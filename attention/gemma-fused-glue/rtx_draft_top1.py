"""Fused softmax top-1 for the frozen-KV MTP draft steps (rtxsys, Gemma 31B kernel program D2-b, 2026-10-06).
Installed as sglang/srt/layers/numinous_draft_top1.py by operations/patches/gemma31_draft_top1_hooks.py.
Env NUMINOUS_DRAFT_TOP1=1.

What it attacks:
- Every draft step of FrozenKVMTPDraftWorker.draft_forward (frozen_kv_mtp_worker_v2.py:577-578 seed, :617-618 steps)
  runs `probs = torch.softmax(logits, -1)` over [bs, 262144] fp32, then `fast_topk(probs, 1)` = torch.max
  (utils/common.py:3680-3683).
- Measured (rtxv41 prof_decode32_composed, 1xH100, 32 sessions):
  - cunn_SoftMaxForward: 6 calls 0.862 ms per step (one is the verify softmax);
  - the MaxOps reduce: 5 calls 0.117 ms.
  The softmax writes a 33.5 MB probability tensor per draft step that only its max is read from.
- With topk == 1 only (max prob, argmax) is needed: p_max = exp(x_max - M) / sum = 1 / sum_j exp(x_j - M).
  This module computes it in one read of the logits, split across CTAs (a [bs, 262144] row per CTA is what starves
  the stock softmax at bs 32), then a tiny combine.

Exactness:
- topk_index = argmax(logits), smallest index on exact ties. topk_p = 1 / sum(exp(x - max)) in fp32.
- torch's softmax gives probs[argmax] = exp(0) / sum = 1 / sum, the same quantity; it differs only in the fp32
  summation order (~1e-7 relative).
- A pair of logits closer than the probability rounding can tie in probs while differing in logits. Stock then
  picks either index; this picks the larger logit. Both are argmaxes of the draft distribution.
- The draft only proposes tokens; the target verify decides the output, so neither changes what is emitted.

Fused draft head (D2-c, env NUMINOUS_DRAFT_HEAD_FUSED=1 on top of NUMINOUS_DRAFT_TOP1=1):
- Each draft step's head is the full tied 262,144 x 1,024 BF16 GEMM (nvjet 512x32, 5 calls 0.937 ms per step,
  measured). The logits processor then copies the bf16 logits into its fp32 buffer (direct_copy, part of 6 calls
  0.28 ms), and softmax / top-1 reads them again.
- head_top1 computes the GEMM tile by tile and reduces each 128-wide vocab tile to (max, argmax, sum-exp) in the
  epilogue. The [bs, 262144] logits never exist.
- The tile logits are rounded to bf16 first, as the stock bf16 GEMM output is.
- The model hook swaps LogitsProcessor._get_logits for that call only, so pruning and hidden-state capture stay
  stock. _get_logits returns a zero-stride [bs, V] placeholder carrying .numinous_top1 = (p, idx), which the worker
  hook consumes.
- It engages only in DECODE draft steps with topk 1, no logprobs, no TP / DP logits gather, no logit scale or softcap.
- NUMINOUS_DRAFT_HEAD_FP8=1 additionally keeps an FP8 e4m3 copy of the head with per-row scales, made once outside
  graph capture. It is upcast to bf16 per tile (W8A16): half the head bytes. It changes draft logits by about FP8
  rounding, so acceptance must be measured; outputs cannot change.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

ON = os.environ.get("NUMINOUS_DRAFT_TOP1", "0") == "1"
HEAD_ON = ON and os.environ.get("NUMINOUS_DRAFT_HEAD_FUSED", "0") == "1"
HEAD_FP8 = HEAD_ON and os.environ.get("NUMINOUS_DRAFT_HEAD_FP8", "0") == "1"
EMU = os.environ.get("TRITON_INTERPRET", "0") == "1"
SPLIT_BLOCK = 4096
HEAD_BN = 128
HEAD_BK = 64        # BK 128 spills (ptxas sm_90a: 255 regs, 0.4-1.9 KB); BK 64 at 8 warps: 250 regs, 0 spills


if triton is not None:

    @triton.jit
    def _top1_partial_kernel(X, PM, PI, PS, stride_x, V, NSPLIT, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        sp = tl.program_id(1)
        cols = sp * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + row * stride_x + cols, mask=cols < V, other=-float("inf")).to(tl.float32)
        m = tl.max(x, axis=0)
        idx = tl.min(tl.where(x == m, cols, 2147483647), axis=0)       # smallest index among the maxima
        s = tl.sum(tl.where(cols < V, tl.exp(x - m), 0.0), axis=0)
        o = row * NSPLIT + sp
        tl.store(PM + o, m)
        tl.store(PI + o, idx)
        tl.store(PS + o, s)

    @triton.jit
    def _top1_combine_kernel(PM, PI, PS, P, I, NSPLIT, BLOCK_S: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, BLOCK_S)
        msk = j < NSPLIT
        m = tl.load(PM + row * NSPLIT + j, mask=msk, other=-float("inf"))
        ii = tl.load(PI + row * NSPLIT + j, mask=msk, other=2147483647)
        s = tl.load(PS + row * NSPLIT + j, mask=msk, other=0.0)
        mx = tl.max(m, axis=0)
        idx = tl.min(tl.where(m == mx, ii, 2147483647), axis=0)
        tot = tl.sum(tl.where(msk, s * tl.exp(m - mx), 0.0), axis=0)
        tl.store(P + row, 1.0 / tot)
        tl.store(I + row, idx.to(tl.int64))


if triton is not None:

    @triton.jit
    def _rne_bf16(x):
        b = x.to(tl.uint32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return b.to(tl.float32, bitcast=True)

    @triton.jit
    def _head_top1_partial_kernel(Hd, W, WS, PM, PI, PS, BS, V, K, stride_h, stride_w, NT,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                  FP8: tl.constexpr, EMU: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(Hd + rows[:, None].to(tl.int64) * stride_h + ks[None, :],
                        mask=(rows[:, None] < BS) & (ks[None, :] < K), other=0.0)
            w = tl.load(W + cols[:, None].to(tl.int64) * stride_w + ks[None, :],
                        mask=(cols[:, None] < V) & (ks[None, :] < K), other=0.0)
            if EMU:   # the interpreter's bf16 dot works on raw bits; fp32 operands are exact for bf16 / e4m3 values
                acc += tl.dot(a.to(tl.float32), tl.trans(w.to(tl.float32)), input_precision="ieee")
            else:
                acc += tl.dot(a.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
        if FP8:
            acc = acc * tl.load(WS + cols, mask=cols < V, other=0.0)[None, :]
        if EMU:
            acc = _rne_bf16(acc)                                   # the stock head writes bf16 logits
        else:
            acc = acc.to(tl.bfloat16).to(tl.float32)
        acc = tl.where(cols[None, :] < V, acc, -float("inf"))
        m = tl.max(acc, axis=1)
        idx = tl.min(tl.where(acc == m[:, None], cols[None, :], 2147483647), axis=1)
        s = tl.sum(tl.exp(acc - m[:, None]), axis=1)
        o = rows.to(tl.int64) * NT + pid_n
        tl.store(PM + o, m, mask=rows < BS)
        tl.store(PI + o, idx, mask=rows < BS)
        tl.store(PS + o, s, mask=rows < BS)


def head_top1(hidden: torch.Tensor, weight: torch.Tensor, vocab: int, wscale: torch.Tensor | None = None):
    """(p_max [bs, 1] fp32, argmax [bs, 1] int64) of softmax(bf16(hidden @ weight[:vocab].T)) without logits.
    weight [>=vocab, K] bf16, or e4m3 with per-row fp32 scales wscale [>=vocab]."""
    bs, k = hidden.shape
    p = torch.empty((bs, 1), dtype=torch.float32, device=hidden.device)
    i = torch.empty((bs, 1), dtype=torch.int64, device=hidden.device)
    if bs == 0:
        return p, i
    nt = triton.cdiv(vocab, HEAD_BN)
    bm = max(16, min(64, triton.next_power_of_2(bs)))
    pm = torch.empty((bs, nt), dtype=torch.float32, device=hidden.device)
    pi = torch.empty((bs, nt), dtype=torch.int32, device=hidden.device)
    ps = torch.empty((bs, nt), dtype=torch.float32, device=hidden.device)
    fp8 = wscale is not None
    _head_top1_partial_kernel[(nt, triton.cdiv(bs, bm))](
        hidden, weight, wscale if fp8 else pm, pm, pi, ps, bs, vocab, k, hidden.stride(0), weight.stride(0), nt,
        BLOCK_M=bm, BLOCK_N=HEAD_BN, BLOCK_K=HEAD_BK, FP8=fp8, EMU=EMU, num_warps=8, num_stages=4)
    _top1_combine_kernel[(bs,)](pm, pi, ps, p, i, nt, BLOCK_S=triton.next_power_of_2(nt), num_warps=8)
    return p, i


def quantize_head_fp8(weight: torch.Tensor):
    """Per-row e4m3 copy of a [V, K] bf16 head: (q [V, K] e4m3, scale [V] fp32), scale = amax / 448."""
    w = weight.float()
    s = (w.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
    q = (w / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, s.contiguous()


def softmax_top1(logits: torch.Tensor):
    """(topk_p [bs, 1] fp32, topk_index [bs, 1] int64) == fast_topk(torch.softmax(logits, -1), 1, -1)."""
    assert logits.dim() == 2 and logits.stride(-1) == 1
    bs, v = logits.shape
    nsplit = triton.cdiv(v, SPLIT_BLOCK)
    pm = torch.empty((bs, nsplit), dtype=torch.float32, device=logits.device)
    pi = torch.empty((bs, nsplit), dtype=torch.int32, device=logits.device)
    ps = torch.empty((bs, nsplit), dtype=torch.float32, device=logits.device)
    p = torch.empty((bs, 1), dtype=torch.float32, device=logits.device)
    i = torch.empty((bs, 1), dtype=torch.int64, device=logits.device)
    if bs == 0:
        return p, i
    _top1_partial_kernel[(bs, nsplit)](logits, pm, pi, ps, logits.stride(0), v, nsplit, BLOCK=SPLIT_BLOCK,
                                       num_warps=8)
    _top1_combine_kernel[(bs,)](pm, pi, ps, p, i, nsplit, BLOCK_S=triton.next_power_of_2(nsplit), num_warps=4)
    return p, i


def draft_top1(logits: torch.Tensor, topk: int):
    """Worker hook body: (p, idx) for topk == 1, else None (stock softmax + fast_topk).
    A placeholder from the fused head carries its result; otherwise the logits are reduced here."""
    if not ON or topk != 1:
        return None
    pre = getattr(logits, "numinous_top1", None)
    if pre is not None:
        return pre
    if (triton is None or not logits.is_cuda or logits.dim() != 2 or logits.stride(-1) != 1
            or logits.dtype not in (torch.float32, torch.bfloat16, torch.float16)):
        return None
    return softmax_top1(logits)


# ------------------------------------------------------------------------------------------------ fused draft head
_FP8_HEADS: dict = {}
_logged = False


def _log(msg: str) -> None:
    global _logged
    if not _logged:
        _logged = True
        import logging

        logging.getLogger(__name__).info("numinous draft head (D2-c): %s", msg)


def _spec_topk() -> int:
    try:
        from sglang.srt.runtime_context import get_spec

        return int(get_spec().speculative_eagle_topk or 1)
    except Exception:
        return -1


def _head_weights(model):
    w = model.lm_head.weight
    if not HEAD_FP8:
        return w, None
    key = id(w)
    got = _FP8_HEADS.get(key)
    if got is None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return w, None          # never quantize inside a capture; this graph keeps the bf16 head
        got = quantize_head_fp8(w.data)
        _FP8_HEADS[key] = got
    return got


def _on_device(t: torch.Tensor) -> bool:
    return t.is_cuda


def draft_head(model, input_ids, hidden_states, projected_states, forward_batch):
    """Model hook body (Gemma4AssistantForCausalLM.forward): LogitsProcessorOutput via the fused head, or None."""
    if not HEAD_ON or triton is None:
        return None
    lp = model.logits_processor
    fm = getattr(forward_batch, "forward_mode", None)
    if (fm is None or not fm.is_decode() or getattr(forward_batch, "return_logprob", False)
            or getattr(lp, "do_tensor_parallel_all_gather", False)
            or getattr(lp, "do_tensor_parallel_all_gather_dp_attn", False)
            or getattr(lp, "logit_scale", None) is not None or getattr(lp, "final_logit_softcapping", None)
            or getattr(model, "use_ordered_embeddings", False) or _spec_topk() != 1
            or not _on_device(hidden_states) or hidden_states.dim() != 2):
        return None
    vocab = int(model.vocab_size)
    weight, wscale = _head_weights(model)
    if weight.shape[0] < vocab or weight.stride(-1) != 1:
        return None

    def fused_get_logits(hs, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
        h = hs if hs.stride(-1) == 1 else hs.contiguous()
        top1 = head_top1(h, weight, vocab, wscale)
        ph = torch.empty((1, 1), dtype=torch.float32, device=hs.device).expand(hs.shape[0], vocab)
        ph.numinous_top1 = top1
        return ph

    _log("ENGAGED" + (" (fp8 head)" if wscale is not None else ""))
    lp._get_logits = fused_get_logits
    try:
        out = lp(input_ids, hidden_states, model.lm_head, forward_batch, hidden_states_before_norm=projected_states)
    finally:
        del lp._get_logits
    if getattr(out.next_token_logits, "numinous_top1", None) is None:   # pruned / indexed: not ours, redo stock
        return None
    return out
