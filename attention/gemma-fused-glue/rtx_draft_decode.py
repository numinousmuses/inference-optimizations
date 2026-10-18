"""GQA split-K decode attention for the frozen-KV MTP draft (rtxsys, Gemma 31B kernel program, 2026-10-06).
Installed as sglang/srt/layers/attention/numinous_draft_decode.py by operations/patches/gemma31_draft_decode_hooks.py.
Env NUMINOUS_DRAFT_DECODE=1.

What it attacks:
- rtxv41's run1 composed profile assigns "Draft attention 3.3 ms per 32-session verify step (floor ~1)" to rtxsys
  (findings rtxv41_gemma31.md section 10).
- The draft's 4 layers x 5 steps run TritonAttnBackend.forward_decode -> decode_attention_fwd ->
  _fwd_grouped_kernel_stage1 (decode_attention.py:768-880): BLOCK_N 32, 2 stages, 4 warps, heads padded to 16.
  Over the target's FP8 window / full KV that reaches ~45% of HBM (stock profile: 20 calls 2.98 ms; ~1 GB per draft
  step at 32 sessions).

This kernel computes the same thing over the same metadata (kv_indptr / kv_indices: window lists on SWA layers):
- one program per (request, kv head, split) loads each K/V tile once for the GROUP q heads, online softmax in fp32,
  bf16 MMAs, 8 warps; BLOCK_N 64 / 3 stages for hd256 FP8, 32 / 4 for hd256 bf16, 32 / 3 for hd512 (0 spills);
- the split count is fixed per (batch, kv heads, window) at capture time;
- a combine kernel then merges the splits by log-sum-exp.
Scales follow the stock path: k_descale folded into sm_scale; v_descale applied to the output.
Scope: page size 1, no sinks / logit cap / score_mod / xai temperature, non-MLA (Dk == Dv <= 512), bf16 q, bf16 or
FP8 e4m3 KV, DCP off. Layers whose K buffer aliases V (rtxattn's KV1 full layers) stay on their own path.
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

ON = os.environ.get("NUMINOUS_DRAFT_DECODE", "0") == "1"
EMU = os.environ.get("TRITON_INTERPRET", "0") == "1"


if triton is not None:

    @triton.jit
    def _dd_stage1(Q, Kb, Vb, KV_INDPTR, KV_INDICES, OUT, LSE, qk_scale,
                   stride_qb, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh,
                   stride_ob, stride_oh, stride_os, stride_lb, stride_lh, NSPLIT,
                   GROUP: tl.constexpr, D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
                   EMU: tl.constexpr):
        b = tl.program_id(0)
        hkv = tl.program_id(1)
        sp = tl.program_id(2)
        start = tl.load(KV_INDPTR + b)
        n = tl.load(KV_INDPTR + b + 1) - start
        per = tl.cdiv(tl.cdiv(n, NSPLIT), BLOCK_N) * BLOCK_N
        lo = sp * per
        hi = tl.minimum(lo + per, n)
        hr = tl.arange(0, BLOCK_H)
        hmask = hr < GROUP
        heads = hkv * GROUP + hr
        dd = tl.arange(0, D)
        q = tl.load(Q + b.to(tl.int64) * stride_qb + heads[:, None] * stride_qh + dd[None, :],
                    mask=hmask[:, None], other=0.0)
        if EMU:
            q = q.to(tl.float32)
        else:
            q = q.to(tl.bfloat16)
        m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)
        for t0 in range(lo, hi, BLOCK_N):
            tt = t0 + tl.arange(0, BLOCK_N)
            tm = tt < hi
            slot = tl.load(KV_INDICES + start + tt, mask=tm, other=0).to(tl.int64)
            k = tl.load(Kb + slot[:, None] * stride_kt + hkv * stride_kh + dd[None, :], mask=tm[:, None], other=0.0)
            v = tl.load(Vb + slot[:, None] * stride_vt + hkv * stride_vh + dd[None, :], mask=tm[:, None], other=0.0)
            if EMU:
                s = tl.dot(q, tl.trans(k.to(tl.float32)), input_precision="ieee")
            else:
                s = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
            s = tl.where(tm[None, :], s * qk_scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            if EMU:
                pv = tl.dot(p, v.to(tl.float32), input_precision="ieee")
            else:
                pv = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
            acc = acc * alpha[:, None] + pv
            m_i = m_new
        o_base = b.to(tl.int64) * stride_ob + heads * stride_oh + sp * stride_os
        has = l_i > 0
        out = acc / tl.where(has, l_i, 1.0)[:, None]
        tl.store(OUT + o_base[:, None] + dd[None, :], out, mask=hmask[:, None])
        lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), -float("inf"))
        tl.store(LSE + b.to(tl.int64) * stride_lb + heads * stride_lh + sp, lse, mask=hmask)

    @triton.jit
    def _dd_stage2(OUT, LSE, O, v_scale, stride_ob, stride_oh, stride_os, stride_lb, stride_lh, stride_fb, stride_fh,
                   NSPLIT, D: tl.constexpr, BLOCK_S: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        js = tl.arange(0, BLOCK_S)
        jm = js < NSPLIT
        base = b.to(tl.int64) * stride_ob + h * stride_oh
        lse = tl.load(LSE + b.to(tl.int64) * stride_lb + h * stride_lh + js, mask=jm, other=-float("inf"))
        mx = tl.max(lse, axis=0)
        w = tl.where(jm & (lse > -float("inf")), tl.exp(lse - mx), 0.0)
        dd = tl.arange(0, D)
        parts = tl.load(OUT + base + js[:, None] * stride_os + dd[None, :], mask=jm[:, None], other=0.0)
        o = tl.sum(parts * w[:, None], axis=0) / tl.maximum(tl.sum(w, axis=0), 1e-30)
        tl.store(O + b.to(tl.int64) * stride_fb + h * stride_fh + dd, (o * v_scale).to(O.dtype.element_ty))


def choose_splits(bs: int, hkv: int, window: int, sm_count: int = 132) -> int:
    """Splits per (request, kv head): ~8 programs per SM, each split >= 128 keys of the usual span (the window on
    SWA layers, ~2K otherwise). Fixed per capture shape: no device data.
    run5 (rtxv41, h100): the old rule (~3 waves, >= 256 keys) gave 1 split at bs 32 x 16 kv heads over a 1,023-token
    window: 512 programs at 254 regs, 1 per SM, 0.189 ms per call vs stock 0.163 (8 splits x 128 keys, 4 warps)."""
    span = window if window > 0 else 2048
    want = max(1, (8 * sm_count + bs * hkv - 1) // max(1, bs * hkv))
    return int(max(1, min(want, max(1, span // 128), 16)))


def _cfg_override():
    """NUMINOUS_DRAFT_DECODE_CFG="splits,block_n,num_warps,num_stages" (0 = default) for the bench sweep."""
    raw = os.environ.get("NUMINOUS_DRAFT_DECODE_CFG", "")
    if not raw:
        return None
    vals = [int(x) for x in raw.split(",")]
    return (vals + [0, 0, 0, 0])[:4]


def decode(q: torch.Tensor, k_buf: torch.Tensor, v_buf: torch.Tensor, o: torch.Tensor, kv_indptr: torch.Tensor,
           kv_indices: torch.Tensor, sm_scale: float, k_scale: float = 1.0, v_scale: float = 1.0, window: int = 0,
           n_splits: int | None = None, block_n: int | None = None, num_warps: int | None = None,
           num_stages: int | None = None) -> torch.Tensor:
    """q [bs, Hq, D] bf16; k_buf / v_buf [slots, Hkv, D] (bf16 or e4m3); o [bs, Hq, D] written."""
    bs, hq, d = q.shape
    hkv = k_buf.shape[-2]
    group = hq // hkv
    ov = _cfg_override()
    if ov is not None:
        n_splits = n_splits or (ov[0] or None)
        block_n = block_n or (ov[1] or None)
        num_warps = num_warps or (ov[2] or None)
        num_stages = num_stages or (ov[3] or None)
    ns = n_splits or choose_splits(bs, hkv, window)
    out = torch.empty((bs, hq, ns, d), dtype=torch.float32, device=q.device)
    lse = torch.empty((bs, hq, ns), dtype=torch.float32, device=q.device)
    fp8 = k_buf.dtype == torch.float8_e4m3fn
    # ptxas sm_90a (rtx_aot_check.py), 0 spills: hd256 fp8 BN32/4w/3st 212 regs (2 CTAs/SM), hd256 bf16 BN32/8w/4st,
    # hd512 BN32/8w/3st. The run5 default (hd256 fp8 BN64/8w, 254 regs, 1 CTA/SM) was slower than stock.
    bn0, nw0, st0 = (32, 8, 3) if d >= 512 else ((32, 4, 3) if fp8 else (32, 8, 4))
    block_n, nw, st = block_n or bn0, num_warps or nw0, num_stages or st0
    _dd_stage1[(bs, hkv, ns)](
        q, k_buf, v_buf, kv_indptr, kv_indices, out, lse, float(sm_scale) * float(k_scale),
        q.stride(0), q.stride(1), k_buf.stride(0), k_buf.stride(1), v_buf.stride(0), v_buf.stride(1),
        out.stride(0), out.stride(1), out.stride(2), lse.stride(0), lse.stride(1), ns,
        GROUP=group, D=d, BLOCK_H=max(16, triton.next_power_of_2(group)), BLOCK_N=block_n, EMU=EMU,
        num_warps=nw, num_stages=st)
    _dd_stage2[(bs, hq)](out, lse, o, float(v_scale), out.stride(0), out.stride(1), out.stride(2),
                         lse.stride(0), lse.stride(1), o.stride(0), o.stride(1), ns, D=d, BLOCK_S=max(2, triton.next_power_of_2(ns)),
                         num_warps=4)
    return o


def decode_ref(q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale, k_scale=1.0, v_scale=1.0):
    """Reference: per request, softmax(q K^T * sm_scale * k_scale) V * v_scale over its kv_indices (fp32)."""
    bs, hq, d = q.shape
    hkv = k_buf.shape[-2]
    g = hq // hkv
    o = torch.zeros(bs, hq, d, dtype=torch.float32)
    for b in range(bs):
        idx = kv_indices[int(kv_indptr[b]):int(kv_indptr[b + 1])].long()
        if idx.numel() == 0:
            continue
        k = k_buf[idx].float()          # [n, hkv, d]
        v = v_buf[idx].float()
        for h in range(hq):
            s = (k[:, h // g] @ q[b, h].float()) * sm_scale * k_scale
            o[b, h] = torch.softmax(s, 0) @ v[:, h // g] * v_scale
    return o


# ----------------------------------------------------------------------------------------------------- hook body
_logged = False


def maybe_decode(backend, q, o, layer, forward_batch, kv_indptr, kv_indices, k_descale, v_descale, logits_soft_cap,
                 sinks, score_mod) -> bool:
    """TritonAttnBackend.forward_decode, before the stock decode call: True when served (o written)."""
    global _logged
    if not ON or triton is None:
        return False
    if (getattr(backend, "page_size", 1) != 1 or getattr(backend, "use_mla", False)
            or getattr(backend, "dcp_size", 1) > 1 or sinks is not None or score_mod is not None
            or (logits_soft_cap or 0) > 0 or (getattr(layer, "xai_temperature_len", -1) or -1) > 0
            or layer.qk_head_dim != layer.v_head_dim or layer.qk_head_dim > 512 or not q.is_cuda
            or q.dtype != torch.bfloat16):
        return False
    pool = backend.token_to_kv_pool
    k_buf = pool.get_key_buffer(layer.layer_id)
    v_buf = pool.get_value_buffer(layer.layer_id)
    if (k_buf.dim() != 3 or v_buf.dim() != 3 or k_buf.data_ptr() == v_buf.data_ptr()
            or k_buf.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or v_buf.dtype != k_buf.dtype
            or k_buf.stride(-1) != 1 or v_buf.stride(-1) != 1):
        return False
    hq, d = layer.tp_q_head_num, layer.qk_head_dim
    if hq % k_buf.shape[-2] != 0:
        return False
    window = int(layer.sliding_window_size) if (layer.sliding_window_size or -1) > 0 else 0
    q3 = q.view(-1, hq, d)
    decode(q3, k_buf, v_buf, o.view(-1, hq, layer.v_head_dim), kv_indptr, kv_indices, layer.scaling,
           float(k_descale), float(v_descale), window)
    if not _logged:
        _logged = True
        import logging

        logging.getLogger(__name__).info("numinous draft decode: ENGAGED (layer %d, hq %d, hkv %d, d %d, window %d)",
                                         layer.layer_id, hq, k_buf.shape[-2], d, window)
    return True
