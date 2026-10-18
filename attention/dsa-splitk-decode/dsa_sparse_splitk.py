"""Split-top-k wrapper for FlashMLA's SM90 sparse attention kernel on small token counts (amdpro, 2026-10-05).

Installed by operations/patches/dsa_sparse_splitk_hooks.py as sglang/srt/layers/attention/dsa/numinous_sparse_splitk.py
and called from DeepseekSparseAttnBackend._forward_flashmla_sparse (dsa_backend.py) when
NUMINOUS_DSA_SPARSE_SPLITK=1.

Why (source + measured):
  - What runs: flash_mla_sparse_fwd is FlashMLA's sparse PREFILL kernel (sgl-project/FlashMLA@3e18517,
    csrc/kernels/sm90/prefill/sparse/phase1.cuh). The KV4 scratch decode path (glm_kv4_sm90._attend) and the bf16
    flashmla_sparse decode both use it.
  - Grid: (h_q/64) * s_q CTAs (phase1.cuh:629, B_H = 64 in config.h:26).
  - Per CTA: all top-k rows, in 64-row blocks (B_TOPK, config.h:27), through a 2-stage cp.async gather pipeline
    (phase1.cuh:466-530), one CTA per SM (~200 KB of shared memory).
  - Effect at GLM shapes (64 heads per rank): a 36-token verify call launches 36 CTAs on 132 SMs, and a 6-token draft
    call launches 6.
  - measured: P2 decode profile (weights-cache:/verify_logs/prof_P2_20261005-051443/decode) has 83 calls per cycle at
    60.6 us = 5.03 ms. rtxsys's H200 kernel check runs the same kernel at 48 tokens in 60.61 us, against a 23.6 us
    bandwidth floor (research/gpu-evidence/rtxsys/hopper_kv4_check_h200_profile.json).
  - 36 CTAs moving 85 MB in 60.6 us is ~39 GB/s per active SM: the kernel is SM-starved, not tile-bound.

Fix: split each token's top-k into S chunks (multiples of the kernel's 64-row block) and run them as S x more query
rows, so S x more CTAs. Then merge the partial outputs with their log-sum-exps. flash_mla_sparse_fwd returns
(out, max_logits, lse) with lse the BASE-2 log-sum-exp of the scaled logits, and +inf for a row with no valid key.
  - The kernel itself writes the natural log: logf(rL) + rM*LN2, +INFINITY when empty (vendor
    csrc/kernels/sm90/prefill/sparse/phase1.cuh:446-449).
  - The binding sparse_prefill_fwd multiplies max_logits and lse by LOG_2_E before returning (vendor
    csrc/python_api.cpp:212-215: "Keep SGL compatibility").
  - measured on H200 (research/gpu-evidence/amdpro/splitk_fix_bench_h200.json, run 2): weighting with exp instead of
    exp2 raised the invalid-row max diff from 2.4e-4 to 0.023 and gave 6-33% error on peaky rows.
    lse = log2(sum_s 2^lse_s),  out = sum_s 2^(lse_s - lse) * out_s
S minimises waves x rows per CTA on the device's SM count; ties go to the smaller S, and S = 1 when the grid already
fills the device. Attention sinks are not split (the wrapper declines). Each partial output is rounded to bf16 once
before the fp32 merge, so the result is not bit-identical to one kernel call: it needs the arm's accuracy gate.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:   # pragma: no cover
    triton = None

ENV = "NUMINOUS_DSA_SPARSE_SPLITK"
B_TOPK = 64            # FlashMLA SM90 sparse prefill top-k block (config.h:27)
B_H = 64               # heads per CTA (config.h:26)
EMPTY_LSE = -1e29      # the kernel initialises max logits to -1e30 (config.h:29); below this a split saw no key


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


@lru_cache(maxsize=8)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


# cost model in "gathered rows per CTA" units (derived, to be fitted on the H200 kernel check):
#   a CTA costs (rows + CTA_FIXED): the Q tile (64 x 576 bf16 = 72 KB, ~64 rows), the 64 x 512 output + lse (~56 rows)
#   and one block of pipeline fill (64 rows); the merge moves 2 x 64 KB per (token, head block, split) through the
#   whole device (~114 row-equivalents / sms)
CTA_FIXED = int(os.environ.get("NUMINOUS_DSA_SPLITK_CTA_FIXED", "192"))
MERGE_ROWS = 114


def split_factor(tokens: int, heads: int, topk: int, sms: int, max_split: Optional[int] = None) -> int:
    """S in {1, 2, 4, ...} minimising waves x (topk/S + CTA_FIXED) + merge traffic; smallest S on ties."""
    if max_split is None:
        # measured on H200 (research/gpu-evidence/amdpro/p2_decode_fix_bench_h200.json): 6 tokens best at S=8
        # (22.3 us vs 27.9 at S=16, stock 57.1); 36 tokens best at S=2 (52.9 vs stock 63.1); 72 tokens: no split
        max_split = int(os.environ.get("NUMINOUS_DSA_SPLITK_MAX", "8"))
    hb = max(1, heads // B_H)
    if tokens <= 0 or topk % B_TOPK or tokens * hb >= sms:
        return 1

    def cost(s: int) -> float:
        waves = math.ceil(tokens * hb * s / sms)
        merge = MERGE_ROWS * s * tokens * hb / sms if s > 1 else 0.0
        return waves * (topk / s + CTA_FIXED) + merge
    best, best_cost = 1, cost(1)
    s = 2
    while s <= max_split and topk % (B_TOPK * s) == 0:
        c = cost(s)
        if c < best_cost:
            best, best_cost = s, c
        s *= 2
    return best


if triton is not None:
    @triton.jit
    def _merge_kernel(o_ptr, lse_ptr, out_ptr, H, S: tl.constexpr, DV: tl.constexpr, BLOCK_D: tl.constexpr):
        # one program per (token, head): weights from the S split lse values, then the weighted sum over S outputs
        pid = tl.program_id(0)
        t = pid // H
        h = pid % H
        s_offs = tl.arange(0, S)
        lse = tl.load(lse_ptr + (t * S + s_offs) * H + h)
        valid = (lse > -1e29) & (lse < 1e29)                   # a NaN / +-inf lse is an empty split
        m = tl.max(tl.where(valid, lse, -1e30), 0)
        # the binding returns a BASE-2 lse: the kernel writes logf(rL) + rM*LN2 (vendor phase1.cuh:446-449), and
        # sparse_prefill_fwd multiplies it by LOG_2_E before returning (vendor csrc/python_api.cpp:212-215,
        # "Keep SGL compatibility: this API historically returns max_logits/lse in log2 space")
        w = tl.where(valid, tl.exp2(lse - m), 0.0)
        den = tl.sum(w, 0)
        w = w / tl.where(den > 0, den, 1.0)
        d_offs = tl.arange(0, BLOCK_D)
        for d0 in range(0, DV, BLOCK_D):
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            for s in tl.static_range(S):
                ws = tl.sum(tl.where(s_offs == s, w, 0.0), 0)
                o = tl.load(o_ptr + ((t * S + s) * H + h) * DV + d0 + d_offs).to(tl.float32)
                # a split with no valid key has weight 0 and may hold NaN (FlashMLA README reference: exp2(P - lse)
                # with P = lse = -inf); 0 * NaN would poison the merge, so its partial is dropped, not multiplied
                acc += tl.where(ws > 0.0, ws * o, 0.0)
            tl.store(out_ptr + (t * H + h) * DV + d0 + d_offs, acc.to(tl.bfloat16))


def merge(o_s: torch.Tensor, lse_s: torch.Tensor, tokens: int, s: int) -> torch.Tensor:
    """o_s [T*S, H, DV] bf16 (row t*S+s), lse_s [T*S, H] fp32 base 2 -> [T, H, DV] bf16."""
    h, dv = o_s.shape[1], o_s.shape[2]
    out = torch.empty(tokens, h, dv, dtype=torch.bfloat16, device=o_s.device)
    if tokens == 0:
        return out
    _merge_kernel[(tokens * h,)](o_s.contiguous(), lse_s.contiguous().float(), out, h, S=s, DV=dv,
                                 BLOCK_D=min(dv, 128), num_warps=4)
    return out


def merge_reference(o_s: torch.Tensor, lse_s: torch.Tensor, tokens: int, s: int) -> torch.Tensor:
    """The same merge in torch (tests)."""
    h, dv = o_s.shape[1], o_s.shape[2]
    o = o_s.view(tokens, s, h, dv).float()
    lse = lse_s.view(tokens, s, h).float()
    valid = (lse > EMPTY_LSE) & (lse < 1e29)
    m = torch.where(valid, lse, torch.full_like(lse, -1e30)).amax(dim=1, keepdim=True)
    w = torch.where(valid, torch.exp2(lse - m), torch.zeros_like(lse))  # base-2 lse (python_api.cpp:212-215)
    den = w.sum(dim=1, keepdim=True)
    w = w / torch.where(den > 0, den, torch.ones_like(den))
    contrib = torch.where(w.unsqueeze(-1) > 0, o * w.unsqueeze(-1), torch.zeros_like(o))
    return contrib.sum(dim=1).to(torch.bfloat16)


def split_inputs(q: torch.Tensor, indices: torch.Tensor, topk_length: Optional[torch.Tensor], s: int):
    """Rows t*S+s: q[t] and the s-th top-k chunk of token t (indices [T, 1, K] -> [T*S, 1, K/S])."""
    t, _, k = indices.shape
    c = k // s
    q_s = q.repeat_interleave(s, dim=0)
    idx_s = indices.reshape(t, 1, s, c).permute(0, 2, 1, 3).reshape(t * s, 1, c).contiguous()
    tl_s = None
    if topk_length is not None:
        starts = torch.arange(s, device=topk_length.device, dtype=topk_length.dtype) * c
        tl_s = (topk_length.view(t, 1) - starts.view(1, s)).clamp(0, c).reshape(t * s).to(torch.int32).contiguous()
    return q_s, idx_s, tl_s


def sparse_fwd_splitk(fwd, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float, d_v: int,
                      topk_length: Optional[torch.Tensor], s: int, merge_fn=None) -> torch.Tensor:
    """flash_mla_sparse_fwd over S top-k splits, merged by lse. `fwd` is flash_mla_sparse_fwd (or a test stand-in)."""
    t = q.shape[0]
    q_s, idx_s, tl_s = split_inputs(q, indices, topk_length, s)
    o_s, _, lse_s = fwd(q=q_s, kv=kv, indices=idx_s, sm_scale=sm_scale, d_v=d_v, attn_sink=None, topk_length=tl_s)
    return (merge_fn or merge)(o_s, lse_s, t, s)


def maybe_split(tokens: int, heads: int, topk: int, device: torch.device, has_sink: bool) -> int:
    if not enabled() or has_sink or device.type != "cuda":
        return 1
    return split_factor(tokens, heads, topk, _sm_count(device.index if device.index is not None else 0))

