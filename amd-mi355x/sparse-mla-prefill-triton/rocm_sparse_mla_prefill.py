"""Prefill-shaped sparse MLA attention for gfx950 (MI355X), Triton, FP8 raw KV (the HIP TileLang layout).

Replaces, for prefill-sized calls, the HIP path of tilelang_sparse_fwd (which runs the decode-shaped
`sparse_mla_fwd_decode_partial_fp8` with a 16-head tile plus a combine kernel for every call; 4.96 ms per layer at
4096 tokens x 64 heads x top-2048, ~5% of the FP8 peak, measured 2026-09-30 by rocm_sparse_mla_prefill_bench.py).

One program = one query token x all local heads (BLOCK_H = the head count padded to 16): the Q tile stays resident,
each 64-key block of the token's top-k is gathered once for all heads, QK^T runs as two FP8 MFMA GEMMs (latent 512 +
rope 64), online softmax in fp32, and PV as an FP8 GEMM on P scaled into the E4M3 range (the TileLang kernel's own
scheme: P * 448 -> fp8, product / 448). No split-K / combine: prefill has enough tokens to fill the GPU.
Defaults (block_n 64, 4 warps, 3 pipeline stages, kpack 2, MFMA non-K dim 16) are the fastest of the sweep in
rocm_sparse_mla_prefill_bench.py: 1.77 ms (1 stage: 1.92 ms) vs 4.43 ms for the stock path at 4096 x 64 heads, same
error vs stock (2026-09-30). head_split 2/4 is slower (3.3-5.7 ms).

    sparse_mla_prefill_fp8(q, kv, indices, sm_scale, d_v=512) -> [1, T, H, d_v] bf16 (the tilelang_sparse_fwd layout)
    sparse_mla_prefill_split_q(q_nope, q_rope, kv, indices, sm_scale) -> same, q as two bf16/fp8 tensors

q [T, H, 576] (bf16 or fp8), kv [slots, 1, 576] float8_e4m3fn, indices [T, 1, K] int32 (-1 = masked).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


def _env_launch(h: int, block_n: int, num_warps: int, num_stages: int, waves_per_eu: int, kpack: int):
    """NUMINOUS_SPARSE_PREFILL_CFG="block_n,num_warps,num_stages[,waves_per_eu,kpack]" overrides the launch for calls
    with at most NUMINOUS_SPARSE_PREFILL_CFG_MAX_H local heads (default 16 = GLM-5.3 TP4). Measured 2026-10-03 (amdflash,
    operations/amdflash/sparse_prefill_sweep16.py, MI355X, 8192 tokens x 16 heads x top-2048): 32,2,1,0,2 = 1.53 ms vs
    2.63 ms for the 64-head defaults (1.72x). Unset: the defaults (64 heads: 64,4,3)."""
    cfg = os.environ.get("NUMINOUS_SPARSE_PREFILL_CFG")
    if not cfg or h > int(os.environ.get("NUMINOUS_SPARSE_PREFILL_CFG_MAX_H", "16")):
        return block_n, num_warps, num_stages, waves_per_eu, kpack
    v = [int(x) for x in cfg.split(",")]
    v += [waves_per_eu, kpack][len(v) - 3:] if len(v) < 5 else []
    return v[0], v[1], v[2], v[3], v[4]


@triton.jit
def _sparse_mla_prefill_kernel(q_ptr, qr_ptr, kv_ptr, idx_ptr, o_ptr, sm_scale_log2,
                               stride_qt, stride_qh, stride_qrt, stride_qrh, stride_kv, stride_it, stride_ot, stride_oh,
                               H: tl.constexpr, BLOCK_H: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr,
                               DV: tl.constexpr, DR: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)  # head group (head_split > 1: several per token)
    hmask = offs_h < H
    offs_d = tl.arange(0, DV)
    offs_r = tl.arange(0, DR)
    q_row = q_ptr + t * stride_qt + offs_h[:, None] * stride_qh
    qr_row = qr_ptr + t * stride_qrt + offs_h[:, None] * stride_qrh
    # q may arrive as fp8 or bf16 (split q_nope / q_rope from the caller); the MFMAs take fp8
    q_nope = tl.load(q_row + offs_d[None, :], mask=hmask[:, None], other=0.0).to(tl.float8e4nv)  # [BH, DV]
    q_rope = tl.load(qr_row + offs_r[None, :], mask=hmask[:, None], other=0.0).to(tl.float8e4nv)  # [BH, DR]

    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, DV], tl.float32)
    idx_row = idx_ptr + t * stride_it
    for n0 in range(0, K, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        idx = tl.load(idx_row + offs_n)
        valid = idx >= 0
        slot = tl.where(valid, idx, 0).to(tl.int64)
        k_base = kv_ptr + slot[:, None] * stride_kv
        # invalid (-1) picks: masked to 0 instead of reading slot 0 (padded rows write their KV there, so it can hold any
        # bytes; measured on gfx950: rel err 1.0 with NaN bytes in slot 0 unmasked, 0.03 masked)
        k_nope = tl.load(k_base + offs_d[None, :], mask=valid[:, None], other=0.0)  # [BN, DV] fp8
        k_rope = tl.load(k_base + DV + offs_r[None, :], mask=valid[:, None], other=0.0)  # [BN, DR] fp8
        s = tl.dot(q_nope, tl.trans(k_nope))
        s = tl.dot(q_rope, tl.trans(k_rope), s)  # [BH, BN] fp32
        s = tl.where(valid[None, :], s * sm_scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p8 = (p * 448.0).to(tl.float8e4nv)
        acc = acc * alpha[:, None] + tl.dot(p8, k_nope).to(tl.float32) * (1.0 / 448.0)
        m_i = m_new
    denom = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / denom[:, None]
    o_row = o_ptr + t * stride_ot + offs_h[:, None] * stride_oh
    tl.store(o_row + offs_d[None, :], out.to(tl.bfloat16), mask=hmask[:, None])


def sparse_mla_prefill_fp8(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float,
                           d_v: int = 512, block_n: int = 64, num_warps: int = 4, num_stages: int = 3,
                           waves_per_eu: int = 0, kpack: int = 2, nonkdim: int = 16,
                           head_split: int = 1) -> torch.Tensor:
    """head_split > 1 runs `head_split` programs per token, each on H / head_split heads (smaller accumulator, more
    programs in flight; the token's KV rows are gathered once per program, mostly from L2)."""
    assert q.shape[-1] > d_v
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
    q = q.contiguous()
    return sparse_mla_prefill_split_q(q[..., :d_v], q[..., d_v:], kv, indices, sm_scale, d_v, block_n, num_warps,
                                      num_stages, waves_per_eu, kpack, nonkdim, head_split)


def sparse_mla_prefill_split_q(q_nope: torch.Tensor, q_rope: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor,
                               sm_scale: float, d_v: int = 512, block_n: int = 64, num_warps: int = 4,
                               num_stages: int = 3, waves_per_eu: int = 0, kpack: int = 2, nonkdim: int = 16,
                               head_split: int = 1) -> torch.Tensor:
    """Same attention with q given as two tensors (q_nope [T, H, d_v], q_rope [T, H, d_rope], bf16 or fp8, any
    strides with a unit last-dim stride): no concatenated q and no separate fp8 cast pass (the kernel casts on load).
    The engine hook passes the absorb output and the rope output straight through."""
    assert kv.dtype == torch.float8_e4m3fn and kv.dim() == 3 and kv.shape[1] == 1, (kv.dtype, kv.shape)
    t, h, dn = q_nope.shape
    dr = q_rope.shape[-1]
    k = indices.shape[-1]
    block_n, num_warps, num_stages, waves_per_eu, kpack = _env_launch(h, block_n, num_warps, num_stages, waves_per_eu,
                                                                      kpack)
    assert dn == d_v and q_rope.shape[:2] == (t, h) and d_v + dr == kv.shape[2] and k % block_n == 0
    assert q_nope.stride(-1) == 1 and q_rope.stride(-1) == 1
    idx = indices.reshape(t, k).contiguous()
    kv2 = kv.view(kv.shape[0], d_v + dr)
    out = torch.empty(1, t, h, d_v, dtype=torch.bfloat16, device=q_nope.device)
    if t == 0:
        return out
    assert h % head_split == 0, (h, head_split)
    block_h = max(16, triton.next_power_of_2(h // head_split))
    extra = {"waves_per_eu": waves_per_eu} if waves_per_eu else {}
    if kpack:
        extra["kpack"] = kpack
    if nonkdim:
        extra["matrix_instr_nonkdim"] = nonkdim
    _sparse_mla_prefill_kernel[(t, head_split)](
        q_nope, q_rope, kv2, idx, out, sm_scale * 1.4426950408889634,
        q_nope.stride(0), q_nope.stride(1), q_rope.stride(0), q_rope.stride(1), kv2.stride(0), idx.stride(0),
        out.stride(1), out.stride(2),
        H=h, BLOCK_H=block_h, K=k, BLOCK_N=block_n, DV=d_v, DR=dr,
        num_warps=num_warps, num_stages=num_stages, **extra)
    return out


# ------------------------------------------------------------------------------------------------ split-K (decode)
@triton.jit
def _sparse_mla_split_kernel(q_ptr, kv_ptr, idx_ptr, po_ptr, plse_ptr, sm_scale_log2,
                             stride_qt, stride_qh, stride_kv, stride_it,
                             stride_pt, stride_ps, stride_ph, stride_lt, stride_ls,
                             H: tl.constexpr, BLOCK_H: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr,
                             SPLIT: tl.constexpr, DV: tl.constexpr, DR: tl.constexpr):
    """Rows x splits programs: each handles K / SPLIT keys of one row for all heads; writes a partial output and
    its log2-sum-exp (flash-decoding), combined by _sparse_mla_combine_kernel."""
    t = tl.program_id(0).to(tl.int64)
    sp = tl.program_id(1)
    offs_h = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)  # head group (head_split > 1)
    hmask = offs_h < H
    offs_d = tl.arange(0, DV)
    offs_r = tl.arange(0, DR)
    q_row = q_ptr + t * stride_qt + offs_h[:, None] * stride_qh
    q_nope = tl.load(q_row + offs_d[None, :], mask=hmask[:, None], other=0.0)
    q_rope = tl.load(q_row + DV + offs_r[None, :], mask=hmask[:, None], other=0.0)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, DV], tl.float32)
    idx_row = idx_ptr + t * stride_it
    per = K // SPLIT
    for n0 in range(sp * per, sp * per + per, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        idx = tl.load(idx_row + offs_n)
        valid = idx >= 0
        slot = tl.where(valid, idx, 0).to(tl.int64)
        k_base = kv_ptr + slot[:, None] * stride_kv
        k_nope = tl.load(k_base + offs_d[None, :], mask=valid[:, None], other=0.0)  # masked: see the prefill kernel
        k_rope = tl.load(k_base + DV + offs_r[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q_nope, tl.trans(k_nope))
        s = tl.dot(q_rope, tl.trans(k_rope), s)
        s = tl.where(valid[None, :], s * sm_scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p8 = (p * 448.0).to(tl.float8e4nv)
        acc = acc * alpha[:, None] + tl.dot(p8, k_nope).to(tl.float32) * (1.0 / 448.0)
        m_i = m_new
    denom = tl.where(l_i == 0.0, 1.0, l_i)
    lse = tl.where(l_i == 0.0, float("-inf"), tl.log2(denom) + tl.where(m_i == float("-inf"), 0.0, m_i))
    po = po_ptr + t * stride_pt + sp * stride_ps + offs_h[:, None] * stride_ph + offs_d[None, :]
    tl.store(po, acc / denom[:, None], mask=hmask[:, None])
    tl.store(plse_ptr + t * stride_lt + sp * stride_ls + offs_h, lse, mask=hmask)


@triton.jit
def _sparse_mla_split_pf_kernel(q_ptr, kv_ptr, idx_ptr, po_ptr, plse_ptr, sm_scale_log2,
                                stride_qt, stride_qh, stride_kv, stride_it,
                                stride_pt, stride_ps, stride_ph, stride_lt, stride_ls,
                                H: tl.constexpr, BLOCK_H: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr,
                                SPLIT: tl.constexpr, DV: tl.constexpr, DR: tl.constexpr):
    """_sparse_mla_split_kernel with the next key block's index load and row gather issued BEFORE the current block's
    MFMAs (software double buffering, rtxsys 2026-10-03). The plain kernel is gather-latency bound: each 64-key block
    is an index load, a dependent 64 x 576 B gather, then the math, strictly in series (amdglm: ~7x above the KV byte
    floor at 72 rows; num_stages 1-3 did not change it, the AMD stream pipeliner does not overlap these indirect
    loads). Same arithmetic in the same order as the plain kernel, so the outputs match it."""
    t = tl.program_id(0).to(tl.int64)
    sp = tl.program_id(1)
    offs_h = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = offs_h < H
    offs_d = tl.arange(0, DV)
    offs_r = tl.arange(0, DR)
    q_row = q_ptr + t * stride_qt + offs_h[:, None] * stride_qh
    q_nope = tl.load(q_row + offs_d[None, :], mask=hmask[:, None], other=0.0)
    q_rope = tl.load(q_row + DV + offs_r[None, :], mask=hmask[:, None], other=0.0)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, DV], tl.float32)
    idx_row = idx_ptr + t * stride_it
    per = K // SPLIT
    start = sp * per
    end = start + per
    offs_b = tl.arange(0, BLOCK_N)
    idx = tl.load(idx_row + start + offs_b)
    valid = idx >= 0
    slot = tl.where(valid, idx, 0).to(tl.int64)
    k_nope = tl.load(kv_ptr + slot[:, None] * stride_kv + offs_d[None, :], mask=valid[:, None], other=0.0)
    k_rope = tl.load(kv_ptr + slot[:, None] * stride_kv + DV + offs_r[None, :], mask=valid[:, None], other=0.0)
    for n0 in range(start, end, BLOCK_N):
        nxt = n0 + BLOCK_N
        has_next = nxt < end
        idx_n = tl.load(idx_row + nxt + offs_b, mask=(offs_b >= 0) & has_next, other=-1)
        valid_n = idx_n >= 0
        slot_n = tl.where(valid_n, idx_n, 0).to(tl.int64)
        k_nope_n = tl.load(kv_ptr + slot_n[:, None] * stride_kv + offs_d[None, :], mask=valid_n[:, None], other=0.0)
        k_rope_n = tl.load(kv_ptr + slot_n[:, None] * stride_kv + DV + offs_r[None, :], mask=valid_n[:, None],
                           other=0.0)
        s = tl.dot(q_nope, tl.trans(k_nope))
        s = tl.dot(q_rope, tl.trans(k_rope), s)
        s = tl.where(valid[None, :], s * sm_scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p8 = (p * 448.0).to(tl.float8e4nv)
        acc = acc * alpha[:, None] + tl.dot(p8, k_nope).to(tl.float32) * (1.0 / 448.0)
        m_i = m_new
        k_nope = k_nope_n
        k_rope = k_rope_n
        valid = valid_n
    denom = tl.where(l_i == 0.0, 1.0, l_i)
    lse = tl.where(l_i == 0.0, float("-inf"), tl.log2(denom) + tl.where(m_i == float("-inf"), 0.0, m_i))
    po = po_ptr + t * stride_pt + sp * stride_ps + offs_h[:, None] * stride_ph + offs_d[None, :]
    tl.store(po, acc / denom[:, None], mask=hmask[:, None])
    tl.store(plse_ptr + t * stride_lt + sp * stride_ls + offs_h, lse, mask=hmask)


@triton.jit
def _sparse_mla_combine_kernel(po_ptr, plse_ptr, o_ptr, stride_pt, stride_ps, stride_ph, stride_lt, stride_ls,
                               stride_ot, stride_oh, H: tl.constexpr, SPLIT: tl.constexpr, DV: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLIT)
    offs_d = tl.arange(0, DV)
    lse = tl.load(plse_ptr + t * stride_lt + offs_s * stride_ls + h)  # [S]
    m = tl.max(lse, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse - m_safe)
    wsum = tl.sum(w, 0)
    w = w / tl.where(wsum == 0.0, 1.0, wsum)
    po = tl.load(po_ptr + t * stride_pt + offs_s[:, None] * stride_ps + h * stride_ph + offs_d[None, :])  # [S, DV]
    out = tl.sum(po * w[:, None], 0)
    tl.store(o_ptr + t * stride_ot + h * stride_oh + offs_d, out.to(tl.bfloat16))


def sparse_mla_split_fp8(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float,
                         d_v: int = 512, split: int = 8, block_n: int = 64, num_warps: int = 4,
                         kpack: int = 2, nonkdim: int = 16, head_split: int = 1, num_stages: int = 0,
                         waves_per_eu: int = 0, prefetch: bool = False) -> torch.Tensor:
    """Split-K version for few rows (decode / verify): rows x `split` x `head_split` programs + a combine. Same output
    layout. head_split > 1 gives each program H / head_split heads (smaller accumulator, more programs resident per
    CU); num_stages / waves_per_eu 0 = the compiler default. prefetch=True runs the software double-buffered kernel
    (next block's indices + gather issued before the current block's math; same arithmetic). Defaults reproduce the
    original kernel launch."""
    t, h, d = q.shape
    k = indices.shape[-1]
    assert k % (split * block_n) == 0, (k, split, block_n)
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
    q = q.contiguous()
    idx = indices.reshape(t, k).contiguous()
    kv2 = kv.view(kv.shape[0], d)
    out = torch.empty(1, t, h, d_v, dtype=torch.bfloat16, device=q.device)
    if t == 0:
        return out
    assert h % head_split == 0, (h, head_split)
    block_h = max(16, triton.next_power_of_2(h // head_split))
    po = torch.empty(t, split, h, d_v, dtype=torch.float32, device=q.device)
    plse = torch.empty(t, split, h, dtype=torch.float32, device=q.device)
    extra = {}
    if num_stages:
        extra["num_stages"] = num_stages
    if waves_per_eu:
        extra["waves_per_eu"] = waves_per_eu
    kern = _sparse_mla_split_pf_kernel if prefetch else _sparse_mla_split_kernel
    kern[(t, split, head_split)](
        q, kv2, idx, po, plse, sm_scale * 1.4426950408889634,
        q.stride(0), q.stride(1), kv2.stride(0), idx.stride(0),
        po.stride(0), po.stride(1), po.stride(2), plse.stride(0), plse.stride(1),
        H=h, BLOCK_H=block_h, K=k, BLOCK_N=block_n, SPLIT=split, DV=d_v, DR=d - d_v,
        num_warps=num_warps, kpack=kpack, matrix_instr_nonkdim=nonkdim, **extra)
    _sparse_mla_combine_kernel[(t, h)](
        po, plse, out, po.stride(0), po.stride(1), po.stride(2), plse.stride(0), plse.stride(1),
        out.stride(1), out.stride(2), H=h, SPLIT=split, DV=d_v, num_warps=4)
    return out
