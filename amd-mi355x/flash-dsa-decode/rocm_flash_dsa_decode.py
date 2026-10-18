"""Split-K sparse MLA DECODE / EAGLE-verify kernel for GLM-5.3-Flash on gfx950 (agent rtxsys, 2026-10-04; coordinator
assignment 01:50: Flash decode per-request cost).

Why: k3amd's 32-vs-48 decode attribution (x_dp4g, one TP1 replica) puts the stock TileLang sparse decode ("main_kernel",
tilelang_sparse_fwd) at ~208 us per extra request per cycle: 11 DSA layers x 8 verify rows (MTP 7/1/8). amdflash's
rope-free Triton kernel (operations/amdflash/flash_sparse_prefill.py) only serves prefill. It has one program per row and
no split-K, it is routed only outside graph capture, and it reads slot 0 unmasked for invalid picks.

This module is the decode-shaped counterpart for Flash's layout: FP8 KV [slots, 1, 512], rope-free (tail_dim 0),
d_v 512, top-2048 indices.
- One program = (row, split, head group); it walks K / SPLIT keys of the row's top-k for all heads of the group and
  writes a partial output plus its log2-sum-exp. The combine kernel merges the splits (flash-decoding).
- Invalid (-1) picks are MASKED loads (other=0), never a slot-0 read: slot 0 is the padded rows' KV target
  (GLM-5.3 Value idle-rank lesson, 2026-10-03).
- Graph-capture safe: no host sync; out / partials are allocated with torch.empty.

Same arithmetic as the prefill kernels (FP8 QK^T, online softmax in fp32, P * 448 -> FP8 PV / 448), so it matches
flash_sparse_prefill_ref.

    flash_sparse_decode_fp8(q [T, H, 512] bf16|fp8, kv [slots, 1, 512] fp8, indices [T, 1, K] int32, sm_scale,
                            split=8, head_split=1) -> [1, T, H, 512] bf16
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _flash_dsa_split_kernel(q_ptr, kv_ptr, idx_ptr, po_ptr, plse_ptr, sm_scale_log2,
                            stride_qt, stride_qh, stride_kv, stride_it,
                            stride_pt, stride_ps, stride_ph, stride_lt, stride_ls,
                            H: tl.constexpr, BLOCK_H: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr,
                            SPLIT: tl.constexpr, DV: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    sp = tl.program_id(1)
    offs_h = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = offs_h < H
    offs_d = tl.arange(0, DV)
    q = tl.load(q_ptr + t * stride_qt + offs_h[:, None] * stride_qh + offs_d[None, :], mask=hmask[:, None],
                other=0.0).to(tl.float8e4nv)
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
        kv = tl.load(kv_ptr + slot[:, None] * stride_kv + offs_d[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kv))
        s = tl.where(valid[None, :], s * sm_scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p8 = (p * 448.0).to(tl.float8e4nv)
        acc = acc * alpha[:, None] + tl.dot(p8, kv).to(tl.float32) * (1.0 / 448.0)
        m_i = m_new
    denom = tl.where(l_i == 0.0, 1.0, l_i)
    lse = tl.where(l_i == 0.0, float("-inf"), tl.log2(denom) + tl.where(m_i == float("-inf"), 0.0, m_i))
    po = po_ptr + t * stride_pt + sp * stride_ps + offs_h[:, None] * stride_ph + offs_d[None, :]
    tl.store(po, acc / denom[:, None], mask=hmask[:, None])
    tl.store(plse_ptr + t * stride_lt + sp * stride_ls + offs_h, lse, mask=hmask)


@triton.jit
def _flash_dsa_combine_kernel(po_ptr, plse_ptr, o_ptr, stride_pt, stride_ps, stride_ph, stride_lt, stride_ls,
                              stride_ot, stride_oh, SPLIT: tl.constexpr, DV: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLIT)
    offs_d = tl.arange(0, DV)
    lse = tl.load(plse_ptr + t * stride_lt + offs_s * stride_ls + h)
    m = tl.max(lse, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse - m_safe)
    wsum = tl.sum(w, 0)
    w = w / tl.where(wsum == 0.0, 1.0, wsum)
    po = tl.load(po_ptr + t * stride_pt + offs_s[:, None] * stride_ps + h * stride_ph + offs_d[None, :])
    tl.store(o_ptr + t * stride_ot + h * stride_oh + offs_d, tl.sum(po * w[:, None], 0).to(tl.bfloat16))


def flash_sparse_decode_fp8(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float,
                            d_v: int = 512, split: int = 8, head_split: int = 1, block_n: int = 64,
                            num_warps: int = 4, num_stages: int = 0, kpack: int = 2,
                            nonkdim: int = 16) -> torch.Tensor:
    assert kv.dtype == torch.float8_e4m3fn and kv.dim() == 3 and kv.shape[1] == 1 and kv.shape[2] == d_v, (
        kv.dtype, kv.shape)
    t, h, dq = q.shape
    k = indices.shape[-1]
    assert dq == d_v and k % (split * block_n) == 0 and q.stride(-1) == 1, (q.shape, k, split, block_n)
    assert h % head_split == 0, (h, head_split)
    idx = indices.reshape(t, k)
    if not idx.is_contiguous():
        idx = idx.contiguous()
    kv2 = kv.view(kv.shape[0], d_v)
    out = torch.empty(1, t, h, d_v, dtype=torch.bfloat16, device=q.device)
    if t == 0:
        return out
    po = torch.empty(t, split, h, d_v, dtype=torch.float32, device=q.device)
    plse = torch.empty(t, split, h, dtype=torch.float32, device=q.device)
    extra = {}
    if torch.version.hip:
        extra = {"kpack": kpack, "matrix_instr_nonkdim": nonkdim} if kpack else {}
    if num_stages:
        extra["num_stages"] = num_stages
    _flash_dsa_split_kernel[(t, split, head_split)](
        q, kv2, idx, po, plse, sm_scale * 1.4426950408889634,
        q.stride(0), q.stride(1), kv2.stride(0), idx.stride(0),
        po.stride(0), po.stride(1), po.stride(2), plse.stride(0), plse.stride(1),
        H=h, BLOCK_H=max(16, triton.next_power_of_2(h // head_split)), K=k, BLOCK_N=block_n, SPLIT=split,
        DV=d_v, num_warps=num_warps, **extra)
    _flash_dsa_combine_kernel[(t, h)](
        po, plse, out, po.stride(0), po.stride(1), po.stride(2), plse.stride(0), plse.stride(1),
        out.stride(1), out.stride(2), SPLIT=split, DV=d_v, num_warps=4)
    return out


# rows -> split, measured on mi355x8b GPU 4, 2026-10-04 (64 heads, top-2048, CUDA graphs; us vs TileLang):
# 64 rows split4 46.9 (TileLang 100.0); 128 split2 78.8 (173.5); 256 split1 137.2 (337.3); 384 split2 231.0 (468.9)
SPLIT_TABLE = ((96, 4), (192, 2), (320, 1), (1 << 30, 2))


def split_for_rows(rows: int) -> int:
    import os
    table = SPLIT_TABLE
    env = os.environ.get("NUMINOUS_FLASH_DSA_DECODE_SPLIT_TABLE")
    if env:
        table = tuple(tuple(int(v) for v in item.split(":")) for item in env.split(","))
    for limit, split in table:
        if rows <= limit:
            return split
    return table[-1][1]


def flash_sparse_decode_ref(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float,
                            d_v: int = 512) -> torch.Tensor:
    """fp32 reference over the valid picks with the kernel's quantization points (q -> fp8; P * 448 -> fp8 per split
    is not modeled; compare with a tolerance)."""
    t, h, _ = q.shape
    k = indices.shape[-1]
    qf = q.to(torch.float8_e4m3fn).float()
    kvf = kv.view(kv.shape[0], d_v).float()
    idx = indices.reshape(t, k).long()
    valid = idx >= 0
    rows = torch.where(valid[..., None], kvf[idx.clamp(min=0)], torch.zeros(()))
    s = torch.einsum("thd,tkd->thk", qf, rows) * sm_scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(s, -1).nan_to_num(0.0)
    return torch.einsum("thk,tkd->thd", p, rows).to(torch.bfloat16).unsqueeze(0)
