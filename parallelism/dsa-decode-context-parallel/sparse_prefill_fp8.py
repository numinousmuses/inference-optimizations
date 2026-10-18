"""Sparse MLA prefill over the FlashMLA fp8 KV layout for small head counts.

FlashMLA's sparse prefill kernel (`flash_mla_sparse_fwd`) tiles 128 query heads on SM100 and wants bf16 KV, so a
rank holding 8 or 16 heads pays for 128 and the KV has to be dequantised first. Under decode context parallel every
rank holds a slice of the heads or a slice of the tokens, so that padding is what made DCP prefill slower than TP.

This kernel takes the fp8 rows as stored (512 fp8 nope | 4 fp32 scales over 128-wide tiles | 64 bf16 rope = 656
bytes) and any head count that is a multiple of 8 (padded to 16 in registers when smaller), with per-row valid
lengths so a row can carry fewer than the top-k width. One program handles one query token and all its heads,
streaming the selected rows in blocks: gather, dequantise in registers, online softmax, accumulate.

Returns the normalised output and the log-sum-exp (base e) so a caller can merge partials across ranks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

DIM_NOPE = tl.constexpr(512)
DIM_ROPE = tl.constexpr(64)
NUM_TILES = tl.constexpr(4)
DIM_QK = 576
ROW_BYTES = 656  # 512 fp8 nope | 4 fp32 scales | 64 bf16 rope


@triton.jit
def _sparse_prefill_fp8_kernel(
    q_ptr,            # [T, H, 576] bf16 (nope 512 | rope 64)
    kv_ptr,           # [N, 656] uint8
    idx_ptr,          # [T, K] int32, -1 = masked
    len_ptr,          # [T] int32 valid entries per row (prefix of the row), or K when null
    o_ptr,            # [T, S, HP, 512] bf16 partial (normalised within the split)
    lse_ptr,          # [T, S, HP] fp32 partial lse
    K: tl.constexpr,
    KS: tl.constexpr,       # entries per split
    H: tl.constexpr,
    HP: tl.constexpr,       # heads padded to a power of two >= 16
    BLOCK_N: tl.constexpr,  # rows per iteration
    sm_scale,
    has_len: tl.constexpr,
    FP8_DOT: tl.constexpr,  # q and k tiles multiplied as e4m3 on the tensor cores; p in e4m3 for PV
    stride_qt, stride_qh, stride_kv, stride_it,
):
    t = tl.program_id(0)
    sp = tl.program_id(1)
    S: tl.constexpr = K // KS
    offs_h = tl.arange(0, HP)
    h_mask = offs_h < H
    offs_nope = tl.arange(0, DIM_NOPE)
    offs_rope = tl.arange(0, DIM_ROPE)
    offs_n = tl.arange(0, BLOCK_N)

    q_base = q_ptr + t * stride_qt + offs_h[:, None] * stride_qh
    q_rope = tl.load(q_base + DIM_NOPE + offs_rope[None, :], mask=h_mask[:, None], other=0.0)

    if has_len:
        valid = tl.load(len_ptr + t)
    else:
        valid = K

    m_i = tl.full([HP], float("-inf"), tl.float32)
    l_i = tl.zeros([HP], tl.float32)

    offs_tile_d = tl.arange(0, 128)
    # per-tile query slices (the nope dim is 4 tiles of 128; each K row carries one fp32 scale per tile)
    q_n0 = tl.load(q_base + offs_tile_d[None, :], mask=h_mask[:, None], other=0.0)
    q_n1 = tl.load(q_base + 128 + offs_tile_d[None, :], mask=h_mask[:, None], other=0.0)
    q_n2 = tl.load(q_base + 256 + offs_tile_d[None, :], mask=h_mask[:, None], other=0.0)
    q_n3 = tl.load(q_base + 384 + offs_tile_d[None, :], mask=h_mask[:, None], other=0.0)
    if FP8_DOT:
        qmax = tl.maximum(tl.maximum(tl.max(tl.abs(q_n0.to(tl.float32)), axis=1), tl.max(tl.abs(q_n1.to(tl.float32)), axis=1)),
                          tl.maximum(tl.max(tl.abs(q_n2.to(tl.float32)), axis=1), tl.max(tl.abs(q_n3.to(tl.float32)), axis=1)))
        qs = tl.where(qmax > 0, qmax / 448.0, 1.0)                       # [HP] per-head scale
        q8_0 = (q_n0.to(tl.float32) / qs[:, None]).to(tl.float8e4nv)
        q8_1 = (q_n1.to(tl.float32) / qs[:, None]).to(tl.float8e4nv)
        q8_2 = (q_n2.to(tl.float32) / qs[:, None]).to(tl.float8e4nv)
        q8_3 = (q_n3.to(tl.float32) / qs[:, None]).to(tl.float8e4nv)
    acc0 = tl.zeros([HP, 128], tl.float32)
    acc1 = tl.zeros([HP, 128], tl.float32)
    acc2 = tl.zeros([HP, 128], tl.float32)
    acc3 = tl.zeros([HP, 128], tl.float32)

    # indices of this split, loaded once; the row gathers of block i+1 are issued before block i's math
    start0 = sp * KS
    n_idx = start0 + offs_n
    n_ok = n_idx < valid
    rows_nxt = tl.load(idx_ptr + t * stride_it + n_idx, mask=n_ok, other=-1)
    ok_nxt = n_ok & (rows_nxt >= 0)
    for it in range(0, KS // BLOCK_N):
        rows = rows_nxt
        row_ok = ok_nxt
        rows_safe = tl.where(row_ok, rows, 0).to(tl.int64)
        row_base = kv_ptr + rows_safe[:, None] * stride_kv
        k8_0 = tl.load(row_base + offs_tile_d[None, :], mask=row_ok[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        k8_1 = tl.load(row_base + 128 + offs_tile_d[None, :], mask=row_ok[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        k8_2 = tl.load(row_base + 256 + offs_tile_d[None, :], mask=row_ok[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        k8_3 = tl.load(row_base + 384 + offs_tile_d[None, :], mask=row_ok[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        if not FP8_DOT:
            k0 = k8_0.to(tl.bfloat16)
            k1 = k8_1.to(tl.bfloat16)
            k2 = k8_2.to(tl.bfloat16)
            k3 = k8_3.to(tl.bfloat16)
        sc_ptr = (kv_ptr + rows_safe * stride_kv + DIM_NOPE).to(tl.pointer_type(tl.float32))
        sc0 = tl.load(sc_ptr, mask=row_ok, other=0.0)
        sc1 = tl.load(sc_ptr + 1, mask=row_ok, other=0.0)
        sc2 = tl.load(sc_ptr + 2, mask=row_ok, other=0.0)
        sc3 = tl.load(sc_ptr + 3, mask=row_ok, other=0.0)
        rope_ptr = (kv_ptr + rows_safe[:, None] * stride_kv + DIM_NOPE + NUM_TILES * 4).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(rope_ptr + offs_rope[None, :], mask=row_ok[:, None], other=0.0)
        # next block's indices (independent of this block's math)
        n_idx = start0 + (it + 1) * BLOCK_N + offs_n
        n_ok = (n_idx < valid) & (n_idx < start0 + KS)
        rows_nxt = tl.load(idx_ptr + t * stride_it + n_idx, mask=n_ok, other=-1)
        ok_nxt = n_ok & (rows_nxt >= 0)

        if FP8_DOT:
            s = tl.dot(q8_0, tl.trans(k8_0), out_dtype=tl.float32) * sc0[None, :]
            s += tl.dot(q8_1, tl.trans(k8_1), out_dtype=tl.float32) * sc1[None, :]
            s += tl.dot(q8_2, tl.trans(k8_2), out_dtype=tl.float32) * sc2[None, :]
            s += tl.dot(q8_3, tl.trans(k8_3), out_dtype=tl.float32) * sc3[None, :]
            s = s * qs[:, None]
        else:
            s = tl.dot(q_n0, tl.trans(k0), out_dtype=tl.float32) * sc0[None, :]
            s += tl.dot(q_n1, tl.trans(k1), out_dtype=tl.float32) * sc1[None, :]
            s += tl.dot(q_n2, tl.trans(k2), out_dtype=tl.float32) * sc2[None, :]
            s += tl.dot(q_n3, tl.trans(k3), out_dtype=tl.float32) * sc3[None, :]
        s += tl.dot(q_rope, tl.trans(k_rope), out_dtype=tl.float32)
        s = s * sm_scale
        s = tl.where(row_ok[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        p = tl.where(row_ok[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if FP8_DOT:
            # p is in [0, 1]; e4m3 keeps ~2 significant digits, the per-tile scale rides on p to keep k as stored
            acc0 = acc0 * alpha[:, None] + tl.dot((p * sc0[None, :]).to(tl.float8e4nv), k8_0, out_dtype=tl.float32)
            acc1 = acc1 * alpha[:, None] + tl.dot((p * sc1[None, :]).to(tl.float8e4nv), k8_1, out_dtype=tl.float32)
            acc2 = acc2 * alpha[:, None] + tl.dot((p * sc2[None, :]).to(tl.float8e4nv), k8_2, out_dtype=tl.float32)
            acc3 = acc3 * alpha[:, None] + tl.dot((p * sc3[None, :]).to(tl.float8e4nv), k8_3, out_dtype=tl.float32)
        else:
            acc0 = acc0 * alpha[:, None] + tl.dot((p * sc0[None, :]).to(tl.bfloat16), k0, out_dtype=tl.float32)
            acc1 = acc1 * alpha[:, None] + tl.dot((p * sc1[None, :]).to(tl.bfloat16), k1, out_dtype=tl.float32)
            acc2 = acc2 * alpha[:, None] + tl.dot((p * sc2[None, :]).to(tl.bfloat16), k2, out_dtype=tl.float32)
            acc3 = acc3 * alpha[:, None] + tl.dot((p * sc3[None, :]).to(tl.bfloat16), k3, out_dtype=tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    lse = tl.where(l_i > 0, m_i + tl.log(l_safe), float("-inf"))
    o_base = o_ptr + ((t * S + sp) * HP + offs_h[:, None]) * DIM_NOPE
    tl.store(o_base + offs_tile_d[None, :], (acc0 / l_safe[:, None]).to(tl.bfloat16))
    tl.store(o_base + 128 + offs_tile_d[None, :], (acc1 / l_safe[:, None]).to(tl.bfloat16))
    tl.store(o_base + 256 + offs_tile_d[None, :], (acc2 / l_safe[:, None]).to(tl.bfloat16))
    tl.store(o_base + 384 + offs_tile_d[None, :], (acc3 / l_safe[:, None]).to(tl.bfloat16))
    tl.store(lse_ptr + (t * S + sp) * HP + offs_h, lse)


@triton.jit
def _combine_splits_kernel(
    po_ptr,   # [T, S, HP, 512] bf16
    plse_ptr, # [T, S, HP] fp32
    o_ptr,    # [T, H, 512] bf16
    lse_ptr,  # [T, H] fp32
    S: tl.constexpr, H: tl.constexpr, HP: tl.constexpr,
    stride_ot, stride_oh,
):
    t = tl.program_id(0)
    offs_h = tl.arange(0, HP)
    h_mask = offs_h < H
    offs_d = tl.arange(0, DIM_NOPE)
    m = tl.full([HP], float("-inf"), tl.float32)
    for sp in range(S):
        l = tl.load(plse_ptr + (t * S + sp) * HP + offs_h)
        m = tl.maximum(m, l)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    acc = tl.zeros([HP, DIM_NOPE], tl.float32)
    denom = tl.zeros([HP], tl.float32)
    for sp in range(S):
        l = tl.load(plse_ptr + (t * S + sp) * HP + offs_h)
        w = tl.where(l == float("-inf"), 0.0, tl.exp(l - m_safe))
        po = tl.load(po_ptr + ((t * S + sp) * HP + offs_h[:, None]) * DIM_NOPE + offs_d[None, :]).to(tl.float32)
        acc += po * w[:, None]
        denom += w
    d_safe = tl.where(denom > 0, denom, 1.0)
    out = acc / d_safe[:, None]
    lse = tl.where(denom > 0, m + tl.log(d_safe), float("-inf"))
    tl.store(o_ptr + t * stride_ot + offs_h[:, None] * stride_oh + offs_d[None, :], out.to(tl.bfloat16), mask=h_mask[:, None])
    tl.store(lse_ptr + t * H + offs_h, lse, mask=h_mask)


_NUM_WARPS = 4
_NUM_STAGES = 3


def sparse_prefill_fp8(
    q: torch.Tensor,
    kv_fp8_rows: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    lengths: torch.Tensor | None = None,
    block_n: int = 64,
    splits: int = 8,
    fp8_dot: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q [T, H, 576] bf16; kv_fp8_rows [N, 656] uint8 (FlashMLA fp8 layout); indices [T, K] int32 with -1 masked;
    lengths [T] int32 valid prefix per row (optional). Returns (o [T, H, 512] bf16, lse [T, H] fp32, base e).
    Split-K over the top-k width: grid (T, splits) then a combine pass, so a chunk of 8k tokens fills the GPU."""
    assert q.dim() == 3 and q.shape[-1] == DIM_QK, q.shape
    assert kv_fp8_rows.dim() == 2 and kv_fp8_rows.shape[-1] == ROW_BYTES, kv_fp8_rows.shape
    T, H, _ = q.shape
    K = indices.shape[1]
    assert indices.shape[0] == T
    assert K % (block_n * splits) == 0, (K, block_n, splits)
    HP = max(16, triton.next_power_of_2(H))
    q = q.contiguous()
    indices = indices.contiguous().to(torch.int32)
    kv_fp8_rows = kv_fp8_rows.contiguous()
    o = torch.empty((T, H, 512), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((T, H), dtype=torch.float32, device=q.device)
    if T == 0:
        return o, lse
    po = torch.empty((T, splits, HP, 512), dtype=torch.bfloat16, device=q.device)
    plse = torch.empty((T, splits, HP), dtype=torch.float32, device=q.device)
    has_len = lengths is not None
    len_arg = lengths.contiguous().to(torch.int32) if has_len else indices
    _sparse_prefill_fp8_kernel[(T, splits)](
        q, kv_fp8_rows, indices, len_arg, po, plse,
        K=K, KS=K // splits, H=H, HP=HP, BLOCK_N=block_n, sm_scale=sm_scale, has_len=has_len, FP8_DOT=fp8_dot,
        stride_qt=q.stride(0), stride_qh=q.stride(1), stride_kv=kv_fp8_rows.stride(0), stride_it=indices.stride(0),
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    _combine_splits_kernel[(T,)](po, plse, o, lse, S=splits, H=H, HP=HP, stride_ot=o.stride(0), stride_oh=o.stride(1), num_warps=4)
    return o, lse


def sparse_prefill_reference(q: torch.Tensor, kv_fp8_rows: torch.Tensor, indices: torch.Tensor, sm_scale: float,
                             lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain torch reference (fp32) for tests: dequantise the selected rows and attend over them."""
    T, H, _ = q.shape
    K = indices.shape[1]
    rows = kv_fp8_rows
    nope_q = rows[:, :512].view(torch.float8_e4m3fn).float()
    scales = rows[:, 512:528].contiguous().view(torch.float32)
    nope = (nope_q.view(-1, 4, 128) * scales[:, :, None]).view(-1, 512)
    rope = rows[:, 528:].contiguous().view(torch.bfloat16).float()
    k_all = torch.cat([nope, rope], dim=-1)  # [N, 576]
    out = torch.zeros((T, H, 512), dtype=torch.float32, device=q.device)
    lse = torch.full((T, H), float("-inf"), dtype=torch.float32, device=q.device)
    qf = q.float()
    for t in range(T):
        idx = indices[t].long()
        valid = idx >= 0
        if lengths is not None:
            valid &= torch.arange(K, device=q.device) < lengths[t]
        sel = idx[valid]
        if sel.numel() == 0:
            continue
        k = k_all[sel]                                  # [n, 576]
        s = (qf[t] @ k.T) * sm_scale                    # [H, n]
        m = s.max(dim=-1, keepdim=True).values
        p = torch.exp(s - m)
        l = p.sum(dim=-1, keepdim=True)
        out[t] = (p / l) @ nope[sel]
        lse[t] = (m + torch.log(l)).squeeze(-1)
    return out.to(torch.bfloat16), lse
