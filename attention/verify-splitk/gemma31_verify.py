"""GQA-packed split-K target verify for Gemma 4 on SM90 (rtxattn, kernel program A3 + A1; coordinator priority).

Measured (rtxv41, findings rtxv41_gemma31.md section 9; 1xH100, 32 sessions x 6 draft tokens, ~1.4K contexts):
  - The stock triton extend kernel (_fwd_kernel) serves TARGET_VERIFY on SM90. It takes 77 of the 111.5 ms step:
      - SWA (50 layers, hd 256, window 1024): 55.4 ms, 13.8x floor;
      - full (10 layers, hd 512, group 8): 21.6 ms, 38.6x floor.
  - It runs one program per (request, q head) over the whole prefix with BLOCK_M rows for 6 valid queries, so each
    kv head's prefix is read once per q head and the tensor-core M dimension is ~10% used.
  - The split/shared-KV verify kernels are gfx95-only (triton_backend.py:87-101, 219-233). The grouped decode
    kernel already does a 32-session decode step at 1.9x floor.

This kernel: one program per (request, kv head, split). Its rows are the request's N_EXT draft tokens x the group's
q heads (row r -> token r // GROUP, head kv_head * GROUP + r % GROUP). The prefix is read ONCE per kv head (M = 12
for SWA, 48 for full) and split across SMs. Split N_SPLITS is the draft-draft block (fresh k/v_extend, causal). An
LSE stage combines the splits.

Semantics match extend_attention_fwd for a topk-1 chain (where the tree custom mask equals causal, the argument
verify_splitkv.py makes):
  - prefix logits are scaled by sm_scale * k_scale, and the prefix PV by v_scale (FP8 pool); draft K/V unscaled;
  - sliding window (extend_attention.py:513-519, 705-710): a prefix key n of a request whose window list holds W'
    keys is visible to draft row m iff W' + m <= n + WINDOW. Draft keys t <= m always lie inside 6 << 1024.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _dot(a, b, LOWP: tl.constexpr):
    if LOWP:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a, b)


@triton.jit
def _verify_gqa_stage1(Q, K_ext, V_ext, K_buf, V_buf, qo_indptr, kv_indptr, kv_indices, Att_Out, Att_Lse,
                       sm_scale, k_scale, v_scale,
                       stride_qt, stride_qh, stride_ket, stride_keh, stride_vet, stride_veh,
                       stride_kbs, stride_kbh, stride_vbs, stride_vbh, stride_ob, stride_or, stride_os,
                       GROUP: tl.constexpr, N_EXT: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
                       BLOCK_N: tl.constexpr, N_SPLITS: tl.constexpr, WINDOW: tl.constexpr, LOWP: tl.constexpr):
    cur_seq = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    split = tl.program_id(2)
    q_start = tl.load(qo_indptr + cur_seq)
    kv_start = tl.load(kv_indptr + cur_seq)
    prefix_len = tl.load(kv_indptr + cur_seq + 1) - kv_start

    rows = tl.arange(0, ROWS)
    tok = rows // GROUP
    head = cur_kv_head * GROUP + rows % GROUP
    mask_r = tok < N_EXT
    offs_d = tl.arange(0, D)
    q = tl.load(Q + (q_start + tok[:, None]) * stride_qt + head[:, None] * stride_qh + offs_d[None, :],
                mask=mask_r[:, None], other=0.0).to(tl.float32)

    e_max = tl.zeros([ROWS], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([ROWS], dtype=tl.float32)
    acc = tl.zeros([ROWS, D], dtype=tl.float32)

    if split < N_SPLITS:
        split_len = tl.cdiv(prefix_len, N_SPLITS)
        lo_n = split * split_len
        hi_n = tl.minimum(lo_n + split_len, prefix_len)
        if WINDOW > 0:
            # keys older than the first draft row's window are invisible to every row: skip their tiles
            lo_n = tl.maximum(lo_n, prefix_len - WINDOW)
        for start_n in range(lo_n, hi_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = (offs_n < hi_n) & (offs_n >= lo_n)
            loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0).to(tl.int64)  # pool offsets > 2^31
            k = tl.load(K_buf + loc[:, None] * stride_kbs + cur_kv_head * stride_kbh + offs_d[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
            v = tl.load(V_buf + loc[:, None] * stride_vbs + cur_kv_head * stride_vbh + offs_d[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
            qk = _dot(q, tl.trans(k), LOWP) * (sm_scale * k_scale)
            keep = mask_r[:, None] & mask_n[None, :]
            if WINDOW > 0:
                keep = keep & ((prefix_len + tok[:, None]) <= (offs_n[None, :] + WINDOW))
            qk = tl.where(keep, qk, float("-inf"))
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            m_safe = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
            re_scale = tl.exp(e_max - m_safe)
            p = tl.exp(qk - m_safe[:, None])
            p = tl.where(keep, p, 0.0)
            acc = acc * re_scale[:, None] + _dot(p, v, LOWP) * v_scale
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
    else:
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_EXT
        k = tl.load(K_ext + (q_start + offs_n[:, None]) * stride_ket + cur_kv_head * stride_keh + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        v = tl.load(V_ext + (q_start + offs_n[:, None]) * stride_vet + cur_kv_head * stride_veh + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        qk = _dot(q, tl.trans(k), LOWP) * sm_scale
        keep = mask_r[:, None] & mask_n[None, :] & (offs_n[None, :] <= tok[:, None])
        qk = tl.where(keep, qk, float("-inf"))
        e_max = tl.max(qk, 1)
        p = tl.where(keep, tl.exp(qk - e_max[:, None]), 0.0)
        acc = _dot(p, v, LOWP)
        e_sum = tl.sum(p, 1)

    has = e_sum > 0
    denom = tl.where(has, e_sum, 1.0)
    o_base = Att_Out + cur_seq * stride_ob + (cur_kv_head * ROWS + rows[:, None]) * stride_or + split * stride_os
    tl.store(o_base + offs_d[None, :], acc / denom[:, None], mask=mask_r[:, None])
    lse = tl.where(has, e_max + tl.log(denom), float("-inf"))
    lse_off = ((cur_seq * tl.num_programs(1) + cur_kv_head) * ROWS + rows) * (N_SPLITS + 1) + split
    tl.store(Att_Lse + lse_off, lse, mask=mask_r)


@triton.jit
def _verify_gqa_stage2(Att_Out, Att_Lse, O, stride_ob, stride_or, stride_os, stride_ot, stride_oh, qo_indptr,
                       GROUP: tl.constexpr, N_EXT: tl.constexpr, ROWS: tl.constexpr, D: tl.constexpr,
                       N_SPLITS: tl.constexpr):
    cur_seq = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    r = tl.program_id(2)
    tok = r // GROUP
    head = cur_kv_head * GROUP + r % GROUP
    if tok < N_EXT:
        offs_d = tl.arange(0, D)
        q_start = tl.load(qo_indptr + cur_seq)
        lse_base = Att_Lse + ((cur_seq * tl.num_programs(1) + cur_kv_head) * ROWS + r) * (N_SPLITS + 1)
        o_base = Att_Out + cur_seq * stride_ob + (cur_kv_head * ROWS + r) * stride_or
        e_max = -float("inf")
        e_sum = 0.0
        acc = tl.zeros([D], dtype=tl.float32)
        for s in range(0, N_SPLITS + 1):
            lse = tl.load(lse_base + s)
            if lse > -float("inf"):
                part = tl.load(o_base + s * stride_os + offs_d)
                n_e_max = tl.maximum(lse, e_max)
                old = tl.exp(e_max - n_e_max)
                w = tl.exp(lse - n_e_max)
                acc = acc * old + w * part
                e_sum = e_sum * old + w
                e_max = n_e_max
        tl.store(O + (q_start + tok) * stride_ot + head * stride_oh + offs_d, (acc / e_sum).to(O.dtype.element_ty))


def choose_splits(n_prefix_tokens: int, bs: int, hkv: int, window: int, sm_count: int = 132) -> int:
    """Static (CUDA-graph safe) split count from shapes only: enough (request, kv head, split) programs to cover the
    SMs, each with >= ~256 keys; a window caps the useful prefix."""
    avg = n_prefix_tokens // max(bs, 1)
    if window > 0:
        avg = min(avg, window)
    want = max(1, triton.cdiv(sm_count, max(1, bs * hkv)))
    by_len = max(1, avg // 256)
    return int(min(16, triton.next_power_of_2(min(want, by_len))))


def verify_gqa(q, k_extend, v_extend, k_buffer, v_buffer, o, qo_indptr, kv_indptr, kv_indices, n_ext, sm_scale,
               k_scale=1.0, v_scale=1.0, window=0, n_splits=None, block_n=None, num_warps=None, num_stages=None):
    """q [bs*n_ext, Hq, D]; k/v_extend [bs*n_ext, Hkv, D] (the draft tokens, bf16); k/v_buffer [pool, Hkv, D];
    kv_indptr / kv_indices: the cached prefix (the window list on SWA layers). o written [bs*n_ext, Hq, D]."""
    nq, hq, d = q.shape
    hkv = k_extend.shape[1]
    group = hq // hkv
    bs = qo_indptr.shape[0] - 1
    assert nq == bs * n_ext, "verify_gqa needs a constant draft length per request (topk 1)"
    rows = max(16, triton.next_power_of_2(n_ext * group))
    # launch defaults MEASURED on 1xH100 (gemma31_verify_bench; sweep of 72 configs at the full-layer shape, 32
    # sessions x 6, 1.4K ctx): hd 512 -> 8 warps, block_n 64, 2 stages (308-312 us; 4 warps / block_n 32 / 3 stages
    # was 518.7); hd 256 (SWA) -> 4 warps, block_n 32, 3 stages (150.0 us; 2 stages measured 245)
    block_n = block_n or (16 if _INTERPRET else (64 if d >= 512 else 32))
    block_n = max(block_n, triton.next_power_of_2(n_ext))
    if n_splits is None:
        n_splits = choose_splits(kv_indices.shape[0], bs, hkv, window)
    launch = {}
    if not _INTERPRET:
        launch = {"num_warps": num_warps or (8 if d >= 512 else 4), "num_stages": num_stages or (2 if d >= 512 else 3)}
    att = torch.empty((bs, hkv * rows, n_splits + 1, d), dtype=torch.float32, device=q.device)
    lse = torch.empty((bs * hkv * rows * (n_splits + 1),), dtype=torch.float32, device=q.device)
    _verify_gqa_stage1[(bs, hkv, n_splits + 1)](
        q, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, att, lse,
        float(sm_scale), float(k_scale), float(v_scale),
        q.stride(0), q.stride(1), k_extend.stride(0), k_extend.stride(1), v_extend.stride(0), v_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1), v_buffer.stride(0), v_buffer.stride(1),
        att.stride(0), att.stride(1), att.stride(2),
        GROUP=group, N_EXT=n_ext, D=d, ROWS=rows, BLOCK_N=block_n, N_SPLITS=n_splits, WINDOW=int(max(0, window)),
        LOWP=not _INTERPRET, **launch)
    _verify_gqa_stage2[(bs, hkv, rows)](
        att, lse, o, att.stride(0), att.stride(1), att.stride(2), o.stride(0), o.stride(1), qo_indptr,
        GROUP=group, N_EXT=n_ext, ROWS=rows, D=d, N_SPLITS=n_splits)
    return o


def verify_ref(q, k_ext, v_ext, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices, n_ext, sm_scale, k_scale=1.0,
               v_scale=1.0, window=0):
    """Reference: per request, prefix keys from the pool (scaled), draft keys causal, window per extend_attention."""
    hq, hkv = q.shape[1], k_ext.shape[1]
    g = hq // hkv
    out = torch.empty(q.shape, dtype=torch.float32)
    for b in range(qo_indptr.shape[0] - 1):
        s, e = int(qo_indptr[b]), int(qo_indptr[b + 1])
        a, z = int(kv_indptr[b]), int(kv_indptr[b + 1])
        loc = kv_indices[a:z].long()
        kp, vp = k_buf[loc].float() * k_scale, v_buf[loc].float() * v_scale
        k = torch.cat([kp, k_ext[s:e].float()], 0).repeat_interleave(g, 1)
        v = torch.cat([vp, v_ext[s:e].float()], 0).repeat_interleave(g, 1)
        w = z - a
        sc = torch.einsum("qhd,khd->hqk", q[s:e].float(), k) * sm_scale
        m = torch.zeros(n_ext, w + n_ext, dtype=torch.bool)
        for i in range(n_ext):
            for j in range(w):
                m[i, j] = (window <= 0) or (w + i <= j + window)
            m[i, w:w + i + 1] = True
        sc = sc.masked_fill(~m[None], float("-inf"))
        out[s:e] = torch.einsum("hqk,khd->qhd", torch.softmax(sc, -1), v)
    return out


# ------------------------------------------------------------------------------------------------ SWA prefill
@triton.jit
def _extend_gqa_window(Q, K_ext, V_ext, K_buf, V_buf, O, qo_indptr, kv_indptr, kv_indices, KMAX, sm_scale, k_scale, v_scale,
                       stride_qt, stride_qh, stride_ket, stride_keh, stride_vet, stride_veh,
                       stride_kbs, stride_kbh, stride_vbs, stride_vbh, stride_ot, stride_oh,
                       GROUP: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       WINDOW: tl.constexpr, LOWP: tl.constexpr, D_SPLIT: tl.constexpr = 1,
                       USE_KMAX: tl.constexpr = False):
    """Prefill / extend for sliding-window layers, GQA-packed: one program = BLOCK_M new tokens x the GROUP q heads of
    one kv head, so K/V tiles load once for the group. The prefix is the window list (kv_indices, W' keys) and the
    new tokens are causal. extend_attention's window rule: q_id <= kv_id + WINDOW, with q_id = W' + m on the prefix
    and m on the new tokens. Both loops start at the first tile the block's first row can see."""
    cur_seq = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    pid2 = tl.program_id(2)
    cur_block_m = pid2 // D_SPLIT
    d_part = pid2 % D_SPLIT                 # split-D: this program accumulates output dims [d_part*DV, +DV)
    ROWS: tl.constexpr = BLOCK_M * GROUP
    DV: tl.constexpr = D // D_SPLIT
    q_start = tl.load(qo_indptr + cur_seq)
    ext_len = tl.load(qo_indptr + cur_seq + 1) - q_start
    kv_start = tl.load(kv_indptr + cur_seq)
    prefix_len = tl.load(kv_indptr + cur_seq + 1) - kv_start
    if cur_block_m * BLOCK_M < ext_len:
        rows = tl.arange(0, ROWS)
        offs_m = cur_block_m * BLOCK_M + rows // GROUP
        head = cur_kv_head * GROUP + rows % GROUP
        mask_m = offs_m < ext_len
        offs_d = tl.arange(0, D)
        offs_dv = d_part * DV + tl.arange(0, DV)
        q = tl.load(Q + (q_start + offs_m[:, None]) * stride_qt + head[:, None] * stride_qh + offs_d[None, :],
                    mask=mask_m[:, None], other=0.0).to(tl.float32)
        e_max = tl.zeros([ROWS], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([ROWS], dtype=tl.float32)
        acc = tl.zeros([ROWS, DV], dtype=tl.float32)
        # prefix (window list)
        lo = 0
        if WINDOW > 0:
            lo = tl.maximum(0, prefix_len + cur_block_m * BLOCK_M - WINDOW) // BLOCK_N * BLOCK_N
        for start_n in range(lo, prefix_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < prefix_len
            loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0).to(tl.int64)
            k = tl.load(K_buf + loc[:, None] * stride_kbs + cur_kv_head * stride_kbh + offs_d[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
            v = tl.load(V_buf + loc[:, None] * stride_vbs + cur_kv_head * stride_vbh + offs_dv[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
            qk = _dot(q, tl.trans(k), LOWP) * (sm_scale * k_scale)
            keep = mask_m[:, None] & mask_n[None, :]
            if WINDOW > 0:
                keep = keep & ((prefix_len + offs_m[:, None]) <= (offs_n[None, :] + WINDOW))
            qk = tl.where(keep, qk, float("-inf"))
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            m_safe = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
            re_scale = tl.exp(e_max - m_safe)
            p = tl.where(keep, tl.exp(qk - m_safe[:, None]), 0.0)
            acc = acc * re_scale[:, None] + _dot(p, v, LOWP) * v_scale
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
        # new tokens: causal (or, for image batches, k <= kmax[q]: an image span is bidirectional) + window
        end_n = tl.minimum((cur_block_m + 1) * BLOCK_M, ext_len)
        if USE_KMAX:
            kmax = tl.load(KMAX + q_start + offs_m, mask=mask_m, other=0)
            end_n = tl.minimum(tl.maximum(end_n, tl.max(kmax, 0) + 1), ext_len)
        lo_e = 0
        if WINDOW > 0:
            lo_e = tl.maximum(0, cur_block_m * BLOCK_M - WINDOW) // BLOCK_N * BLOCK_N
        for start_n in range(lo_e, end_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < ext_len
            k = tl.load(K_ext + (q_start + offs_n[:, None]) * stride_ket + cur_kv_head * stride_keh + offs_d[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
            v = tl.load(V_ext + (q_start + offs_n[:, None]) * stride_vet + cur_kv_head * stride_veh + offs_dv[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
            qk = _dot(q, tl.trans(k), LOWP) * sm_scale
            if USE_KMAX:
                keep = mask_m[:, None] & mask_n[None, :] & (offs_n[None, :] <= kmax[:, None])
            else:
                keep = mask_m[:, None] & mask_n[None, :] & (offs_n[None, :] <= offs_m[:, None])
            if WINDOW > 0:
                keep = keep & (offs_m[:, None] <= (offs_n[None, :] + WINDOW))
            qk = tl.where(keep, qk, float("-inf"))
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            m_safe = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
            re_scale = tl.exp(e_max - m_safe)
            p = tl.where(keep, tl.exp(qk - m_safe[:, None]), 0.0)
            acc = acc * re_scale[:, None] + _dot(p, v, LOWP)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
        o = acc / tl.where(e_sum > 0, e_sum, 1.0)[:, None]
        tl.store(O + (q_start + offs_m[:, None]) * stride_ot + head[:, None] * stride_oh + offs_dv[None, :],
                 o.to(O.dtype.element_ty), mask=mask_m[:, None])


def extend_gqa_window(q, k_extend, v_extend, k_buffer, v_buffer, o, qo_indptr, kv_indptr, kv_indices,
                      max_len_extend, sm_scale, k_scale=1.0, v_scale=1.0, window=1024, block_m=None, block_n=None,
                      num_warps=None, num_stages=None, d_split=1, kmax=None):
    """Sliding-window prefill / extend (no custom mask): q / k_extend / v_extend [N, H, D] for the new tokens,
    kv_indptr / kv_indices = the window list of the cached prefix."""
    n, hq, d = q.shape
    hkv = k_extend.shape[1]
    group = hq // hkv
    # defaults MEASURED on 1xH100 (gemma31_swa_prefill_bench sweep, hd 256 / group 2 / window 1024): block_m 32,
    # block_n 64, 8 warps (8K chunk 2,419 us vs stock 5,381 and window-skip-patched stock 3,007)
    # hd 512 (full layers, group 8) MEASURED best of 24 configs on 1xH100: block_m 4 (32 rows), block_n 32, 4 warps,
    # 2 stages (8K chunk 36,466 -> 13,380 us per layer)
    big = d >= 512
    block_m = block_m or (max(1, 16 // group) if _INTERPRET else (4 if big else 32))
    block_m = max(block_m, triton.cdiv(16, group))
    block_n = block_n or (16 if _INTERPRET else (32 if big else 64))
    bs = qo_indptr.shape[0] - 1
    launch = {} if _INTERPRET else {"num_warps": num_warps or (4 if big else 8), "num_stages": num_stages or 2}
    _extend_gqa_window[(bs, hkv, triton.cdiv(max_len_extend, block_m) * d_split)](
        q, k_extend, v_extend, k_buffer, v_buffer, o, qo_indptr, kv_indptr, kv_indices,
        kmax if kmax is not None else qo_indptr, float(sm_scale), float(k_scale), float(v_scale),
        q.stride(0), q.stride(1), k_extend.stride(0), k_extend.stride(1), v_extend.stride(0), v_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1), v_buffer.stride(0), v_buffer.stride(1), o.stride(0), o.stride(1),
        GROUP=group, D=d, BLOCK_M=block_m, BLOCK_N=block_n, WINDOW=int(window), LOWP=not _INTERPRET,
        D_SPLIT=int(d_split), USE_KMAX=kmax is not None, **launch)
    return o


def extend_window_ref(q, k_ext, v_ext, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices, sm_scale, k_scale, v_scale,
                      window):
    """window <= 0: plain causal extend (prefix fully visible)."""
    hq, hkv = q.shape[1], k_ext.shape[1]
    g = hq // hkv
    out = torch.empty(q.shape, dtype=torch.float32)
    for b in range(qo_indptr.shape[0] - 1):
        s, e = int(qo_indptr[b]), int(qo_indptr[b + 1])
        a, z = int(kv_indptr[b]), int(kv_indptr[b + 1])
        loc = kv_indices[a:z].long()
        w, n = z - a, e - s
        k = torch.cat([k_buf[loc].float() * k_scale, k_ext[s:e].float()], 0).repeat_interleave(g, 1)
        v = torch.cat([v_buf[loc].float() * v_scale, v_ext[s:e].float()], 0).repeat_interleave(g, 1)
        sc = torch.einsum("qhd,khd->hqk", q[s:e].float(), k) * sm_scale
        m = torch.zeros(n, w + n, dtype=torch.bool)
        for i in range(n):
            for j in range(w):
                m[i, j] = window <= 0 or (w + i <= j + window)
            for t in range(n):
                m[i, w + t] = (t <= i) and (window <= 0 or i <= t + window)
        sc = sc.masked_fill(~m[None], float("-inf"))
        out[s:e] = torch.einsum("hqk,khd->qhd", torch.softmax(sc, -1), v)
    return out


# ------------------------------------------------------------------------------------------------ serving glue
# Installed as sglang/srt/layers/attention/numinous_gemma_verify.py by operations/patches/gemma31_verify_hooks.py.
ENV = "NUMINOUS_GEMMA_VERIFY_GQA"
_DISABLED: dict = {}        # kernel name -> first launch error; once set, that path returns False (stock kernels)


_ENGAGED: set = set()


def _engaged(name: str) -> None:
    if name not in _ENGAGED:
        _ENGAGED.add(name)
        import sys as _sys
        print(f"numinous gemma {name}: engaged", file=_sys.stderr, flush=True)


def _guarded(name, fn) -> bool:
    """Run fn(); a Triton compile / resource error at launch (raised before the kernel runs) latches this path off
    and returns False so the stock kernel serves (rtxv41 run2: a resource error must never take the server down)."""
    if name in _DISABLED:
        return False
    try:
        fn()
        _engaged(name)
        return True
    except Exception as e:  # noqa: BLE001
        if "OutOfResources" in type(e).__name__ or "CompilationError" in type(e).__name__ \
                or "out of resource" in str(e).lower() \
                or (name.startswith("extend_fa4") and not isinstance(e, (KeyboardInterrupt, SystemExit))):
            _DISABLED[name] = f"{type(e).__name__}: {str(e)[:200]}"
            import logging
            logging.getLogger(__name__).warning("numinous gemma %s disabled, stock kernel serves: %s", name,
                                                _DISABLED[name])
            return False
        raise


def maybe_verify(backend, q, k, v, o, layer, forward_batch, kv_indptr, kv_indices, sliding_window_size, k_descale,
                 v_descale, logit_cap, sinks, score_mod) -> bool:
    """TritonAttnBackend.forward_extend, right before the stock verify / extend call: serve a topk-1 target verify
    with verify_gqa and return True; anything it cannot serve exactly returns False (stock path)."""
    if os.environ.get(ENV) != "1" or not forward_batch.forward_mode.is_target_verify():
        return False
    if getattr(backend, "topk", 0) != 1 or getattr(backend, "page_size", 1) != 1:
        return False
    if sinks is not None or score_mod is not None or (logit_cap and logit_cap > 0) or k is None or v is None:
        return False
    if getattr(layer, "xai_temperature_len", -1) not in (None, -1) and layer.xai_temperature_len > 0:
        return False
    md = backend.forward_metadata
    n_ext = md.max_extend_len
    try:
        n_ext = int(n_ext)
    except (TypeError, ValueError):
        return False
    hq, d, hkv = layer.tp_q_head_num, layer.qk_head_dim, layer.tp_k_head_num
    if n_ext < 1 or layer.v_head_dim != d or d > 512 or d % 16:
        return False
    q3 = q.view(-1, hq, d)
    bs = md.qo_indptr.shape[0] - 1
    if q3.shape[0] != bs * n_ext:
        return False
    return _guarded(f"verify_d{d}", lambda: verify_gqa(
        q3, k.contiguous().view(-1, hkv, d), v.contiguous().view(-1, hkv, d),
        backend.token_to_kv_pool.get_key_buffer(layer.layer_id),
        backend.token_to_kv_pool.get_value_buffer(layer.layer_id), o.view(-1, hq, layer.v_head_dim),
        md.qo_indptr, kv_indptr, kv_indices, n_ext, layer.scaling, float(k_descale), float(v_descale),
        window=int(sliding_window_size) if sliding_window_size and sliding_window_size > 0 else 0))



def maybe_extend_window(backend, q, k, v, o, layer, forward_batch, kv_indptr, kv_indices, sliding_window_size,
                        k_descale, v_descale, logit_cap, sinks, score_mod) -> bool:
    """TritonAttnBackend.forward_extend (prefill / chunked extend, not verify): serve a layer with extend_gqa_window
    and return True; anything else returns False (stock path). Env:
      - NUMINOUS_GEMMA_EXTEND_GQA=1: SWA and full layers;
      - NUMINOUS_GEMMA_SWA_EXTEND=1: SWA layers only.
    Measured 1xH100, 8K chunk per layer: SWA 5,381 -> 2,397 us, full hd512 36,466 -> 13,380 us."""
    all_layers = os.environ.get("NUMINOUS_GEMMA_EXTEND_GQA") == "1"          # SWA + full layers
    swa_only = os.environ.get("NUMINOUS_GEMMA_SWA_EXTEND") == "1"
    if not (all_layers or swa_only):
        return False
    fm = forward_batch.forward_mode
    if not fm.is_extend() or fm.is_target_verify() or fm.is_draft_extend_v2():
        return False
    windowed = bool(sliding_window_size) and sliding_window_size > 0
    if not windowed and not all_layers:
        return False
    md = backend.forward_metadata
    if getattr(backend, "page_size", 1) != 1:
        return False
    kmax = None
    if md.custom_mask is not None:
        # image batch: a structured mask (bidirectional image spans) runs on the Triton kernel with a per-query key
        # bound; FA4 has no per-query bound. Anything else keeps the stock kernel
        if os.environ.get("NUMINOUS_GEMMA_IMAGE_MASK") != "1" or md.mask_indptr is None:
            return False
        cache = getattr(forward_batch, "_numinous_kmax", None)
        if cache is None:
            cache = kmax_from_mask(md.custom_mask, md.mask_indptr, md.qo_indptr, md.kv_indptr)
            forward_batch._numinous_kmax = cache if cache is not None else False
        if cache is False or cache is None:
            return False
        kmax = cache
    if sinks is not None or score_mod is not None or (logit_cap and logit_cap > 0) or k is None or v is None:
        return False
    hq, d, hkv = layer.tp_q_head_num, layer.qk_head_dim, layer.tp_k_head_num
    if layer.v_head_dim != d or d > 512 or d % 16:
        return False
    if kmax is not None:
        return _guarded(f"extend_img_d{d}_w{int(windowed)}", lambda: extend_gqa_window(
            q.view(-1, hq, d), k.contiguous().view(-1, hkv, d), v.contiguous().view(-1, hkv, d),
            backend.token_to_kv_pool.get_key_buffer(layer.layer_id),
            backend.token_to_kv_pool.get_value_buffer(layer.layer_id), o.view(-1, hq, layer.v_head_dim),
            md.qo_indptr, kv_indptr, kv_indices, int(md.max_extend_len), layer.scaling, float(k_descale),
            float(v_descale), window=int(sliding_window_size) if windowed else 0, kmax=kmax))
    plan = None
    if ((windowed and d <= 256 and os.environ.get("NUMINOUS_GEMMA_SWA_FA4") == "1")
            or (not windowed and d == 512 and os.environ.get("NUMINOUS_GEMMA_EXTEND_FA4") == "1")):
        plan = fa4_plan(forward_batch, backend, kv_indptr, kv_indices, windowed)
    if (plan is not None and windowed and d <= 256 and os.environ.get("NUMINOUS_GEMMA_SWA_FA4") == "1"
            and _guarded("extend_fa4_swa", lambda: extend_fa4_hd512(
                q.view(-1, hq, d), k.contiguous().view(-1, hkv, d), v.contiguous().view(-1, hkv, d),
                backend.token_to_kv_pool.get_key_buffer(layer.layer_id),
                backend.token_to_kv_pool.get_value_buffer(layer.layer_id), o.view(-1, hq, layer.v_head_dim),
                md.qo_indptr, kv_indptr, kv_indices, layer.scaling, float(k_descale), float(v_descale),
                window=int(sliding_window_size), plan=plan))):
        return True
    if (plan is not None and not windowed and d == 512 and os.environ.get("NUMINOUS_GEMMA_EXTEND_FA4") == "1"
            and _guarded("extend_fa4_d512", lambda: extend_fa4_hd512(
                q.view(-1, hq, d), k.contiguous().view(-1, hkv, d), v.contiguous().view(-1, hkv, d),
                backend.token_to_kv_pool.get_key_buffer(layer.layer_id),
                backend.token_to_kv_pool.get_value_buffer(layer.layer_id), o.view(-1, hq, layer.v_head_dim),
                md.qo_indptr, kv_indptr, kv_indices, layer.scaling, float(k_descale), float(v_descale),
                plan=plan))):
        return True
    return _guarded(f"extend_d{d}_w{int(windowed)}", lambda: extend_gqa_window(
        q.view(-1, hq, d), k.contiguous().view(-1, hkv, d), v.contiguous().view(-1, hkv, d),
        backend.token_to_kv_pool.get_key_buffer(layer.layer_id),
        backend.token_to_kv_pool.get_value_buffer(layer.layer_id), o.view(-1, hq, layer.v_head_dim),
        md.qo_indptr, kv_indptr, kv_indices, int(md.max_extend_len), layer.scaling, float(k_descale),
        float(v_descale), window=int(sliding_window_size) if windowed else 0))


# ------------------------------------------------------------------------------------------------ hd512 prefill, FA4
# Measured (1xH100 h100b, g31_fa4_hd512_probe.py): FA4's CuTe-DSL SM90 forward (flash-attn-4 4.0.0b19, in the image)
# runs head dim 512 once its hdim<=256 gate (interface.py:102-105) is lifted.
#   - Q.K^T is computed over 512; the output comes in two passes over 256-wide V halves.
#   - Tile 128x32, intra-WG overlap: 5.82 ms per full layer per 8K chunk, 10 layers 58.2 ms (Triton extend_gqa_window
#     133, stock 365); rel L2 vs fp32 0.0023.
_FA4_TILE = (128, 32, True, True)          # (tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap), hd > 256 only
_FA4_TILE_256 = None                        # same for the hd256 SWA layers; None = FA4's own choice
_FA4_PACK_GQA_256 = False


def _fa4():
    import flash_attn.cute.interface as fi
    if not getattr(fi, "_numinous_hd512", False):
        orig_tile, orig_validate = fi._tile_size_fwd_sm90, fi._validate_head_dims

        def tile(hd, hdv, c, l, sparse_block_size_q=None):
            if hd > 256:
                return fi.FwdConfig(*_FA4_TILE)
            return orig_tile(hd, hdv, c, l, sparse_block_size_q=sparse_block_size_q)

        def validate(head_dim, head_dim_v, compute_capability, alignment):
            if compute_capability == 9 and head_dim == 512 and head_dim_v <= 256 and head_dim % alignment == 0:
                return None
            return orig_validate(head_dim, head_dim_v, compute_capability, alignment)
        fi._tile_size_fwd_sm90, fi._validate_head_dims = tile, validate
        fi._numinous_hd512 = True
    return fi


@triton.jit
def _assemble_kv_kernel(K_buf, V_buf, K_ext, V_ext, K_all, V_all, SRC, k_scale, v_scale,
                        KROW: tl.constexpr, VROW: tl.constexpr, KBLOCK: tl.constexpr, VBLOCK: tl.constexpr):
    """One row of FA4's contiguous K/V: SRC >= 0 is a pool location (dequantized), SRC < 0 is new row -(SRC + 1)."""
    r = tl.program_id(0).to(tl.int64)
    s = tl.load(SRC + r)
    ok = tl.arange(0, KBLOCK)
    ov = tl.arange(0, VBLOCK)
    mk = ok < KROW
    mv = ov < VROW
    if s >= 0:
        k = tl.load(K_buf + s * KROW + ok, mask=mk, other=0.0).to(tl.float32) * k_scale
        v = tl.load(V_buf + s * VROW + ov, mask=mv, other=0.0).to(tl.float32) * v_scale
    else:
        e = -s - 1
        k = tl.load(K_ext + e * KROW + ok, mask=mk, other=0.0).to(tl.float32)
        v = tl.load(V_ext + e * VROW + ov, mask=mv, other=0.0).to(tl.float32)
    tl.store(K_all + r * KROW + ok, k.to(K_all.dtype.element_ty), mask=mk)
    tl.store(V_all + r * VROW + ov, v.to(V_all.dtype.element_ty), mask=mv)


class Fa4Plan:
    """Per-forward FA4 metadata built from HOST lengths (forward_batch.extend_seq_lens_cpu / extend_prefix_lens_cpu;
    the SWA window list holds min(prefix, backend.sliding_window_size) tokens, triton_backend.py:2530-2534): one
    H2D copy and one device gather per forward and layer kind, no device->host sync per layer."""

    def __init__(self, ext_lens, pre_lens, kv_indptr, kv_indices, device):
        ext_lens = [int(x) for x in ext_lens]
        pre_lens = [int(x) for x in pre_lens]
        bs = len(ext_lens)
        cu = [0]
        for e, p in zip(ext_lens, pre_lens):
            cu.append(cu[-1] + p + e)
        self.total = cu[-1]
        self.max_q = max(ext_lens) if bs else 0
        self.max_k = max((cu[b + 1] - cu[b]) for b in range(bs)) if bs else 0
        self.no_prefix = sum(pre_lens) == 0
        self.cu_k = torch.tensor(cu, dtype=torch.int32).pin_memory().to(device, non_blocking=True) \
            if torch.device(device).type == "cuda" else torch.tensor(cu, dtype=torch.int32, device=device)
        self.src = None
        if self.no_prefix:
            return
        src = torch.empty(self.total, dtype=torch.int64)
        pre_rows, pre_seg, pre_j = [], [], []
        eo = 0
        for b, (e, p) in enumerate(zip(ext_lens, pre_lens)):
            base = cu[b]
            if p:
                pre_rows.append(torch.arange(base, base + p))
                pre_seg.append(torch.full((p,), b, dtype=torch.int64))
                pre_j.append(torch.arange(p))
            src[base + p:base + p + e] = -(torch.arange(eo, eo + e) + 1)
            eo += e
        rows, seg, j = torch.cat(pre_rows), torch.cat(pre_seg), torch.cat(pre_j)
        cuda = torch.device(device).type == "cuda"
        h2d = (lambda t: t.pin_memory().to(device, non_blocking=True)) if cuda else (lambda t: t.to(device))
        src_d, rows_d, seg_d, j_d = h2d(src), h2d(rows), h2d(seg), h2d(j)
        # prefix row j of request b is kv_indices[kv_indptr[b] + j] (the window list on SWA layers)
        src_d[rows_d] = kv_indices[kv_indptr.to(torch.int64)[seg_d] + j_d].to(torch.int64)
        self.src = src_d


def fa4_plan(forward_batch, backend, kv_indptr, kv_indices, windowed):
    """Cached per forward and layer kind; None when the host lengths are missing (the caller then uses the Triton
    kernel)."""
    key = "_numinous_fa4_plan_w" if windowed else "_numinous_fa4_plan_f"
    plan = getattr(forward_batch, key, None)
    if plan is not None:
        return plan
    ext = getattr(forward_batch, "extend_seq_lens_cpu", None)
    pre = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if ext is None or pre is None or len(ext) != len(pre) or len(ext) != kv_indptr.shape[0] - 1:
        return None
    if windowed:
        sw = int(getattr(backend, "sliding_window_size", 0) or 0)
        if sw <= 0:
            return None
        pre = [min(int(x), sw) for x in pre]
    plan = Fa4Plan(ext, pre, kv_indptr, kv_indices, kv_indices.device)
    setattr(forward_batch, key, plan)
    return plan


def extend_fa4_hd512(q, k_extend, v_extend, k_buffer, v_buffer, o, qo_indptr, kv_indptr, kv_indices, sm_scale,
                     k_scale=1.0, v_scale=1.0, flash=None, window=0, plan=None):
    """Prefill / extend through FA4: full hd512 layers (no window; two V passes) or, with window > 0, the hd256
    sliding-window layers (one pass, window_size=(window, 0), the stock rule q - k <= window with window =
    layer.sliding_window_size). Each request's cached prefix (the window list on SWA layers) is gathered from the
    pool (dequantized: buffer x scale, as bf16) and put in front of its new K/V. FA4 varlen then runs with bottom-right
    causal alignment (query i sees keys <= i + prefix), twice: once per 256-wide V half. `flash` lets a CPU test
    stand in for FA4."""
    n, hq, d = q.shape
    hkv = k_extend.shape[1]
    if plan is None:            # tests / callers without host lengths: one sync here, never per layer in serving
        qo = qo_indptr.to(torch.int64).tolist()
        kp = kv_indptr.to(torch.int64).tolist()
        plan = Fa4Plan([qo[b + 1] - qo[b] for b in range(len(qo) - 1)], [kp[b + 1] - kp[b] for b in range(len(kp) - 1)],
                       kv_indptr, kv_indices, q.device)
    if plan.no_prefix:
        k_all, v_all = k_extend, v_extend
    else:
        dv0 = v_extend.shape[-1]
        k_all = torch.empty(plan.total, hkv, d, dtype=q.dtype, device=q.device)
        v_all = torch.empty(plan.total, hkv, dv0, dtype=q.dtype, device=q.device)
        _assemble_kv_kernel[(plan.total,)](
            k_buffer, v_buffer, k_extend.contiguous(), v_extend.contiguous(), k_all, v_all, plan.src,
            float(k_scale), float(v_scale), KROW=hkv * d, VROW=hkv * dv0,
            KBLOCK=triton.next_power_of_2(hkv * d), VBLOCK=triton.next_power_of_2(hkv * dv0), num_warps=4)
    cu_q = qo_indptr if qo_indptr.dtype == torch.int32 else qo_indptr.to(torch.int32)
    cu_kt = plan.cu_k
    max_q, max_k = plan.max_q, plan.max_k
    dv = v_all.shape[-1]
    step = dv // 2 if dv > 256 else dv          # hd512: two 256-wide V passes; hd <= 256: one pass
    if flash is None:
        # FA4 writes straight into o (a strided 256-wide view for the hd512 halves): no output copy per layer
        fi = _fa4()
        kw = {}
        if window and window > 0:
            kw.update(window_size_left=int(window), window_size_right=0)
        if d <= 256 and _FA4_TILE_256 is not None:
            tm, tn, rs, ov = _FA4_TILE_256
            kw.update(tile_mn=(tm, tn), mma_pv_is_rs=rs, intra_wg_overlap=ov)
        for lo in range(0, dv, step):
            vv = v_all[..., lo:lo + step]
            fi._flash_attn_fwd(q, k_all, vv if step == dv else vv.contiguous(), cu_seqlens_q=cu_q,
                               cu_seqlens_k=cu_kt, max_seqlen_q=max_q, max_seqlen_k=max_k, softmax_scale=sm_scale,
                               causal=True, pack_gqa=bool(_FA4_PACK_GQA_256) if d <= 256 else False,
                               out=o[..., lo:lo + step], **kw)
        return o
    kw = {"window_size": (int(window), 0)} if window and window > 0 else {}
    for lo in range(0, dv, step):
        vv = v_all[..., lo:lo + step]
        out = flash(q, k_all, vv if step == dv else vv.contiguous(), cu_seqlens_q=cu_q, cu_seqlens_k=cu_kt,
                    max_seqlen_q=max_q, max_seqlen_k=max_k, softmax_scale=sm_scale, causal=True, pack_gqa=False, **kw)
        out = out[0] if isinstance(out, tuple) else out
        o[..., lo:lo + step].copy_(out.view(n, hq, step))
    return o



def kmax_from_mask(custom_mask, mask_indptr, qo_indptr, kv_indptr):
    """Image batches (gemma4_mm.py:644-656, 368-411) carry a dense custom mask per request of shape
    [ext x (prefix + ext)]. When every row is 'all prefix keys + new keys 0..kmax' (bidirectional image spans, causal
    elsewhere), return kmax int32 [N_ext], relative to each request's first new token. Otherwise return None, and
    the caller keeps the stock kernel. The check reads the mask once on the GPU; prefill is not CUDA-graphed, so the
    host sync is allowed."""
    qo = qo_indptr.to(torch.int64).tolist()
    kp = kv_indptr.to(torch.int64).tolist()
    mi = mask_indptr.to(torch.int64).tolist()
    out = []
    for b in range(len(qo) - 1):
        ext, pre = qo[b + 1] - qo[b], kp[b + 1] - kp[b]
        if ext == 0:
            continue
        m = custom_mask[mi[b]:mi[b] + ext * (pre + ext)].view(ext, pre + ext).bool()
        if pre and not bool(m[:, :pre].all()):
            return None
        new = m[:, pre:]
        ar = torch.arange(ext, device=new.device)
        # last visible new key per row; the row must then be exactly 0..kmax, and kmax >= the row's own index
        kmax = (ext - 1) - torch.flip(new, dims=[1]).int().argmax(dim=1)
        if not bool((new == (ar[None, :] <= kmax[:, None])).all()) or not bool((kmax >= ar).all()):
            return None
        out.append(kmax.to(torch.int32))
    return torch.cat(out) if out else None
