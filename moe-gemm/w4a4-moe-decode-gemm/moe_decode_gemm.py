"""Grouped GEMM for MoE decode at tiny per-expert M, written for Qwen 3.8's shape on Blackwell.

The shape, per GPU under TP8:
    512 experts, hidden K=8192, expert intermediate 2048 -> 256 per GPU
    gate+up:  [m_e, 8192] x [8192, 512]      (gate and up concatenated along N)
    down:     [m_e, 256]  x [256, 8192]
    m_e = tokens routed to expert e, 1..16 in decode, top-10 of 512

Why this kernel exists. At c >= 32 with 4 draft tokens, a decode step touches over 90% of the experts, so
the step reads almost all 1.2 TB of NVFP4 expert weights, ~147 GB per GPU: 18 ms at 8 TB/s. The measured
step was 26 ms. The kernel that decides the gap is the one that streams each touched expert's weight slice
once, at bandwidth, with only a handful of rows to multiply against it. Compute is irrelevant; every design
choice below is about keeping the weight stream saturated.

Design:
  - work list, not a dense grid: one program per (touched expert, N tile). Untouched experts cost nothing.
  - BLOCK_M = 16 covers m_e; rows beyond m_e are masked. The weight tile is read once regardless of m_e.
  - the K loop streams W[e] in BLOCK_K slabs; X[e] rows are tiny and stay in registers/L1.
  - two weight formats share one kernel body: bf16 (reference) and 4-bit packed with per-block scales
    (tl.dot_scaled), so the fp4 path's bandwidth can be compared with bf16 on identical routing.

This file is importable on CPU (routing, offsets, roofline) and runs on GPU via operations/modal/moe_kernel_bench.py.
"""
from __future__ import annotations

import math

try:
    import torch
except ImportError:  # CPU-only tests of the pure-python parts
    torch = None


# -----------------------------------------------------------------------------------------------------------
# routing and work-list construction (pure python / torch-cpu, tested)
# -----------------------------------------------------------------------------------------------------------

def experts_touched_fraction(tokens: int, experts: int = 512, topk: int = 10) -> float:
    """Expected fraction of experts that receive at least one token."""
    return 1.0 - (1.0 - topk / experts) ** tokens


def roofline_ms(tokens: int, *, experts: int = 512, topk: int = 10, layers: int = 92, hidden: int = 8192,
                inter_per_gpu: int = 256, bytes_per_param: float = 0.5, hbm_tb_s: float = 8.0) -> dict:
    """Weight-stream floor for one decode step on one GPU, MoE only."""
    per_expert = hidden * (2 * inter_per_gpu + inter_per_gpu) * bytes_per_param  # gate, up, down slices
    touched = experts_touched_fraction(tokens, experts, topk)
    gb = layers * experts * touched * per_expert / 1e9
    return {"tokens": tokens, "experts_touched": round(touched, 3), "expert_gb_per_gpu": round(gb, 1),
            "ms_at_hbm": round(gb / hbm_tb_s / 1000 * 1000, 2)}


def group_by_expert(topk_ids, experts: int):
    """Sort (token, slot) pairs by expert. Returns sorted token index, expert offsets [E+1], and counts.

    topk_ids: [M, topk] int tensor. Output row r of the grouped activation is token sorted_tok[r].
    """
    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_expert = flat[order]
    counts = torch.bincount(sorted_expert, minlength=experts)
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device=topk_ids.device)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    sorted_tok = (order // topk_ids.shape[1]).to(torch.int32)
    return sorted_tok, offsets, counts.to(torch.int32)


def work_list(counts, block_n: int, n: int, block_m: int = 16):
    """(expert, n_tile, m_tile) triples for experts with at least one token; the launch grid is this long.

    M is tiled on the host so the kernel has no dynamic outer loop (a trip count that varies per program
    cost the bf16 kernel 20% when it lived inside). Fully vectorised: a Python loop here cost more than the
    kernel it launched. At BLOCK_M=128 every decode expert is one M tile.
    """
    touched = torch.nonzero(counts > 0).flatten()
    n_tiles = math.ceil(n / block_n)
    mt = (counts[touched] + block_m - 1) // block_m               # M tiles per touched expert
    expert_of_mtile = touched.repeat_interleave(mt)               # [total_mt]
    first = torch.cumsum(mt, 0) - mt
    mtile_idx = torch.arange(int(mt.sum()), device=counts.device) - first.repeat_interleave(mt)
    e = expert_of_mtile.repeat_interleave(n_tiles).to(torch.int32)
    m = mtile_idx.repeat_interleave(n_tiles).to(torch.int32)
    t = torch.arange(n_tiles, device=counts.device, dtype=torch.int32).repeat(int(mt.sum()))
    return e, t, m


# -----------------------------------------------------------------------------------------------------------
# triton kernels (only importable where triton is)
# -----------------------------------------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _moe_gemm_bf16_kernel(
        x_ptr, w_ptr, out_ptr, sorted_tok_ptr, offsets_ptr, work_e_ptr, work_t_ptr, work_m_ptr,
        K, N, stride_xm, stride_xk, stride_we, stride_wn, stride_wk, stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """out[r, :] = x[sorted_tok[r], :] @ W[e].T for rows r in one (expert, M tile, N tile) work item."""
        pid = tl.program_id(0)
        e = tl.load(work_e_ptr + pid)
        nt = tl.load(work_t_ptr + pid)
        mt = tl.load(work_m_ptr + pid)
        row0 = tl.load(offsets_ptr + e) + mt * BLOCK_M
        row1 = tl.load(offsets_ptr + e + 1)
        m_e = row1 - row0
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < m_e
        n_mask = offs_n < N
        tok = tl.load(sorted_tok_ptr + row0 + offs_m, mask=m_mask, other=0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        w_base = w_ptr + e.to(tl.int64) * stride_we
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            x = tl.load(x_ptr + tok[:, None].to(tl.int64) * stride_xm + kk[None, :] * stride_xk,
                        mask=m_mask[:, None] & (kk[None, :] < K), other=0.0)
            w = tl.load(w_base + offs_n[:, None].to(tl.int64) * stride_wn + kk[None, :] * stride_wk,
                        mask=n_mask[:, None] & (kk[None, :] < K), other=0.0)
            acc += tl.dot(x, tl.trans(w))
        out_rows = row0 + offs_m
        tl.store(out_ptr + out_rows[:, None].to(tl.int64) * stride_om + offs_n[None, :] * stride_on,
                 acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _moe_gemm_fp4_kernel(
        x_ptr, xs_ptr, w_ptr, ws_ptr, out_ptr, sorted_tok_ptr, offsets_ptr, work_e_ptr, work_t_ptr, work_m_ptr,
        K, N, stride_xm, stride_xk, stride_xsm, stride_xsk,
        stride_we, stride_wn, stride_wk, stride_se, stride_sn, stride_sk, stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_KB: tl.constexpr, BLOCK_KS: tl.constexpr, A_FP4: tl.constexpr,
    ):
        """W packed e2m1 (two per byte along K) with e8m0 scales per 32. A_FP4=True: activations are also
        e2m1 + e8m0. Blackwell's fp4 MMA needs M of 64 or 128, so BLOCK_M is padded up to that and the
        tensor core multiplies zero rows: compute is free here, the weight stream is what costs."""
        pid = tl.program_id(0)
        e = tl.load(work_e_ptr + pid)
        nt = tl.load(work_t_ptr + pid)
        mt = tl.load(work_m_ptr + pid)
        row0 = tl.load(offsets_ptr + e) + mt * BLOCK_M
        row1 = tl.load(offsets_ptr + e + 1)
        m_e = row1 - row0
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < m_e
        n_mask = offs_n < N
        tok = tl.load(sorted_tok_ptr + row0 + offs_m, mask=m_mask, other=0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        w_base = w_ptr + e.to(tl.int64) * stride_we
        s_base = ws_ptr + e.to(tl.int64) * stride_se
        offs_kb = tl.arange(0, BLOCK_KB)
        offs_ks = tl.arange(0, BLOCK_KS)
        offs_k = tl.arange(0, BLOCK_K)
        for k0 in range(0, K, BLOCK_K):
            kb = k0 // 2 + offs_kb
            ks = k0 // 32 + offs_ks
            w = tl.load(w_base + offs_n[:, None].to(tl.int64) * stride_wn + kb[None, :] * stride_wk,
                        mask=n_mask[:, None], other=0)
            ws = tl.load(s_base + offs_n[:, None].to(tl.int64) * stride_sn + ks[None, :] * stride_sk,
                         mask=n_mask[:, None], other=127)
            if A_FP4:
                xq = tl.load(x_ptr + tok[:, None].to(tl.int64) * stride_xm + kb[None, :] * stride_xk,
                             mask=m_mask[:, None], other=0)
                xs = tl.load(xs_ptr + tok[:, None].to(tl.int64) * stride_xsm + ks[None, :] * stride_xsk,
                             mask=m_mask[:, None], other=127)
                acc = tl.dot_scaled(xq, xs, "e2m1", tl.trans(w), ws, "e2m1", acc)
            else:
                kk = k0 + offs_k
                x = tl.load(x_ptr + tok[:, None].to(tl.int64) * stride_xm + kk[None, :] * stride_xk,
                            mask=m_mask[:, None], other=0.0)
                acc = tl.dot_scaled(x, None, "bf16", tl.trans(w), ws, "e2m1", acc)
        out_rows = row0 + offs_m
        tl.store(out_ptr + out_rows[:, None].to(tl.int64) * stride_om + offs_n[None, :] * stride_on,
                 acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _moe_gemm_fp4_tiled_kernel(
        x_ptr, xs_ptr, w_ptr, ws_ptr, out_ptr, sorted_tok_ptr, offsets_ptr, work_e_ptr, work_t_ptr, work_m_ptr,
        K, N, stride_xm, stride_xk, stride_xsm, stride_xsk, stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_KB: tl.constexpr, BLOCK_KS: tl.constexpr, N_TILES: tl.constexpr, K_TILES: tl.constexpr,
        W_TRANSPOSED: tl.constexpr,
    ):
        """W4A4 with weights pre-arranged tile-major: W[e, n_tile, k_tile] is one contiguous [BLOCK_N, BLOCK_KB]
        block and its scales one contiguous [BLOCK_N, BLOCK_KS] block, so every weight load in the K loop is a
        single dense chunk instead of BLOCK_N short runs. This is what TRT-LLM's pre-shuffle buys."""
        pid = tl.program_id(0)
        e = tl.load(work_e_ptr + pid)
        nt = tl.load(work_t_ptr + pid)
        mt = tl.load(work_m_ptr + pid)
        row0 = tl.load(offsets_ptr + e) + mt * BLOCK_M
        row1 = tl.load(offsets_ptr + e + 1)
        m_e = row1 - row0
        offs_m = tl.arange(0, BLOCK_M)
        offs_n_local = tl.arange(0, BLOCK_N)
        offs_n = nt * BLOCK_N + offs_n_local
        m_mask = offs_m < m_e
        n_mask = offs_n < N
        tok = tl.load(sorted_tok_ptr + row0 + offs_m, mask=m_mask, other=0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        offs_kb = tl.arange(0, BLOCK_KB)
        offs_ks = tl.arange(0, BLOCK_KS)
        # tile bases: W laid out [E, N_TILES, K_TILES, BLOCK_N, BLOCK_KB]; scales [E, N_TILES, K_TILES, BLOCK_N, BLOCK_KS]
        w_tile0 = w_ptr + ((e.to(tl.int64) * N_TILES + nt) * K_TILES) * (BLOCK_N * BLOCK_KB)
        s_tile0 = ws_ptr + ((e.to(tl.int64) * N_TILES + nt) * K_TILES) * (BLOCK_N * BLOCK_KS)
        if W_TRANSPOSED:
            # tile stored [BLOCK_KB, BLOCK_N]: loads land in the rhs shape, no shuffle through shared memory
            w_local = offs_kb[:, None] * BLOCK_N + offs_n_local[None, :]
        else:
            w_local = offs_n_local[:, None] * BLOCK_KB + offs_kb[None, :]
        s_local = offs_n_local[:, None] * BLOCK_KS + offs_ks[None, :]
        for kt in range(0, K_TILES):
            k0 = kt * BLOCK_K
            w = tl.load(w_tile0 + kt * (BLOCK_N * BLOCK_KB) + w_local)
            ws = tl.load(s_tile0 + kt * (BLOCK_N * BLOCK_KS) + s_local)
            xq = tl.load(x_ptr + tok[:, None].to(tl.int64) * stride_xm + (k0 // 2 + offs_kb)[None, :] * stride_xk,
                         mask=m_mask[:, None], other=0)
            xs = tl.load(xs_ptr + tok[:, None].to(tl.int64) * stride_xsm + (k0 // 32 + offs_ks)[None, :] * stride_xsk,
                         mask=m_mask[:, None], other=127)
            if W_TRANSPOSED:
                acc = tl.dot_scaled(xq, xs, "e2m1", w, ws, "e2m1", acc)
            else:
                acc = tl.dot_scaled(xq, xs, "e2m1", tl.trans(w), ws, "e2m1", acc)
        out_rows = row0 + offs_m
        tl.store(out_ptr + out_rows[:, None].to(tl.int64) * stride_om + offs_n[None, :] * stride_on,
                 acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


    @triton.jit
    def _moe_gemm_fp4_tma_kernel(
        x_ptr, xs_ptr, w_desc, ws_ptr, out_ptr, sorted_tok_ptr, offsets_ptr, work_e_ptr, work_t_ptr, work_m_ptr,
        K, N, stride_xm, stride_xk, stride_xsm, stride_xsk, stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_KB: tl.constexpr, BLOCK_KS: tl.constexpr, N_TILES: tl.constexpr, K_TILES: tl.constexpr,
    ):
        """Same tile-major W4A4 contraction; the weight tile arrives by TMA (one async bulk copy per k-slab)
        instead of BLOCK_N x BLOCK_KB ordinary loads. Tiles are addressed as rows of a flat [tiles*BLOCK_N, BLOCK_KB]."""
        pid = tl.program_id(0)
        e = tl.load(work_e_ptr + pid)
        nt = tl.load(work_t_ptr + pid)
        mt = tl.load(work_m_ptr + pid)
        row0 = tl.load(offsets_ptr + e) + mt * BLOCK_M
        row1 = tl.load(offsets_ptr + e + 1)
        m_e = row1 - row0
        offs_m = tl.arange(0, BLOCK_M)
        offs_n_local = tl.arange(0, BLOCK_N)
        offs_n = nt * BLOCK_N + offs_n_local
        m_mask = offs_m < m_e
        n_mask = offs_n < N
        tok = tl.load(sorted_tok_ptr + row0 + offs_m, mask=m_mask, other=0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        offs_kb = tl.arange(0, BLOCK_KB)
        offs_ks = tl.arange(0, BLOCK_KS)
        tile0 = (e * N_TILES + nt) * K_TILES
        s_tile0 = ws_ptr + tile0.to(tl.int64) * (BLOCK_N * BLOCK_KS)
        s_local = offs_n_local[:, None] * BLOCK_KS + offs_ks[None, :]
        for kt in range(0, K_TILES):
            k0 = kt * BLOCK_K
            w = w_desc.load([(tile0 + kt) * BLOCK_N, 0])
            ws = tl.load(s_tile0 + kt * (BLOCK_N * BLOCK_KS) + s_local)
            xq = tl.load(x_ptr + tok[:, None].to(tl.int64) * stride_xm + (k0 // 2 + offs_kb)[None, :] * stride_xk,
                         mask=m_mask[:, None], other=0)
            xs = tl.load(xs_ptr + tok[:, None].to(tl.int64) * stride_xsm + (k0 // 32 + offs_ks)[None, :] * stride_xsk,
                         mask=m_mask[:, None], other=127)
            acc = tl.dot_scaled(xq, xs, "e2m1", tl.trans(w), ws, "e2m1", acc)
        out_rows = row0 + offs_m
        tl.store(out_ptr + out_rows[:, None].to(tl.int64) * stride_om + offs_n[None, :] * stride_on,
                 acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


    @triton.jit
    def _moe_gemm_fp4_swap_kernel(
        x_ptr, xs_ptr, w_ptr, ws_ptr, out_ptr, sorted_tok_ptr, offsets_ptr, work_e_ptr, work_t_ptr, work_m_ptr,
        K, N, stride_xm, stride_xk, stride_xsm, stride_xsk, stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_KB: tl.constexpr, BLOCK_KS: tl.constexpr, N_TILES: tl.constexpr, K_TILES: tl.constexpr,
    ):
        """Operand-swapped W4A4: acc[n, m] = W[e][n, :] . x[m, :]. The weight tile is the MMA's M side (128 real
        rows), the tokens are its N side (16 is legal), so no padded activation rows go through shared memory.
        Per stage: 16 KB of weights and 2 KB of activations instead of 16 KB + 16 KB, so two CTAs fit per SM."""
        pid = tl.program_id(0)
        e = tl.load(work_e_ptr + pid)
        nt = tl.load(work_t_ptr + pid)
        mt = tl.load(work_m_ptr + pid)
        row0 = tl.load(offsets_ptr + e) + mt * BLOCK_M
        row1 = tl.load(offsets_ptr + e + 1)
        m_e = row1 - row0
        offs_m = tl.arange(0, BLOCK_M)
        offs_n_local = tl.arange(0, BLOCK_N)
        offs_n = nt * BLOCK_N + offs_n_local
        m_mask = offs_m < m_e
        n_mask = offs_n < N
        tok = tl.load(sorted_tok_ptr + row0 + offs_m, mask=m_mask, other=0)
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        offs_kb = tl.arange(0, BLOCK_KB)
        offs_ks = tl.arange(0, BLOCK_KS)
        tile0 = ((e.to(tl.int64) * N_TILES + nt) * K_TILES)
        w_tile0 = w_ptr + tile0 * (BLOCK_N * BLOCK_KB)
        s_tile0 = ws_ptr + tile0 * (BLOCK_N * BLOCK_KS)
        w_local = offs_n_local[:, None] * BLOCK_KB + offs_kb[None, :]
        s_local = offs_n_local[:, None] * BLOCK_KS + offs_ks[None, :]
        for kt in range(0, K_TILES):
            k0 = kt * BLOCK_K
            w = tl.load(w_tile0 + kt * (BLOCK_N * BLOCK_KB) + w_local)
            ws = tl.load(s_tile0 + kt * (BLOCK_N * BLOCK_KS) + s_local)
            xq = tl.load(x_ptr + tok[:, None].to(tl.int64) * stride_xm + (k0 // 2 + offs_kb)[None, :] * stride_xk,
                         mask=m_mask[:, None], other=0)
            xs = tl.load(xs_ptr + tok[:, None].to(tl.int64) * stride_xsm + (k0 // 32 + offs_ks)[None, :] * stride_xsk,
                         mask=m_mask[:, None], other=127)
            acc = tl.dot_scaled(w, ws, "e2m1", tl.trans(xq), xs, "e2m1", acc)
        out_rows = row0 + offs_m
        tl.store(out_ptr + out_rows[None, :].to(tl.int64) * stride_om + offs_n[:, None] * stride_on,
                 acc.to(tl.bfloat16), mask=m_mask[None, :] & n_mask[:, None])


    @triton.jit
    def _moe_gemm_fp4_swap_k1_kernel(
        x_ptr, xs_ptr, w_ptr, ws_ptr, out_ptr, sorted_tok_ptr, offsets_ptr, work_e_ptr, work_t_ptr, work_m_ptr,
        N, stride_xm, stride_xk, stride_xsm, stride_xsk, stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr, BLOCK_KS: tl.constexpr,
        N_TILES: tl.constexpr, N_PER_PROG: tl.constexpr,
    ):
        """Single-K-slab shape (the down projection: K=256 per GPU). The activation tile is loaded once and the
        program walks N_PER_PROG weight tiles of the same expert, so the pipelined loop runs over N tiles and the
        prologue (work item, offsets, token gather) is paid once per N_PER_PROG tiles instead of once per tile."""
        pid = tl.program_id(0)
        e = tl.load(work_e_ptr + pid)
        ng = tl.load(work_t_ptr + pid)
        mt = tl.load(work_m_ptr + pid)
        row0 = tl.load(offsets_ptr + e) + mt * BLOCK_M
        row1 = tl.load(offsets_ptr + e + 1)
        m_e = row1 - row0
        offs_m = tl.arange(0, BLOCK_M)
        offs_n_local = tl.arange(0, BLOCK_N)
        m_mask = offs_m < m_e
        tok = tl.load(sorted_tok_ptr + row0 + offs_m, mask=m_mask, other=0)
        offs_kb = tl.arange(0, BLOCK_KB)
        offs_ks = tl.arange(0, BLOCK_KS)
        xq = tl.load(x_ptr + tok[:, None].to(tl.int64) * stride_xm + offs_kb[None, :] * stride_xk, mask=m_mask[:, None], other=0)
        xs = tl.load(xs_ptr + tok[:, None].to(tl.int64) * stride_xsm + offs_ks[None, :] * stride_xsk, mask=m_mask[:, None], other=127)
        xqt = tl.trans(xq)
        out_rows = row0 + offs_m
        w_local = offs_n_local[:, None] * BLOCK_KB + offs_kb[None, :]
        s_local = offs_n_local[:, None] * BLOCK_KS + offs_ks[None, :]
        e_base = e.to(tl.int64) * N_TILES
        for j in tl.static_range(N_PER_PROG):  # unrolled: a dot in a pipelined loop trips the TMEM lowering
            nt = ng * N_PER_PROG + j
            tile = e_base + nt
            w = tl.load(w_ptr + tile * (BLOCK_N * BLOCK_KB) + w_local)
            ws = tl.load(ws_ptr + tile * (BLOCK_N * BLOCK_KS) + s_local)
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            acc = tl.dot_scaled(w, ws, "e2m1", xqt, xs, "e2m1", acc)
            offs_n = nt * BLOCK_N + offs_n_local
            tl.store(out_ptr + out_rows[None, :].to(tl.int64) * stride_om + offs_n[:, None] * stride_on,
                     acc.to(tl.bfloat16), mask=m_mask[None, :] & (offs_n < N)[:, None])


def moe_gemm_bf16(x, w, sorted_tok, offsets, counts, block_m=16, block_n=128, block_k=128, num_warps=4, num_stages=3, work=None):
    """x [M, K] bf16; w [E, N, K] bf16. Returns out [rows, N] bf16 in expert-sorted row order."""
    E, N, K = w.shape
    rows = int(offsets[-1])
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=x.device)
    we, wt, wm = work if work is not None else work_list(counts, block_n, N, block_m)
    if we.numel() == 0:
        return out
    _moe_gemm_bf16_kernel[(we.numel(),)](
        x, w, out, sorted_tok, offsets, we, wt, wm, K, N,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages)
    return out


def moe_gemm_fp4(x, w_packed, w_scale, sorted_tok, offsets, counts, block_m=16, block_n=128, block_k=128,
                 num_warps=4, num_stages=3, x_scale=None, work=None):
    """W4A16 when x is bf16 [M, K]; W4A4 when x is packed e2m1 [M, K//2] uint8 and x_scale is [M, K//32]."""
    E, N, Kh = w_packed.shape
    K = Kh * 2
    rows = int(offsets[-1])
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=w_packed.device)
    we, wt, wm = work if work is not None else work_list(counts, block_n, N, block_m)
    if we.numel() == 0:
        return out
    a_fp4 = x.dtype == torch.uint8
    xs = x_scale if a_fp4 else x  # placeholder pointer when unused
    _moe_gemm_fp4_kernel[(we.numel(),)](
        x, xs, w_packed, w_scale, out, sorted_tok, offsets, we, wt, wm, K, N,
        x.stride(0), x.stride(1), xs.stride(0), xs.stride(1),
        w_packed.stride(0), w_packed.stride(1), w_packed.stride(2),
        w_scale.stride(0), w_scale.stride(1), w_scale.stride(2), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_KB=block_k // 2, BLOCK_KS=block_k // 32,
        A_FP4=a_fp4, num_warps=num_warps, num_stages=num_stages)
    return out


def tile_layout(w_packed, w_scale, block_n: int, block_k: int, transposed: bool = False):
    """Re-arrange [E, N, K//2] packed weights and [E, N, K//32] scales tile-major for the tiled kernel.

    Done once at load time in a real server; here it is the shuffle TRT-LLM applies to its fp4 experts.
    N and K must be multiples of the tile. Returns contiguous (w_tiles, s_tiles, n_tiles, k_tiles).
    """
    E, N, KB = w_packed.shape
    K = KB * 2
    assert N % block_n == 0 and K % block_k == 0, (N, K, block_n, block_k)
    n_tiles, k_tiles = N // block_n, K // block_k
    bkb, bks = block_k // 2, block_k // 32
    w_t = w_packed.reshape(E, n_tiles, block_n, k_tiles, bkb).permute(0, 1, 3, 2, 4)
    if transposed:
        w_t = w_t.transpose(3, 4)  # [E, n_tiles, k_tiles, bkb, block_n]
    s_t = w_scale.reshape(E, n_tiles, block_n, k_tiles, bks).permute(0, 1, 3, 2, 4).contiguous()
    return w_t.contiguous(), s_t, n_tiles, k_tiles


def moe_gemm_fp4_tiled(xq, x_scale, w_tiles, s_tiles, n_tiles, k_tiles, N, K, sorted_tok, offsets, counts,
                       block_m=128, block_n=128, block_k=256, num_warps=8, num_stages=3, work=None, transposed=False):
    """W4A4 over tile-major weights. xq [M, K//2] uint8, x_scale [M, K//32] uint8. transposed must match tile_layout."""
    rows = int(offsets[-1])
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=xq.device)
    we, wt, wm = work if work is not None else work_list(counts, block_n, N, block_m)
    if we.numel() == 0:
        return out
    _moe_gemm_fp4_tiled_kernel[(we.numel(),)](
        xq, x_scale, w_tiles, s_tiles, out, sorted_tok, offsets, we, wt, wm, K, N,
        xq.stride(0), xq.stride(1), x_scale.stride(0), x_scale.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_KB=block_k // 2, BLOCK_KS=block_k // 32,
        N_TILES=n_tiles, K_TILES=k_tiles, W_TRANSPOSED=transposed, num_warps=num_warps, num_stages=num_stages)
    return out


def moe_gemm_fp4_tma(xq, x_scale, w_tiles, s_tiles, n_tiles, k_tiles, N, K, sorted_tok, offsets, counts,
                     block_m=128, block_n=128, block_k=256, num_warps=4, num_stages=4, work=None):
    """W4A4 over tile-major (non-transposed) weights with the weight tile fetched by TMA."""
    from triton.tools.tensor_descriptor import TensorDescriptor
    bkb = block_k // 2
    flat = w_tiles.reshape(-1, bkb)  # [E*n_tiles*k_tiles*block_n, bkb], rows of one tile are contiguous
    desc = TensorDescriptor.from_tensor(flat, [block_n, bkb])
    rows = int(offsets[-1])
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=xq.device)
    we, wt, wm = work if work is not None else work_list(counts, block_n, N, block_m)
    if we.numel() == 0:
        return out
    _moe_gemm_fp4_tma_kernel[(we.numel(),)](
        xq, x_scale, desc, s_tiles, out, sorted_tok, offsets, we, wt, wm, K, N,
        xq.stride(0), xq.stride(1), x_scale.stride(0), x_scale.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_KB=bkb, BLOCK_KS=block_k // 32,
        N_TILES=n_tiles, K_TILES=k_tiles, num_warps=num_warps, num_stages=num_stages)
    return out


def moe_gemm_fp4_swap(xq, x_scale, w_tiles, s_tiles, n_tiles, k_tiles, N, K, sorted_tok, offsets, counts,
                      block_m=16, block_n=128, block_k=256, num_warps=4, num_stages=4, work=None):
    """Operand-swapped W4A4 over tile-major weights. block_m is the token tile (16 or 32)."""
    rows = int(offsets[-1])
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=xq.device)
    we, wt, wm = work if work is not None else work_list(counts, block_n, N, block_m)
    if we.numel() == 0:
        return out
    _moe_gemm_fp4_swap_kernel[(we.numel(),)](
        xq, x_scale, w_tiles, s_tiles, out, sorted_tok, offsets, we, wt, wm, K, N,
        xq.stride(0), xq.stride(1), x_scale.stride(0), x_scale.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_KB=block_k // 2, BLOCK_KS=block_k // 32,
        N_TILES=n_tiles, K_TILES=k_tiles, num_warps=num_warps, num_stages=num_stages)
    return out


def moe_gemm_fp4_swap_k1(xq, x_scale, w_tiles, s_tiles, n_tiles, k_tiles, N, K, sorted_tok, offsets, counts,
                         block_m=16, block_n=128, n_per_prog=8, num_warps=4, num_stages=2, work=None):
    """Operand-swapped W4A4 for K == one slab (tile_layout called with block_k=K). Each program covers
    n_per_prog N tiles of one expert; the work list's tile index is the group index."""
    assert k_tiles == 1, "single-slab kernel: call tile_layout with block_k=K"
    assert n_tiles % n_per_prog == 0, (n_tiles, n_per_prog)
    rows = int(offsets[-1])
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=xq.device)
    we, wt, wm = work if work is not None else work_list(counts, block_n * n_per_prog, N, block_m)
    if we.numel() == 0:
        return out
    _moe_gemm_fp4_swap_k1_kernel[(we.numel(),)](
        xq, x_scale, w_tiles, s_tiles, out, sorted_tok, offsets, we, wt, wm, N,
        xq.stride(0), xq.stride(1), x_scale.stride(0), x_scale.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_KB=K // 2, BLOCK_KS=K // 32, N_TILES=n_tiles, N_PER_PROG=n_per_prog,
        num_warps=num_warps, num_stages=num_stages)
    return out


def reference_bf16(x, w, sorted_tok, offsets):
    """Per-expert torch matmul, the correctness oracle."""
    E, N, K = w.shape
    rows = int(offsets[-1])
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=x.device)
    for e in range(E):
        a, b = int(offsets[e]), int(offsets[e + 1])
        if b > a:
            out[a:b] = (x[sorted_tok[a:b].long()].float() @ w[e].float().T).to(torch.bfloat16)
    return out


def pack_e2m1(w_bf16, block: int = 32):
    """Quantise [..., K] bf16 to e2m1 nibbles with e8m0 per-block scales (mx layout Triton consumes).

    Works for weights [E, N, K] and activations [M, K] alike. Returns (packed uint8 [..., K//2],
    scales uint8 [..., K//block], dequantised bf16 for the oracle).
    """
    *lead, K = w_bf16.shape
    wf = w_bf16.float().reshape(*lead, K // block, block)
    amax = wf.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    # e2m1 max magnitude is 6.0; pick a power-of-two scale so amax maps to <= 6
    exp = torch.ceil(torch.log2(amax / 6.0))
    scale = torch.exp2(exp)
    q = wf / scale
    # e2m1 representable magnitudes
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w_bf16.device)
    idx = torch.bucketize(q.abs(), (levels[1:] + levels[:-1]) / 2)
    mag = levels[idx]
    deq = torch.where(q < 0, -mag, mag) * scale
    # nibble encoding: sign(1) exp(2) mant(1); index into levels gives the low 3 bits, sign is bit 3
    code = (idx | ((q < 0).to(torch.int64) << 3)).to(torch.uint8).reshape(*lead, K)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
    e8m0 = (exp.reshape(*lead, K // block) + 127).clamp(0, 254).to(torch.uint8).contiguous()
    return packed, e8m0, deq.reshape(*lead, K).to(torch.bfloat16)
