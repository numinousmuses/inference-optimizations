"""DeepSeek V4 Pro: mHC post (block end) + next mHC pre (sinkhorn 4-stream mix + RMSNorm) in ONE Triton kernel.

What it replaces (SGLang v0.5.20 `mhc_fused_post_pre`, default on for CUDA):
  <= 32 tokens: TileLang FMA post+GEMM-partials kernel, then `mhc_pre_big_fuse_with_norm_tilelang`  (2 launches)
  >  32 tokens: `mhc_post_tilelang`, DeepGEMM `tf32_hc_prenorm_gemm`, `mhc_pre_big_fuse_with_norm_tilelang` (3)
Those passes read the new residual (4 x 7168 bf16 per token) three times. This kernel reads the old residual and
the block output once, writes the new residual and the normed layer input once (plus L1/L2-hot re-reads).

Math, per token (H = hidden, 4 streams, fn [24, 4H] fp32):
  new[j]   = bf16(post[j] * x + sum_k comb[k, j] * res[k])                  (mHC post)
  mixes    = (new.flat @ fn.T) * rsqrt(sum(new^2) / 4H + rms_eps)
  pre      = sigmoid(mixes[0:4] * s0 + b[0:4]) + pre_eps
  post'    = sigmoid(mixes[4:8] * s1 + b[4:8]) * post_mult
  comb'    = sinkhorn(mixes[8:24] * s2 + b[8:24]), row-softmax start, `iters` alternating normalizations
  ol       = sum_j pre[j] * new[j]                                          (fp32)
  out      = bf16(bf16(ol) * rsqrt(sum(ol^2) / H + norm_eps) * w)           (the reference's bf16 stash)

Layout: grid (S, G). G token groups of BLOCK_M rows; S hidden splits of CHUNK columns, walked in BLOCK_H sub-tiles.
Phase 1 (every program): the post for its [BLOCK_M, 4, CHUNK] slice, written as the new residual; partial mix
dot-products (tensor cores, fn split into bf16 terms) and the 4x4 Gram matrix G[j][k] = sum_h new[j,h] new[k,h].
The Gram matrix is what makes one kernel possible: sum(ol^2) = pre^T G pre, so the RMSNorm of the layer input
needs no second reduction over H once `pre` is known. The S programs of a group then meet at a global-memory
barrier (arrive counter + spin, reset by the last leaver so CUDA-graph replays start from zero). Phase 2: every
program loads the 16 critical partial columns of all S splits in one round trip, computes pre and the norm factor,
and writes its slice of the layer input (residual re-read, L1/L2 hot). Split 0 then finishes post' and the
sinkhorn comb' off the other programs' critical path (sinkhorn in per-token scalar form, no shuffles).

Two measured Triton pitfalls shaped the code: (1) using the post result directly as a tl.dot operand makes Triton
rematerialize the post in the warp-replicated dot-operand layout (255 registers, spills, 10-50x slower), so the
dot re-loads the residual it just wrote; (2) a sequential loop over S partials pays one L2 round trip per split.

Barrier safety: a group's S programs are consecutive in launch order (split is grid axis 0) and S <= SM count, so
in-order CTA dispatch always makes progress (the assumption CUTLASS serial split-K semaphores rely on). Counters
and workspace are per (device, stream); concurrent launches on different streams never share them.
`phase="two_pass"` splits the same kernel into two launches (partials, then finalize) with no barrier; it is the
fallback and what the Triton CPU interpreter runs (its programs execute sequentially, a spin would never end).

Env (engine hook): NUMINOUS_MHC_TWO_PASS=1 forces the two-launch variant.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

N_HC, N_MIX, N_WS = 4, 24, 64  # host-side ints
HC = tl.constexpr(N_HC)
MIX = tl.constexpr(N_MIX)  # HC * (2 + HC)
MIX_PAD = tl.constexpr(32)
WS_COLS = tl.constexpr(N_WS)
# workspace columns per (split, token): 0..3 pre logits, 4..13 Gram upper triangle, 16..35 post+comb logits
C_GRAM = tl.constexpr(4)
C_REST = tl.constexpr(16)


@triton.jit
def _col(tile, cols, k: tl.constexpr):
    return tl.sum(tl.where(cols[None, :] == k, tile, 0.0), axis=1)


@triton.jit
def _rcp(x):
    """Reciprocal via div.full.f32 (~2 ulp): the sinkhorn multiplies by it instead of 16 IEEE divides per
    half-iteration (measured: the divides made split 0's sinkhorn an 8.6 us tail at 96 tokens)."""
    return tl.fdiv(tl.full(x.shape, 1.0, tl.float32), x, ieee_rounding=False)


@triton.jit
def _mix_dot(acc, n, wt, PREC: tl.constexpr):
    """acc[BM, 32] += n[BM, BH] (bf16, exact) @ wt[BH, 32] (fp32).
    bf16x2 / bf16x3: wt split into 2 / 3 bf16 terms (products exact, fp32 accumulate): 2^-16 / ~2^-24 relative.
    tf32 / tf32x3 / ieee: Triton's fp32 dot modes."""
    if PREC == "bf16x2":
        hi = wt.to(tl.bfloat16)
        lo = (wt - hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(n, hi, acc)
        acc = tl.dot(n, lo, acc)
    elif PREC == "bf16x3":
        hi = wt.to(tl.bfloat16)
        r1 = wt - hi.to(tl.float32)
        mid = r1.to(tl.bfloat16)
        lo = (r1 - mid.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(n, hi, acc)
        acc = tl.dot(n, mid, acc)
        acc = tl.dot(n, lo, acc)
    else:
        acc = tl.dot(n.to(tl.float32), wt, acc, input_precision=PREC)
    return acc


@triton.jit
def _mhc_fused_kernel(
    x_ptr, stride_x,
    res_ptr,
    post_ptr, stride_post,
    comb_ptr,
    fn_ptr,
    scale_ptr, base_ptr,
    nw_ptr,
    res_out_ptr, post_out_ptr, comb_out_ptr, li_ptr, stride_li,
    ws_ptr, cnt_ptr,
    T,
    rms_eps, pre_eps, sk_eps, post_mult, norm_eps,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHUNK: tl.constexpr,  # hidden columns per program
    BLOCK_H: tl.constexpr,  # sub-tile width inside the chunk
    S: tl.constexpr,
    S_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    HAS_NORM: tl.constexpr,
    PHASE: tl.constexpr,  # 0 fused (barrier), 1 partials only, 2 finalize only
    DOT_PREC: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    s = tl.program_id(0)
    g = tl.program_id(1)
    rows = g * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < T
    rows64 = rows.to(tl.int64)
    h_base = s * CHUNK

    if PHASE != 2:
        pb = post_ptr + rows64 * stride_post
        cb = comb_ptr + rows64 * (HC * HC)
        a0 = tl.load(pb + 0, mask=rmask, other=0.0)[:, None]
        a1 = tl.load(pb + 1, mask=rmask, other=0.0)[:, None]
        a2 = tl.load(pb + 2, mask=rmask, other=0.0)[:, None]
        a3 = tl.load(pb + 3, mask=rmask, other=0.0)[:, None]
        c00 = tl.load(cb + 0, mask=rmask, other=0.0)[:, None]
        c01 = tl.load(cb + 1, mask=rmask, other=0.0)[:, None]
        c02 = tl.load(cb + 2, mask=rmask, other=0.0)[:, None]
        c03 = tl.load(cb + 3, mask=rmask, other=0.0)[:, None]
        c10 = tl.load(cb + 4, mask=rmask, other=0.0)[:, None]
        c11 = tl.load(cb + 5, mask=rmask, other=0.0)[:, None]
        c12 = tl.load(cb + 6, mask=rmask, other=0.0)[:, None]
        c13 = tl.load(cb + 7, mask=rmask, other=0.0)[:, None]
        c20 = tl.load(cb + 8, mask=rmask, other=0.0)[:, None]
        c21 = tl.load(cb + 9, mask=rmask, other=0.0)[:, None]
        c22 = tl.load(cb + 10, mask=rmask, other=0.0)[:, None]
        c23 = tl.load(cb + 11, mask=rmask, other=0.0)[:, None]
        c30 = tl.load(cb + 12, mask=rmask, other=0.0)[:, None]
        c31 = tl.load(cb + 13, mask=rmask, other=0.0)[:, None]
        c32 = tl.load(cb + 14, mask=rmask, other=0.0)[:, None]
        c33 = tl.load(cb + 15, mask=rmask, other=0.0)[:, None]
        ocols = tl.arange(0, MIX_PAD)
        omask = ocols < MIX
        acc = tl.zeros((BLOCK_M, MIX_PAD), dtype=tl.float32)
        # Gram products: 8-wide partial sums per thread (reshape keeps them in-thread), one reduction at the end
        t00 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        t01 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        t02 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        t03 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        t11 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        t12 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        t13 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        t22 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        t23 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        t33 = tl.zeros((BLOCK_M, BLOCK_H // 8), dtype=tl.float32)
        for h0 in tl.range(0, CHUNK, BLOCK_H, num_stages=NUM_STAGES):
            hs = h_base + h0 + tl.arange(0, BLOCK_H)
            hmask = hs < H
            m2 = rmask[:, None] & hmask[None, :]
            res_row = rows64[:, None] * (HC * H) + hs[None, :]
            xv = tl.load(x_ptr + rows64[:, None] * stride_x + hs[None, :], mask=m2, other=0.0).to(tl.float32)
            r0 = tl.load(res_ptr + res_row, mask=m2, other=0.0).to(tl.float32)
            r1 = tl.load(res_ptr + res_row + H, mask=m2, other=0.0).to(tl.float32)
            r2 = tl.load(res_ptr + res_row + 2 * H, mask=m2, other=0.0).to(tl.float32)
            r3 = tl.load(res_ptr + res_row + 3 * H, mask=m2, other=0.0).to(tl.float32)
            # new[j] = post[j] * x + sum_k comb[k, j] * res[k], summed in the reference's order
            f0 = (a0 * xv + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3).to(tl.bfloat16)
            f1 = (a1 * xv + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3).to(tl.bfloat16)
            f2 = (a2 * xv + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3).to(tl.bfloat16)
            f3 = (a3 * xv + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3).to(tl.bfloat16)
            tl.store(res_out_ptr + res_row, f0, mask=m2)
            tl.store(res_out_ptr + res_row + H, f1, mask=m2)
            tl.store(res_out_ptr + res_row + 2 * H, f2, mask=m2)
            tl.store(res_out_ptr + res_row + 3 * H, f3, mask=m2)
            e0 = f0.to(tl.float32)
            e1 = f1.to(tl.float32)
            e2 = f2.to(tl.float32)
            e3 = f3.to(tl.float32)
            t00 += tl.sum(tl.reshape(e0 * e0, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            t01 += tl.sum(tl.reshape(e0 * e1, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            t02 += tl.sum(tl.reshape(e0 * e2, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            t03 += tl.sum(tl.reshape(e0 * e3, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            t11 += tl.sum(tl.reshape(e1 * e1, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            t12 += tl.sum(tl.reshape(e1 * e2, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            t13 += tl.sum(tl.reshape(e1 * e3, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            t22 += tl.sum(tl.reshape(e2 * e2, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            t23 += tl.sum(tl.reshape(e2 * e3, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            t33 += tl.sum(tl.reshape(e3 * e3, (BLOCK_M, BLOCK_H // 8, 8)), axis=2)
            # mixes[t, o] += new[t, j, h] * fn[o, j*H + h]. The dot operand is re-loaded from the residual just
            # written (L1/L2 hot) instead of taken from registers (pitfall 1 in the module docstring).
            tl.debug_barrier()
            wtp = fn_ptr + (ocols[None, :] * (HC * H) + hs[:, None])  # int32 offsets: 24 * 4H < 2^31
            wm = hmask[:, None] & omask[None, :]
            acc = _mix_dot(acc, tl.load(res_out_ptr + res_row, mask=m2, other=0.0),
                           tl.load(wtp, mask=wm, other=0.0), DOT_PREC)
            acc = _mix_dot(acc, tl.load(res_out_ptr + res_row + H, mask=m2, other=0.0),
                           tl.load(wtp + H, mask=wm, other=0.0), DOT_PREC)
            acc = _mix_dot(acc, tl.load(res_out_ptr + res_row + 2 * H, mask=m2, other=0.0),
                           tl.load(wtp + 2 * H, mask=wm, other=0.0), DOT_PREC)
            acc = _mix_dot(acc, tl.load(res_out_ptr + res_row + 3 * H, mask=m2, other=0.0),
                           tl.load(wtp + 3 * H, mask=wm, other=0.0), DOT_PREC)

        wsb = ws_ptr + (s * T + rows64) * WS_COLS
        # mixes: cols 0..3 -> ws 0..3 (pre), cols 4..23 -> ws 16..35 (post, comb)
        dst = tl.where(ocols < HC, ocols, ocols + (C_REST - HC))
        tl.store(wsb[:, None] + dst[None, :], acc, mask=rmask[:, None] & omask[None, :])
        tl.store(wsb + C_GRAM + 0, tl.sum(t00, axis=1), mask=rmask)
        tl.store(wsb + C_GRAM + 1, tl.sum(t01, axis=1), mask=rmask)
        tl.store(wsb + C_GRAM + 2, tl.sum(t02, axis=1), mask=rmask)
        tl.store(wsb + C_GRAM + 3, tl.sum(t03, axis=1), mask=rmask)
        tl.store(wsb + C_GRAM + 4, tl.sum(t11, axis=1), mask=rmask)
        tl.store(wsb + C_GRAM + 5, tl.sum(t12, axis=1), mask=rmask)
        tl.store(wsb + C_GRAM + 6, tl.sum(t13, axis=1), mask=rmask)
        tl.store(wsb + C_GRAM + 7, tl.sum(t22, axis=1), mask=rmask)
        tl.store(wsb + C_GRAM + 8, tl.sum(t23, axis=1), mask=rmask)
        tl.store(wsb + C_GRAM + 9, tl.sum(t33, axis=1), mask=rmask)

    if PHASE == 0:
        # own residual writes must be visible to the whole program before the re-read below
        tl.debug_barrier()
        if S > 1:
            # group barrier: arrive, spin; the last leaver resets (below)
            tl.atomic_add(cnt_ptr + g * 2, 1, sem="release")
            while tl.atomic_add(cnt_ptr + g * 2, 0, sem="acquire") < S:
                pass
            tl.debug_barrier()

    if PHASE != 1:
        # ---- the 16 critical partial columns of all S splits, one round trip, bypassing L1 ----
        sidx = tl.arange(0, S_PAD)
        k16 = tl.arange(0, 16)
        crit_p = (ws_ptr + ((sidx[:, None, None] * T + rows64[None, :, None]) * WS_COLS + k16[None, None, :]))
        crit_m = (sidx[:, None, None] < S) & rmask[None, :, None] & (k16[None, None, :] < 14)
        crit = tl.sum(tl.load(crit_p, mask=crit_m, other=0.0, cache_modifier=".cg"), axis=0)  # [BM, 16]

        if PHASE == 0:
            if S > 1:
                tl.debug_barrier()
                left = tl.atomic_add(cnt_ptr + g * 2 + 1, 1, sem="acq_rel")
                if left == S - 1:
                    tl.atomic_xchg(cnt_ptr + g * 2, 0)
                    tl.atomic_xchg(cnt_ptr + g * 2 + 1, 0)

        q00 = _col(crit, k16, C_GRAM + 0)
        q01 = _col(crit, k16, C_GRAM + 1)
        q02 = _col(crit, k16, C_GRAM + 2)
        q03 = _col(crit, k16, C_GRAM + 3)
        q11 = _col(crit, k16, C_GRAM + 4)
        q12 = _col(crit, k16, C_GRAM + 5)
        q13 = _col(crit, k16, C_GRAM + 6)
        q22 = _col(crit, k16, C_GRAM + 7)
        q23 = _col(crit, k16, C_GRAM + 8)
        q33 = _col(crit, k16, C_GRAM + 9)
        rms = tl.rsqrt((q00 + q11 + q22 + q33) / (HC * H) + rms_eps)
        s0 = tl.load(scale_ptr + 0)
        p0 = tl.sigmoid(_col(crit, k16, 0) * rms * s0 + tl.load(base_ptr + 0)) + pre_eps
        p1 = tl.sigmoid(_col(crit, k16, 1) * rms * s0 + tl.load(base_ptr + 1)) + pre_eps
        p2 = tl.sigmoid(_col(crit, k16, 2) * rms * s0 + tl.load(base_ptr + 2)) + pre_eps
        p3 = tl.sigmoid(_col(crit, k16, 3) * rms * s0 + tl.load(base_ptr + 3)) + pre_eps
        if HAS_NORM:
            # sum(ol^2) = pre^T G pre (G symmetric: off-diagonals count twice)
            ss = (p0 * p0 * q00 + p1 * p1 * q11 + p2 * p2 * q22 + p3 * p3 * q33
                  + 2.0 * (p0 * p1 * q01 + p0 * p2 * q02 + p0 * p3 * q03
                           + p1 * p2 * q12 + p1 * p3 * q13 + p2 * p3 * q23))
            rn = tl.rsqrt(ss / H + norm_eps)[:, None]
        p0 = p0[:, None]
        p1 = p1[:, None]
        p2 = p2[:, None]
        p3 = p3[:, None]

        # ---- layer input for this chunk: bf16(bf16(ol) * rn * w) ----
        for h0 in tl.range(0, CHUNK, BLOCK_H, num_stages=NUM_STAGES):
            hs = h_base + h0 + tl.arange(0, BLOCK_H)
            hmask = hs < H
            m2 = rmask[:, None] & hmask[None, :]
            res_row = rows64[:, None] * (HC * H) + hs[None, :]
            ol = p0 * tl.load(res_out_ptr + res_row, mask=m2, other=0.0).to(tl.float32)
            ol += p1 * tl.load(res_out_ptr + res_row + H, mask=m2, other=0.0).to(tl.float32)
            ol += p2 * tl.load(res_out_ptr + res_row + 2 * H, mask=m2, other=0.0).to(tl.float32)
            ol += p3 * tl.load(res_out_ptr + res_row + 3 * H, mask=m2, other=0.0).to(tl.float32)
            out = ol.to(tl.bfloat16).to(tl.float32)
            if HAS_NORM:
                w = tl.load(nw_ptr + hs, mask=hmask, other=0.0).to(tl.float32)
                out = out * rn * w[None, :]
            tl.store(li_ptr + rows64[:, None] * stride_li + hs[None, :], out.to(tl.bfloat16), mask=m2)

        if s == 0:
            # ---- post' and comb' (split 0 only, after its layer-input work) ----
            k32 = tl.arange(0, 32)
            rest_p = (ws_ptr + ((sidx[:, None, None] * T + rows64[None, :, None]) * WS_COLS + C_REST
                                + k32[None, None, :]))
            rest_m = (sidx[:, None, None] < S) & rmask[None, :, None] & (k32[None, None, :] < MIX - HC)
            rest = tl.sum(tl.load(rest_p, mask=rest_m, other=0.0, cache_modifier=".cg"), axis=0)  # [BM, 32]
            s1 = tl.load(scale_ptr + 1)
            s2 = tl.load(scale_ptr + 2)
            pob = post_out_ptr + rows64 * HC
            for j in tl.static_range(HC):
                v = tl.sigmoid(_col(rest, k32, j) * rms * s1 + tl.load(base_ptr + HC + j)) * post_mult
                tl.store(pob + j, v, mask=rmask)
            # sinkhorn in per-token scalar form: 16 [BM] vectors, row/col sums are plain adds (no shuffles)
            b8 = base_ptr + 2 * HC
            m00 = _col(rest, k32, 4) * rms * s2 + tl.load(b8 + 0)
            m01 = _col(rest, k32, 5) * rms * s2 + tl.load(b8 + 1)
            m02 = _col(rest, k32, 6) * rms * s2 + tl.load(b8 + 2)
            m03 = _col(rest, k32, 7) * rms * s2 + tl.load(b8 + 3)
            m10 = _col(rest, k32, 8) * rms * s2 + tl.load(b8 + 4)
            m11 = _col(rest, k32, 9) * rms * s2 + tl.load(b8 + 5)
            m12 = _col(rest, k32, 10) * rms * s2 + tl.load(b8 + 6)
            m13 = _col(rest, k32, 11) * rms * s2 + tl.load(b8 + 7)
            m20 = _col(rest, k32, 12) * rms * s2 + tl.load(b8 + 8)
            m21 = _col(rest, k32, 13) * rms * s2 + tl.load(b8 + 9)
            m22 = _col(rest, k32, 14) * rms * s2 + tl.load(b8 + 10)
            m23 = _col(rest, k32, 15) * rms * s2 + tl.load(b8 + 11)
            m30 = _col(rest, k32, 16) * rms * s2 + tl.load(b8 + 12)
            m31 = _col(rest, k32, 17) * rms * s2 + tl.load(b8 + 13)
            m32 = _col(rest, k32, 18) * rms * s2 + tl.load(b8 + 14)
            m33 = _col(rest, k32, 19) * rms * s2 + tl.load(b8 + 15)
            # row softmax (+eps)
            mx = tl.maximum(tl.maximum(m00, m01), tl.maximum(m02, m03))
            m00 = tl.exp(m00 - mx)
            m01 = tl.exp(m01 - mx)
            m02 = tl.exp(m02 - mx)
            m03 = tl.exp(m03 - mx)
            r = _rcp(m00 + m01 + m02 + m03)
            m00 = m00 * r + sk_eps
            m01 = m01 * r + sk_eps
            m02 = m02 * r + sk_eps
            m03 = m03 * r + sk_eps
            mx = tl.maximum(tl.maximum(m10, m11), tl.maximum(m12, m13))
            m10 = tl.exp(m10 - mx)
            m11 = tl.exp(m11 - mx)
            m12 = tl.exp(m12 - mx)
            m13 = tl.exp(m13 - mx)
            r = _rcp(m10 + m11 + m12 + m13)
            m10 = m10 * r + sk_eps
            m11 = m11 * r + sk_eps
            m12 = m12 * r + sk_eps
            m13 = m13 * r + sk_eps
            mx = tl.maximum(tl.maximum(m20, m21), tl.maximum(m22, m23))
            m20 = tl.exp(m20 - mx)
            m21 = tl.exp(m21 - mx)
            m22 = tl.exp(m22 - mx)
            m23 = tl.exp(m23 - mx)
            r = _rcp(m20 + m21 + m22 + m23)
            m20 = m20 * r + sk_eps
            m21 = m21 * r + sk_eps
            m22 = m22 * r + sk_eps
            m23 = m23 * r + sk_eps
            mx = tl.maximum(tl.maximum(m30, m31), tl.maximum(m32, m33))
            m30 = tl.exp(m30 - mx)
            m31 = tl.exp(m31 - mx)
            m32 = tl.exp(m32 - mx)
            m33 = tl.exp(m33 - mx)
            r = _rcp(m30 + m31 + m32 + m33)
            m30 = m30 * r + sk_eps
            m31 = m31 * r + sk_eps
            m32 = m32 * r + sk_eps
            m33 = m33 * r + sk_eps
            for it in tl.static_range(ITERS * 2 - 1):
                if it % 2 == 0:  # column normalize
                    c = _rcp(m00 + m10 + m20 + m30 + sk_eps)
                    m00 = m00 * c
                    m10 = m10 * c
                    m20 = m20 * c
                    m30 = m30 * c
                    c = _rcp(m01 + m11 + m21 + m31 + sk_eps)
                    m01 = m01 * c
                    m11 = m11 * c
                    m21 = m21 * c
                    m31 = m31 * c
                    c = _rcp(m02 + m12 + m22 + m32 + sk_eps)
                    m02 = m02 * c
                    m12 = m12 * c
                    m22 = m22 * c
                    m32 = m32 * c
                    c = _rcp(m03 + m13 + m23 + m33 + sk_eps)
                    m03 = m03 * c
                    m13 = m13 * c
                    m23 = m23 * c
                    m33 = m33 * c
                else:  # row normalize
                    c = _rcp(m00 + m01 + m02 + m03 + sk_eps)
                    m00 = m00 * c
                    m01 = m01 * c
                    m02 = m02 * c
                    m03 = m03 * c
                    c = _rcp(m10 + m11 + m12 + m13 + sk_eps)
                    m10 = m10 * c
                    m11 = m11 * c
                    m12 = m12 * c
                    m13 = m13 * c
                    c = _rcp(m20 + m21 + m22 + m23 + sk_eps)
                    m20 = m20 * c
                    m21 = m21 * c
                    m22 = m22 * c
                    m23 = m23 * c
                    c = _rcp(m30 + m31 + m32 + m33 + sk_eps)
                    m30 = m30 * c
                    m31 = m31 * c
                    m32 = m32 * c
                    m33 = m33 * c
            cbo = comb_out_ptr + rows64 * 16
            tl.store(cbo + 0, m00, mask=rmask)
            tl.store(cbo + 1, m01, mask=rmask)
            tl.store(cbo + 2, m02, mask=rmask)
            tl.store(cbo + 3, m03, mask=rmask)
            tl.store(cbo + 4, m10, mask=rmask)
            tl.store(cbo + 5, m11, mask=rmask)
            tl.store(cbo + 6, m12, mask=rmask)
            tl.store(cbo + 7, m13, mask=rmask)
            tl.store(cbo + 8, m20, mask=rmask)
            tl.store(cbo + 9, m21, mask=rmask)
            tl.store(cbo + 10, m22, mask=rmask)
            tl.store(cbo + 11, m23, mask=rmask)
            tl.store(cbo + 12, m30, mask=rmask)
            tl.store(cbo + 13, m31, mask=rmask)
            tl.store(cbo + 14, m32, mask=rmask)
            tl.store(cbo + 15, m33, mask=rmask)


_COUNTERS: dict = {}
_WS: dict = {}


def _counters(device: torch.device, n: int) -> torch.Tensor:
    key = (device, torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else 0)
    buf = _COUNTERS.get(key)
    if buf is None or buf.numel() < n:
        buf = torch.zeros(max(n, 1 << 14), dtype=torch.int32, device=device)
        _COUNTERS[key] = buf
    return buf


def _workspace(device: torch.device, n: int) -> torch.Tensor:
    key = (device, torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else 0)
    buf = _WS.get(key)
    if buf is None or buf.numel() < n:
        buf = torch.empty(max(n, 1 << 20), dtype=torch.float32, device=device)
        _WS[key] = buf
    return buf


def pick_config(num_tokens: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, CHUNK, BLOCK_H, num_warps). Measured on B200 (see agents/status/kernels.md)."""
    if num_tokens <= 256:
        return 16, 256, 64, 4
    return 32, 512, 64, 4


def mhc_post_pre_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_iters: int,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: Optional[float] = None,
    *,
    layer_input_out: Optional[torch.Tensor] = None,
    phase: str = "fused",
    config: Optional[tuple[int, int, int, int]] = None,
    dot_precision: str = "bf16x3",
    num_stages: int = 1,
):
    """Returns (residual_cur [T, 4, H] bf16, post_cur [T, 4] fp32, comb_cur [T, 4, 4] fp32, layer_input [T, H] bf16).

    x [T, H] bf16 (block output), residual [T, 4, H] bf16, post [T, 4] or [T, 4, 1] fp32, comb [T, 4, 4] fp32,
    fn [24, 4H] fp32, hc_scale [3], hc_base [24] fp32, norm_weight [H] (any float dtype, read as bf16 like the
    reference). phase: "fused" (one launch, group barrier) or "two_pass" (two launches, no barrier).
    """
    assert residual.dtype == torch.bfloat16 and x.dtype == torch.bfloat16
    T, hc, H = residual.shape
    assert hc == N_HC and x.shape == (T, H) and fn.shape == (N_MIX, N_HC * H)
    residual = residual.contiguous()
    if x.stride(-1) != 1:
        x = x.contiguous()
    post2 = post.reshape(T, N_HC)
    if post2.stride(-1) != 1:
        post2 = post2.contiguous()
    comb = comb.reshape(T, N_HC, N_HC).contiguous().float()
    fn = fn.contiguous().float()
    dev = residual.device
    res_out = torch.empty_like(residual)
    post_out = torch.empty(T, N_HC, dtype=torch.float32, device=dev)
    comb_out = torch.empty(T, N_HC, N_HC, dtype=torch.float32, device=dev)
    li = layer_input_out if layer_input_out is not None else torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if T == 0:
        return res_out, post_out, comb_out, li
    has_norm = norm_weight is not None
    nw = norm_weight.to(torch.bfloat16).contiguous() if has_norm else li  # dummy pointer when unused
    bm, chunk, bh, nw_warps = config or pick_config(T)
    S = triton.cdiv(H, chunk)
    assert S <= 64, "partials are reduced in one [S_PAD, BM, 16] load; use a larger CHUNK"
    G = triton.cdiv(T, bm)
    ws = _workspace(dev, S * T * N_WS)
    cnt = _counters(dev, 2 * G)
    common = dict(H=H, BLOCK_M=bm, CHUNK=chunk, BLOCK_H=bh, S=S, S_PAD=triton.next_power_of_2(S), ITERS=sinkhorn_iters, HAS_NORM=has_norm,
                  DOT_PREC=dot_precision, NUM_STAGES=num_stages, num_warps=nw_warps)
    args = (x, x.stride(0), residual, post2, post2.stride(0), comb, fn, hc_scale.float().contiguous(),
            hc_base.float().contiguous(), nw, res_out, post_out, comb_out, li, li.stride(0), ws, cnt, T,
            float(rms_eps), float(hc_pre_eps), float(hc_sinkhorn_eps), float(hc_post_mult_value),
            float(norm_eps if norm_eps is not None else 0.0))
    if phase == "fused":
        if dev.type == "cuda":
            assert S <= torch.cuda.get_device_properties(dev).multi_processor_count
        _mhc_fused_kernel[(S, G)](*args, PHASE=0, **common)
    elif phase == "two_pass":
        _mhc_fused_kernel[(S, G)](*args, PHASE=1, **common)
        _mhc_fused_kernel[(S, G)](*args, PHASE=2, **common)
    else:
        raise ValueError(phase)
    return res_out, post_out, comb_out, li


def default_phase() -> str:
    return "two_pass" if os.environ.get("NUMINOUS_MHC_TWO_PASS", "0") == "1" else "fused"


# ------------------------------------------------------------------------------------------------ reference
def mhc_post_pre_reference(x, residual, post, comb, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                           hc_post_mult_value, sinkhorn_iters, norm_weight=None, norm_eps=None,
                           dtype=torch.float32):
    """Torch reference with the engine's rounding points (bf16 residual, bf16 stash before the RMSNorm).
    dtype=torch.float64 gives the high-precision reference used to judge both implementations."""
    T, hc, H = residual.shape
    post = post.reshape(T, hc).to(dtype)
    comb = comb.reshape(T, hc, hc).to(dtype)
    new = (post[:, :, None] * x[:, None, :].to(dtype)
           + torch.einsum("tkj,tkh->tjh", comb, residual.to(dtype))).to(torch.bfloat16)
    xf = new.to(dtype).reshape(T, hc * H)
    mixes = xf @ fn.to(dtype).T
    rms = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / (hc * H) + rms_eps)
    mixes = mixes * rms
    sc, b = hc_scale.to(dtype), hc_base.to(dtype)
    pre = torch.sigmoid(mixes[:, :hc] * sc[0] + b[:hc]) + hc_pre_eps
    post_out = torch.sigmoid(mixes[:, hc:2 * hc] * sc[1] + b[hc:2 * hc]) * hc_post_mult_value
    cm = (mixes[:, 2 * hc:] * sc[2] + b[2 * hc:]).reshape(T, hc, hc)
    cm = torch.exp(cm - cm.amax(dim=2, keepdim=True))
    cm = cm / cm.sum(dim=2, keepdim=True) + hc_sinkhorn_eps
    cm = cm / (cm.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        cm = cm / (cm.sum(dim=2, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)
    ol = (pre[:, :, None] * new.to(dtype)).sum(1)
    if norm_weight is not None:
        rn = torch.rsqrt((ol * ol).sum(-1, keepdim=True) / H + norm_eps)
        li = ol.to(torch.bfloat16).to(dtype) * rn * norm_weight.to(torch.bfloat16).to(dtype)
    else:
        li = ol
    return new, post_out.float(), cm.float(), li.to(torch.bfloat16)


def make_inputs(T: int, H: int = 7168, device="cuda", seed: int = 0):
    """Synthetic tensors at Pro's scales: fn ~ N(0, 1e-2), post in (0, 2), comb doubly stochastic-ish."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(T, H, generator=g).bfloat16()
    residual = torch.randn(T, N_HC, H, generator=g).bfloat16()
    post = 2 * torch.sigmoid(torch.randn(T, N_HC, generator=g))
    comb = torch.softmax(torch.randn(T, N_HC, N_HC, generator=g), dim=-1)
    fn = torch.randn(N_MIX, N_HC * H, generator=g) * 1e-2
    hc_scale = torch.rand(3, generator=g) + 0.5
    hc_base = torch.randn(N_MIX, generator=g) * 0.1
    nw = (1 + 0.1 * torch.randn(H, generator=g)).bfloat16()
    out = [x, residual, post, comb, fn, hc_scale, hc_base, nw]
    return [t.to(device) for t in out]


PRO_EPS = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.0, sinkhorn_iters=20,
               norm_eps=1e-6)


# ------------------------------------------------------------------------------------------------ engine glue
_LOGGED: set = set()


def _log_once(key: str, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        import logging

        logging.getLogger("numinous_pro").warning(msg)


def engine_enabled() -> bool:
    return os.environ.get("NUMINOUS_MHC_FUSED", "0") == "1"


def engine_boundary(layer_input, residual, post, comb, hc_fn, hc_scale, hc_base, hc_mult, rms_eps, hc_eps,
                    hc_post_mult, sinkhorn_iters, norm_weight, norm_eps):
    """Drop-in for apply_mhc_post_pre_boundary (hook `mhc_fused`): returns
    (residual, layer_input, post, comb, norm_applied) or None to let the engine path run."""
    if hc_mult != N_HC or layer_input.dim() != 2 or residual.dim() != 3 or not layer_input.is_cuda:
        _log_once("mhc_decline", f"numinous mhc_fused declined: hc_mult={hc_mult} dims={layer_input.dim()}/"
                                 f"{residual.dim()} cuda={layer_input.is_cuda}")
        return None
    from sglang.srt.distributed.device_communicators.pynccl_allocator import use_symmetric_memory
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.layers.dp_attention import is_allocation_symmetric

    T, H = layer_input.shape
    # the FFN-side layer input feeds the MoE; keep the engine's symmetric-memory allocation for its all-reduce
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        li = torch.empty(T, H, dtype=torch.bfloat16, device=layer_input.device)
    phase = default_phase()
    _log_once("mhc_active", f"numinous mhc_fused active (phase={phase}, one Triton kernel per mHC boundary)")
    res, post_out, comb_out, li = mhc_post_pre_fused(
        layer_input, residual, post, comb, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, hc_eps, hc_post_mult,
        sinkhorn_iters, norm_weight, norm_eps, layer_input_out=li, phase=phase)
    return res, li, post_out, comb_out, norm_weight is not None
