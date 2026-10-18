"""Split-K, swap-AB FP8 block-scaled GEMM for decode-sized M: DeepSeek V4 Pro's dense projections.

out[M, N] = X[M, K] @ W[N, K]^T with DeepSeek's recipe: X per-token 1x128 scales, W 128x128 block scales, FP8 e4m3.

Why. At decode every GPU multiplies ~96 rows (16 requests x 6 DSpark verify rows) against each weight, so each
projection is a weight stream. DeepGEMM tiles the output: wqkv_a (N 2,048) gives 16 tiles, so 16 of 148 SMs stream
14.7 MB and the call takes 10.8 us against a 2 us floor; the TP8 shared expert (N 768) 11 us against 1.5; wo_b
(56 tiles, K 16,384) 30 us against 15 (operations/modal/dense_gemm_bench.py, CUDA-graph timed, B300).

Design.
  - swap-AB: the weight tile fills the MMA's M side (BLOCK_N = 128, one weight-scale row per tile), the few
    activation rows its N side (BLOCK_M 16..128).
  - split-K: the K range divides across programs until the grid is ~2 programs per SM, so every SM streams weights.
  - per 128-K slab: one FP8 dot, rescaled by the weight block scale x the token scales, accumulated in fp32.
  - partials [SPLIT, M, N] fp32, then one reduce-and-cast kernel: deterministic, no atomics, no shared workspace
    (two projections on two streams, as the shared expert runs, cannot race).
"""
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _skinny_fp8_kernel(X, XS, W, WS, P, M, N, K, stride_xm, stride_xsm, stride_wn, stride_wsn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SLABS: tl.constexpr, DOT_F32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_ok = offs_n < N
    m_ok = offs_m < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for s in range(SLABS):
        kb = pid_k * SLABS + s
        offs_k = kb * 128 + tl.arange(0, 128)
        k_ok = offs_k < K
        w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :], mask=n_ok[:, None] & k_ok[None, :], other=0.0)
        x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :], mask=m_ok[:, None] & k_ok[None, :], other=0.0)
        if DOT_F32:
            part = tl.dot(w.to(tl.float32), tl.trans(x.to(tl.float32)))
        else:
            part = tl.dot(w, tl.trans(x))
        live = kb * 128 < K
        ws = tl.load(WS + (offs_n // 128) * stride_wsn + kb, mask=n_ok & live, other=0.0)
        xs = tl.load(XS + offs_m * stride_xsm + kb, mask=m_ok & live, other=0.0)
        acc += part * ws[:, None] * xs[None, :]
    tl.store(P + pid_k * M * N + offs_m[None, :] * N + offs_n[:, None], acc, mask=n_ok[:, None] & m_ok[None, :])


@triton.jit
def _reduce_cast_kernel(P, OUT, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT):
        acc += tl.load(P + s * MN + offs, mask=ok, other=0.0)
    tl.store(OUT + offs, acc.to(OUT.dtype.element_ty), mask=ok)


def plan(m: int, n: int, k: int, sms: int = 148, block_n: int = 128, per_sm: int = 2,
         block_m: int | None = None) -> dict:
    """Tile sizes and split for one call: ~per_sm programs per SM, at least one 128-slab per split."""
    if block_m is None:
        block_m = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
    tiles = math.ceil(n / block_n) * math.ceil(m / block_m)
    kb = math.ceil(k / GROUP)
    split = max(1, min(kb, math.ceil(per_sm * sms / tiles)))
    slabs = math.ceil(kb / split)
    split = math.ceil(kb / slabs)            # no empty trailing splits
    return {"BLOCK_M": block_m, "BLOCK_N": block_n, "SPLIT": split, "SLABS": slabs}


def gemm(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor,
         out_dtype: torch.dtype = torch.bfloat16, out: torch.Tensor | None = None, *, per_sm: int = 2,
         block_n: int = 128, block_m: int | None = None, num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    """xq [M, K] fp8 (or fp32 holding fp8 values under the interpreter), xs [M, K/128] fp32 row-major,
    wq [N, K], ws [N/128, K/128] fp32. Returns out [M, N]."""
    m, k = xq.shape
    n = wq.shape[0]
    if out is None:
        out = torch.empty(m, n, dtype=out_dtype, device=xq.device)
    if m == 0:
        return out
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    sms = 4 if interp else torch.cuda.get_device_properties(xq.device).multi_processor_count
    cfg = plan(m, n, k, sms, block_n=block_n, per_sm=per_sm, block_m=block_m)
    p = torch.empty(cfg["SPLIT"], m, n, dtype=torch.float32, device=xq.device)
    grid = (triton.cdiv(n, cfg["BLOCK_N"]), cfg["SPLIT"], triton.cdiv(m, cfg["BLOCK_M"]))
    _skinny_fp8_kernel[grid](xq, xs, wq, ws, p, m, n, k, xq.stride(0), xs.stride(0), wq.stride(0), ws.stride(0),
                             BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], SLABS=cfg["SLABS"], DOT_F32=interp,
                             num_warps=num_warps, num_stages=num_stages)
    mn = m * n
    _reduce_cast_kernel[(triton.cdiv(mn, 1024),)](p, out, mn, SPLIT=cfg["SPLIT"], BLOCK=1024)
    return out


def quantize_rows(x: torch.Tensor):
    """Per-token 1x128 FP8 quantization with row-major fp32 scales (the reference; the engine uses SGLang's kernel)."""
    m, k = x.shape
    xb = x.float().view(m, k // GROUP, GROUP)
    s = xb.abs().amax(dim=2).clamp(min=1e-4) / 448.0
    q = (xb / s[:, :, None]).to(torch.float8_e4m3fn).view(m, k)
    return q, s


def reference(xq, xs, wq, ws) -> torch.Tensor:
    """fp32 reference of the block-scaled product."""
    m, k = xq.shape
    n = wq.shape[0]
    x = xq.float() * xs.repeat_interleave(GROUP, dim=1)[:, :k]
    w = wq.float() * ws.repeat_interleave(GROUP, dim=0)[:n].repeat_interleave(GROUP, dim=1)[:, :k]
    return x @ w.t()
