#!/usr/bin/env python3
"""Split-K FP8 blockscale GEMM for small M (GLM-5.3 decode / verify rows) on gfx950, reading the CK bpreshuffle
weight layout in place.

    y[M, N] = (x[M, K] * xs[M, K/128]) @ (w[N, K] * ws[ceil(N/128), K/128])^T      (fp8 e4m3 in, bf16 out)

Why (measured, amdglm 2026-10-04, research/gpu-evidence/amdglm/fp8gemm_1002_partial.jsonl + prof_amdglm_P4_1002):
FP8 blockscale GEMMs are 21% of the Value decode cycle and the CK bpreshuffle kernel serving them reaches only
2-5 TB/s of the MI355X's 8 TB/s at M 6-288 (qkv_a 2624x6144 M24: 7.6 us = 2.1 TB/s; o_proj 6144x16384 M72:
~50 us = 2 TB/s). At these M a GEMM is a weight stream: the floor is bytes(w) / HBM BW. CK tiles N only, so a
6144-wide output gives 48-96 workgroups for 256 CUs and most of the GPU idles. This kernel splits K as well:
(N / BLOCK_N) x SPLIT_K programs, each streaming a [BLOCK_N, K / SPLIT_K] weight slab once with all M rows resident,
fp32 partials reduced by a second small kernel (deterministic, no atomics).

Weight layout: aiter.ops.shuffle.shuffle_weight(w, layout=(16, 16)) for fp8 is
    w.view(N/16, 16, K/32, 2, 16).permute(0, 2, 3, 1, 4)
so byte (n, k) sits at ((n/16 * K/32 + k/32) * 2 + (k%32)/16) * 256 + (n%16) * 16 + k%16; each 16-row group stores
512 contiguous bytes per 32-column chunk, and a [BLOCK_N, 128] tile is 16-byte vector loads. SHUFFLED=False reads a
plain row-major [N, K] weight (reference / tests).

Activation scales: logical [M, K/128] with any strides (SGLang hands the gfx95 bpreshuffle path a column-major view,
materialize_bpreshuffle_fp8_scale), so pass xs as-is.

    python3 rocm_fp8_splitk_gemm.py --check            # CPU (TRITON_INTERPRET=1) numerics vs the fp32 reference
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_bs_splitk_kernel(x_ptr, xs_ptr, w_ptr, ws_ptr, p_ptr, M, N, K,
                          stride_xm, stride_xsm, stride_xsk, stride_wsn, stride_wsk,
                          stride_ps, stride_pm,
                          KB_PER_SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          SHUFFLED: tl.constexpr, DOT_FP8: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)
    BLOCK_K: tl.constexpr = 128
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kk = tl.arange(0, BLOCK_K)
    mmask = offs_m < M
    nmask = offs_n < N
    n64 = offs_n.to(tl.int64)
    kb_total = K // BLOCK_K
    kb0 = pid_s * KB_PER_SPLIT
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for i in range(KB_PER_SPLIT):
        kb = kb0 + i
        kvalid = kb < kb_total
        offs_k = kb * BLOCK_K + offs_kk
        a = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :],
                    mask=mmask[:, None] & kvalid, other=0.0)
        if SHUFFLED:
            woff = (((n64[:, None] // 16) * (K // 32) + offs_k[None, :] // 32) * 2 + (offs_k[None, :] % 32) // 16) \
                * 256 + (n64[:, None] % 16) * 16 + offs_k[None, :] % 16
        else:
            woff = n64[:, None] * K + offs_k[None, :]
        w = tl.load(w_ptr + woff, mask=nmask[:, None] & kvalid, other=0.0)
        if DOT_FP8:
            d = tl.dot(a, tl.trans(w))
        else:  # interpreter / reference path: fp8 -> fp32 before the dot
            d = tl.dot(a.to(tl.float32), tl.trans(w.to(tl.float32)))
        xs = tl.load(xs_ptr + offs_m * stride_xsm + kb * stride_xsk, mask=mmask & kvalid, other=0.0)
        wsc = tl.load(ws_ptr + (offs_n // 128) * stride_wsn + kb * stride_wsk, mask=nmask & kvalid, other=0.0)
        acc += d * xs[:, None] * wsc[None, :]
    p = p_ptr + pid_s * stride_ps + offs_m[:, None].to(tl.int64) * stride_pm + offs_n[None, :]
    tl.store(p, acc, mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _splitk_reduce_kernel(p_ptr, y_ptr, MN, SPLIT: tl.constexpr, stride_ps, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(SPLIT):
        acc += tl.load(p_ptr + s * stride_ps + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, acc.to(tl.bfloat16), mask=mask)


def pick_split(n: int, k: int, block_n: int, cus: int = 256, waves: float = 2.0) -> int:
    """Smallest power-of-two SPLIT_K giving >= `waves` x CUs programs, capped at K/128 blocks."""
    tiles = triton.cdiv(n, block_n)
    kb = k // 128
    s = 1
    while tiles * s < waves * cus and s * 2 <= kb:
        s *= 2
    return s


def fp8_blockscale_splitk(x: torch.Tensor, xs: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, *,
                          shuffled: bool = True, split: int = 0, block_n: int = 128, block_m: int = 0,
                          num_warps: int = 4, num_stages: int = 2, dot_fp8: bool = True) -> torch.Tensor:
    m, k = x.shape
    n = w.shape[0] if not shuffled else w.numel() // k
    assert k % 128 == 0 and (not shuffled or (n % 16 == 0 and k % 32 == 0))
    assert xs.shape == (m, k // 128) and ws.shape == ((n + 127) // 128, k // 128), (xs.shape, ws.shape)
    split = split or pick_split(n, k, block_n)
    kb_per = triton.cdiv(k // 128, split)
    split = triton.cdiv(k // 128, kb_per)
    block_m = block_m or max(16, min(64, triton.next_power_of_2(m)))
    part = torch.empty(split, m, n, dtype=torch.float32, device=x.device)
    y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(n, block_n), split, triton.cdiv(m, block_m))
    extra = {} if os.environ.get("TRITON_INTERPRET") == "1" else {"num_warps": num_warps, "num_stages": num_stages}
    _fp8_bs_splitk_kernel[grid](x, xs, w, ws, part, m, n, k, x.stride(0), xs.stride(0), xs.stride(1),
                                ws.stride(0), ws.stride(1), part.stride(0), part.stride(1),
                                KB_PER_SPLIT=kb_per, BLOCK_M=block_m, BLOCK_N=block_n, SHUFFLED=shuffled,
                                DOT_FP8=dot_fp8, **extra)
    blk = 1024
    _splitk_reduce_kernel[(triton.cdiv(m * n, blk),)](part, y, m * n, SPLIT=split, stride_ps=part.stride(0),
                                                      BLOCK=blk)
    return y


_TABLE: dict | None = None


def table_lookup(n: int, k: int, m: int):
    """Launch config for (N, K, M) from the JSON table at $NUMINOUS_FP8_SPLITK_TABLE, or None (= stay on CK).
    Table: {"N,K": [[max_m, block_n, split, num_warps, num_stages], ...]} sorted by max_m; first max_m >= M wins;
    block_n 0 marks a measured M where CK was faster. M above the last row stays on CK."""
    global _TABLE
    if _TABLE is None:
        import json
        path = os.environ.get("NUMINOUS_FP8_SPLITK_TABLE", "")
        raw = json.load(open(path)) if path and os.path.exists(path) else {}
        _TABLE = {tuple(int(v) for v in key.split(",")): sorted(rows) for key, rows in raw.items()}
    for max_m, block_n, split, warps, stages in _TABLE.get((n, k), ()):
        if m <= max_m:  # block_n 0 = measured row where CK won: stay on CK
            return {"block_n": block_n, "split": split, "num_warps": warps, "num_stages": stages} if block_n else None
    return None


def shuffle_16x16(w: torch.Tensor) -> torch.Tensor:
    """aiter shuffle_weight(w, layout=(16, 16)) for 1-byte dtypes, reproduced for CPU tests."""
    n, k = w.shape
    return w.view(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous().view(n, k)


def reference(x, xs, w, ws):
    n, k = w.shape
    xd = x.float() * xs.float().repeat_interleave(128, 1)
    wd = w.float() * ws.float().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return xd @ wd.T


def _check() -> int:
    os.environ.setdefault("TRITON_INTERPRET", "1")
    dev = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1" else "cpu"
    torch.manual_seed(0)
    worst = 0.0
    for m, n, k in ((1, 256, 256), (6, 400, 512), (24, 2624 // 8, 768), (72, 128, 1024)):
        n = (n + 15) // 16 * 16
        w = (torch.randn(n, k) * 0.5).to(torch.float8_e4m3fn)
        ws = torch.rand((n + 127) // 128, k // 128) * 0.02 + 0.001
        x = (torch.randn(m, k) * 0.5).to(torch.float8_e4m3fn)
        xs_rm = torch.rand(m, k // 128) * 0.02 + 0.001
        xs_cm = xs_rm.t().contiguous().t()  # the column-major view SGLang passes on gfx95
        ref = reference(x, xs_rm, w, ws)
        for shuf, xs, split in ((False, xs_rm, 0), (True, xs_cm, 0), (True, xs_rm, 3)):
            wt = shuffle_16x16(w) if shuf else w
            y = fp8_blockscale_splitk(x.to(dev), xs.to(dev), wt.to(dev), ws.to(dev), shuffled=shuf, split=split,
                                      dot_fp8=(dev == "cuda"))
            rel = ((y.float().cpu() - ref).abs().max() / ref.abs().max()).item()
            worst = max(worst, rel)
            print(f"M={m} N={n} K={k} shuffled={shuf} split={split or 'auto'} rel={rel:.4f}", flush=True)
    ok = worst < 0.02
    print("PASS" if ok else "FAIL", f"worst rel {worst:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(_check() if a.check else 0)
