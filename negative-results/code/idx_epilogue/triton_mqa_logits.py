"""Triton SM100 prototype of GLM-5.3's DSA indexer prefill scores (agent glmidx), toward a fused logits + top-k kernel.

score[r, j] = k_scale[j] * sum_h w[r, h] * relu(q[r, h, :] . k[j, :]),  q FP8 [R, 32, 128] (per-(r,h) scale folded
into w), k FP8 [T, 128] with one fp32 scale per key, causal windows [ks[r], ke[r]).

Tile: a program owns BK keys (MMA M) and loops over row groups of BQ rows x 32 heads (MMA N = BQ*32 = 128), so each
thread's accumulator row is one key and the head sum is thread-local, like DeepGEMM's layout.

  mqa_logits(q8, w, k8, ks_, ke, ...)                  -> fp32 [R, T] scores (reference / timing)
  mqa_topk_threshold(q8, w, k8, k_scale, ks, ke, thr)  -> per-row candidates above thr (fused, no score matrix)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scores_tile(k_tile, q_ptr, w_ptr, row0, n_rows, BQ: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    """k_tile [BK, D] fp8 -> scores [BK, BQ] fp32 (without the key scale) for rows row0..row0+BQ."""
    qh = tl.arange(0, BQ * H)
    rows = row0 + qh // H
    heads = qh % H
    d = tl.arange(0, D)
    qmask = rows < n_rows
    q = tl.load(q_ptr + (rows * H + heads)[:, None] * D + d[None, :], mask=qmask[:, None], other=0.0)  # [BQ*H, D]
    acc = tl.dot(k_tile, tl.trans(q))                                                              # [BK, BQ*H]
    w = tl.load(w_ptr + rows * H + heads, mask=qmask, other=0.0)                                    # [BQ*H]
    s = tl.maximum(acc, 0.0) * w[None, :]
    s = tl.reshape(s, (acc.shape[0], BQ, H))
    return tl.sum(s, axis=2)                                                                        # [BK, BQ]


@triton.jit
def _mqa_logits_kernel(q_ptr, w_ptr, k_ptr, ksc_ptr, ks_ptr, ke_ptr, out_ptr, n_rows, n_keys, out_stride,
                       ROWS_PER_PROG: tl.constexpr, BK: tl.constexpr, BQ: tl.constexpr, H: tl.constexpr,
                       D: tl.constexpr, STORE: tl.constexpr = True, WS: tl.constexpr = False):
    pid_k = tl.program_id(0)
    pid_r = tl.program_id(1)
    keys = pid_k * BK + tl.arange(0, BK)
    kmask = keys < n_keys
    d = tl.arange(0, D)
    k_tile = tl.load(k_ptr + keys[:, None].to(tl.int64) * D + d[None, :], mask=kmask[:, None], other=0.0)
    ksc = tl.load(ksc_ptr + keys, mask=kmask, other=0.0)
    r_begin = pid_r * ROWS_PER_PROG
    total = tl.zeros((BK, BQ), dtype=tl.float32)
    for rb in tl.range(0, ROWS_PER_PROG, BQ, warp_specialize=WS):
        row0 = r_begin + rb
        sc = _scores_tile(k_tile, q_ptr, w_ptr, row0, n_rows, BQ, H, D) * ksc[:, None]            # [BK, BQ]
        rr = row0 + tl.arange(0, BQ)
        if STORE:
            tl.store(out_ptr + rr[None, :].to(tl.int64) * out_stride + keys[:, None], sc,
                     mask=kmask[:, None] & (rr[None, :] < n_rows))
        else:
            total += sc
    if not STORE:
        tl.store(out_ptr + pid_r * out_stride + keys, tl.sum(total, axis=1), mask=kmask)


def mqa_logits(q8: torch.Tensor, w: torch.Tensor, k8: torch.Tensor, k_scale: torch.Tensor, ks: torch.Tensor,
               ke: torch.Tensor, *, bk: int = 128, bq: int = 4, rows_per_prog: int = 64, num_warps: int = 4,
               num_stages: int = 2, store: bool = True, ws: bool = False) -> torch.Tensor:
    r, h, d = q8.shape
    t = k8.shape[0]
    stride = (t + 255) // 256 * 256
    out = torch.empty((r, stride), dtype=torch.float32, device=q8.device)
    grid = (triton.cdiv(t, bk), triton.cdiv(r, rows_per_prog))
    _mqa_logits_kernel[grid](q8, w, k8, k_scale, ks, ke, out, r, t, stride, ROWS_PER_PROG=rows_per_prog, BK=bk,
                             BQ=bq, H=h, D=d, STORE=store, WS=ws, num_warps=num_warps,
                             num_stages=num_stages)
    return out[:, :t]
