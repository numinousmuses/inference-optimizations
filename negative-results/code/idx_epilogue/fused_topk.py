"""Exact indexer top-k for GLM-5.3 DSA prefill without reading back a rows x context fp32 score matrix (agent glmidx).

Needs DeepGEMM's sm100_mqa_logits.cuh patched by patch_mqa_cand.py (bitmap candidate epilogue in compressed-logits
mode; GLM's DSA path never uses compressed logits, so every stock call is unchanged).

    idx, ok = indexer_topk_fused(q8, w, k8, k_scale, ks, ke, topk=2048)

1. threshold: stock DeepGEMM logits over every S-th key; thr[r] = the m-th largest sampled score of row r's window,
   m = ceil(topk / S * margin) (rows whose window holds <= topk keys: thr = -inf, every key selected).
2. candidate pass: the patched kernel scores every key once, stores only scores > thr[r] (sparse) and marks them in a
   per-row bitmap (no score matrix is read back, no top-k over the full context).
3. compaction (Triton, below): bitmap -> (offset, score) lists, then top-k over the lists (SGLang fast_topk_v2).
Exact whenever row r has >= min(topk, window) candidates (then its true top-k all exceed thr[r]); `ok` flags rows
that must be recomputed on the stock path (margin 3 makes that astronomically rare on real score distributions).
Returns key offsets relative to ks[r], -1 padded.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

CAPC = 16384  # max candidates kept per row after compaction


@triton.jit
def _word_prefix_kernel(bitmap_ptr, words_per_row, len_ptr, prefix_ptr, cnt_ptr, WB: tl.constexpr):
    """prefix[r, w] = number of set bits in words < w of row r (exclusive), cnt[r] = total."""
    r = tl.program_id(0)
    n_words = (tl.load(len_ptr + r) + 31) // 32
    bits = tl.arange(0, 32)
    base = 0
    for w0 in range(0, n_words, WB):
        wi = w0 + tl.arange(0, WB)
        words = tl.load(bitmap_ptr + r.to(tl.int64) * words_per_row + wi, mask=wi < n_words, other=0)
        pc = tl.sum(((words[:, None] >> bits[None, :]) & 1).to(tl.int32), axis=1)
        tl.store(prefix_ptr + r.to(tl.int64) * words_per_row + wi, tl.cumsum(pc, axis=0) - pc + base, mask=wi < n_words)
        base += tl.sum(pc, axis=0)
    tl.store(cnt_ptr + r, base)


@triton.jit
def _scatter_kernel(bitmap_ptr, prefix_ptr, words_per_row, scores_ptr, score_stride, len_ptr, out_idx_ptr, out_val_ptr,
                    CAP: tl.constexpr, WB: tl.constexpr):
    r = tl.program_id(0)
    wb = tl.program_id(1)
    n_words = (tl.load(len_ptr + r) + 31) // 32
    wi = wb * WB + tl.arange(0, WB)
    wm = wi < n_words
    words = tl.load(bitmap_ptr + r.to(tl.int64) * words_per_row + wi, mask=wm, other=0)
    pre = tl.load(prefix_ptr + r.to(tl.int64) * words_per_row + wi, mask=wm, other=0)
    bits = tl.arange(0, 32)
    hit = ((words[:, None] >> bits[None, :]) & 1).to(tl.int32)                      # [WB, 32]
    pos = pre[:, None] + tl.cumsum(hit, axis=1) - hit
    off = wi[:, None] * 32 + bits[None, :]
    keep = (hit == 1) & (pos < CAP)
    val = tl.load(scores_ptr + r.to(tl.int64) * score_stride + off, mask=keep, other=0.0)
    tl.store(out_idx_ptr + r.to(tl.int64) * CAP + pos, off, mask=keep)
    tl.store(out_val_ptr + r.to(tl.int64) * CAP + pos, val, mask=keep)


def compact(bitmap, words_per_row, scores, window, cidx, cval, cnt, prefix):
    rows = window.shape[0]
    _word_prefix_kernel[(rows,)](bitmap, words_per_row, window, prefix, cnt, WB=2048, num_warps=8)
    wb = 128
    _scatter_kernel[(rows, triton.cdiv(words_per_row, wb))](bitmap, prefix, words_per_row, scores, scores.stride(0),
                                                            window, cidx, cval, CAP=cidx.shape[1], WB=wb, num_warps=4)


def _topk_rows(vals: torch.Tensor, lengths: torch.Tensor, k: int) -> torch.Tensor:
    """Exact top-k positions within the first lengths[r] entries of each row (-1 padded). Uses SGLang's DSV4 top-k
    v2 ragged kernel (exact on these inputs, measured); NOT sgl_kernel.fast_topk_v2, which measured inexact (0.77-0.98
    overlap) on candidate lists whose scores sit in a narrow band."""
    rows = vals.shape[0]
    try:
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
    except Exception:
        topk_transform_ragged_v2 = None
    if topk_transform_ragged_v2 is not None and k == 2048:
        out = torch.empty(rows, k, dtype=torch.int32, device=vals.device)
        zeros = torch.zeros(rows, dtype=torch.int32, device=vals.device)
        topk_transform_ragged_v2(vals, lengths.to(torch.int32), out_offsets=zeros, out_indices=out, row_starts=zeros)
        return out
    masked = vals.masked_fill(torch.arange(vals.shape[1], device=vals.device)[None] >= lengths[:, None], float("-inf"))
    kk = min(k, vals.shape[1])
    idx = torch.topk(masked, kk, dim=-1).indices.to(torch.int32)
    return torch.where(torch.arange(kk, device=vals.device)[None] < lengths[:, None], idx, torch.full_like(idx, -1))


class Workspace:
    """Reusable buffers for one (rows, context) shape class."""

    def __init__(self, rows: int, max_len: int, device, capc: int = CAPC):
        self.rows, self.capc = rows, capc
        self.r4 = (rows + 3) // 4 * 4
        self.stride = (max_len + 255) // 256 * 256
        self.words = self.stride // 32
        self.aux = torch.empty(2 * self.r4 + rows * self.words, dtype=torch.int32, device=device)
        self.cidx = torch.empty(rows, capc, dtype=torch.int32, device=device)
        self.cval = torch.empty(rows, capc, dtype=torch.float32, device=device)
        self.cnt = torch.empty(rows, dtype=torch.int32, device=device)
        self.prefix = torch.empty(rows, self.words, dtype=torch.int32, device=device)


def indexer_topk_fused(q8, w, k8, k_scale, ks, ke, topk: int = 2048, sample_stride: int = 64, margin: float = 3.0,
                       ws: Workspace | None = None, return_stats: bool = False):
    import deep_gemm

    rows = q8.shape[0]
    dev = q8.device
    window = (ke - ks).to(torch.int32)
    max_len = int(window.max())
    if ws is None or ws.rows != rows or ws.stride < (max_len + 255) // 256 * 256:
        ws = Workspace(rows, max_len, dev)
    s = sample_stride
    # 1. threshold from the sampled keys j*s
    ks_s = ((ks + s - 1) // s).to(torch.int32)
    ke_s = ((ke + s - 1) // s).to(torch.int32)
    ls = deep_gemm.fp8_mqa_logits(q8, (k8[::s].contiguous(), k_scale[::s].contiguous()), w, ks_s, ke_s,
                                  clean_logits=False)
    col = torch.arange(ls.shape[1], device=dev, dtype=torch.int32)
    ls = ls.masked_fill((col[None] < ks_s[:, None]) | (col[None] >= ke_s[:, None]), float("-inf"))
    m = min(max(1, math.ceil(topk / s * margin)), ls.shape[1])
    thr = torch.topk(ls, m, dim=-1, sorted=False).values.amin(-1)
    thr = torch.where(window <= topk, torch.full_like(thr, float("-inf")), thr)
    # 2. candidate pass (patched kernel, compressed-logits mode): sparse scores + bitmap
    r4, aux = ws.r4, ws.aux
    aux[:rows].copy_(ks)
    aux[r4:r4 + rows].copy_(thr.view(torch.int32))
    aux[2 * r4:].zero_()
    scores = deep_gemm.fp8_mqa_logits(q8, (k8, k_scale), w, aux[:rows], ke, clean_logits=False, max_seqlen_k=max_len)
    assert scores.stride(0) == ws.stride, (scores.stride(), ws.stride)
    # 3. compaction + top-k over the candidates
    compact(aux[2 * r4:], ws.words, scores, window, ws.cidx, ws.cval, ws.cnt, ws.prefix)
    n = ws.cnt.clamp(max=ws.capc)
    sel = _topk_rows(ws.cval, n, topk).long()  # fast_topk_v2 reads only the first n[r] entries of each row
    out = torch.gather(ws.cidx, 1, sel.clamp(min=0))
    need = torch.minimum(window, torch.full_like(window, topk))
    out = torch.where((sel >= 0) & (torch.arange(topk, device=dev)[None] < need[:, None]), out,
                      torch.full_like(out, -1))
    ok = (ws.cnt >= need) & (ws.cnt <= ws.capc)
    if return_stats:
        c = ws.cnt.float()
        return out, ok, {"cnt_mean": c.mean().item(), "cnt_max": int(ws.cnt.max()), "cnt_min": int(ws.cnt.min()),
                         "fail_rows": int((~ok).sum())}
    return out, ok
