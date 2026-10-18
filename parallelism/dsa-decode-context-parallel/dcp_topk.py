"""Numinous: index math for decode context parallel (DCP) on the DeepSeek sparse attention (DSA) path.

Under DCP with size S, token position p lives on rank p % S at local slot p // S (the owner rule SGLang's dense-MLA
DCP already uses, layers/dcp/layout.py). The DSA indexer scores every token of a request and keeps the top-k; with
sharded keys each rank can only score its own tokens, so the selection becomes:

    local scores over local slots  ->  local top-k with global positions
    -> all-gather (S x k) candidates -> global top-k -> each rank keeps the candidates it owns, as local slots

Everything here is plain torch so it is unit-tested on CPU; the kernels that consume the outputs are the existing
FlashMLA sparse kernels, which accept -1 for absent entries.
"""
from __future__ import annotations

import torch


def local_to_global(local_idx: torch.Tensor, dcp_size: int, dcp_rank: int) -> torch.Tensor:
    """Global positions of this rank's local slots. -1 stays -1."""
    out = local_idx * dcp_size + dcp_rank
    return torch.where(local_idx < 0, torch.full_like(out, -1), out)


def global_to_local_owned(global_idx: torch.Tensor, dcp_size: int, dcp_rank: int) -> torch.Tensor:
    """Local slots for the positions this rank owns; every other entry becomes -1. Shape preserved."""
    owned = (global_idx >= 0) & (global_idx % dcp_size == dcp_rank)
    return torch.where(owned, global_idx // dcp_size, torch.full_like(global_idx, -1))


def merge_topk(candidate_scores: torch.Tensor, candidate_pos: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Global top-k from the gathered candidates.

    candidate_scores: [B, S*k] float, candidate_pos: [B, S*k] int64 with -1 for padding (scored as -inf).
    Returns (scores [B, k], positions [B, k]) sorted by score descending, -1 where fewer than k valid candidates.
    """
    scores = torch.where(candidate_pos >= 0, candidate_scores, torch.full_like(candidate_scores, float("-inf")))
    top_s, top_i = torch.topk(scores, k=min(k, scores.shape[-1]), dim=-1)
    top_p = torch.gather(candidate_pos, -1, top_i)
    top_p = torch.where(torch.isfinite(top_s), top_p, torch.full_like(top_p, -1))
    if top_p.shape[-1] < k:  # fewer candidates than k: pad
        pad = k - top_p.shape[-1]
        top_p = torch.cat([top_p, torch.full((top_p.shape[0], pad), -1, dtype=top_p.dtype)], dim=-1)
        top_s = torch.cat([top_s, torch.full((top_s.shape[0], pad), float("-inf"), dtype=top_s.dtype)], dim=-1)
    return top_s, top_p


def compact_owned(local_idx: torch.Tensor) -> torch.Tensor:
    """Move valid (>= 0) entries to the front of each row, -1 after them. Kernels that iterate a contiguous
    prefix of selected tokens want this; kernels that mask -1 natively do not need it."""
    valid = local_idx >= 0
    order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
    return torch.gather(local_idx, -1, order)


def reference_topk_global(scores_full: torch.Tensor, k: int) -> torch.Tensor:
    """What a single rank holding every token would select: positions of the top-k scores. For tests."""
    return torch.topk(scores_full, k=min(k, scores_full.shape[-1]), dim=-1).indices


def select_dcp(scores_full: torch.Tensor, k: int, dcp_size: int) -> list[torch.Tensor]:
    """Simulate the whole DCP selection on one device: returns each rank's owned local slots [B, k] (with -1).

    Used by the tests to prove the sharded selection reproduces the single-rank top-k set exactly, and as the
    executable specification for the GPU implementation.
    """
    B, T = scores_full.shape
    cands_s, cands_p = [], []
    for r in range(dcp_size):
        local = scores_full[:, r::dcp_size]                               # this rank's tokens, in local-slot order
        kk = min(k, local.shape[-1])
        s, i = torch.topk(local, k=kk, dim=-1)
        p = local_to_global(i, dcp_size, r)
        if kk < k:
            s = torch.cat([s, torch.full((B, k - kk), float("-inf"), dtype=s.dtype)], dim=-1)
            p = torch.cat([p, torch.full((B, k - kk), -1, dtype=p.dtype)], dim=-1)
        cands_s.append(s)
        cands_p.append(p)
    gathered_s, gathered_p = torch.cat(cands_s, dim=-1), torch.cat(cands_p, dim=-1)  # the all-gather
    _, top_p = merge_topk(gathered_s, gathered_p, k)
    return [compact_owned(global_to_local_owned(top_p, dcp_size, r)) for r in range(dcp_size)]


def owned_local_slots(slots: torch.Tensor, dcp_size: int, dcp_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel-facing form of the owner rule on selected KV slots: [rows, k] virtual slots (with -1 padding) ->
    ([rows, k] int32 local slots compacted to the front with -1 after, [rows] int32 valid counts)."""
    local = global_to_local_owned(slots.to(torch.int64), dcp_size, dcp_rank)
    local = compact_owned(local)
    counts = (local >= 0).sum(dim=-1).to(torch.int32)
    return local.to(torch.int32), counts
