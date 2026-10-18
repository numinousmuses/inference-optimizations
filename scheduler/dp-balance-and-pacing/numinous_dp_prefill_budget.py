"""Per-pass prefill token budget balanced across DP-attention ranks (k3amd for amdpro, 2026-10-03).

Measured motive (DeepSeek V4 Pro, 8x MI355X, DP8 attention + TP-MoE, amdpro's v4 base):
  - decode window: 48% of a decoding rank's kernel time is NCCL, and 95% of that is waiting for the slowest rank;
    the waits sit at the MoE layers of a prefill forward running on ONE rank (rank 1 late in 121 of 199 calls);
  - balanced prefill load: 41% of RCCL time is skew wait.
Under DP attention every rank runs every forward in lock-step, so a forward lasts as long as its LARGEST per-rank
prefill: a rank with a 16K-token chunk holds the other seven (with 1K or 0) for the whole forward.

Mechanism (no extra collective): the PrefillDelayer already all-gathers five ints per rank every scheduler pass
(prefillable, watermark, running batch, max prefill bs, queue length). The hooks append a sixth, this rank's prefill
backlog (tokens still to prefill: the chunked request's rest plus every queued request's uncached tokens, using the
prefix lengths calc_priority matched on the previous pass), and turn the gathered backlogs into a per-pass token
target T: when two or more ranks have backlog, T = max(min_tokens, mean of the active backlogs capped at the chunk
size), rounded up to the page size; with one or no active rank there is no cap. The next pass caps this rank's
chunked_prefill_size and max_prefill_tokens at T, so a long prefill is split across passes and the forward length
follows the typical rank instead of the outlier. Splitting only moves chunk boundaries: the computed KV and logits are
the chunked-prefill result at a different chunk size (the gate's accuracy stage checks it).

Env: NUMINOUS_DP_PREFILL_BUDGET=1 (inert otherwise; the gather keeps its stock 5-wide shape),
NUMINOUS_DP_PREFILL_BUDGET_MIN (default 2048), NUMINOUS_DP_PREFILL_BUDGET_MODE = mean | median (default mean).
Requires --enable-prefill-delayer (the gather it rides on); without it the target stays 0 (no cap).
"""
from __future__ import annotations

import os

LOCAL_BACKLOG = 0
TARGET = 0
STATS = {"passes": 0, "capped": 0}


def enabled() -> bool:
    return os.environ.get("NUMINOUS_DP_PREFILL_BUDGET") == "1"


def _len(x) -> int:
    # prefix_indices is a torch tensor: `x or ()` raises on an empty tensor (measured on mi355x8, v7 boot), so no
    # truthiness tests on these fields
    return 0 if x is None else len(x)


def _uncached(req) -> int:
    total = _len(getattr(req, "origin_input_ids", None)) + _len(getattr(req, "output_ids", None))
    cached = _len(getattr(req, "prefix_indices", None)) + int(getattr(req, "host_hit_length", 0) or 0)
    return max(0, total - cached)


def local_backlog(scheduler) -> int:
    n = 0
    cr = getattr(scheduler, "chunked_req", None)
    if cr is not None:
        n += _uncached(cr)
    for req in getattr(scheduler, "waiting_queue", ()) or ():
        n += _uncached(req)
    return int(n)


def compute_target(backlogs, chunk: int, min_tokens: int = 2048, mode: str = "mean", page: int = 256) -> int:
    """0 means no cap."""
    if not chunk or chunk <= 0:
        return 0
    b = [min(int(x), chunk) for x in backlogs if int(x) > 0]
    if len(b) <= 1:
        return 0
    if mode == "median":
        t = sorted(b)[len(b) // 2]
    else:
        t = sum(b) / len(b)
    t = max(int(min_tokens), int(-(-t // 1)))
    if page and page > 1:
        t = -(-t // page) * page
    if t >= max(b):
        return 0
    return int(t)


def on_gather(backlogs, chunk: int, page: int = 256) -> None:
    global TARGET
    TARGET = compute_target(backlogs, chunk, int(os.environ.get("NUMINOUS_DP_PREFILL_BUDGET_MIN", "2048")),
                            os.environ.get("NUMINOUS_DP_PREFILL_BUDGET_MODE", "mean"), page)


def _log_every() -> int:
    return int(os.environ.get("NUMINOUS_DP_PREFILL_BUDGET_LOG_EVERY", "500"))


def apply(chunked_prefill_size, max_prefill_tokens):
    """Cap this pass's chunk and prefill-token budget at TARGET (no-op when off or uncapped)."""
    if not enabled():
        return chunked_prefill_size, max_prefill_tokens
    STATS["passes"] += 1
    n = _log_every()
    if n > 0 and STATS["passes"] % n == 0:
        import logging
        logging.getLogger(__name__).info(
            f"[numinous] dp prefill budget: {STATS['capped']} of {STATS['passes']} prefill passes capped; "
            f"target now {TARGET}, local backlog {LOCAL_BACKLOG}")
    if TARGET <= 0 or chunked_prefill_size is None:
        return chunked_prefill_size, max_prefill_tokens
    if TARGET < chunked_prefill_size:
        STATS["capped"] += 1
        return TARGET, min(max_prefill_tokens, TARGET) if max_prefill_tokens else TARGET
    return chunked_prefill_size, max_prefill_tokens


def after_pass(scheduler) -> None:
    global LOCAL_BACKLOG
    if enabled():
        LOCAL_BACKLOG = local_backlog(scheduler)
