"""Numinous: decode-burst admission rule behind --num-continuous-decode-steps.

Pure so it can be unit-tested without the scheduler's GPU dependencies.
"""


def hold_prefill(
    burst: int,
    streak: int,
    last_batch_was_extend: bool | None,
    running_nonempty: bool,
    running_prefill_only: bool,
    chunked_prefill_pending: bool = False,
) -> bool:
    """True when this scheduler pass should skip prefill admission and let decode take the step.

    burst <= 1 disables the rule. The hold applies only after a non-extend (decode) pass, while fewer than
    `burst` consecutive decode steps have run, and only when there are running decode requests; an idle
    scheduler admits prefill immediately so first-token latency on an idle node is unchanged. A request that is
    part-way through a chunked prefill is never held: its next chunk must be scheduled or it never finishes.

    Under DP attention every rank must make the same decision, so callers pass running_nonempty=True there and
    derive last_batch_was_extend and streak from the synced batch (see next_streak_from_mode).
    """
    if burst <= 1 or last_batch_was_extend is None or last_batch_was_extend:
        return False
    if chunked_prefill_pending:
        return False
    if not running_nonempty or running_prefill_only:
        return False
    return streak < burst


def next_streak(burst: int, streak: int, prefill_admitted: bool) -> int:
    if burst <= 1:
        return 0
    return 0 if prefill_admitted else streak + 1
