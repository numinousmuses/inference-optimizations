"""Prefill-token-balanced, cache-aware DP dispatch (agent rtxv41, 2026-10-01; GLM-5.3-Flash on RTX, SGLang v0.5.20).

Why: under DP attention every forward step is a lockstep across the DP ranks (the MoE / EP collectives), so a step
lasts as long as the slowest rank's prefill. Round-robin (or the gateway's session pin i % dp) ignores how many
uncached prompt tokens each rank still has to prefill: one 125K-token cold first turn on a rank makes the other
ranks wait on it, step after step. Upstream's `total_tokens` method balances KV tokens in use plus queued prompt
lengths, which is neither the prefill work (cached prefixes count as work) nor sticky (a session's next turn can
land on a rank without its prefix and re-prefill 40-60K tokens).

Policy (per new request, in the DP controller):
  cost(r) = pending_uncached(r) + uncached_if_routed_to(r)
  rank    = argmin_r cost(r) (ties: fewer requests, then lower rank)
  - pending_uncached(r): the scheduler's own `num_waiting_uncached_tokens` from its load snapshot (waiting queue +
    the chunked request's remainder; source_code v0.5.20 load_inquirer.get_num_waiting_uncached_tokens), plus the
    estimated uncached tokens of every request dispatched to r AFTER that snapshot's timestamp (the snapshot has not
    seen them yet; without this a burst all lands on the rank whose stale snapshot looks emptiest).
  - uncached_if_routed_to(r) = len(prompt) - matched_prefix(r), where matched_prefix comes from a controller-side
    prefix index: a chained hash per BLOCK tokens of every prompt routed to r (LRU-bounded). A returning session's
    rank matches its whole previous prompt, so it stays put unless another rank is idle by more than the prefix it
    would have to recompute: stickiness falls out of the same cost, no separate threshold.
  - `stick_bias` (>= 1) scales the recompute term for ranks without the prefix, to prefer stickiness more strongly
    (KV capacity, decode locality); 1.0 = pure prefill-work balance.

Everything here is pure Python and CPU-tested (test_dp_prefill_balance.py); the engine hook is
operations/patches/dp_prefill_balance_hooks.py (load_balance_method total_tokens + NUMINOUS_DP_PREFILL_BALANCE=1).
"""
from __future__ import annotations

import collections
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

BLOCK = int(os.environ.get("NUMINOUS_DP_BALANCE_BLOCK", "256"))
MAX_ENTRIES = int(os.environ.get("NUMINOUS_DP_BALANCE_MAX_BLOCKS", "4000000"))


def block_hashes(tokens: Sequence[int], block: int = BLOCK) -> list[int]:
    """Chained hash of each full block: h_i covers tokens[0 : (i + 1) * block] (a prefix identity, like the radix)."""
    out, h = [], 0
    n = len(tokens) // block
    for i in range(n):
        h = hash((h, tuple(tokens[i * block:(i + 1) * block])))
        out.append(h)
    return out


class PrefixIndex:
    """Which ranks hold which prompt prefix (controller-side estimate of the per-rank radix caches), LRU-bounded.

    Each block hash maps to [rank bitmask, deepest block index any prompt through it reached].
    - Bitmask: a prefix shared by many sessions (agent system prompt + tool definitions) ends up on every rank. The
      first version mapped a block to its LAST inserter, so returning sessions "stuck" to whichever rank last saw
      the shared system prompt. Measured as a prefix-cache hit drop from 0.81 to 0.65-0.70 in composed run g156o.
    - Deepest index: tells a returning session from a new one that only shares a common prefix. If the match stops
      at a block from which another prompt continued for a long way, the matched part is shared. If the other path
      ended within TAIL blocks (this session's own previous turn, whose last block differs because the next turn
      re-renders its tail), the session is returning."""

    TAIL = 2

    def __init__(self, max_entries: int = MAX_ENTRIES, block: int = BLOCK):
        self.block = block
        self.max_entries = max_entries
        self._map: "collections.OrderedDict[int, list]" = collections.OrderedDict()

    def walk(self, hashes: Sequence[int], dp: int):
        """(matched tokens per rank, index of the last block matched on any rank, its deepest index)."""
        matched = [0] * dp
        alive = (1 << dp) - 1
        last, deepest = -1, -1
        for i, h in enumerate(hashes):
            e = self._map.get(h)
            alive &= e[0] if e else 0
            if not alive:
                break
            last, deepest = i, e[1]
            for r in range(dp):
                if alive >> r & 1:
                    matched[r] = (i + 1) * self.block
        return matched, last, deepest

    def match(self, hashes: Sequence[int], dp: int) -> list[int]:
        return self.walk(hashes, dp)[0]

    def insert(self, hashes: Sequence[int], rank: int) -> None:
        m = self._map
        bit = 1 << rank
        end = len(hashes) - 1
        for h in hashes:
            e = m.pop(h, None)
            m[h] = [bit, end] if e is None else [e[0] | bit, max(e[1], end)]
        while len(m) > self.max_entries:
            m.popitem(last=False)

    def __len__(self) -> int:
        return len(self._map)


@dataclass
class _Dispatched:
    t: float
    rank: int
    tokens: int


@dataclass
class PrefillBalancer:
    """mode "affinity" (default, the measured-better policy in dp_prefill_balance_sim.py): a request whose prompt is
    mostly cached on some rank (a returning session) stays there unless that rank's waiting uncached tokens exceed
    the least-loaded rank's by more than `spill_tokens`; a new session goes to the rank with the fewest active
    requests (decode rows set the lockstep step), ties broken by waiting uncached tokens + its own prompt.
    mode "tokens": the pure prefill-work cost argmin (kept for comparison; it loses to the sticky pin in replay)."""

    dp: int
    stick_bias: float = float(os.environ.get("NUMINOUS_DP_BALANCE_STICK_BIAS", "1.0"))
    mode: str = os.environ.get("NUMINOUS_DP_BALANCE_MODE", "affinity")
    # default: never move a returning session (spill disabled). The composed run g156o (2026-10-01) measured prefix-
    # cache hit 0.65-0.69 with dpbal vs 0.81-0.82 pinned; moving a session costs its whole prefix.
    spill_tokens: int = int(os.environ.get("NUMINOUS_DP_BALANCE_SPILL_TOKENS", str(1 << 62)))
    sticky_frac: float = 0.5
    sticky_min_tokens: int = int(os.environ.get("NUMINOUS_DP_BALANCE_STICKY_MIN", "2048"))
    index: PrefixIndex = field(default_factory=PrefixIndex)
    snap_tokens: list = field(default_factory=list)  # num_waiting_uncached_tokens per rank (last snapshot)
    snap_reqs: list = field(default_factory=list)
    snap_time: list = field(default_factory=list)
    recent: "collections.deque[_Dispatched]" = field(default_factory=collections.deque)
    active: Optional[list] = None  # allowed ranks (None = all)
    stats: dict = field(default_factory=dict)  # decision counters (logged every 200 dispatches)

    def __post_init__(self):
        self.snap_tokens = [0] * self.dp
        self.snap_reqs = [0] * self.dp
        self.snap_time = [0.0] * self.dp

    # ------------------------------------------------------------------ load snapshots
    def update(self, loads) -> None:
        """loads: iterable of objects with dp_rank, timestamp, num_waiting_uncached_tokens, num_running_reqs,
        num_waiting_reqs (sglang LoadSnapshot)."""
        for ld in loads:
            r = int(ld.dp_rank)
            if r >= self.dp or ld.timestamp <= self.snap_time[r]:
                continue
            self.snap_time[r] = float(ld.timestamp)
            self.snap_tokens[r] = int(ld.num_waiting_uncached_tokens)
            self.snap_reqs[r] = int(ld.num_running_reqs) + int(ld.num_waiting_reqs)
        # drop dispatch records every rank's snapshot has already seen
        oldest = min(self.snap_time) if self.snap_time else 0.0
        while self.recent and self.recent[0].t <= oldest:
            self.recent.popleft()

    def active_reqs(self) -> list[int]:
        a = list(self.snap_reqs)
        for d in self.recent:
            if d.t > self.snap_time[d.rank]:
                a[d.rank] += 1
        return a

    def pending(self) -> list[int]:
        p = list(self.snap_tokens)
        for d in self.recent:
            if d.t > self.snap_time[d.rank]:
                p[d.rank] += d.tokens
        return p

    # ------------------------------------------------------------------ dispatch
    def choose(self, tokens: Sequence[int], now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        hashes = block_hashes(tokens, self.index.block)
        matched, last, deepest = self.index.walk(hashes, self.dp)
        # returning: the match covers our whole prompt, or stops where only a short tail (our previous turn's) went on
        returning = last >= 0 and (last == len(hashes) - 1 or deepest <= last + self.index.TAIL)
        pend = self.pending()
        n = len(tokens)
        ranks = list(self.active) if self.active else list(range(self.dp))
        act = self.active_reqs()
        if self.mode == "affinity":
            best, kind = self._affinity(n, matched, pend, ranks, act, returning)
            return self._commit(best, n - matched[best], hashes, now, kind)
        best, best_key = None, None
        for r in ranks:
            unc = n - matched[r]
            recompute = max(matched) - matched[r]  # prefix tokens another rank already holds
            cost = pend[r] + unc + (self.stick_bias - 1.0) * recompute
            key = (cost, act[r], r)
            if best_key is None or key < best_key:
                best, best_key = r, key
        return self._commit(best, n - matched[best], hashes, now)

    def _affinity(self, n: int, matched: list, pend: list, ranks: list, act: list, returning: bool = True):
        # longest prefix holder; ties (a prefix on several ranks, e.g. only the shared system prompt) -> fewer active
        holder = max(ranks, key=lambda r: (matched[r], -act[r], -r))
        # a returning session: a real shared prefix (>= sticky_min_tokens, or most of the prompt). A fraction alone is
        # wrong for agent sessions whose next prompt can be > 2x the previous one (tool output), measured in replay.
        if returning and n > 0 and matched[holder] > 0 and (matched[holder] >= self.sticky_min_tokens
                                                            or matched[holder] >= self.sticky_frac * n):
            if pend[holder] - min(pend[r] for r in ranks) <= self.spill_tokens:
                return holder, "sticky"
            # overloaded holder: fall back to the work cost (moving pays the prefix recompute)
            return min(ranks, key=lambda r: (pend[r] + n - matched[r], act[r], r)), "spill"
        kind = "new" if max(matched) == 0 else ("shared_prefix" if not returning else "weak_match")
        return min(ranks, key=lambda r: (act[r], pend[r] + n, r)), kind

    def _commit(self, best: int, unc: int, hashes: list, now: float, kind: str = "") -> int:
        self.recent.append(_Dispatched(now, best, unc))
        self.index.insert(hashes, best)
        if kind:
            self.stats[kind] = self.stats.get(kind, 0) + 1
            self.stats["n"] = self.stats.get("n", 0) + 1
            if self.stats["n"] % 200 == 0:
                import logging

                logging.getLogger(__name__).info("numinous dp balance decisions: %s", dict(self.stats))
        return best
