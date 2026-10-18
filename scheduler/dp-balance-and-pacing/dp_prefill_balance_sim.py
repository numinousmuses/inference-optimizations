"""Replay simulator: DP-attention lockstep serving of the OR session file under different DP dispatch policies
(agent rtxv41, 2026-10-01; GLM-5.3-Flash on 4 x RTX, DP4 + EP4).

Model (calibrated to rtxsys's measured 4 x RTX numbers, agents/status/rtxsys.md 03:25):
- Ranks step in lockstep: a step lasts max over ranks of the rank's own forward. A rank either extends (prefill up
  to CHUNK uncached tokens of its waiting queue, FIFO, several requests per chunk) or decodes all its running
  requests. That is v0.5.20 without mixed chunk.
  - Prefill: 28.2 us per token per rank (measured, 8.85K tok/s/rank).
  - Decode: 27.5 + 3.0 ms x running (measured 29 / 42 / 50 ms at 1 / 4 / 8 running), x EAGER_PENALTY on steps where
    any rank extends (an extend anywhere disables the decode CUDA graphs).
  - MTP accept: ACCEPT tokens per decode step per request.
- Sessions are closed-loop (the gate's shape). N sessions all start at t=0; turn k sends prompt turn_tokens[k] and
  decodes OUT tokens (ignore_eos 659, the OR mean).
- Cache: the next turn's prompt is the recorded conversation, so the cached prefix on a rank is the session's
  previous prompt (not the generated output), kept while the session's prefix is resident (per-rank LRU over
  KV_CAP tokens).
- Delayer (optional): a rank with waiting prefill defers while fewer than DELAY_MIN ranks want to prefill, for at
  most DELAY_PASSES passes. This is upstream's --enable-prefill-delayer, simplified.

Policies: pin (session i -> rank i % dp, what the gate and gateway do), rr (per request, no stickiness),
total_tokens (upstream: KV in use + queued prompt lengths), balanced (operations/kernels/dp_prefill_balance.py:
waiting uncached tokens + this prompt's uncached tokens on the rank, sticky through the prefix cost).

    python3 operations/kernels/dp_prefill_balance_sim.py --sessions operations/modal/data/real_sessions_flash_or30k_t8_ctx262k.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PREFILL_US = 28.2
DEC_BASE_MS, DEC_PER_MS = 27.5, 3.0
EAGER_PENALTY = 1.15
ACCEPT = 2.7
CHUNK = 8192
MAX_RUN = 16
KV_CAP = 1_400_000
OUT = 659


@dataclass
class Req:
    sess: int
    turn: int
    prompt: int
    uncached: int
    arrive: float
    rank: int = -1
    first_tok: float = -1.0
    out_left: float = OUT
    done: float = -1.0


@dataclass
class Rank:
    waiting: collections.deque = field(default_factory=collections.deque)
    running: list = field(default_factory=list)
    cache: "collections.OrderedDict[int, int]" = field(default_factory=collections.OrderedDict)  # sess -> tokens
    cache_tokens: int = 0
    delayed: int = 0
    busy_s: float = 0.0
    prefill_tokens: int = 0

    def cached_for(self, sess: int) -> int:
        return self.cache.get(sess, 0)

    def put(self, sess: int, tokens: int):
        old = self.cache.pop(sess, 0)
        self.cache_tokens += tokens - old
        self.cache[sess] = tokens
        while self.cache_tokens > KV_CAP and len(self.cache) > 1:
            _, t = self.cache.popitem(last=False)
            self.cache_tokens -= t

    def waiting_uncached(self) -> int:
        return sum(r.uncached for r in self.waiting)

    def total_tokens(self) -> int:  # upstream total_tokens: KV in use + queued prompt lengths
        return self.cache_tokens + sum(r.prompt for r in self.waiting)


class Router:
    def __init__(self, policy: str, dp: int, stick_bias: float = 1.0, req_tokens: float = 0.0):
        self.policy, self.dp, self.rr = policy, dp, 0
        self.stick_bias = stick_bias
        self.spill_tokens = 131072
        # tokens-equivalent of one more active request on a rank (its decode rows lengthen every lockstep step):
        # 3.0 ms per row per step x (OUT / ACCEPT) steps / 28.2 us per prefill token = ~26K (derived)
        self.req_tokens = req_tokens if req_tokens else DEC_PER_MS * 1e-3 * (OUT / ACCEPT) / (PREFILL_US * 1e-6)

    def route(self, req: Req, ranks: list, prev_prompt: int) -> int:
        dp = self.dp
        if self.policy == "pin":
            return req.sess % dp
        if self.policy == "pin_hash":  # a gateway hashing session ids: uniform random, fixed per session
            return hash(("sess", req.sess)) % dp
        if self.policy == "affinity_nospill":
            self.spill_tokens = 1 << 60
            return Router.route(type("R", (), {"policy": "affinity", "dp": dp, "spill_tokens": 1 << 60,
                                               "rr": 0, "stick_bias": 1.0, "req_tokens": 0})(), req, ranks, prev_prompt)
        if self.policy == "rr":
            r = self.rr % dp
            self.rr += 1
            return r
        if self.policy == "total_tokens":
            return min(range(dp), key=lambda i: (ranks[i].total_tokens(), len(ranks[i].waiting) + len(ranks[i].running), i))
        if self.policy == "affinity":
            # PrefillBalancer mode "affinity" on the simulator's true state
            hold = [min(ranks[i].cached_for(req.sess), prev_prompt) for i in range(dp)]
            pend = [ranks[i].waiting_uncached() for i in range(dp)]
            act = [len(ranks[i].waiting) + len(ranks[i].running) for i in range(dp)]
            h = max(range(dp), key=lambda i: hold[i])
            if req.prompt > 0 and hold[h] > 0 and (hold[h] >= 2048 or hold[h] >= 0.5 * req.prompt):
                if pend[h] - min(pend) <= self.spill_tokens:
                    return h
                return min(range(dp), key=lambda i: (pend[i] + req.prompt - hold[i], act[i], i))
            return min(range(dp), key=lambda i: (act[i], pend[i] + req.prompt, i))
        if self.policy in ("balanced", "balanced_req"):
            # same cost as PrefillBalancer.choose, with the true per-rank cached prefix (the controller's prefix
            # index tracks exactly this; LRU eviction is the only divergence)
            best = None
            hold = [min(ranks[i].cached_for(req.sess), prev_prompt) for i in range(dp)]
            for i in range(dp):
                unc = req.prompt - hold[i]
                cost = ranks[i].waiting_uncached() + unc + (self.stick_bias - 1.0) * (max(hold) - hold[i])
                if self.policy == "balanced_req":
                    cost += self.req_tokens * (len(ranks[i].waiting) + len(ranks[i].running))
                key = (cost, len(ranks[i].waiting) + len(ranks[i].running), i)
                if best is None or key < best[0]:
                    best = (key, i)
            return best[1]
        raise ValueError(self.policy)


def simulate(sessions: list, n: int, policy: str, dp: int = 4, delayer: bool = False, delay_min: int = 0,
             delay_passes: int = 30, extend_cap: int = 0, stick_bias: float = 1.0, max_steps: int = 2_000_000,
             arrival_rate: float = 0.0, seed: int = 0) -> dict:
    """arrival_rate 0: the gate's shape (n sessions start at t=0, closed loop). > 0: production-like open loop,
    n sessions arrive as a Poisson process (sessions/s), each then runs its turns back to back; pin = arrival
    index % dp (what a session-id hash gives)."""
    import random
    rng = random.Random(seed)
    sess = sessions[:n]
    ranks = [Rank() for _ in range(dp)]
    router = Router(policy, dp, stick_bias)
    t = 0.0
    done_reqs: list = []
    turn_of = [0] * n
    prev_prompt = [0] * n
    pending_arrivals: list = [(0.0, i) for i in range(n)]
    delay_min = delay_min or dp

    def submit(i, now):
        k = turn_of[i]
        p = sess[i]["turn_tokens"][k]
        req = Req(sess=i, turn=k, prompt=p, uncached=p, arrive=now)
        r = router.route(req, ranks, prev_prompt[i])
        hold = min(ranks[r].cached_for(i), prev_prompt[i]) if k > 0 else 0
        req.uncached = p - hold
        req.rank = r
        ranks[r].waiting.append(req)

    arrivals = collections.deque()
    if arrival_rate > 0:
        ta = 0.0
        for i in range(n):
            arrivals.append((ta, i))
            ta += rng.expovariate(arrival_rate)
    else:
        for _, i in pending_arrivals:
            submit(i, 0.0)
    steps = 0
    out_tokens = 0
    stall_s = 0.0
    while arrivals or any(r.waiting or r.running for r in ranks):
        while arrivals and arrivals[0][0] <= t:
            submit(arrivals.popleft()[1], t)
        if arrivals and not any(r.waiting or r.running for r in ranks):
            t = arrivals[0][0]
            continue
        steps += 1
        if steps > max_steps:
            break
        want = [bool(r.waiting) and len(r.running) < MAX_RUN for r in ranks]
        n_want = sum(want)
        extend = []
        for k, r in enumerate(ranks):
            ext = want[k]
            if ext and delayer and n_want < delay_min and r.delayed < delay_passes:
                r.delayed += 1
                ext = False
            extend.append(ext)
        any_ext = any(extend)
        n_ext = sum(extend)
        times = []
        plans = []
        for k, r in enumerate(ranks):
            if extend[k]:
                r.delayed = 0
                budget = CHUNK
                if extend_cap and n_ext <= 2 and n_ext < dp and any(ranks[j].running for j in range(dp) if not extend[j]):
                    budget = min(budget, extend_cap)
                take, used = [], 0
                for q in r.waiting:
                    if used >= budget or len(r.running) + len(take) >= MAX_RUN:
                        break
                    c = min(q.uncached, budget - used)
                    take.append((q, c))
                    used += c
                plans.append(("ext", take))
                times.append(used * PREFILL_US * 1e-6)
            elif r.running:
                plans.append(("dec", None))
                ms = (DEC_BASE_MS + DEC_PER_MS * len(r.running)) * (EAGER_PENALTY if any_ext else 1.0)
                times.append(ms * 1e-3)
            else:
                plans.append(("idle", None))
                times.append(0.0)
        dt = max(times) if times else 0.0
        if dt <= 0:
            dt = 1e-4
        t += dt
        for k, r in enumerate(ranks):
            r.busy_s += times[k]
            stall_s += dt - times[k]
            kind, take = plans[k]
            if kind == "ext":
                for q, c in take:
                    q.uncached -= c
                    r.prefill_tokens += c
                    if q.uncached <= 0:
                        r.waiting.remove(q)
                        q.first_tok = t
                        q.out_left = OUT - 1
                        r.running.append(q)
                        r.put(q.sess, q.prompt)
            elif kind == "dec":
                still = []
                for q in r.running:
                    adv = min(ACCEPT, q.out_left)
                    q.out_left -= adv
                    out_tokens += adv
                    if q.out_left <= 1e-9:
                        q.done = t
                        done_reqs.append(q)
                        i = q.sess
                        prev_prompt[i] = q.prompt
                        turn_of[i] += 1
                        if turn_of[i] < len(sess[i]["turn_tokens"]):
                            submit(i, t)
                    else:
                        still.append(q)
                r.running = still
    out_tokens += len(done_reqs)  # the first token of each turn comes from the prefill
    ttft = [q.first_tok - q.arrive for q in done_reqs]
    per_user = [(OUT - 1) / max(1e-9, q.done - q.first_tok) for q in done_reqs]
    busy = [r.busy_s for r in ranks]
    return {"policy": policy, "n": n, "arrival_rate": arrival_rate, "delayer": delayer, "extend_cap": extend_cap, "makespan_s": round(t, 1),
            "agg_tok_s": round(out_tokens / t, 1), "per_user_p50": round(statistics.median(per_user), 1),
            "ttft_p50_s": round(statistics.median(ttft), 2), "ttft_p90_s": round(sorted(ttft)[int(0.9 * len(ttft))], 2),
            "stall_frac": round(stall_s / (t * dp), 3), "busy_imbalance": round(max(busy) / (sum(busy) / dp), 3),
            "prefill_tokens": sum(r.prefill_tokens for r in ranks), "turns": len(done_reqs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=os.path.join(os.path.dirname(__file__), "..", "modal", "data",
                                                       "real_sessions_flash_or30k_t8_ctx262k.json"))
    ap.add_argument("--ladder", default="16,24,32")
    ap.add_argument("--policies", default="pin,rr,total_tokens,balanced,affinity")
    ap.add_argument("--delayer", default="0,1")
    ap.add_argument("--extend-cap", default="0")
    ap.add_argument("--offset", type=int, default=0, help="first session index (robustness: other session draws)")
    ap.add_argument("--arrival-rate", type=float, default=0.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    sessions = json.load(open(a.sessions))[a.offset:]
    rows = []
    for n in [int(x) for x in a.ladder.split(",")]:
        for d in [x == "1" for x in a.delayer.split(",")]:
            for cap in [int(x) for x in a.extend_cap.split(",")]:
                for p in a.policies.split(","):
                    for sd in [int(x) for x in a.seeds.split(",")]:
                        r = simulate(sessions, n, p, delayer=d, extend_cap=cap, arrival_rate=a.arrival_rate, seed=sd)
                        r["seed"] = sd
                        rows.append(r)
                        print(json.dumps(r), flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
