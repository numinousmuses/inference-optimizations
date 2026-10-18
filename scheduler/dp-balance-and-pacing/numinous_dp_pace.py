"""Pure decision rule for DP-attention prefill pacing on a monolithic spec-decoding server (amdpro 2026-10-05, GLM arm
M decode-idle; installed as sglang/srt/managers/numinous_dp_pace.py by operations/patches/m_dp_pace_hooks.py).

Every DP rank calls decide() once per scheduler pass with the SAME gathered rows (the prefill delayer's all-gather
plus one 3-int all-gather), and the SAME streak, so every rank takes the same action.

Inputs per rank i:
  prefillable[i]  the delayer's gathered flag (a mid-chunk rank reports True in v0.5.20, schedule_policy.py:1154-1163)
  cont[i]         the rank holds a chunked-prefill continuation this pass (scheduler.chunked_req is not None)
  running[i]      requests in the rank's running (decode) batch
  waiting[i]      requests in the rank's waiting queue
Actions:
  "park"     no rank prefills this pass. The continuation is not added (the v0.5.20 SWA park path,
             schedule_policy.py:1149-1151) and new work is held, so every rank runs a decode step instead of an IDLE
             batch. Taken only when:
               - only continuations would prefill (no rank has new work);
               - at most k_max ranks would prefill;
               - some other rank holds running decodes;
               - every waiting queue is <= q_max;
               - fewer than `steps` consecutive parks so far (the continuation runs at least every steps+1 passes).
  "release"  a continuation runs this pass anyway, so queued new work on other ranks joins it instead of being held
             (peer release). The forward happens either way, and joining costs only per-token time.
  "stock"    leave the delayer's verdict alone.
"""
from __future__ import annotations

import os


def params(env=None) -> dict:
    e = os.environ if env is None else env
    return {"steps": int(e.get("NUMINOUS_DP_PACE_STEPS", "2")), "k_max": int(e.get("NUMINOUS_DP_PACE_K", "2")),
            "q_max": int(e.get("NUMINOUS_DP_PACE_QMAX", "0")),
            "peer": e.get("NUMINOUS_DP_PEER_RELEASE", "1") == "1"}


def decide(prefillable, cont, running, waiting, streak: int, steps: int = 2, k_max: int = 2, q_max: int = 0,
           peer: bool = True) -> tuple[str, int]:
    """(action, next streak). Pure."""
    pre = [bool(x) for x in prefillable]
    con = [bool(x) for x in cont]
    run = [int(x) for x in running]
    wq = [int(x) for x in waiting]
    k = sum(1 for p, c in zip(pre, con) if p or c)
    any_cont = any(con)
    new_work = any(p and not c for p, c in zip(pre, con))
    decode_idle = any(r > 0 and not (p or c) for r, p, c in zip(run, pre, con))
    eligible = (steps > 0 and any_cont and not new_work and 1 <= k <= k_max and decode_idle
                and max(wq, default=0) <= q_max)
    if eligible and streak < steps:
        return "park", streak + 1
    if any_cont and peer:
        return "release", 0
    return "stock", 0
