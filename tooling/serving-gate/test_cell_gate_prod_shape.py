"""cell_gate production-shape row (info only; amdglm findings 9g): the per-session mix from the recorded turns, the
exponential tilt that meets the OR mix's uncached tokens per output token, and the closed-loop runner (session
replacement, staggered starts, window accounting) against a fake server."""
import asyncio
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("GATE_LOCAL", "1")


def _cg():
    import cell_gate
    return importlib.reload(cell_gate)


def session(turn_chars, answer_chars):
    """messages: system + (user, assistant) per turn; turn t's prompt ends before its assistant message."""
    msgs, ends = [{"role": "system", "content": "s" * 100}], []
    for u, a in zip(turn_chars, answer_chars):
        msgs.append({"role": "user", "content": "u" * u})
        ends.append(len(msgs))
        msgs.append({"role": "assistant", "content": "a" * a})
    return {"messages": msgs, "turn_ends": ends}


def test_session_mix_counts_turns_like_the_replay():
    cg = _cg()
    s = session([1000, 50, 50], [200, 200, 200])
    u, c, o = cg.session_mix(s)
    # prompts: t1 = 100 + 1000; t2 = t1 + 200 + 50; t3 = t2 + 200 + 50
    p1, p2, p3 = 1100, 1350, 1600
    assert (u, c, o) == (p1 + (p2 - p1) + (p3 - p2), p1 + p2, 600)


def test_mix_weights_meet_the_target():
    cg = _cg()
    stats = [cg.session_mix(session([5000 + 3000 * i, 100, 100, 100], [400, 400, 400, 400])) for i in range(30)]
    pool = sum(s[0] for s in stats) / sum(s[2] for s in stats)
    target = 0.6 * pool
    w, mix = cg.mix_weights(stats, target)
    assert abs(sum(w) - 1) < 1e-9 and abs(mix["uncached_per_out"] - target) / target < 0.01
    assert mix["pool_uncached_per_out"] == round(pool, 3)
    assert w[0] > w[-1]                      # the light sessions are favoured when the target is below the pool


def test_closed_loop_replaces_sessions_and_counts_the_window(monkeypatch):
    cg = _cg()

    async def fake_request(http, url, engine, ids, rank, sampling):
        t0 = time.perf_counter()
        await asyncio.sleep(0.01)
        return {"prompt": len(ids), "cached": 0 if ids[0] == 0 else len(ids) - 10, "out": 5, "ttft": 0.002,
                "wall": 0.01, "t_start": t0, "t_end": time.perf_counter(), "decode_tok_s": 500.0}
    pool = [[[0] * 100, [1] * 120, [1] * 140, [1] * 160] for _ in range(10)]
    row = cg._prod_rung("http://x", "sglang", 2, lambda i: pool[i], [0.1] * 10, n=4, warmup_s=0.1, window_s=0.4,
                        sampling={}, stagger_s=0.02, request=fake_request)
    assert row["prod_sessions"] == 4 and row["prod_requests"] > 0
    assert row["prod_sessions_started"] > 4                          # finished sessions were replaced
    assert 0.15 < row["prod_cold_share"] < 0.4                       # ~1 cold turn in 4
    assert abs(row["prod_out_tok_s"] - row["prod_requests"] * 5 / 0.4) < 1e-6
