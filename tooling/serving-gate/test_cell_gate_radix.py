"""cell_gate per-turn prefix reuse (hit_by_turn, warm_low_hit_requests) and the per-rung radix accounting delta
(operations/kernels/q38_radix_stats.py lines in the server log), plus the per-arm _ladder override."""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("GATE_LOCAL", "1")


def _cg():
    import cell_gate
    return importlib.reload(cell_gate)


def test_hit_by_turn_from_rung(monkeypatch):
    cg = _cg()
    t = [0.0]

    async def fake_request(http, url, engine, ids, rank, sampling):
        t[0] += 1.0
        prompt = len(ids)
        cached = 0 if ids[0] == 0 else (prompt - 100 if ids[0] == 1 else 0)   # turn 1 cold, turn 2 hit, turn 3 miss
        return {"prompt": prompt, "cached": cached, "out": 10, "ttft": 0.1, "wall": 1.0, "t_start": t[0],
                "t_end": t[0] + 1.0, "decode_tok_s": 10.0}

    monkeypatch.setattr(cg, "_request", fake_request)
    sessions = [[[0] * 1000, [1] * 1200, [2] * 1400] for _ in range(3)]
    row = cg._rung("http://x", "sglang", 1, sessions, {})
    assert row["hit_by_turn"] == [0.0, round(1100 / 1200, 4), 0.0]
    assert row["warm_requests"] == 6 and row["warm_low_hit_requests"] == 3


def test_radix_delta_per_round():
    cg = _cg()
    snap = lambda **k: "INFO numinous radix stats " + json.dumps(k)  # noqa: E731
    head = "\n".join([snap(requests=5, prompt_tokens=100, usable_tokens=50, flushes=1),
                      snap(requests=8, prompt_tokens=400, usable_tokens=60, flushes=2, full_available_now=9)])
    seg = "\n".join(["noise", snap(requests=10, prompt_tokens=500, usable_tokens=100, mamba_gap_tokens=0,
                                   kv_absent_tokens=0, flushes=2, full_available_now=3),
                     snap(requests=18, prompt_tokens=1400, usable_tokens=960, mamba_gap_tokens=40,
                          kv_absent_tokens=0, flushes=2, full_available_now=1)])
    d = cg._radix_delta(head, seg)
    assert d["requests"] == 10 and d["prompt_tokens"] == 1000 and d["usable_tokens"] == 900
    assert d["full_available_now"] == 1 and d["usable_share"] == 0.9 and d["mamba_gap_share"] == 0.04
    assert cg._radix_delta(head, "no stats here") is None
    assert cg._radix_delta("", seg)["requests"] == 18


def test_arm_ladder_override_in_source():
    src = open(os.path.join(os.path.dirname(__file__), "cell_gate.py")).read()
    assert 'ladder = list(delta.get("_ladder") or ladder)' in src


def test_flush_retries_until_success(monkeypatch):
    cg = _cg()
    calls = []
    monkeypatch.setattr(cg.time, "sleep", lambda s: None)
    monkeypatch.setattr(cg, "_post", lambda url: calls.append(url) or len(calls) >= 3)
    assert cg._flush("http://x/flush_cache") is True and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(cg, "_post", lambda url: calls.append(url) and False)
    assert cg._flush("http://x/flush_cache", tries=4) is False and len(calls) == 4
