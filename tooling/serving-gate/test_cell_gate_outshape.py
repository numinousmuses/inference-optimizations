"""GATE_MAX_OUT / GATE_IGNORE_EOS: OR-shaped output length (rtxv41: the 46K ladder averaged ~240 out tokens per turn,
r = 15-17, ~4x the OpenRouter mix's prefill weight). Defaults must stay 400 / off so earlier gates stay comparable."""
import asyncio
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("GATE_LOCAL", "1")


def _load(monkeypatch, **env):
    for k in ("GATE_MAX_OUT", "GATE_IGNORE_EOS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import cell_gate
    return importlib.reload(cell_gate)


class _Stop(Exception):
    pass


class _Http:
    def __init__(self):
        self.body = None

    def post(self, url, json=None, headers=None):
        self.body = json
        raise _Stop


def _body(cg, engine):
    http = _Http()
    try:
        asyncio.run(cg._request_once(http, "http://x", engine, [1, 2, 3], None, {"temperature": 1.0}))
    except _Stop:
        pass
    return http.body


def test_defaults_unchanged(monkeypatch):
    cg = _load(monkeypatch)
    assert cg.MAX_OUT == 400 and cg.IGNORE_EOS is False
    b = _body(cg, "sglang")
    assert b["sampling_params"]["max_new_tokens"] == 400 and "ignore_eos" not in b["sampling_params"]


def test_or_shape(monkeypatch):
    cg = _load(monkeypatch, GATE_MAX_OUT="988", GATE_IGNORE_EOS="1")
    b = _body(cg, "sglang")
    assert b["sampling_params"]["max_new_tokens"] == 988 and b["sampling_params"]["ignore_eos"] is True
    v = _body(cg, "vllm")
    assert v["max_tokens"] == 988 and v["ignore_eos"] is True
    _load(monkeypatch)  # restore module defaults for other tests


def test_no_pin_env(monkeypatch):
    """GATE_NO_PIN=1: _rung's requests carry no routed_dp_rank (the engine's DP controller places them)."""
    cg = _load(monkeypatch)
    seen = []

    async def fake_request(http, url, engine, ids, rank, sampling):
        seen.append(rank)
        return {"prompt": 1, "cached": 0, "out": 1, "ttft": 0.1, "wall": 0.2, "t_start": 0.0, "t_end": 0.2,
                "decode_tok_s": None}

    monkeypatch.setattr(cg, "_request", fake_request)
    for env, want in (("0", [0, 1, 2, 3]), ("1", [None] * 4)):
        monkeypatch.setenv("GATE_NO_PIN", env)
        seen.clear()
        try:
            cg._rung("http://x", "sglang", 4, [[[1]] for _ in range(4)], {})
        except Exception:  # noqa: BLE001 - only the routing of the requests matters here
            pass
        assert sorted(seen, key=lambda x: -1 if x is None else x) == want
    monkeypatch.delenv("GATE_NO_PIN")
