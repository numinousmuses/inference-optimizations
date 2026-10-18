"""CPU tests for m_dp_pace_hooks.py + numinous_dp_pace.py against the real SGLang v0.5.20 text and the hopper
decode-burst patched tree (amdpro 2026-10-05). The patched prefill_delayer module runs with its sglang imports stubbed
and the all-gathers replaced by per-rank tables, so the stock negotiation plus the hook decide together; the patched
add_chunked_req is executed in isolation on a fake adder."""
from __future__ import annotations

import ast
import glob
import importlib.util
import os
import pathlib
import shutil
import sys
import types

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m_dp_pace_hooks as h  # noqa: E402
import numinous_dp_pace as ndp  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", str(pathlib.Path.home() / ".cache/numinous/sgl-v0520/python")))
PKG = SRC / "sglang"
if not (PKG / h.DELAYER).exists():
    pytest.skip("v0.5.20 source tree not present", allow_module_level=True)


# ---------------------------------------------------------------- the pure rule
def test_decide_parks_a_lone_continuation_then_lets_it_run():
    rows = dict(prefillable=[0, 0, 1, 0], cont=[0, 0, 1, 0], running=[3, 2, 1, 4], waiting=[0, 0, 0, 0])
    a1, s1 = ndp.decide(**rows, streak=0, steps=2)
    a2, s2 = ndp.decide(**rows, streak=s1, steps=2)
    a3, s3 = ndp.decide(**rows, streak=s2, steps=2)
    assert (a1, a2, a3) == ("park", "park", "release") and (s1, s2, s3) == (1, 2, 0)


@pytest.mark.parametrize("change,expect", [
    (dict(prefillable=[1, 0, 1, 0]), "release"),            # rank 0 has new work: never park, let it join
    (dict(waiting=[0, 1, 0, 0]), "release"),                # a queued request anywhere: no park (q_max 0)
    (dict(running=[0, 0, 1, 0]), "release"),                # nobody else decodes: nothing to win
    (dict(prefillable=[1, 1, 1, 0], cont=[1, 1, 1, 0]), "release"),   # 3 continuations > k_max 2
    (dict(cont=[0, 0, 0, 0], prefillable=[0, 0, 1, 0]), "stock"),     # no continuation: stock delayer
])
def test_decide_eligibility(change, expect):
    rows = dict(prefillable=[0, 0, 1, 0], cont=[0, 0, 1, 0], running=[3, 2, 1, 4], waiting=[0, 0, 0, 0])
    rows.update(change)
    assert ndp.decide(**rows, streak=0)[0] == expect


def test_decide_knobs():
    rows = dict(prefillable=[0, 1], cont=[0, 1], running=[5, 0], waiting=[0, 0])
    assert ndp.decide(**rows, streak=0, steps=0)[0] == "release"           # steps 0: release only
    assert ndp.decide(**rows, streak=0, peer=False, steps=0)[0] == "stock"
    assert ndp.params({"NUMINOUS_DP_PACE_STEPS": "4", "NUMINOUS_DP_PEER_RELEASE": "0"}) == {
        "steps": 4, "k_max": 2, "q_max": 0, "peer": False}


# ---------------------------------------------------------------- the patched trees
def _burst_tree():
    for f in sorted(glob.glob(str(pathlib.Path.home() / ".cache/numinous/preflight_src/*/python/sglang") + "/" + h.DELAYER)):
        t = open(f).read()
        if "decode_streak" in t and "numinous-hook:m_dp_pace" not in t and "numinous-hook:pd_delayer_peer" not in t:
            root = pathlib.Path(f[: -len(h.DELAYER) - 1])
            if "decode_streak" in (root / h.POLICY).read_text():
                return root
    return None


TREES = ["stock"] + (["burst"] if _burst_tree() else [])


@pytest.fixture(scope="module", params=TREES)
def tree(request, tmp_path_factory):
    base = PKG if request.param == "stock" else _burst_tree()
    root = tmp_path_factory.mktemp(f"sgl_{request.param}") / "sglang"
    for rel in (h.SCHED, h.DELAYER, h.POLICY):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(base / rel, root / rel)
    out = h.apply(str(root))
    assert sum("applied to" in x for x in out) == len(h.HOOKS) and (root / h.MODULE_DST).exists(), out
    again = h.apply(str(root))
    assert all("already applied" in x for x in again if not x.startswith("module"))
    return root


def _load_delayer(root: pathlib.Path):
    sched = types.SimpleNamespace(prefill_delayer_queue_min_ratio=None, prefill_delayer_max_delay_ms=None,
                                  prefill_max_requests=None, disable_overlap_schedule=False)
    par = types.SimpleNamespace(enable_dp_attention=True)
    managers = types.ModuleType("sglang.srt.managers")
    spec = importlib.util.spec_from_file_location("sglang.srt.managers.numinous_dp_pace", root / h.MODULE_DST)
    pace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pace)
    managers.numinous_dp_pace = pace
    stubs = {
        "sglang": types.ModuleType("sglang"), "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.managers": managers, "sglang.srt.managers.numinous_dp_pace": pace,
        "sglang.srt.environ": types.SimpleNamespace(envs=types.SimpleNamespace(
            SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=types.SimpleNamespace(get=lambda: False))),
        "sglang.srt.runtime_context": types.SimpleNamespace(get_parallel=lambda: par, get_schedule=lambda: sched),
        "sglang.srt.utils": types.SimpleNamespace(get_bool_env_var=lambda *a, **k: False),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("m_pace_delayer", root / h.DELAYER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, stubs


class _Gather:
    """Info rows (5 stock / 6 decode-burst columns) from `info`, pacing rows (3 ints) from `pace`; rank 0 is local."""

    def __init__(self, info, pace):
        self.info, self.pace, self.widths = info, pace, []

    def __call__(self, out, local, group=None):
        n = local.numel()
        self.widths.append(n)
        src = self.pace if n == 3 else self.info
        table = torch.tensor([(list(r) + [0] * n)[:n] for r in src], dtype=torch.int64)
        table[0, :n] = local
        out.copy_(table.flatten())


def _negotiate(tree, monkeypatch, info, pace, local_prefillable, streak=0, env="1"):
    mod, stubs = _load_delayer(tree)
    if env:
        monkeypatch.setenv(h.ENV, env)
    else:
        monkeypatch.delenv(h.ENV, raising=False)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", _Gather(info, pace))
    d = mod.PrefillDelayer(dp_size=4, attn_tp_size=1, cpu_group=None, max_delay_passes=30,
                           token_usage_low_watermark=None, debug_log_enabled=False)
    d._numinous_local = tuple(pace[0])
    d._numinous_pace_streak = streak
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)                       # the decide hook imports numinous_dp_pace lazily
    try:
        out = d._negotiate_should_allow_prefill(local_prefillable=local_prefillable, token_usage=0.5)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return d, out


# info: prefillable, watermark, running(delayer field), max_prefill_bs, waiting ; pace: continuation, running, waiting
LONE_INFO = [[1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
LONE_PACE = [[1, 1, 0], [0, 3, 0], [0, 2, 0], [0, 4, 0]]         # rank 0 mid-prompt; ranks 1-3 hold decodes


def test_lone_continuation_parks(tree, monkeypatch):
    d, out = _negotiate(tree, monkeypatch, LONE_INFO, LONE_PACE, local_prefillable=True)
    assert not out.output_allow and out.output_reason == "numinous_pace" and d._numinous_park
    assert d._numinous_pace_streak == 1


def test_cap_lets_the_chunk_run(tree, monkeypatch):
    d, out = _negotiate(tree, monkeypatch, LONE_INFO, LONE_PACE, local_prefillable=True, streak=2)
    assert not d._numinous_park and d._numinous_pace_streak == 0
    assert out.output_allow                       # released (or stock allow): the continuation runs this pass


def test_held_new_work_joins_a_continuation_pass(tree, monkeypatch):
    # rank 1 has queued new work, rank 0 runs a continuation, ranks 2-3 have nothing: stock 'mixed' would delay rank 1
    info = [[1, 0, 0, 1, 0], [1, 0, 0, 1, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    pace = [[1, 1, 0], [0, 0, 2], [0, 2, 0], [0, 1, 0]]
    d, out = _negotiate(tree, monkeypatch, info, pace, local_prefillable=True)
    assert out.output_allow and out.output_reason == "numinous_peer_forward" and not d._numinous_park


def test_env_off_is_stock(tree, monkeypatch):
    d, out = _negotiate(tree, monkeypatch, LONE_INFO, LONE_PACE, local_prefillable=True, env="")
    assert out.output_reason in ("delay", "") and not getattr(d, "_numinous_park", False)


# ---------------------------------------------------------------- add_chunked_req, executed on a fake adder
def _add_chunked_req(tree):
    src = (tree / h.POLICY).read_text()
    mod = ast.parse(src)
    fn = next(n for n in ast.walk(mod) if isinstance(n, ast.FunctionDef) and n.name == "add_chunked_req")
    code = compile(ast.Module(body=[fn], type_ignores=[]), "add_chunked_req", "exec")
    ns = {}
    exec(code, ns)
    return ns["add_chunked_req"]


class _Exec:
    def __init__(self, park):
        self.calls = 0
        self._prefill_delayer = types.SimpleNamespace(_numinous_park=park)

    def negotiate_should_allow_prefill(self, **kw):
        self.calls += 1
        return True


def _adder(park):
    a = types.SimpleNamespace(dllm_config=None, rem_chunk_tokens=2048, rem_total_tokens=10 ** 6, is_hybrid_swa=False,
                              prefill_delayer_single_pass=_Exec(park), running_batch=types.SimpleNamespace(
                                  batch_size=lambda: 3), max_prefill_bs=8, max_running_requests=64,
                              waiting_queue_len=0, can_run_list=[], decode_streak=0, decode_burst=8,
                              _update_prefill_budget=lambda *x, **k: None)
    return a


class _Req:
    def __init__(self):
        self.full_untruncated_fill_ids = list(range(10000))
        self.prefix_indices = list(range(4096))
        self.extend_range = None

    def set_extend_range(self, a, b):
        self.extend_range = types.SimpleNamespace(length=b - a, start=a, end=b)


def test_add_chunked_req_parks(tree, monkeypatch):
    monkeypatch.setenv(h.ENV, "1")
    f = _add_chunked_req(tree)
    a, r = _adder(park=True), _Req()
    assert f(a, r) is r and a.can_run_list == [] and r.extend_range is None
    assert a.prefill_delayer_single_pass.calls == 1


def test_add_chunked_req_runs_when_not_parked(tree, monkeypatch):
    monkeypatch.setenv(h.ENV, "1")
    f = _add_chunked_req(tree)
    a, r = _adder(park=False), _Req()
    try:
        f(a, r)
    except (AttributeError, TypeError, NameError):
        pass                                    # stock tail touches fields the fake does not model
    assert a.can_run_list == [r] and r.extend_range.length == 2048


def test_scheduler_mark_placement(tree):
    s = (tree / h.SCHED).read_text()
    i = s.index("numinous-hook:m_dp_pace_mark")
    fn = s.rfind("def get_new_batch_prefill(self, running_batch", 0, i)
    assert fn != -1 and i < s.index("self._get_new_batch_prefill_raw(", i)
    assert "int(running_batch.batch_size())" in s[i:i + 400]


def test_preflight():
    ctx = types.SimpleNamespace(env={h.ENV: "1"}, args={"enable_prefill_delayer": True, "enable_dp_attention": True,
                                                        "dp_size": 8, "speculative_algorithm": "EAGLE",
                                                        "disaggregation_mode": "null"})
    assert h.numinous_preflight(ctx) == []
    ctx.args["enable_prefill_delayer"] = False
    assert any("inert" in x for x in h.numinous_preflight(ctx))
    ctx.env = {}
    assert h.numinous_preflight(ctx) == []
