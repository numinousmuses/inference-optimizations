"""Composition test (final exam): m_dp_pace's behaviour on the FULL composed GLM M tree, i.e. the v0.5.20 tree with
every hook of with_dp_pace(arm_m6()) applied exactly as the boot applies them (hopper_cells.patched_src: the files at
the cell's pinned ref, install entries first, each patcher's --root mode), not only on a tree with this hook alone.
amdpro 2026-10-05."""
from __future__ import annotations

import pathlib
import py_compile
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hopper"))
import m_dp_pace_hooks as h  # noqa: E402
import test_m_dp_pace_hooks as t  # noqa: E402


@pytest.fixture(scope="module", params=["m6_pace", "final_gx1"])
def composed(request):
    """with_dp_pace(arm_m6()) and the exact final-exam step-1 cell (amdglm's final_chain()[0]: M6 + FP8 passthrough +
    lazy dequant + w4a8dg + pacing, hooks pinned at its frozen ref)."""
    import glm_hopper_cells as g
    import glm_m_pace as mp
    import hopper_cells as hc
    if request.param == "m6_pace":
        if not hasattr(g, "arm_m6"):
            pytest.skip("no arm_m6")
        cell = mp.with_dp_pace(g.arm_m6())
    else:
        if not hasattr(g, "final_chain"):
            pytest.skip("no final_chain")
        cell = g.final_chain()[0]
        assert cell["config"]["env"].get(h.ENV) == "1"
    root = pathlib.Path(hc.patched_src(cell["config"])) / "sglang"
    return cell, root


def test_every_hook_applied_and_files_compile(composed):
    cell, root = composed
    for rel in (h.SCHED, h.DELAYER, h.POLICY, h.MODULE_DST):
        py_compile.compile(str(root / rel), doraise=True)
    for name, rel, _anchor, text in h.HOOKS:
        assert text in (root / rel).read_text(), f"{name} not applied in the composed tree"
    runs = [r[0] for r in cell["config"]["hooks"]["run"]]
    assert "operations/patches/m_dp_pace_hooks.py" in runs and len(runs) >= 8


def test_park_release_stock_on_the_composed_delayer(composed, monkeypatch):
    _, root = composed
    d, out = t._negotiate(root, monkeypatch, t.LONE_INFO, t.LONE_PACE, local_prefillable=True)
    assert not out.output_allow and out.output_reason == "numinous_pace" and d._numinous_park
    d, out = t._negotiate(root, monkeypatch, t.LONE_INFO, t.LONE_PACE, local_prefillable=True, streak=2)
    assert out.output_allow and not d._numinous_park
    info = [[1, 0, 0, 1, 0], [1, 0, 0, 1, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    pace = [[1, 1, 0], [0, 0, 2], [0, 2, 0], [0, 1, 0]]
    d, out = t._negotiate(root, monkeypatch, info, pace, local_prefillable=True)
    assert out.output_allow and out.output_reason == "numinous_peer_forward"
    d, out = t._negotiate(root, monkeypatch, t.LONE_INFO, t.LONE_PACE, local_prefillable=True, env="")
    assert not getattr(d, "_numinous_park", False) and out.output_reason != "numinous_pace"


def test_add_chunked_req_on_the_composed_policy(composed, monkeypatch):
    _, root = composed
    monkeypatch.setenv(h.ENV, "1")
    f = t._add_chunked_req(root)
    a, r = t._adder(park=True), t._Req()
    assert f(a, r) is r and a.can_run_list == []
    a, r = t._adder(park=False), t._Req()
    try:
        f(a, r)
    except (AttributeError, TypeError, NameError):
        pass
    assert a.can_run_list == [r]


def test_scheduler_mark_on_the_composed_scheduler(composed):
    _, root = composed
    t.test_scheduler_mark_placement(root)
