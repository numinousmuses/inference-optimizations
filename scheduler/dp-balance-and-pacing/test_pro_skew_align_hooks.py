"""Tests for pro_skew_align_hooks.py (ride-along and min-ranks release of the DP prefill delayer).

CPU: the pure decision, and the wrapper plumbing on a fake delayer (FORCED reaches the gather only inside
add_chunked_req; overrides replace a "delay" verdict; the prefillable count is not inflated).
Box (sglang v0.5.20 importable): the wrappers on a subclass of the REAL PrefillDelayer, so the stock mixed-state
branch decides and the hooks override it; plus the anchor check and apply() on a copy of scheduler.py.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections import namedtuple

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _mod(ride=False, min_ranks=0, idle_release=False):
    import pro_skew_align_hooks as hooks
    ns = {"__name__": "numinous_pro_skew_align_test"}
    exec(hooks.MODULE_SOURCE, ns)
    ns["_ride"], ns["_min_ranks"], ns["_idle_release"] = ride, min_ranks, idle_release
    return ns


# ---- pure decision ---------------------------------------------------------------------------------------------

class TestDecide:
    def d(self, col0, local, allow=False, reason="delay", status="mixed", min_ranks=0, ride=True):
        return _mod()["decide"](col0, local, allow, reason, status, min_ranks, ride)

    def test_ride_releases_prefillable_rank_when_another_is_mid_chunk(self):
        assert self.d([2, 1, 0, 0, 0, 0, 0, 0], 1) == "ride"

    def test_forced_rank_itself_keeps_stock_verdict(self):
        assert self.d([2, 1, 0, 0, 0, 0, 0, 0], 2) is None

    def test_not_prefillable_rank_unchanged(self):
        assert self.d([2, 1, 0, 0, 0, 0, 0, 0], 0) is None

    def test_no_forced_rank_no_ride(self):
        assert self.d([1, 1, 0, 0, 0, 0, 0, 0], 1) is None

    def test_ride_off(self):
        assert self.d([2, 1, 0, 0, 0, 0, 0, 0], 1, ride=False) is None

    def test_only_mixed_delay_is_overridden(self):
        assert self.d([2, 1, 1, 1, 1, 1, 1, 1], 1, status="all") is None
        assert self.d([2, 1, 0, 0, 0, 0, 0, 0], 1, allow=True, reason="wait_timeout") is None
        assert self.d([2, 1, 0, 0, 0, 0, 0, 0], 1, reason="token_watermark") is None

    def test_align_counts_forced_as_prefillable(self):
        assert self.d([2, 1, 1, 0, 0, 0, 0, 0], 1, min_ranks=3, ride=False) == "align"
        assert self.d([1, 1, 0, 0, 0, 0, 0, 0], 1, min_ranks=3, ride=False) is None

    def test_bool_local_prefillable(self):
        assert self.d([2, 1, 0, 0, 0, 0, 0, 0], True) == "ride"

    def test_no_gather(self):
        assert self.d(None, 1) is None

    def test_idle_release_when_no_other_rank_runs(self):
        m = _mod()
        dec = m["decide"]
        col0, idle = [1, 0, 0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0, 0]
        assert dec(col0, 1, False, "delay", "mixed", 0, False, idle, True) == "idle"
        # a non-prefillable rank is decoding: holding protects it, keep the stock verdict
        assert dec(col0, 1, False, "delay", "mixed", 0, False, [5, 0, 3, 0, 0, 0, 0, 0], True) is None
        # prefillable ranks' own running requests do not matter
        assert dec([1, 1, 0, 0, 0, 0, 0, 0], 1, False, "delay", "mixed", 0, False, [9, 9, 0, 0, 0, 0, 0, 0], True) == "idle"
        # off, or no running column
        assert dec(col0, 1, False, "delay", "mixed", 0, False, idle, False) is None
        assert dec(col0, 1, False, "delay", "mixed", 0, False, None, True) is None
        assert dec(col0, 1, False, "delay", "mixed", 0, False, idle[:3], True) is None


class TestParse:
    def test_min_ranks(self):
        m = _mod()
        assert m["parse_min_ranks"](None) == 0 and m["parse_min_ranks"]("0") == 0
        assert m["parse_min_ranks"]("1") == m["DEFAULT_MIN_RANKS"] and m["parse_min_ranks"]("4") == 4
        with pytest.raises(ValueError):
            m["parse_min_ranks"]("-2")

    def test_idle_release_env_uses_ride_parser(self):
        m = _mod()
        assert m["parse_ride"]("1") and not m["parse_ride"]("0")

    def test_ride(self):
        m = _mod()
        assert not m["parse_ride"](None) and not m["parse_ride"]("0") and not m["parse_ride"]("")
        assert m["parse_ride"]("1")


# ---- wrapper plumbing on a fake delayer ------------------------------------------------------------------------

Out = namedtuple("Out", "next_state input_estimation output_allow output_reason num_prefillable "
                        "num_token_watermark_force_allow wait_forward_passes wait_seconds")


def _fake_classes(others):
    """A delayer whose pure step gathers [local] + others and applies the stock mixed rule (always delay)."""
    seen = {}

    class FakeDelayer:
        def _gather_info(self, local_prefillable, *a, **k):
            import types
            seen["local"] = local_prefillable
            rows = [int(local_prefillable)] + list(others)

            class T:  # minimal tensor stand-in: out[:, 0].tolist()
                def __getitem__(self, idx):
                    return types.SimpleNamespace(tolist=lambda: rows)
            return T()

        def _negotiate_should_allow_prefill_pure(self, prev_state, local_prefillable, token_usage, **k):
            self._gather_info(local_prefillable)
            rows = [int(local_prefillable)] + list(others)
            status = "all" if min(rows) > 0 else "none" if max(rows) == 0 else "mixed"
            return Out("state", status, status != "mixed", "delay" if status == "mixed" else "no_wait",
                       sum(rows), 0, 0, 0.0)

    class FakeAdder:
        def __init__(self, delayer):
            self.d = delayer

        def add_chunked_req(self, req):
            return self.d._negotiate_should_allow_prefill_pure(prev_state=None, local_prefillable=True, token_usage=0.5)

        def add_one_req(self):
            return self.d._negotiate_should_allow_prefill_pure(prev_state=None, local_prefillable=True, token_usage=0.5)

    return FakeDelayer, FakeAdder, seen


def _wrap(m, D, A):
    m["_patch"](D, "_negotiate_should_allow_prefill_pure", m["_wrap_negotiate_pure"])
    m["_patch"](D, "_gather_info", m["_wrap_gather_info"])
    m["_patch"](A, "add_chunked_req", m["_wrap_add_chunked_req"])


def test_forced_value_reaches_gather_only_inside_add_chunked_req():
    m = _mod(ride=True)
    D, A, seen = _fake_classes([0] * 7)
    _wrap(m, D, A)
    adder = A(D())
    adder.add_chunked_req(None)
    assert seen["local"] == 2
    adder.add_one_req()
    assert seen["local"] is True
    assert m["_in_chunked"] is False


def test_ride_overrides_delay_and_reports_existing_reason():
    m = _mod(ride=True)
    D, A, _ = _fake_classes([2, 0, 0, 0, 0, 0, 0])  # rank 1 is mid-chunk
    _wrap(m, D, A)
    out = A(D()).add_one_req()
    assert out.output_allow and out.output_reason == "wait_success" and out.next_state is None
    assert out.num_prefillable == 2  # FORCED counted once, not twice
    assert m["_counts"]["ride"] == 1


def test_mid_chunk_rank_itself_not_overridden_and_count_fixed():
    m = _mod(ride=True)
    D, A, _ = _fake_classes([1, 0, 0, 0, 0, 0, 0])
    _wrap(m, D, A)
    out = A(D()).add_chunked_req(None)  # this rank is the forced one: verdict stays (it extends anyway)
    assert not out.output_allow and out.output_reason == "delay"
    assert out.num_prefillable == 2


def test_disabled_is_pass_through():
    m = _mod(ride=False, min_ranks=0)
    D, A, seen = _fake_classes([2, 0, 0, 0, 0, 0, 0])
    _wrap(m, D, A)
    out = A(D()).add_chunked_req(None)
    assert seen["local"] is True and not out.output_allow


def test_wrap_is_idempotent():
    m = _mod(ride=True)
    D, A, _ = _fake_classes([0] * 7)
    _wrap(m, D, A)
    f = D.__dict__["_gather_info"]
    _wrap(m, D, A)
    assert D.__dict__["_gather_info"] is f


def test_module_source_parses():
    import pro_skew_align_hooks as hooks
    compile(hooks.MODULE_SOURCE, "numinous_pro_skew_align.py", "exec")


# ---- box: the real v0.5.20 delayer ------------------------------------------------------------------------------

def _real_delayer_subclass(others, m, others_running=10):
    torch = pytest.importorskip("torch")
    try:
        from sglang.srt.managers.prefill_delayer import PrefillDelayer
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"sglang prefill_delayer not importable: {e}")
    seen = {}

    class Scripted(PrefillDelayer):
        def _gather_info(self, local_prefillable, local_token_watermark_force_allow, running_batch=0,
                         max_prefill_bs=0, waiting_queue_len=0):
            seen["local"] = local_prefillable
            seen["running"] = running_batch
            rows = [[int(local_prefillable), 0, running_batch, max_prefill_bs, waiting_queue_len]]
            rows += [[p, 0, others_running, 1, 1 if p else 0] for p in others]
            return torch.tensor(rows, dtype=torch.int64)

    d = object.__new__(Scripted)
    d.__dict__.update(_max_delay_passes=30, _token_usage_low_watermark=None, _debug_log_enabled=False,
                      _queue_min_ratio=None, _max_delay_ms=5000.0, _queue_trigger_enabled=False,
                      _prefill_max_requests=None, dp_size=8, enable_dp_attention=True, _metrics_collector=None,
                      _curr_state=None, skip_first_delayer=False)
    m["_patch"](Scripted, "_negotiate_should_allow_prefill_pure", m["_wrap_negotiate_pure"])
    m["_patch"](Scripted, "_gather_info", m["_wrap_gather_info"])
    return d, seen


def _negotiate(d):
    return d._negotiate_should_allow_prefill(local_prefillable=True, token_usage=0.5, running_batch=10,
                                             max_prefill_bs=1, max_running_requests=512, waiting_queue_len=1)


def test_real_delayer_stock_delays_mixed():
    m = _mod(ride=False)
    d, _ = _real_delayer_subclass([2, 0, 0, 0, 0, 0, 0], m)
    out = _negotiate(d)
    assert out.input_estimation == "mixed" and not out.output_allow and out.output_reason == "delay"


def test_real_delayer_ride_releases_and_resets_state():
    m = _mod(ride=True)
    d, _ = _real_delayer_subclass([2, 0, 0, 0, 0, 0, 0], m)
    out = _negotiate(d)
    assert out.output_allow and out.output_reason == "wait_success" and out.num_prefillable == 2
    assert d._curr_state is None  # the delayer's held-pass counter restarts, as on a stock release


def test_real_delayer_ride_without_mid_chunk_rank_keeps_delay():
    m = _mod(ride=True)
    d, _ = _real_delayer_subclass([1, 0, 0, 0, 0, 0, 0], m)
    out = _negotiate(d)
    assert not out.output_allow and d._curr_state is not None and d._curr_state.delayed_count == 1


def test_real_delayer_timeout_path_untouched():
    m = _mod(ride=True)
    d, _ = _real_delayer_subclass([1, 0, 0, 0, 0, 0, 0], m)
    for _ in range(29):
        _negotiate(d)
    out = _negotiate(d)
    assert out.output_allow and out.output_reason == "wait_timeout"


def test_real_record_accepts_override_reason():
    """_record_single_pass_result asserts the reason set under the delayer debug log: wait_success is in it."""
    m = _mod(ride=True)
    d, _ = _real_delayer_subclass([2, 0, 0, 0, 0, 0, 0], m)
    from sglang.srt.managers import prefill_delayer as pd
    pd._record_single_pass_result(True, _negotiate(d), None, debug_log_enabled=True)


def test_real_delayer_idle_release_lone_request_at_low_load():
    m = _mod(idle_release=True)
    d, _ = _real_delayer_subclass([0] * 7, m, others_running=0)
    out = _negotiate(d)
    assert out.output_allow and out.output_reason == "wait_success" and d._curr_state is None
    assert m["_counts"]["idle"] == 1


def test_real_delayer_idle_release_holds_while_others_decode():
    m = _mod(idle_release=True)
    d, _ = _real_delayer_subclass([0] * 7, m, others_running=10)
    out = _negotiate(d)
    assert not out.output_allow and out.output_reason == "delay"


def test_real_finalize_path_reports_true_running_under_idle_release():
    """A non-prefillable rank negotiates through finalize() with running_batch=0; idle-release must report the
    running count recorded from get_new_batch_prefill instead (the idle_d120 gate released every mixed pass)."""
    m = _mod(idle_release=True)
    d, seen = _real_delayer_subclass([1] + [0] * 6, m, others_running=0)
    from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor

    class FakeScheduler:
        def get_new_batch_prefill(self, running_batch):
            ex = PrefillDelayerSinglePassExecutor(d, token_usage=0.5)
            ex.finalize(actual_prefill_bs=0)
            return ex._result

    class RB:
        def batch_size(self):
            return 7

    m["_patch"](FakeScheduler, "get_new_batch_prefill", m["_wrap_get_new_batch_prefill"])
    FakeScheduler().get_new_batch_prefill(RB())
    assert seen["local"] is False and seen["running"] == 7


def test_real_finalize_path_stock_reports_zero_running():
    """Documents the stock behaviour the fix works around."""
    m = _mod()
    d, seen = _real_delayer_subclass([1] + [0] * 6, m, others_running=0)
    from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor
    PrefillDelayerSinglePassExecutor(d, token_usage=0.5).finalize(actual_prefill_bs=0)
    assert seen["running"] == 0


def test_real_delayer_stock_holds_lone_request_at_low_load():
    """The hazard idle-release removes: stock holds a lone request although every other rank is idle."""
    m = _mod()
    d, _ = _real_delayer_subclass([0] * 7, m, others_running=0)
    assert not _negotiate(d).output_allow


def test_anchors_and_apply_on_copy():
    import pro_skew_align_hooks as hooks
    try:
        root = hooks.sglang_root()
    except SystemExit:
        pytest.skip("sglang not importable")
    tmp = tempfile.mkdtemp()
    try:
        dst = os.path.join(tmp, "sglang")
        os.makedirs(os.path.join(dst, "srt/managers"))
        shutil.copy2(os.path.join(root, "srt/managers/scheduler.py"), os.path.join(dst, "srt/managers/scheduler.py"))
        lines = hooks.apply(dst)
        assert any("applied" in ln for ln in lines)
        src = open(os.path.join(dst, "srt/managers/scheduler.py")).read()
        assert "numinous-hook:pro_skew_align_install" in src and "NUMINOUS_PREFILL_RIDE" in src
        assert "NUMINOUS_PREFILL_IDLE_RELEASE" in src
        assert all("already applied" in ln or "module:" in ln for ln in hooks.apply(dst))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
