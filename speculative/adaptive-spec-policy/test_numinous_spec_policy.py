"""CPU tests of the DSpark load policy and the TBO capture helper (numinous_spec_policy.py). The sglang-dependent
checks (SGLang's own SpsAdditiveCostTable and budget optimiser on our emitted table) skip without sglang and run on
the dev box."""
from __future__ import annotations

import bisect
import importlib.util
import json
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numinous_spec_policy as nsp  # noqa: E402

HAS_SGLANG = importlib.util.find_spec("sglang") is not None
B200 = nsp.StepCostTable.from_sps(nsp.PRO_B200_SPS)
ACCEPT7 = nsp.AcceptanceModel.from_accept_length(4.27, 5, positions=7)


# ---------------------------------------------------------------- step cost
def test_step_cost_matches_profile_and_interpolates():
    for t, s in zip(nsp.PRO_B200_SPS["sample_batch_tokens"], nsp.PRO_B200_SPS["sample_steps_per_sec"]):
        assert B200.seconds_at(t) == pytest.approx(1 / s)
    mid = (1 / 19.51902201892076 + 1 / 17.110012030477208) / 2
    assert B200.seconds_at(120) == pytest.approx(mid)
    assert B200.seconds_at(1) == B200.seconds_at(6)  # never cheaper than the smallest probe


def test_extension_continues_the_last_segment():
    ext = B200.extended(512)
    slope = (B200.seconds[-1] - B200.seconds[-2]) / 48
    assert ext.tokens[-1] == 512
    assert ext.seconds_at(512) == pytest.approx(B200.seconds[-1] + slope * 320)
    assert B200.extended(100) is B200


def test_from_records_takes_the_median_per_token_count():
    recs = [{"num_verify_tokens": 6, "step_time": x} for x in (0.02, 0.03, 0.025)]
    recs += [json.dumps({"num_verify_tokens": 12, "step_time": x}) for x in (0.04, 0.05)]
    t = nsp.StepCostTable.from_records(recs)
    assert t.tokens == (6, 12) and t.seconds == pytest.approx((0.025, 0.045))


def test_additive_json_reproduces_the_interpolated_curve():
    add = B200.extended(512).to_additive_json()
    for n, budget in [(1, 5), (16, 80), (24, 100), (32, 160), (64, 400)]:
        assert nsp.additive_step_seconds(add, n, budget) == pytest.approx(B200.extended(512).seconds_at(n + budget))


def _floor_sps(tokens: int) -> float:
    xs = nsp.PRO_B200_SPS["sample_batch_tokens"]
    i = max(0, min(bisect.bisect_right(xs, tokens) - 1, len(xs) - 1))
    return nsp.PRO_B200_SPS["sample_steps_per_sec"][i]


def _stock_budget(n: int, price) -> int:
    """SGLang's compute_verify_token_budget objective, CPU copy: max over b of (n + top-b survival) x speed."""
    p = ACCEPT7.q[0]
    surv = sorted([p ** k for k in range(1, 6) for _ in range(n)], reverse=True)
    vals = [(n + sum(surv[:b])) * price(n, b) for b in range(len(surv) + 1)]
    return max(range(len(vals)), key=vals.__getitem__)


def test_floor_table_underprices_tokens_below_each_probe():
    """Why the measured compact arm lost accept length (4.04-4.09 vs 4.21-4.30): with the floor lookup the planner
    stops one token below a probe, and at 24 requests per rank it verifies 95 of 144 rows (derived)."""
    floor = _stock_budget(24, lambda n, b: _floor_sps(n + b))
    add = B200.extended(512).to_additive_json()
    interp = _stock_budget(24, lambda n, b: 1 / nsp.additive_step_seconds(add, n, b))
    assert 24 + floor == 95
    assert 24 + interp == 144
    assert 16 + _stock_budget(16, lambda n, b: _floor_sps(n + b)) == 95


# ---------------------------------------------------------------- acceptance
def test_accept_length_fit_reproduces_the_measurement():
    a = nsp.AcceptanceModel.from_accept_length(4.27, 5, positions=5)
    assert a.expected_tokens(6) == pytest.approx(4.27, abs=1e-6)
    assert a.q[0] == pytest.approx(0.8617, abs=1e-3)
    assert a.expected_tokens(1) == 1.0


def test_tail_decay_only_touches_positions_past_the_measured_block():
    a = nsp.AcceptanceModel.from_accept_length(4.27, 5, positions=7, tail_decay=0.5)
    assert a.q[:5] == pytest.approx([a.q[0]] * 5)
    assert a.q[5] == pytest.approx(a.q[0] * 0.5) and a.q[6] == pytest.approx(a.q[0] * 0.25)
    with pytest.raises(ValueError):
        nsp.AcceptanceModel.from_accept_length(7.5, 5, positions=5)


# ---------------------------------------------------------------- policy
def test_default_gamma7_policy_transitions_are_pinned():
    pol = nsp.default_policy(7)
    assert pol.transitions() == [(1, 8), (21, 7), (30, 6), (44, 5)]
    assert all(a >= b for a, b in zip(pol.rows_by_load, pol.rows_by_load[1:]))


def test_gamma5_policy_is_verify_all_at_served_loads():
    """Pro Standard serves ~16 per rank at gamma 5: verify-all is optimal there, compact alone cannot gain."""
    pol = nsp.default_policy(5)
    assert all(pol.rows(n) == 6 for n in range(1, 33))


def test_budget_cap_bounds_the_step_tokens():
    pol = nsp.default_policy(7)
    assert pol.budget_cap(16) == 16 * 7
    assert pol.budget_cap(48) == 48 * 4
    assert pol.cap(48, 10_000) == 192 and pol.cap(48, 50) == 50 and pol.cap(48, -3) == 0
    assert pol.rows(0) == 8 and pol.rows(10_000) == pol.rows_by_load[-1]


def test_cap_never_loses_to_verify_all_under_the_model():
    ext = B200.extended(64 * 8)
    pol = nsp.DSparkLoadPolicy.build(B200, ACCEPT7, 7, 64)
    for n in range(1, 65):
        capped = nsp.rank_tokens_per_second(n, pol.rows(n), ext, ACCEPT7)
        full = nsp.rank_tokens_per_second(n, 8, ext, ACCEPT7)
        assert capped >= full * (1 - 0.01)  # the monotone guard may cost < 1% at a kink


def test_policy_json_round_trip_and_validation(tmp_path):
    pol = nsp.default_policy(6)
    p = tmp_path / "pol.json"
    p.write_text(json.dumps(pol.to_json()))
    back = nsp.DSparkLoadPolicy.from_json(str(p))
    assert back.rows_by_load == pol.rows_by_load and back.gamma == 6
    with pytest.raises(ValueError):
        nsp.DSparkLoadPolicy(5, [7])
    with pytest.raises(ValueError):
        nsp.DSparkLoadPolicy.from_json({"kind": "other"})


def test_recommend_gamma_prefers_the_wide_block_under_the_measured_curve():
    rec = nsp.recommend_gamma([1, 2, 4, 8, 16, 24, 32], B200, ACCEPT7)
    assert rec["gamma"] == 7
    assert rec["mean_rank_tokens_per_s"][7] > rec["mean_rank_tokens_per_s"][5]


# ---------------------------------------------------------------- runtime glue
class _Decision:
    def __init__(self, budget):
        self.budget = budget


def test_runtime_policy_env_and_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(nsp, "_POLICY", {})
    monkeypatch.setenv("NUMINOUS_DSPARK_ADAPTIVE", "1")
    assert nsp.adaptive_enabled()
    d = nsp.cap_decision(_Decision(10_000), num_requests=48, gamma=7)
    assert d.budget == 192
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps(nsp.default_policy(6).to_json()))
    monkeypatch.setattr(nsp, "_POLICY", {})
    monkeypatch.setenv("NUMINOUS_DSPARK_POLICY", str(pol))
    with pytest.raises(ValueError, match="gamma 6"):
        nsp.runtime_policy(7)
    assert nsp.runtime_policy(6).gamma == 6
    monkeypatch.setenv("NUMINOUS_DSPARK_ADAPTIVE", "0")
    assert not nsp.adaptive_enabled()


def test_planner_init_warns_when_static(monkeypatch, caplog):
    monkeypatch.setattr(nsp, "_LOGGED", set())
    monkeypatch.setenv("NUMINOUS_DSPARK_ADAPTIVE", "1")
    with caplog.at_level("WARNING"):
        nsp.on_planner_init(mode="static", gamma=5, tp_rank=0)
    assert "inert" in caplog.text


def test_cli_build_writes_policy_and_additive_table(tmp_path, capsys):
    out, add = tmp_path / "pol.json", tmp_path / "sps_add.json"
    assert nsp.main(["build", "--gamma", "7", "--out", str(out), "--additive-out", str(add)]) == 0
    assert nsp.DSparkLoadPolicy.from_json(str(out)).gamma == 7
    table = json.loads(add.read_text())
    assert table["m_probes"][-1] == 64 * 8 and "bias_seconds" in table
    assert json.loads(capsys.readouterr().out)["gamma_choice"]["gamma"] == 7


# ---------------------------------------------------------------- TBO capture
class _FakeStrategy:
    def __init__(self, operations, deep_gemm_num_sms=None, tbo_delta_stages=0):
        self.operations, self.deep_gemm_num_sms, self.tbo_delta_stages = operations, deep_gemm_num_sms, tbo_delta_stages

    @staticmethod
    def init_new_tbo(layers, forward_mode):
        (layer,) = layers
        return _FakeStrategy([layer.op_a, layer.op_b], None, 0)

    @classmethod
    def concat(cls, items):
        return cls([op for it in items for op in it.operations], None, 0)


class _FakeLayer:
    """Adds its id to every hidden value (a stand-in for a decoder layer's two TBO ops)."""

    def __init__(self, lid):
        self.lid = lid

    def op_a(self, state, **kw):
        return kw

    def op_b(self, state, **kw):
        return {**kw, "hidden_states": kw["hidden_states"] + self.lid}


class _FB:
    def __init__(self, rng):
        self.tbo_parent_token_range = rng


def _run(ops, inputs):
    out = inputs
    for op in ops:
        out = op(state=None, **out)
    return out


def test_tbo_capture_matches_the_eager_loop():
    layers = [_FakeLayer(i) for i in range(10, 16)]  # layers 10..15, start_layer 10
    capture = [13, 15, 99]
    strat, cap = nsp.tbo_strategy_with_dspark_capture(layers, 10, capture, "EXTEND", strategy_cls=_FakeStrategy)
    assert len(strat.operations) == 6 * 2 + 2
    hc, hidden, total, split = 4, 3, 7, 3
    x = torch.randn(total, hc, hidden)
    # eager reference: capture layer outputs' hc-mean at layers 13 and 15
    ref, h = [], x
    for layer in layers:
        h = h + layer.lid
        if layer.lid in capture:
            ref.append(h.mean(dim=1))
    # two ubatches, the second padded by one row
    ina = {"hidden_states": x[:split], "forward_batch": _FB((0, split)), "tbo_subbatch_index": 0}
    xb = torch.cat([x[split:], torch.zeros(1, hc, hidden)])
    inb = {"hidden_states": xb, "forward_batch": _FB((split, total)), "tbo_subbatch_index": 1}
    outs = [_run(strat.operations, ina), _run(strat.operations, inb)]
    got = cap.merge(outs, total)
    assert len(got) == 2
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r)
    assert cap.store == {0: [], 1: []}  # released after the merge


def test_tbo_capture_fails_loudly_on_a_missing_ubatch():
    cap = nsp.TboDSparkCapture([3])
    cap.store[0].append(torch.zeros(2, 4))
    with pytest.raises(RuntimeError, match="expected 1 layers"):
        cap.merge([{"forward_batch": _FB((0, 2))}, {"forward_batch": _FB((2, 2))}], 2)


# ---------------------------------------------------------------- against the installed sglang (dev box)
@pytest.mark.skipif(not HAS_SGLANG, reason="needs sglang")
def test_sglang_loads_our_additive_table_and_optimises_with_the_cap(tmp_path):
    from sglang.srt.speculative.dspark_components import dspark_planner as dp
    from sglang.srt.speculative.dspark_components.dspark_sps import SpsAdditiveCostTable, load_sps_table_from_path

    path = tmp_path / "add.json"
    path.write_text(json.dumps(B200.extended(512).to_additive_json()))
    table = load_sps_table_from_path(str(path))
    assert isinstance(table, SpsAdditiveCostTable)
    add = json.loads(path.read_text())
    for n, b in [(16, 80), (24, 100), (48, 300)]:
        assert table.step_time(num_reqs=n, budget=b) == pytest.approx(nsp.additive_step_seconds(add, n, b))
    p = ACCEPT7.q[0]
    n = 48
    conf = torch.full((n, 7), p)
    surv = torch.cumprod(conf, dim=1)
    cfg = dp.DSparkScheduleConfig(gamma=7)
    decision = dp.compute_verify_token_budget(history_survival_probs=surv, sps_table=table, cfg=cfg)
    pol = nsp.default_policy(7)
    capped = nsp.cap_decision(decision, num_requests=n, gamma=7)
    assert capped is decision and decision.budget <= pol.budget_cap(n)
