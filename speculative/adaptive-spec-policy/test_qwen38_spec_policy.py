"""CPU-safe tests for qwen38_spec_policy.py's roofline-derived policy adapter, plus a GPU bandwidth-probe test."""
import pytest
import torch

from operations.kernels import qwen38_spec_policy as m


def test_accept_lengths_within_valid_range():
    for name, table in m.ACCEPT_LENGTHS.items():
        gamma = 7 if "gamma7" in name else (8 if "gamma8" in name else 3)
        for c, v in table.items():
            assert 1.0 <= v <= gamma + 1, f"{name}@{c}: {v} outside [1, {gamma + 1}]"


def test_roofline_cost_table_monotonic_and_positive():
    table = m.roofline_cost_table(gamma=7, max_per_rank=32, bandwidth_gbps=1400.0)
    secs = [table.seconds_at(t) for t in sorted(table.tokens)]
    assert all(s > 0 for s in secs)
    assert secs == sorted(secs)  # more tokens never gets cheaper


def test_roofline_floor_dominates_at_low_token_counts():
    """The point of the dense-model argument: near gamma+1 tokens, cost is close to the weight-read floor, not
    to the floor times some multiple (i.e. a few more verify rows are nearly free)."""
    table = m.roofline_cost_table(gamma=7, max_per_rank=32, bandwidth_gbps=1400.0)
    floor = table.seconds_at(1)
    near_gamma = table.seconds_at(8)
    assert near_gamma / floor < 1.05


def test_build_qwen38_policy_is_a_valid_policy_object():
    pol = m.build_qwen38_policy(gamma=7, max_per_rank=16, bandwidth_gbps=1400.0, accept_key="dspark_nvfp4_gamma7")
    assert pol.gamma == 7
    assert len(pol.rows_by_load) == 16
    assert all(1 <= r <= 8 for r in pol.rows_by_load)
    # rows are non-increasing in load (DSparkLoadPolicy.build's own invariant, exercised end to end here)
    assert pol.rows_by_load == sorted(pol.rows_by_load, reverse=True)


def test_build_qwen38_policy_rejects_unknown_draft():
    with pytest.raises(KeyError):
        m.build_qwen38_policy(gamma=7, max_per_rank=8, bandwidth_gbps=1400.0, accept_key="nope")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_measure_hbm_bandwidth_reports_a_plausible_rtx_pro_6000_number():
    gbps = m.measure_hbm_bandwidth_gbps(size_mb=128, iters=5)
    # RTX PRO 6000 Blackwell publishes ~1.6-1.8 TB/s; a device-copy probe should land in a broad plausible band,
    # not assert the vendor number (measured here, not assumed).
    assert 200.0 < gbps < 4000.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_report_runs_end_to_end_on_gpu():
    rep = m.report(max_per_rank=8)
    assert rep["bandwidth_gbps"] > 0
    assert len(rep["rank_tok_s_by_load"]) == 4  # (1, 2, 4, 8) at max_per_rank=8
    for row in rep["rank_tok_s_by_load"]:
        assert row["tok_s"] > 0
