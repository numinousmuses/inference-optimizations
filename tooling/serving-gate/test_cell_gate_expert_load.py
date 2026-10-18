"""cell_gate expert-load summary (per-layer EP rank skew from SGLang's expert-distribution dump) and the
_expert_record arm plumbing (start before the measured round, dump after)."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("GATE_LOCAL", "1")


def _cg():
    import cell_gate
    return importlib.reload(cell_gate)


def test_balanced_load_is_skew_one():
    cg = _cg()
    s = cg.expert_load_summary([[10.0] * 16, [5.0] * 16], ep=4)
    assert s["layers"] == 2 and s["rank_skew_mean"] == 1.0 and s["rank_skew_max"] == 1.0


def test_hot_rank_skew():
    cg = _cg()
    # 4 ranks x 4 experts; rank 0's experts carry 3x the others' load in layer 0, balanced in layer 1
    layer0 = [30.0] * 4 + [10.0] * 12
    layer1 = [10.0] * 16
    s = cg.expert_load_summary([layer0, layer1, [0.0] * 16], ep=4)
    assert s["layers"] == 2                       # all-zero (dense / unrecorded) layers are dropped
    assert s["rank_skew_max"] == round(120 / 60, 3)
    assert s["rank_skew_p50"] == 2.0
    assert s["rank_skew_mean"] == 1.5


def test_record_plumbing_in_source():
    src = open(os.path.join(os.path.dirname(__file__), "cell_gate.py")).read()
    assert '_post(url + "/start_expert_distribution_record")' in src
    assert 'row["expert_load"] = _expert_record_dump(' in src
