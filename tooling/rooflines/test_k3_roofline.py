"""CPU tests for operations/kernels/k3_roofline.py."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3_roofline as R  # noqa: E402


def test_weights_match_checkpoint():
    w = R.weights_gb()
    assert abs(w["routed_experts"] + w["dense_bf16"] - 1561) / 1561 < 0.02   # measured total_size 1,560.9 GB


def test_floor_grows_with_seats_and_per_user_falls():
    rows = [R.or_floor("MI355X", s, 71_733, 2.3, R.OR_NEW_PER_OUT) for s in (16, 32, 64, 96, 160)]
    assert all(a["floor_tok_s"] < b["floor_tok_s"] for a, b in zip(rows, rows[1:]))
    assert all(a["floor_per_user_tok_s"] > b["floor_per_user_tok_s"] for a, b in zip(rows, rows[1:]))
    assert rows[3]["experts_touched"] > 0.99      # 96 x 8 x 16 routes over 896 experts


def test_calibration_is_an_efficiency():
    c = R.calibrate()
    assert 0.2 < c["efficiency"] < 0.6 and c["measured_tok_s"] == 920
