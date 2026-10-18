"""CPU tests for megamoe_bench's reference helpers (the correctness oracle of the MegaMoE benchmark).

The GPU side (8-rank kernel vs reference) runs through operations/patches/test_deepgemm_megamoe_hooks.py and the
bench itself on the box.
"""
from __future__ import annotations

import os
import pathlib
import sys
import types

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.modules.setdefault("flashinfer", types.ModuleType("flashinfer"))  # the bench imports it first on GPUs

import megamoe_bench as mb  # noqa: E402

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_fp4_decode_table_and_sign():
    codes = torch.arange(16, dtype=torch.int8)
    vals = mb.fp4_decode(codes)
    assert vals[:8].tolist() == E2M1
    assert vals[8:].tolist() == [-v for v in E2M1]


def test_dequant_packed_low_nibble_first_and_scales():
    packed = torch.tensor([[0x21, 0x7F]], dtype=torch.int8)  # elements: 1, 2, 15, 7 -> 0.5, 1.0, -6.0, 6.0
    sf = torch.tensor([[0.25]])
    out = mb.dequant_fp4_packed(packed, sf, gran=4)
    assert out.tolist() == [[0.125, 0.25, -1.5, 1.5]]


def test_ue8m0_ceil_is_power_of_two_at_or_above():
    x = torch.tensor([1.0, 1.5, 3.0, 0.3, 448.0])
    y = mb.ue8m0_ceil(x)
    assert torch.all(y >= x)
    assert torch.all(torch.log2(y) == torch.round(torch.log2(y)))
    assert y.tolist()[:3] == [1.0, 2.0, 4.0]


@pytest.mark.parametrize("kind,grid_max", [("fp8", 448.0), ("fp4", 6.0)])
def test_fake_quant_is_idempotent_and_bounded(kind, grid_max):
    torch.manual_seed(0)
    x = torch.randn(4, 64) * 3
    q = mb.fake_quant(x, kind)
    assert torch.equal(mb.fake_quant(q, kind), q)  # values on the grid stay put
    err = (q - x).abs().max() / x.abs().max()
    assert err < (0.07 if kind == "fp8" else 0.35)
    v = x.view(4, 2, 32)
    sf = mb.ue8m0_ceil(v.abs().amax(-1) / grid_max)
    assert torch.all(q.view(4, 2, 32).abs() <= sf.unsqueeze(-1) * grid_max + 1e-6)


def test_fp4_fake_quant_rounds_to_e2m1_grid():
    x = torch.tensor([[6.0, 2.4, 2.6, 0.2, 0.3, -5.1] + [0.0] * 26])  # amax 6 -> scale 1
    q = mb.fake_quant(x, "fp4")
    assert q[0, :6].tolist() == [6.0, 2.0, 3.0, 0.0, 0.5, -6.0]


def test_calc_diff():
    a = torch.randn(10, 10)
    assert mb.calc_diff(a, a) == pytest.approx(0.0, abs=1e-12)
    assert mb.calc_diff(a, -a) == pytest.approx(2.0)


def test_apply_config_sets_and_clears_overrides(monkeypatch):
    for k in mb.CONFIG_KEYS:
        monkeypatch.delenv(f"DG_MEGA_MOE_{k}", raising=False)
    mb.apply_config("BLOCK_K=256,MAX_STAGES=8")
    assert os.environ["DG_MEGA_MOE_BLOCK_K"] == "256" and os.environ["DG_MEGA_MOE_MAX_STAGES"] == "8"
    mb.apply_config("default")
    assert not any(f"DG_MEGA_MOE_{k}" in os.environ for k in mb.CONFIG_KEYS)
    with pytest.raises(AssertionError):
        mb.apply_config("NUM_SMS=10")


def test_touched_bytes_counts_unique_local_experts():
    idx = torch.tensor([[0, 1, 47, 48], [1, 2, 95, -1]])
    n, b = mb.touched_bytes(idx, rank=0, nl=48)
    assert n == 4  # experts 0, 1, 2, 47
    assert b == 4 * (3 * mb.INTER * mb.HIDDEN // 2 + 3 * mb.INTER * mb.HIDDEN // 32)


def test_wait_decomposition_on_the_recorded_megamoe_profile():
    import json

    import megamoe_wait_decomposition as wd
    gate = HERE.parents[1] / "research/gpu-evidence/gate/deepseek-v4-pro_standard_reg_chunk_32k+megamoe_20260927-115148.json"
    out = wd.decompose(json.load(open(gate)))
    decode = [r for r in out["rows"] if r["block_m"] == 32][0]
    assert decode["calls"] == 14976 and abs(decode["us_per_call"] - 1879.9) < 1
    assert out["wait_share_lower"] > 0.8
    assert abs(out["per_rank_step_ms"] - 159.8) < 0.5
