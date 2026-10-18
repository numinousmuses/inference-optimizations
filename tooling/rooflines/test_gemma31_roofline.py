"""Tests for gemma31_roofline (agent rtxv41): config-exact sizes, the v0.5.20 pool formulas, and the calibration
reproducing the measured August H100 rows.
    pytest -p no:cacheprovider operations/kernels/test_gemma31_roofline.py
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gemma31_roofline as g  # noqa: E402

M = g.Gemma31()


def test_params_match_the_checkpoint():
    # RedHatAI/gemma-4-31B-it-FP8-block model.safetensors.index.json total_size = 33,263,025,112 B (measured):
    # FP8 linears + BF16 tied embedding + the BF16 vision tower and scales.
    text = M.linear_params() * 1.0 + M.embed_params() * 2.0
    assert 32.0e9 < text < 33.263e9
    assert 33.263e9 - text < 1.2e9            # what remains is the vision tower + block scales + norms
    assert M.linear_params() + M.embed_params() == pytest.approx(30.8e9, rel=0.01)   # "31B"


def test_kv_rows_follow_sglang_gemma4_mapping():
    # utils/hf_transformers/config.py:212-239: full = 4 KV heads x 512, SWA = 16 KV heads x 256; K and V rows, FP8.
    assert M.kv_full_per_token_layer() == 4 * 1024
    assert M.kv_swa_per_token_layer() == 16 * 512
    assert M.kv_full_length_per_token() == 50 * 8192 + 10 * 4096 == 450_560


def test_hybrid_cell_is_the_configurator_formula():
    # pool_configurator.py:664-691: F x n_full + ratio x S x n_swa
    assert M.hybrid_cell(0.8) == 10 * 4096 + 0.8 * 50 * 8192 == 368_640
    assert M.hybrid_cell(0.35) == 10 * 4096 + 0.35 * 50 * 8192
    assert M.hybrid_cell(1.0) == M.kv_full_length_per_token()


def test_august_pool_is_explained_by_full_length_kv():
    """The August std arm reported a 100,120-token pool at mem 0.85. Full-length rows (no hybrid pool) need ~45 GB for
    it, hybrid at the stock 0.8 ratio ~36.9 GB; with 33.3 GB of weights the 0.85 budget is ~38.7 GB, so the measured
    pool is consistent with the hybrid pool at ratio 0.8 or with full-length rows on a larger budget: either way the
    ratio, not the weights, set it."""
    budget = g.kv_budget(33.263e9, mem_fraction=0.85, draft_bytes=0)
    assert 36e9 < budget < 41e9
    assert budget / M.hybrid_cell(0.8) == pytest.approx(100_120, rel=0.12)


def test_pool_grows_as_ratio_falls_and_holds_the_or_mix():
    budget = g.kv_budget(33.263e9)
    pools = [g.pool(M, budget, r) for r in (0.8, 0.5, 0.35, 0.25)]
    fulls = [p["full_tokens"] for p in pools]
    assert fulls == sorted(fulls)
    p35 = g.pool(M, budget, 0.35)
    n = g.concurrency(M, p35, g.OR_PROMPT + g.OR_OUTPUT, M.window + 8, 8192)
    n80 = g.concurrency(M, pools[0], g.OR_PROMPT + g.OR_OUTPUT, M.window + 8, 8192)
    assert n >= 60 and n > n80
    # at ratio 0.35 neither pool is idle: SWA- and full-bound concurrencies within 30% of each other
    full_bound = p35["full_tokens"] // (g.OR_PROMPT + g.OR_OUTPUT)
    swa_bound = (p35["swa_tokens"] - 8192) // (M.window + 8)
    assert abs(full_bound - swa_bound) / max(full_bound, swa_bound) < 0.3


def test_calibration_reproduces_the_august_rows():
    c = g.calibrate()
    ctx = 8192 + 400
    std_per_user = 1.0 / g.cycle_time(M, c, 10, ctx, 0)
    assert std_per_user == pytest.approx(45.84, rel=0.01)
    mtp_per_user = 5.44 / g.cycle_time(M, c, 9, ctx, 5)
    assert mtp_per_user == pytest.approx(68.77, rel=0.01)
    assert 1.0 / g.prefill_token_time(M, c, 8192) == pytest.approx(7100, rel=0.01)
    assert 0.15 < c.gemm_eff < 0.4          # the August image's FP8 GEMMs ran far below DeepGEMM rates


def test_or_shape_model_monotonic_and_bounded():
    c = g.nominal_v0520(g.calibrate())
    rows = [g.or_shape(M, c, n, 3.5, 5) for n in (8, 16, 32, 48)]
    warm = [r["warm_tok_s"] for r in rows]
    per_user = [r["per_user"] for r in rows]
    assert warm == sorted(warm) and per_user == sorted(per_user, reverse=True)
    # the warm rate can never exceed the prefill-only bound (uncached prompt tokens per output token)
    pf_bound = rows[-1]["prefill_tok_s"] / (g.OR_PROMPT * (1 - g.OR_CACHE_HIT) / g.OR_OUTPUT)
    assert all(w < pf_bound for w in warm)


def test_cli_runs(tmp_path):
    out = tmp_path / "r.json"
    assert g.main(["--json", str(out)]) == 0
    assert out.exists() and "or_rows" in out.read_text()


def test_weight_formats_at_the_or_mix():
    """W4A16 loses at the operating point (BF16 MMA for prefill and a compute-bound verify); W4A8 only pays at low
    concurrency, where verify is weight-bandwidth-bound."""
    c = g.nominal_v0520(g.calibrate())
    fp8 = {n: g.or_shape(M, c, n, 3.5, 5)["warm_tok_s"] for n in (8, 32, 48)}
    w4a16 = {n: g.or_shape(M, c, n, 3.5, 5, weights="w4a16")["warm_tok_s"] for n in (8, 32, 48)}
    w4a8 = {n: g.or_shape(M, c, n, 3.5, 5, weights="w4a8")["warm_tok_s"] for n in (8, 32, 48)}
    assert all(w4a16[n] < 0.9 * fp8[n] for n in (32, 48))
    assert w4a8[8] > fp8[8] and w4a8[48] <= fp8[48]
    # 4-bit KV: a few percent at most once the pool is not the limit
    kv4 = g.or_shape(M, c, 48, 3.5, 5, kv_bytes=0.5)["warm_tok_s"]
    assert fp8[48] < kv4 < 1.10 * fp8[48]
