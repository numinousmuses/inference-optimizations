import glmflash_roofline as r


def test_touched_saturates():
    assert abs(r.touched_experts(1) - r.K) < 1e-9
    assert r.touched_experts(384) > 0.999 * r.E


def test_moe_dominates_tp1_and_halves_at_tp2():
    f1, f2 = r.decode_floor(tp=1, batch=64), r.decode_floor(tp=2, batch=128)
    assert 19 < f1["moe"] < 24                    # 42 x 289 x 25.2M x 0.5625 B / 8 TB/s
    assert abs(f2["moe"] - f1["moe"] / 2) < 0.2
    assert f1["moe"] / f1["total"] > 0.85


def test_kda_state_floor_vs_measured():
    # decode-only profile (run2, 64 per replica): KDA decode update 9.8 ms; state floor at bf16 ~1.1 ms
    f = r.decode_floor(tp=1, batch=64)
    assert 0.9 < f["kda_state"] < 1.3


def test_h200_marlin_moe_is_bandwidth_bound_in_decode_and_compute_bound_in_prefill():
    d = r.decode_floor(tp=4, batch=32, verify=6, context=30000, sku="h200")
    # every expert touched at 192 tokens: 42 x 289 x 25.2M x 0.5625 B / 4 / 4.8 TB/s = 8.9 ms
    assert 8.5 < d["moe"] < 9.3
    p = r.prefill_floor(tokens=16384, tp=4, context=30000, sku="h200")
    # 2 x 16384 x 42 x 9 x 25.2M / 4 / 0.989 PFLOP/s (Marlin W4A16 computes at BF16) = 79 ms
    assert 75 < p["moe"] < 83
    assert p["moe"] / p["total"] > 0.5
    assert 25_000 < p["tok_s_per_gpu_bound"] < 35_000


def test_default_sku_unchanged():
    assert r.decode_floor(tp=1, batch=64) == r.decode_floor(tp=1, batch=64, sku="b300")
