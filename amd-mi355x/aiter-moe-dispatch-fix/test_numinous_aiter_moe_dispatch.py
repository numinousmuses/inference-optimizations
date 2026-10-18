"""CPU checks for numinous_aiter_moe_dispatch: metadata built only for (cktile_*, opus a8w4) rows, with the CK-Tile
stage-1 partial and the OPUS stage-2 partial wired the way fused_moe's own branches wire them (k3amd)."""
import types

import numinous_aiter_moe_dispatch as nd


def _fake_fm():
    fm = types.SimpleNamespace()

    class MOEMetadata:
        def __init__(self, stage1, stage2, block_m, ksplit, **kw):
            self.stage1, self.stage2, self.block_m, self.ksplit, self.kw = stage1, stage2, block_m, ksplit, kw

    def cktile_moe_stage1(*a, **k):
        return "s1"

    opus = types.SimpleNamespace(
        is_opus_a8w4_stage2_kernel=lambda n: n.startswith("opus_moe2_afp8_wfp4"),
        parse_stage2_config=lambda cfg, bm: ("launch", bm),
        opus_a8w4_stage2_wrapper=lambda *a, **k: "s2",
    )
    fm.MOEMetadata, fm.cktile_moe_stage1, fm._opus_a8w4 = MOEMetadata, cktile_moe_stage1, opus
    return fm


def test_builds_for_cktile_opus_row():
    fm = _fake_fm()
    cfg = {"kernelName1": "cktile_a8w4_bm64", "kernelName2": "opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3584",
           "block_m": 64, "ksplit": 0}
    md = nd.build_metadata(fm, cfg, activation="silu", dtype="bf16", use_g1u1=True, hidden_pad=0, intermediate_pad=0)
    assert md is not None and md.block_m == 64 and md.ksplit == 0
    assert md.stage1.func is fm.cktile_moe_stage1 and md.stage1.keywords["split_k"] == 1
    assert md.stage2.keywords["kernelName"] == cfg["kernelName2"] and md.stage2.keywords["launch"] == ("launch", 64)
    assert md.kw["fuse_quant"] == "" and md.kw["stage2_has_bias"] is False


def test_skips_other_rows():
    fm = _fake_fm()
    for kn1, kn2 in [("flydsl_moe1_afp8_wfp4_bf16_t32x64x256", "opus_moe2_afp8_wfp4_x"),
                     ("cktile_a8w4_bm64", "cktile_a8w4_bm64"), ("cktile_a8w4_bm64", "flydsl_moe2_layout_x")]:
        assert nd.build_metadata(fm, {"kernelName1": kn1, "kernelName2": kn2, "block_m": 64}, activation="silu",
                                 dtype="bf16", use_g1u1=True, hidden_pad=0, intermediate_pad=0) is None


def test_install_inert_without_env(monkeypatch):
    monkeypatch.delenv(nd.ENV, raising=False)
    assert nd.install() is False
