"""CPU tests for k27_fp4_convert: INT4 unpacking against a hand packing, E2M1 rounding (ties to even), pack/unpack
round trips, exactness when an INT4 block fits the E2M1 grid, MSE-search never worse than the amax rule, and the
streaming convert on a synthetic two-expert checkpoint (index, config, hf_quant_config, borrowed input scales).
    python3 -m pytest operations/kernels/test_k27_fp4_convert.py -q
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import k27_fp4_convert as k  # noqa: E402

torch.manual_seed(0)


def _pack_int4(q: torch.Tensor) -> torch.Tensor:
    u = q.int() + 8
    packed = torch.zeros((q.shape[0], q.shape[1] // 8), dtype=torch.int32)
    for i in range(8):
        packed |= u[:, i::8] << (4 * i)
    return packed


def test_unpack_int4_matches_hand_packing():
    q = torch.randint(-8, 8, (6, 128), dtype=torch.int8)
    assert torch.equal(k.unpack_int4(_pack_int4(q), (6, 128)), q)


def test_dequant_int4_applies_group_scale():
    q = torch.randint(-8, 8, (2, 64), dtype=torch.int8)
    s = torch.tensor([[0.5, 2.0], [1.0, 0.25]], dtype=torch.bfloat16)
    w = k.dequant_int4(_pack_int4(q), s, (2, 64))
    assert torch.allclose(w[0, :32], q[0, :32].float() * 0.5) and torch.allclose(w[1, 32:], q[1, 32:].float() * 0.25)


def test_e2m1_round_to_nearest_even_and_saturate():
    x = torch.tensor([[0.0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -6, 0.24, 0.26, 0.75, 1.25, 5.0, 7.0, -2.5, -3.5]])
    want = torch.tensor([[0.0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -6, 0.0, 0.5, 1.0, 1.0, 4.0, 6.0, -2.0, -4.0]])
    assert torch.equal(k.e2m1_decode(k.e2m1_encode(x)), want)
    assert int(k.e2m1_encode(torch.tensor([[-0.1]]))[0, 0]) == 0   # no negative zero


def test_nibble_pack_roundtrip():
    codes = torch.randint(0, 16, (3, 32), dtype=torch.uint8)
    assert torch.equal(k.unpack_nibbles(k.pack_nibbles(codes)), codes)
    assert int(k.pack_nibbles(torch.tensor([[1, 2]], dtype=torch.uint8))[0, 0]) == 0x21   # low nibble = even column


def test_e8m0_roundtrip():
    x = torch.exp2(torch.arange(-20, 20).float())
    assert torch.equal(k.from_e8m0(k.to_e8m0(x)), x)


def test_nvfp4_exact_when_int4_block_fits_grid():
    # codes in {0, +-1, +-2, +-3, +-4, +-6} at scale s are E2M1 x s exactly
    q = torch.tensor([[0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 0, 1, 2, 3, 4]], dtype=torch.float32) * 0.013
    w, sc, g = k.quant_nvfp4(q, search=True)
    rec = k.dequant_nvfp4(w, sc, g)
    assert k.compare(q, rec)["rel_rms_w"] < 3e-3   # E4M3 scale rounding only


def test_search_never_worse_than_amax_rule():
    w = torch.randint(-8, 8, (64, 512)).float() * torch.rand(64, 16).repeat_interleave(32, 1) * 0.01
    e_amax = k.compare(w, k.dequant_nvfp4(*k.quant_nvfp4(w, search=False)))["rel_rms_w"]
    e_srch = k.compare(w, k.dequant_nvfp4(*k.quant_nvfp4(w, search=True)))["rel_rms_w"]
    assert e_srch <= e_amax + 1e-6
    m_pow2 = k.compare(w, k.dequant_mxfp4(*k.quant_mxfp4(w, search=False)))["rel_rms_w"]
    m_srch = k.compare(w, k.dequant_mxfp4(*k.quant_mxfp4(w, search=True)))["rel_rms_w"]
    assert m_srch <= m_pow2 + 1e-6
    assert e_srch < 0.2 and m_srch < 0.3   # INT4 grid -> FP4: bounded, not exact


def test_nvfp4_shapes_and_dtypes():
    w = torch.randn(8, 256)
    packed, sc, g = k.quant_nvfp4(w)
    assert packed.shape == (8, 128) and packed.dtype == torch.uint8
    assert sc.shape == (8, 16) and sc.dtype == torch.float8_e4m3fn and g.dtype == torch.float32
    packed, sc = k.quant_mxfp4(w)
    assert packed.shape == (8, 128) and sc.shape == (8, 8) and sc.dtype == torch.uint8


def _synthetic_checkpoint(tmp_path, n_layers=2, n_experts=2, out=64, hidden=128):
    from safetensors.torch import save_file
    src = tmp_path / "src"
    src.mkdir()
    wmap = {}
    for layer in range(n_layers):
        tensors = {f"language_model.model.layers.{layer}.input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16)}
        for e in range(n_experts):
            for proj, shape in (("gate_proj", (out, hidden)), ("up_proj", (out, hidden)), ("down_proj", (hidden, out))):
                p = f"language_model.model.layers.{layer}.mlp.experts.{e}.{proj}"
                q = torch.randint(-8, 8, shape, dtype=torch.int8)
                tensors[p + ".weight_packed"] = _pack_int4(q)
                tensors[p + ".weight_scale"] = (torch.rand(shape[0], shape[1] // 32) * 0.02 + 0.001).to(torch.bfloat16)
                tensors[p + ".weight_shape"] = torch.tensor(shape, dtype=torch.int32)
        sh = f"model-{layer + 1:05d}-of-{n_layers:05d}.safetensors"
        save_file(tensors, str(src / sh), metadata={"format": "pt"})
        wmap.update({kk: sh for kk in tensors})
    (src / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    (src / "config.json").write_text(json.dumps({"architectures": ["KimiK25ForConditionalGeneration"], "text_config": {
        "num_hidden_layers": n_layers, "first_k_dense_replace": 1, "quantization_config": {"quant_method": "compressed-tensors"}}}))
    (src / "tokenizer_config.json").write_text("{}")
    # a "calibrated" checkpoint to borrow input scales from
    cal = tmp_path / "cal"
    cal.mkdir()
    scales = {f"language_model.model.layers.{layer}.mlp.experts.{e}.{proj}.input_scale": torch.tensor(0.5 + e, dtype=torch.float32)
              for layer in range(n_layers) for e in range(n_experts) for proj in ("gate_proj", "up_proj", "down_proj")}
    save_file(scales, str(cal / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (cal / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {kk: "model-00001-of-00001.safetensors" for kk in scales}}))
    return src, cal


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_convert_streams_checkpoint(tmp_path, fmt):
    from safetensors import safe_open
    src, cal = _synthetic_checkpoint(tmp_path)
    dst = tmp_path / fmt
    stats = k.convert(str(src), str(dst), fmt, search=True, input_scales_from=str(cal) if fmt == "nvfp4" else "")
    assert stats["experts"] == 2 * 2 * 3 and stats["shards"] == 2 and stats["mean_rel_rms_w"] < 0.3
    idx = json.load(open(dst / "model.safetensors.index.json"))["weight_map"]
    p = "language_model.model.layers.1.mlp.experts.1.gate_proj"
    assert p + ".weight" in idx and p + ".weight_scale" in idx and p + ".weight_packed" not in idx
    assert "language_model.model.layers.1.input_layernorm.weight" in idx
    with safe_open(str(dst / idx[p + ".weight"]), "pt") as f:
        w = f.get_tensor(p + ".weight")
        assert w.dtype == torch.uint8 and w.shape == (64, 64)
        if fmt == "nvfp4":
            assert f.get_tensor(p + ".weight_scale").dtype == torch.float8_e4m3fn
            assert float(f.get_tensor(p + ".input_scale")) == 1.5 and f.get_tensor(p + ".weight_scale_2").ndim == 0
        else:
            assert f.get_tensor(p + ".weight_scale").shape == (64, 4)
    cfg = json.load(open(dst / "config.json"))
    if fmt == "nvfp4":
        hq = json.load(open(dst / "hf_quant_config.json"))["quantization"]
        assert hq["quant_algo"] == "NVFP4" and "language_model.model.layers.0*" in hq["exclude_modules"]
        assert "language_model.model.layers.1.self_attn*" in hq["exclude_modules"]
        assert "quantization_config" not in cfg["text_config"]
        assert cfg["quantization_config"]["quant_method"] == "modelopt" and cfg["quantization_config"]["quant_algo"] == "NVFP4"
    else:
        assert cfg["text_config"]["quantization_config"]["quant_method"] == "mxfp4"
    assert (dst / "tokenizer_config.json").exists() and (dst / "numinous_convert.json").exists()
