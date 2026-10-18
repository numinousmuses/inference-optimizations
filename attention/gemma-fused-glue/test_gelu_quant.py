"""CPU tests of the G1a math and layout (the GPU bit-exactness check runs in the bundled H100 microbench):
- the reference fused math equals the stock two-step semantics (bf16 activation with one rounding, then the stock
  per-token-group quant formulas) element for element;
- dequantized q x scale reconstructs the bf16 activation within e4m3 resolution;
- the scale buffer has the layout of sglang_per_token_group_quant_fp8(column_major_scales=True, scale_tma_aligned=True)
  (create_per_token_group_quant_fp8_output_scale, fp8_kernel.py:464-517: [T, H/128] with strides (1, ceil4(T)));
- the .cuh keeps the stock activation constants and the stock quant formulas verbatim."""
import pathlib
import re
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gelu_quant as gq  # noqa: E402

SRC = pathlib.Path("~/.cache/numinous/sgl-v0520/python/sglang").expanduser()


def stock_two_step(gate_up):
    """gelu_tanh_and_mul (act_and_mul_kernel: one rounding) then per_token_group_quant (fp32-scale branch)."""
    x = gate_up.float()
    h = x.shape[1] // 2
    g, u = x[:, :h], x[:, h:]
    gelu = g * (0.5 * (1.0 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))))
    act = (gelu * u).to(torch.bfloat16)
    groups = act.float().view(act.shape[0], h // 128, 128)
    amax = torch.maximum(groups.abs().amax(-1), torch.tensor(1e-10))
    scale = amax * (1.0 / 448.0)
    q = torch.minimum(groups * (448.0 / amax).unsqueeze(-1), torch.tensor(448.0))
    return q.reshape(act.shape).to(torch.float8_e4m3fn), scale, act


@pytest.mark.parametrize("tokens", [1, 3, 64])
def test_reference_equals_stock_two_step(tokens):
    g = torch.Generator().manual_seed(tokens)
    gate_up = (torch.randn(tokens, 2 * 512, generator=g) * 3).to(torch.bfloat16)
    q, s = gq.reference(gate_up)
    q2, s2, act = stock_two_step(gate_up)
    assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s, s2)
    deq = (q.float().view(tokens, 4, 128) * s.unsqueeze(-1)).view(tokens, 512)
    rel = (deq - act.float()).norm() / act.float().norm()
    assert rel < 0.04                                           # e4m3: 3 mantissa bits


@pytest.mark.parametrize("tokens", [1, 2, 5, 288, 8191])
def test_scale_layout_matches_stock(tokens):
    s = gq.scale_buffer(tokens, 21504, "cpu")
    assert tuple(s.shape) == (tokens, 21504 // 128)
    assert s.stride() == (1, (tokens + 3) // 4 * 4)


@pytest.mark.skipif(not SRC.is_dir(), reason="needs the v0.5.20 tree")
def test_cuh_keeps_the_stock_math_verbatim():
    mine = (HERE / "gelu_tanh_quant.cuh").read_text()
    act = (SRC / "kernels/jit/csrc/elementwise/activation.cuh").read_text()
    quant = (SRC / "kernels/jit/csrc/gemm/per_token_group_quant.cuh").read_text()
    for line in ("constexpr auto kGeluTanhAlpha = 0.044715f;", "constexpr auto kGeluTanhBeta = 0.7978845608028654f;",
                 "const float cdf = 0.5f * (1.0f + tanhf(kGeluTanhBeta * (x_f32 + kGeluTanhAlpha * x_f32 * x_f32 * x_f32)));"):
        assert line in mine and line in act
    for line in ("const auto amax = math::max(math::max(amax2.x, amax2.y), 1e-10f);",
                 "const float raw_scale = amax * kMaxValueInv;",
                 "const float quant_scale = kMaxValue / amax;",
                 "out[i] = WTrait::quant(detail::mul2(cast<float2>(in[i]), quant_scale2));",
                 "params.scale.store<kUe8m0, kRowMajor, kAligned>(expert_idx, token_idx, group_idx, scale_inv);"):
        assert line in mine and line in quant, line
    # the stock act_and_mul single rounding: out[i] = cast<T>(apply_activation_f32(gate_f32) * up_f32)
    assert "out[i] = device::cast<T>(apply_activation_f32<kAct>(gate_f32) * up_f32);" in act
    assert re.search(r"cast<T2>\(float2\{gelu_tanh_f32\(gate\.x\) \* upf\.x, gelu_tanh_f32\(gate\.y\) \* upf\.y\}\)", mine)
    # the build flags of both stock modules on SM90
    assert '"--use_fast_math"' in (SRC / "kernels/ops/activation/activation.py").read_text()
    assert '"--use_fast_math"' in (SRC / "kernels/ops/quantization/per_token_group_quant.py").read_text()
