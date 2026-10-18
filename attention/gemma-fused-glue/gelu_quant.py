"""k3amd G1a: one-pass gelu_tanh(gate) * up + per-token-group-128 FP8 quant for Gemma 4's MLP (installed into the
hooked SGLang tree as srt/layers/numinous_gemma_gelu_quant.py by operations/patches/gemma31_mlp_hooks.py).

gelu_tanh_mul_quant_fp8(gate_up [T, 2H] bf16) -> (q [T, H] e4m3, s [T, H/128] fp32, column-major TMA-aligned):
exactly what sglang_per_token_group_quant_fp8(gelu_tanh_and_mul(gate_up), 128, column_major_scales=True,
scale_tma_aligned=True) returns on SM90 (bit-identical by construction: see gelu_tanh_quant.cuh), from one kernel that
reads gate_up once and never writes the bf16 activation. The pair feeds the FP8 down projection's pre-quantized path
(Fp8LinearMethod.apply with a tuple -> deepgemm_w8a8_block_fp8_linear_with_fallback(input_scale=s)).
"""
from __future__ import annotations

import os
import pathlib

import torch

GROUP = 128
_CUH = "numinous_gemma_gelu_tanh_quant.cuh"      # installed next to this module by the hook


def _cuh_path() -> str:
    here = pathlib.Path(__file__).resolve().parent
    for p in (here / _CUH, here / "gelu_tanh_quant.cuh"):
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"{_CUH} not found next to {__file__}")


_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit
        from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
        pdl = "true" if is_arch_support_pdl() else "false"
        _MODULE = load_jit(
            "numinous_gemma_gelu_tanh_quant", pdl,
            cuda_files=[_cuh_path()],
            cuda_wrappers=[("run", f"numinous_gemma::GeluTanhQuantFlatKernel<{pdl}>::run")],
            # the stock activation and quant JIT modules are both built with fast math on SM90
            # (kernels/ops/activation/activation.py _fast_math_flags, per_token_group_quant.py)
            extra_cuda_cflags=["--use_fast_math"],
            extra_include_paths=[str(KERNEL_PATH / "csrc")],
        )
    return _MODULE


def scale_buffer(tokens: int, hidden: int, device) -> torch.Tensor:
    """The scale layout of sglang_per_token_group_quant_fp8(column_major_scales=True, scale_tma_aligned=True):
    [T, H/128] fp32 viewed column-major with the token axis padded to a multiple of 4 (TMA alignment)."""
    aligned = (tokens + 3) // 4 * 4
    return torch.empty((hidden // GROUP, aligned), device=device, dtype=torch.float32).transpose(0, 1)[:tokens, :]


def gelu_tanh_mul_quant_fp8(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert gate_up.dim() == 2 and gate_up.dtype == torch.bfloat16 and gate_up.is_contiguous()
    tokens, h2 = gate_up.shape
    hidden = h2 // 2
    assert hidden % GROUP == 0
    q = torch.empty((tokens, hidden), device=gate_up.device, dtype=torch.float8_e4m3fn)
    s = scale_buffer(tokens, hidden, gate_up.device)
    if tokens:
        _module().run(gate_up, q, s)
    return q, s


def reference(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The stock two-step semantics in torch (CPU-runnable; exact math, no fast-math approximations): bf16
    activation with one rounding, then per-group amax in bf16, raw scale amax / 448, quant scale 448 / amax, clamp,
    round-to-nearest e4m3. Used by the CPU tests; the GPU check compares against the real stock kernels."""
    x = gate_up.float()
    hidden = x.shape[1] // 2
    g, u = x[:, :hidden], x[:, hidden:]
    act = (g * (0.5 * (1.0 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))) * u).to(torch.bfloat16)
    a = act.float().view(act.shape[0], hidden // GROUP, GROUP)
    amax = a.abs().amax(-1).clamp_min(1e-10)
    raw = amax * (1.0 / 448.0)
    qs = (448.0 / amax).unsqueeze(-1)
    q = torch.clamp(a * qs, max=448.0).clamp(min=-448.0).view_as(act).to(torch.float8_e4m3fn)
    return q, raw


def enabled() -> bool:
    return os.environ.get("NUMINOUS_GEMMA_GELU_QUANT") == "1"


def selftest(hidden: int, device, tokens: int = 37) -> tuple:
    """First-call check in the serving process (the hook runs it once, outside any CUDA-graph capture): the fused
    kernel against the stock gelu_tanh_and_mul + sglang_per_token_group_quant_fp8 on a random batch. Expected
    bit-identical; accepted within a hair (<= 1 in 10^4 q bytes, scales rel <= 1e-6, dequant rel <= 1e-3) so a
    fast-math corner does not cost the lever, and refused beyond that. Returns (ok, info)."""
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
    from sglang.srt.layers.activation import gelu_tanh_and_mul
    g = torch.Generator(device=device).manual_seed(4321)
    gu = (torch.randn(tokens, 2 * hidden, device=device, generator=g) * 2).to(torch.bfloat16)
    q0, s0 = sglang_per_token_group_quant_fp8(gelu_tanh_and_mul(gu), GROUP, column_major_scales=True,
                                              scale_tma_aligned=True)
    q1, s1 = gelu_tanh_mul_quant_fp8(gu)
    return compare(q0, s0, q1, s1)


def compare(q0, s0, q1, s1, max_frac: float = 1e-4, srel_tol: float = 1e-6, rel_tol: float = 1e-3) -> tuple:
    t = q0.shape[0]
    diff = int((q0.view(torch.uint8) != q1.view(torch.uint8)).sum())
    srel = float(((s0 - s1).abs() / s0.abs().clamp_min(1e-30)).max())
    d0 = q0.float().view(t, -1, GROUP) * s0.unsqueeze(-1)
    d1 = q1.float().view(t, -1, GROUP) * s1.unsqueeze(-1)
    rel = float((d0 - d1).norm() / (d0.norm() + 1e-30))
    ok = diff <= max(0, int(q0.numel() * max_frac)) and srel <= srel_tol and rel <= rel_tol
    return ok, f"q bytes differing {diff}/{q0.numel()}, scale max rel {srel:.2e}, dequant rel {rel:.2e}"
