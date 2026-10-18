"""k3amd G1b: RMSNorm (+ residual add) + per-token-group-128 FP8 quant for Gemma 4's decoder layer (installed into
the hooked SGLang tree as srt/layers/numinous_gemma_norm_quant.py by operations/patches/gemma31_norm_quant_hooks.py).

rmsnorm_quant_fp8(x, w, eps) -> (q, s): the pair sglang_per_token_group_quant_fp8(rmsnorm(x, w, eps), 128,
column_major_scales=True, scale_tma_aligned=True) would give, from one kernel that never writes the bf16 norm output.
fused_add_rmsnorm_quant_fp8(x, residual, w, eps) -> (q, s); residual <- x + residual (bf16) in place, as
fused_add_rmsnorm does. Accuracy contract in norm_quant.cuh (not bit-exact with the flashinfer norm).
"""
from __future__ import annotations

import pathlib

import torch

GROUP = 128
_CUH = "numinous_gemma_norm_quant.cuh"
JIT_MODULES = ["_module_plain", "_module_add"]


def _cuh_path() -> str:
    here = pathlib.Path(__file__).resolve().parent
    for p in (here / _CUH, here / "norm_quant.cuh"):
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"{_CUH} not found next to {__file__}")


_MODULES: dict = {}


def _module(add: bool):
    if add not in _MODULES:
        from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit
        from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
        pdl = "true" if is_arch_support_pdl() else "false"
        a = "true" if add else "false"
        _MODULES[add] = load_jit(
            "numinous_gemma_norm_quant", a, pdl,
            cuda_files=[_cuh_path()],
            cuda_wrappers=[("run", f"numinous_gemma::RmsNormQuantKernel<{a}, {pdl}>::run")],
            extra_cuda_cflags=["--use_fast_math"],
            extra_include_paths=[str(KERNEL_PATH / "csrc")],
        )
    return _MODULES[add]


def _module_plain():
    return _module(False)


def _module_add():
    return _module(True)


def scale_buffer(tokens: int, hidden: int, device) -> torch.Tensor:
    aligned = (tokens + 3) // 4 * 4
    return torch.empty((hidden // GROUP, aligned), device=device, dtype=torch.float32).transpose(0, 1)[:tokens, :]


def supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    h = x.shape[-1]
    return (x.dim() == 2 and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and x.stride(-1) == 1
            and h % 256 == 0 and h <= 8192 and x.shape[0] > 0)


def rmsnorm_quant_fp8(x: torch.Tensor, weight: torch.Tensor, eps: float):
    t, h = x.shape
    q = torch.empty((t, h), device=x.device, dtype=torch.float8_e4m3fn)
    s = scale_buffer(t, h, x.device)
    _module(False).run(x, x, weight, q, s, float(eps))
    return q, s


def fused_add_rmsnorm_quant_fp8(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    t, h = x.shape
    q = torch.empty((t, h), device=x.device, dtype=torch.float8_e4m3fn)
    s = scale_buffer(t, h, x.device)
    _module(True).run(x, residual, weight, q, s, float(eps))
    return q, s


def reference(x, weight, eps, residual=None):
    """CPU semantics of the kernel (exact fp32 math): optional fp32 add (residual stored bf16), fp32 row mean of
    squares, bf16(x * rsqrt(ms + eps) * w), then the stock per_token_group_quant formulas. Returns
    (q, scale, residual_out, normed_bf16)."""
    xf = x.float()
    res_out = None
    if residual is not None:
        xf = xf + residual.float()
        res_out = xf.to(torch.bfloat16)
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = (xf * rms * weight.float()).to(torch.bfloat16)
    g = y.float().view(y.shape[0], -1, GROUP)
    amax = g.abs().amax(-1).clamp_min(1e-10)
    q = torch.clamp(g * (448.0 / amax).unsqueeze(-1), min=-448.0, max=448.0).view_as(y).to(torch.float8_e4m3fn)
    return q, amax * (1.0 / 448.0), res_out, y


def selftest(norm, device, tokens: int = 37) -> tuple:
    """First-call check in the serving process (once, outside capture): the fused norm (+ add) + quant against the
    stock norm module's own forward + sglang_per_token_group_quant_fp8 on a random batch, both variants. Accepted if
    the dequantized activations agree within rel 2e-3 and the residual stream is bit-identical; refused otherwise.
    Returns (ok, info)."""
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
    h = norm.weight.shape[0]
    g = torch.Generator(device=device).manual_seed(8765)
    x = (torch.randn(tokens, h, device=device, generator=g) * 3).to(torch.bfloat16)
    r = torch.randn(tokens, h, device=device, generator=g).to(torch.bfloat16)
    eps = norm.variance_epsilon
    infos, ok = [], True
    q0, s0 = sglang_per_token_group_quant_fp8(norm(x.clone()), GROUP, column_major_scales=True, scale_tma_aligned=True)
    q1, s1 = rmsnorm_quant_fp8(x, norm.weight, eps)
    a_ok, a_info = _rel(q0, s0, q1, s1)
    ok &= a_ok
    infos.append(f"plain {a_info}")
    xs, rs = x.clone(), r.clone()
    y, r0 = norm(xs, rs)
    q0, s0 = sglang_per_token_group_quant_fp8(y, GROUP, column_major_scales=True, scale_tma_aligned=True)
    r1 = r.clone()
    q1, s1 = fused_add_rmsnorm_quant_fp8(x, r1, norm.weight, eps)
    b_ok, b_info = _rel(q0, s0, q1, s1)
    res_eq = bool(torch.equal(r0, r1))
    ok &= b_ok and res_eq
    infos.append(f"add {b_info}, residual equal {res_eq}")
    return bool(ok), "; ".join(infos)


def _rel(q0, s0, q1, s1, tol: float = 2e-3) -> tuple:
    t = q0.shape[0]
    d0 = q0.float().view(t, -1, GROUP) * s0.unsqueeze(-1)
    d1 = q1.float().view(t, -1, GROUP) * s1.unsqueeze(-1)
    rel = float((d0 - d1).norm() / (d0.norm() + 1e-30))
    diff = int((q0.view(torch.uint8) != q1.view(torch.uint8)).sum())
    return rel <= tol, f"dequant rel {rel:.2e}, q bytes differing {diff}/{q0.numel()}"
