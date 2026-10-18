"""DeepSeek V4 Pro: RMSNorm + FP8 per-token-group quant (UE8M0 scales) in one Triton kernel.

Target: the q path. `q_lora = q_norm(qkv_a[:, :1536])` feeds two FP8 DeepGEMM GEMMs: the attention wq_b
(1536 -> 128 x 512) and the C4 indexer wq_b (1536 -> 64 x 128). In SGLang v0.5.20 that is one RMSNorm launch plus
one `per_token_group_quant` launch inside EACH fp8 linear (the same bf16 tensor quantized twice, on two streams).
This kernel emits the bf16 normed row, the FP8 bytes and the scales in exactly the layout
`deepgemm_w8a8_block_fp8_linear_with_fallback` builds on Blackwell (column-major, TMA-aligned, UE8M0 packed:
int32 [ceil(G/4), align(M, 4)] viewed transposed; byte b of int32 (g // 4, t) is group 4 * (g // 4) + b), so both
GEMMs consume it with no further quant (`fp8_linear_prequant`).

Arithmetic, chosen to match the engine bit for bit:
  RMSNorm   y = bf16(x * rsqrt(sum(x^2) / K + eps) * w)                (fp32 math, like the engine's rmsnorm)
  quant     amax_g = max(|y_g|, 1e-10); e = ceil_pow2_exponent(amax_g * fp32(1/448))
            q = fp8_e4m3(y * 2^-e)   (the product is exact, one rounding, like the engine's bf16 multiply)
The sum-of-squares reduction order differs from the engine's kernel, so rsqrt can differ by an ulp and move a
bf16 value; the GPU test reports the byte mismatch rate (measured 0 on Pro shapes, see agents/status/kernels.md).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

FP8_MAX_INV = 1.0 / 448.0


@triton.jit
def _rmsnorm_fp8_kernel(
    x_ptr, stride_x, w_ptr, y_ptr, stride_y, q_ptr, stride_q, s_ptr, s_mn,
    M, eps,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    G_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    WRITE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    G: tl.constexpr = K // GROUP
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    rows64 = rows.to(tl.int64)
    gi = tl.arange(0, G_PAD)
    ci = tl.arange(0, GROUP)
    gmask = gi < G
    col = gi[:, None] * GROUP + ci[None, :]  # [G_PAD, GROUP]
    m3 = rmask[:, None, None] & gmask[None, :, None]
    x = tl.load(x_ptr + rows64[:, None, None] * stride_x + col[None, :, :], mask=m3, other=0.0).to(tl.float32)
    ss = tl.sum(tl.sum(x * x, axis=2), axis=1)
    r = tl.rsqrt(ss / K + eps)
    w = tl.load(w_ptr + col, mask=gmask[:, None], other=0.0).to(tl.float32)
    y = (x * r[:, None, None] * w[None, :, :]).to(tl.bfloat16)
    if WRITE_Y:
        tl.store(y_ptr + rows64[:, None, None] * stride_y + col[None, :, :], y, mask=m3)
    yf = y.to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(yf), axis=2), 1e-10)  # [ROWS, G_PAD]
    raw = amax * 0.002232142857142857  # fp32(1/448), as the engine multiplies by kMaxValueInv
    bits = raw.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)  # UE8M0 biased exponent (ceil)
    inv = ((254 - e) << 23).to(tl.float32, bitcast=True)  # 2^-(e - 127)
    q = (yf * inv[:, :, None]).to(tl.float8e4nv)
    tl.store(q_ptr + rows64[:, None, None] * stride_q + col[None, :, :], q, mask=m3)
    # packed UE8M0 scales: byte offset (g // 4) * s_mn * 4 + t * 4 + g % 4
    sb = (gi[None, :] // 4).to(tl.int64) * s_mn * 4 + rows64[:, None] * 4 + (gi[None, :] % 4)
    tl.store(s_ptr + sb, e.to(tl.uint8), mask=rmask[:, None] & gmask[None, :])
    if G % 4 != 0:
        # zero the tail bytes of the last packed int32 so garbage never reaches the GEMM
        tail = G + tl.arange(0, 4)
        tb = (tail[None, :] // 4).to(tl.int64) * s_mn * 4 + rows64[:, None] * 4 + (tail[None, :] % 4)
        tl.store(s_ptr + tb, tl.zeros((ROWS, 4), dtype=tl.uint8),
                 mask=rmask[:, None] & (tail[None, :] < ((G + 3) // 4) * 4))


def alloc_ue8m0_scales(m: int, k: int, group: int, device) -> torch.Tensor:
    """The engine's layout (create_per_token_group_quant_fp8_output_scale, ue8m0 + column-major + TMA-aligned)."""
    g = k // group
    aligned_mn = (m + 3) // 4 * 4
    aligned_k = (g + 3) // 4 * 4
    return torch.empty((aligned_k // 4, aligned_mn), device=device, dtype=torch.int32).transpose(0, 1)[:m, :]


def rmsnorm_fp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group: int = 128,
    write_bf16: bool = True,
    rows_per_program: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """x [M, K] bf16 (rows may be strided, last dim contiguous) -> (y bf16 [M, K] or None, q fp8 [M, K],
    scales int32 [M, ceil(K/group/4)] in DeepGEMM's packed UE8M0 layout)."""
    assert x.dim() == 2 and x.stride(-1) == 1 and x.shape[1] % group == 0
    M, K = x.shape
    dev = x.device
    y = torch.empty((M, K), dtype=torch.bfloat16, device=dev) if write_bf16 else None
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    s = alloc_ue8m0_scales(M, K, group, dev)
    if M == 0:
        return y, q, s
    rows = rows_per_program or (1 if M <= 256 else 2)
    G = K // group
    s_mn = s.stride(1)  # aligned_mn (int32 units)
    _rmsnorm_fp8_kernel[(triton.cdiv(M, rows),)](
        x, x.stride(0), weight, y if y is not None else q, y.stride(0) if y is not None else 0, q, q.stride(0),
        _bytes(s), s_mn, M, float(eps),
        K=K, GROUP=group, G_PAD=triton.next_power_of_2(G), ROWS=rows, WRITE_Y=write_bf16,
        num_warps=4 if K <= 2048 else 8)
    return y, q, s


def _bytes(s: torch.Tensor) -> torch.Tensor:
    """uint8 view of the whole underlying int32 buffer (the transposed/sliced view cannot be .view()-ed)."""
    base = s.as_strided((s.stride(1) * ((s.shape[1] - 1) + 1),), (1,), s.storage_offset())
    return base.view(torch.uint8)


def fp8_linear_prequant(layer, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Run an SGLang Fp8LinearMethod layer (block [128, 128] weights, DeepGEMM UE8M0 scales) on pre-quantized
    activations: the same call deepgemm_w8a8_block_fp8_linear_with_fallback makes after its own quant."""
    from sglang.kernels.ops.quantization.fp8_kernel import w8a8_block_fp8_matmul_deepgemm

    ws = layer.weight_scale_inv if hasattr(layer, "weight_scale_inv") else layer.weight_scale
    out = w8a8_block_fp8_matmul_deepgemm(q, layer.weight, s, ws, [128, 128], output_dtype=torch.bfloat16)
    if getattr(layer, "bias", None) is not None:
        out += layer.bias
    return out


# ------------------------------------------------------------------------------------------------ reference
def reference(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int = 128):
    """Torch reference of the same arithmetic; scales returned as [M, G] uint8 exponents."""
    xf = x.float()
    r = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / x.shape[1] + eps)
    y = (xf * r * weight.float()).bfloat16()
    M, K = y.shape
    yg = y.float().view(M, K // group, group)
    amax = yg.abs().amax(-1).clamp_min(1e-10)
    raw = amax * torch.tensor(FP8_MAX_INV, dtype=torch.float32)
    bits = raw.view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    inv = ((254 - e) << 23).view(torch.float32)
    q = (yg * inv[..., None]).view(M, K).to(torch.float8_e4m3fn)
    return y, q, e.to(torch.uint8)


def unpack_scales(s: torch.Tensor, groups: int) -> torch.Tensor:
    """[M, ceil(G/4)] packed int32 -> [M, G] uint8 exponents."""
    M = s.shape[0]
    b = s.contiguous().view(torch.uint8).view(M, -1)
    return b[:, :groups]


# ------------------------------------------------------------------------------------------------ engine glue
_LOGGED: set = set()


def _log_once(key: str, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        import logging

        logging.getLogger("numinous_pro").warning(msg)


def engine_enabled() -> bool:
    import os

    return os.environ.get("NUMINOUS_NORM_QUANT", "0") == "1"


def engine_q_norm(q_norm, q: torch.Tensor) -> torch.Tensor:
    """Hook `norm_quant` in MQALayer._compute_q_a: RMSNorm + quant once; the bf16 result carries its FP8 twin
    (`_numinous_fp8`), which Fp8LinearMethod.apply (hook `norm_quant_linear`) and the indexer (hook `indexer_q`)
    consume instead of quantizing again."""
    if q.dim() != 2 or q.stride(-1) != 1 or q.shape[1] % 128 or q.dtype != torch.bfloat16:
        _log_once("nq_decline", f"numinous norm_quant declined: shape {tuple(q.shape)} stride {q.stride()}")
        return q_norm(q)
    _log_once("nq_active", "numinous norm_quant active (q_norm + FP8 UE8M0 quant in one kernel, shared by wq_b "
                           "and indexer wq_b)")
    y, q8, s = rmsnorm_fp8_quant(q, q_norm.weight, q_norm.variance_epsilon)
    y._numinous_fp8 = (q8, s)
    return y


def engine_linear(layer, x: torch.Tensor, prequant, bias):
    """Fp8LinearMethod.apply with a pre-quantized twin: DeepGEMM directly, or None when the layer is not the
    Blackwell DeepGEMM UE8M0 block-FP8 case this layout is for."""
    from sglang.srt.layers import deep_gemm_wrapper

    w = layer.weight
    ws = getattr(layer, "weight_scale_inv", None)
    ok = (deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
          and w.dtype == torch.float8_e4m3fn and ws is not None and ws.dtype == torch.int32
          and w.shape[0] % 64 == 0 and w.shape[1] % 128 == 0 and w.shape[1] == x.shape[-1] and x.dim() == 2)
    if not ok:
        _log_once(f"nql_decline_{tuple(w.shape)}", f"numinous norm_quant_linear declined for weight "
                                                   f"{tuple(w.shape)} {w.dtype}")
        return None
    q8, s = prequant
    out = fp8_linear_prequant(layer, q8, s)
    if bias is not None:
        out += bias
    return out
