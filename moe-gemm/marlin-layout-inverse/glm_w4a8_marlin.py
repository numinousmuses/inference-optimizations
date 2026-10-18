"""W4AFP8 MoE experts (int4 g128, cutlass_w4a8 layout) -> Marlin W4A16 layout, for running GLM-5.3's experts through
fused_marlin_moe (k3amd, coordinator 2026-10-05: the decode MoE GEMM is the top kernel of the P2 decode cycle, 58%).

Source layouts (SGLang v0.5.20):
  W4AFp8MoEMethod.create_weights (w4afp8.py): w13_weight int8 [E, 2N, K/2], w2_weight int8 [E, K, N/2]: two signed
  int4 per byte along the reduction dim, low nibble = even index (test/manual/quant/test_cutlass_w4a8_moe.py
  pack_int4_values_to_int8); w13_weight_scale_inv / w2_weight_scale_inv fp32 [E, rows, cols/128] before
  process_weights_after_loading interleaves them (interleave_scales). The dequantized weight is int4 x scale.
  Marlin (fused_marlin_moe.py:135, num_bits 4, uint4b8): gptq_marlin_moe_repack (gptq_kernels.py:62) takes GPTQ
  int32 [E, K/8, N] (8 nibbles along K, value + 8, nibble i at bits 4i) and marlin_moe_permute_scales
  (marlin_utils.py:334) takes scales [E, K/128, N] in the activation dtype.

The pure-torch packing below is CPU-tested (operations/patches/test_glm_w4a8_marlin.py); the repack calls are GPU ops.
"""
from __future__ import annotations

import torch

GROUP = 128


def unpack_int4_pairs(w: torch.Tensor) -> torch.Tensor:
    """int8 [..., K/2] (two signed int4, low nibble first) -> int8 [..., K] in [-8, 7]."""
    assert w.dtype == torch.int8
    lo = torch.bitwise_right_shift(torch.bitwise_left_shift(w, 4), 4)    # sign-extend the low nibble
    hi = torch.bitwise_right_shift(w, 4)                                 # arithmetic shift sign-extends
    out = torch.empty(*w.shape[:-1], w.shape[-1] * 2, dtype=torch.int8, device=w.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def pack_int4_pairs(q: torch.Tensor) -> torch.Tensor:
    """Inverse of unpack_int4_pairs (the cutlass_w4a8 layout)."""
    q = q.to(torch.int8)
    return (torch.bitwise_left_shift(q[..., 1::2], 4) | (q[..., 0::2] & 0x0F)).to(torch.int8)


def to_gptq_int32(q: torch.Tensor) -> torch.Tensor:
    """int8 [E, N, K] signed int4 values (rows = output features, K = reduction) -> GPTQ uint4b8 int32 [E, K/8, N]."""
    e, n, k = q.shape
    assert k % 8 == 0
    u = (q.to(torch.int64) + 8).permute(0, 2, 1).reshape(e, k // 8, 8, n)        # [E, K/8, 8, N], 0..15
    shifts = (torch.arange(8, device=q.device, dtype=torch.int64) * 4).view(1, 1, 8, 1)
    packed = (torch.bitwise_left_shift(u, shifts)).sum(dim=2)                       # < 2^32
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed)               # two's-complement into int32
    return packed.to(torch.int32).contiguous()


def scales_to_marlin_input(s: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """fp32 [E, N, K/128] (W4AFP8 checkpoint layout, rows = output features) -> [E, K/128, N] in the activation dtype."""
    return s.permute(0, 2, 1).to(dtype).contiguous()


@torch.no_grad()
def w4afp8_to_marlin(w_packed: torch.Tensor, s: torch.Tensor, chunk: int = 32, dtype=torch.bfloat16):
    """One projection of a W4AFP8 MoE layer -> (Marlin int32 weights [E, K/16, 2 N], Marlin scales [E, K/128, N]).
    w_packed int8 [E, N, K/2], s fp32/bf16 [E, N, K/128]. Experts in chunks bound the transient memory."""
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import gptq_marlin_moe_repack
    from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales
    e, n, k2 = w_packed.shape
    k = k2 * 2
    out_w, out_s = [], []
    for a in range(0, e, chunk):
        q = unpack_int4_pairs(w_packed[a:a + chunk])
        g = to_gptq_int32(q)
        del q
        perm = torch.empty((g.shape[0], 0), dtype=torch.int32, device=g.device)
        out_w.append(gptq_marlin_moe_repack(g, perm, k, n, 4))
        del g
        out_s.append(marlin_moe_permute_scales(scales_to_marlin_input(s[a:a + chunk], dtype), k, n, GROUP))
    return torch.cat(out_w), torch.cat(out_s)


def dequant_reference(w_packed: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp32 [E, N, K] = int4 x group scale (the weight both kernels compute with)."""
    q = unpack_int4_pairs(w_packed).to(torch.float32)
    e, n, k = q.shape
    return (q.view(e, n, k // GROUP, GROUP) * s.to(torch.float32).unsqueeze(-1)).view(e, n, k)
