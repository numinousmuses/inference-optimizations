"""Format bridge between our NVFP4 layout and what the CUTLASS block-scaled GEMM reads.

Our layout (matches nvidia ModelOpt):
  weight codes   uint8 [N, K//2]   two e2m1 nibbles per byte (low nibble = even k)
  block scale    e4m3  [N, K//16]  per-16-element scale
  global scale   fp32  scalar      per-tensor

CUTLASS block-scaled GEMM (NVF4, sf_vec_size=16) needs, per operand:
  values     Float4E2M1FN [rows, K]   (the DSL stores 2 codes/byte, same as ours)
  scales     Float8E4M3FN in a SWIZZLED blocked layout (atom (32,4) x 4), NOT plain [rows,K//16]
  a per-tensor fp32 scale is applied in the epilogue (alpha = gscale_A * gscale_B), since the
  block-scaled MMA consumes only the e4m3 per-block scales.

This module has NO GPU/DSL dependency (pure torch), so the round-trip is unit-testable
anywhere torch runs; the kernel binding that consumes it lives in grouped_gemm.py.
"""
from __future__ import annotations

import torch

SF_VEC = 16   # NVFP4 block size (per-16-element e4m3 scale)


def quantize_nvfp4(x: torch.Tensor):
    """bf16/fp32 activations or weights [M, K] -> (codes uint8 [M, K//2], block_scale e4m3
    [M, K//SF_VEC], global_scale fp32 scalar). Same NVFP4 quantizer as our weight packer, so
    activations and weights share one format. K is padded to a multiple of SF_VEC."""
    from kernels.reference.nvfp4 import E2M1_GRID, quantize
    blocks, block_scale, global_scale, n = quantize(x)          # blocks:[M,nblk,16] floats
    scale = (block_scale * global_scale).unsqueeze(-1)
    grid = E2M1_GRID.to(x.device)
    codes = (blocks / scale.clamp(min=1e-12)).unsqueeze(-1)
    idx = (codes.abs() - grid).abs().argmin(dim=-1)             # nearest e2m1 magnitude
    idx = idx + (blocks < 0).long() * 8                        # sign bit
    flat = idx.flatten(-2)[..., :n].to(torch.uint8)            # [M, K] 4-bit codes
    if flat.shape[-1] % 2:                                      # pad odd K for nibble packing
        flat = torch.nn.functional.pad(flat, (0, 1))
    lo, hi = flat[:, 0::2], flat[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()                    # [M, K//2]
    return packed, block_scale.to(torch.float8_e4m3fn), global_scale.float().reshape(())


def to_blocked_scale(sf: torch.Tensor) -> torch.Tensor:
    """Rearrange a plain per-block scale [rows, k_blocks] into the CUTLASS swizzled blocked
    layout the tensor cores read (atom (32,4) over rows, 4 over k). Rows padded to 128, k to
    4. This mirrors the reference `to_blocked` in the CUTLASS tutorials so the scales land
    where the MMA expects them."""
    rows, cols = sf.shape
    rpad = (128 - rows % 128) % 128
    cpad = (4 - cols % 4) % 4
    if rpad or cpad:
        sf = torch.nn.functional.pad(sf, (0, cpad, 0, rpad))
    R, C = sf.shape
    nrb, ncb = R // 128, C // 4
    blocks = sf.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)     # [nrb, ncb, 128, 4]
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def epilogue_alpha(gscale_a: torch.Tensor, gscale_b: torch.Tensor) -> float:
    """Per-tensor fp32 scale applied after the block-scaled MMA: the MMA uses only the e4m3
    per-block scales, so the two global scales multiply the accumulator."""
    return float(gscale_a) * float(gscale_b)
