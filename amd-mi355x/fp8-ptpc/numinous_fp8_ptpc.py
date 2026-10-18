"""Block-FP8 dense linears served through AITER's per-token / per-channel (ptpc) FP8 GEMM at every M (k3amd for
amdflash, 2026-10-04, coordinator 01:50: Flash on AMD prefill).

Measured motive:
  - FP8 GEMM is 24% of GLM-5.3-Flash prefill kernel time on MI355X even with amdflash's tuned CK blockscale configs:
    the CK a8w8 blockscale bpreshuffle kernels top out at ~1.8-2.0 PFLOP/s at the Flash prefill shapes (amdflash's
    flash_bs_tuned.csv: 24576x4096 @16384 rows 1,815 us = 1.82 PF; 4096x16384 1.98 PF).
  - AITER's own tuned ptpc FP8 table (a8w8_bpreshuffle_tuned_gemm.csv, gfx950) reaches 2.93-3.05 PFLOP/s at
    comparable large shapes (flydsl_bpreshuffle_8w_256x256x128 / cktile 256x256x128), i.e. ~1.5-1.65x.
SGLang already carries the mechanism for one case: dense linears converted from MXFP8 get a rowwise-FP8 copy
(weight._ptpc_weight / _ptpc_scale) that aiter_w8a8_block_fp8_linear uses at M <= 128
(fp8.py process_weights_after_loading_block_quant, fp8_utils.py). These hooks (a) build the same rowwise copy for
plain block-FP8 checkpoints (dequantize the [128, 128] blocks, one FP8 scale per output row) and (b) raise the M
threshold so prefill-sized calls take it too. Numerics change (per-channel weights, per-token activations instead of
128-blocks / 128-groups): the gate's accuracy stage decides.

Env: NUMINOUS_FP8_BLOCK_TO_PTPC=1 (build the copies), NUMINOUS_PTPC_MAX_M (default: the upstream 128; set e.g.
1048576 to route every M). Memory: the block weight stays (other paths may read it); the copy adds one FP8 byte per
weight (Flash TP1 attention: ~5.7 GB).
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("NUMINOUS_FP8_BLOCK_TO_PTPC") == "1"


def max_m(default: int) -> int:
    v = os.environ.get("NUMINOUS_PTPC_MAX_M")
    return int(v) if v else default


def rowwise_from_block(weight, scale_inv, block=(128, 128)):
    """Dequantize a block-FP8 weight [N, K] with scales [ceil(N/bn), ceil(K/bk)] and requantize it to FP8 with one
    scale per output row. Returns (fp8 weight [N, K], fp32 row scale [N, 1])."""
    import torch

    n, k = weight.shape
    bn, bk = block
    s = scale_inv.float().repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
    w32 = weight.float() * s
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    row = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fp8_max
    q = (w32 / row).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, row


def attach(layer, block=(128, 128)) -> bool:
    """Give a block-FP8 linear the rowwise copy aiter_w8a8_block_fp8_linear's ptpc branch reads. Call before the
    block weight is bpreshuffled (it reads the plain row-major layout)."""
    import torch

    w = layer.weight
    s = getattr(layer, "weight_scale_inv", None)
    if s is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or s.dim() != 2:
        return False
    if getattr(w, "_ptpc_weight", None) is not None or getattr(layer, "keep_plain_weight_layout", False):
        return False
    from aiter.ops.shuffle import shuffle_weight

    q, row = rowwise_from_block(w.data, s.data, block)
    w._ptpc_weight = shuffle_weight(q, (16, 16))
    w._ptpc_scale = row
    return True
