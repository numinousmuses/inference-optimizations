"""DeepSeek V4 Pro: FP8 LM head. The bf16 head [129280, 7168] (1.85 GB read per decode step) becomes FP8 E4M3 with
128x128 UE8M0 block scales (0.93 GB), and logits come from DeepGEMM's FP8 GEMM, the same kernel family and scale
layout as every other dense FP8 projection in the model.

At load: `quantize_lm_head(lm_head)` stores `lm_head._numinous_fp8 = (w_fp8, w_scale)` (engine helpers
quant_weight_ue8m0 + transform_scale_ue8m0, so the layout is what DeepGEMM asserts). The bf16 weight stays (the
DSpark draft and tied embeddings may read it); NUMINOUS_FP8_LM_HEAD_DROP_BF16=1 is not offered on purpose.
Per call: `fp8_logits(hidden, lm_head)` quantizes the hidden rows per 128-group (UE8M0, engine kernel) and runs
DeepGEMM; bf16 out, upcast to fp32 when the logits processor asks for fp32.

--enable-dp-lm-head: the head is a VocabParallelEmbedding sharded over the attention-TP group; this module works on
whatever shard `lm_head.weight` holds (rows = local vocab slice, 1010 x 128 for the full Pro vocab), so the gather
the logits processor does afterwards is unchanged. Vocab shard rows must be a multiple of 64 (DeepGEMM N), K a
multiple of 128; otherwise `supported()` is False and the bf16 path runs.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def supported(weight: torch.Tensor) -> bool:
    return (weight.is_cuda and weight.dim() == 2 and weight.dtype == torch.bfloat16
            and weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0)


def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (fp8 [N, K], DeepGEMM UE8M0 packed block scales). Done in row chunks to bound temp memory."""
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0

    n = weight.shape[0]
    qs, ss = [], []
    step = 16384  # multiple of 128: block boundaries stay aligned
    for i in range(0, n, step):
        q, s = quant_weight_ue8m0(weight[i:i + step].contiguous(), weight_block_size=[128, 128])
        qs.append(q)
        ss.append(s)
    q = torch.cat(qs)
    s = transform_scale_ue8m0(torch.cat(ss), mn=n)
    return q, s


def quantize_lm_head(lm_head) -> bool:
    """Attach the FP8 copy to the module; returns False (and leaves the module alone) when unsupported."""
    w = getattr(lm_head, "weight", None)
    if w is None or not supported(w):
        logger.warning("numinous_fp8_lm_head: unsupported head %s, keeping bf16",
                       None if w is None else (tuple(w.shape), w.dtype))
        return False
    with torch.no_grad():
        lm_head._numinous_fp8 = quantize_weight(w.data)
    return True


def fp8_logits(hidden: torch.Tensor, lm_head, out_fp32: bool = False) -> torch.Tensor:
    from sglang.kernels.ops.quantization.fp8_kernel import (sglang_per_token_group_quant_fp8,
                                                            w8a8_block_fp8_matmul_deepgemm)

    wq, ws = lm_head._numinous_fp8
    h = hidden.to(torch.bfloat16)
    if not h.is_contiguous():
        h = h.contiguous()
    if h.shape[0] == 0:
        out = h.new_empty((0, wq.shape[0]))
    else:
        q, s = sglang_per_token_group_quant_fp8(h, 128, column_major_scales=True, scale_tma_aligned=True,
                                                scale_ue8m0=True)
        out = w8a8_block_fp8_matmul_deepgemm(q, wq, s, ws, [128, 128], output_dtype=torch.bfloat16)
    return out.float() if out_fp32 else out


def maybe_fp8_logits(hidden: torch.Tensor, lm_head, out_fp32: bool) -> Optional[torch.Tensor]:
    """Engine hook entry: FP8 logits if the head was quantized at load, else None (caller runs its bf16 path)."""
    if getattr(lm_head, "_numinous_fp8", None) is None:
        return None
    return fp8_logits(hidden, lm_head, out_fp32)


# ------------------------------------------------------------------------------------------------ engine glue
_LOGGED: set = set()


def engine_enabled() -> bool:
    import os

    return os.environ.get("NUMINOUS_FP8_LM_HEAD", "0") == "1"


def engine_post_load(model) -> None:
    """Hook `pro_post_load`: quantize the head once, at load (never inside CUDA-graph capture)."""
    head = getattr(model, "lm_head", None)
    if head is not None and quantize_lm_head(head):
        logger.warning("numinous fp8_lm_head active (head %s quantized to FP8 128x128 UE8M0, DeepGEMM logits)",
                       tuple(head.weight.shape))


def engine_logits(hidden: torch.Tensor, lm_head, use_fp32: bool) -> torch.Tensor:
    return fp8_logits(hidden, lm_head, out_fp32=use_fp32)
