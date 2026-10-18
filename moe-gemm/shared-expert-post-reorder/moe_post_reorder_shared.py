"""MoE finalize with the shared-expert add folded in: post_reorder for the cutlass W4A8 MoE plus `+ shared`, in one
pass (rtxattn, Hopper kernel program 2026-10-05, lever "GLM fusions").

Installed by operations/patches/moe_shared_post_reorder_hooks.py as sglang/srt/layers/moe/numinous_post_reorder_shared.py.

Profile row (measured: M5-spec314 rung-32 cold-wave prefill, rank 4, 8 forwards of 2,048 tok/rank; weights-cache:
/verify_logs/prof_M5s314_cold32_20261005-134343): `vectorized_elementwise_kernel<8, CUDAFunctor_add<bf16>>`, 10.0 ms per
forward, 74.9 calls at 134 us. That is `routed += shared` (mxfp4_flashinfer_trtllm_moe.py:415-416 via deepseek_v2.py
:1164-1169) on the DP-gathered 16,384-row buffer: it reads two and writes one 16,384 x 6144 bf16 tensor, right after
post_reorder_triton_kernel_for_cutlass_moe (ep_moe_kernels.py:822, 422 us/call) has written the routed one.
Fusion: post_reorder loads the shared row once more and adds it before its store. That costs one extra 201 MB read
(~42 us) against the separate kernel's 604 MB (~134 us measured), so about -92 us per MoE layer, ~-6.9 ms per
prefill forward (derived; the 1xH200 microbench measures it).
Numerics: the routed sum, the routed scaling and the shared add stay in fp32 with one bf16 store. Stock rounds the
routed output to bf16, then adds in bf16. This is not bit-identical; the error vs fp32 is lower.

The stash: forward_normal computes the shared output before the experts and stash()es it. The cutlass W4A8 runner
take()s it at its post_reorder and marks it consumed. forward_normal then skips its own add only if consumed()
(otherwise it falls back to the stock add).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

ENV = "NUMINOUS_MOE_SHARED_IN_POST_REORDER"
_tls = threading.local()


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def stash(shared: torch.Tensor) -> None:
    _tls.shared, _tls.consumed = shared, False


def take() -> Optional[torch.Tensor]:
    s = getattr(_tls, "shared", None)
    _tls.shared = None
    return s


def mark_consumed() -> None:
    _tls.consumed = True


def consumed() -> bool:
    c = getattr(_tls, "consumed", False)
    _tls.consumed = False
    return c


def clear() -> None:
    _tls.shared, _tls.consumed = None, False


if HAVE_TRITON:

    @triton.jit
    def _post_reorder_shared_kernel(down_output_ptr, output_ptr, shared_ptr, src2dst_ptr, topk_ids_ptr,
                                    topk_weights_ptr, num_local_experts, topk, num_tokens, hidden_size,
                                    routed_scaling_factor: float, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr,
                                    INTERPRET: tl.constexpr):
        """ep_moe_kernels.post_reorder_triton_kernel_for_cutlass_moe (v0.5.20 :822-874) + the shared row before the
        store."""
        OutDtype = output_ptr.dtype.element_ty
        offset = BLOCK_SIZE * tl.program_id(1) + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        down_output_ptr_offs = down_output_ptr + offset
        output_ptr_offs = output_ptr + offset
        shared_ptr_offs = shared_ptr + offset
        start_src_idx = tl.program_id(0)
        step = tl.num_programs(0)
        for src_idx_int32 in tl.range(start_src_idx, num_tokens, step, num_stages=NUM_STAGES):
            if INTERPRET:  # the CPU interpreter yields Python ints for the loop variable
                src_idx = src_idx_int32
            else:
                src_idx = src_idx_int32.to(tl.int64)
            token_src2dst_ptr = src2dst_ptr + src_idx * topk
            token_topk_ids_ptr = topk_ids_ptr + src_idx * topk
            token_topk_weights_ptr = topk_weights_ptr + src_idx * topk
            sum_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for idx in range(topk):
                expert_id = tl.load(token_topk_ids_ptr + idx)
                if expert_id != num_local_experts:
                    dst_idx = tl.load(token_src2dst_ptr + idx).to(tl.int64)
                    weight_scale = tl.load(token_topk_weights_ptr + idx).to(tl.float32)
                    in_data = tl.load(down_output_ptr_offs + dst_idx * hidden_size, mask=mask).to(tl.float32)
                    sum_vec += in_data * weight_scale
            sum_vec *= routed_scaling_factor
            sum_vec += tl.load(shared_ptr_offs + src_idx * hidden_size, mask=mask).to(tl.float32)
            tl.store(output_ptr_offs + src_idx * hidden_size, sum_vec.to(OutDtype), mask=mask)


def launch_config(device, m: int, n: int, sm_count: Optional[int] = None):
    """The stock launcher's grid (ep_moe_kernels._get_launch_config_2d, v0.5.20 :52-76), with an injectable SM count for
    CPU tests."""
    max_threads_per_block, min_threads_per_block, max_waves = 1024, 512, 8
    if sm_count is None:
        props = torch.cuda.get_device_properties(device)
        sm_count, max_threads_per_sm = props.multi_processor_count, props.max_threads_per_multi_processor
    else:
        max_threads_per_sm = 2048
    max_num_blocks = sm_count * max_threads_per_sm // max_threads_per_block
    block_dim = max_threads_per_block

    def nblocks(bd):
        return m * -(-n // bd)

    while block_dim > min_threads_per_block and nblocks(block_dim // 2) <= max_num_blocks:
        block_dim //= 2
    gx = -(-n // block_dim)
    gy = max(min(m, max_num_blocks * max_waves // gx), 1)
    return (gy, gx), block_dim


def post_reorder_shared(down_output, output, src2dst, topk_ids, topk_weights, num_local_experts: int, topk: int,
                        num_tokens: int, hidden_size: int, routed_scaling_factor: float, shared: torch.Tensor,
                        sm_count: Optional[int] = None) -> None:
    """output[t] = routed_scaling_factor * sum_k w[t,k] * down_output[src2dst[t,k]] + shared[t] (fp32 math, one
    store)."""
    assert shared.shape == output.shape and shared.dtype == output.dtype and shared.is_contiguous(), (
        shared.shape, output.shape, shared.dtype, output.dtype)
    if num_tokens == 0:
        return
    grid, block_dim = launch_config(down_output.device, num_tokens, hidden_size, sm_count)
    _post_reorder_shared_kernel[grid](down_output, output, shared, src2dst, topk_ids, topk_weights, num_local_experts,
                                      topk, num_tokens, hidden_size, routed_scaling_factor, BLOCK_SIZE=block_dim,
                                      NUM_STAGES=3, INTERPRET=os.environ.get("TRITON_INTERPRET") == "1")


def numinous_preflight(ctx) -> list:
    """CPU attach check (operations/hopper/hook_preflight.py). With NUMINOUS_MOE_SHARED_IN_POST_REORDER=1, report the
    resolved configurations under which the fusion silently does nothing: only the cutlass W4A8 runner consumes the
    stash (W4AFP8 checkpoints), and a TP1 shared expert adds after the all-reduce (deepseek_v2.py:1171-1173), so the
    model hook keeps the stock add."""
    env = dict(getattr(ctx, "env", None) or {})
    if env.get("NUMINOUS_MOE_SHARED_IN_POST_REORDER") != "1":
        return []
    probs = []
    if env.get("SGLANG_SHARED_EXPERT_TP1") == "1":
        probs.append("finalize+shared fusion is inert with SGLANG_SHARED_EXPERT_TP1=1 (the shared add runs after the "
                     "all-reduce)")
    quant = str(getattr(ctx, "checkpoint_quant", "") or ctx.args.get("quantization") or "")
    if quant != "w4afp8":
        probs.append(f"finalize+shared fusion is inert: only the cutlass W4A8 runner consumes it (checkpoint quant "
                     f"{quant or 'none'})")
    return probs
