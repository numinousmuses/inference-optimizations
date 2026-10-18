"""Reuse the DP gather's group-128 FP8 payload as the MoE / MLP input quant (rtxsys, kernel program 2026-10-05).

Installed as sglang/srt/layers/numinous_fp8_passthrough.py by operations/patches/dp_fp8_passthrough_hooks.py.
Env NUMINOUS_DP_FP8_PASSTHROUGH=1 (prefill role). It only acts with SGLANG_ENABLE_DP_GATHER_FP8=1 under gatherv.
Unset: every entry point is a no-op and the stock quant runs.

What it attacks (measured, F4 prefill trace TP0, 4,096 tokens/rank/step, per MoE layer):
- `_dp_gather_via_all_gatherv_fp8` (dp_attention.py:709-745) already quantizes each rank's rows to per-token-group-128
  FP8. It all-gathers q and scales, then dequantizes into the bf16 global buffer (49 us).
- The MoE then quantizes that same buffer twice more:
  - shared expert: the DeepGEMM w8a8 linear quant (fp8_utils.py:1112-1119), column-major TMA-aligned scales, 54 us;
  - routed experts: cutlass_fused_experts_fp8 (cutlass_moe.py:214), row-major scales, 48 us.
- Both requants recompute the gathered (q, s): see "Why it is exact".
- Saving = 54 + 48 us minus a 2-MB scale transpose. 43 MoE layers plus the 3 dense layers' gate_up quant come to
  ~-4.3 ms per step.

Why it is exact:
- The gather quant is sglang_per_token_group_quant_fp8(x, 128): s = amax/448 per 128-group, q = e4m3(x/s).
- Requantizing dequant(q, s) gives s' = amax(bf16(q*s))/448. The group's max element dequantizes to 448*s, and bf16
  rounding moves it by <= 2^-9 relative, so s' = s(1 +- 2^-9). Then q' = e4m3(q*s/s') = q, because e4m3's spacing
  is 2^-3 relative.
- So the stock path's (q', s') equals the gathered (q, s) up to that scale rounding: the passthrough skips a lossy
  round trip. Pad rows: (0, eps) in both paths.

Safety. A consumer takes the stash only when all of these hold; otherwise it returns None and the stock quant runs:
- the tensor is the same buffer (data_ptr, shape, bf16, contiguous);
- it is the same CUDA stream as the gather. Under DP TBO the gather runs on the comm stream, and the shared FP8
  buffers could be overwritten by the other micro-batch's gather;
- this consumer kind has not used it yet ("linear" once, "moe" once).
Every DP gather clears the stash first (dp_attention._dp_gather), so a non-FP8 gather can never leave a stale payload.

Lifetime fix after the FX1 boot crash (2026-10-06 03:54, glm-5.3-flash FX1 prefill log):
- The stash used to live until the next _dp_gather. In GLM-5.3-Flash MoE layers no consumer takes the "linear" kind,
  so the payload outlived its MoE.
- On the 1-token health prefill (DP0 only), the next layer's KDA attention qkv_proj input had the stashed buffer's
  data_ptr and shape. for_linear took the stale stash and crashed on its scale tensor.
- The stash is now also cleared at every attention boundary (LayerCommunicator.prepare_attn, before any attention
  projection) and at every forward start (dp_attention.set_dp_buffer_len). It lives only from the MoE input gather
  to the next attention.
- _take checks the payload shapes (q [rows, hidden], s fp32 [rows, hidden/128]) and declines (stock quant) instead
  of raising.
"""
from __future__ import annotations

import os

import torch

GROUP = 128
_ON = os.environ.get("NUMINOUS_DP_FP8_PASSTHROUGH", "0") == "1"
_st: dict | None = None


def enabled() -> bool:
    return _ON


def _stream_id() -> int:
    return torch.cuda.current_stream().cuda_stream if torch.cuda.is_available() else 0


def clear() -> None:
    global _st
    _st = None


def stash(global_tokens: torch.Tensor, gq: torch.Tensor, gs: torch.Tensor) -> None:
    """Called right after the FP8 gather dequantized into global_tokens: gq [rows, hidden] (uint8 view of e4m3),
    gs [rows, hidden/128] fp32 row-major."""
    global _st
    _st = None
    if not _ON or global_tokens.dim() != 2 or global_tokens.shape[0] == 0:
        return
    st = {"ptr": global_tokens.data_ptr(), "rows": global_tokens.shape[0], "hidden": global_tokens.shape[1],
          "q": gq, "s": gs, "stream": _stream_id(), "used": set()}
    if _valid(st):
        _st = st


def _valid(st: dict) -> bool:
    """The payload is exactly what the gather produced for this buffer: q [rows, hidden] 1-byte, s fp32
    [rows, hidden/128]. Anything else is declined: the stock quant runs."""
    q, s = st["q"], st["s"]
    rows, hidden = st["rows"], st["hidden"]
    return (hidden % GROUP == 0 and q.dim() == 2 and tuple(q.shape) == (rows, hidden) and q.element_size() == 1
            and s.dim() == 2 and tuple(s.shape) == (rows, hidden // GROUP) and s.dtype == torch.float32)


def _take(x: torch.Tensor, who: str) -> dict | None:
    global _st
    st = _st
    if st is None or who in st["used"]:
        return None
    if not _valid(st):
        _st = None
        return None
    if (x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() != st["ptr"]
            or x.shape[0] != st["rows"] or x.shape[1] != st["hidden"]):
        return None
    if _stream_id() != st["stream"]:
        return None
    st["used"].add(who)
    return st


def for_routed(a: torch.Tensor):
    """(q e4m3 [rows, hidden], s fp32 [rows, hidden/128] row-major) for cutlass_fused_experts_fp8, or None."""
    st = _take(a, "moe")
    if st is None:
        return None
    return st["q"].view(torch.float8_e4m3fn), st["s"]


def for_linear(x2d: torch.Tensor, group: int):
    """(q e4m3 [M, K], s fp32 [M, K/128] column-major with stride (1, align4(M))) for the DeepGEMM w8a8 linear:
    the layout create_per_token_group_quant_fp8_output_scale makes for column_major_scales + scale_tma_aligned
    (fp8_kernel.py:496-507). Returns None for other groups."""
    if group != GROUP:
        return None
    st = _take(x2d, "linear")
    if st is None:
        return None
    m, g = st["rows"], st["s"].shape[1]
    xs = torch.empty((g, (m + 3) // 4 * 4), dtype=torch.float32, device=x2d.device).transpose(0, 1)[:m, :]
    xs.copy_(st["s"])
    return st["q"].view(torch.float8_e4m3fn), xs
