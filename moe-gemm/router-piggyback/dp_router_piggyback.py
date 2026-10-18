"""Route before the DP gather: each rank runs the MoE router on its own rows and sends topk weights/ids along with the
FP8 gather's scales. No new collective is added. rtxattn, Hopper kernel program 2026-10-05, lever "GLM fusions".

Profile row (measured: M5-spec314 rung-32 cold-wave prefill, rank 4, per 2,048-tok/rank forward). Today every rank
runs the router on the whole DP-gathered buffer (16,384 rows), so 7/8 of that work repeats on all 8 ranks:
  - router GEMM 6144 -> 256, bf16: ~4.8 ms;
  - _router_triton topk: ~2.2 ms.
  Source: deepseek_v2.py:1082-1094 (forward_normal) after the gather in dp_attention.py:709-744.

Wire format (env NUMINOUS_DP_ROUTER_BEFORE_GATHER=1):
  - The scale gatherv carries [rows, NGROUPS + 2*KMAX] fp32 instead of [rows, NGROUPS]. Columns NGROUPS..+KMAX hold
    topk weights; the next KMAX hold topk ids bit-cast to int32.
  - The width depends only on the env, so every rank issues the same collective shapes (dp_attention.py:806-812).
  - The pending router affects only the payload. All ranks run the same decoder-layer code, so it is uniform too.
  - One Triton pass dequantizes the payload into the bf16 global buffer (same math as the stock
    _dequant_per_token_group_fp8_kernel) and copies the topk columns out.
  - forward_normal consumes the result through take(). A missing or mismatched stash runs the stock router.

Numerics: the router sees the exact bf16 local rows, not the FP8-dequantized gathered rows. Routing then matches the
bf16-gather reference and can differ from the stock FP8-gather path on near-ties.
"""
from __future__ import annotations

import threading

import torch
import triton
import triton.language as tl

KMAX = 8
_tls = threading.local()
_bufs: dict = {}


def set_pending(mlp) -> None:
    lz = getattr(_tls, "lazy", None)
    if lz is not None and not lz["taken"] and lz["rows"] > 0:
        # The previous layer's deferred buffer was neither taken by pre_reorder_fp8 nor materialized, so some reader
        # this module does not know ran on it. Fail loudly instead of serving it.
        _tls.lazy = None
        raise RuntimeError("numinous lazy dequant: a deferred DP-gather buffer reached the end of its layer unread "
                           "by pre_reorder_fp8 and unmaterialized (an unlisted reader): unset NUMINOUS_DP_LAZY_DEQUANT")
    _tls.pending = mlp
    _tls.stash = None
    _tls.lazy = None


def clear_pending() -> None:
    _tls.pending = None


def take_pending():
    mlp = getattr(_tls, "pending", None)
    _tls.pending = None
    return mlp


def stash(mlp, topk_output) -> None:
    _tls.stash = (mlp, topk_output)


def take(mlp, rows: int):
    st = getattr(_tls, "stash", None)
    _tls.stash = None
    if st is None or st[0] is not mlp or st[1].topk_ids.shape[0] != rows:
        return None
    return st[1]


@triton.jit
def _dequant_scales_topk_kernel(q_ptr, s_ptr, si_ptr, out_ptr, w_ptr, ids_ptr,
                                HIDDEN: tl.constexpr, NGROUPS: tl.constexpr, GROUP: tl.constexpr,
                                S_STRIDE: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, KMAXP: tl.constexpr,
                                WITH_TOPK: tl.constexpr, BLOCK: tl.constexpr, WITH_DEQUANT: tl.constexpr = True):
    row = tl.program_id(0).to(tl.int64)
    if WITH_DEQUANT:
        for start in tl.static_range(0, HIDDEN, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HIDDEN
            qv = tl.load(q_ptr + row * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
            sv = tl.load(s_ptr + row * S_STRIDE + offs // GROUP, mask=mask, other=0.0)
            tl.store(out_ptr + row * HIDDEN + offs, (qv * sv).to(tl.bfloat16), mask=mask)
    if WITH_TOPK:
        k = tl.arange(0, KP)
        km = k < K
        w = tl.load(s_ptr + row * S_STRIDE + NGROUPS + k, mask=km, other=0.0)
        i = tl.load(si_ptr + row * S_STRIDE + NGROUPS + KMAXP + k, mask=km, other=0)
        tl.store(w_ptr + row * K + k, w, mask=km)
        tl.store(ids_ptr + row * K + k, i, mask=km)



def dequant_and_split(gq: torch.Tensor, gs: torch.Tensor, out: torch.Tensor, ngroups: int, group: int, k: int,
                      with_topk: bool, with_dequant: bool = True):
    """gq uint8/fp8 [rows, hidden]; gs fp32 [rows, ngroups + 2*KMAX]; out bf16 [rows, hidden] (filled unless
    with_dequant is False: the lazy-dequant path). Returns (weights fp32 [rows, k], ids int32 [rows, k]) when
    with_topk, else (None, None)."""
    rows, hidden = out.shape[0], out.shape[-1]
    w = torch.empty((rows, max(k, 1)), dtype=torch.float32, device=out.device)
    ids = torch.empty((rows, max(k, 1)), dtype=torch.int32, device=out.device)
    if rows and (with_topk or with_dequant):
        _dequant_scales_topk_kernel[(rows,)](
            gq.view(torch.float8_e4m3fn), gs, gs.view(torch.int32), out, w, ids,
            HIDDEN=hidden, NGROUPS=ngroups, GROUP=group, S_STRIDE=ngroups + 2 * KMAX, K=max(k, 1),
            KP=triton.next_power_of_2(max(k, 1)), KMAXP=KMAX, WITH_TOPK=with_topk, BLOCK=2048,
            WITH_DEQUANT=with_dequant)
    return (w[:, :k], ids[:, :k]) if with_topk else (None, None)


def pack_scales(s: torch.Tensor, ngroups: int, weights=None, ids=None) -> torch.Tensor:
    """[T, ngroups] fp32 scales (+ optional topk [T, k]) -> [T, ngroups + 2*KMAX] fp32 wire rows."""
    t = s.shape[0]
    wide = torch.zeros((t, ngroups + 2 * KMAX), dtype=torch.float32, device=s.device)
    wide[:, :ngroups] = s
    if weights is not None and t:
        k = weights.shape[1]
        wide[:, ngroups:ngroups + k] = weights.float()
        wide.view(torch.int32)[:, ngroups + KMAX:ngroups + KMAX + k] = ids.to(torch.int32)
    return wide


def _wire_bufs(rows: int, hidden: int, width: int, device):
    key = (str(device), hidden, width)
    b = _bufs.get(key)
    if b is None or b[0].shape[0] < rows:
        b = (torch.empty((rows, hidden), dtype=torch.uint8, device=device),
             torch.empty((rows, width), dtype=torch.float32, device=device))
        _bufs[key] = b
    return b[0][:rows], b[1][:rows]


def static_topk_width(mlp) -> int:
    """Rank-uniform decision: the number of topk columns this MoE module's router produces in the STANDARD format, or 0
    when the piggyback must not be used. TP-sharded experts need identical routing on every rank, so this reads only
    static configuration (the same on all ranks, including idle ones), mirroring TopK.forward_cuda's format choice
    (topk.py:639-660)."""
    if mlp is None or getattr(mlp, "is_hash", False) or getattr(mlp, "topk", None) is None:
        return 0
    from sglang.srt.layers.moe import get_moe_runner_backend
    from sglang.srt.layers.moe.topk import TopKOutputFormat
    cfg = mlp.topk.topk_config
    if getattr(cfg, "num_fused_shared_experts", 0):
        return 0
    if cfg.output_format is not None:
        ok = cfg.output_format == TopKOutputFormat.STANDARD
    else:
        b = get_moe_runner_backend()
        ok = not (b.is_triton_kernels() or b.is_experimental_sgl_trtllm() or b.is_flashinfer_trtllm()
                  or (b.is_flashinfer_mxfp4() and not getattr(mlp.topk, "is_fp4_experts", False)))
    k = int(cfg.top_k)
    return k if ok and 1 <= k <= KMAX else 0


def local_topk(mlp, x: torch.Tensor, k: int):
    """The MoE module's own gate + TopK on this rank's rows (same math per row as on the gathered buffer)."""
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    out = mlp.topk(x, mlp.gate(x))
    if not isinstance(out, StandardTopKOutput) or out.topk_ids.shape[1] != k:
        raise RuntimeError(f"numinous router-before-gather: expected STANDARD topk with {k} columns, got "
                           f"{type(out).__name__} {tuple(getattr(out, 'topk_ids', torch.empty(0)).shape)}")
    return out.topk_weights, out.topk_ids


def _passthrough_stash(global_tokens, gq, gs, ngroups) -> None:
    """Keep rtxsys's dp_fp8_passthrough working: the stock gather stashes (q, s) after its dequant, and this replacement
    must do the same. That module's consumers expect row-major [rows, ngroups] scales, so the scale columns are copied
    out of the wide wire rows (rows x ngroups x 4 B), and only when the passthrough is on."""
    try:
        from sglang.srt.layers import numinous_fp8_passthrough as _nfp
    except ImportError:
        return
    if _nfp.enabled():
        _nfp.stash(global_tokens, gq, gs[:, :ngroups].contiguous())


def gather_fp8_with_router(global_tokens: torch.Tensor, local_real: torch.Tensor, sizes, group: int) -> None:
    """Drop-in for dp_attention._dp_gather_via_all_gatherv_fp8 with the widened scale gatherv."""
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    rows, hidden = global_tokens.shape[0], global_tokens.shape[-1]
    ngroups = hidden // group
    q, s = sglang_per_token_group_quant_fp8(local_real.contiguous(), group)
    mlp = take_pending()
    k = static_topk_width(mlp)                      # rank-uniform
    routed = local_topk(mlp, local_real, k) if (k and local_real.shape[0] > 0) else None
    wide = pack_scales(s, ngroups, *(routed if routed is not None else (None, None)))
    gq, gs = _wire_bufs(rows, hidden, ngroups + 2 * KMAX, global_tokens.device)
    tp = get_tp_group()
    tp.all_gatherv(q.view(torch.uint8), sizes=sizes, output=gq)
    tp.all_gatherv(wide, sizes=sizes, output=gs)
    _tls.lazy = None
    lazy = bool(k) and lazy_eligible(mlp)
    w, ids = dequant_and_split(gq, gs, global_tokens, ngroups, group, k, bool(k), with_dequant=not lazy)
    _passthrough_stash(global_tokens, gq, gs, ngroups)
    if lazy:
        _tls.lazy = {"ptr": global_tokens.data_ptr(), "rows": rows, "hidden": hidden, "out": global_tokens,
                     "q": gq, "s": gs, "ngroups": ngroups, "group": group, "taken": False}
    if k:
        stash(mlp, StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=w.new_empty((w.shape[0], 0))))


# Backends whose TopK output is not STANDARD (topk.py:639-660): the router then cannot move and the lever is inert.
_NON_STANDARD_BACKENDS = ("triton_kernel", "flashinfer_trtllm", "flashinfer_mxfp4", "experimental_sgl_trtllm")


def numinous_preflight(ctx) -> list:
    """CPU attach check (operations/hopper/hook_preflight.py): with NUMINOUS_DP_ROUTER_BEFORE_GATHER=1, report every
    resolved configuration under which the lever silently does nothing. The lever:
      - rides on the FP8 gatherv (dp_attention.py:806-812, is_dp_gatherv_active);
      - needs a STANDARD TopK (static_topk_width);
      - is skipped by forward_normal under EPLB (dispatch_info)."""
    env = dict(getattr(ctx, "env", None) or {})
    if env.get("NUMINOUS_DP_ROUTER_BEFORE_GATHER") != "1":
        return []
    a = ctx.args
    probs = []
    if env.get("SGLANG_DP_USE_GATHERV") != "1" or env.get("SGLANG_ENABLE_DP_GATHER_FP8") != "1":
        probs.append("router-before-gather is inert: it rides on the FP8 gatherv (SGLANG_DP_USE_GATHERV=1 and "
                     "SGLANG_ENABLE_DP_GATHER_FP8=1)")
    if not a.get("enable_dp_attention") or int(a.get("tp_size") or 1) != int(a.get("dp_size") or 1):
        probs.append("router-before-gather is inert: the gatherv path needs --enable-dp-attention with tp == dp "
                     "(attention TP 1)")
    if str(a.get("moe_runner_backend") or "auto") in _NON_STANDARD_BACKENDS:
        probs.append(f"router-before-gather is inert: --moe-runner-backend {a.get('moe_runner_backend')} makes TopK "
                     "non-STANDARD")
    if a.get("enable_eplb"):
        probs.append("router-before-gather is inert under --enable-eplb (forward_normal routes with dispatch_info)")
    k = int(getattr(ctx.hf_config, "num_experts_per_tok", 0) or 0)
    if k > KMAX:
        probs.append(f"router-before-gather is inert: top-k {k} > {KMAX} wire columns")
    if env.get("NUMINOUS_DP_LAZY_DEQUANT") == "1":
        quant = str(getattr(ctx, "checkpoint_quant", "") or a.get("quantization") or "")
        if env.get("NUMINOUS_DP_FP8_PASSTHROUGH") != "1":
            probs.append("lazy dequant is inert without NUMINOUS_DP_FP8_PASSTHROUGH=1 (the shared expert must take FP8)")
        if str(env.get("SGLANG_OPT_MOE_QUANT_ONCE", "0")).lower() in ("1", "true"):
            probs.append("lazy dequant is inert with SGLANG_OPT_MOE_QUANT_ONCE (it reads the bf16 buffer)")
        if env.get("NUMINOUS_W4A8_MARLIN"):
            probs.append("lazy dequant is inert with NUMINOUS_W4A8_MARLIN (Marlin reads the bf16 buffer)")
        if quant != "w4afp8":
            probs.append(f"lazy dequant is inert: only the cutlass W4A8 runner takes the FP8 form (quant {quant or 'none'})")
    return probs



# ---------------------------------------------------------------------------------------------------------------
# Lazy dequant (env NUMINOUS_DP_LAZY_DEQUANT=1, on top of router-before-gather + rtxsys's FP8 passthrough).
#
# After the router moves before the gather and the shared expert takes the FP8 payload, the gathered bf16 buffer has
# one reader left: the routed W4A8 MoE's pre_reorder. pre_reorder_fp8 below reads the FP8 payload and group scales
# instead, so the gather skips its dequant entirely.
#   Profile row: _dequant_per_token_group_fp8, 5.86 ms per M5 prefill pass. Measured stock dequant: 77.3 us at
#   16,384 rows (glm_kernel_benches_h200_glue.json router_wire), plus a smaller pre_reorder read.
#
# Every reader of the buffer either takes the FP8 form or forces materialize() first:
#   - DeepseekV2MoE.forward: any dispatch other than forward_normal materializes (hook lazy_moe_entry);
#   - forward_normal entry (hook lazy_normal_check): materializes unless all three hold:
#       - the router stash is there;
#       - the passthrough would hand the shared-expert linear its FP8 input;
#       - no SBO-fused shared expert and no quant-once;
#   - cutlass_w4a8_moe: the absmax path (dynamic a1) materializes. pre_reorder takes the FP8 form, and a lazy record
#     whose buffer is not the runner's input raises (never silent);
#   - set_pending / the next gather drop the record, so it never outlives its layer.
# Numerics: out = fp8(q * s / a1) in fp32. That is one rounding fewer than the stock fp8(bf16(q * s) / a1).


def lazy_enabled() -> bool:
    import os
    return os.environ.get("NUMINOUS_DP_LAZY_DEQUANT") == "1"


def lazy_eligible(mlp) -> bool:
    """Static, per module: the lazy path needs the passthrough (shared expert takes FP8), the cutlass W4A8 routed
    runner (pre_reorder_fp8), no quant-once and no a2a."""
    import os
    if not lazy_enabled() or mlp is None:
        return False
    if os.environ.get("SGLANG_OPT_MOE_QUANT_ONCE", "0").lower() in ("1", "true"):
        return False
    try:
        from sglang.srt.layers import numinous_fp8_passthrough as _nfp
    except ImportError:
        return False
    if not _nfp.enabled() or getattr(mlp, "_enable_a2a_moe", False):
        return False
    if os.environ.get("NUMINOUS_W4A8_MARLIN"):          # k3amd's Marlin dispatch reads the bf16 input
        return False
    qm = getattr(getattr(mlp, "experts", None), "quant_method", None)
    return type(qm).__name__ == "W4AFp8MoEMethod"


def is_lazy(x: torch.Tensor) -> bool:
    lz = getattr(_tls, "lazy", None)
    return lz is not None and not lz["taken"] and x.data_ptr() == lz["ptr"] and x.shape[0] == lz["rows"]


def materialize(x: torch.Tensor | None = None) -> bool:
    """Run the deferred dequant into the bf16 buffer now (stock math), and drop the record."""
    lz = getattr(_tls, "lazy", None)
    if lz is None or (x is not None and x.data_ptr() != lz["ptr"]):
        return False
    _tls.lazy = None
    if lz["taken"]:
        raise RuntimeError("numinous lazy dequant: the buffer was materialized after pre_reorder took the FP8 form")
    dequant_and_split(lz["q"], lz["s"], lz["out"], lz["ngroups"], lz["group"], 0, False, with_dequant=True)
    return True


def _passthrough_would_take(x: torch.Tensor) -> bool:
    """dp_fp8_passthrough._take's conditions for the 'linear' consumer, without consuming the stash."""
    try:
        from sglang.srt.layers import numinous_fp8_passthrough as _nfp
    except ImportError:
        return False
    st = getattr(_nfp, "_st", None)
    if st is None or "linear" in st["used"]:
        return False
    return (x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() == st["ptr"]
            and x.shape[0] == st["rows"] and x.shape[1] == st["hidden"] and _nfp._stream_id() == st["stream"])


def lazy_check_normal(mlp, x: torch.Tensor, skip_shared_experts: bool) -> None:
    """forward_normal entry: keep the buffer lazy only when every reader below takes the FP8 form."""
    if not is_lazy(x):
        return
    st = getattr(_tls, "stash", None)
    router_ok = st is not None and st[0] is mlp and st[1].topk_ids.shape[0] == x.shape[0]
    shared_ok = skip_shared_experts or (not getattr(mlp, "_fuse_shared_experts_inside_sbo", False)
                                        and _passthrough_would_take(x))
    if not (router_ok and shared_ok):
        materialize(x)


def lazy_take(a: torch.Tensor):
    """For the W4A8 runner's pre_reorder: (q e4m3 [rows, hidden], wide scales fp32 [rows, S], ngroups, group) when `a`
    is the lazy buffer, else None. A pending record for a different tensor means some step copied the unmaterialized
    buffer, so it raises."""
    lz = getattr(_tls, "lazy", None)
    if lz is None or lz["taken"]:
        return None
    if a.data_ptr() != lz["ptr"] or a.shape[0] != lz["rows"]:
        raise RuntimeError("numinous lazy dequant: the routed MoE input is not the gathered buffer it deferred")
    lz["taken"] = True
    return lz["q"].view(torch.float8_e4m3fn), lz["s"], lz["ngroups"], lz["group"]


@triton.jit
def _pre_reorder_fp8_kernel(q_ptr, s_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr,
                            num_local_experts, topk, num_tokens, hidden_size,
                            S_STRIDE: tl.constexpr, GROUP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                            NUM_STAGES: tl.constexpr, INTERPRET: tl.constexpr):
    """pre_reorder_triton_kernel_for_cutlass_moe (ep_moe_kernels.py:242-283) reading the gathered FP8 payload +
    per-token-group scales instead of the bf16 buffer: out = fp8(q * s / a1)."""
    OutDtype = gateup_input_ptr.dtype.element_ty
    a1_scale = 1.0 / tl.load(a1_scales_ptr)
    offset = BLOCK_SIZE * tl.program_id(1) + tl.arange(0, BLOCK_SIZE)
    mask = offset < hidden_size
    start_src_idx = tl.program_id(0)
    step = tl.num_programs(0)
    for src_idx_int32 in tl.range(start_src_idx, num_tokens, step, num_stages=NUM_STAGES):
        if INTERPRET:
            src_idx = src_idx_int32
        else:
            src_idx = src_idx_int32.to(tl.int64)
        token_src2dst_ptr = src2dst_ptr + src_idx * topk
        token_topk_ids_ptr = topk_ids_ptr + src_idx * topk
        qv = tl.load(q_ptr + src_idx * hidden_size + offset, mask=mask, other=0.0).to(tl.float32)
        sv = tl.load(s_ptr + src_idx * S_STRIDE + offset // GROUP, mask=mask, other=0.0)
        out_data = (qv * sv * a1_scale).to(OutDtype)
        dst_ptr_offs = gateup_input_ptr + offset
        for idx in range(topk):
            expert_id = tl.load(token_topk_ids_ptr + idx)
            if expert_id != num_local_experts:
                dst_idx = tl.load(token_src2dst_ptr + idx).to(tl.int64)
                tl.store(dst_ptr_offs + dst_idx * hidden_size, out_data, mask=mask)


def pre_reorder_fp8(q, s_wide, ngroups, group, gateup_input, src2dst, topk_ids, a1_scales, num_local_experts, topk,
                    num_tokens, hidden_size, grid=None, block_dim=None):
    """Drop-in for pre_reorder_for_cutlass_moe(a, ...) on the lazy buffer (same launch geometry as the stock call)."""
    import os
    if grid is None:
        from sglang.kernels.ops.moe.ep_moe_kernels import _get_launch_config_2d
        grid, block_dim = _get_launch_config_2d(q.device, num_tokens, hidden_size)
    _pre_reorder_fp8_kernel[grid](
        q, s_wide, gateup_input, src2dst, topk_ids, a1_scales, num_local_experts, topk, num_tokens, hidden_size,
        S_STRIDE=s_wide.stride(0), GROUP=group, BLOCK_SIZE=block_dim, NUM_STAGES=3,
        INTERPRET=os.environ.get("TRITON_INTERPRET") == "1")
