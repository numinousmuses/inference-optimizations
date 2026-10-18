"""SGLang glue for the kv4tri NVFP4-KV Triton path: installed by operations/patches/kv4tri_hooks.py as
sglang/srt/layers/attention/numinous_kv4tri.py and called from hooks in TritonAttnBackend.forward_decode /
forward_extend, the KV-pool configurator and the pool-size calculator.

What it does when the KV pool holds NVFP4 (`--kv-cache-dtype nvfp4 --attention-backend triton`):
  * forward_decode / forward_extend: store K/V through the pool (its own NVFP4 quantizer), then run the native FP4
    kernels (operations/kernels/kv4tri_nvfp4_attention.py) on the packed pool + block scales + the layer's global
    scale. Features the kernels do not implement (attention sinks, score_mod, DCP, deterministic unified kernel,
    MLA, paged layouts > 1, KV read-back with k=None, xai temperature) raise NotImplementedError instead of falling
    through to the stock kernels, which would read the packed bytes as bf16/fp8 (silent garbage or a crash).
  * NVFP4TritonKVCacheMethod: the stock NVFP4 method allocates a full-pool FP8 dequant workspace for FlashInfer
    prefill; the Triton path never reads it, so this variant declares none (more tokens in the same memory).
  * swa_quant_kwargs: SWA pools (Gemma 4) get one NVFP4 method per sub-pool (v0.5.20 passes none, and allocating
    the FP4 dtype without one crashes), with the checkpoint's per-layer scales remapped to sub-pool layer indices.

Global-scale rule (must match what the store path used; see NVFP4 docs in kv4tri_nvfp4_attention.py):
  layer.k_scale set      -> layer.k_scale_float (the backend passes layer.k_scale to set_kv_buffer)
  layer.k_scale is None  -> SWAKVPool / HybridLinearKVPool: 1.0 (their set_kv_buffer default);
                            plain MHATokenToKVPool: quant_method.k_scales_float[layer_id] (its None fallback).

NUMINOUS_KV4TRI=0 disables every hook (stock behaviour).
"""
from __future__ import annotations

import logging
import os

import torch

try:  # installed inside SGLang by kv4tri_hooks.py
    from sglang.kernels.ops.attention import numinous_kv4tri_attention as _k
    from sglang.kernels.ops.attention import numinous_kv4tri_gqa as _gqa
except ImportError:  # repo layout (tests)
    import extend_attention_gqa as _gqa  # type: ignore[no-redef]
    import kv4tri_nvfp4_attention as _k  # type: ignore[no-redef]

# Long prefill chunks are tensor-core bound: there the prefix is gathered + dequantized once per layer into a dense
# workspace and the GQA-packed FP8 kernel (operations/kernels/extend_attention_gqa.py) runs on it. Measured on
# rtx-qwen38, 4096 new tokens over a 32k prefix (CUDA graph): Qwen 3.8 16.8 ms (native FP4 33 ms tuned, stock Triton
# FP8 33.9 ms), Qwen 3.6 9.3 ms (32.0 / 22.7), Gemma global 20.8 ms (108.7 / 68.3); the gather-dequant itself is
# 0.02-0.12 ms. Extends up to NUMINOUS_KV4TRI_SPLIT_EXTEND_MAX new tokens (default 128) and masked extends use the
# split-KV FP4 kernel (nvfp4_verify_fwd): few new tokens over long prefixes are memory bound, and without a KV split
# they ran on a handful of SMs (the round-3 repeat-to-repeat slowdown suspect).
# workspace bound in prefix tokens (K+V): 256k tokens = 0.54 GB for Qwen 3.8 / Gemma global in FP8; above it the
# native kernel runs (correct, slower)
DQ_MAX_TOKENS = int(os.environ.get("NUMINOUS_KV4TRI_DQ_MAX_TOKENS", "262144"))
DQ_DTYPE = os.environ.get("NUMINOUS_KV4TRI_DQ_DTYPE", "fp8")  # fp8 | bf16 (sliding-window layers always use bf16)

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get("NUMINOUS_KV4TRI", "1") != "0"


def triton_only(server_args) -> bool:
    """True when both prefill and decode attention resolve to the Triton backend."""
    if not enabled():
        return False
    from sglang.srt.arg_groups.overrides import attention_backends_of, resolved_view

    prefill, decode = attention_backends_of(resolved_view(server_args))
    return prefill == "triton" and decode == "triton"


# ---- quant method variant ----------------------------------------------------------------------------------------
_TRITON_CLS = None


def _triton_cls():
    global _TRITON_CLS
    if _TRITON_CLS is None:
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import NVFP4KVCacheMethod

        class NVFP4TritonKVCacheMethod(NVFP4KVCacheMethod):
            """NVFP4 KV read natively by the kv4tri Triton kernels: no FP8 dequant workspace."""

            def dequant_workspace_dtype(self):
                return None

        _TRITON_CLS = NVFP4TritonKVCacheMethod
    return _TRITON_CLS


def adapt_quant_method(quant_method, server_args):
    """Called on every FP4 KV quant method SGLang builds; switches NVFP4 to the workspace-free variant when the
    Triton backend serves both phases."""
    if quant_method is not None and getattr(quant_method, "name", "") == "nvfp4" and triton_only(server_args):
        quant_method.__class__ = _triton_cls()
        logger.info("kv4tri: NVFP4 KV served by native Triton kernels (no FP8 dequant workspace)")
    return quant_method


def workspace_bytes_per_token(server_args, n_heads: int, head_dim: int, kv_size: int) -> int:
    """Bytes the pool-size calculator reserved per token for the FlashInfer FP8 dequant workspace that the Triton
    path does not allocate (to be subtracted back)."""
    return n_heads * head_dim * 2 * kv_size if triton_only(server_args) else 0


def swa_quant_kwargs(configurator, swa_layer_ids, full_layer_ids) -> dict:
    """NVFP4 quant methods for the full and SWA sub-pools of an SWAKVPool, scales indexed by sub-pool layer."""
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import get_kv_cache_quant_method

    all_ids = list(full_layer_ids) + list(swa_layer_ids)
    loaded = configurator._build_fp4_quant_method(num_layers=(max(all_ids) + 1) if all_ids else 1)
    if loaded is None:
        return {}
    out = []
    for ids in (full_layer_ids, swa_layer_ids):
        qm = get_kv_cache_quant_method(loaded.name, num_layers=max(len(ids), 1), device=configurator.device)
        if isinstance(loaded, _triton_cls()):
            qm.__class__ = _triton_cls()
        for i, g in enumerate(ids):
            if g < len(loaded.k_scales_float):
                qm.k_scales_float[i] = loaded.k_scales_float[g]
                qm.v_scales_float[i] = loaded.v_scales_float[g]
        qm.k_scales_gpu.copy_(torch.tensor(qm.k_scales_float, dtype=torch.float32))
        qm.v_scales_gpu.copy_(torch.tensor(qm.v_scales_float, dtype=torch.float32))
        out.append(qm)
    return {"numinous_quant_methods": tuple(out)}


# ---- pool access -------------------------------------------------------------------------------------------------
def _nvfp4_pools(pool) -> list:
    subs = [pool] + [getattr(pool, a) for a in ("full_kv_pool", "swa_kv_pool") if getattr(pool, a, None) is not None]
    return [p for p in subs if getattr(getattr(p, "quant_method", None), "name", "") == "nvfp4"]


def is_nvfp4_pool(pool) -> bool:
    return bool(_nvfp4_pools(pool))


def _state(backend) -> bool:
    st = getattr(backend, "_numinous_kv4tri", None)
    if st is None:
        st = enabled() and is_nvfp4_pool(backend.token_to_kv_pool)
        backend._numinous_kv4tri = st
        if st:
            logger.info("kv4tri: TritonAttnBackend routes NVFP4 KV to the native FP4 kernels")
    return st


def _layer_global_scale(layer, fallback: float) -> tuple[float, float]:
    if getattr(layer, "k_scale", None) is None:
        return fallback, fallback
    ks, vs = getattr(layer, "k_scale_float", None), getattr(layer, "v_scale_float", None)
    if ks is None or vs is None:
        cached = getattr(layer, "_numinous_kv4tri_gs", None)
        if cached is None:
            cached = (float(layer.k_scale), float(layer.v_scale))
            layer._numinous_kv4tri_gs = cached
        return cached
    return float(ks), float(vs)


def layer_view(pool, layer):
    """(k_fp4, v_fp4, k_sf, v_sf, k_global_scale, v_global_scale) for one attention layer of an NVFP4 pool."""
    lid = layer.layer_id
    if hasattr(pool, "layers_mapping") and hasattr(pool, "swa_kv_pool"):  # SWAKVPool
        if hasattr(pool, "_wait_for_layer"):
            pool._wait_for_layer(lid)
        local, is_swa = pool.layers_mapping[lid]
        sub = pool.swa_kv_pool if is_swa else pool.full_kv_pool
        kp, vp, ks, vs = sub.get_raw_kv_buffer(local + sub.start_layer)
        kg, vg = _layer_global_scale(layer, 1.0)
    elif hasattr(pool, "full_attention_layer_id_mapping"):  # HybridLinearKVPool
        kp, vp, ks, vs = pool.get_raw_kv_buffer(lid)
        kg, vg = _layer_global_scale(layer, 1.0)
    else:  # MHATokenToKVPool
        if getattr(pool, "layer_transfer_counter", None) is not None:
            pool.layer_transfer_counter.wait_until(lid - pool.start_layer)
        kp, vp, ks, vs = pool.get_raw_kv_buffer(lid)
        if getattr(layer, "k_scale", None) is None:
            kg, vg = pool.quant_method.k_scales_float[lid], pool.quant_method.v_scales_float[lid]
        else:
            kg, vg = _layer_global_scale(layer, 1.0)
    return kp, vp, ks, vs, kg, vg


def _unsupported(backend, layer, forward_batch, sinks, score_mod, aux_tensors, phase: str) -> None:
    why = []
    if sinks is not None:
        why.append("attention sinks")
    if score_mod is not None or aux_tensors is not None:
        why.append("score_mod")
    if getattr(backend, "dcp_size", 1) > 1:
        why.append("decode context parallel")
    if getattr(backend, "use_mla", False):
        why.append("MLA")
    if getattr(backend, "page_size", 1) != 1:
        why.append(f"page_size {backend.page_size}")
    if (getattr(layer, "xai_temperature_len", -1) or -1) > 0:
        why.append("xai temperature")
    if layer.qk_head_dim != layer.v_head_dim:
        why.append("qk_head_dim != v_head_dim")
    if phase == "extend":
        if getattr(backend, "enable_deterministic", False):
            why.append("deterministic unified extend")
        if getattr(forward_batch, "mha_one_shot", False):
            why.append("MHA one-shot prefill")
    if why:
        raise NotImplementedError(f"kv4tri NVFP4 Triton {phase}: unsupported with FP4 KV: {', '.join(why)}")


def _store_scales(backend, layer):
    """Global scales handed to the pool's NVFP4 store, as device tensors. The pools' own default is the python
    float 1.0 (SWAKVPool / HybridLinearKVPool) which NVFP4KVQuantizeUtil.quantize turns into a fresh CUDA tensor
    every call: a host-to-device copy that CUDA-graph capture rejects (measured at the first Qwen 3.6 boot:
    "Cannot copy between CPU and CUDA tensors during CUDA graph capture"). Same values, no copy."""
    if getattr(layer, "k_scale", None) is not None:
        return layer.k_scale, layer.v_scale
    pool = backend.token_to_kv_pool
    if not (hasattr(pool, "layers_mapping") or hasattr(pool, "full_attention_layer_id_mapping")):
        qm = pool.quant_method  # plain MHA pool: its None fallback reads the method's per-layer scales
        lid = layer.layer_id
        return qm.k_scales_gpu[lid:lid + 1], qm.v_scales_gpu[lid:lid + 1]
    one = getattr(backend, "_numinous_kv4tri_one", None)
    if one is None:
        one = backend._numinous_kv4tri_one = torch.ones(1, dtype=torch.float32, device=pool.device
                                                        if hasattr(pool, "device") else "cuda")
    return one, one


def _write_loc(backend, forward_batch):
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc

    md = backend.forward_metadata
    return KVWriteLoc(forward_batch.out_cache_loc, md.swa_out_cache_loc,
                      full_loc=getattr(md, "out_cache_loc_full_physical", None))


def _logit_cap(layer) -> float:
    from sglang.srt.layers.attention.triton_backend import logit_capping_mod

    return float(logit_capping_mod(layer.logit_capping_method, layer.logit_cap) or 0.0)


def forward_decode(backend, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None, score_mod=None,
                   aux_tensors=None):
    """Returns the attention output, or None when the pool is not NVFP4 (stock path continues)."""
    if not _state(backend):
        return None
    _unsupported(backend, layer, forward_batch, sinks, score_mod, aux_tensors, "decode")
    md = backend.forward_metadata
    H, D, Dv = layer.tp_q_head_num, layer.qk_head_dim, layer.v_head_dim
    q = q.reshape(-1, H * D)
    o = q.new_empty((q.shape[0], H * Dv))
    if save_kv_cache:
        # RadixAttention hands 3-D K/V; the NVFP4 quantizer requires [tokens, heads, dim]
        k = k.reshape(-1, layer.tp_k_head_num, D)
        v = v.reshape(-1, layer.tp_k_head_num, Dv)
        backend._set_kv_buffer(forward_batch, layer, _write_loc(backend, forward_batch), k, v,
                               *_store_scales(backend, layer))
    num_kv_splits = md.num_kv_splits
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        kv_indptr, kv_indices = md.window_kv_indptr, md.window_kv_indices
        if getattr(md, "window_num_kv_splits", None) is not None:
            num_kv_splits = md.window_num_kv_splits
    else:
        kv_indptr, kv_indices = md.kv_indptr, md.kv_indices
    attn_logits = md.attn_logits
    if md.swa_attn_logits is not None and layer.v_head_dim == backend.swa_v_head_dim:
        attn_logits = md.swa_attn_logits
    kp, vp, ks, vs, kg, vg = layer_view(backend.token_to_kv_pool, layer)
    _k.nvfp4_decode_fwd(q.view(-1, H, D), kp, vp, ks, vs, o.view(-1, H, Dv), kv_indptr, kv_indices, attn_logits,
                        md.attn_lse, num_kv_splits, backend.max_kv_splits, layer.scaling, kg, vg, _logit_cap(layer))
    return o


def forward_extend(backend, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None, score_mod=None,
                   aux_tensors=None):
    """Returns the attention output, or None when the pool is not NVFP4 (stock path continues)."""
    if not _state(backend):
        return None
    if k is None or v is None:
        raise NotImplementedError("kv4tri NVFP4 Triton extend: KV read-back (k=None) is unsupported with FP4 KV")
    _unsupported(backend, layer, forward_batch, sinks, score_mod, aux_tensors, "extend")
    from sglang.srt.layers.radix_attention import AttentionType

    md = backend.forward_metadata
    H, KVH, D, Dv = layer.tp_q_head_num, layer.tp_k_head_num, layer.qk_head_dim, layer.v_head_dim
    attn_out = getattr(forward_batch, "_attn_output", None)
    o = attn_out if attn_out is not None else q.new_empty((q.shape[0], H * Dv))
    k = k.reshape(-1, KVH, D)
    v = v.reshape(-1, KVH, Dv)
    if save_kv_cache:
        backend._set_kv_buffer(forward_batch, layer, _write_loc(backend, forward_batch), k, v,
                               *_store_scales(backend, layer))
    causal = not (
        layer.is_cross_attention
        or layer.attn_type == AttentionType.ENCODER_ONLY
        or (layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
            and getattr(backend, "allow_bidirectional_attention_in_extend", False))
    )
    window = -1
    kv_indptr, kv_indices = md.kv_indptr, md.kv_indices
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        window = layer.sliding_window_size
        kv_indptr, kv_indices = md.window_kv_indptr, md.window_kv_indices
    # custom masks (EAGLE verify; Gemma 4 image prefill builds a bidirectional mask) span the unwindowed sequence;
    # on sliding layers SGLang's window_kv_offsets locate the windowed prefix inside each mask row
    woff = md.window_kv_offsets if (window > 0 and md.custom_mask is not None) else None
    kp, vp, ks, vs, kg, vg = layer_view(backend.token_to_kv_pool, layer)
    cap = _logit_cap(layer)
    max_len = int(md.max_extend_len)
    lens = _cpu_lens(forward_batch, window)
    use_dq = (causal and md.custom_mask is None and not cap and lens is not None
              and max_len > _k.SPLIT_EXTEND_MAX)
    if causal and not use_dq:
        # EAGLE verify, draft extend, incremental prefills (radix hits), chunk tails, masked (image) prefills:
        # split-KV over the FP4 prefix, so a few new tokens over a 100k prefix still spread over the whole GPU
        _k.nvfp4_verify_fwd(q.reshape(-1, H, D), k.contiguous(), v.contiguous(), o.view(-1, H, Dv), kp, vp, ks, vs,
                            md.qo_indptr, kv_indptr, kv_indices, max_len, layer.scaling, kg, vg,
                            sliding_window_size=window, custom_mask=md.custom_mask, mask_indptr=md.mask_indptr,
                            logit_cap=cap, window_kv_offsets=woff,
                            num_splits=_verify_splits(backend, md, KVH) if max_len <= _k.verify_max_len(H // KVH)
                            else None)
        return o
    if use_dq:
        # long prefill chunks: tensor-core bound; dequantize each group of requests' prefixes (<= workspace cap,
        # grown to the largest single request) and run the GQA-packed FP8 kernel per group
        _dq_extend(backend, q.reshape(-1, H, D), k.contiguous(), v.contiguous(), o.view(-1, H, Dv), kp, vp, ks, vs,
                   md, kv_indptr, kv_indices, lens, window, layer.scaling, kg, vg, KVH, D, H // KVH)
        return o
    _k.nvfp4_extend_fwd(q.reshape(-1, H, D), k.contiguous(), v.contiguous(), o.view(-1, H, Dv), kp, vp, ks, vs,
                        md.qo_indptr, kv_indptr, kv_indices, md.max_extend_len, layer.scaling, kg, vg,
                        sliding_window_size=window, custom_mask=md.custom_mask, mask_indptr=md.mask_indptr,
                        is_causal=causal, logit_cap=cap, window_kv_offsets=woff)
    return o


def _cpu_lens(forward_batch, window: int):
    """(prefix tokens read per request, new tokens per request) on the host, or None (gpu-only batches)."""
    pre = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    new = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if pre is None or new is None:
        return None
    pre = [int(x) for x in pre]
    if window > 0:
        pre = [min(x, window) for x in pre]
    return pre, [int(x) for x in new]


def _dq_groups(prefix_lens, cap: int):
    """Consecutive request ranges [i, j) whose prefix sum fits the workspace (a single request may exceed it)."""
    groups, i, acc = [], 0, 0
    for r, p in enumerate(prefix_lens):
        if r > i and acc + p > cap:
            groups.append((i, r))
            i, acc = r, 0
        acc += p
    groups.append((i, len(prefix_lens)))
    return groups


def _dq_extend(backend, q, k, v, o, kp, vp, ks, vs, md, kv_indptr, kv_indices, lens, window, sm_scale, kg, vg,
               kvh, d, group):
    pre, new = lens
    ws_dtype = torch.bfloat16 if (window > 0 or DQ_DTYPE == "bf16") else torch.float8_e4m3fn
    cap = max(DQ_MAX_TOKENS, max(pre) if pre else 0)
    starts = [0]
    for p in pre:
        starts.append(starts[-1] + p)
    cfg = gqa_config(d, group)
    for i, j in _dq_groups(pre, cap):
        n = starts[j] - starts[i]
        kw, vw, ident = _workspace(backend, max(n, 1), kvh, d, ws_dtype)
        if n:
            _k.nvfp4_gather_dequant(kp, ks, kv_indices[starts[i]:starts[j]], kw)
            _k.nvfp4_gather_dequant(vp, vs, kv_indices[starts[i]:starts[j]], vw)
        kvp = kv_indptr[i:j + 1] - starts[i] if starts[i] else kv_indptr[i:j + 1]
        mx = max(new[i:j]) if j > i else 0
        if mx == 0:
            continue
        _gqa.extend_attention_gqa(q, k, v, o, kw, vw, md.qo_indptr[i:j + 1], kvp, ident, mx, sm_scale, kg, vg,
                                  window, fp8=ws_dtype == torch.float8_e4m3fn, **cfg)


def gqa_config(head_dim: int, group: int) -> dict:
    """GQA-packed kernel tiles measured on RTX PRO 6000 (same table as gemma_extend_hooks._numinous_gqa_config)."""
    if head_dim >= 512:
        return dict(block_t=8, block_n=128, dk=64, dv=256, num_warps=8, num_stages=3)
    if group >= 4:
        return dict(block_t=8, block_n=128, dk=64, dv=head_dim, num_warps=8, num_stages=3)
    return dict(block_t=32, block_n=64, dk=64, dv=head_dim, num_warps=8, num_stages=1)


def _verify_splits(backend, md, kv_heads: int) -> int:
    sm = getattr(backend, "device_core_count", None) or 188
    return _k.verify_splits(md.qo_indptr.shape[0] - 1, kv_heads, sm_count=int(sm))


def _workspace(backend, n_tokens: int, kvh: int, d: int, dtype):
    """Grow-only dense K/V workspace (+ identity indices), one per (dtype, heads, dim), reused by every layer."""
    cache = getattr(backend, "_numinous_kv4tri_ws", None)
    if cache is None:
        cache = backend._numinous_kv4tri_ws = {}
    key = (dtype, kvh, d)
    ws = cache.get(key)
    if ws is None or ws[0].shape[0] < n_tokens:
        size = -(-n_tokens // 16384) * 16384
        cache.pop(key, None)
        dev = backend.token_to_kv_pool.device if hasattr(backend.token_to_kv_pool, "device") else "cuda"
        ws = (torch.empty(size, kvh, d, dtype=dtype, device=dev), torch.empty(size, kvh, d, dtype=dtype, device=dev),
              torch.arange(size, dtype=torch.int64, device=dev))
        cache[key] = ws
    return ws
