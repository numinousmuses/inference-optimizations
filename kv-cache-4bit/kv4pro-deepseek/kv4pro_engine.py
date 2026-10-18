"""Engine glue for the 4-bit DeepSeek V4 Pro KV cache (installed as sglang/srt/layers/attention/numinous_kv4pro_engine.py).

Called from the anchored hooks in operations/patches/kv4pro_hooks.py; inert unless NUMINOUS_KV4PRO is c4 or all
(see numinous_kv4pro, the kernel module, for the cell format and modes).

- ``kv4_pool_classes(pool)``: which DeepSeekV4TokenToKVPool sub-pools use the 4-bit cell (SWA window, c4); validates
  the configuration (non-unified KV, FlashMLA attention, no HiSparse, not SM120) and fails loudly otherwise.
- ``DeepSeekV4FP4KVPool``: a DeepSeekV4SingleKVPool whose rows are contiguous 4-bit cells; ``set_key_buffer_fused``
  quantizes normed + roped bf16 rows (the SWA store and the unfused compressor store both land here).
- ``forward_kv4(backend, ...)``: DeepseekV4AttnBackend.forward for layers that read a 4-bit pool.
    * prompt extends routed to sparse prefill by the stock rule (see _use_sparse_prefill): the upstream scheme (per-request positional
      gather of the SWA window + compressed rows into a bf16 workspace, flash_mla_sparse_fwd), with the 4-bit rows
      dequantized exactly by kv4_dequant_gather;
    * decode / target-verify / draft (graph-captured): the 4-bit rows each query selects are requantized into a
      FlashMLA 584-byte FP8 workspace (kv4_to_fp8_workspace: token t*W + j, remapped indices), then the stock FlashMLA
      sparse decode kernel runs on it. Query rows are processed in chunks of NUMINOUS_KV4PRO_WS_TOKENS (default 256)
      so the workspace stays bounded (~0.2 GB at SWA width 192 + c4 top-1024); workspaces are allocated once per
      width and never freed (captured graphs keep pointers into them); the pool configurator reserves their bound
      (numinous_kv4pro.workspace_bytes) so the non-KV headroom equals stock.
  Layers whose pools are all FP8 (c128 layers in mode c4) return None and run the stock code.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from sglang.srt.layers.attention import numinous_kv4pro as K

_POOL_CLS = None


def fp4_pool_cls():
    global _POOL_CLS
    if _POOL_CLS is not None:
        return _POOL_CLS
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool

    class DeepSeekV4FP4KVPool(DeepSeekV4SingleKVPool):
        """Rows are contiguous 4-bit cells (numinous_kv4pro format): token t of layer L at byte t * CELL."""

        numinous_kv4 = True

        def __init__(self, *args, **kwargs):
            self.kv4_rope = K.mode()[2]
            super().__init__(*args, **kwargs)

        def get_bytes_per_token(self) -> int:
            return K.cell_bytes(self.kv4_rope)

        def create_buffer(self, *, num_pages: int):
            bpt = self.get_bytes_per_token()
            self.kv_cache_total_dim = bpt
            self.bytes_per_page_padded = self.page_size * bpt
            return torch.zeros(num_pages, self.page_size * bpt, dtype=torch.uint8, device=self.device)

        def get_key_buffer(self, layer_id: int):
            return self.kv_buffer[layer_id]

        def set_key_buffer(self, layer_id, loc, cache_nope_fp8_rope_bf16_pack):
            raise NotImplementedError("the packed FlashMLA store does not apply to the 4-bit pool; use set_key_buffer_fused")

        def set_key_buffer_fused(self, layer_id: int, loc: torch.Tensor, cache_k: torch.Tensor) -> None:
            K.kv4_store(self.kv_buffer[layer_id], loc, cache_k.reshape(-1, K.D), self.kv4_rope)

    _POOL_CLS = DeepSeekV4FP4KVPool
    return _POOL_CLS


def kv4_pool_classes(pool):
    """(swa_cls or None, c4_cls or None) for DeepSeekV4TokenToKVPool.__init__."""
    swa4, c44, _ = K.mode()
    if not (swa4 or c44):
        return None, None
    from sglang.srt.runtime_context import get_exec, get_platform

    problems = []
    if getattr(pool, "_unified_kv", False):
        problems.append("unified KV (SGLANG_DSV4_UNIFIED_KV*)")
    if get_exec().kernel.dsv4_attn_backend == "trtllm":
        problems.append("--dsv4-attn-backend trtllm (reads only its uniform FP8 cell)")
    if get_platform().is_sm120:
        problems.append("SM120 (no FlashMLA sparse decode)")
    try:
        from sglang.srt.runtime_context import get_memory

        if get_memory().enable_hisparse:
            problems.append("HiSparse")
    except Exception:
        pass
    if problems:
        raise ValueError("NUMINOUS_KV4PRO is incompatible with: " + ", ".join(problems))
    cls = fp4_pool_cls()
    return (cls if swa4 else None), (cls if c44 else None)


def _is4(p) -> bool:
    return p is not None and getattr(p, "numinous_kv4", False)


_WS: dict = {}
_KEEP: list = []  # every workspace ever handed to a (possibly captured) kernel stays alive


def _chunk_tokens() -> int:
    return K.ws_tokens()


def _workspace(device, width: int, page: int) -> torch.Tensor:
    key = (str(device), width, page)
    ws = _WS.get(key)
    if ws is None:
        rows = _chunk_tokens() * width
        ws = torch.empty(-(-rows // page), K.fp8_cell_page_bytes(page), dtype=torch.uint8, device=device)
        _WS[key] = ws
        _KEEP.append(ws)
    return ws


def _paged_view(buf: torch.Tensor, page: int) -> torch.Tensor:
    return buf[:, : page * K.FP8_CELL].view(buf.shape[0], page, 1, K.FP8_CELL).view(torch.float8_e4m3fn)


def _match(x, n, value):
    if x is None or x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    pad = torch.full((n - x.shape[0], *x.shape[1:]), value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], 0)


def forward_kv4(backend, *, q, layer, forward_batch, compress_ratio, core_attn_metadata, attn_sink) -> Optional[torch.Tensor]:
    pool = backend.token_to_kv_pool
    layer_id = layer.layer_id
    swa_pool = pool.swa_kv_pool
    ext_pool = pool.layer_mapping[layer_id].compress_kv_pool if compress_ratio else None
    swa4, ext4 = _is4(swa_pool), _is4(ext_pool)
    if not (swa4 or ext4):
        return None
    rope = K.mode()[2]
    if _use_sparse_prefill(forward_batch, q):
        return _prefill(backend, q, layer_id, compress_ratio, forward_batch, pool, core_attn_metadata, attn_sink,
                        swa4, ext4, rope)
    return _decode(backend, q, layer_id, compress_ratio, pool, core_attn_metadata, attn_sink, swa4, ext4, rope)


def _use_sparse_prefill(forward_batch, q) -> bool:
    """The stock routing (deepseek_v4_backend.forward, v0.5.20): prompt extends above the indexer threshold, or all
    of them with SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1 (the kv4pro arms set it: the prefill snapshot logic keys on
    the same condition, and the FP8-workspace decode path would re-gather every query's rows)."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsv4.metadata import _LARGE_INDEXER_QUERY_THRESHOLD

    return forward_batch.forward_mode.is_extend_without_speculative() and (
        q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get())


def _decode(backend, q, layer_id, compress_ratio, pool, core, attn_sink, swa4, ext4, rope):
    from sgl_kernel.flash_mla import FlashMLASchedMeta, flash_mla_with_kvcache

    if q.ndim == 3:
        q = q.unsqueeze(1)
    T = q.shape[0]
    if T == 0:  # idle DP rank
        return q.new_empty(0, q.shape[2], backend.head_dim_v)
    swa_idx = _match(core.swa_page_indices, T, 0)
    swa_len = _match(core.swa_topk_lengths, T, 1)
    ext_idx = ext_len = None
    if compress_ratio == 4:
        ext_idx = _match(core.c4_sparse_page_indices, T, -1)
        ext_len = _match(core.c4_sparse_topk_lengths, T, 1)
    elif compress_ratio == 128:
        ext_idx = _match(core.c128_page_indices, T, -1)
        ext_len = _match(core.c128_topk_lengths_clamp1, T, 1)
    swa_idx = swa_idx.reshape(T, -1)
    if ext_idx is not None:
        ext_idx = ext_idx.reshape(T, -1)
    swa_page = pool.swa_page_size
    swa_buf = pool.get_swa_key_buffer_radix(layer_id)
    ext_buf = pool.get_extra_key_buffer(layer_id) if compress_ratio else None
    ext_page = pool.get_extra_key_page_size(layer_id) if compress_ratio else None
    ws_s = _workspace(q.device, swa_idx.shape[1], swa_page) if swa4 else None
    ws_e = _workspace(q.device, ext_idx.shape[1], ext_page) if ext4 else None
    chunk = _chunk_tokens()
    outs = []
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        si, sl = swa_idx[t0:t1], swa_len[t0:t1]
        if swa4:
            si = K.kv4_to_fp8_workspace(swa_buf, si, sl, ws_s, swa_page, rope)
            sv = _paged_view(ws_s, swa_page)
        else:
            sv = _paged_view(swa_buf.view(torch.uint8), swa_page)
        ei = el = ev = None
        if ext_idx is not None:
            ei, el = ext_idx[t0:t1], ext_len[t0:t1]
            if ext4:
                ei = K.kv4_to_fp8_workspace(ext_buf, ei, el, ws_e, ext_page, rope)
                ev = _paged_view(ws_e, ext_page)
            else:
                ev = _paged_view(ext_buf.view(torch.uint8), ext_page)
        # the batch-wide scheduler metadata matches only the full batch shape; chunks get their own
        meta = core.get_flashmla_metadata(compress_ratio) if (t0 == 0 and t1 == T) else FlashMLASchedMeta()
        o = flash_mla_with_kvcache(
            q=q[t0:t1], k_cache=sv, head_dim_v=backend.head_dim_v, block_table=None, cache_seqlens=None,
            tile_scheduler_metadata=meta, softmax_scale=backend.softmax_scale, is_fp8_kvcache=True,
            indices=si.unsqueeze(1), topk_length=sl, attn_sink=attn_sink, extra_k_cache=ev,
            extra_indices_in_kvcache=None if ei is None else ei.unsqueeze(1), extra_topk_length=el)[0]
        outs.append(o.squeeze(1))
    return outs[0] if len(outs) == 1 else torch.cat(outs, 0)


def _prefill(backend, q, layer_id, compress_ratio, forward_batch, pool, core, attn_sink, swa4, ext4, rope):
    """DeepseekV4AttnBackend._forward_prefill_sparse with 4-bit-aware dequantization (v0.5.20 semantics)."""
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged

    q_flat = q.squeeze(1) if q.ndim == 4 else q
    cache = backend.forward_metadata.sparse_prefill_cache
    if cache is None:
        cache = backend._build_sparse_prefill_chunk_cache(forward_batch, num_qo_tokens=q_flat.shape[0])
        backend.forward_metadata.sparse_prefill_cache = cache
    compressed_slice = extra_k_cache = extra_page_size = flat_token_ids = None
    if compress_ratio == 0:
        workspace = backend.sparse_prefill_workspace.get(cache.swa_token_ids.shape[0])
        combined_indices, combined_lens = cache.c0_combined_indices, cache.c0_combined_lens
        swa_slice = workspace
    else:
        extra_page_size = pool.get_extra_key_page_size(layer_id)
        extra_k_cache = pool.get_extra_key_buffer(layer_id)
        if compress_ratio == 128:
            cache.ensure_c128(core.c128_page_indices)
            flat_token_ids = cache.c128_flat_token_ids
            combined_indices, combined_lens = cache.c128_combined_indices, cache.c128_combined_lens
        else:
            cache.ensure_c4(core.page_table, extra_page_size)
            flat_token_ids = cache.c4_flat_token_ids
            combined_indices, combined_lens = cache.combine_c4_layer(
                c4_sparse_raw_indices=core.c4_sparse_raw_indices[: cache.num_qo_tokens])
        n_compressed = flat_token_ids.shape[0]
        workspace = backend.sparse_prefill_workspace.get(n_compressed + cache.swa_token_ids.shape[0])
        compressed_slice = workspace[:n_compressed]
        swa_slice = workspace[n_compressed:]
    if compressed_slice is not None:
        if ext4:
            K.kv4_dequant_gather(extra_k_cache, flat_token_ids, rope, out=compressed_slice)
        else:
            dequantize_k_cache_paged(extra_k_cache, flat_token_ids, page_size=extra_page_size, out=compressed_slice)
    swa_buf = pool.get_swa_key_buffer_radix(layer_id)
    if swa4:
        K.kv4_dequant_gather(swa_buf, cache.swa_token_ids, rope, out=swa_slice)
    else:
        dequantize_k_cache_paged(swa_buf, cache.swa_token_ids, page_size=cache.swa_page_size, out=swa_slice)
    o, _, _ = flash_mla_sparse_fwd(q=q_flat, kv=workspace, indices=combined_indices.unsqueeze(1),
                                   sm_scale=backend.softmax_scale, d_v=backend.head_dim_v, attn_sink=attn_sink,
                                   topk_length=combined_lens)
    return o
