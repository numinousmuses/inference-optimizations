"""GLM-5.3 mixed FP8 + FP4 KV cache inside SGLang v0.5.20: the pool, the attention routing, the scratch.

Installed by operations/patches/kvmix_glm_hooks.py as sglang/srt/layers/attention/dsa/kvmix_glm_engine.py next to
glm_kv4.py (FP4 row format and dequant tiles) and kvmix_glm.py (mixed kernels); kvmix_core.py (allocator, policy,
promotion) goes to sglang/srt/mem_cache/. Everything is behind NUMINOUS_KVMIX=1.

What changes when it is on (GLM-5.3, --kv-cache-dtype fp8_e4m3, `trtllm` DSA roles, SM100):
  pricing  calculate_mla_kv_cache_dim returns the blended row (kvmix_core.blended_row_bytes: FP4 fraction f of the
           latent byte budget) so the configurator's token count already reflects the split; the indexer cache is
           priced per token as before (it is shared by both pools).
  pool     KVMixDSATokenToKVPool: FP8 storage = ONE tensor holding every layer's [N8 + page, 576] buffer back to back
           plus an FP8 scratch tail of NUMINOUS_KVMIX_SCRATCH_ROWS rows (default 589,824); kv_buffer[l] are views into it, so the
           TRT-LLM op for layer l can be handed the tail of the tensor starting at layer l (its pages = layer l's
           pool followed by the later layers and the scratch) and one scratch serves all 78 layers. FP4 storage =
           one tensor of [layers x (N4 + page), 352]. Indexer cache, page 64, spec tail slots: inherited, sized for
           N8 + N4 tokens (unified slot space, kvmix_core.KVMixAllocator).
  write    the trtllm path keeps the bf16 latent from before the FP8 quantize + RoPE kernel and hands it, with the
           FP8 rows, to write_mixed: FP8 slots get the FP8 bytes, FP4 slots a quantization of the bf16 latent.
  read     engine_forward replaces the TRT-LLM call. Batches without any FP4 row (checked once per eager batch)
           run the stock call on layer l's view. Otherwise decode / verify / draft extend run redirect_gather (FP4
           entries -> scratch rows, FP8 entries unchanged, fixed shapes, captured in the CUDA graph) and one
           TRT-LLM call over the whole batch; plain prefill dequantizes each FP4-holding request's sequence once
           (seq_dequant + redirect_seq), chunked by request ranges when it does not fit the scratch; a single
           request longer than the scratch sends its batch through the gather path (correct, per-token cost).
  graphs   redirect_gather in a captured graph needs max_num_tokens x K_pad <= scratch rows (asserted in
           init_cuda_graph_state; 192 x 2048 = 393,216 <= 524,288 default).

Memory (derived): scratch 589,824 x 576 B = 324 MB once per rank (not per layer); slot_to_pos int32 per slot;
tables int32 [tokens, K].
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.attention.dsa import glm_kv4 as kv4
from sglang.srt.layers.attention.dsa import kvmix_glm as mix
from sglang.srt.mem_cache import kvmix_core as core
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

logger = logging.getLogger("sglang.numinous")

SCRATCH_ROWS_ENV = "NUMINOUS_KVMIX_SCRATCH_ROWS"
DEFAULT_SCRATCH_ROWS = 9 * 64 * 1024  # 589,824 rows (324 MB): covers a full 524k-token context plus draft slots
ARCH_PREFIX = "GlmMoeDsa"
SCRATCH_PAGE = 64
ROW_BYTES = mix.ROW_BYTES
FP8_ROW_BYTES = mix.FP8_ROW_BYTES
enabled = core.enabled
log_once = core.log_once


def scratch_rows() -> int:
    s = int(os.environ.get(SCRATCH_ROWS_ENV, str(DEFAULT_SCRATCH_ROWS)))
    if s <= 0 or s % SCRATCH_PAGE:
        raise ValueError(f"{SCRATCH_ROWS_ENV}={s}: expected a positive multiple of {SCRATCH_PAGE}")
    return s


def enabled_for(hf_config, kv_cache_dtype) -> bool:
    """The mixed pool applies to GLM-5.3 (target and its NextN draft) served with an FP8 KV flag."""
    if not enabled():
        return False
    archs = getattr(hf_config, "architectures", None) or []
    if not any(str(a).startswith(ARCH_PREFIX) for a in archs):
        log_once("arch", "numinous kvmix: %s=1 but architectures %s are not GLM DSA; mixed pool off", core.ENV, archs)
        return False
    if kv_cache_dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"numinous kvmix: needs --kv-cache-dtype fp8_e4m3 (the trtllm DSA path and the FP8 pool "
                           f"are E4M3); got {kv_cache_dtype}")
    return True


def blended_dim() -> int:
    return core.blended_row_bytes(FP8_ROW_BYTES, ROW_BYTES, core.fp4_frac())


def split(total_tokens: int, page_size: int):
    return core.split_tokens(total_tokens, FP8_ROW_BYTES, ROW_BYTES, blended_dim(), core.fp4_frac(), page_size)


# ------------------------------------------------------------------------------------------------ pool


class KVMixDSATokenToKVPool(DSATokenToKVPool):
    """DSATokenToKVPool over two storages: FP8 rows (unified slots < boundary) and FP4 rows (slots >= boundary)."""

    numinous_kvmix = True

    def __init__(self, *args, **kwargs):
        size = kwargs.get("size", args[0] if args else None)
        page_size = kwargs.get("page_size")
        assert size is not None and page_size is not None, "kvmix pool: size and page_size are keyword/positional"
        self.n8, self.n4 = split(int(size), int(page_size))
        self.lay = mix.Layout(self.n8, self.n4, int(page_size))
        self.scratch_rows = scratch_rows()
        super().__init__(*args, **kwargs)
        self.dsa_kv_cache_store_fp8 = False  # raw 576-byte layout, not the 656-byte scaled one
        self.kv_cache_dim = FP8_ROW_BYTES
        assert self.store_dtype == torch.uint8 and self.dtype == torch.float8_e4m3fn
        assert self.kv_lora_rank == kv4.DIM_NOPE and self.qk_rope_head_dim == kv4.DIM_ROPE, (
            self.kv_lora_rank, self.qk_rope_head_dim)
        logger.warning("numinous kvmix pool: %d layers; %s; FP8 %d B/token/layer + FP4 %d B/token/layer (priced as "
                       "%d); scratch %d rows (%.0f MB); latent bytes %.2f GB (FP8 %.2f + FP4 %.2f)",
                       self.layer_num, self.lay.describe(), FP8_ROW_BYTES, ROW_BYTES, blended_dim(),
                       self.scratch_rows, self.scratch_rows * FP8_ROW_BYTES / 2**20,
                       (self.fp8_all.numel() + self.fp4_all.numel()) / 2**30, self.fp8_all.numel() / 2**30,
                       self.fp4_all.numel() / 2**30)

    def _create_buffers(self):
        from contextlib import nullcontext

        from sglang.srt.mem_cache.memory_pool import GPU_MEMORY_TYPE_KV_CACHE

        self.kv_cache_dim = FP8_ROW_BYTES
        lay = self.lay
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (torch.cuda.use_mem_pool(self.custom_mem_pool) if self.custom_mem_pool else nullcontext()):
                self.fp8_all = torch.zeros((self.layer_num * lay.fp8_rows + self.scratch_rows, 1, FP8_ROW_BYTES),
                                           dtype=torch.uint8, device=self.device)
                self.fp4_all = torch.zeros((self.layer_num * lay.fp4_rows, 1, ROW_BYTES), dtype=torch.uint8,
                                           device=self.device)
        r8, r4 = lay.fp8_rows, lay.fp4_rows
        self.kv_buffer = [self.fp8_all[l * r8:(l + 1) * r8] for l in range(self.layer_num)]
        self.fp4_buffer = [self.fp4_all[l * r4:(l + 1) * r4] for l in range(self.layer_num)]
        self.scratch = self.fp8_all[self.layer_num * r8:]
        assert self.scratch.shape[0] == self.scratch_rows

    def _clear_buffers(self):
        super()._clear_buffers()
        del self.fp4_buffer, self.fp8_all, self.fp4_all, self.scratch

    def get_kv_size_bytes(self):
        from sglang.srt.mem_cache.memory_pool import get_tensor_size_bytes

        n = get_tensor_size_bytes(self.fp8_all) + get_tensor_size_bytes(self.fp4_all)
        for index_k_cache in self.index_k_with_scale_buffer:
            n += get_tensor_size_bytes(index_k_cache)
        return n

    # ---- views the attention path uses ---------------------------------------------------------------------
    def layer_view(self, local_layer: int) -> torch.Tensor:
        """Rows from layer `local_layer`'s buffer to the end of the FP8 tensor (later layers + scratch), as the
        [pages, 1, 64, 576] fp8 tensor the TRT-LLM op reads. Slots of layer l are valid indices; the scratch starts at
        row scratch_base(local_layer)."""
        rows = self.fp8_all[local_layer * self.lay.fp8_rows:]
        return rows.view(torch.float8_e4m3fn).view(-1, 1, SCRATCH_PAGE, FP8_ROW_BYTES)

    def scratch_base(self, local_layer: int) -> int:
        return (self.layer_num - local_layer) * self.lay.fp8_rows

    # ---- doors -------------------------------------------------------------------------------------------------
    def write_mixed(self, layer, loc: torch.Tensor, k_fp8: torch.Tensor, k_rope_fp8: torch.Tensor,
                    k_bf16: Optional[torch.Tensor]) -> None:
        l = layer.layer_id - self.start_layer
        mix.write_mixed(self.lay, self.kv_buffer[l], self.fp4_buffer[l], loc, k_fp8, k_rope_fp8, k_bf16)

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope, layer_id_override=None):
        """Generic door (dense MHA prefill and other non-trtllm callers): any float dtype in."""
        from sglang.srt.runtime_context import get_parallel

        assert not get_parallel().dcp_enabled, "kvmix pool has no DCP write path"
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        l = layer_id - self.start_layer
        t = loc.numel()
        nope = cache_k_nope.reshape(t, kv4.DIM_NOPE)
        rope = cache_k_rope.reshape(t, kv4.DIM_ROPE)
        bf16 = None if nope.dtype == torch.float8_e4m3fn else nope
        n8 = nope if nope.dtype == torch.float8_e4m3fn else nope.to(torch.float8_e4m3fn)
        r8 = rope if rope.dtype == torch.float8_e4m3fn else rope.to(torch.float8_e4m3fn)
        mix.write_mixed(self.lay, self.kv_buffer[l], self.fp4_buffer[l], loc, n8, r8, bf16)

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("kvmix pool: combined-row set_kv_buffer is not a DSA write door (use set_mla_kv_buffer)")

    def get_value_buffer(self, layer_id: int):
        raise RuntimeError("kvmix pool: rows are mixed FP8/FP4; read them through kvmix_glm_engine.engine_forward")

    def get_mla_kv_buffer(self, layer, loc: torch.Tensor, dst_dtype: Optional[torch.dtype] = None):
        """Rows at unified slots loc as (nope [T, 1, 512], rope [T, 1, 64]) in dst_dtype (default bf16)."""
        dst_dtype = dst_dtype or torch.bfloat16
        l = layer.layer_id - self.start_layer
        out = torch.empty(loc.numel(), kv4.DIM_QK, dtype=torch.bfloat16, device=self.fp8_all.device)
        mix.gather_bf16(self.lay, self.kv_buffer[l], self.fp4_buffer[l], loc, out)
        out = out.to(dst_dtype)
        return out[:, : kv4.DIM_NOPE].unsqueeze(1).contiguous(), out[:, kv4.DIM_NOPE:].unsqueeze(1).contiguous()

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        tgt = tgt_loc.reshape(-1).to(torch.int64)
        src = src_loc.reshape(-1).to(torch.int64)
        mix.move_rows(self.lay, self.fp8_all, self.fp4_all, self.layer_num, tgt, src)
        for buf in self.index_key_cache.buffer:
            mix.index_move(buf, self.page_size, tgt, src)

    def kvmix_promote_pages(self, src_slots: torch.Tensor, dst_slots: torch.Tensor) -> None:
        """Requantize whole FP8 pages (src_slots, unified) into FP4 pages (dst_slots) for all layers and move the
        indexer pages with them. Both are [pages * page_size] with page-contiguous slots."""
        ps = self.page_size
        src = src_slots.reshape(-1).to(torch.int64)
        dst = dst_slots.reshape(-1).to(torch.int64)
        assert src.numel() == dst.numel() and src.numel() % ps == 0, (src.shape, dst.shape)
        mix.promote_rows(self.lay, self.fp8_all, self.fp4_all, self.layer_num, src, dst)
        src_pages = src[::ps] // ps
        dst_pages = dst[::ps] // ps
        for buf in self.index_key_cache.buffer:
            if buf.shape[0]:
                buf[dst_pages] = buf[src_pages]

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("kvmix pool: host offload of mixed rows is not supported")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("kvmix pool: host offload of mixed rows is not supported")


def build_allocator(pool, page_size: int, dtype, device, need_sort: bool, forward_stream=None):
    """The KVMixAllocator matching a KVMixDSATokenToKVPool (from the configurator hook)."""
    return core.KVMixAllocator(pool.n8, pool.n4, page_size, dtype, device, pool, need_sort, forward_stream)


# ------------------------------------------------------------------------------------------------ backend glue


class _State:
    def __init__(self, backend):
        self.pool = backend.token_to_kv_pool
        self.lay = self.pool.lay
        self.device = torch.device(backend.device)
        self.graph_table: Optional[torch.Tensor] = None
        self.eager_table: Optional[torch.Tensor] = None
        self.slot_to_pos: Optional[torch.Tensor] = None
        self.batch_key = None  # id(forward_batch) of the last eager batch examined
        self.batch_fp4_reqs: Optional[list] = None  # per request: holds any FP4 row (eager batches only)
        self.calls = {"stock": 0, "gather": 0, "seq": 0}

    def table(self, tokens: int, k: int) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            assert self.graph_table is not None, "kvmix: init_cuda_graph_state did not reserve the graph table"
            assert tokens <= self.graph_table.shape[0] and k == self.graph_table.shape[1], (
                tokens, k, self.graph_table.shape)
            return self.graph_table[:tokens]
        if self.eager_table is None or self.eager_table.shape[0] < tokens or self.eager_table.shape[1] != k:
            self.eager_table = torch.empty(tokens, k, dtype=torch.int32, device=self.device)
        return self.eager_table[:tokens]

    def slot_map(self) -> torch.Tensor:
        if self.slot_to_pos is None:
            self.slot_to_pos = torch.full((self.lay.n_slots,), -1, dtype=torch.int32, device=self.device)
        return self.slot_to_pos

    def fp4_requests(self, forward_batch, metadata) -> Optional[list]:
        """Per-row "holds an FP4 slot" flags of metadata.page_table_1 for an eager batch: one flag per request for
        plain extend / decode, one per token for verify and draft extend (None when unknown: take the mixed path)."""
        key = id(forward_batch)
        if self.batch_key == key:
            return self.batch_fp4_reqs
        pt = getattr(metadata, "page_table_1", None)  # [bs, max_len] slots (extend) or [T, max_len] (verify)
        flags = None
        if pt is not None and pt.dim() == 2 and pt.shape[0] > 0:
            flags = (pt >= self.lay.boundary).any(dim=1).tolist()  # one sync per eager batch
        self.batch_key, self.batch_fp4_reqs = key, flags
        return flags


def _state(backend) -> _State:
    st = getattr(backend, "_numinous_kvmix_state", None)
    if st is None:
        st = _State(backend)
        backend._numinous_kvmix_state = st
    return st


def attach(backend, model_runner) -> bool:
    """From DeepseekSparseAttnBackend.__init__: True when this backend serves the mixed pool."""
    pool = model_runner.token_to_kv_pool
    if not getattr(pool, "numinous_kvmix", False):
        if enabled():
            log_once("nopool", "numinous kvmix: %s=1 but the pool is %s; mixed attention off", core.ENV,
                     type(pool).__name__)
        return False
    why = (
        f"dsa roles prefill={backend.dsa_prefill_impl} decode={backend.dsa_decode_impl} (need trtllm/trtllm)"
        if backend.dsa_prefill_impl != "trtllm" or backend.dsa_decode_impl != "trtllm"
        else f"sm_{backend.device_sm_major}x (need SM100)" if backend.device_sm_major != 10
        else f"kv dtype {backend.kv_cache_dtype}" if backend.kv_cache_dtype != torch.float8_e4m3fn
        else ""
    )
    if why:
        raise RuntimeError(f"numinous kvmix: the mixed pool is built but this backend cannot read it: {why}")
    if os.environ.get("NUMINOUS_GLM_FLASHMLA_PREFILL", "0") == "1":
        raise RuntimeError("numinous kvmix: NUMINOUS_GLM_FLASHMLA_PREFILL=1 reads the pool as FP8; incompatible")
    if os.environ.get("NUMINOUS_GLM_KV4", "0") == "1":
        raise RuntimeError("numinous kvmix: NUMINOUS_GLM_KV4=1 (all-FP4 pool) and NUMINOUS_KVMIX=1 are exclusive")
    alloc = getattr(model_runner, "token_to_kv_pool_allocator", None)
    if alloc is not None and getattr(alloc, "kvmix", False):
        assert (alloc.n8, alloc.n4) == (pool.n8, pool.n4), ("allocator/pool split differ", alloc.n8, alloc.n4,
                                                             pool.n8, pool.n4)
    st = _state(backend)
    log_once("attach", "numinous kvmix attention active: %d heads, top-k %d, %s, scratch %d rows",
             backend.num_q_heads, backend.dsa_index_topk, st.lay.describe(), pool.scratch_rows)
    return True


def reserve_graph_scratch(backend, max_num_tokens: int) -> None:
    """From init_cuda_graph_state: the redirect table for captured batches; the scratch must hold tokens x K_pad."""
    st = _state(backend)
    k = ((backend.dsa_index_topk + 3) // 4) * 4
    tokens = max(int(max_num_tokens), 1)
    if tokens * k > st.pool.scratch_rows:
        raise RuntimeError(f"numinous kvmix: CUDA-graph batches of {tokens} tokens x top-k {k} need {tokens * k} "
                           f"scratch rows but {SCRATCH_ROWS_ENV} is {st.pool.scratch_rows}; raise it")
    if st.graph_table is None or st.graph_table.shape[0] < tokens:
        assert not torch.cuda.is_current_stream_capturing()
        st.graph_table = torch.empty(tokens, k, dtype=torch.int32, device=st.device)
        logger.warning("numinous kvmix: graph table %d tokens x %d, scratch %d rows", tokens, k, st.pool.scratch_rows)
    warm_kernels(st)


def warm_kernels(st: _State) -> None:
    """Run every kernel the captured path uses once, outside capture: the byte LUT is a CPU -> CUDA copy and the
    Triton kernels compile on first use; both are illegal / slow inside a CUDA graph capture. The capture warm-up
    runs do not reach them: their dummy batches hold no FP4 slot and take the stock call."""
    assert not torch.cuda.is_current_stream_capturing()
    pool, lay = st.pool, st.lay
    kv4.byte_lut(st.device)
    k = st.graph_table.shape[1] if st.graph_table is not None else 4
    table = torch.full((2, k), -1, dtype=torch.int32, device=st.device)
    table[0, 0] = lay.boundary  # the FP4 dummy page (never allocated): row 64 of the FP4 buffer
    table[1, 0] = 0
    out = torch.empty_like(table)
    mix.redirect_gather(lay, table, out, pool.fp4_buffer[0], pool.scratch, 0)
    # eager prefill kernels too, so the first long prefill does not pay the compile
    pt = torch.full((1, 8), lay.boundary, dtype=torch.int32, device=st.device)
    mix.seq_dequant(lay, pool.fp4_buffer[0], pt, torch.zeros(1, dtype=torch.int32, device=st.device),
                    torch.full((1,), 8, dtype=torch.int32, device=st.device),
                    torch.zeros(1, dtype=torch.int32, device=st.device), 8, pool.scratch, st.slot_map())
    mix.redirect_seq(lay, table, st.slot_map(), 0, out)
    st.slot_map()[lay.boundary] = -1
    torch.cuda.synchronize()


def _bmm1_scale(layer) -> float:
    k_scale = getattr(layer, "k_scale_float", None)
    return float(layer.scaling) * (1.0 if k_scale is None else float(k_scale))


def _trtllm(backend, q_rows: torch.Tensor, kv: torch.Tensor, table: torch.Tensor, seq_lens: torch.Tensor,
            max_seq_len: int, top_k: int, scale: float, out: torch.Tensor) -> None:
    """The stock TRT-LLM sparse MLA call: q_rows [T, H, 576] fp8, kv [pages, 1, 64, 576] fp8, table [T, K] int32."""
    import flashinfer.decode
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.trtllm_mla_backend import grow_multi_ctas_kv_counter_buffer_if_needed

    t, h, d = q_rows.shape
    backend._multi_ctas_kv_counter_buffer = grow_multi_ctas_kv_counter_buffer_if_needed(
        backend._multi_ctas_kv_counter_buffer, torch.device(backend.device), backend.num_q_heads, t)
    flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q_rows.view(t, 1, h, d),
        kv_cache=kv,
        workspace_buffer=backend.workspace_buffer,
        qk_nope_head_dim=backend.qk_nope_head_dim,
        kv_lora_rank=backend.kv_lora_rank,
        qk_rope_head_dim=backend.qk_rope_head_dim,
        block_tables=table.unsqueeze(1),
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=top_k,
        bmm1_scale=scale,
        backend="trtllm-gen",
        skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        multi_ctas_kv_counter_buffer=backend._multi_ctas_kv_counter_buffer,
        out=out,
    )


def _gather_path(backend, st: _State, l: int, q_all, page_table_1, top_k, metadata, seq_lens, scale, out) -> None:
    """Decode / verify / draft extend / fallback: FP4 entries -> scratch rows, one TRT-LLM call per chunk."""
    t, k = page_table_1.shape
    pool = st.pool
    kv = pool.layer_view(l)
    base = pool.scratch_base(l)
    capturing = torch.cuda.is_current_stream_capturing()
    per_call = max(1, pool.scratch_rows // k)
    if capturing:
        assert t <= per_call, (t, per_call)
    for c0 in range(0, t, per_call):
        c1 = min(t, c0 + per_call)
        tab = st.table(c1 - c0, k)
        mix.redirect_gather(st.lay, page_table_1[c0:c1], tab, pool.fp4_buffer[l], pool.scratch, base)
        _trtllm(backend, q_all[c0:c1], kv, tab, seq_lens[c0:c1], metadata.max_seq_len_k, top_k, scale, out[c0:c1])
    st.calls["gather"] += 1


def _sequence_path(backend, st: _State, l: int, q_all, page_table_1, top_k, forward_batch, metadata, seq_lens,
                   scale, out, fp4_reqs: list) -> bool:
    """Plain prefill: dequantize each FP4-holding request's sequence once. Returns False when it cannot apply."""
    pt = metadata.page_table_1
    seq_lens_cpu = forward_batch.seq_lens_cpu
    ext_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    bs = forward_batch.batch_size
    if pt is None or seq_lens_cpu is None or ext_cpu is None or pt.dim() != 2 or pt.shape[0] != bs:
        return False
    ext = [int(x) for x in ext_cpu]
    if sum(ext) != page_table_1.shape[0] or len(ext) != bs or len(fp4_reqs) != bs:
        return False
    pool = st.pool
    kv = pool.layer_view(l)
    base = pool.scratch_base(l)
    lens = [int(x) for x in seq_lens_cpu]
    cap = pool.scratch_rows
    if any(fp4_reqs[i] and lens[i] > cap for i in range(bs)):
        # a sequence longer than the scratch: the gather path (chunked per token) serves this batch
        log_once("seq_cap", "numinous kvmix: a %d-token request exceeds the %d-row scratch; its prefill takes the "
                 "per-token gather path (raise %s to avoid this)", max(lens), cap, SCRATCH_ROWS_ENV)
        return False
    cu = [0]
    for e in ext:
        cu.append(cu[-1] + e)
    slot_map = st.slot_map()
    i0 = 0
    while i0 < bs:
        # the longest contiguous request range [i0, i1) whose FP4-holding sequences fit the scratch
        rows, i1 = 0, i0
        while i1 < bs:
            add = lens[i1] if fp4_reqs[i1] else 0
            if rows + add > cap and i1 > i0:
                break
            assert add <= cap, (add, cap)
            rows += add
            i1 += 1
        req_ids = [i for i in range(i0, i1) if fp4_reqs[i]]
        t0, t1 = cu[i0], cu[i1]
        tab = st.table(t1 - t0, page_table_1.shape[1])
        if req_ids:
            bases = [0]
            for i in req_ids[:-1]:
                bases.append(bases[-1] + lens[i])
            dev = st.device
            req_t = torch.tensor(req_ids, dtype=torch.int32, device=dev)
            base_t = torch.tensor(bases, dtype=torch.int32, device=dev)
            max_len = min(max(lens[i] for i in req_ids), pt.shape[1])
            mix.seq_dequant(st.lay, pool.fp4_buffer[l], pt.contiguous(), req_t, metadata.cache_seqlens_int32, base_t,
                            max_len, pool.scratch, slot_map)
        mix.redirect_seq(st.lay, page_table_1[t0:t1], slot_map, base, tab)
        if t1 > t0:
            _trtllm(backend, q_all[t0:t1], kv, tab, seq_lens[t0:t1], metadata.max_seq_len_k, top_k, scale,
                    out[t0:t1])
        i0 = i1
    st.calls["seq"] += 1
    return True


def engine_forward(backend, q_all: torch.Tensor, page_table_1: torch.Tensor, sparse_mla_top_k: int, layer,
                   forward_batch, metadata, seq_lens: Optional[torch.Tensor], is_prefill: bool) -> torch.Tensor:
    """Replaces the TRT-LLM call in DeepseekSparseAttnBackend._forward_trtllm. q_all [T, H, 576] fp8, page_table_1
    [T, K] int32 unified slots padded with -1 (K a multiple of 4). Returns [T, 1, H, 512] bf16."""
    st = _state(backend)
    pool = st.pool
    l = layer.layer_id - pool.start_layer
    t, h, _ = q_all.shape
    out = torch.empty(t, 1, h, kv4.DIM_NOPE, dtype=torch.bfloat16, device=q_all.device)
    if t == 0:
        return out
    if q_all.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"kvmix: expected FP8 q from the trtllm path, got {q_all.dtype}")
    if not page_table_1.is_contiguous():
        page_table_1 = page_table_1.contiguous()
    if seq_lens is None:
        seq_lens = metadata.cache_seqlens_int32
    scale = _bmm1_scale(layer)
    capturing = torch.cuda.is_current_stream_capturing()
    mode = forward_batch.forward_mode
    if not capturing:
        fp4_reqs = st.fp4_requests(forward_batch, metadata)
        if fp4_reqs is not None and not any(fp4_reqs):
            _trtllm(backend, q_all, pool.layer_view(l), page_table_1, seq_lens, metadata.max_seq_len_k,
                    sparse_mla_top_k, scale, out)
            st.calls["stock"] += 1
            return out
        if (is_prefill and fp4_reqs is not None and (mode.is_extend_or_draft_extend_or_mixed() or mode.is_mixed())
                and not mode.is_target_verify()
                and _sequence_path(backend, st, l, q_all, page_table_1, sparse_mla_top_k, forward_batch, metadata,
                                   seq_lens, scale, out, fp4_reqs)):
            return out
    _gather_path(backend, st, l, q_all, page_table_1, sparse_mla_top_k, metadata, seq_lens, scale, out)
    return out


def write_door(backend, k_nope_bf16: Optional[torch.Tensor], k_fp8: torch.Tensor, k_rope_fp8: torch.Tensor, layer,
               cache_loc: torch.Tensor) -> None:
    """The trtllm-path write: FP8 bytes for FP8 slots, an FP4 quantization of the bf16 latent for FP4 slots."""
    bf16 = k_nope_bf16 if (k_nope_bf16 is not None and k_nope_bf16.shape[0] == k_fp8.shape[0]) else None
    backend.token_to_kv_pool.write_mixed(layer, cache_loc, k_fp8, k_rope_fp8, bf16)
