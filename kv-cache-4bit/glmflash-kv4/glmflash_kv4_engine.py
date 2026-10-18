"""GLM-5.3-Flash 4-bit KV cache inside SGLang v0.5.20: glm_kv4_engine (GLM-5.3) re-targeted at the rope-free row.

Installed by operations/patches/glmflash_kv4_hooks.py as sglang/srt/layers/attention/dsa/glmflash_kv4_engine.py next
to glmflash_kv4.py (kernels) and glm_kv4.py (the shared numerics and geometry-free passes). Behind
NUMINOUS_GLMFLASH_KV4=1; without it the hooks are one attribute test.

Differences against glm_kv4_engine.py (everything else is that module's logic, kept line for line):
  arch      Glm5NextForConditionalGeneration (and its NextN draft) instead of GlmMoeDsa
  row       288 B FP4 rows (glmflash_kv4.ROW_BYTES), 512 B FP8 scratch rows (the raw TRT-LLM layout on Flash)
  rope      qk_rope_head_dim must be 0; the write door ignores the zero-width k_rope the FP8 path passes;
            get_mla_kv_buffer returns a zero-width rope tensor; q_all is [T, H, 512]
  trtllm    the scratch call passes sparse_mla_top_k_lens = prepare_trtllm_nope_sparse_metadata(table), as
            dsa_backend._forward_trtllm does for rope 0 (the kernel needs per-row valid counts without a rope part)
Env: NUMINOUS_GLMFLASH_KV4, NUMINOUS_GLMFLASH_KV4_ATTN (scratch | triton), NUMINOUS_GLMFLASH_KV4_SEQ_ROWS,
NUMINOUS_GLMFLASH_KV4_SEL_ROWS, NUMINOUS_GLMFLASH_KV4_DEDUP. UNTESTED on a GPU as of 2026-09-29 (no box).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.attention.dsa import glmflash_kv4 as kv4
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

logger = logging.getLogger("sglang.numinous")
from sglang.kernels.ops.attention.dsa.transform_index import prepare_trtllm_nope_sparse_metadata

ENV = "NUMINOUS_GLMFLASH_KV4"
ATTN_ENV = "NUMINOUS_GLMFLASH_KV4_ATTN"
SEQ_ROWS_ENV = "NUMINOUS_GLMFLASH_KV4_SEQ_ROWS"
SEL_ROWS_ENV = "NUMINOUS_GLMFLASH_KV4_SEL_ROWS"
DEDUP_ENV = "NUMINOUS_GLMFLASH_KV4_DEDUP"  # 1 (default): batches with several tokens per request dequantize shared rows once
DEDUP_MIN_ROWS = 128  # below this the seven dedup passes cost more than the rows they save (measured, B200)
ROW_BYTES = kv4.ROW_BYTES
FP8_ROW_BYTES = kv4.FP8_ROW_BYTES
ARCH_PREFIX = "Glm5Next"
SCRATCH_PAGE = 64  # trtllm-gen accepts block sizes 32 and 64; the served pool uses 64
_logged: set = set()


def log_once(key: str, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.warning(msg, *args)


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def attn_mode() -> str:
    m = os.environ.get(ATTN_ENV, "scratch")
    if m not in ("scratch", "triton"):
        raise ValueError(f"{ATTN_ENV}={m!r}: expected scratch or triton")
    return m


def enabled_for(hf_config, kv_cache_dtype) -> bool:
    """The FP4 pool applies to GLM-5.3 (target and its NextN draft) served with an FP8 KV flag."""
    if not enabled():
        return False
    archs = getattr(hf_config, "architectures", None) or []
    if not any(str(a).startswith(ARCH_PREFIX) for a in archs):
        log_once("arch", "numinous glmflash_kv4: %s=1 but architectures %s are not GLM DSA; FP4 pool off", ENV, archs)
        return False
    if kv_cache_dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"numinous glmflash_kv4: needs --kv-cache-dtype fp8_e4m3 (the trtllm DSA path quantizes q to FP8 and the "
            f"scratch is FP8); got {kv_cache_dtype}")
    return True


def pool_class(default_cls, kv_cache_dim):
    """For HybridLinearKVPool.__init__ (memory_pool.py hook glmflash_kv4_hybrid_pool): Flash's DSA pool is built inside
    the hybrid wrapper, not through kv_cache_configurator._build_dsa_kv_pool, so the class swap happens here. The FP4
    class is returned only when the dim hook priced 288-byte rows, so a stock-dim pool never gets FP4 semantics."""
    if enabled() and kv_cache_dim == ROW_BYTES:
        return GLMFlashKV4DSATokenToKVPool
    if enabled() and kv_cache_dim != ROW_BYTES:
        log_once("dim", "numinous glmflash_kv4: %s=1 but kv_cache_dim %s != %d; stock pool", ENV, kv_cache_dim, ROW_BYTES)
    return default_cls


def _inner(pool):
    """The DSA pool behind Flash's HybridLinearKVPool (the KDA layers' state lives elsewhere)."""
    return getattr(pool, "full_kv_pool", pool)


def _dense_layer(pool, layer_id: int) -> int:
    """kv_buffer index of a global layer id: the hybrid wrapper maps the 11 DSA layers onto 0..10."""
    if hasattr(pool, "_transfer_full_attention_id"):
        return pool._transfer_full_attention_id(layer_id) - _inner(pool).start_layer
    return layer_id - pool.start_layer


class GLMFlashKV4DSATokenToKVPool(DSATokenToKVPool):
    """DSATokenToKVPool with 288-byte FP4 rows. kv_buffer[l]: [size + page_size, 1, 288] uint8."""

    numinous_kv4 = True

    def __init__(self, *args, **kwargs):
        kv_cache_dim = kwargs.get("kv_cache_dim")
        assert kv_cache_dim == ROW_BYTES, f"glmflash_kv4 pool expects kv_cache_dim {ROW_BYTES}, got {kv_cache_dim}"
        super().__init__(*args, **kwargs)
        # MLATokenToKVPool takes any overridden dim with an FP8 dtype for the 656-byte scaled layout; ours is not.
        self.dsa_kv_cache_store_fp8 = False
        assert self.store_dtype == torch.uint8 and self.kv_cache_dim == ROW_BYTES
        assert self.kv_lora_rank == kv4.DIM_NOPE and self.qk_rope_head_dim == kv4.DIM_ROPE, (
            self.kv_lora_rank, self.qk_rope_head_dim)
        logger.warning(
            "numinous glmflash_kv4 pool: %d layers x %d B/token (E2M1 latent + E4M3 scale per 16, no rope) = %d B/token "
            "vs %d for the FP8 raw layout; %d tokens",
            self.layer_num, ROW_BYTES, ROW_BYTES * self.layer_num, FP8_ROW_BYTES * self.layer_num, self.size)

    def _write_mla_kv_buffer(self, dst_buffer, loc, cache_k_nope, cache_k_rope) -> None:
        from sglang.srt.runtime_context import get_parallel

        assert not get_parallel().dcp_enabled, "glmflash_kv4 pool has no DCP write path"
        assert cache_k_rope is None or cache_k_rope.numel() == 0, "Flash DSA rows have no rope part"
        kv4.quant_write(dst_buffer, loc, cache_k_nope)

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("glmflash_kv4 pool: combined-row set_kv_buffer is not a DSA write door (use set_mla_kv_buffer)")

    def get_value_buffer(self, layer_id: int):
        raise RuntimeError("glmflash_kv4 pool: the latent is FP4; read it through glmflash_kv4_engine.engine_forward")

    def get_mla_kv_buffer(self, layer, loc: torch.Tensor, dst_dtype: Optional[torch.dtype] = None):
        """Dequantized rows at loc: (nope [T, 1, 512], rope [T, 1, 0]) in dst_dtype (default bf16)."""
        dst_dtype = dst_dtype or torch.bfloat16
        buf = self.kv_buffer[layer.layer_id - self.start_layer]
        out = torch.empty(loc.numel(), kv4.DIM_QK, dtype=torch.bfloat16, device=buf.device)
        kv4.dequant_rows(buf, loc, out)
        out = out.to(dst_dtype)
        return out.unsqueeze(1).contiguous(), out.new_empty(loc.numel(), 1, 0)


# ------------------------------------------------------------------------------------------------ backend glue


class _State:
    """Per-attention-backend scratch buffers."""

    def __init__(self, device: torch.device, n_pool: int):
        self.device = device
        self.n_pool = n_pool
        self.mode = attn_mode()
        self.graph_scratch: Optional[torch.Tensor] = None  # fp8 [rows, 512], fixed after init_cuda_graph_state
        self.graph_table: Optional[torch.Tensor] = None  # int32 [tokens, K]
        self.graph_work = None  # (counts, owners, pos) for the dedup passes
        self.eager_work = None
        self.eager_scratch: Optional[torch.Tensor] = None  # fp8 [rows, 512], grows outside capture
        self.eager_table: Optional[torch.Tensor] = None
        self.slot_to_pos: Optional[torch.Tensor] = None  # int32 [n_pool]
        self.stamp: Optional[torch.Tensor] = None  # int32 [n_pool], STAMP_CLEAN between calls
        self.dedup = os.environ.get(DEDUP_ENV, "1") == "1"
        self.sel_tokens = int(os.environ.get(SEL_ROWS_ENV, "256"))
        # the sequence path must cover the served context (a 632k-token session fell back to the chunked
        # selected-row path at the 512k default and paid +0.8 s warm TTFT, measured 2026-09-29): 1.1M rows = 634 MB,
        # allocated lazily and only as large as the batch's contexts need
        self.seq_rows_cap = int(os.environ.get(SEQ_ROWS_ENV, str(1_100_000)))
        self.calls = {"seq": 0, "sel": 0, "triton": 0}
        self.eager_by_mode: dict = {}  # forward mode -> eager (non-captured) engine_forward calls
        self.captured_by_mode: dict = {}  # forward mode -> calls made under CUDA-graph capture

    def _alloc(self, rows: int):
        rows = -(-rows // SCRATCH_PAGE) * SCRATCH_PAGE  # the TRT-LLM op views the scratch as 64-token pages
        return torch.empty(rows, FP8_ROW_BYTES, dtype=torch.float8_e4m3fn, device=self.device)

    def reserve_graph(self, tokens: int, k: int) -> None:
        rows = tokens * k
        if self.graph_scratch is None or self.graph_scratch.shape[0] < rows:
            assert not torch.cuda.is_current_stream_capturing(), "glmflash_kv4: graph scratch requested during capture"
            self.graph_scratch = self._alloc(rows)
            self.graph_table = torch.empty(tokens, k, dtype=torch.int32, device=self.device)
            self.graph_work = self._work(rows)
            logger.warning("numinous glmflash_kv4: graph scratch %d tokens x %d rows = %.0f MB", tokens, k,
                           rows * FP8_ROW_BYTES / 2**20)

    def eager(self, rows: int, tokens: int, k: int):
        """Eager scratch with at least `rows` rows and a [tokens, k] table; grows only outside capture."""
        assert not torch.cuda.is_current_stream_capturing()
        if self.eager_scratch is None or self.eager_scratch.shape[0] < rows:
            self.eager_scratch = self._alloc(rows)
        if self.eager_table is None or self.eager_table.shape[0] < tokens or self.eager_table.shape[1] != k:
            self.eager_table = torch.empty(tokens, k, dtype=torch.int32, device=self.device)
            self.eager_work = self._work(tokens * k)
        return self.eager_scratch, self.eager_table

    def _work(self, entries: int):
        """(counts, owners, pos) for dequant_rows_dedup over up to `entries` table entries."""
        return (torch.empty(-(-entries // 1024), dtype=torch.int32, device=self.device),
                torch.empty(entries, dtype=torch.int32, device=self.device),
                torch.empty(entries, dtype=torch.int32, device=self.device))

    def slot_map(self) -> torch.Tensor:
        if self.slot_to_pos is None:
            self.slot_to_pos = torch.full((self.n_pool,), -1, dtype=torch.int32, device=self.device)
        return self.slot_to_pos

    def stamps(self) -> torch.Tensor:
        """int32 [pool] owner stamps for dequant_rows_dedup, STAMP_CLEAN between calls. Allocated before any graph
        capture (init_cuda_graph_state) or at first eager use."""
        if self.stamp is None:
            assert not torch.cuda.is_current_stream_capturing(), "glmflash_kv4: stamp array requested during capture"
            self.stamp = torch.full((self.n_pool,), kv4.STAMP_CLEAN, dtype=torch.int32, device=self.device)
        return self.stamp


def _state(backend) -> _State:
    st = getattr(backend, "_numinous_fkv4_state", None)
    if st is None:
        pool = _inner(backend.token_to_kv_pool)
        st = _State(torch.device(backend.device), pool.kv_buffer[0].shape[0])
        backend._numinous_fkv4_state = st
    return st


def attach(backend, model_runner) -> bool:
    """From DeepseekSparseAttnBackend.__init__: True when this backend serves the FP4 pool."""
    pool = _inner(model_runner.token_to_kv_pool)
    if not getattr(pool, "numinous_kv4", False):
        if enabled():
            log_once("nopool", "numinous glmflash_kv4: %s=1 but the pool is %s; FP4 attention off", ENV,
                     type(pool).__name__)
        return False
    why = (
        f"dsa roles prefill={backend.dsa_prefill_impl} decode={backend.dsa_decode_impl} (need trtllm/trtllm)"
        if backend.dsa_prefill_impl != "trtllm" or backend.dsa_decode_impl != "trtllm"
        else f"sm_{backend.device_sm_major}x (need SM100)" if backend.device_sm_major != 10
        else f"kv dtype {backend.kv_cache_dtype}" if backend.kv_cache_dtype != torch.float8_e4m3fn
        else "use_mha" if getattr(backend, "use_mha", False)
        else f"qk_rope_head_dim {backend.qk_rope_head_dim} (Flash rows have none)" if backend.qk_rope_head_dim != 0
        else ""
    )
    if why:
        raise RuntimeError(f"numinous glmflash_kv4: the FP4 pool is built but this backend cannot read it: {why}")
    st = _state(backend)
    log_once("attach", "numinous glmflash_kv4 attention active: mode=%s, %d heads, top-k %d, pool %d slots",
             st.mode, backend.num_q_heads, backend.dsa_index_topk, st.n_pool)
    return True


def reserve_graph_scratch(backend, max_num_tokens: int) -> None:
    """From init_cuda_graph_state: size the fixed decode/verify scratch (TRT-LLM pads top-k to a multiple of 4)."""
    st = _state(backend)
    if st.mode == "scratch":
        # the k-pool indexer (Flash: index_kpool 4) hands over index_topk + kpool - 1 columns, which the TRT-LLM
        # path pads to a multiple of 4: 2048 + 3 -> 2052 (measured 07:54 on B300: width 2052 vs 2048 reserved)
        kpool = int(getattr(backend, "dsa_index_kpool", 1) or 1)
        k = ((backend.dsa_index_topk + kpool - 1 + 3) // 4) * 4
        st.reserve_graph(max(int(max_num_tokens), 1), k)
        if st.dedup:
            st.stamps()


def _bmm1_scale(layer) -> float:
    k_scale = getattr(layer, "k_scale_float", None)
    return float(layer.scaling) * (1.0 if k_scale is None else float(k_scale))


def _trtllm(backend, q_rows: torch.Tensor, kv: torch.Tensor, table: torch.Tensor, seq_lens: torch.Tensor,
            max_seq_len: int, top_k: int, scale: float, out: torch.Tensor) -> None:
    """The stock TRT-LLM sparse MLA call on an FP8 scratch: q_rows [T, H, 512] fp8, kv [rows, 512] fp8,
    table [T, K] int32 into kv rows (token slots inside 64-row pages, as the pool's), out [T, 1, H, 512] bf16."""
    assert kv.shape[0] % SCRATCH_PAGE == 0, kv.shape
    import flashinfer.decode
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.trtllm_mla_backend import grow_multi_ctas_kv_counter_buffer_if_needed

    t, h, d = q_rows.shape
    backend._multi_ctas_kv_counter_buffer = grow_multi_ctas_kv_counter_buffer_if_needed(
        backend._multi_ctas_kv_counter_buffer, torch.device(backend.device), backend.num_q_heads, t)
    flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q_rows.view(t, 1, h, d),
        kv_cache=kv.view(-1, 1, SCRATCH_PAGE, FP8_ROW_BYTES),
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
        sparse_mla_top_k_lens=prepare_trtllm_nope_sparse_metadata(table),
        out=out,
    )


def _sequence_prefill(backend, st: _State, q_all, kv_buffer, page_table_1, top_k, forward_batch, metadata, seq_lens,
                      scale, out) -> bool:
    """Plain prefill: dequantize every request's sequence once. Returns False when the batch does not fit."""
    pt = metadata.page_table_1
    seq_lens_cpu = forward_batch.seq_lens_cpu
    if pt is None or seq_lens_cpu is None or pt.dim() != 2 or pt.shape[0] != seq_lens_cpu.numel():
        return False
    if pt.shape[0] != forward_batch.batch_size:
        return False
    lens = seq_lens_cpu.to(torch.int64)
    rows = int(lens.sum())
    if rows == 0 or rows > st.seq_rows_cap:
        return False
    max_len = min(int(lens.max()), pt.shape[1])
    bases = torch.zeros(lens.numel(), dtype=torch.int32)
    bases[1:] = torch.cumsum(lens, 0)[:-1].to(torch.int32)
    bases = bases.to(st.device, non_blocking=True)
    scratch, table = st.eager(rows, page_table_1.shape[0], page_table_1.shape[1])
    kv4.dequant_seq(kv_buffer, pt.contiguous(), metadata.cache_seqlens_int32, bases, max_len, scratch,
                    st.slot_map())
    tab = table[: page_table_1.shape[0]]
    kv4.remap_table(page_table_1, st.slot_to_pos, st.n_pool, tab)
    _trtllm(backend, q_all, scratch, tab, seq_lens, metadata.max_seq_len_k, top_k, scale, out)
    st.calls["seq"] += 1
    return True


def _selected_rows(backend, st: _State, q_all, kv_buffer, page_table_1, top_k, metadata, seq_lens, scale, out,
                   dedup: bool):
    """Gather + dequantize exactly the selected rows, in chunks of tokens that fit the scratch. With dedup, rows
    several tokens of the chunk select are dequantized once (verify: a request's draft tokens share most picks)."""
    t, k = page_table_1.shape
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        assert st.graph_scratch is not None, "glmflash_kv4: init_cuda_graph_state did not reserve the graph scratch"
        scratch, table, work = st.graph_scratch, st.graph_table, st.graph_work
    else:
        scratch, table = st.eager(min(t, st.sel_tokens) * k, min(t, st.sel_tokens), k)
        work = st.eager_work
    # the fused top-k path hands the indexer's own tensor over as page_table_1 (SGLANG_DSA_FUSE_TOPK defaults on);
    # the row kernels index it flat, so a strided view is made contiguous here (a captured copy is graph-pool memory)
    if not page_table_1.is_contiguous():
        page_table_1 = page_table_1.contiguous()
    if table.shape[1] != k:
        # width not anticipated at graph init: the scratch and table are re-sized here; under capture the memory
        # comes from the graph's private pool, which is legal and deterministic per captured batch size
        logger.warning("numinous glmflash_kv4: table width %d != page_table_1 width %d (capturing=%s); resizing",
                       table.shape[1], k, capturing)
        if capturing:
            tokens = max(st.graph_table.shape[0], t)
            st.graph_scratch = st._alloc(tokens * k)
            st.graph_table = torch.empty(tokens, k, dtype=torch.int32, device=st.device)
            st.graph_work = st._work(tokens * k)
            scratch, table, work = st.graph_scratch, st.graph_table, st.graph_work
        else:
            st.eager_table = None
            scratch, table = st.eager(min(t, st.sel_tokens) * k, min(t, st.sel_tokens), k)
            work = st.eager_work
    per_call = max(1, min(scratch.shape[0] // k, table.shape[0]))
    for c0 in range(0, t, per_call):
        c1 = min(t, c0 + per_call)
        sub = page_table_1[c0:c1]
        tab = table[: c1 - c0]
        if dedup:
            kv4.dequant_rows_dedup(kv_buffer, sub, st.stamps(), scratch, tab, work=work)
        else:
            kv4.dequant_rows(kv_buffer, sub, scratch)
            kv4.identity_table(sub, tab)
        _trtllm(backend, q_all[c0:c1], scratch, tab, seq_lens[c0:c1], metadata.max_seq_len_k, top_k, scale,
                out[c0:c1])
    st.calls["sel"] += 1


def engine_forward(backend, q_all: torch.Tensor, page_table_1: torch.Tensor, sparse_mla_top_k: int, layer,
                   forward_batch, metadata, seq_lens: Optional[torch.Tensor], is_prefill: bool) -> torch.Tensor:
    """Replaces the TRT-LLM call in DeepseekSparseAttnBackend._forward_trtllm. q_all [T, H, 512] fp8 (nope only), page_table_1 [T, K] int32 pool slots padded with -1 (K a multiple of 4). Returns [T, 1, H, 512]."""
    st = _state(backend)
    pool = backend.token_to_kv_pool
    kv_buffer = _inner(pool).kv_buffer[_dense_layer(pool, layer.layer_id)]
    t, h, _ = q_all.shape
    out = torch.empty(t, 1, h, kv4.DIM_NOPE, dtype=torch.bfloat16, device=q_all.device)
    if t == 0:
        return out
    if q_all.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"glmflash_kv4: expected FP8 q from the trtllm path, got {q_all.dtype}")
    if not page_table_1.is_contiguous():
        page_table_1 = page_table_1.contiguous()
    scale = _bmm1_scale(layer)
    if st.mode == "triton":
        kv4.sparse_mla(q_all, kv_buffer, page_table_1, scale, out=out.view(t, h, kv4.DIM_NOPE))
        st.calls["triton"] += 1
        return out
    if seq_lens is None:
        seq_lens = metadata.cache_seqlens_int32
    mode = forward_batch.forward_mode
    _count_call(st, mode, t)
    if (is_prefill and not torch.cuda.is_current_stream_capturing()
            and (mode.is_extend_or_draft_extend_or_mixed() or mode.is_mixed())
            and not mode.is_target_verify()
            and _sequence_prefill(backend, st, q_all, kv_buffer, page_table_1, sparse_mla_top_k, forward_batch,
                                  metadata, seq_lens, scale, out)):
        return out
    # dedup pays only with many rows (measured B200: 192 rows 145 vs 163 us, 64 rows 60.7 vs 58.2, 1 row 28.5 vs
    # 11.9): plain decode and small verify batches take the two-launch gather
    dedup = st.dedup and not mode.is_decode_or_idle() and t >= DEDUP_MIN_ROWS
    _selected_rows(backend, st, q_all, kv_buffer, page_table_1, sparse_mla_top_k, metadata, seq_lens, scale, out,
                   dedup)
    return out


def _count_call(st: _State, mode, t: int) -> None:
    """Eager vs captured calls per forward mode; logged at 1, 100, 1000, ... eager calls (the gate keeps `numinous`
    lines), so a boot says whether verify/decode steps replay graphs or run this path eagerly per layer."""
    d = st.captured_by_mode if torch.cuda.is_current_stream_capturing() else st.eager_by_mode
    d[mode.name] = d.get(mode.name, 0) + 1
    n = sum(st.eager_by_mode.values())
    if n in (1, 100, 1000, 10000, 100000):
        logger.warning("numinous glmflash_kv4 calls: eager %s captured %s (last mode %s, %d rows)",
                       st.eager_by_mode, st.captured_by_mode, mode.name, t)


def write_bf16_latent(backend, k_nope_bf16: Optional[torch.Tensor], k_fp8: torch.Tensor, k_rope_fp8: torch.Tensor,
                      layer, cache_loc: torch.Tensor) -> None:
    """The FP4 write door: prefers the bf16 latent stashed before the FP8 quantize kernel (same rows), else FP8."""
    src = k_nope_bf16 if (k_nope_bf16 is not None and k_nope_bf16.shape[0] == k_fp8.shape[0]) else k_fp8
    backend.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, src, k_rope_fp8)
