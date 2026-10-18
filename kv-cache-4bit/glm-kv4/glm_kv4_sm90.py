"""GLM-5.3 4-bit KV cache on Hopper (SM90): the `flashmla_kv` DSA roles reading the 352-byte FP4 pool.

Installed by operations/patches/glm_kv4_sm90_hooks.py as sglang/srt/layers/attention/dsa/glm_kv4_sm90.py next to
glm_kv4.py (kernels) and glm_kv4_engine.py (pool class + scratch state, shared with the SM100 and ROCm paths).
Everything is behind NUMINOUS_GLM_KV4=1, like the other two paths.

Why (arm P of agents/briefs/glm-h200-composed.md, measured 2026-10-05): the in-box PD decode role on 4 H200s holds
296,256 FP8 tokens per rank (1.19M, ~21 agent sessions of 55K), and PD ships raw KV pages, so a smaller row must be on
BOTH roles. The FP4 row is 352 B against the SM90 FP8 row's 656 B (512 E4M3 + 4 fp32 scales + 64 bf16 rope).

SM90 differences from the SM100 path (glm_kv4_engine.engine_forward):
  roles   prefill and decode both `flashmla_kv` with --kv-cache-dtype fp8_e4m3 (the SM90 FP8 DSA path). Draft extend
          and target verify run the decode role, which is the same function.
  write   unchanged code path: dsa_backend.forward_decode / forward_extend call set_mla_kv_buffer(k bf16, k_rope
          bf16); GLMKV4DSATokenToKVPool quantizes from bf16 (glm_kv4.quant_write).
  read    `forward_flashmla_kv` replaces DeepseekSparseAttnBackend._forward_flashmla_kv. The stock SM90 FP8 decode
          kernel only reads its own 656-byte row, so the selected rows are dequantized EXACTLY into a bf16 [rows, 576]
          scratch (E2M1 x E4M3 is exact in bf16; rope E4M3 -> bf16 exact) and the stock SM90 bf16 sparse kernel
          (flash_mla_sparse_fwd, through dsa_backend._forward_flashmla_sparse, which pads heads to 64) runs on it:
            prefill (EXTEND / MIXED / draft extend, eager): every request's sequence is dequantized once
                    (sum(seq_lens) rows) and the top-k table is remapped through a pool-sized slot -> row map;
            decode / verify / draft decode (CUDA graphs): the selected rows per token, chunked to the scratch
                    reserved in init_cuda_graph_state; many-token calls dedup rows shared between tokens.
          NUMINOUS_GLM_KV4_ATTN=triton runs the fused Triton kernel with in-kernel FP4 dequant instead (q cast to FP8
          as on SM100).
  mha     MHA_ONE_SHOT prefill reads the cached prefix through dequantize_k_cache_paged, which only knows the 656-byte
          row (forward_mha.py: the FP8 DSA branch for non-trtllm roles), so attach() turns it off for this backend
          (supports_mha_one_shot = False; dsa_backend consults it per batch). Every prefill takes the MLA path.
  graphs  the decode/verify scratch is reserved in init_cuda_graph_state (max_num_tokens x top-k rows, bf16:
          1,152 B per row) and never reallocated; eager calls larger than the scratch are chunked by tokens.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.attention.dsa import glm_kv4 as kv4
from sglang.srt.layers.attention.dsa import glm_kv4_engine as eng

logger = logging.getLogger("sglang.numinous")

ROLE = "flashmla_kv"
# dedup pays when a call's tokens share rows (verify: a request's draft tokens; draft extend); with a bf16 scratch a
# row costs 1,152 B written + read, so the break-even is lower than the B200 FP8 measurement (128 rows); the GPU check
# (operations/kernels/hopper_kv4_check.py) times both and the env overrides the default
DEDUP_MIN_ROWS = int(os.environ.get("NUMINOUS_GLM_KV4_SM90_DEDUP_MIN", "64"))
# decode / verify / draft-decode scratch format (prefill always takes the exact bf16 sequence path):
#   bf16 (default): exact dequant, 1,152 B per selected row, the stock bf16 sparse kernel (flash_mla_sparse_fwd);
#   v32: the FP4 rows re-quantized into FlashMLA's own 656-byte V32 row (512 E4M3 + four fp32 per-128 scales + 64 bf16
#        rope, exactly what quant_k_cache writes for the stock FP8 pool), read by the stock SM90 FP8 sparse DECODE kernel
#        (flash_mla_with_kvcache, 57 us per 48-token layer call vs 116 us for the bf16 route, measured 1xH200).
DECODE_ENV = "NUMINOUS_GLM_KV4_SM90_DECODE"
V32_ROW = 656
# prefill (the sequence path: EXTEND / MIXED / draft extend) attention kernel on the exact bf16 scratch:
#   bf16 (default): the stock SM90 bf16 sparse prefill kernel (flash_mla_sparse_fwd);
#   q8: v0.5.20's native FP8 SM90 sparse prefill kernel (sparse_mla_q8kv8_prefill_fwd, what --dsa-prefill-backend
#       flashmla_sparse_q8 runs on the stock FP8 pool; FP8 GMMA, 2x the bf16 tensor rate). The scratch rows are cast
#       to E4M3 into [rows + topk, 576] with `topk` trailing zero rows (the kernel clamps a -1 slot to row
#       s_kv - topk + slot, kernel.cuh:793-823), q is cast to E4M3, and the per-tensor scales follow
#       NUMINOUS_GLM_KV4_SM90_Q8_SCALE:
#         identity (default): scale 1, v0.5.20's contract (dsa_backend._forward_flashmla_sparse_q8kv8);
#         amax: amax / 448 per tensor for q and for kv, computed on device (no host sync).
#   Decode / verify / draft decode keep their route (decode_mode()).
#   Measured 2026-10-05 (amdpro, research/gpu-evidence/amdpro/glm_sparse_q8kv8_bench_h200.json, real GLM-5.3 layers
#   3/40). This route is NOT adopted for M6; it stays opt-in for a future kernel.
#     - Identity scales fail on layer 3: 89% of the KV sits below E4M3's smallest normal, rel L2 up to 0.147,
#       cos 0.989. amax scales hold rel L2 <= 0.035.
#     - The q8 kernel runs 1.14x / 0.95x / 1.17x the bf16 kernel at s_q 2048 / 4096 / 8192. With the casts that is
#       ~1.03x at M's 2,048 tokens/rank, and amax's extra passes make it net slower. The sparse prefill is
#       gather-bound, not MMA-bound.
PREFILL_ENV = "NUMINOUS_GLM_KV4_SM90_PREFILL"
Q8_SCALE_ENV = "NUMINOUS_GLM_KV4_SM90_Q8_SCALE"
E4M3_MAX = 448.0
Q8_CAST_ROWS = 131072  # amax mode: rows per fp32 rescale chunk (bounds the temporary to ~290 MB)


def prefill_mode() -> str:
    m = os.environ.get(PREFILL_ENV, "bf16")
    if m not in ("bf16", "q8"):
        raise ValueError(f"{PREFILL_ENV}={m!r}: expected bf16 or q8")
    return m


def q8_scale_mode() -> str:
    m = os.environ.get(Q8_SCALE_ENV, "identity")
    if m not in ("identity", "amax"):
        raise ValueError(f"{Q8_SCALE_ENV}={m!r}: expected identity or amax")
    return m


def decode_mode() -> str:
    m = os.environ.get(DECODE_ENV, "bf16")
    if m not in ("bf16", "v32"):
        raise ValueError(f"{DECODE_ENV}={m!r}: expected bf16 or v32")
    return m


try:  # the CPU-only host imports this module for the glue tests
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _kv4_to_v32_kernel(src_ptr, lut_ptr, idx_ptr, n_rows, n_pool, dst_u8, dst_f32, dst_bf16,
                           BLOCK_R: tl.constexpr):
        """dst[r] (656 B) = FlashMLA V32 row of pool[idx[r]]: the exact FP4 x E4M3 values re-quantized per 128 with
        quant_k_cache's convention (scale = amax / 448, q = clamp(x * (1 / scale)) -> E4M3), the E4M3 rope as bf16.
        Rows with idx -1 are left untouched (the decode kernel masks -1 entries)."""
        r = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < n_rows
        slot = tl.load(idx_ptr + r, mask=rmask, other=-1).to(tl.int64)
        valid = rmask & (slot >= 0) & (slot < n_pool)
        src = src_ptr + tl.multiple_of(tl.where(valid, slot, 0) * 352, 16)[:, None]
        pair = tl.max_contiguous(tl.multiple_of(tl.arange(0, 256), 16), 256)
        b = tl.load(src + pair[None, :], mask=valid[:, None], other=0).to(tl.int32)  # [R, 256]
        soffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 16), 32)
        sc = tl.load(src + 256 + soffs[None, :], mask=valid[:, None], other=0).to(
            tl.float8e4nv, bitcast=True).to(tl.float32)  # [R, 32]
        sc_pair = tl.reshape(tl.broadcast_to(sc[:, :, None], [BLOCK_R, 32, 8]), [BLOCK_R, 256])
        lo, hi = kv4._decode_pairs(lut_ptr, b)
        v = tl.reshape(tl.join(lo * sc_pair, hi * sc_pair), [BLOCK_R, 4, 128])  # exact values, 4 tiles of 128
        amax = tl.max(tl.abs(v), axis=2)  # [R, 4]
        s = amax / 448.0
        inv = tl.where(s > 0, 1.0 / s, 0.0)
        q = tl.clamp(v * inv[:, :, None], -448.0, 448.0)
        q8 = tl.reshape(q, [BLOCK_R, 512]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        doffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 16), 512)
        tl.store(dst_u8 + (r * 656)[:, None] + doffs[None, :], q8, mask=valid[:, None])
        s4 = tl.arange(0, 4)
        tl.store(dst_f32 + (r * 164 + 128)[:, None] + s4[None, :], s, mask=valid[:, None])  # byte 512 = f32 128
        roffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
        rope = tl.load(src + 288 + roffs[None, :], mask=valid[:, None], other=0).to(
            tl.float8e4nv, bitcast=True).to(tl.float32).to(tl.bfloat16)
        tl.store(dst_bf16 + (r * 328 + 264)[:, None] + roffs[None, :], rope, mask=valid[:, None])  # byte 528


def kv4_to_v32(kv_buffer: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, block_r: int = 16,
               num_warps: int = 4) -> None:
    """dst[r] = V32 row of pool[idx[r]] for idx[r] >= 0. dst: uint8 [R, 656], contiguous, R >= idx.numel()."""
    rows = kv4.pool_rows(kv_buffer)
    idx = idx.reshape(-1)
    n = idx.numel()
    if n == 0:
        return
    assert dst.dtype == torch.uint8 and dst.shape[-1] == V32_ROW and dst.is_contiguous() and dst.shape[0] >= n, (
        dst.dtype, dst.shape, n)
    _kv4_to_v32_kernel[(triton.cdiv(n, block_r),)](rows, kv4.byte_lut(rows.device), idx, n, rows.shape[0], dst,
                                                   dst.view(torch.float32), dst.view(torch.bfloat16),
                                                   BLOCK_R=block_r, num_warps=num_warps)


class _State90(eng._State):
    """The engine's scratch state with bf16 rows (the SM90 bf16 sparse kernel reads [rows, 1, 576] bf16), plus the
    V32 decode scratch when NUMINOUS_GLM_KV4_SM90_DECODE=v32."""

    def __init__(self, device, n_pool):
        super().__init__(device, n_pool)
        self.decode = decode_mode()
        self.v32_graph: Optional[torch.Tensor] = None  # uint8 [rows, 656], fixed after init_cuda_graph_state
        self.v32_eager: Optional[torch.Tensor] = None
        self.calls["v32"] = 0
        self.prefill = prefill_mode()
        self.q8_scale = q8_scale_mode()
        self.q8_kv: Optional[torch.Tensor] = None   # E4M3 [rows + topk, 576], grow-only (eager prefill only)
        self.q8_q: Optional[torch.Tensor] = None    # E4M3 [tokens, heads padded to 64, 576]
        self.q8_out: Optional[tuple] = None         # (out bf16, max_logits fp32, lse fp32)
        self.q8_one: Optional[torch.Tensor] = None  # the identity scale, one device scalar (no per-call H2D copy)
        self.calls["q8"] = 0

    def q8_buffers(self, kv_rows: int, tokens: int, heads: int, d_v: int):
        """(kv, q, (out, max_logits, lse), one) for a q8 prefill call; grown outside capture only."""
        assert not torch.cuda.is_current_stream_capturing(), "glm_kv4: q8 prefill buffers requested during capture"
        if self.q8_kv is None or self.q8_kv.shape[0] < kv_rows:
            rows = -(-kv_rows // eng.SCRATCH_PAGE) * eng.SCRATCH_PAGE
            self.q8_kv = torch.empty(rows, kv4.DIM_QK, dtype=torch.float8_e4m3fn, device=self.device)
        if self.q8_q is None or self.q8_q.shape[0] < tokens or self.q8_q.shape[1] != heads:
            self.q8_q = torch.zeros(tokens, heads, kv4.DIM_QK, dtype=torch.float8_e4m3fn, device=self.device)
        if self.q8_out is None or self.q8_out[0].shape[0] < tokens or self.q8_out[0].shape[1] != heads:
            self.q8_out = (torch.empty(tokens, heads, d_v, dtype=torch.bfloat16, device=self.device),
                           torch.empty(tokens, heads, dtype=torch.float32, device=self.device),
                           torch.empty(tokens, heads, dtype=torch.float32, device=self.device))
        if self.q8_one is None:
            self.q8_one = torch.ones(1, dtype=torch.float32, device=self.device)
        return self.q8_kv, self.q8_q, self.q8_out, self.q8_one

    def _alloc(self, rows: int):
        rows = -(-rows // eng.SCRATCH_PAGE) * eng.SCRATCH_PAGE
        return torch.empty(rows, kv4.DIM_QK, dtype=torch.bfloat16, device=self.device)

    def _alloc_v32(self, rows: int):
        rows = -(-rows // eng.SCRATCH_PAGE) * eng.SCRATCH_PAGE  # the decode kernel views the scratch as 64-row pages
        return torch.zeros(rows, V32_ROW, dtype=torch.uint8, device=self.device)

    def reserve_graph(self, tokens: int, k: int) -> None:
        rows = tokens * k
        if self.decode == "v32":
            if self.v32_graph is None or self.v32_graph.shape[0] < rows:
                assert not torch.cuda.is_current_stream_capturing(), "glm_kv4: v32 scratch requested during capture"
                self.v32_graph = self._alloc_v32(rows)
                self.graph_table = torch.empty(tokens, k, dtype=torch.int32, device=self.device)
                logger.warning("numinous glm_kv4 (SM90): v32 graph scratch %d tokens x %d rows = %.0f MB", tokens, k,
                               rows * V32_ROW / 2**20)
            return
        if self.graph_scratch is None or self.graph_scratch.shape[0] < rows:
            assert not torch.cuda.is_current_stream_capturing(), "glm_kv4: graph scratch requested during capture"
            self.graph_scratch = self._alloc(rows)
            self.graph_table = torch.empty(tokens, k, dtype=torch.int32, device=self.device)
            self.graph_work = self._work(rows)
            logger.warning("numinous glm_kv4 (SM90): graph scratch %d tokens x %d rows = %.0f MB bf16", tokens, k,
                           rows * kv4.DIM_QK * 2 / 2**20)

    def v32_buffers(self, tokens: int, k: int):
        """(scratch, table) for a v32 call of `tokens` tokens: the graph pair under capture, else the eager pair
        (grown outside capture only)."""
        if torch.cuda.is_current_stream_capturing():
            assert self.v32_graph is not None, "glm_kv4: init_cuda_graph_state did not reserve the v32 scratch"
            return self.v32_graph, self.graph_table
        rows = tokens * k
        if self.v32_eager is None or self.v32_eager.shape[0] < rows:
            self.v32_eager = self._alloc_v32(rows)
        if self.eager_table is None or self.eager_table.shape[0] < tokens or self.eager_table.shape[1] != k:
            self.eager_table = torch.empty(tokens, k, dtype=torch.int32, device=self.device)
            self.eager_work = self._work(tokens * k)
        return self.v32_eager, self.eager_table


def full_pool(pool):
    """The DSA pool that holds the FP4 rows: the pool itself, or the full-attention pool inside a hybrid
    (KDA/SSM + DSA) HybridLinearKVPool (GLM-5.3-Flash; memory_pool.py:3870-3892)."""
    return getattr(pool, "full_kv_pool", pool)


def layer_buffer(pool, layer_id: int) -> torch.Tensor:
    """kv_buffer of a model layer: the hybrid wrapper maps global layer ids to its dense full-attention index
    (HybridLinearKVPool._transfer_full_attention_id); a plain DSA pool offsets by start_layer."""
    inner = full_pool(pool)
    if inner is not pool:
        return inner.kv_buffer[pool._transfer_full_attention_id(layer_id)]
    return pool.kv_buffer[layer_id - pool.start_layer]


def _state(backend) -> _State90:
    st = getattr(backend, "_numinous_kv4_state", None)
    if st is None:
        pool = full_pool(backend.token_to_kv_pool)
        st = _State90(torch.device(backend.device), pool.kv_buffer[0].shape[0])
        backend._numinous_kv4_state = st
    assert isinstance(st, _State90), type(st)
    return st


def attach_problem(is_hip, device_sm_major, prefill_impl, decode_impl, kv_cache_dtype, hisparse, index_kpool) -> str:
    """Why this SM90 DSA backend cannot read the FP4 pool; "" when it can. attach() raises on it; the CPU preflight
    (numinous_preflight) evaluates it on the resolved server args."""
    return (
        "a ROCm build (use glm_kv4_rocm_hooks.py)" if is_hip
        else f"sm_{device_sm_major}x (need SM90; SM100 uses glm_kv4_hooks.py)" if device_sm_major != 9
        else (f"dsa roles prefill={prefill_impl} decode={decode_impl} (need {ROLE}/{ROLE}: "
              f"every other SM90 role reads the pool as its own layout)")
        if prefill_impl != ROLE or decode_impl != ROLE
        else f"kv dtype {kv_cache_dtype} (need fp8_e4m3)" if kv_cache_dtype != torch.float8_e4m3fn
        else "hisparse (swaps pool pages itself)" if hisparse
        else f"index kpool {index_kpool} (GLM-5.3 has none)" if index_kpool > 1
        else ""
    )


def numinous_preflight(ctx) -> list:
    """CPU preflight (operations/hopper/hook_preflight.py) of what attach() refuses at boot, on the resolved server
    args. The backend fields come from where v0.5.20 dsa_backend.__init__ reads them: the DSA roles from the resolved
    server args (dsa_backend.py:356-357), the KV dtype from the model runner (:441; ctx.kv_cache_torch_dtype is the
    stock kv_cache_dtype.configure_kv_cache_dtype of the flag), hisparse from --enable-hisparse (:351,
    model_runner.py:940-952), the index kpool from the HF config (:336, model_config.get_dsa_index_kpool). Only when
    the FP4 pool is built (glm_kv4_engine.enabled_for, which glm_kv4_engine.numinous_preflight checks itself)."""
    try:
        decode_mode()
    except ValueError as e:
        return [str(e)]
    try:
        if not eng.enabled_for(ctx.hf_config, ctx.kv_cache_torch_dtype):
            return []
    except (ValueError, RuntimeError):
        return []                    # reported by glm_kv4_engine.numinous_preflight
    if ctx.device.get("sm_major") != 9 and not ctx.device.get("is_hip"):
        return []                    # SM100 cells attach through glm_kv4_engine
    why = attach_problem(bool(ctx.device.get("is_hip")), ctx.device.get("sm_major"), ctx.args.get("dsa_prefill_backend"),
                         ctx.args.get("dsa_decode_backend"), ctx.kv_cache_torch_dtype,
                         bool(ctx.args.get("enable_hisparse")), int(getattr(ctx.hf_config, "index_kpool", 1) or 1))
    return [f"numinous glm_kv4 (SM90): the FP4 pool is built but this backend cannot read it: {why}"] if why else []


def attach(backend, model_runner) -> bool:
    """From DeepseekSparseAttnBackend.__init__ (CUDA SM90): True when this backend serves the FP4 pool."""
    pool = full_pool(model_runner.token_to_kv_pool)
    if not getattr(pool, "numinous_kv4", False):
        if eng.enabled():
            eng.log_once("nopool", "numinous glm_kv4: %s=1 but the pool is %s; FP4 attention off", eng.ENV,
                         type(pool).__name__)
        return False
    why = attach_problem(torch.version.hip is not None, backend.device_sm_major, backend.dsa_prefill_impl,
                         backend.dsa_decode_impl, backend.kv_cache_dtype,
                         getattr(backend, "hisparse_coordinator", None) is not None,
                         getattr(backend, "dsa_index_kpool", 1))
    if why:
        raise RuntimeError(f"numinous glm_kv4 (SM90): the FP4 pool is built but this backend cannot read it: {why}")
    # forward_mha's FP8 DSA prefix fetch (dequantize_k_cache_paged) only knows the 656-byte row
    backend.supports_mha_one_shot = False
    st = _state(backend)
    eng.log_once("attach", "numinous glm_kv4 attention active (SM90 %s): mode=%s, %d heads, top-k %d, pool %d slots, "
                           "MHA one-shot off", ROLE, st.mode, backend.num_q_heads, backend.dsa_index_topk, st.n_pool)
    return True


def reserve_graph_scratch(backend, max_num_tokens: int) -> None:
    """From init_cuda_graph_state: the fixed decode/verify scratch (max_num_tokens x top-k bf16 rows) and table."""
    st = _state(backend)
    if st.mode == "scratch":
        st.reserve_graph(max(int(max_num_tokens), 1), int(backend.dsa_index_topk))
        if st.dedup and st.decode == "bf16":
            st.stamps()


def note_batch(backend, forward_batch) -> None:
    """From forward_decode / forward_extend: the batch the next _forward_flashmla_kv call belongs to (the sequence
    prefill needs its mode and seq_lens_cpu; _forward_flashmla_kv itself is not given the batch)."""
    backend._numinous_kv4_fb = forward_batch


def _attend(backend, q_rows: torch.Tensor, scratch: torch.Tensor, table: torch.Tensor, sm_scale: float,
            v_head_dim: int, topk_length: Optional[torch.Tensor]) -> torch.Tensor:
    """The stock SM90 bf16 sparse MLA call (head padding to 64 included) on the scratch."""
    return backend._forward_flashmla_sparse(
        q_all=q_rows, kv_cache=scratch.view(-1, 1, kv4.DIM_QK), v_head_dim=v_head_dim, page_table_1=table,
        sm_scale=sm_scale, topk_length=topk_length)


def _sequence_prefill(backend, st: _State90, q_all, kv_buffer, page_table_1, forward_batch, metadata, sm_scale,
                      v_head_dim, topk_length) -> Optional[torch.Tensor]:
    """Plain prefill: dequantize every request's sequence once. None when the batch does not fit or lacks tables."""
    pt = getattr(metadata, "page_table_1", None)
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if pt is None or seq_lens_cpu is None or pt.dim() != 2 or pt.shape[0] != seq_lens_cpu.numel():
        return None
    if pt.shape[0] != forward_batch.batch_size:
        return None
    lens = seq_lens_cpu.to(torch.int64)
    rows = int(lens.sum())
    if rows == 0 or rows > st.seq_rows_cap:
        return None
    max_len = min(int(lens.max()), pt.shape[1])
    bases = torch.zeros(lens.numel(), dtype=torch.int32)
    bases[1:] = torch.cumsum(lens, 0)[:-1].to(torch.int32)
    bases = bases.to(st.device, non_blocking=True)
    scratch, table = st.eager(rows, page_table_1.shape[0], page_table_1.shape[1])
    seq_lens_dev = getattr(metadata, "cache_seqlens_int32", None)
    if seq_lens_dev is None or seq_lens_dev.numel() != lens.numel():
        seq_lens_dev = lens.to(torch.int32).to(st.device, non_blocking=True)
    kv4.dequant_seq(kv_buffer, pt.contiguous(), seq_lens_dev, bases, max_len, scratch, st.slot_map())
    tab = table[: page_table_1.shape[0]]
    kv4.remap_table(page_table_1, st.slot_to_pos, st.n_pool, tab)
    st.calls["seq"] += 1
    if st.prefill == "q8":
        return _attend_q8(st, q_all, scratch, rows, tab, sm_scale, v_head_dim, topk_length)
    return _attend(backend, q_all, scratch, tab, sm_scale, v_head_dim, topk_length)


def _cast_e4m3(dst: torch.Tensor, src: torch.Tensor, scale: Optional[torch.Tensor]) -> None:
    """dst (E4M3) = src (bf16) / scale, one rounding from the fp32 quotient; scale None = identity, the plain
    bf16 -> E4M3 cast copy v0.5.20's q8 path uses (kv_padded[:n].copy_(kv_bf16))."""
    if scale is None:
        dst.copy_(src)
        return
    for s in range(0, src.shape[0], Q8_CAST_ROWS):
        e = min(s + Q8_CAST_ROWS, src.shape[0])
        dst[s:e].copy_(src[s:e].float().div_(scale))


def _amax_scale(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor E4M3 scale amax / 448 as a 1-element fp32 device tensor (no host sync); 1 for an all-zero x."""
    a = x.abs().amax().float().reshape(1)
    return torch.where(a > 0, a / E4M3_MAX, torch.ones_like(a))


def _attend_q8(st: _State90, q_all: torch.Tensor, scratch: torch.Tensor, rows: int, table: torch.Tensor,
               sm_scale: float, v_head_dim: int, topk_length: Optional[torch.Tensor]) -> torch.Tensor:
    """The sequence prefill through v0.5.20's native FP8 SM90 sparse prefill kernel (see PREFILL_ENV): the first
    `rows` scratch rows cast to E4M3 plus `topk` zero rows, q cast to E4M3 with heads zero-padded to 64."""
    from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90 import sparse_mla_q8kv8_prefill_fwd

    t, h, d = q_all.shape
    hp = -(-h // 64) * 64
    k = table.shape[1]
    kv_buf, q_buf, (o_buf, ml_buf, lse_buf), one = st.q8_buffers(rows + k, t, hp, v_head_dim)
    kv = kv_buf[: rows + k]
    src = scratch[:rows]
    if st.q8_scale == "amax":
        kv_scale, q_scale = _amax_scale(src), _amax_scale(q_all)
    else:
        kv_scale = q_scale = one
    _cast_e4m3(kv[:rows], src, kv_scale if st.q8_scale == "amax" else None)
    kv[rows:].zero_()   # every call: a previous, larger call may have left rows here
    q = q_buf[:t]
    _cast_e4m3(q[:, :h], q_all, q_scale if st.q8_scale == "amax" else None)
    if hp != h:
        q[:, h:].zero_()
    # topk_length stays None: the full-topk entry is v0.5.20's default (SGLANG_ENABLE_DSA_Q8KV8_TOPK_LENGTH off). Only
    # that branch clamps a -1 slot onto a zero pad row before the load (kernel.cuh:803-822); masking is identical.
    out = sparse_mla_q8kv8_prefill_fwd(
        q=q, kv=kv.view(-1, 1, d), indices=table.view(t, 1, k), sm_scale=sm_scale, q_scale=q_scale,
        kv_scale=kv_scale, d_v=v_head_dim, attn_sink=None, topk_length=None, out=o_buf[:t],
        max_logits=ml_buf[:t], lse=lse_buf[:t])[0]
    st.calls["q8"] += 1
    return out[:, :h] if hp != h else out


def _selected_rows(backend, st: _State90, q_all, kv_buffer, page_table_1, sm_scale, v_head_dim, topk_length,
                   dedup: bool) -> torch.Tensor:
    """Gather + dequantize exactly the selected rows, in token chunks that fit the scratch (fixed under capture)."""
    t, k = page_table_1.shape
    if torch.cuda.is_current_stream_capturing():
        assert st.graph_scratch is not None, "glm_kv4: init_cuda_graph_state did not reserve the graph scratch"
        assert st.graph_table.shape[1] == k, (st.graph_table.shape, k)
        scratch, table, work = st.graph_scratch, st.graph_table, st.graph_work
    else:
        scratch, table = st.eager(min(t, st.sel_tokens) * k, min(t, st.sel_tokens), k)
        work = st.eager_work
    per_call = max(1, min(scratch.shape[0] // k, table.shape[0]))
    outs = []
    for c0 in range(0, t, per_call):
        c1 = min(t, c0 + per_call)
        sub = page_table_1[c0:c1]
        tab = table[: c1 - c0]
        if dedup:
            kv4.dequant_rows_dedup(kv_buffer, sub, st.stamps(), scratch, tab, work=work)
        else:
            kv4.dequant_rows(kv_buffer, sub, scratch)
            kv4.identity_table(sub, tab)
        outs.append(_attend(backend, q_all[c0:c1], scratch, tab, sm_scale, v_head_dim,
                            None if topk_length is None else topk_length[c0:c1]))
    st.calls["sel"] += 1
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def _flashmla_fp8_decode(backend, q_rows: torch.Tensor, scratch: torch.Tensor, table: torch.Tensor, sm_scale: float,
                         v_head_dim: int, cache_seqlens: torch.Tensor, meta) -> torch.Tensor:
    """The stock SM90 FP8 sparse decode call of dsa_backend._forward_flashmla_kv, on the V32 scratch: q [T, H, 576]
    bf16 padded to flashmla_kv_num_q_heads, the scratch as 64-row pages, `table` [T, K] scratch rows. -> [T, H, 512]."""
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

    t, h, d = q_rows.shape
    hp = int(getattr(backend, "flashmla_kv_num_q_heads", h))
    q_in = q_rows.view(t, 1, h, d)
    if hp != h:
        q_pad = q_rows.new_zeros(t, 1, hp, d)
        q_pad[:, :, :h, :] = q_in
        q_in = q_pad
    if meta is None:
        m, splits = get_mla_metadata(cache_seqlens=cache_seqlens, num_q_tokens_per_head_k=hp, num_heads_k=1,
                                     num_heads_q=hp, is_fp8_kvcache=True, topk=table.shape[1])
    else:
        m, splits = meta.flashmla_metadata, meta.num_splits
    o, _ = flash_mla_with_kvcache(
        q=q_in, k_cache=scratch.view(-1, eng.SCRATCH_PAGE, 1, V32_ROW), cache_seqlens=cache_seqlens,
        head_dim_v=v_head_dim, tile_scheduler_metadata=m, num_splits=splits, softmax_scale=sm_scale,
        indices=table.unsqueeze(1), block_table=torch.empty((t, 0), dtype=torch.int32, device=q_rows.device),
        is_fp8_kvcache=True)
    o = o.view(t, hp, v_head_dim)
    return o[:, :h, :] if hp != h else o


def _selected_rows_v32(backend, st: _State90, q_all, kv_buffer, page_table_1, sm_scale, v_head_dim,
                       metadata) -> torch.Tensor:
    """Decode / verify / draft decode: the selected FP4 rows re-quantized into V32 rows in table order, then the stock
    FP8 sparse decode kernel with an identity table. The batch's own FlashMLA tile metadata is used when the call
    covers the whole batch (always under capture: the graph scratch holds max_num_tokens x top-k rows); eager chunks
    get their own."""
    t, k = page_table_1.shape
    capturing = torch.cuda.is_current_stream_capturing()
    scratch, table = st.v32_buffers(t if capturing else min(t, st.sel_tokens), k)
    per_call = max(1, min(scratch.shape[0] // k, table.shape[0]))
    if capturing:
        assert per_call >= t, ("glm_kv4: the v32 graph scratch is smaller than a captured batch", t, per_call)
    seqlens = getattr(metadata, "dsa_cache_seqlens_int32", None)
    if seqlens is None or seqlens.shape[0] != t:
        seqlens = torch.ones(t, dtype=torch.int32, device=q_all.device)
    batch_meta = getattr(metadata, "flashmla_metadata", None)
    outs = []
    for c0 in range(0, t, per_call):
        c1 = min(t, c0 + per_call)
        sub = page_table_1[c0:c1]
        tab = table[: c1 - c0]
        kv4_to_v32(kv_buffer, sub, scratch)
        kv4.identity_table(sub, tab)
        whole = c0 == 0 and c1 == t
        outs.append(_flashmla_fp8_decode(backend, q_all[c0:c1], scratch, tab, sm_scale, v_head_dim, seqlens[c0:c1],
                                         batch_meta if whole else None))
    st.calls["v32"] += 1
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def forward_flashmla_kv(backend, q_all: torch.Tensor, page_table_1: torch.Tensor, sm_scale: float, v_head_dim: int,
                        layer, metadata) -> torch.Tensor:
    """Replaces DeepseekSparseAttnBackend._forward_flashmla_kv for the FP4 pool.

    q_all [T, H, 576] bf16 (nope | rope, RoPE applied; any view with T*H*576 elements), page_table_1 [T, K] int32
    pool slots (-1 padded). Returns [T, H, 512] bf16 (the flashmla_sparse output shape; the caller flattens)."""
    st = _state(backend)
    kv_buffer = layer_buffer(backend.token_to_kv_pool, layer.layer_id)
    q_all = q_all.reshape(-1, layer.tp_q_head_num, layer.head_dim)
    t = q_all.shape[0]
    if t == 0:
        return q_all.new_empty(0, layer.tp_q_head_num, v_head_dim)
    if layer.head_dim == kv4.DIM_NOPE:
        # zero-rope DSA (GLM-5.3-Flash): the rows dequantize to 576 with zero rope; pad q to match
        q_all = torch.nn.functional.pad(q_all, (0, kv4.DIM_ROPE))
    if not page_table_1.is_contiguous():
        page_table_1 = page_table_1.contiguous()
    if page_table_1.dtype != torch.int32:
        page_table_1 = page_table_1.to(torch.int32)
    assert page_table_1.shape[0] == t, (page_table_1.shape, q_all.shape)
    topk_length = getattr(metadata, "dsa_cache_seqlens_int32", None)
    if topk_length is not None and topk_length.shape[0] != t:
        topk_length = None
    if st.mode == "triton":
        out = kv4.sparse_mla(q_all, kv_buffer, page_table_1, sm_scale)
        st.calls["triton"] += 1
        return out
    fb = getattr(backend, "_numinous_kv4_fb", None)
    mode = getattr(fb, "forward_mode", None)
    if mode is not None:
        eng._count_call(st, mode, t)
    if (fb is not None and mode is not None and not torch.cuda.is_current_stream_capturing()
            and (mode.is_extend_or_draft_extend_or_mixed() or mode.is_mixed())
            and not mode.is_target_verify()):
        out = _sequence_prefill(backend, st, q_all, kv_buffer, page_table_1, fb, metadata, sm_scale, v_head_dim,
                                topk_length)
        if out is not None:
            return out
    if st.decode == "v32":
        return _selected_rows_v32(backend, st, q_all, kv_buffer, page_table_1, sm_scale, v_head_dim, metadata)
    dedup = st.dedup and t >= DEDUP_MIN_ROWS and not (mode is not None and mode.is_decode_or_idle())
    return _selected_rows(backend, st, q_all, kv_buffer, page_table_1, sm_scale, v_head_dim, topk_length, dedup)
