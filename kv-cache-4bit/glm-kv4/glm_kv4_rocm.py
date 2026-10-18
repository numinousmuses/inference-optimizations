"""GLM-5.3 4-bit KV cache on ROCm (MI355X gfx950): the TileLang DSA read path over the 352-byte FP4 pool.

Installed by operations/patches/glm_kv4_rocm_hooks.py as sglang/srt/layers/attention/dsa/glm_kv4_rocm.py next to
glm_kv4.py (kernels, unchanged: the Triton quantize-write / dequantize / dedup kernels are bit-exact on gfx950,
operations/kernels/test_glm_kv4.py 20 passed on the MI355X box, 2026-09-29) and glm_kv4_engine.py (the pool class
and scratch state, shared with the CUDA path). Behind NUMINOUS_GLM_KV4=1 like the CUDA path.

ROCm differences from the SM100 path (glm_kv4_engine.engine_forward):
  roles   the ROCm DSA backend serves prefill and decode with TileLang (`tilelang_sparse_fwd`); with an FP8 KV flag
          it reads the raw 576-byte layout [512 E4M3 latent | 64 E4M3 rope] with no per-block scale, the same
          layout the FP4 scratch produces, so the stock kernel runs on the scratch unchanged.
  write   the fused AITER rope + concat + cache kernel writes FP8 rows straight into the pool on gfx95; the hooks
          turn that fusion off when the pool is FP4 (RoPE runs in prepare, the backend's save_kv_cache path calls
          set_mla_kv_buffer with the bf16 latent + rope, and GLMKV4DSATokenToKVPool quantizes from bf16).
  read    `forward_tilelang` replaces DeepseekSparseAttnBackend._forward_tilelang: the selected rows (top-k pool
          slots per token, -1 padded) are dequantized into an FP8 scratch (plain gather, or dedup when many tokens
          of the call share rows: verify, draft extend, prefill) and TileLang runs on the scratch with the table
          remapped to scratch rows, padded to a multiple of 64 columns as the stock path pads.
  graphs  the decode/verify scratch is reserved in init_cuda_graph_state (max_num_tokens x top-k rows) and never
          reallocated; eager calls larger than the scratch are chunked by tokens.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.dsa import glm_kv4 as kv4
from sglang.srt.layers.attention.dsa import glm_kv4_engine as eng

TILE = 64  # TileLang processes indices in 64-column blocks


def attach(backend, model_runner) -> bool:
    """From DeepseekSparseAttnBackend.__init__ (ROCm): True when this backend serves the FP4 pool."""
    pool = model_runner.token_to_kv_pool
    if not getattr(pool, "numinous_kv4", False):
        if eng.enabled():
            eng.log_once("nopool", "numinous glm_kv4: %s=1 but the pool is %s; FP4 attention off", eng.ENV,
                         type(pool).__name__)
        return False
    why = (
        "not a ROCm build" if torch.version.hip is None
        else f"dsa roles prefill={backend.dsa_prefill_impl} decode={backend.dsa_decode_impl} (need tilelang/tilelang)"
        if backend.dsa_prefill_impl != "tilelang" or backend.dsa_decode_impl != "tilelang"
        else f"kv dtype {backend.kv_cache_dtype}" if backend.kv_cache_dtype != torch.float8_e4m3fn
        else ""
    )
    if why:
        raise RuntimeError(f"numinous glm_kv4 (ROCm): the FP4 pool is built but this backend cannot read it: {why}")
    st = eng._state(backend)
    if st.mode != "scratch":
        raise RuntimeError("numinous glm_kv4 (ROCm): only NUMINOUS_GLM_KV4_ATTN=scratch is supported on ROCm (the fused "
                           "Triton sparse_mla aborts on gfx950 at block_n 64)")
    eng.log_once("attach", "numinous glm_kv4 attention active (ROCm TileLang): mode=%s, %d heads, top-k %d, "
                           "pool %d slots", st.mode, backend.num_q_heads, backend.dsa_index_topk, st.n_pool)
    return True


def _pad_cols(k: int) -> int:
    return -(-k // TILE) * TILE


def reserve_graph_scratch(backend, max_num_tokens: int) -> None:
    """From init_cuda_graph_state: the fixed decode/verify scratch (max_num_tokens x top-k rows) and table."""
    st = eng._state(backend)
    k = _pad_cols(backend.dsa_index_topk)
    st.reserve_graph(max(int(max_num_tokens), 1), k)
    if st.dedup:
        st.stamps()


def forward_tilelang(backend, q_all: torch.Tensor, kv_cache: torch.Tensor, page_table_1: torch.Tensor,
                     sm_scale: float, v_head_dim: int) -> torch.Tensor:
    """Replaces DeepseekSparseAttnBackend._forward_tilelang for the FP4 pool.

    q_all [T, H, 576] (bf16 or fp8), kv_cache = the layer's FP4 pool buffer [slots, 1, 352] uint8, page_table_1
    [T, K] int32 pool slots (-1 padded). Returns what tilelang_sparse_fwd returns for the same call on an FP8 pool."""
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    st = eng._state(backend)
    kv_buffer = kv_cache.view(kv_cache.shape[0], -1) if kv_cache.dim() == 3 else kv_cache
    t, k = page_table_1.shape
    if not page_table_1.is_contiguous():
        page_table_1 = page_table_1.contiguous()
    kp = _pad_cols(k)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        assert st.graph_scratch is not None, "glm_kv4: init_cuda_graph_state did not reserve the graph scratch"
        assert st.graph_table.shape[1] >= k, (st.graph_table.shape, k)
        scratch, work = st.graph_scratch, st.graph_work
        per_call = max(1, min(scratch.shape[0] // k, st.graph_table.shape[0]))
    else:
        scratch, _ = st.eager(min(t, st.sel_tokens) * k, min(t, st.sel_tokens), kp)
        work = st.eager_work
        per_call = max(1, min(scratch.shape[0] // k, st.sel_tokens))
    dedup = st.dedup and t >= eng.DEDUP_MIN_ROWS
    outs = []
    for c0 in range(0, t, per_call):
        c1 = min(t, c0 + per_call)
        sub = page_table_1[c0:c1]
        tab = torch.empty(c1 - c0, k, dtype=torch.int32, device=q_all.device)
        if dedup:
            kv4.dequant_rows_dedup(kv_buffer, sub, st.stamps(), scratch, tab, work=work)
        else:
            kv4.dequant_rows(kv_buffer, sub, scratch)
            kv4.identity_table(sub, tab)
        if kp != k:
            tab = torch.cat((tab, tab.new_full((c1 - c0, kp - k), -1)), dim=1)
        outs.append(tilelang_sparse_fwd(q=q_all[c0:c1], kv=scratch.view(-1, 1, eng.FP8_ROW_BYTES),
                                        indices=tab.unsqueeze(1), sm_scale=sm_scale, d_v=v_head_dim))
    st.calls["sel"] += 1
    if len(outs) == 1:
        return outs[0]
    # the HIP combine kernel returns [1, T, H, d_v] (batch 1): tokens are dim 1
    return torch.cat(outs, dim=1)
