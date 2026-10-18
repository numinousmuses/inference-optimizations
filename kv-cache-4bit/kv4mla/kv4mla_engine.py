"""kv4mla engine glue: routes GLM-5.3's FP4 DSA attention (kv4glm's 352-byte pool) to the native kernel.

Installed by operations/patches/kv4mla_hooks.py as sglang/srt/layers/attention/dsa/kv4mla_engine.py, next to kv4glm's
glm_kv4_engine.py (whose pool, write door and attach it reuses unchanged). Active with NUMINOUS_GLM_KV4=1 (the FP4 pool)
and NUMINOUS_GLM_KV4MLA=1:

  decode, target verify, draft extend, and prefill batches the sequence path cannot take
        -> kv4mla.Decoder (native sparse MLA over the FP4 rows, in-kernel dequant; no FP8 scratch)
  plain prefill (EXTEND / MIXED, eager) -> kv4glm's sequence path (each request's context dequantized once into an
        FP8 scratch + the TRT-LLM op: level with FP8 at prefill, measured), falling back to the native kernel when it
        does not fit

Graph safety: the split-KV schedule depends only on (T, K); `prepare_graph` builds it for every T up to the largest
graph batch before capture, so replays run exactly the attention kernel + combine. Whether a batch uses split-KV follows
kv4mla.split_policy (measured table); batches without split-KV need no workspace. Math mode: KV4MLA_MODE=fp8 (default,
E4M3 tensor cores like the FP8 TRT-LLM op) or exact (fp16 over exactly dequantized rows, slower).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.attention.dsa import glm_kv4_engine as kv4e
from sglang.srt.layers.attention.dsa.kv4mla import kv4mla as native

logger = logging.getLogger("sglang.numinous")
ENV = "NUMINOUS_GLM_KV4MLA"
# batches of at most MIN_T tokens go to kv4glm's selected-row scratch path instead (measured on B200: T=1 12.0 vs 22.1 us,
# T=8 17.0 vs 22.6; the native kernel wins from T=32: 27.6 vs 33.5). 0 = native for every batch (the gated default).
MIN_T = int(os.environ.get("NUMINOUS_GLM_KV4MLA_MIN_T", "0"))


def enabled() -> bool:
    return kv4e.enabled() and os.environ.get(ENV, "0") == "1"


class _State:
    def __init__(self, device):
        self.dec = native.Decoder(device)
        self.calls = {"native": 0, "seq": 0}


def _state(backend) -> Optional[_State]:
    st = getattr(backend, "_numinous_kv4mla_state", None)
    if st is None:
        if backend.num_q_heads != native.H_Q:
            kv4e.log_once("kv4mla_heads", "numinous kv4mla: %d q heads per rank (kernel needs 64); using kv4glm's path",
                          backend.num_q_heads)
            backend._numinous_kv4mla_state = False
            return None
        st = _State(torch.device(backend.device))
        backend._numinous_kv4mla_state = st
        kv4e.log_once("kv4mla_attach", "numinous kv4mla native FP4 sparse MLA active: %d heads, top-k %d, %d SM parts, "
                      "mode %s", backend.num_q_heads, backend.dsa_index_topk, st.dec.parts, st.dec.mode)
    return st or None


def _k(backend) -> int:
    return -(-backend.dsa_index_topk // native.B_TOPK) * native.B_TOPK


def prepare_graph(backend, max_num_tokens: int) -> bool:
    """From init_cuda_graph_state (before capture): schedules for every batch up to max_num_tokens and the split-KV
    workspace. Returns False when the native path does not apply (the caller keeps kv4glm's scratch reservation)."""
    st = _state(backend)
    if st is None:
        return False
    if MIN_T > 0:   # small graph batches take the scratch path: reserve its graph scratch for them only
        kv4e.reserve_graph_scratch(backend, min(int(max_num_tokens), MIN_T))
    k = _k(backend)
    n = 0
    for t in range(1, int(max_num_tokens) + 1):
        if native.split_policy(t, k):
            st.dec.prepare(t, k)
            n += 1
    torch.cuda.synchronize()
    logger.warning("numinous kv4mla: graph schedules for %d batch sizes up to %d tokens (K %d), no FP8 scratch reserved",
                   n, int(max_num_tokens), k)
    return True


def engine_forward(backend, q_all: torch.Tensor, page_table_1: torch.Tensor, sparse_mla_top_k: int, layer,
                   forward_batch, metadata, seq_lens: Optional[torch.Tensor], is_prefill: bool) -> torch.Tensor:
    """Replaces the TRT-LLM call in DeepseekSparseAttnBackend._forward_trtllm (same contract as
    glm_kv4_engine.engine_forward): q_all [T, H, 576] fp8, page_table_1 [T, K] pool slots (-1 padded)."""
    st = _state(backend)
    if st is None:
        return kv4e.engine_forward(backend, q_all, page_table_1, sparse_mla_top_k, layer, forward_batch, metadata,
                                   seq_lens, is_prefill)
    pool = backend.token_to_kv_pool
    kv_buffer = pool.kv_buffer[layer.layer_id - pool.start_layer]
    t, h, _ = q_all.shape
    out = torch.empty(t, 1, h, native.D_V, dtype=torch.bfloat16, device=q_all.device)
    if t == 0:
        return out
    if q_all.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"numinous kv4mla: expected FP8 q from the trtllm path, got {q_all.dtype}")
    scale = kv4e._bmm1_scale(layer)
    mode = forward_batch.forward_mode
    if (is_prefill and not torch.cuda.is_current_stream_capturing()
            and (mode.is_extend_or_draft_extend_or_mixed() or mode.is_mixed())
            and not mode.is_target_verify()):
        kst = kv4e._state(backend)
        sl = seq_lens if seq_lens is not None else metadata.cache_seqlens_int32
        if kv4e._sequence_prefill(backend, kst, q_all, kv_buffer, page_table_1, sparse_mla_top_k, forward_batch,
                                  metadata, sl, scale, out):
            st.calls["seq"] += 1
            return out
    if t <= MIN_T:
        return kv4e.engine_forward(backend, q_all, page_table_1, sparse_mla_top_k, layer, forward_batch, metadata,
                                   seq_lens, is_prefill)
    table = page_table_1 if page_table_1.dtype == torch.int32 else page_table_1.to(torch.int32)
    if table.stride(-1) != 1:
        table = table.contiguous()
    st.dec(q_all, kv_buffer, table, scale, out=out.view(t, h, native.D_V))
    st.calls["native"] += 1
    return out
