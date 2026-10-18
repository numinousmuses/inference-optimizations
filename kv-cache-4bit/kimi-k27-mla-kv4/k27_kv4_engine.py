"""Kimi K2.7 Code (dense MLA, DeepseekV3ForCausalLM) 4-bit KV cache inside SGLang v0.5.20: pool class and the
attention routing for the TRT-LLM / TokenSpeed MLA backends. Installed by operations/patches/k27_kv4_hooks.py as
sglang/srt/layers/attention/dsa/k27_kv4_engine.py next to glm_kv4.py (row format + Triton quantize / dequantize
kernels) and the kv4mla package (the native FP4 decode kernel, dense mode). Everything is behind NUMINOUS_K27_KV4=1.

What changes when it is on (dense MLA model, --kv-cache-dtype fp8_e4m3, tokenspeed_mla or trtllm_mla, TP attention
with <= 64 q heads per rank, no DCP):
  pool     kv_cache_configurator.calculate_mla_kv_cache_dim returns 352 (not 576) for non-DSA MLA models, so the
           pool configurator prices 352 B/token/layer (1.636x tokens at the same bytes, derived) and _build_mla_kv_pool
           builds K27KV4MLATokenToKVPool: MLATokenToKVPool whose rows are kv4glm's 352 bytes [256 B E2M1 latent | 32 B
           E4M3 block scales (16 values each) | 64 B E4M3 rope] and whose write door quantizes (glm_kv4.quant_write,
           from bf16 or FP8 input) and whose read door for prefill (get_mla_kv_buffer) dequantizes to bf16.
  write    every pool write already goes through set_mla_kv_buffer except the fused fp8 "scatter + q concat" decode
           path, which the hook turns off when the pool is FP4 (it writes raw 576-byte rows).
  prefill  the MHA prefill path reads prefix KV through get_mla_kv_buffer (dequantized rows); new tokens are attended
           from the bf16/FP8 K, V of the current chunk, never from the pool. The tokenspeed prepare_prefill_qkv write
           hands the bf16 latent to the pool instead of its FP8 copy (one rounding less).
  decode   forward_decode (non-DCP) and the target-verify / draft-extend branch of forward_extend call
           `decode`/`verify` below instead of the FP8 paged kernel: the kv4mla kernel in dense mode reads the 352-byte
           rows straight from the pool through the backend's block table (page ids), one q tile of 64 rows per
           sequence = tokens x heads (TP8: 8 heads, up to 8 tokens), with per-row position limits for the draft rows.
Not covered (loud asserts): DCP, DP attention with more than 64 rows per sequence tile (64 heads x several tokens),
the flashinfer fallback prefill (its ragged kernel reads bf16 rows from get_key_buffer): that fallback is what the
v0.5.20 prefill CUDA graph uses, so the FP4 pool requires --cuda-graph-backend-prefill disabled (checked at attach).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.attention.dsa import glm_kv4 as kv4
from sglang.srt.layers.attention.dsa.kv4mla import kv4mla
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

logger = logging.getLogger("sglang.numinous")

ENV = "NUMINOUS_K27_KV4"
ROW_BYTES = kv4.ROW_BYTES          # 352
FP8_ROW_BYTES = kv4.DIM_QK         # 576
H_TILE = kv4mla.H_Q                # 64 q rows per sequence tile
_logged: set = set()


def log_once(key: str, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.warning(msg, *args)


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def enabled_for(hf_config, kv_cache_dtype) -> bool:
    """On for dense (non-DSA) MLA models with the FP8 KV flag: the same trigger the pool builder and the pool
    configurator see."""
    if not enabled():
        return False
    cfg = getattr(hf_config, "text_config", None) or hf_config
    if getattr(cfg, "kv_lora_rank", None) != kv4.DIM_NOPE or getattr(cfg, "qk_rope_head_dim", None) != kv4.DIM_ROPE:
        log_once("dims", "numinous k27_kv4: off, kv_lora_rank/rope %s/%s are not 512/64",
                 getattr(cfg, "kv_lora_rank", None), getattr(cfg, "qk_rope_head_dim", None))
        return False
    if getattr(cfg, "index_topk", None) or getattr(cfg, "index_n_heads", None):
        log_once("dsa", "numinous k27_kv4: off, DSA model (use glm_kv4)")
        return False
    if kv_cache_dtype != torch.float8_e4m3fn:
        log_once("dtype", "numinous k27_kv4: off, needs --kv-cache-dtype fp8_e4m3 (got %s)", kv_cache_dtype)
        return False
    return True


class K27KV4MLATokenToKVPool(MLATokenToKVPool):
    """MLATokenToKVPool with 352-byte FP4 rows: kv_buffer[l] is [size + page_size, 1, 352] uint8."""

    numinous_kv4 = True

    def _create_buffers(self):
        # MLATokenToKVPool sets kv_cache_dim = kv_lora_rank + rope before calling this; our rows are 352 bytes
        assert self.kv_lora_rank == kv4.DIM_NOPE and self.qk_rope_head_dim == kv4.DIM_ROPE, (
            self.kv_lora_rank, self.qk_rope_head_dim)
        assert self.store_dtype == torch.uint8, f"k27_kv4 pool needs the FP8 KV flag (store dtype uint8), got {self.store_dtype}"
        self.kv_cache_dim = ROW_BYTES
        super()._create_buffers()
        logger.warning(
            "numinous k27_kv4 pool: %d layers x %d B/token (E2M1 latent + E4M3 scale per 16 + FP8 rope) = %d B/token "
            "vs %d for FP8; %d tokens", self.layer_num, ROW_BYTES, ROW_BYTES * self.layer_num,
            FP8_ROW_BYTES * self.layer_num, self.size)

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("k27_kv4 pool: combined-row set_kv_buffer is not an MLA write door (use set_mla_kv_buffer)")

    def _write_mla_kv_buffer(self, dst_buffer, loc, cache_k_nope, cache_k_rope) -> None:
        """The write door set_mla_kv_buffer (base: OOB checks, layer_id_override) ends in: quantize into FP4 rows."""
        from sglang.srt.runtime_context import get_parallel

        assert not get_parallel().dcp_enabled, "k27_kv4 pool has no DCP write path"
        assert cache_k_rope is not None and cache_k_rope.numel() > 0, "MLA rows always carry the rope part"
        nope = cache_k_nope.reshape(-1, kv4.DIM_NOPE)
        rope = cache_k_rope.reshape(-1, kv4.DIM_ROPE)
        if rope.dtype == torch.uint8:
            rope = rope.view(torch.float8_e4m3fn)
        if nope.dtype == torch.uint8:
            nope = nope.view(torch.float8_e4m3fn)
        kv4.quant_write(dst_buffer, loc.reshape(-1), nope, rope)

    def get_value_buffer(self, layer_id: int):
        raise RuntimeError("k27_kv4 pool: the latent is FP4; decode reads it through k27_kv4_engine")

    def get_mla_kv_buffer(self, layer, loc: torch.Tensor, dst_dtype: Optional[torch.dtype] = None):
        """Dequantized rows at loc: (nope [T, 1, 512], rope [T, 1, 64]) in dst_dtype (default bf16); the prefill
        prefix read door (forward_mha._get_mla_kv_buffer)."""
        dst_dtype = dst_dtype or torch.bfloat16
        buf = self.kv_buffer[layer.layer_id - self.start_layer]
        loc = loc.reshape(-1)
        out = torch.empty(loc.numel(), kv4.DIM_QK, dtype=torch.bfloat16, device=buf.device)
        kv4.dequant_rows(buf, loc, out)
        out = out.to(dst_dtype)
        return out[:, : kv4.DIM_NOPE].unsqueeze(1).contiguous(), out[:, kv4.DIM_NOPE:].unsqueeze(1).contiguous()


# ------------------------------------------------------------------------------------------------ backend glue


class _State:
    """Per-backend: the dense decoder, the q tile / row-limit buffers per batch size (graph-stable)."""

    def __init__(self, device: torch.device, heads: int, max_context_len: int, page_size: int):
        self.dec = kv4mla.DenseDecoder(device)
        self.heads = heads
        self.max_context_len = int(max_context_len)
        self.page_size = int(page_size)
        self.device = device
        self._q: dict = {}
        self._lim: dict = {}
        self.calls = {"decode": 0, "verify": 0}

    def q_tile(self, b: int) -> torch.Tensor:
        t = self._q.get(b)
        if t is None:
            t = torch.zeros(b, H_TILE, kv4.DIM_QK, dtype=torch.float8_e4m3fn, device=self.device)
            self._q[b] = t
        return t

    def limits(self, b: int) -> torch.Tensor:
        t = self._lim.get(b)
        if t is None:
            t = torch.zeros(b, H_TILE, dtype=torch.int32, device=self.device)
            self._lim[b] = t
        return t


def _state(backend) -> _State:
    st = getattr(backend, "_numinous_k27_kv4", None)
    if st is None:
        raise RuntimeError("k27_kv4: backend not attached (the init hook did not run)")
    return st


def attach(backend, model_runner) -> bool:
    """Called from the MLA backend __init__ hook. Verifies the layout the kernel supports and builds the state."""
    pool = model_runner.token_to_kv_pool
    if not getattr(pool, "numinous_kv4", False):
        return False
    from sglang.srt.runtime_context import get_parallel

    assert not get_parallel().dcp_enabled, "k27_kv4: DCP is not supported (TP attention only)"
    assert torch.cuda.get_device_capability()[0] == 10, "k27_kv4: the kv4mla kernel is SM100 only"
    assert backend.data_type == torch.float8_e4m3fn, f"k27_kv4: expects the FP8 q path, got {backend.data_type}"
    heads = int(backend.num_q_heads)
    assert heads <= H_TILE, f"k27_kv4: {heads} q heads per rank do not fit the 64-row tile"
    # the prefill CUDA graph (v0.5.20 default "breakable") falls back to the flashinfer MLA kernel, which reads the pool
    # as bf16 rows through get_key_buffer (trtllm_mla_backend.init_forward_metadata: is_in_breakable_cuda_graph); our
    # rows are 352 bytes, so the FP4 pool needs --cuda-graph-backend-prefill disabled (arms k27_kv4*, k27_ctl_nopfg)
    try:
        from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, check_cuda_graph_backend
        prefill_graph_off = check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
    except Exception:   # config not published (unit tests): nothing to check
        prefill_graph_off = True
    assert prefill_graph_off, (
        "k27_kv4: the prefill CUDA graph routes extend batches to the flashinfer MLA kernel, which cannot read FP4 rows; "
        "launch with --cuda-graph-backend-prefill disabled")
    assert not getattr(backend, "disable_chunked_prefix_cache", False), (
        "k27_kv4: --disable-chunked-prefix-cache routes prefix prefill to the flashinfer ragged kernel, which reads "
        "raw pool rows")
    max_ctx = int(getattr(backend, "max_context_len", model_runner.model_config.context_len))
    backend._numinous_k27_kv4 = _State(pool.kv_buffer[0].device, heads, max_ctx, backend.page_size)
    # the fused FP8 "quantize + scatter + q concat" decode write bypasses the pool's write door (raw 576-byte rows)
    backend._fused_set_kv_concat_q_fp8 = False
    backend._fused_set_kv_concat_q = False
    log_once("attach", "numinous k27_kv4: attached (%d q heads/rank, page %d, max context %d, %d tokens/rank at %d B)",
             heads, backend.page_size, max_ctx, pool.size, ROW_BYTES)
    return True


def prepare_graph(backend, max_bs: int) -> None:
    """Allocate every buffer the decode / verify calls touch before CUDA-graph capture."""
    st = _state(backend)
    for b in range(1, max_bs + 1):
        st.q_tile(b)
        st.limits(b)
    st.dec.prepare(max_bs)


def _rows(kv_buffer: torch.Tensor) -> torch.Tensor:
    return kv4.pool_rows(kv_buffer)


def decode(backend, query: torch.Tensor, layer, forward_batch, metadata) -> torch.Tensor:
    """query [b, (1,) heads, 576] fp8 with RoPE applied -> [b, heads * 512] bf16. seq_lens include the current token
    (its row was written before attention, as the FP8 path does)."""
    st = _state(backend)
    q = query.reshape(-1, st.heads, kv4.DIM_QK)
    b = q.shape[0]
    seq_lens = metadata.seq_lens_k if getattr(metadata, "seq_lens_k", None) is not None else forward_batch.seq_lens
    seq_lens = seq_lens[:b]
    tile = st.q_tile(b)
    tile[:, : st.heads].copy_(q)
    kv = _rows(backend.token_to_kv_pool.kv_buffer[layer.layer_id - backend.token_to_kv_pool.start_layer])
    out = st.dec(tile, kv, metadata.block_kv_indices[:b], seq_lens, st.page_size, st.max_context_len, None,
                 float(layer.scaling))
    st.calls["decode"] += 1
    return out[:, : st.heads].reshape(b, st.heads * kv4.DIM_NOPE)


def verify(backend, q: torch.Tensor, layer, forward_batch, metadata, n_tok: int,
           actual_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
    """q [b, n_tok, heads, 576] fp8 (RoPE applied; padded tokens allowed) -> [b, n_tok, heads, 512] bf16 (the shape
    the backend's unpad / view tail expects from its FP8 kernel).
    metadata.seq_lens_k counts the sequence with its n_tok new rows (target verify: seq_lens + draft_token_num;
    draft extend: seq_lens - extend_len + max_extend_len, i.e. padded to n_tok new rows per sequence, see
    init_forward_metadata). Row (t, h) attends seq_lens_k - n_tok + t + 1 positions; with actual_lens [b] (draft
    extend with ragged new-token counts) rows t >= actual attend nothing (their outputs are dropped by the unpad)."""
    st = _state(backend)
    b = q.shape[0]
    assert q.shape[1] == n_tok and q.shape[2] == st.heads, (q.shape, n_tok, st.heads)
    assert n_tok * st.heads <= H_TILE, f"k27_kv4: {n_tok} tokens x {st.heads} heads exceed the 64-row tile"
    total = metadata.seq_lens_k[:b].to(torch.int32)
    tile = st.q_tile(b)
    tile[:, : n_tok * st.heads].copy_(q.reshape(b, n_tok * st.heads, kv4.DIM_QK))
    lim = st.limits(b)
    t_idx = torch.arange(n_tok, device=q.device, dtype=torch.int32)
    per_tok = (total - n_tok)[:, None] + t_idx[None, :] + 1                 # [b, n_tok]
    if actual_lens is not None:
        per_tok = torch.where(t_idx[None, :] < actual_lens[:b, None], per_tok, torch.zeros_like(per_tok))
    lim[:, : n_tok * st.heads].copy_(per_tok.repeat_interleave(st.heads, dim=1))
    if n_tok * st.heads < H_TILE:
        lim[:, n_tok * st.heads:].zero_()
    kv = _rows(backend.token_to_kv_pool.kv_buffer[layer.layer_id - backend.token_to_kv_pool.start_layer])
    out = st.dec(tile, kv, metadata.block_kv_indices[:b], total, st.page_size, st.max_context_len + n_tok, lim,
                 float(layer.scaling))
    st.calls["verify"] += 1
    return out[:, : n_tok * st.heads].reshape(b, n_tok, st.heads, kv4.DIM_NOPE)
