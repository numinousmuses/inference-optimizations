"""k3state engine glue: two-tier quantized KDA state for Kimi K3 on SGLang v0.5.20 and main 172b1b4825 (the ROCm
image tree) (NUMINOUS_K3STATE=fp8|fp4|bf16).

Installed by operations/patches/k3state_hooks.py as sglang/srt/layers/attention/numinous_k3state.py (the kernels as
numinous_k3state_quant.py). Inert unless NUMINOUS_K3STATE is fp8 or fp4.

Layout
------
* Pool Q (every mamba slot: running requests, radix-cached prefixes, extra_buffer track slots): the KDA state stored
  quantized, uint8 [layers, slots+1, HV, V, row_bytes] with the scales inside each row (k3state_quant). All engine byte
  copies (copy-on-write, track, clear, host) stay correct without knowing the format.
* Working state W (one row per request-pool row, fp32 [layers, R, HV, V, K]): the exact recurrent state of each
  running request. Speculative verify reads W; the commit fold advances W in fp32 (bit-identical to SGLang's fold)
  and writes the quantized mirror of the result to the request's Q slot (and to its track slot on the interval
  crossing), so Q is always current for the radix cache.
* Conv working copy Cw [layers, R, 3, dim] (bf16): the fused verify kernel indexes conv and SSM state with one index
  vector, so it reads a per-request copy of the conv window gathered from the pool once per verify step.
* ReplaySSM ring (raw v, k, gate, beta per draft step): indexed by the request row too, so it is sized by requests
  (R rows) instead of by every mamba slot (stock KDA sizes it per slot: 20.4 MB/slot/rank on K3).

Error model: the running recurrence never sees quantization. A quantized state is read back only when a prefill
chunk starts from the pool (a radix-cache hit, or the next chunk of a chunked prefill), so the error enters once per
turn / prefill chunk, not per token (see research/k3state.md for the drift measurements).

Paths (hooked in kda_backend.KDAAttnBackend and kda_replayssm_spec_decode):
  forward_extend  prefill: dequant Q[slot] -> W[row], the stock extend kernel on (W, row), quant W[row] -> Q[slot];
                  prefix-cache track snapshots are quantized into Q directly.
  target verify   the stock _forward_target_verify (fused Kimi-K3/DSpARK CuTe kernel or the Triton fallbacks) run
                  against a proxy layer cache (conv=Cw, temporal=W, rings) with the per-row index vector.
  commit          k3state_quant.commit_fold_all_layers + the stock conv rollback.
  forward_decode  non-speculative decode (not used by the DSPARK recipe) through the same proxy, then conv and state
                  written back to the pool and the stock track copy run on the pool.
Requirements (checked at pool build): --enable-linear-replayssm-spec, no page-major envelope / unified memory pool.
"""
from __future__ import annotations

import dataclasses
import logging
import os

import torch

try:  # installed next to this module by the patcher
    from sglang.srt.layers.attention import numinous_k3state_quant as kq
except ImportError:  # repo checkout (tests)
    import k3state_quant as kq  # type: ignore[no-redef]

logger = logging.getLogger(__name__)
_POOLS: dict[int, object] = {}   # id(pool.mamba_cache) -> MambaPool, for the commit entry point
WIDX_CAP = 8192                  # request rows per verify batch (padded CUDA-graph batch sizes included)


def mode():
    m = os.environ.get("NUMINOUS_K3STATE", "off").strip().lower()
    if m in ("", "0", "off", "none"):
        return None
    if m not in kq.FORMATS:
        raise ValueError(f"NUMINOUS_K3STATE={m!r}: expected off, fp8, fp4 or bf16")
    return m


# ------------------------------------------------------------------------------------------------ sizing
def _conv_bytes(cache_params) -> int:
    import numpy as np
    return int(sum(np.prod(s) for s in cache_params.shape.conv)) * cache_params.dtype.conv.itemsize


def slot_bytes(cache_params) -> int:
    """Per mamba slot, all KDA layers: conv window (unchanged) + quantized state."""
    hv, v, k = cache_params.shape.temporal
    return (_conv_bytes(cache_params) + hv * v * kq.row_bytes(mode(), k)) * len(cache_params.layers)


def request_bytes(cache_params) -> int:
    """Per request row, all KDA layers: fp32 working state + conv working copy (the ring is added by the caller)."""
    hv, v, k = cache_params.shape.temporal
    return (hv * v * k * 4 + _conv_bytes(cache_params)) * len(cache_params.layers)


def ring_rows(spec_state_size: int) -> int:
    return spec_state_size + 2


# ------------------------------------------------------------------------------------------------ pool
def pool_temporal(num_layers: int, size: int, temporal_shape, device) -> torch.Tensor:
    hv, v, k = temporal_shape
    return torch.zeros((num_layers, size + 1, hv, v, kq.row_bytes(mode(), k)), dtype=torch.uint8, device=device)


def attach(pool, *, num_layers: int, mamba_layer_ids, spec_state_size: int, temporal_shape, device,
           replayssm_spec: bool, envelope: bool, is_kda: bool) -> None:
    fmt = mode()
    if not is_kda:
        raise RuntimeError("NUMINOUS_K3STATE applies to KDA (Kimi linear) models only")
    if not replayssm_spec:
        raise RuntimeError("NUMINOUS_K3STATE needs --enable-linear-replayssm-spec (the verify/commit path it hooks)")
    if envelope:
        raise RuntimeError("NUMINOUS_K3STATE does not support the page-major envelope mamba layout")
    if list(mamba_layer_ids) != sorted(mamba_layer_ids):
        raise RuntimeError("NUMINOUS_K3STATE expects ascending mamba layer ids")
    hv, v, k = temporal_shape
    rows = ring_rows(spec_state_size)
    conv = pool.mamba_cache.conv
    pool.k3s_fmt = fmt
    pool.k3s_rows = rows
    pool.k3s_w = torch.zeros((num_layers, rows, hv, v, k), dtype=torch.float32, device=device)
    pool.k3s_cw = [torch.zeros((num_layers, rows) + tuple(c.shape[2:]), dtype=c.dtype, device=device) for c in conv]
    pool.k3s_widx = torch.full((WIDX_CAP,), -1, dtype=torch.int32, device=device)
    # slot -> request row of the request that last verified in that slot (the commit's lookup; a slot belongs to one
    # running request, so the map stays right even if another batch verifies between a verify and its commit).
    # Last entry: the discard target for padded (-1) rows.
    n_slots = pool.mamba_cache.temporal.shape[1]
    pool.k3s_slot2row = torch.full((n_slots + 1,), -1, dtype=torch.int32, device=device)
    _POOLS[id(pool.mamba_cache)] = pool
    gb = 1 << 30
    pool.mem_usage += (pool.k3s_w.numel() * 4 + sum(c.numel() * c.element_size() for c in pool.k3s_cw)) / gb
    logger.info(
        "numinous k3state: KDA state %s in the pool (%d slots, %.2f GB, %d B/row), fp32 working state %d rows "
        "(%.2f GB), conv working copy %.3f GB, ReplaySSM ring by request row", fmt,
        pool.mamba_cache.temporal.shape[1], pool.mamba_cache.temporal.numel() / gb, kq.row_bytes(fmt, k), rows,
        pool.k3s_w.numel() * 4 / gb, sum(c.numel() * c.element_size() for c in pool.k3s_cw) / gb)


def _pool_of(backend):
    return backend.req_to_token_pool.mamba_pool


def _slot_or_discard(pool, slots: torch.Tensor) -> torch.Tensor:
    discard = pool.k3s_slot2row.shape[0] - 1
    return torch.where(slots >= 0, slots.long(), torch.full_like(slots, discard, dtype=torch.long))


def _rows_for(slots: torch.Tensor, req_pool_indices: torch.Tensor) -> torch.Tensor:
    return torch.where(slots >= 0, req_pool_indices[: slots.shape[0]].to(torch.int32),
                       torch.full_like(slots, -1, dtype=torch.int32))


class _ProxyReqPool:
    """Forwards everything to the real HybridReqToTokenPool except the per-layer state view."""

    def __init__(self, real, pool):
        self._real, self._pool = real, pool

    def mamba2_layer_cache(self, layer_id: int):
        li = self._real.mamba2_layer_index(layer_id)
        lc = self._real.mamba2_layer_cache(layer_id)
        return dataclasses.replace(lc, conv=[c[li] for c in self._pool.k3s_cw], temporal=self._pool.k3s_w[li])

    def __getattr__(self, name):
        return getattr(self._real, name)


def _run_proxied(backend, rows: torch.Tensor, fn, *args, **kwargs):
    fm = backend.forward_metadata
    real, orig = backend.req_to_token_pool, fm.mamba_cache_indices
    backend.req_to_token_pool = _ProxyReqPool(real, real.mamba_pool)
    fm.mamba_cache_indices = rows
    backend._k3s_inside = True   # the hooked forward_decode falls through to the stock body
    try:
        return fn(*args, **kwargs)
    finally:
        backend._k3s_inside = False
        backend.req_to_token_pool = real
        fm.mamba_cache_indices = orig


# ------------------------------------------------------------------------------------------------ verify
def forward_target_verify(backend, stock_verify, layer, forward_batch, mixed_qkv, a, b):
    rtp = backend.req_to_token_pool
    pool = rtp.mamba_pool
    li = rtp.mamba2_layer_index(layer.layer_id)
    slots = backend.forward_metadata.mamba_cache_indices
    bs = slots.shape[0]
    if bs > WIDX_CAP:
        raise RuntimeError(f"k3state: verify batch {bs} > {WIDX_CAP}")
    rows = pool.k3s_widx[:bs]
    if li == 0:
        # once per verify step, for every KDA layer: the request rows and their conv windows (read-only in verify)
        rows.copy_(_rows_for(slots, forward_batch.req_pool_indices))
        pool.k3s_slot2row.index_copy_(0, _slot_or_discard(pool, slots), rows)
        for conv, cw in zip(pool.mamba_cache.conv, pool.k3s_cw):
            kq.copy_rows_all_layers(conv, slots, cw, rows)
    return _run_proxied(backend, rows, stock_verify, backend, layer, forward_batch, mixed_qkv, a, b)


# ------------------------------------------------------------------------------------------------ commit
def commit(*, spec_state, state_batch_indices, accept_lens, last_correct_step_indices, mamba_track_indices=None,
           mamba_steps_to_track=None, null_block_id: int = -1) -> None:
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_conv_window_scatter_with_mask

    pool = _POOLS[id(spec_state)]
    slots = state_batch_indices.to(torch.int32)
    rows = torch.where(slots > null_block_id, pool.k3s_slot2row[_slot_or_discard(pool, slots)],
                       torch.full_like(slots, -1))
    kq.commit_fold_all_layers(
        w_state=pool.k3s_w, q_pool=spec_state.temporal, rawv=spec_state.replayssm_rawv, rawk=spec_state.replayssm_rawk,
        gk=spec_state.replayssm_g, beta=spec_state.replayssm_beta, w_indices=rows, slot_indices=slots,
        accept_lens=accept_lens, fmt=pool.k3s_fmt, num_k_heads=spec_state.replayssm_rawk.shape[2],
        track_indices=mamba_track_indices, steps_to_track=mamba_steps_to_track)
    # conv rollback + track conv snapshot: verbatim from commit_kda_replayssm_after_verify
    for conv_states, interm_conv in zip(spec_state.conv, spec_state.intermediate_conv_window):
        fused_conv_window_scatter_with_mask(conv_states, interm_conv, state_batch_indices, last_correct_step_indices)
        if mamba_track_indices is not None and mamba_steps_to_track is not None:
            fused_conv_window_scatter_with_mask(conv_states, interm_conv, mamba_track_indices, mamba_steps_to_track)


# ------------------------------------------------------------------------------------------------ extend (prefill)
def forward_extend(backend, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    """Stock KDAAttnBackend.forward_extend (v0.5.20 / main 172b) with the SSM state on (W, request row) and the pool mirrored."""
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    if forward_batch.forward_mode.is_target_verify():
        return forward_target_verify(backend, type(backend)._forward_target_verify, layer, forward_batch,
                                     mixed_qkv, a, b)
    if forward_batch.forward_mode.is_draft_extend_v2():  # noqa: SIM102
        raise RuntimeError("k3state: draft_extend_v2 on the KDA target is not supported (DSPARK does not use it)")
    fm = backend.forward_metadata
    query_start_loc = fm.query_start_loc
    slots = fm.mamba_cache_indices
    rtp = backend.req_to_token_pool
    pool = rtp.mamba_pool
    li = rtp.mamba2_layer_index(layer.layer_id)
    mamba_cache_params = rtp.mamba2_layer_cache(layer.layer_id)
    conv_states = mamba_cache_params.conv[0].transpose(-1, -2)
    q_pool = mamba_cache_params.temporal
    w_state = pool.k3s_w[li]
    rows = _rows_for(slots, forward_batch.req_pool_indices)
    # the initial state always comes from the pool: a radix-cache hit was copied into the slot by the scheduler
    kq.dequant_rows(q_pool, slots, w_state, rows, pool.k3s_fmt)

    if forward_batch.extend_prefix_lens is None:
        raise RuntimeError("extend_prefix_lens cannot be None in non-TARGET_VERIFY mode.")
    has_initial_state = forward_batch.extend_prefix_lens > 0
    physical_num_tokens = mixed_qkv.shape[0]
    # main (172b1b4825, the ROCm image) carries the logical count on the metadata; v0.5.20 derives it
    logical_num_tokens = getattr(fm, "logical_num_tokens", None)
    if logical_num_tokens is None:
        logical_num_tokens = int(query_start_loc[-1])
    if logical_num_tokens < physical_num_tokens:
        mixed_qkv = mixed_qkv[:logical_num_tokens]
        a = a[:, :logical_num_tokens]
        b = b[:, :logical_num_tokens]
    if fm.has_mamba_track_mask:
        mamba_cache_params.conv[0][fm.conv_states_mask_indices] = mixed_qkv[fm.track_conv_indices]
    qkv = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1), layer.conv_weights, layer.bias, activation="silu", conv_states=conv_states,
        has_initial_state=has_initial_state, cache_indices=slots, query_start_loc=query_start_loc,
        seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
    ).transpose(0, 1)
    q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
    q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
    k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
    v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
    gate_was_flat = a.ndim == 3
    if gate_was_flat:
        a = a.unflatten(-1, (-1, layer.head_k_dim))

    track_ssm = fm.has_mamba_track_mask
    track_chunk_idx = fm.track_chunk_idx
    h_track_buf = None
    if track_ssm and track_chunk_idx is not None and fm.track_ssm_h_src.numel() > 0:
        extend_kernel = backend.kernel_dispatcher.effective_extend_kernel(layer.lower_bound)
        assert extend_kernel.supports_track_state_snapshot, (
            f"{type(extend_kernel).__name__} cannot write the fp32 track snapshot required by the mamba track path")
        h_track_buf = torch.empty((track_chunk_idx.shape[0], *w_state.shape[1:]), dtype=torch.float32,
                                  device=w_state.device)
    core_attn_out = backend.kernel_dispatcher.extend(
        q=q, k=k, v=v, g=a, beta=b, ssm_states=w_state, cache_indices=rows, query_start_loc=query_start_loc,
        A_log=layer.A_log, dt_bias=layer.dt_bias, lower_bound=layer.lower_bound, beta_is_raw=gate_was_flat,
        extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu, is_spec_decode=False,
        return_intermediate_states=track_ssm,
        track_ssm_h_src=(fm.track_ssm_h_src if track_ssm else None), track_state=h_track_buf,
        track_chunk_idx=(track_chunk_idx if h_track_buf is not None else None),
    )
    # the exact final state stays in W for the verify steps; the pool slot gets its quantized mirror
    kq.quant_rows(w_state, rows, q_pool, slots, pool.k3s_fmt)
    if track_ssm:
        core_attn_out, h = core_attn_out
        _track_extend(fm, h, q_pool, h_track_buf, pool.k3s_fmt)

    if logical_num_tokens < physical_num_tokens:
        pad = core_attn_out.new_zeros((1, physical_num_tokens - logical_num_tokens) + tuple(core_attn_out.shape[2:]))
        core_attn_out = torch.cat((core_attn_out, pad), dim=1)
    if getattr(backend, "accept_lens_pool", None) is not None:
        raise RuntimeError("k3state: the fused-accept verify path is not supported (it is off under ReplaySSM)")
    return core_attn_out


def _track_extend(fm, h, q_pool, h_track_buf, fmt) -> None:
    """MambaAttnBackendBase._track_mamba_state_extend with quantizing stores into the pool."""
    if not fm.has_mamba_track_mask:
        return
    if fm.track_ssm_h_src.numel() > 0:
        if h_track_buf is not None:
            kq.quant_rows(h_track_buf, fm.track_ssm_h_batch_src, q_pool, fm.track_ssm_h_dst, fmt)
        else:
            assert h is not None
            hh = h.squeeze(0)
            if not hh.is_contiguous():
                hh = hh.contiguous()
            kq.quant_rows(hh, fm.track_ssm_h_src, q_pool, fm.track_ssm_h_dst, fmt)
    if fm.track_ssm_recompute_dst is not None and fm.track_ssm_recompute_dst.numel() > 0:
        raise RuntimeError("k3state: KDA extend has no recompute track states")
    if fm.track_ssm_final_src.numel() > 0:
        q_pool[fm.track_ssm_final_dst] = q_pool[fm.track_ssm_final_src]


# ------------------------------------------------------------------------------------------------ decode
def forward_decode(backend, stock_decode, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    """Non-speculative decode: the stock kernels on (Cw, W, request row), then conv + quantized state written back
    to the pool slots and the stock track copy run on the pool."""
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import track_mamba_states_if_needed

    rtp = backend.req_to_token_pool
    pool = rtp.mamba_pool
    li = rtp.mamba2_layer_index(layer.layer_id)
    fm = backend.forward_metadata
    slots = fm.mamba_cache_indices
    bs = slots.shape[0]
    rows = pool.k3s_widx[:bs]
    if li == 0:
        rows.copy_(_rows_for(slots, forward_batch.req_pool_indices))
        for conv, cw in zip(pool.mamba_cache.conv, pool.k3s_cw):
            kq.copy_rows_all_layers(conv, slots, cw, rows)
    track_mask = forward_batch.mamba_track_mask
    forward_batch.mamba_track_mask = None   # the stock track copy would index the pool with request rows
    try:
        out = _run_proxied(backend, rows, stock_decode, backend, layer, forward_batch, mixed_qkv, a, b, **kwargs)
    finally:
        forward_batch.mamba_track_mask = track_mask
    lc = rtp.mamba2_layer_cache(layer.layer_id)
    for c, cw in zip(lc.conv, pool.k3s_cw):
        kq.copy_rows_all_layers(cw[li:li + 1], rows, c.unsqueeze(0), slots)
    kq.quant_rows(pool.k3s_w[li], rows, lc.temporal, slots, pool.k3s_fmt)
    if track_mask is not None:
        track_mamba_states_if_needed(lc.conv[0], lc.temporal, slots, track_mask, fm.mamba_track_indices, bs)
    return out
