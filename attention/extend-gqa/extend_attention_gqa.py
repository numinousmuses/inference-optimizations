"""Extend (prefill-over-cache) attention with grouped-query packing, for large heads on small-shared-memory GPUs.

Why. On an RTX PRO 6000 the shipped Gemma 4 recipe spends 54% of GPU time in SGLang's Triton extend kernel at
64k-token sessions (research/gpu-evidence/profile). That kernel runs one program per query head, so for Gemma's
global layers (16 query heads on 2 KV heads, head_dim 512) every KV tile is streamed once per query head, 8 times
per KV head, and on SM120 it tiles 512-wide heads at 32x32 with no software pipelining (num_stages=1). Measured:
75 TFLOPS against the card's 383 bf16 / 674 fp8 GEMM TFLOPS (research/gpu-evidence/dc_arch/rtxattn_run_*).

What this does differently.
  * One program per (request, KV head, block of BLOCK_T query tokens). Its rows are the G query heads that share
    the KV head times the BLOCK_T tokens, so a KV tile is read once for all G heads (GQA packing).
  * Q.K^T is accumulated over head-dimension chunks of DK, so a 512-wide head never needs a 512-wide K tile.
  * The P.V accumulator can be split across programs by DV (for 512-wide values the full accumulator does not fit
    a CTA's registers at useful row counts); each DV slice recomputes the scores, trading a little tensor-core
    work for the register room that lets BLOCK_M grow.
  * The cached prefix is read from the paged pool (fp8 or bf16, dequantized in registers with the layer's scale);
    the new tokens' K/V come from the contiguous extend tensors in bf16, exactly as the stock kernel does.
  * Masking matches SGLang's extend semantics: every new token sees the whole prefix (within the sliding window
    when one is set) and the new tokens causally. Sliding window: a query at position q sees keys kv with
    q - kv <= sliding_window_size, the stock Triton extend rule (`q_id <= kv_id + SLIDING_WINDOW_SIZE`). Gemma 4
    passes sliding_window_size = config.sliding_window - 1 = 1023, so each query sees 1024 keys including itself,
    as in HF (`q - kv < sliding_window`). Before 2026-09-28 this kernel used `<`, i.e. 1023 keys (one fewer).

Interface mirrors sglang.kernels.ops.attention.extend_attention.extend_attention_fwd for the paths Gemma 4 takes
(no custom mask, no sinks, no logit cap, page_size 1), so it can be dispatched from there for head_dim >= 256.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _extend_gqa_kernel(
    Q, K_ext, V_ext, O, K_buf, V_buf, Q8, QS,
    qo_indptr, kv_indptr, kv_indices,
    sm_scale, k_scale, v_scale,
    stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh, stride_ot, stride_oh,
    stride_kbt, stride_kbh, stride_vbt, stride_vbh,
    stride_q8t, stride_q8h, stride_qst, stride_qsh,
    num_t_blocks,
    G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
    D: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, NUM_DV: tl.constexpr,
    WINDOW: tl.constexpr, DOT_F32: tl.constexpr, FP8: tl.constexpr,
):
    # DOT_F32: Triton's CPU interpreter multiplies bf16 dot operands as their raw uint16 storage, so the CPU tests
    # run the dots in float32; on a GPU this is False and the dots are bf16 tensor-core MMAs
    req = tl.program_id(0)
    kvh = tl.program_id(1)
    pid2 = tl.program_id(2)
    tblk = pid2 // NUM_DV
    dvb = pid2 % NUM_DV

    q_start = tl.load(qo_indptr + req)
    q_len = tl.load(qo_indptr + req + 1) - q_start
    if tblk * BLOCK_T >= q_len:
        return
    p_start = tl.load(kv_indptr + req)
    p_len = tl.load(kv_indptr + req + 1) - p_start

    # rows = BLOCK_T tokens x the group's query heads, the group padded to a power of two (Qwen 3.8 puts 6 query
    # heads on each KV head); padding rows are masked everywhere
    BLOCK_M: tl.constexpr = BLOCK_T * G_PAD
    rows = tl.arange(0, BLOCK_M)
    t_local = tblk * BLOCK_T + rows // G_PAD      # new-token index within this request
    g = rows % G_PAD
    head = kvh * G + tl.minimum(g, G - 1)          # query head of each row (padding rows alias a real head)
    row_ok = (t_local < q_len) & (g < G)
    q_pos = p_len + t_local                        # absolute position of each row's token

    offs_dk = tl.arange(0, DK)
    offs_dv = dvb * DV + tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, DV], tl.float32)
    qk_scale = sm_scale * 1.44269504  # exp2

    q_row_ptr = Q + (q_start + t_local)[:, None] * stride_qt + head[:, None] * stride_qh
    if FP8:
        # Q arrives quantized (the wrapper scales each query row by its own absolute max over the full head onto
        # e4m3's range), so Q.K^T runs on fp8 tensor cores against the cache's fp8 keys with no conversion inside
        # the key loop
        # Q8 and its scales are fresh contiguous tensors with their own strides; Q itself is usually a view into
        # the fused QKV projection, so its strides must not be reused for them
        q8_row_ptr = Q8 + (q_start + t_local)[:, None] * stride_q8t + head[:, None] * stride_q8h
        q_scale = tl.load(QS + (q_start + t_local) * stride_qst + head * stride_qsh, mask=row_ok, other=1.0)

    # ---- stage 1: the cached prefix, read through kv_indices from the pool ----
    lo = 0
    if WINDOW > 0:
        # keys older than every row's window are skipped outright (a row at q_pos sees keys >= q_pos - WINDOW)
        lo = tl.maximum(p_len + tblk * BLOCK_T - WINDOW, 0)
        lo = (lo // BLOCK_N) * BLOCK_N
    for n0 in range(lo, p_len, BLOCK_N):
        n_idx = n0 + offs_n
        n_ok = n_idx < p_len
        slot = tl.load(kv_indices + p_start + n_idx, mask=n_ok, other=0).to(tl.int64)
        s = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for d0 in range(0, D, DK):
            if not FP8:
                q = tl.load(q_row_ptr + (d0 + offs_dk)[None, :], mask=row_ok[:, None], other=0.0)
            if FP8:
                k8 = tl.load(K_buf + slot[None, :] * stride_kbt + kvh * stride_kbh + (d0 + offs_dk)[:, None],
                             mask=n_ok[None, :], other=0.0)
                q8 = tl.load(q8_row_ptr + (d0 + offs_dk)[None, :], mask=row_ok[:, None], other=0.0)
                s += tl.dot(q8, k8)
            else:
                k = tl.load(K_buf + slot[None, :] * stride_kbt + kvh * stride_kbh + (d0 + offs_dk)[:, None],
                            mask=n_ok[None, :], other=0.0).to(tl.bfloat16)
                if DOT_F32:
                    s += tl.dot(q.to(tl.float32), k.to(tl.float32))
                else:
                    s += tl.dot(q, k)
        if FP8:
            s = s * (q_scale[:, None] * (k_scale * qk_scale))
        else:
            s = s * (k_scale * qk_scale)
        valid = n_ok[None, :] & row_ok[:, None]
        if WINDOW > 0:
            valid = valid & ((q_pos[:, None] - n_idx[None, :]) <= WINDOW)
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        if FP8:
            v8 = tl.load(V_buf + slot[:, None] * stride_vbt + kvh * stride_vbh + offs_dv[None, :],
                         mask=n_ok[:, None], other=0.0)
            pv = tl.dot((p * 448.0).to(tl.float8e4nv), v8) * (1.0 / 448.0)
        else:
            v = tl.load(V_buf + slot[:, None] * stride_vbt + kvh * stride_vbh + offs_dv[None, :],
                        mask=n_ok[:, None], other=0.0).to(tl.bfloat16)
            if DOT_F32:
                pv = tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))
            else:
                pv = tl.dot(p.to(tl.bfloat16), v)
        acc = acc * alpha[:, None] + pv * v_scale
        m_i = m_new

    # ---- stage 2: the new tokens themselves, causal, from the contiguous extend tensors ----
    hi = tl.minimum(q_len, (tblk + 1) * BLOCK_T)
    for n0 in range(0, hi, BLOCK_N):
        n_idx = n0 + offs_n
        n_ok = n_idx < q_len
        s = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for d0 in range(0, D, DK):
            q = tl.load(q_row_ptr + (d0 + offs_dk)[None, :], mask=row_ok[:, None], other=0.0)
            k = tl.load(K_ext + (q_start + n_idx)[None, :] * stride_kt + kvh * stride_kh + (d0 + offs_dk)[:, None],
                        mask=n_ok[None, :], other=0.0)
            if DOT_F32:
                s += tl.dot(q.to(tl.float32), k.to(tl.float32))
            else:
                s += tl.dot(q, k)
        s = s * qk_scale
        valid = n_ok[None, :] & row_ok[:, None] & (n_idx[None, :] <= t_local[:, None])
        if WINDOW > 0:
            valid = valid & ((t_local[:, None] - n_idx[None, :]) <= WINDOW)
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V_ext + (q_start + n_idx)[:, None] * stride_vt + kvh * stride_vh + offs_dv[None, :],
                    mask=n_ok[:, None], other=0.0)
        if DOT_F32:
            pv = tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))
        else:
            pv = tl.dot(p.to(tl.bfloat16), v)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    o_ptr = O + (q_start + t_local)[:, None] * stride_ot + head[:, None] * stride_oh + offs_dv[None, :]
    tl.store(o_ptr, out.to(O.dtype.element_ty), mask=row_ok[:, None])


def extend_attention_gqa(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr, kv_indptr,
                         kv_indices, max_len_extend: int, sm_scale: float, k_scale: float = 1.0,
                         v_scale: float = 1.0, sliding_window_size: int = -1, *, block_t: int | None = None,
                         block_n: int = 32, dk: int = 64, dv: int | None = None, num_warps: int = 4,
                         num_stages: int = 2, fp8: bool | None = None):
    """q/o [tokens, H, D] bf16; k/v_extend [tokens, KVH, D] bf16; k/v_buffer [slots, KVH, D] (fp8 or bf16)."""
    T, H, D = q_extend.shape
    KVH = k_extend.shape[1]
    G = H // KVH
    Dv = v_extend.shape[-1]
    if dv is None:
        dv = Dv if Dv <= 256 else 256
    g_pad = triton.next_power_of_2(G)
    if block_t is None:
        block_t = max(1, 32 // g_pad)
    batch = qo_indptr.shape[0] - 1
    num_dv = Dv // dv
    t_blocks = triton.cdiv(max_len_extend, block_t)
    if fp8 is None:
        # fp8 tensor-core path when the cache is fp8 (as the shipped Gemma recipe runs it)
        fp8 = k_buffer.dtype == torch.float8_e4m3fn and os.environ.get("TRITON_INTERPRET") != "1"
    if fp8:
        q_scale = (q_extend.abs().amax(-1).float().clamp_min(1e-6) / 448.0).contiguous()   # [tokens, heads]
        q8 = (q_extend.float() / q_scale[..., None]).to(torch.float8_e4m3fn).contiguous()
    else:
        q_scale = torch.empty(1, 1, device=q_extend.device)   # unused on the bf16 path
        q8 = q_extend
    grid = (batch, KVH, t_blocks * num_dv)
    _extend_gqa_kernel[grid](
        q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, q8, q_scale,
        qo_indptr, kv_indptr, kv_indices,
        sm_scale, k_scale, v_scale,
        q_extend.stride(0), q_extend.stride(1), k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1), o_extend.stride(0), o_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1), v_buffer.stride(0), v_buffer.stride(1),
        q8.stride(0), q8.stride(1), q_scale.stride(0), q_scale.stride(1),
        t_blocks,
        G=G, G_PAD=g_pad, BLOCK_T=block_t, BLOCK_N=block_n, D=D, DK=dk, DV=dv, NUM_DV=num_dv,
        WINDOW=sliding_window_size if sliding_window_size and sliding_window_size > 0 else 0,
        DOT_F32=os.environ.get("TRITON_INTERPRET") == "1", FP8=bool(fp8),
        num_warps=num_warps, num_stages=num_stages,
    )
    return o_extend


def reference(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, sm_scale,
              k_scale: float = 1.0, v_scale: float = 1.0, sliding_window_size: int = -1) -> torch.Tensor:
    """Dense float32 attention with SGLang's extend mask, for tests."""
    T, H, D = q_extend.shape
    KVH = k_extend.shape[1]
    G = H // KVH
    out = torch.empty(T, H, v_extend.shape[-1], dtype=torch.float32, device=q_extend.device)
    for r in range(qo_indptr.shape[0] - 1):
        qs, qe = int(qo_indptr[r]), int(qo_indptr[r + 1])
        ps, pe = int(kv_indptr[r]), int(kv_indptr[r + 1])
        P, Tn = pe - ps, qe - qs
        idx = kv_indices[ps:pe].long()
        k = torch.cat([k_buffer[idx].float() * k_scale, k_extend[qs:qe].float()])
        v = torch.cat([v_buffer[idx].float() * v_scale, v_extend[qs:qe].float()])
        qpos = P + torch.arange(Tn, device=q_extend.device)
        kpos = torch.arange(P + Tn, device=q_extend.device)
        mask = kpos[None, :] <= qpos[:, None]
        if sliding_window_size and sliding_window_size > 0:
            mask &= (qpos[:, None] - kpos[None, :]) <= sliding_window_size
        for h in range(H):
            s = (q_extend[qs:qe, h].float() @ k[:, h // G].T) * sm_scale
            s = s.masked_fill(~mask, float("-inf"))
            out[qs:qe, h] = torch.softmax(s, -1) @ v[:, h // G]
    return out
