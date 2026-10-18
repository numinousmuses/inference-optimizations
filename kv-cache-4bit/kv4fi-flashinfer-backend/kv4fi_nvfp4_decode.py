"""Native NVFP4-KV attention for decode and speculative verify (Triton), page size 1 token indices.

Reads SGLang's `nvfp4` KV pool directly: packed E2M1 nibbles `[tokens, kv_heads, D/2]` (uint8, low nibble = even
element, high nibble = odd element, FlashInfer `nvfp4_kv_quantize` layout), per-16 FP8 E4M3 block scales
`[tokens, kv_heads, D/16]` (uint8 view) and one FP32 global scale per layer for K and for V
(value = E2M1[nibble] * e4m3(scale) * global). No dequant workspace, no extra pass: FP4 bytes are decoded in
registers, K/V tiles are formed as exact bf16 (a 2-significant-bit E2M1 times a 4-significant-bit E4M3 fits in
bf16's 8 bits), the global scales are folded into the softmax scale (K) and the epilogue (V). Q/K products are
computed as two half-width dots over the even and odd elements, so no nibble interleave is needed.

Interface mirrors what the FlashInfer wrappers plan, so the flashinfer backend can call it with the exact same
metadata FlashInfer would use (CUDA-graph safe: only device tensors are read; the split count depends on batch size
and heads only):
  kv_indptr [B+1] int32, kv_indices [total] int32 (token slots), QL query tokens per request laid out as
  q[b*QL + i], optional packed custom mask (FlashInfer `segment_packbits`, little bit order, byte offsets in
  mask_indptr [B+1]); without a mask, query i of a request sees kv positions n <= kv_len - QL + i (causal chain).

Split-KV: stage 1 writes per-split partial outputs and log-sum-exp; stage 2 merges (same structure as SGLang's
`decode_attention.py`, kept self-contained here).

Tests: operations/kernels/test_kv4fi_nvfp4_decode.py (fp32 reference, FlashInfer byte-format cross-check, bench).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

SF_BLOCK = 16  # elements per E4M3 block scale (NVFP4)


# Two packed E2M1 codes (one byte, in the low 8 bits of a 32-bit register) -> two f16 (one 32-bit register).
# Hardware conversion on sm_100a / sm_120a (PTX cvt.rn.f16x2.e2m1x2; the bf16x2 form needs CUDA >= 13.2 ptxas).
_CVT_E2M1X2_F16X2 = tl.constexpr(
    "{\n.reg .b8 fp4_byte;\nmov.b32 {fp4_byte, _, _, _}, $1;\ncvt.rn.f16x2.e2m1x2 $0, fp4_byte;\n}"
)


@triton.jit
def _e2m1x2_to_f16_pair(byte_i32, USE_CVT: tl.constexpr):
    """byte_i32: int32 tensor holding one packed byte -> (f16 of low nibble, f16 of high nibble)."""
    if USE_CVT:
        y = tl.inline_asm_elementwise(
            _CVT_E2M1X2_F16X2, "=r,r", [byte_i32], dtype=tl.int32, is_pure=True, pack=1
        )
        lo = (y & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        hi = (y >> 16).to(tl.int16).to(tl.float16, bitcast=True)
    else:
        lo = _e2m1_to_f16(byte_i32 & 15)
        hi = _e2m1_to_f16(byte_i32 >> 4)
    return lo, hi


@triton.jit
def _e2m1_to_f16(nib):
    # nib: int32 in [0, 16), code (s e1 e0 m). f16 bits: codes 2..7 -> ((n&7) << 9) + (14 << 10) (exponent e+14,
    # mantissa msb m); code 1 -> 0.5 (14 << 10); code 0 -> 0; sign bit 3 -> bit 15.
    mag = nib & 7
    bits = tl.where(mag < 2, (nib & 1) * 14336, (mag << 9) + 14336) | ((nib & 8) << 12)
    return bits.to(tl.int16).to(tl.float16, bitcast=True)


@triton.jit
def _e4m3_to_f32(b):
    # b: uint8 tensor holding an FP8 E4M3 (fn) code. 0x7F/0xFF (NaN) is not produced by the quantizer.
    s = (b >> 7).to(tl.int32)
    e = ((b >> 3) & 15).to(tl.int32)
    m = (b & 7).to(tl.float32)
    mant = tl.where(e == 0, m * 0.125, 1.0 + m * 0.125)
    expo = tl.where(e == 0, -6.0, e.to(tl.float32) - 7.0)
    v = mant * tl.exp2(expo)
    return tl.where(s != 0, -v, v)


@triton.jit
def _kv4fi_stage1(
    Q,
    K,
    KS,
    V,
    VS,
    qo_indptr,
    kv_indptr,
    kv_indices,
    Mask,
    mask_indptr,
    Mid_O,
    Mid_Lse,
    sm_scale_k,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_ksbs,
    stride_ksh,
    stride_vbs,
    stride_vh,
    stride_vsbs,
    stride_vsh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_ob,
    stride_lse_oh,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROW_TILES: tl.constexpr,
    HALF_D: tl.constexpr,
    NSF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    QL_MAX: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    USE_CVT: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    logit_cap: tl.constexpr,
):
    # One program: one request, one kv head, one KV split, BLOCK_M rows = (query token qi, head in group) pairs,
    # so all query tokens of a request (verify drafts, extend chunk) share every KV byte read. Requests may have
    # different query counts (qo_indptr); rows beyond a request's count are masked.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_kv_head = tl.program_id(1) // ROW_TILES
    row_tile = tl.program_id(1) % ROW_TILES
    split_kv_id = tl.program_id(2)

    qo_start = tl.load(qo_indptr + cur_batch).to(tl.int64)
    qo_len = (tl.load(qo_indptr + cur_batch + 1) - qo_start).to(tl.int32)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    row_qi = rows // kv_group_num
    row_head = cur_kv_head * kv_group_num + rows % kv_group_num
    mask_m = (row_qi < qo_len) & (row_head < q_head_num) & (rows < QL_MAX * kv_group_num)
    row_token = qo_start + row_qi.to(tl.int64)

    offs_half = tl.arange(0, HALF_D)
    offs_sf = tl.arange(0, NSF)

    kv_start = tl.load(kv_indptr + cur_batch).to(tl.int64)
    kv_len = (tl.load(kv_indptr + cur_batch + 1) - kv_start).to(tl.int32)
    # causal chain: query i of qo_len sees kv positions < kv_len - qo_len + i + 1 (FlashInfer paged causal)
    if CAUSAL:
        causal_end = kv_len - (qo_len - 1) + row_qi
    else:
        causal_end = kv_len + tl.zeros_like(row_qi)
    if HAS_MASK:
        mask_base = tl.load(mask_indptr + cur_batch).to(tl.int64)
        row_bit = row_qi.to(tl.int64) * kv_len.to(tl.int64)

    kv_len_per_split = tl.cdiv(tl.cdiv(kv_len, NUM_KV_SPLITS), BLOCK_N) * BLOCK_N
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, kv_len)

    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_even = tl.zeros([BLOCK_M, HALF_D], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_M, HALF_D], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        offs_q = row_token[:, None] * stride_qbs + row_head[:, None] * stride_qh + 2 * offs_half[None, :]
        # K/V tiles are exact in f16 (E2M1 x E4M3 has <= 6 significant bits, magnitude <= 2688); q is clamped to
        # the f16 range (|q| < 65504 always holds for normed queries) and rounded to f16.
        q_even = tl.load(Q + offs_q, mask=mask_m[:, None], other=0.0).to(tl.float32)
        q_odd = tl.load(Q + offs_q + 1, mask=mask_m[:, None], other=0.0).to(tl.float32)
        q_even = tl.minimum(tl.maximum(q_even, -65504.0), 65504.0).to(tl.float16)
        q_odd = tl.minimum(tl.maximum(q_odd, -65504.0), 65504.0).to(tl.float16)

        base_k = cur_kv_head * stride_kh + offs_half[None, :]
        base_v = cur_kv_head * stride_vh + offs_half[None, :]
        base_ks = cur_kv_head * stride_ksh + offs_sf[None, :]
        base_vs = cur_kv_head * stride_vsh + offs_sf[None, :]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N, num_stages=NUM_STAGES):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0).to(tl.int64)

            # ---- K: [BLOCK_N, HALF_D] packed bytes -> even/odd f16 tiles (exact), transposed for the dot
            kb = tl.load(K + kv_loc[:, None] * stride_kbs + base_k, mask=mask_n[:, None], other=0).to(tl.int32)
            ks_small = _e4m3_to_f32(
                tl.load(KS + kv_loc[:, None] * stride_ksbs + base_ks, mask=mask_n[:, None], other=0)
            ).to(tl.float16)
            ks = tl.reshape(tl.broadcast_to(ks_small[:, :, None], [BLOCK_N, NSF, 8]), [BLOCK_N, HALF_D])
            k_lo, k_hi = _e2m1x2_to_f16_pair(kb, USE_CVT)
            k_lo = k_lo * ks
            k_hi = k_hi * ks
            qk = tl.dot(q_even, tl.trans(k_lo)) + tl.dot(q_odd, tl.trans(k_hi))
            qk *= sm_scale_k
            if logit_cap > 0:
                qk = logit_cap * (2.0 / (1.0 + tl.exp(-2.0 * qk / logit_cap)) - 1.0)

            valid = mask_m[:, None] & mask_n[None, :]
            if HAS_MASK:
                bitpos = row_bit[:, None] + offs_n[None, :].to(tl.int64)
                mbyte = tl.load(Mask + mask_base + bitpos // 8, mask=valid, other=0)
                mbit = (mbyte.to(tl.int32) >> (bitpos % 8).to(tl.int32)) & 1
                valid = valid & (mbit != 0)
            else:
                valid = valid & (offs_n[None, :] < causal_end[:, None])
            qk = tl.where(valid, qk, float("-inf"))

            # ---- V: [BLOCK_N, HALF_D]
            vb = tl.load(V + kv_loc[:, None] * stride_vbs + base_v, mask=mask_n[:, None], other=0).to(tl.int32)
            vs_small = _e4m3_to_f32(
                tl.load(VS + kv_loc[:, None] * stride_vsbs + base_vs, mask=mask_n[:, None], other=0)
            ).to(tl.float16)
            vs = tl.reshape(tl.broadcast_to(vs_small[:, :, None], [BLOCK_N, NSF, 8]), [BLOCK_N, HALF_D])
            v_lo, v_hi = _e2m1x2_to_f16_pair(vb, USE_CVT)
            v_lo = v_lo * vs
            v_hi = v_hi * vs

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            # a fully masked tile for a row leaves n_e_max at -inf; guard the rescale
            n_e_max_safe = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
            re_scale = tl.exp(e_max - n_e_max_safe)
            p = tl.exp(qk - n_e_max_safe[:, None])
            acc_even *= re_scale[:, None]
            acc_odd *= re_scale[:, None]
            p16 = p.to(tl.float16)
            acc_even += tl.dot(p16, v_lo)
            acc_odd += tl.dot(p16, v_hi)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid = (
            row_token[:, None] * stride_mid_ob
            + row_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + 2 * offs_half[None, :]
        )
        # a split with no visible keys for a row: e_sum 0 -> write 0 and lse -inf (stage 2 skips it)
        safe_sum = tl.where(e_sum > 0, e_sum, 1.0)
        tl.store(Mid_O + offs_mid, acc_even / safe_sum[:, None], mask=mask_m[:, None])
        tl.store(Mid_O + offs_mid + 1, acc_odd / safe_sum[:, None], mask=mask_m[:, None])
        lse = tl.where(e_sum > 0, e_max + tl.log(safe_sum), float("-inf"))
        tl.store(
            Mid_Lse + row_token * stride_lse_ob + row_head * stride_lse_oh + split_kv_id,
            lse,
            mask=mask_m,
        )


@triton.jit
def _kv4fi_stage2(
    Mid_O,
    Mid_Lse,
    O,
    Lse,
    qo_indptr,
    v_scale,
    kv_indptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_ob,
    stride_lse_oh,
    stride_obs,
    stride_oh,
    stride_lse_out,
    bs,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    BS_POW2: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    cur_token = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)
    # request of this token: number of request starts (qo_indptr[1:]) at or before it
    offs_b = tl.arange(0, BS_POW2)
    starts = tl.load(qo_indptr + 1 + offs_b, mask=offs_b < bs, other=2147483647)
    cur_batch = tl.sum((starts <= cur_token).to(tl.int32), 0).to(tl.int64)

    kv_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    kv_len_per_split = tl.cdiv(tl.cdiv(kv_len, NUM_KV_SPLITS), BLOCK_N) * BLOCK_N

    offs_d = tl.arange(0, D)
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([D], dtype=tl.float32)
    offs_v = cur_token * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_l = cur_token * stride_lse_ob + cur_head * stride_lse_oh

    for split_kv_id in tl.range(0, NUM_KV_SPLITS, num_stages=2):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, kv_len)
        if split_kv_end > split_kv_start:
            tlogic = tl.load(Mid_Lse + offs_l + split_kv_id)
            if tlogic > float("-inf"):
                tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os)
                n_e_max = tl.maximum(tlogic, e_max)
                old_scale = tl.exp(e_max - n_e_max)
                acc *= old_scale
                exp_logic = tl.exp(tlogic - n_e_max)
                acc += exp_logic * tv
                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max

    safe_sum = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(O + cur_token * stride_obs + cur_head * stride_oh + offs_d, acc / safe_sum * v_scale)
    if WRITE_LSE:
        # FlashInfer convention: log-sum-exp in base 2 (what merge_state / cascade expect)
        lse2 = tl.where(e_sum > 0, (e_max + tl.log(safe_sum)) * 1.4426950408889634, float("-inf"))
        tl.store(Lse + cur_token * stride_lse_out + cur_head, lse2)


def hardware_e2m1_cvt_available(device=None) -> bool:
    """PTX cvt.rn.f16x2.e2m1x2 exists on sm_100a / sm_120a targets (SM 10.x and 12.x); Triton compiles for the
    arch-specific target there."""
    major = torch.cuda.get_device_capability(device)[0]
    return major in (10, 12)


def choose_num_kv_splits(num_tokens: int, num_kv_heads: int, sm_count: Optional[int] = None) -> int:
    """Split count from batch/head shape only (never from sequence lengths), so a CUDA graph captured for a batch
    size keeps a valid, identical split count at every replay."""
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    programs = max(1, num_tokens * num_kv_heads)
    want = math.ceil(8 * sm_count / programs)
    return int(min(128, max(1, 1 << (want - 1).bit_length())))


def nvfp4_attention(
    q: torch.Tensor,
    k_fp4: torch.Tensor,
    k_sf: torch.Tensor,
    v_fp4: torch.Tensor,
    v_sf: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    k_global_scale: float,
    v_global_scale: float,
    sm_scale: float,
    *,
    ql: int = 1,
    qo_indptr: Optional[torch.Tensor] = None,
    causal: bool = True,
    mask: Optional[torch.Tensor] = None,
    mask_indptr: Optional[torch.Tensor] = None,
    logit_cap: float = 0.0,
    num_kv_splits: Optional[int] = None,
    block_n: Optional[int] = None,
    use_cvt: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    num_warps: Optional[int] = None,
    num_stages: int = 3,
):
    """q [num_tokens, HQ, D] (bf16/fp16); k_fp4/v_fp4 [T, HK, D/2] uint8; k_sf/v_sf [T, HK, D/16] (uint8 or fp8
    view); kv_indptr [B+1], kv_indices [>= kv_indptr[-1]] int32 token slots.
    Query layout: `qo_indptr` [B+1] (device) gives each request's query rows; `ql` is the maximum per request
    (uniform ql -> qo_indptr may be None). `causal` = FlashInfer paged causal (query i of a request's qo_len sees
    kv positions < kv_len - qo_len + i + 1); `mask` = packed bits + byte `mask_indptr` [B+1] (segment_packbits,
    little bit order) overrides causal. Returns out [num_tokens, HQ, D] in q.dtype (and base-2 lse [num_tokens, HQ]
    fp32 when return_lse)."""
    assert q.dim() == 3 and q.is_cuda
    num_tokens, hq, d = q.shape
    bs = kv_indptr.numel() - 1
    if qo_indptr is None:
        assert num_tokens == bs * ql, (num_tokens, bs, ql)
        qo_indptr = torch.arange(0, (bs + 1) * ql, ql, dtype=torch.int32, device=q.device)
    hk = k_fp4.shape[1]
    assert k_fp4.shape[2] * 2 == d and v_fp4.shape[2] * 2 == d, (k_fp4.shape, v_fp4.shape, d)
    assert k_sf.shape[2] * SF_BLOCK == d and v_sf.shape[2] * SF_BLOCK == d, (k_sf.shape, v_sf.shape)
    assert hq % hk == 0 and d % 32 == 0 and (d & (d - 1)) == 0, (hq, hk, d)
    if k_sf.dtype != torch.uint8:
        k_sf = k_sf.view(torch.uint8)
    if v_sf.dtype != torch.uint8:
        v_sf = v_sf.view(torch.uint8)
    if use_cvt is None:
        use_cvt = hardware_e2m1_cvt_available(q.device)
    kv_group_num = hq // hk
    # measured sweeps (research/gpu-evidence/rtx/kv4fi_decode_bench.json and the 99K single-user check in
    # agents/status/kv4fi.md): wide groups (Qwen 3.6, 8 q heads per kv head) want 128-token tiles with 8 warps and
    # at most 64 splits; 6-wide groups (Qwen 3.8) want 32-token tiles for one query row, 64 for verify rows.
    wide = kv_group_num >= 8
    if block_n is None:
        block_n = 128 if wide else (32 if ql == 1 else 64)
    if num_warps is None:
        num_warps = 8 if wide else 4
    rows = ql * kv_group_num  # (query token, head in group) pairs per kv head
    block_m = max(16, min(64, triton.next_power_of_2(rows)))
    row_tiles = triton.cdiv(rows, block_m)
    if num_kv_splits is None:
        num_kv_splits = choose_num_kv_splits(bs * row_tiles, hk)
        if wide or ql > 1:
            num_kv_splits = min(num_kv_splits, 64)
    if out is None:
        out = torch.empty_like(q)
    mid_o = torch.empty((num_tokens, hq, num_kv_splits, d), dtype=torch.float32, device=q.device)
    mid_lse = torch.empty((num_tokens, hq, num_kv_splits), dtype=torch.float32, device=q.device)
    has_mask = mask is not None
    if has_mask:
        assert mask_indptr is not None and mask.dtype == torch.uint8, (mask.dtype,)
    else:
        mask, mask_indptr = kv_indptr, kv_indptr  # never read; valid device pointers
    lse = torch.empty((num_tokens, hq), dtype=torch.float32, device=q.device) if return_lse else mid_lse
    grid1 = (bs, hk * row_tiles, num_kv_splits)
    _kv4fi_stage1[grid1](
        q, k_fp4, k_sf, v_fp4, v_sf, qo_indptr, kv_indptr, kv_indices, mask, mask_indptr, mid_o, mid_lse,
        float(sm_scale) * float(k_global_scale),
        q.stride(0), q.stride(1),
        k_fp4.stride(0), k_fp4.stride(1), k_sf.stride(0), k_sf.stride(1),
        v_fp4.stride(0), v_fp4.stride(1), v_sf.stride(0), v_sf.stride(1),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        kv_group_num=kv_group_num, q_head_num=hq, BLOCK_M=block_m, ROW_TILES=row_tiles, HALF_D=d // 2,
        NSF=d // SF_BLOCK, BLOCK_N=block_n, NUM_KV_SPLITS=num_kv_splits, QL_MAX=ql, CAUSAL=causal,
        HAS_MASK=has_mask, USE_CVT=bool(use_cvt), NUM_STAGES=num_stages, logit_cap=float(logit_cap),
        num_warps=num_warps,
    )
    _kv4fi_stage2[(num_tokens, hq)](
        mid_o, mid_lse, out, lse, qo_indptr, float(v_global_scale), kv_indptr,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        out.stride(0), out.stride(1), lse.stride(0) if return_lse else 0, bs,
        D=d, BLOCK_N=block_n, NUM_KV_SPLITS=num_kv_splits, BS_POW2=triton.next_power_of_2(bs),
        WRITE_LSE=return_lse, num_warps=4,
    )
    if return_lse:
        return out, lse
    return out


# ------------------------------------------------------------------------------------------ torch reference
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_nvfp4_ref(fp4: torch.Tensor, sf: torch.Tensor, global_scale: float) -> torch.Tensor:
    """[..., D/2] uint8 + [..., D/16] scale bytes -> [..., D] fp32 (FlashInfer nvfp4_kv_dequantize semantics)."""
    lut = E2M1_VALUES.to(fp4.device)
    lo = lut[(fp4 & 15).long()]
    hi = lut[(fp4 >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)  # even, odd interleave
    scales = sf.view(torch.float8_e4m3fn).float().repeat_interleave(SF_BLOCK, dim=-1)
    return vals * scales * float(global_scale)


def unpack_mask_ref(mask: torch.Tensor, mask_indptr: torch.Tensor, b: int, ql: int, kv_len: int) -> torch.Tensor:
    """Packed little-bit-order per-request mask -> [ql, kv_len] bool."""
    start = int(mask_indptr[b])
    nbits = ql * kv_len
    bytes_ = mask[start : start + (nbits + 7) // 8].long()
    bits = ((bytes_[:, None] >> torch.arange(8, device=mask.device)) & 1).flatten()[:nbits]
    return bits.view(ql, kv_len).bool()


def nvfp4_attention_ref(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, k_gs, v_gs, sm_scale, *, ql=1,
                        qo_indptr=None, causal=True, mask=None, mask_indptr=None, logit_cap=0.0,
                        return_lse=False):
    num_tokens, hq, d = q.shape
    bs = kv_indptr.numel() - 1
    hk = k_fp4.shape[1]
    g = hq // hk
    if qo_indptr is None:
        qo_indptr = torch.arange(0, (bs + 1) * ql, ql, device=q.device)
    out = torch.empty((num_tokens, hq, d), dtype=torch.float32, device=q.device)
    lse_out = torch.empty((num_tokens, hq), dtype=torch.float32, device=q.device)
    indptr = kv_indptr.tolist()
    qptr = qo_indptr.tolist()
    for b in range(bs):
        idx = kv_indices[indptr[b] : indptr[b + 1]].long()
        kv_len = idx.numel()
        qo_len = qptr[b + 1] - qptr[b]
        k = dequant_nvfp4_ref(k_fp4[idx], k_sf[idx], k_gs)  # [L, HK, D]
        v = dequant_nvfp4_ref(v_fp4[idx], v_sf[idx], v_gs)
        if mask is not None:
            allowed = unpack_mask_ref(mask, mask_indptr, b, qo_len, kv_len)
        elif causal:
            pos = torch.arange(kv_len, device=q.device)
            allowed = pos[None, :] < (kv_len - qo_len + 1 + torch.arange(qo_len, device=q.device))[:, None]
        else:
            allowed = torch.ones((qo_len, kv_len), dtype=torch.bool, device=q.device)
        kk = k.repeat_interleave(g, dim=1)  # [L, HQ, D]
        vv = v.repeat_interleave(g, dim=1)
        for i in range(qo_len):
            qi = q[qptr[b] + i].float()  # [HQ, D]
            s = torch.einsum("hd,lhd->hl", qi, kk) * sm_scale
            if logit_cap > 0:
                s = logit_cap * torch.tanh(s / logit_cap)
            s = s.masked_fill(~allowed[i][None, :], float("-inf"))
            p = torch.softmax(s, dim=-1)
            p = torch.nan_to_num(p)
            out[qptr[b] + i] = torch.einsum("hl,lhd->hd", p, vv)
            lse_out[qptr[b] + i] = torch.logsumexp(s, dim=-1) / math.log(2.0)
    if return_lse:
        return out.to(q.dtype), lse_out
    return out.to(q.dtype)


def pack_mask_ref(allowed_per_req: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """List of [ql, kv_len] bool -> (packed uint8 little bit order, byte indptr [B+1]) like segment_packbits."""
    chunks, indptr = [], [0]
    for m in allowed_per_req:
        bits = m.flatten().to(torch.uint8)
        pad = (-bits.numel()) % 8
        bits = torch.cat([bits, bits.new_zeros(pad)])
        by = (bits.view(-1, 8) << torch.arange(8, device=bits.device, dtype=torch.uint8)).sum(-1, dtype=torch.int32)
        chunks.append(by.to(torch.uint8))
        indptr.append(indptr[-1] + by.numel())
    return torch.cat(chunks), torch.tensor(indptr, dtype=torch.int32, device=chunks[0].device)
