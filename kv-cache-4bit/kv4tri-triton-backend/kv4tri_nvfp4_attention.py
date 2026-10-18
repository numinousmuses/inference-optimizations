"""Native NVFP4-KV attention kernels for SGLang's Triton backend (decode + extend), dequantizing in registers.

Why. SGLang v0.5.20 accepts `--kv-cache-dtype nvfp4 --attention-backend triton` at argument validation, but the
Triton backend has no FP4 read path: it hands the packed pool (viewed as torch.float4_e2m1fn_x2, [slots, H, D/2])
to kernels written for bf16/fp8 and crashes (measured on RTX PRO 6000: `KeyError: 'float4_e2m1fn_x2'`; see
operations/kernels/kv4tri_probe.py). The only upstream NVFP4 consumers are FlashInfer prefill through a full-pool
FP8 dequant workspace and TRT-LLM XQA decode (whose batched decode is known-broken on SM120, sglang #31641).

What this is. Two kernels that read the NVFP4 pool exactly as SGLang's NVFP4KVCacheMethod writes it and never
materialize a dequantized copy:
  * packed data  uint8 [slots, H, D/2]: element 2j in the low nibble of byte j, element 2j+1 in the high nibble,
    each an E2M1 code (sign bit 3, magnitude {0, .5, 1, 1.5, 2, 3, 4, 6});
  * block scales uint8 [slots, H, D/16]: one FP8 E4M3 scale per 16 consecutive elements (linear layout);
  * a per-layer FP32 global scale;  x = e2m1 * block_scale * global_scale.
Decoding uses bit placement instead of lookup tables: an E2M1 magnitude code `eem` shifted to fp16 bits 11..9 is the
fp16 number 2^-14 * value (normal and subnormal cases alike), and an E4M3 byte shifted left by 7 is the fp16 number
2^-8 * value, so one multiply by 2^22 restores both. Dots contract even and odd head-dimension halves separately
(q_even . k_lo + q_odd . k_hi), which avoids interleaving nibbles in registers.

  nvfp4_decode_fwd : split-KV grouped decode (one program per request x KV head x split, all query heads of the
                     group as MMA rows) + a split reduction; same buffers and split rule as SGLang's decode kernel.
  nvfp4_extend_fwd : GQA-packed extend (prefix from the FP4 pool, new tokens from bf16 extend tensors, causal or a
                     custom mask on the new-token part, sliding window with SGLang's `q - kv <= W` rule), derived from
                     operations/kernels/extend_attention_gqa.py.

The torch helpers (`nvfp4_quantize_ref`, `nvfp4_dequantize_ref`, `decode_reference`, `extend_reference`) define the
format for the CPU tests (operations/kernels/test_kv4tri_nvfp4_attention.py); the box test
(operations/kernels/test_kv4tri_nvfp4_gpu.py) checks the kernels against SGLang's own pools and FlashInfer quantizer.
Routing inside SGLang: operations/patches/kv4tri_hooks.py.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SPLIT_ALIGN = 32  # SGLang's _MIN_BLOCK_KV: split lengths are rounded up to this many tokens


def _interp() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


# ---------------------------------------------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _e2m1x2_to_f16(packed, USE_ASM: tl.constexpr):
    """packed uint8 tile -> (low-nibble values, high-nibble values) as float16, unscaled (exact)."""
    if USE_ASM:
        # SM100/SM120 hardware conversion: one cvt per byte (two E2M1 -> f16x2), then split the halves
        lo, hi = tl.inline_asm_elementwise(
            asm="""{
            .reg .b8 b0, b1, b2, b3;
            .reg .b32 r0, r1, r2, r3;
            mov.b32 {b0, b1, b2, b3}, $4;
            cvt.rn.f16x2.e2m1x2 r0, b0;
            cvt.rn.f16x2.e2m1x2 r1, b1;
            cvt.rn.f16x2.e2m1x2 r2, b2;
            cvt.rn.f16x2.e2m1x2 r3, b3;
            prmt.b32 $0, r0, r1, 0x5410;
            prmt.b32 $1, r2, r3, 0x5410;
            prmt.b32 $2, r0, r1, 0x7632;
            prmt.b32 $3, r2, r3, 0x7632;
            }""",
            constraints="=r,=r,=r,=r,r", args=[packed], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    else:
        # bit placement: E2M1 magnitude bits `eem` at fp16 bits 11..9 is 2^-14 * value (normal and subnormal)
        p = packed.to(tl.int32)
        lo = (((p & 0x7) << 9) | ((p & 0x8) << 12)).to(tl.uint16).to(tl.float16, bitcast=True) * 16384.0
        hi = (((p & 0x70) << 5) | ((p & 0x80) << 8)).to(tl.uint16).to(tl.float16, bitcast=True) * 16384.0
    return lo, hi


@triton.jit
def _e4m3_to_f16(sf, USE_ASM: tl.constexpr):
    """uint8 E4M3 bytes -> float16 (exact)."""
    if USE_ASM:
        return sf.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    else:
        s = sf.to(tl.int32)
        # E4M3 byte shifted left by 7 is the fp16 number 2^-8 * value
        return (((s & 0x7F) << 7) | ((s & 0x80) << 8)).to(tl.uint16).to(tl.float16, bitcast=True) * 256.0


@triton.jit
def _fp4_pair(packed, sf):
    """Reference-path helper kept for the bit-exactness test: (lo, hi) float32 with the block scale applied, from a
    packed tile and a same-shape tile of scale bytes."""
    lo, hi = _e2m1x2_to_f16(packed, False)
    s = _e4m3_to_f16(sf, False).to(tl.float32)
    return lo.to(tl.float32) * s, hi.to(tl.float32) * s


@triton.jit
def _fp4_tile_t(packed, sf, HD: tl.constexpr, N: tl.constexpr, USE_ASM: tl.constexpr):
    """Transposed tile: packed uint8 [HD, N] (packed dim first), sf uint8 [HD // 8, N] -> scaled fp16 (lo, hi)."""
    lo, hi = _e2m1x2_to_f16(packed, USE_ASM)
    s = _e4m3_to_f16(sf, USE_ASM)[:, None, :]
    lo = tl.reshape(tl.reshape(lo, [HD // 8, 8, N]) * s, [HD, N])
    hi = tl.reshape(tl.reshape(hi, [HD // 8, 8, N]) * s, [HD, N])
    return lo, hi


@triton.jit
def _fp4_tile_n(packed, sf, N: tl.constexpr, HD: tl.constexpr, USE_ASM: tl.constexpr):
    """Natural tile: packed uint8 [N, HD], sf uint8 [N, HD // 8] -> scaled fp16 (lo, hi)."""
    lo, hi = _e2m1x2_to_f16(packed, USE_ASM)
    s = _e4m3_to_f16(sf, USE_ASM)[:, :, None]
    lo = tl.reshape(tl.reshape(lo, [N, HD // 8, 8]) * s, [N, HD])
    hi = tl.reshape(tl.reshape(hi, [N, HD // 8, 8]) * s, [N, HD])
    return lo, hi


# ---------------------------------------------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _nvfp4_decode_stage1(
    Q, K, V, KS, VS, kv_indptr, kv_indices, Att, Lse, num_kv_splits,
    sm_scale_k, logit_cap,
    stride_qb, stride_qh,
    stride_kt, stride_kh, stride_vt, stride_vh,
    stride_kst, stride_ksh, stride_vst, stride_vsh,
    stride_ab, stride_ah, stride_as, stride_lb, stride_lh,
    G: tl.constexpr, HQ: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    D: tl.constexpr, DV: tl.constexpr, SPLIT_ALIGN: tl.constexpr, LOGIT_CAP: tl.constexpr,
    DOT_F32: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    hblk = tl.program_id(1)
    split = tl.program_id(2)
    TILES: tl.constexpr = (G + BLOCK_H - 1) // BLOCK_H
    kvh = hblk // TILES
    sub = hblk % TILES
    hg = sub * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = hg < G
    head = kvh * G + tl.minimum(hg, G - 1)

    kv_start = tl.load(kv_indptr + b)
    seq_len = tl.load(kv_indptr + b + 1) - kv_start
    nsplit = tl.load(num_kv_splits + b)
    per_split = tl.cdiv(tl.cdiv(seq_len, nsplit), SPLIT_ALIGN) * SPLIT_ALIGN
    s_lo = per_split * split
    s_hi = tl.minimum(s_lo + per_split, seq_len)
    if s_hi <= s_lo:
        return

    HD: tl.constexpr = D // 2
    HDV: tl.constexpr = DV // 2
    j = tl.arange(0, HD)
    jb = tl.arange(0, HD // 8)
    jv = tl.arange(0, HDV)
    jvb = tl.arange(0, HDV // 8)
    q_ptr = Q + b * stride_qb + head[:, None] * stride_qh
    q_e = tl.load(q_ptr + 2 * j[None, :], mask=mask_h[:, None], other=0.0)
    q_o = tl.load(q_ptr + 2 * j[None, :] + 1, mask=mask_h[:, None], other=0.0)
    if DOT_F32:
        q_e = q_e.to(tl.float32)
        q_o = q_o.to(tl.float32)
    else:
        q_e = q_e.to(tl.float16)
        q_o = q_o.to(tl.float16)

    e_max = tl.full([BLOCK_H], float("-inf"), tl.float32)
    e_sum = tl.zeros([BLOCK_H], tl.float32)
    acc_e = tl.zeros([BLOCK_H, HDV], tl.float32)
    acc_o = tl.zeros([BLOCK_H, HDV], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for n0 in range(s_lo, s_hi, BLOCK_N):
        n_idx = n0 + offs_n
        n_ok = n_idx < s_hi
        slot = tl.load(kv_indices + kv_start + n_idx, mask=n_ok, other=0).to(tl.int64)
        kp = tl.load(K + slot[None, :] * stride_kt + kvh * stride_kh + j[:, None], mask=n_ok[None, :], other=0)
        ks = tl.load(KS + slot[None, :] * stride_kst + kvh * stride_ksh + jb[:, None], mask=n_ok[None, :], other=0)
        k_lo, k_hi = _fp4_tile_t(kp, ks, HD, BLOCK_N, not DOT_F32)
        if DOT_F32:
            qk = tl.dot(q_e, k_lo.to(tl.float32)) + tl.dot(q_o, k_hi.to(tl.float32))
        else:
            qk = tl.dot(q_e, k_lo) + tl.dot(q_o, k_hi)
        qk = qk * sm_scale_k
        if LOGIT_CAP:
            qk = logit_cap * (2.0 * tl.sigmoid(2.0 * qk / logit_cap) - 1.0)
        qk = tl.where(mask_h[:, None] & n_ok[None, :], qk, float("-inf"))

        vp = tl.load(V + slot[:, None] * stride_vt + kvh * stride_vh + jv[None, :], mask=n_ok[:, None], other=0)
        vs = tl.load(VS + slot[:, None] * stride_vst + kvh * stride_vsh + jvb[None, :], mask=n_ok[:, None], other=0)
        v_lo, v_hi = _fp4_tile_n(vp, vs, BLOCK_N, HDV, not DOT_F32)

        n_max = tl.maximum(tl.max(qk, 1), e_max)
        re = tl.exp(e_max - n_max)
        p = tl.exp(qk - n_max[:, None])
        acc_e *= re[:, None]
        acc_o *= re[:, None]
        if DOT_F32:
            pb = p.to(tl.float16).to(tl.float32)
            acc_e += tl.dot(pb, v_lo.to(tl.float32))
            acc_o += tl.dot(pb, v_hi.to(tl.float32))
        else:
            pb = p.to(tl.float16)
            acc_e += tl.dot(pb, v_lo)
            acc_o += tl.dot(pb, v_hi)
        e_sum = e_sum * re + tl.sum(p, 1)
        e_max = n_max

    a_ptr = Att + b * stride_ab + head[:, None] * stride_ah + split * stride_as
    tl.store(a_ptr + 2 * jv[None, :], acc_e / e_sum[:, None], mask=mask_h[:, None])
    tl.store(a_ptr + 2 * jv[None, :] + 1, acc_o / e_sum[:, None], mask=mask_h[:, None])
    tl.store(Lse + b * stride_lb + head * stride_lh + split, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _nvfp4_decode_stage2(
    Att, Lse, O, kv_indptr, num_kv_splits, v_scale,
    stride_ab, stride_ah, stride_as, stride_lb, stride_lh, stride_ob, stride_oh,
    DV: tl.constexpr, BLOCK_DV: tl.constexpr, MAX_SPLITS: tl.constexpr, SPLIT_ALIGN: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    seq_len = tl.load(kv_indptr + b + 1) - tl.load(kv_indptr + b)
    nsplit = tl.load(num_kv_splits + b)
    per_split = tl.cdiv(tl.cdiv(seq_len, nsplit), SPLIT_ALIGN) * SPLIT_ALIGN
    offs = tl.arange(0, BLOCK_DV)
    m_d = offs < DV
    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], tl.float32)
    for s in range(0, MAX_SPLITS):
        lo = per_split * s
        hi = tl.minimum(lo + per_split, seq_len)
        if hi > lo:
            tv = tl.load(Att + b * stride_ab + h * stride_ah + s * stride_as + offs, mask=m_d, other=0.0)
            tlse = tl.load(Lse + b * stride_lb + h * stride_lh + s)
            n_max = tl.maximum(tlse, e_max)
            old = tl.exp(e_max - n_max)
            w = tl.exp(tlse - n_max)
            acc = acc * old + w * tv
            e_sum = e_sum * old + w
            e_max = n_max
    e_safe = tl.where(e_sum == 0.0, 1.0, e_sum)
    tl.store(O + b * stride_ob + h * stride_oh + offs, (acc / e_safe * v_scale).to(O.dtype.element_ty), mask=m_d)


def decode_config(D: int) -> dict:
    """Stage-1 geometry, CUDA-graph swept on RTX PRO 6000 (operations/kernels/kv4tri_tune.py, B=16, 104k context):
    D 256 -> 1.30 ms (Qwen 3.8, 24/4 heads) and 0.83 ms (Qwen 3.6, 16/2); D 512 -> 1.54 ms (Gemma global, 16/2);
    stock Triton FP8 on the same shapes 2.28 / 1.46 / 2.40 ms, FlashInfer FP8 2.25 / 1.14 / 2.62 ms (measured)."""
    if D >= 512:
        return dict(block_n=64, num_warps=4, num_stages=1)
    return dict(block_n=32, num_warps=4, num_stages=3)


def nvfp4_decode_fwd(q, k_fp4, v_fp4, k_sf, v_sf, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits,
                     max_kv_splits: int, sm_scale: float, k_scale: float = 1.0, v_scale: float = 1.0,
                     logit_cap: float = 0.0, *, block_n: int | None = None, block_h: int = 16,
                     num_warps: int | None = None, num_stages: int | None = None):
    """q/o [B, HQ, D] / [B, HQ, DV]; k_fp4/v_fp4 uint8 [slots, KVH, D/2]; k_sf/v_sf uint8 (or e4m3 view)
    [slots, KVH, D/16]; attn_logits fp32 [>=B, HQ, >=max_kv_splits, DV]; attn_lse fp32 [>=B, HQ, >=max_kv_splits];
    num_kv_splits int32 [>=B]. k_scale / v_scale are the layer's NVFP4 global scales."""
    B, HQ, D = q.shape
    KVH = k_fp4.shape[1]
    DV = o.shape[-1]
    G = HQ // KVH
    cfg = decode_config(D)
    block_n = block_n or cfg["block_n"]
    num_warps = num_warps or cfg["num_warps"]
    num_stages = num_stages or cfg["num_stages"]
    assert k_fp4.shape[-1] * 2 == D and v_fp4.shape[-1] * 2 == DV and D % 32 == 0 and DV % 32 == 0
    k_sf = k_sf.view(torch.uint8)
    v_sf = v_sf.view(torch.uint8)
    k_fp4 = k_fp4.view(torch.uint8)
    v_fp4 = v_fp4.view(torch.uint8)
    tiles = triton.cdiv(G, block_h)
    grid = (B, KVH * tiles, max_kv_splits)
    _nvfp4_decode_stage1[grid](
        q, k_fp4, v_fp4, k_sf, v_sf, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits,
        float(sm_scale) * float(k_scale), float(logit_cap),
        q.stride(0), q.stride(1),
        k_fp4.stride(0), k_fp4.stride(1), v_fp4.stride(0), v_fp4.stride(1),
        k_sf.stride(0), k_sf.stride(1), v_sf.stride(0), v_sf.stride(1),
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        attn_lse.stride(0), attn_lse.stride(1),
        G=G, HQ=HQ, BLOCK_H=block_h, BLOCK_N=block_n, D=D, DV=DV, SPLIT_ALIGN=SPLIT_ALIGN,
        LOGIT_CAP=bool(logit_cap and logit_cap > 0), DOT_F32=_interp(),
        num_warps=num_warps, num_stages=num_stages,
    )
    _nvfp4_decode_stage2[(B, HQ)](
        attn_logits, attn_lse, o, kv_indptr, num_kv_splits, float(v_scale),
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        attn_lse.stride(0), attn_lse.stride(1), o.stride(0), o.stride(1),
        DV=DV, BLOCK_DV=triton.next_power_of_2(DV), MAX_SPLITS=max_kv_splits, SPLIT_ALIGN=SPLIT_ALIGN,
        num_warps=4, num_stages=2,
    )
    return o


# ---------------------------------------------------------------------------------------------------------------
# extend
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _nvfp4_extend_kernel(
    Q, K_ext, V_ext, O, Kp, Vp, KS, VS, qo_indptr, kv_indptr, kv_indices, Mask, mask_indptr, WOff,
    sm_scale, k_scale, v_scale, logit_cap,
    stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh, stride_ot, stride_oh,
    stride_kbt, stride_kbh, stride_vbt, stride_vbh, stride_kst, stride_ksh, stride_vst, stride_vsh,
    G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
    D: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, NUM_DV: tl.constexpr,
    WINDOW: tl.constexpr, IS_CAUSAL: tl.constexpr, USE_MASK: tl.constexpr, LOGIT_CAP: tl.constexpr,
    DOT_F32: tl.constexpr, HAS_WOFF: tl.constexpr = False,
):
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

    BLOCK_M: tl.constexpr = BLOCK_T * G_PAD
    rows = tl.arange(0, BLOCK_M)
    t_local = tblk * BLOCK_T + rows // G_PAD
    g = rows % G_PAD
    head = kvh * G + tl.minimum(g, G - 1)
    row_ok = (t_local < q_len) & (g < G)
    q_pos = p_len + t_local

    HDK: tl.constexpr = DK // 2
    HDV: tl.constexpr = DV // 2
    jk = tl.arange(0, HDK)
    jkb = tl.arange(0, HDK // 8)
    offs_dk = tl.arange(0, DK)
    jv = dvb * HDV + tl.arange(0, HDV)       # packed value columns of this DV slice
    jvb = dvb * (HDV // 8) + tl.arange(0, HDV // 8)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc_e = tl.zeros([BLOCK_M, HDV], tl.float32)
    acc_o = tl.zeros([BLOCK_M, HDV], tl.float32)
    q_row_ptr = Q + (q_start + t_local)[:, None] * stride_qt + head[:, None] * stride_qh
    if USE_MASK:
        # mask rows span the full (unwindowed) sequence: the windowed prefix starts `woff` tokens in (SGLang's
        # window_kv_offsets), so row length is woff + prefix + q_len and new-token columns start at woff + prefix
        woff = 0
        if HAS_WOFF:
            woff = tl.load(WOff + req).to(tl.int64)
        mask_base = tl.load(mask_indptr + req) + t_local.to(tl.int64) * (p_len + q_len + woff) + woff

    # ---- stage 1: the cached prefix from the NVFP4 pool (fp16 tensor-core dots on exactly dequantized values) ----
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum(p_len + tblk * BLOCK_T - WINDOW, 0)
        lo = (lo // BLOCK_N) * BLOCK_N
    for n0 in range(lo, p_len, BLOCK_N):
        n_idx = n0 + offs_n
        n_ok = n_idx < p_len
        slot = tl.load(kv_indices + p_start + n_idx, mask=n_ok, other=0).to(tl.int64)
        s = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for d0 in range(0, D, DK):
            q_e = tl.load(q_row_ptr + (d0 + 2 * jk)[None, :], mask=row_ok[:, None], other=0.0)
            q_o = tl.load(q_row_ptr + (d0 + 2 * jk + 1)[None, :], mask=row_ok[:, None], other=0.0)
            kp = tl.load(Kp + slot[None, :] * stride_kbt + kvh * stride_kbh + (d0 // 2 + jk)[:, None],
                         mask=n_ok[None, :], other=0)
            ks = tl.load(KS + slot[None, :] * stride_kst + kvh * stride_ksh + (d0 // 16 + jkb)[:, None],
                         mask=n_ok[None, :], other=0)
            k_lo, k_hi = _fp4_tile_t(kp, ks, HDK, BLOCK_N, not DOT_F32)
            if DOT_F32:
                s += tl.dot(q_e.to(tl.float32), k_lo.to(tl.float32)) + tl.dot(q_o.to(tl.float32), k_hi.to(tl.float32))
            else:
                s += tl.dot(q_e.to(tl.float16), k_lo)
                s += tl.dot(q_o.to(tl.float16), k_hi)
        s = s * (sm_scale * k_scale)
        if LOGIT_CAP:
            s = logit_cap * (2.0 * tl.sigmoid(2.0 * s / logit_cap) - 1.0)
        valid = n_ok[None, :] & row_ok[:, None]
        if WINDOW > 0:
            valid = valid & ((q_pos[:, None] - n_idx[None, :]) <= WINDOW)
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        vp = tl.load(Vp + slot[:, None] * stride_vbt + kvh * stride_vbh + jv[None, :], mask=n_ok[:, None], other=0)
        vs = tl.load(VS + slot[:, None] * stride_vst + kvh * stride_vsh + jvb[None, :], mask=n_ok[:, None], other=0)
        v_lo, v_hi = _fp4_tile_n(vp, vs, BLOCK_N, HDV, not DOT_F32)
        if DOT_F32:
            pb = p.to(tl.float16).to(tl.float32)
            pv_e = tl.dot(pb, v_lo.to(tl.float32))
            pv_o = tl.dot(pb, v_hi.to(tl.float32))
        else:
            pb = p.to(tl.float16)
            pv_e = tl.dot(pb, v_lo)
            pv_o = tl.dot(pb, v_hi)
        acc_e = acc_e * alpha[:, None] + pv_e * v_scale
        acc_o = acc_o * alpha[:, None] + pv_o * v_scale
        m_i = m_new

    # ---- stage 2: the new tokens (bf16 extend tensors) ----
    if IS_CAUSAL and not USE_MASK:
        hi = tl.minimum(q_len, (tblk + 1) * BLOCK_T)
    else:
        hi = q_len
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
                s += tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16))
        s = s * sm_scale
        if LOGIT_CAP:
            s = logit_cap * (2.0 * tl.sigmoid(2.0 * s / logit_cap) - 1.0)
        valid = n_ok[None, :] & row_ok[:, None]
        if USE_MASK:
            cm = tl.load(Mask + mask_base[:, None] + p_len + n_idx[None, :], mask=valid, other=0)
            valid = valid & (cm != 0)
        elif IS_CAUSAL:
            valid = valid & (n_idx[None, :] <= t_local[:, None])
        if WINDOW > 0:
            valid = valid & ((t_local[:, None] - n_idx[None, :]) <= WINDOW)
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v_base = V_ext + (q_start + n_idx)[:, None] * stride_vt + kvh * stride_vh
        v_e = tl.load(v_base + (2 * jv)[None, :], mask=n_ok[:, None], other=0.0)
        v_o = tl.load(v_base + (2 * jv + 1)[None, :], mask=n_ok[:, None], other=0.0)
        if DOT_F32:
            pb = p.to(tl.bfloat16).to(tl.float32)
            pv_e = tl.dot(pb, v_e.to(tl.float32))
            pv_o = tl.dot(pb, v_o.to(tl.float32))
        else:
            pb = p.to(tl.bfloat16)
            pv_e = tl.dot(pb, v_e.to(tl.bfloat16))
            pv_o = tl.dot(pb, v_o.to(tl.bfloat16))
        acc_e = acc_e * alpha[:, None] + pv_e
        acc_o = acc_o * alpha[:, None] + pv_o
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    o_ptr = O + (q_start + t_local)[:, None] * stride_ot + head[:, None] * stride_oh
    tl.store(o_ptr + (2 * jv)[None, :], (acc_e / l_safe[:, None]).to(O.dtype.element_ty), mask=row_ok[:, None])
    tl.store(o_ptr + (2 * jv + 1)[None, :], (acc_o / l_safe[:, None]).to(O.dtype.element_ty), mask=row_ok[:, None])


def extend_config(D: int, G: int, max_len_extend: int = 1 << 30) -> dict:
    """Native-extend geometry (SM120). Used for short extends (EAGLE target verify, draft extend, short chunks); long
    prefill chunks go through gather-dequant + the GQA-packed FP8 kernel instead (kv4tri_backend.py). All query
    tokens of a verify request fit one token block so each KV tile is read once (the regime is memory bound).
    Prefill-shaped sweep (kv4tri_tune.py extend, Qwen 3.8, 4096 new over 32k): block_t 4, block_n 64, dk 128,
    4 warps, 3 stages = 33 ms (the first default, block_n 32 / dk 64, was 60 ms)."""
    g_pad = triton.next_power_of_2(G)
    bt_max = max(1, 64 // g_pad)
    bt = min(bt_max, triton.next_power_of_2(max(1, int(max_len_extend))))
    if bt >= bt_max and max_len_extend > bt_max:
        bt = max(1, 32 // g_pad)
    bt = max(bt, max(1, 16 // g_pad))  # tensor-core dots need >= 16 rows
    if D >= 512:
        return dict(block_t=bt, block_n=64, dk=64, dv=256, num_warps=4, num_stages=1)
    return dict(block_t=bt, block_n=64, dk=128, dv=D, num_warps=4, num_stages=3)


def nvfp4_extend_fwd(q_extend, k_extend, v_extend, o_extend, k_fp4, v_fp4, k_sf, v_sf, qo_indptr, kv_indptr,
                     kv_indices, max_len_extend: int, sm_scale: float, k_scale: float = 1.0, v_scale: float = 1.0,
                     sliding_window_size: int = -1, custom_mask=None, mask_indptr=None, is_causal: bool = True,
                     logit_cap: float = 0.0, window_kv_offsets=None, *, block_t: int | None = None, block_n: int | None = None,
                     dk: int | None = None, dv: int | None = None, num_warps: int | None = None,
                     num_stages: int | None = None):
    """q/o [T, HQ, D]; k/v_extend [T, KVH, D] bf16 (the new tokens); k/v_fp4 uint8 [slots, KVH, D/2];
    k/v_sf uint8 or e4m3 [slots, KVH, D/16]; kv_indptr/kv_indices address the cached prefix only (SGLang extend
    convention). custom_mask (flattened bool/uint8, per request [q_len, prefix + q_len] rows starting at
    mask_indptr[r]) applies to the new-token part only (SGLang's skip_prefix_custom_mask=True)."""
    T, HQ, D = q_extend.shape
    KVH = k_extend.shape[1]
    G = HQ // KVH
    DVfull = v_extend.shape[-1]
    assert DVfull == D, "NVFP4 extend assumes qk_head_dim == v_head_dim"
    cfg = extend_config(D, G, int(max_len_extend))
    block_t = block_t or cfg["block_t"]
    block_n = block_n or cfg["block_n"]
    dk = min(dk or cfg["dk"], D)
    dv = min(dv or cfg["dv"], DVfull)
    num_warps = num_warps or cfg["num_warps"]
    num_stages = num_stages or cfg["num_stages"]
    assert D % dk == 0 and DVfull % dv == 0 and dk % 16 == 0 and dv % 32 == 0
    g_pad = triton.next_power_of_2(G)
    batch = qo_indptr.shape[0] - 1
    num_dv = DVfull // dv
    t_blocks = triton.cdiv(int(max_len_extend), block_t)
    use_mask = custom_mask is not None
    if use_mask:
        assert mask_indptr is not None
        mask = custom_mask.view(torch.uint8) if custom_mask.dtype == torch.bool else custom_mask
    else:
        mask = kv_indices  # unused pointer
        mask_indptr = kv_indptr
    k_sf = k_sf.view(torch.uint8)
    v_sf = v_sf.view(torch.uint8)
    k_fp4 = k_fp4.view(torch.uint8)
    v_fp4 = v_fp4.view(torch.uint8)
    grid = (batch, KVH, t_blocks * num_dv)
    _nvfp4_extend_kernel[grid](
        q_extend, k_extend, v_extend, o_extend, k_fp4, v_fp4, k_sf, v_sf, qo_indptr, kv_indptr, kv_indices,
        mask, mask_indptr, window_kv_offsets if window_kv_offsets is not None else kv_indptr,
        float(sm_scale), float(k_scale), float(v_scale), float(logit_cap),
        q_extend.stride(0), q_extend.stride(1), k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1), o_extend.stride(0), o_extend.stride(1),
        k_fp4.stride(0), k_fp4.stride(1), v_fp4.stride(0), v_fp4.stride(1),
        k_sf.stride(0), k_sf.stride(1), v_sf.stride(0), v_sf.stride(1),
        G=G, G_PAD=g_pad, BLOCK_T=block_t, BLOCK_N=block_n, D=D, DK=dk, DV=dv, NUM_DV=num_dv,
        WINDOW=sliding_window_size if sliding_window_size is not None and sliding_window_size > 0 else 0,
        IS_CAUSAL=bool(is_causal), USE_MASK=use_mask, LOGIT_CAP=bool(logit_cap and logit_cap > 0),
        DOT_F32=_interp(), HAS_WOFF=use_mask and window_kv_offsets is not None, num_warps=num_warps,
        num_stages=num_stages,
    )
    return o_extend


# ---------------------------------------------------------------------------------------------------------------
# short extend / EAGLE target verify: split-KV over the prefix (the regime is memory bound, like decode)
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _nvfp4_verify_stage1(
    Q, K_ext, V_ext, Kp, Vp, KS, VS, qo_indptr, kv_indptr, kv_indices, Mask, mask_indptr, WOff, Att, Lse,
    sm_scale, k_scale, logit_cap,
    stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh,
    stride_kbt, stride_kbh, stride_vbt, stride_vbh, stride_kst, stride_ksh, stride_vst, stride_vsh,
    stride_at, stride_ah, stride_as, stride_lt, stride_lh,
    NSPLIT: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
    D: tl.constexpr, DK: tl.constexpr, WINDOW: tl.constexpr, USE_MASK: tl.constexpr, LOGIT_CAP: tl.constexpr,
    DOT_F32: tl.constexpr, HAS_WOFF: tl.constexpr = False,
):
    """Programs split < NSPLIT: one slice of the cached prefix (FP4). Program split == NSPLIT: the new tokens (bf16),
    causal or custom mask. Each writes a normalized partial output + log-sum-exp per (token, head) row; empty
    partials write lse = -inf."""
    req = tl.program_id(0)
    kvh = tl.program_id(1)
    pid2 = tl.program_id(2)
    tblk = pid2 // (NSPLIT + 1)            # token block (short extends have exactly one)
    split = pid2 % (NSPLIT + 1)
    q_start = tl.load(qo_indptr + req)
    q_len = tl.load(qo_indptr + req + 1) - q_start
    p_start = tl.load(kv_indptr + req)
    p_len = tl.load(kv_indptr + req + 1) - p_start
    if tblk * BLOCK_T >= q_len:
        return

    BLOCK_M: tl.constexpr = BLOCK_T * G_PAD
    HD: tl.constexpr = D // 2
    rows = tl.arange(0, BLOCK_M)
    t_local = tblk * BLOCK_T + rows // G_PAD
    g = rows % G_PAD
    head = kvh * G + tl.minimum(g, G - 1)
    row_ok = (t_local < q_len) & (g < G)
    q_pos = p_len + t_local
    q_row_ptr = Q + (q_start + t_local)[:, None] * stride_qt + head[:, None] * stride_qh
    offs_n = tl.arange(0, BLOCK_N)
    j = tl.arange(0, HD)
    jb = tl.arange(0, HD // 8)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc_e = tl.zeros([BLOCK_M, HD], tl.float32)
    acc_o = tl.zeros([BLOCK_M, HD], tl.float32)
    if split < NSPLIT:
        per = tl.cdiv(tl.cdiv(p_len, NSPLIT), BLOCK_N) * BLOCK_N
        lo = per * split
        hi = tl.minimum(lo + per, p_len)
        if WINDOW > 0:
            lo = tl.maximum(lo, (tl.maximum(p_len + tblk * BLOCK_T - WINDOW, 0) // BLOCK_N) * BLOCK_N)
        q_e = tl.load(q_row_ptr + 2 * j[None, :], mask=row_ok[:, None], other=0.0)
        q_o = tl.load(q_row_ptr + 2 * j[None, :] + 1, mask=row_ok[:, None], other=0.0)
        if DOT_F32:
            q_e = q_e.to(tl.float32)
            q_o = q_o.to(tl.float32)
        else:
            q_e = q_e.to(tl.float16)
            q_o = q_o.to(tl.float16)
        for n0 in range(lo, hi, BLOCK_N):
            n_idx = n0 + offs_n
            n_ok = n_idx < hi
            slot = tl.load(kv_indices + p_start + n_idx, mask=n_ok, other=0).to(tl.int64)
            kp = tl.load(Kp + slot[None, :] * stride_kbt + kvh * stride_kbh + j[:, None], mask=n_ok[None, :], other=0)
            ks = tl.load(KS + slot[None, :] * stride_kst + kvh * stride_ksh + jb[:, None], mask=n_ok[None, :],
                         other=0)
            k_lo, k_hi = _fp4_tile_t(kp, ks, HD, BLOCK_N, not DOT_F32)
            if DOT_F32:
                s = tl.dot(q_e, k_lo.to(tl.float32)) + tl.dot(q_o, k_hi.to(tl.float32))
            else:
                s = tl.dot(q_e, k_lo) + tl.dot(q_o, k_hi)
            s = s * (sm_scale * k_scale)
            if LOGIT_CAP:
                s = logit_cap * (2.0 * tl.sigmoid(2.0 * s / logit_cap) - 1.0)
            valid = n_ok[None, :] & row_ok[:, None]
            if WINDOW > 0:
                valid = valid & ((q_pos[:, None] - n_idx[None, :]) <= WINDOW)
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            vp = tl.load(Vp + slot[:, None] * stride_vbt + kvh * stride_vbh + j[None, :], mask=n_ok[:, None], other=0)
            vs = tl.load(VS + slot[:, None] * stride_vst + kvh * stride_vsh + jb[None, :], mask=n_ok[:, None],
                         other=0)
            v_lo, v_hi = _fp4_tile_n(vp, vs, BLOCK_N, HD, not DOT_F32)
            if DOT_F32:
                pb = p.to(tl.float16).to(tl.float32)
                acc_e = acc_e * alpha[:, None] + tl.dot(pb, v_lo.to(tl.float32))
                acc_o = acc_o * alpha[:, None] + tl.dot(pb, v_hi.to(tl.float32))
            else:
                pb = p.to(tl.float16)
                acc_e = acc_e * alpha[:, None] + tl.dot(pb, v_lo)
                acc_o = acc_o * alpha[:, None] + tl.dot(pb, v_hi)
            m_i = m_new
    else:
        offs_dk_x = tl.arange(0, DK)
        if USE_MASK:
            woff_x = 0
            if HAS_WOFF:
                woff_x = tl.load(WOff + req).to(tl.int64)
            mask_base_x = tl.load(mask_indptr + req) + t_local.to(tl.int64) * (p_len + q_len + woff_x) + woff_x
        if USE_MASK:
            hi_x = q_len
        else:
            hi_x = tl.minimum(q_len, (tblk + 1) * BLOCK_T)
        for n0_x in range(0, hi_x, BLOCK_N):
            n_idx_x = n0_x + offs_n
            n_ok_x = n_idx_x < q_len
            s_x = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for d0_x in range(0, D, DK):
                q_x = tl.load(q_row_ptr + (d0_x + offs_dk_x)[None, :], mask=row_ok[:, None], other=0.0)
                k_x = tl.load(K_ext + (q_start + n_idx_x)[None, :] * stride_kt + kvh * stride_kh + (d0_x + offs_dk_x)[:, None],
                            mask=n_ok_x[None, :], other=0.0)
                if DOT_F32:
                    s_x += tl.dot(q_x.to(tl.float32), k_x.to(tl.float32))
                else:
                    s_x += tl.dot(q_x.to(tl.bfloat16), k_x.to(tl.bfloat16))
            s_x = s_x * sm_scale
            if LOGIT_CAP:
                s_x = logit_cap * (2.0 * tl.sigmoid(2.0 * s_x / logit_cap) - 1.0)
            valid_x = n_ok_x[None, :] & row_ok[:, None]
            if USE_MASK:
                cm_x = tl.load(Mask + mask_base_x[:, None] + p_len + n_idx_x[None, :], mask=valid_x, other=0)
                valid_x = valid_x & (cm_x != 0)
            else:
                valid_x = valid_x & (n_idx_x[None, :] <= t_local[:, None])
            if WINDOW > 0:
                valid_x = valid_x & ((t_local[:, None] - n_idx_x[None, :]) <= WINDOW)
            s_x = tl.where(valid_x, s_x, float("-inf"))
            m_new_x = tl.maximum(m_i, tl.max(s_x, 1))
            m_safe_x = tl.where(m_new_x == float("-inf"), 0.0, m_new_x)
            p_x = tl.exp(s_x - m_safe_x[:, None])
            alpha_x = tl.exp(m_i - m_safe_x)
            l_i = l_i * alpha_x + tl.sum(p_x, 1)
            v_base_x = V_ext + (q_start + n_idx_x)[:, None] * stride_vt + kvh * stride_vh
            v_e_x = tl.load(v_base_x + (2 * j)[None, :], mask=n_ok_x[:, None], other=0.0)
            v_o_x = tl.load(v_base_x + (2 * j + 1)[None, :], mask=n_ok_x[:, None], other=0.0)
            if DOT_F32:
                pb_x = p_x.to(tl.bfloat16).to(tl.float32)
                acc_e = acc_e * alpha_x[:, None] + tl.dot(pb_x, v_e_x.to(tl.float32))
                acc_o = acc_o * alpha_x[:, None] + tl.dot(pb_x, v_o_x.to(tl.float32))
            else:
                pb_x = p_x.to(tl.bfloat16)
                acc_e = acc_e * alpha_x[:, None] + tl.dot(pb_x, v_e_x.to(tl.bfloat16))
                acc_o = acc_o * alpha_x[:, None] + tl.dot(pb_x, v_o_x.to(tl.bfloat16))
            m_i = m_new_x
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    lse = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log(l_safe))
    a_ptr = Att + (q_start + t_local)[:, None].to(tl.int64) * stride_at + head[:, None] * stride_ah + split * stride_as
    tl.store(a_ptr + 2 * j[None, :], acc_e / l_safe[:, None], mask=row_ok[:, None])
    tl.store(a_ptr + 2 * j[None, :] + 1, acc_o / l_safe[:, None], mask=row_ok[:, None])
    tl.store(Lse + (q_start + t_local).to(tl.int64) * stride_lt + head * stride_lh + split, lse, mask=row_ok)


@triton.jit
def _nvfp4_verify_stage2(Att, Lse, O, v_scale, stride_at, stride_ah, stride_as, stride_lt, stride_lh, stride_ot,
                         stride_oh, NSPLIT: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    """Combine: prefix partials carry the V global scale, the new-token partial (split NSPLIT) does not."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    m_d = offs < D
    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_D], tl.float32)
    for s in range(0, NSPLIT + 1):
        tlse = tl.load(Lse + t * stride_lt + h * stride_lh + s)
        if tlse > float("-inf"):
            tv = tl.load(Att + t * stride_at + h * stride_ah + s * stride_as + offs, mask=m_d, other=0.0)
            if s < NSPLIT:
                tv = tv * v_scale
            n_max = tl.maximum(tlse, e_max)
            old = tl.exp(e_max - n_max)
            w = tl.exp(tlse - n_max)
            acc = acc * old + w * tv
            e_sum = e_sum * old + w
            e_max = n_max
    e_safe = tl.where(e_sum == 0.0, 1.0, e_sum)
    tl.store(O + t * stride_ot + h * stride_oh + offs, (acc / e_safe).to(O.dtype.element_ty), mask=m_d)


# Longest unmasked extend routed to the split-KV FP4 kernel; longer ones use gather-dequant + GQA FP8. Measured
# crossover (kv4tri_extend_ab.py midq, 1 x 100k prefix, per layer): Qwen 3.8 32 new tokens split 0.71 ms vs dq 5.0 ms
# (the unsplit native kernel: 6.85 ms), 300 tokens split 5.86 vs dq 4.96; Gemma global 32: 1.56 vs 5.95, 300: 12.1
# vs 6.2.
SPLIT_EXTEND_MAX = int(os.environ.get("NUMINOUS_KV4TRI_SPLIT_EXTEND_MAX", "128"))


def verify_max_len(G: int) -> int:
    """Longest per-request extend the split-KV verify kernel takes (all its query rows in one 64-row block)."""
    return max(1, 64 // triton.next_power_of_2(G))


def verify_splits(batch: int, kv_heads: int, sm_count: int = 188, max_splits: int = 64) -> int:
    """Prefix splits so batch x KV heads x splits covers about eight waves of SMs. CUDA-graph sweep on RTX PRO 6000
    (kv4tri_extend_ab.py vsweep, block_n 64, 1 stage; measured): Qwen 3.8 B=16 x 104k, 4 draft tokens 1.78 ms at
    24 splits (6 splits: 3.0 ms); B=8 x 32k 0.28 ms (FlashInfer FP8 verify on that shape: 0.42 ms, kv4fi bench);
    B=1 x 104k 0.14 ms at 64; Qwen 3.6 B=16 x 104k, 5 tokens 1.51 ms at 48; stock Triton FP8 verify 31.3 / 11.8 ms."""
    return int(max(1, min(max_splits, -(-8 * sm_count // max(1, batch * kv_heads)))))


def nvfp4_verify_fwd(q_extend, k_extend, v_extend, o_extend, k_fp4, v_fp4, k_sf, v_sf, qo_indptr, kv_indptr,
                     kv_indices, max_len_extend: int, sm_scale: float, k_scale: float = 1.0, v_scale: float = 1.0,
                     sliding_window_size: int = -1, custom_mask=None, mask_indptr=None, logit_cap: float = 0.0,
                     window_kv_offsets=None, *, num_splits: int | None = None, block_n: int = 64, dk: int = 64, num_warps: int = 4,
                     num_stages: int = 1):
    """Same contract as nvfp4_extend_fwd (causal only), for max_len_extend <= verify_max_len(G): EAGLE target
    verify, draft extend and short chunks. Partials live in a transient fp32 buffer [T, HQ, splits + 1, D]."""
    T, HQ, D = q_extend.shape
    KVH = k_extend.shape[1]
    G = HQ // KVH
    assert v_extend.shape[-1] == D
    g_pad = triton.next_power_of_2(G)
    # up to 64 rows (tokens x padded group) per program; >= 16 rows for the MMAs; verify fits one token block
    block_t = min(max(triton.next_power_of_2(max(1, int(max_len_extend))), 16 // g_pad, 1), verify_max_len(G))
    batch = qo_indptr.shape[0] - 1
    n_tb = triton.cdiv(max(1, int(max_len_extend)), block_t)
    if num_splits:
        ns = num_splits
    else:
        ns = verify_splits(batch * n_tb, KVH)
        # partial-output buffer bound (T x HQ x (splits + 1) x D fp32 <= 512 MB)
        ns = max(1, min(ns, (512 << 20) // max(1, T * HQ * D * 4) - 1))
    att = torch.empty(T, HQ, ns + 1, D, dtype=torch.float32, device=q_extend.device)
    lse = torch.empty(T, HQ, ns + 1, dtype=torch.float32, device=q_extend.device)
    use_mask = custom_mask is not None
    if use_mask:
        mask = custom_mask.view(torch.uint8) if custom_mask.dtype == torch.bool else custom_mask
    else:
        mask, mask_indptr = kv_indices, kv_indptr
    k_sf = k_sf.view(torch.uint8)
    v_sf = v_sf.view(torch.uint8)
    k_fp4 = k_fp4.view(torch.uint8)
    v_fp4 = v_fp4.view(torch.uint8)
    _nvfp4_verify_stage1[(batch, KVH, n_tb * (ns + 1))](
        q_extend, k_extend, v_extend, k_fp4, v_fp4, k_sf, v_sf, qo_indptr, kv_indptr, kv_indices, mask, mask_indptr,
        window_kv_offsets if window_kv_offsets is not None else kv_indptr, att, lse, float(sm_scale), float(k_scale), float(logit_cap),
        q_extend.stride(0), q_extend.stride(1), k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        k_fp4.stride(0), k_fp4.stride(1), v_fp4.stride(0), v_fp4.stride(1),
        k_sf.stride(0), k_sf.stride(1), v_sf.stride(0), v_sf.stride(1),
        att.stride(0), att.stride(1), att.stride(2), lse.stride(0), lse.stride(1),
        NSPLIT=ns, G=G, G_PAD=g_pad, BLOCK_T=block_t, BLOCK_N=block_n, D=D, DK=min(dk, D),
        WINDOW=sliding_window_size if sliding_window_size is not None and sliding_window_size > 0 else 0,
        USE_MASK=use_mask, LOGIT_CAP=bool(logit_cap and logit_cap > 0), DOT_F32=_interp(),
        HAS_WOFF=use_mask and window_kv_offsets is not None, num_warps=num_warps, num_stages=num_stages,
    )
    _nvfp4_verify_stage2[(T, HQ)](
        att, lse, o_extend, float(v_scale), att.stride(0), att.stride(1), att.stride(2), lse.stride(0), lse.stride(1),
        o_extend.stride(0), o_extend.stride(1), NSPLIT=ns, D=D, BLOCK_D=triton.next_power_of_2(D), num_warps=4,
    )
    return o_extend


# ---------------------------------------------------------------------------------------------------------------
# gather + dequantize (long prefill chunks: FP4 prefix -> dense workspace for a tensor-core-bound kernel)
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _nvfp4_gather_dequant_kernel(P, S, idx, Out, n_tok, gscale, stride_pt, stride_ph, stride_st, stride_sh,
                                 stride_ot, stride_oh, BLOCK_T: tl.constexpr, HD: tl.constexpr, FP8_OUT: tl.constexpr,
                                 USE_ASM: tl.constexpr):
    tb = tl.program_id(0)
    h = tl.program_id(1)
    t = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    t_ok = t < n_tok
    slot = tl.load(idx + t, mask=t_ok, other=0).to(tl.int64)
    j = tl.arange(0, HD)
    jb = tl.arange(0, HD // 8)
    p = tl.load(P + slot[:, None] * stride_pt + h * stride_ph + j[None, :], mask=t_ok[:, None], other=0)
    sf = tl.load(S + slot[:, None] * stride_st + h * stride_sh + jb[None, :], mask=t_ok[:, None], other=0)
    lo, hi = _fp4_tile_n(p, sf, BLOCK_T, HD, USE_ASM)
    x = tl.reshape(tl.join(lo.to(tl.float32), hi.to(tl.float32)), [BLOCK_T, 2 * HD]) * gscale
    d = tl.arange(0, 2 * HD)
    o_ptr = Out + t[:, None].to(tl.int64) * stride_ot + h * stride_oh + d[None, :]
    if FP8_OUT:
        tl.store(o_ptr, tl.clamp(x, -448.0, 448.0).to(tl.float8e4nv), mask=t_ok[:, None])
    else:
        tl.store(o_ptr, x.to(Out.dtype.element_ty), mask=t_ok[:, None])


def nvfp4_gather_dequant(packed, sf, idx, out, global_scale: float = 1.0, block_t: int = 32):
    """out[t] = dequant(pool[idx[t]]) for t < len(idx): packed uint8 [slots, H, D/2], sf [slots, H, D/16],
    out [>=len(idx), H, D] (bf16/fp16 exact; fp8 e4m3 rounds and saturates at +-448, the FP8-KV range)."""
    n = idx.numel()
    H, HD = packed.shape[1], packed.shape[2]
    packed = packed.view(torch.uint8)
    sf = sf.view(torch.uint8)
    if n == 0:
        return out
    _nvfp4_gather_dequant_kernel[(triton.cdiv(n, block_t), H)](
        packed, sf, idx, out, n, float(global_scale), packed.stride(0), packed.stride(1), sf.stride(0), sf.stride(1),
        out.stride(0), out.stride(1), BLOCK_T=block_t, HD=HD, FP8_OUT=out.dtype == torch.float8_e4m3fn,
        USE_ASM=not _interp(), num_warps=4)
    return out


# ---------------------------------------------------------------------------------------------------------------
# torch references (format definition + tests)
# ---------------------------------------------------------------------------------------------------------------
def nvfp4_quantize_ref(x: torch.Tensor, global_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., D] -> (packed uint8 [..., D/2], block scales uint8(E4M3 bits) [..., D/16]); x ~= e2m1*sf*gs."""
    *lead, D = x.shape
    xb = x.float().reshape(-1, D // 16, 16) / global_scale
    amax = xb.abs().amax(-1, keepdim=True)
    sf = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    sff = sf.float()
    y = torch.where(sff > 0, xb / sff.clamp_min(1e-30), torch.zeros_like(xb))
    vals = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=x.device)
    code = (y.abs().unsqueeze(-1) - vals).abs().argmin(-1).to(torch.uint8)
    code = code | ((y < 0).to(torch.uint8) << 3)
    code = code.reshape(-1, D)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.reshape(*lead, D // 2), sf.view(torch.uint8).reshape(*lead, D // 16)


def nvfp4_dequantize_ref(packed: torch.Tensor, sf: torch.Tensor, global_scale: float = 1.0) -> torch.Tensor:
    *lead, HD = packed.shape
    D = HD * 2
    p = packed.reshape(-1, HD).to(torch.int64)
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).reshape(-1, D)
    vals = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    x = vals[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    s = sf.reshape(-1, D // 16).view(torch.float8_e4m3fn).float()
    x = x.reshape(-1, D // 16, 16) * s[..., None] * global_scale
    return x.reshape(*lead, D)


def decode_reference(q, k, v, kv_indptr, kv_indices, sm_scale, logit_cap: float = 0.0) -> torch.Tensor:
    """q [B, HQ, D]; k/v dequantized pools [slots, KVH, D] -> [B, HQ, DV] float32."""
    B, HQ, D = q.shape
    G = HQ // k.shape[1]
    out = torch.zeros(B, HQ, v.shape[-1], dtype=torch.float32, device=q.device)
    for b in range(B):
        idx = kv_indices[int(kv_indptr[b]):int(kv_indptr[b + 1])].long()
        if idx.numel() == 0:
            continue
        kk = k[idx].float().repeat_interleave(G, dim=1)
        vv = v[idx].float().repeat_interleave(G, dim=1)
        s = torch.einsum("hd,thd->ht", q[b].float(), kk) * sm_scale
        if logit_cap > 0:
            s = logit_cap * torch.tanh(s / logit_cap)
        out[b] = torch.einsum("ht,thd->hd", s.softmax(-1), vv)
    return out


def extend_reference(q_extend, k_extend, v_extend, k_pool, v_pool, qo_indptr, kv_indptr, kv_indices, sm_scale,
                     sliding_window_size: int = -1, custom_mask=None, mask_indptr=None,
                     logit_cap: float = 0.0, window_kv_offsets=None) -> torch.Tensor:
    """Dense float32 attention with SGLang's extend semantics; k/v_pool are dequantized [slots, KVH, D]."""
    T, HQ, D = q_extend.shape
    G = HQ // k_extend.shape[1]
    out = torch.zeros(T, HQ, v_extend.shape[-1], dtype=torch.float32, device=q_extend.device)
    for r in range(qo_indptr.shape[0] - 1):
        qs, qe = int(qo_indptr[r]), int(qo_indptr[r + 1])
        ps, pe = int(kv_indptr[r]), int(kv_indptr[r + 1])
        P, Tn = pe - ps, qe - qs
        if Tn == 0:
            continue
        idx = kv_indices[ps:pe].long()
        k = torch.cat([k_pool[idx].float(), k_extend[qs:qe].float()])
        v = torch.cat([v_pool[idx].float(), v_extend[qs:qe].float()])
        qpos = P + torch.arange(Tn, device=q_extend.device)
        kpos = torch.arange(P + Tn, device=q_extend.device)
        if custom_mask is not None:
            m0 = int(mask_indptr[r])
            off = int(window_kv_offsets[r]) if window_kv_offsets is not None else 0
            cm = custom_mask[m0:m0 + Tn * (off + P + Tn)].reshape(Tn, off + P + Tn)[:, off:].bool()
            mask = cm.clone()
            mask[:, :P] = True
        else:
            mask = kpos[None, :] <= qpos[:, None]
        if sliding_window_size is not None and sliding_window_size > 0:
            mask &= (qpos[:, None] - kpos[None, :]) <= sliding_window_size
        for h in range(HQ):
            s = (q_extend[qs:qe, h].float() @ k[:, h // G].T) * sm_scale
            if logit_cap > 0:
                s = logit_cap * torch.tanh(s / logit_cap)
            s = s.masked_fill(~mask, float("-inf"))
            out[qs:qe, h] = torch.softmax(s, -1) @ v[:, h // G]
    return out
