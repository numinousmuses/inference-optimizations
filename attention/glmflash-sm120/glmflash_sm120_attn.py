"""GLM-5.3-Flash sparse MLA (DSA layers) on SM120 (RTX PRO 6000): row formats, torch reference, Triton kernels.

Flash's 11 DSA layers are MLA with kv_lora_rank 512 and NO rope (`qk_rope_head_dim 0`, source_code config.json), so
attention is q [T, H, 512] (absorbed) against one 512-value latent per token (K = V), top-k 2048 pooled tokens plus up
to 3 kpool tail tokens: the index table is [T, 2051] of pool slots, compact, -1 padded (source_code
kpool_topk_transform.cuh: valid = min(4 * closed_pools, 2048) + seq_len % 4 entries from column 0).

Row formats (one row per token per DSA layer; `kv_buffer[l]` is [pool_rows, 1, row] in SGLang's DSA pool):

    FMT_BF16     1024 B  512 bf16                                      (stock bf16 pool, `--kv-cache-dtype bfloat16`)
    FMT_FP8_656   656 B  [0,512) e4m3 latent | [512,528) 4 fp32 scales, one per 128 values, scale = amax/448
                         (arbitrary, the GLM-5.3 writer's convention) | [528,656) 128 zero bytes (a zero bf16 rope)
                         = FlashInfer 0.6.18's SM120 `GLM_NSA` row (kv_cache_traits.cuh), so FlashInfer's kernels read
                         it with a zero q_rope (glmflash_sm120_fi.py); our Triton kernel reads it too
    FMT_FP4_288   288 B  [0,256) 512 E2M1 nibbles (element 2i low nibble of byte i) | [256,288) 32 E4M3 scales, one
                         per 16 values = operations/kernels/glmflash_kv4.py's row exactly (served on B300 at GSM8K
                         0.975 = FP8 lead, research/glmflash.md)
    FMT_FP8_528   528 B  [0,512) e4m3 latent | [512,528) 4 fp32 scales = the STOCK rope-free FP8 DSA row
                         (kv_cache_configurator: kv_lora_rank + kv_lora_rank/128*4 + rope 0; written by
                         quantize_k_cache_separate). No SM120/SM90 vendor kernel reads it with a 2051-wide kpool table;
                         our Triton kernel does (Hopper FP8 KV for Flash, agents/findings/hopper_fp8kv.md)

Kernels:
  sparse_mla          q [T, H, 512] bf16 (any strides, last dim contiguous; a [T, H, 576] buffer's [:, :, :512]
                      view works) x rows in one of the three formats at idx [T, K] (-1 skipped, any K) ->
                      out [T, H, 512] bf16 and lse [T, H] fp32 in FlashInfer's convention (log2 of the sum of
                      exp2(score * sm_scale * log2 e); -1e30 for a row with no valid index). Split-K over the
                      top-k (flash decoding) when T x head-blocks cannot fill the GPU, plus a combine kernel.
  merge_attn          (out_a, lse_a) += (out_b, lse_b) in place (log2 LSE): joins the kpool tail to a stock
                      FlashInfer top-2048 call (the no-JIT fallback of glmflash_sm120_fi.py)
  nsa656_quant_write  latent [T, 512] -> 656-byte rows at pool[loc[t]]; loc 0 (the CUDA-graph padding slot) and
                      loc < 0 are skipped; an all-zero block gets scale 1 (the stock writer divides by amax and
                      writes NaN codes there, which FlashInfer's -1 -> slot 0 gather would then multiply by P = 0)

SM120 tile math (derived; 99 KB shared memory per block, mma.sync m16n8k16 bf16, no TMA gather in Triton):
BLOCK_H 16 heads x BLOCK_N 32 rows: q 16 KB + the dequantized row tile (32 x 512 bf16 = 32 KB, two operand copies
for QK^T and PV) = 80 KB; BLOCK_N 64 needs 144 KB and does not fit. acc [16, 512] fp32 = 64 registers/thread at
4 warps. Per row the math is 16 x 1024 x 2 = 32.8 kFLOP per 16 heads against 288/656/1024 bytes: 114/50/32 FLOP/B,
below SM120's ~250 FLOP/B bf16 balance (assumed ~450 TFLOPS dense bf16 / 1.8 TB/s), so the kernel is gather-bound
at every head count up to 64 (4 head blocks re-read the rows; L2 128 MB holds them).

CPU tests: operations/kernels/test_glmflash_sm120_attn.py (Triton interpreter; the interpreter's bf16 tl.dot is
wrong, so under TRITON_INTERPRET=1 the dots run in fp32: DOT_F32). GPU tests: test_glmflash_sm120_gpu.py (not run:
no GPU; research/rtx-flash/attention.md section 8 is the plan).
"""
from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # CPU box without triton: the reference still imports
    HAVE_TRITON = False

D = 512
FMT_BF16, FMT_FP8_656, FMT_FP4_288, FMT_FP8_528 = 0, 1, 2, 3
FMT_NAMES = {FMT_BF16: "bf16", FMT_FP8_656: "fp8_656", FMT_FP4_288: "fp4_288", FMT_FP8_528: "fp8_528"}
ROW_BYTES = {FMT_BF16: 1024, FMT_FP8_656: 656, FMT_FP4_288: 288, FMT_FP8_528: 528}
FP8_FMTS = (FMT_FP8_656, FMT_FP8_528)
STOCK_FP8_ROW = 528
NSA_ROW, NSA_SCALE_OFF, NSA_ROPE_OFF, NSA_TILE = 656, 512, 528, 128
FP4_ROW, FP4_SCALE_OFF, FP4_BLOCK = 288, 256, 16
FP8_MAX = 448.0
LOG2E = 1.4426950408889634
LSE_EMPTY = -1e30  # FlashInfer's sparse MLA writes this for a row with no valid candidate
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def interpret_mode() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


# ------------------------------------------------------------------------------------------------ torch reference


def pow2_ceil_ref(r: torch.Tensor) -> torch.Tensor:
    """Smallest power of two >= r (r > 0 finite fp32), by exponent bits: exact, no log2 rounding."""
    bits = r.float().contiguous().view(torch.int32)
    e = (bits >> 23) & 0xFF
    e = e + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (e << 23).view(torch.float32)


def nsa656_quantize_ref(latent: torch.Tensor, pow2: bool = False) -> torch.Tensor:
    """latent [T, 512] -> uint8 [T, 656]: the stock quantize_k_cache_separate numerics (scale = amax/128-block / 448,
    code = e4m3(clamp(x * (1/scale), +-448))) except that an all-zero block gets scale 1 instead of NaN codes.
    pow2=True rounds every scale UP to a power of two (DeepSeek V3.2 / FlashMLA convention): the row is then FlashInfer's
    DSV3_2 type (UE8M0-exact scales -> block-scaled QK MMA, single-pass FP8 P in PV) instead of GLM_NSA (arbitrary fp32
    scales -> per-block fp32 QK commits and a TWO-pass hi+residual FP8 P in PV; scale_mma.cuh WeightFp8PassTraits)."""
    t = latent.shape[0]
    x = latent.float().reshape(t, D // NSA_TILE, NSA_TILE)
    amax = x.abs().amax(dim=-1)
    s = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    if pow2:
        s = pow2_ceil_ref(s)
    inv = 1.0 / s
    codes = (x * inv.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    out = torch.zeros(t, NSA_ROW, dtype=torch.uint8, device=latent.device)
    out[:, :D] = codes.reshape(t, D).view(torch.uint8)
    out[:, NSA_SCALE_OFF:NSA_ROPE_OFF] = s.contiguous().view(torch.uint8).reshape(t, 16)
    return out


def nsa656_dequant_ref(rows: torch.Tensor) -> torch.Tensor:
    """uint8 [R, 656] or the stock rope-free [R, 528] -> fp32 [R, 512] (both carry e4m3 [0,512) + fp32 scales
    [512,528))."""
    r = rows.shape[0]
    codes = rows[:, :D].contiguous().view(torch.float8_e4m3fn).float().reshape(r, D // NSA_TILE, NSA_TILE)
    s = rows[:, NSA_SCALE_OFF:NSA_ROPE_OFF].contiguous().view(torch.float32)
    return (codes * s.unsqueeze(-1)).reshape(r, D)


def fp4_288_dequant_ref(rows: torch.Tensor) -> torch.Tensor:
    """uint8 [R, 288] (glmflash_kv4 layout) -> fp32 [R, 512]."""
    r = rows.shape[0]
    packed = rows[:, :FP4_SCALE_OFF].to(torch.int32)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(r, D)
    mag = E2M1_VALUES.to(rows.device)[codes & 7]
    val = torch.where(((codes & 8) != 0) & (mag != 0), -mag, mag)
    s = rows[:, FP4_SCALE_OFF:].contiguous().view(torch.float8_e4m3fn).float()
    return (val.reshape(r, D // FP4_BLOCK, FP4_BLOCK) * s.unsqueeze(-1)).reshape(r, D)


def pool_rows(kv_buffer: torch.Tensor, fmt: int) -> torch.Tensor:
    """A pool buffer ([N, 1, row] or [N, row], any dtype view) as [N, 512] bf16 (FMT_BF16) or uint8 [N, row_bytes]."""
    if fmt == FMT_BF16:
        b = kv_buffer if kv_buffer.dtype == torch.bfloat16 else kv_buffer.view(torch.bfloat16)
        b = b.reshape(b.shape[0], -1)
        assert b.shape[1] == D, b.shape
        return b
    b = kv_buffer if kv_buffer.dtype == torch.uint8 else kv_buffer.view(torch.uint8)
    b = b.reshape(b.shape[0], -1)
    assert b.shape[1] == ROW_BYTES[fmt], (b.shape, FMT_NAMES[fmt])
    return b


def dequant_ref(rows: torch.Tensor, fmt: int) -> torch.Tensor:
    if fmt == FMT_BF16:
        return rows.float()
    if fmt in FP8_FMTS:
        return nsa656_dequant_ref(rows)
    if fmt == FMT_FP4_288:
        return fp4_288_dequant_ref(rows)
    raise ValueError(fmt)


def sparse_mla_ref(q: torch.Tensor, kv: torch.Tensor, idx: torch.Tensor, sm_scale: float):
    """fp32 reference. q [T, H, 512], kv [N, 512] fp32 (dequantized rows), idx [T, K] (-1 masked) ->
    (out [T, H, 512] fp32, lse [T, H] fp32 log2-domain, LSE_EMPTY where no valid index)."""
    t, h, _ = q.shape
    out = torch.zeros(t, h, D, dtype=torch.float32, device=q.device)
    lse = torch.full((t, h), LSE_EMPTY, dtype=torch.float32, device=q.device)
    qf = q.float()
    for i in range(t):
        sel = idx[i].long()
        sel = sel[sel >= 0]
        if sel.numel() == 0:
            continue
        k = kv[sel].float()
        s = (qf[i] @ k.T) * (sm_scale * LOG2E)  # [H, n] log2 domain
        m = s.amax(dim=-1, keepdim=True)
        p = torch.exp2(s - m)
        l_ = p.sum(dim=-1, keepdim=True)
        out[i] = (p / l_) @ k
        lse[i] = (m + torch.log2(l_)).squeeze(-1)
    return out, lse


def merge_ref(out_a, lse_a, out_b, lse_b):
    """Merge two partial attentions (log2 LSE, LSE_EMPTY = no mass)."""
    la, lb = lse_a.float(), lse_b.float()
    m = torch.maximum(la, lb)
    wa = torch.where(la <= LSE_EMPTY / 2, torch.zeros_like(la), torch.exp2(la - m))
    wb = torch.where(lb <= LSE_EMPTY / 2, torch.zeros_like(lb), torch.exp2(lb - m))
    tot = wa + wb
    safe = torch.where(tot > 0, tot, torch.ones_like(tot))
    out = (out_a.float() * wa.unsqueeze(-1) + out_b.float() * wb.unsqueeze(-1)) / safe.unsqueeze(-1)
    lse = torch.where(tot > 0, m + torch.log2(safe), torch.full_like(m, LSE_EMPTY))
    return out, lse


# ------------------------------------------------------------------------------------------------ split heuristic


def choose_splits(t: int, h: int, k: int, block_h: int = 16, block_n: int = 32, num_sms: int = 188,
                  min_tiles_per_split: int = 4, waves: int = 2) -> int:
    """Flash-decoding split count: fill `waves` x SMs with (T x head blocks x splits) CTAs, keep >= min_tiles_per_split
    row tiles per split. 188 = RTX PRO 6000 SM count (NVIDIA spec). Returns 1 for prefill-sized T."""
    hb = triton_cdiv(h, block_h)
    tiles = max(1, triton_cdiv(k, block_n))
    ctas = t * hb
    if ctas >= waves * num_sms:
        return 1
    want = triton_cdiv(waves * num_sms, ctas)
    cap = max(1, tiles // min_tiles_per_split)
    return int(max(1, min(want, cap)))


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def device_sms(device=None) -> int:
    """SM count of the device (188 RTX PRO 6000, 132 H100/H200 SXM); 188 off-GPU (interpreter tests)."""
    if torch.cuda.is_available() and (device is None or torch.device(device).type == "cuda"):
        return int(torch.cuda.get_device_properties(device).multi_processor_count)
    return 188


# ------------------------------------------------------------------------------------------------ Triton kernels

if HAVE_TRITON:

    @triton.jit
    def _e2m1_value(c):
        """E2M1 code 0..15 (int32, sign in bit 3) -> fp32 value."""
        m = c & 7
        e = m >> 1
        man = (m & 1).to(tl.float32)
        pw = tl.where(e == 3, 4.0, tl.where(e == 2, 2.0, 1.0))
        mag = tl.where(e == 0, 0.5 * man, (1.0 + 0.5 * man) * pw)
        return tl.where((c & 8) != 0, -mag, mag)

    @triton.jit
    def _e2m1_bf16(c):
        """E2M1 code 0..15 (int16, sign in bit 3) -> bf16 by bit construction: e = (c >> 1) & 3, m = c & 1;
        e > 0: (1 + m/2) * 2^(e-1) = bits (e + 126) << 7 | m << 6; e = 0: m * 0.5 = m * 0x3F00; sign << 15."""
        e = (c >> 1) & 3
        m = c & 1
        mag = tl.where(e > 0, ((e + 126) << 7) | (m << 6), m * 0x3F00)
        bits = mag | ((c & 8) << 12)
        return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def _softmax_step(s, valid, m_i, l_i):
        """Online-softmax update in log2 domain. Returns (p, alpha, m_new, l_new)."""
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        return p, alpha, m_new, l_i * alpha + tl.sum(p, axis=1)

    @triton.jit
    def _dot(a, b, DOT_F32: tl.constexpr):
        if DOT_F32:
            return tl.dot(a.to(tl.float32), b.to(tl.float32))
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))

    @triton.jit
    def _store_block(acc, l_i, m_i, t, h, hmask, col, out_ptr, stride_ot, stride_oh, part_o_ptr, row,
                     SPLITS: tl.constexpr):
        """acc [BLOCK_H, n] for output columns `col` [n]: final (normalized, bf16) or split partial (fp32)."""
        if SPLITS == 1:
            l_safe = tl.where(l_i > 0, l_i, 1.0)
            op = out_ptr + t * stride_ot + h.to(tl.int64)[:, None] * stride_oh + col[None, :]
            tl.store(op, (acc / l_safe[:, None]).to(tl.bfloat16), mask=hmask[:, None])
        else:
            tl.store(part_o_ptr + row[:, None] * 512 + col[None, :], acc, mask=hmask[:, None])

    @triton.jit
    def _store_ml(l_i, m_i, t, h, hmask, lse_ptr, part_ml_ptr, row, H: tl.constexpr, SPLITS: tl.constexpr):
        if SPLITS == 1:
            has = l_i > 0
            tl.store(lse_ptr + t * H + h, tl.where(has, m_i + tl.log2(tl.where(has, l_i, 1.0)), -1e30), mask=hmask)
        else:
            tl.store(part_ml_ptr + row * 2, m_i, mask=hmask)
            tl.store(part_ml_ptr + row * 2 + 1, l_i, mask=hmask)

    @triton.jit
    def _rtx_sparse_mla_kernel(q_ptr, stride_qt, stride_qh, kv_ptr, kv_f32_ptr, n_rows, idx_ptr, stride_it, K,
                               out_ptr, stride_ot, stride_oh, lse_ptr, part_o_ptr, part_ml_ptr, sm_scale_log2,
                               H: tl.constexpr, FMT: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
                               SPLITS: tl.constexpr, TILES_PER_SPLIT, DOT_F32: tl.constexpr,
                               NUM_STAGES: tl.constexpr, ROW_B: tl.constexpr):
        """bf16 (FMT 0) and e4m3 rows of ROW_B bytes (656: FMT 1; the stock rope-free 528: FMT 3; fp32 scales at
        byte 512). Split-dim: the 512 latent columns are 4 blocks of 128, one
        per FP8 scale, so the scale is folded into the scores (QK) and into P (PV) instead of dequantizing the tile:
        s = sum_b sc[:, b] * (q_b . c_b), acc_b += (p * sc[:, b]) @ c_b. e4m3 codes are exact in bf16."""
        t = tl.program_id(0).to(tl.int64)
        hb = tl.program_id(1)
        sp = tl.program_id(2)
        h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = h < H
        c = tl.arange(0, 128)
        qb = q_ptr + t * stride_qt + h.to(tl.int64)[:, None] * stride_qh + c[None, :]
        q0 = tl.load(qb, mask=hmask[:, None], other=0.0)
        q1 = tl.load(qb + 128, mask=hmask[:, None], other=0.0)
        q2 = tl.load(qb + 256, mask=hmask[:, None], other=0.0)
        q3 = tl.load(qb + 384, mask=hmask[:, None], other=0.0)
        m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_H], tl.float32)
        a0 = tl.zeros([BLOCK_H, 128], tl.float32)
        a1 = tl.zeros([BLOCK_H, 128], tl.float32)
        a2 = tl.zeros([BLOCK_H, 128], tl.float32)
        a3 = tl.zeros([BLOCK_H, 128], tl.float32)
        n_tiles = tl.cdiv(K, BLOCK_N)
        tile_lo = sp * TILES_PER_SPLIT
        tile_hi = tl.minimum(tile_lo + TILES_PER_SPLIT, n_tiles)
        irow = idx_ptr + t * stride_it
        for tile in tl.range(tile_lo, tile_hi, num_stages=NUM_STAGES):
            j = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            slot = tl.load(irow + j, mask=j < K, other=-1).to(tl.int64)
            valid = (slot >= 0) & (slot < n_rows)
            slot = tl.where(valid, slot, 0)
            if FMT == 0:
                rp = kv_ptr + tl.multiple_of(slot * 512, 16)[:, None] + c[None, :]
                c0 = tl.load(rp, mask=valid[:, None], other=0.0)
                c1 = tl.load(rp + 128, mask=valid[:, None], other=0.0)
                c2 = tl.load(rp + 256, mask=valid[:, None], other=0.0)
                c3 = tl.load(rp + 384, mask=valid[:, None], other=0.0)
                s = _dot(q0, tl.trans(c0), DOT_F32) + _dot(q1, tl.trans(c1), DOT_F32)
                s += _dot(q2, tl.trans(c2), DOT_F32) + _dot(q3, tl.trans(c3), DOT_F32)
                p, alpha, m_i, l_i = _softmax_step(s * sm_scale_log2, valid, m_i, l_i)
                a0 = a0 * alpha[:, None] + _dot(p, c0, DOT_F32)
                a1 = a1 * alpha[:, None] + _dot(p, c1, DOT_F32)
                a2 = a2 * alpha[:, None] + _dot(p, c2, DOT_F32)
                a3 = a3 * alpha[:, None] + _dot(p, c3, DOT_F32)
            else:
                rp = kv_ptr + tl.multiple_of(slot * ROW_B, 16)[:, None] + c[None, :]
                c0 = tl.load(rp, mask=valid[:, None], other=0).to(tl.float8e4nv, bitcast=True)
                c1 = tl.load(rp + 128, mask=valid[:, None], other=0).to(tl.float8e4nv, bitcast=True)
                c2 = tl.load(rp + 256, mask=valid[:, None], other=0).to(tl.float8e4nv, bitcast=True)
                c3 = tl.load(rp + 384, mask=valid[:, None], other=0).to(tl.float8e4nv, bitcast=True)
                sb = kv_f32_ptr + slot * (ROW_B // 4) + 128
                s0 = tl.load(sb, mask=valid, other=0.0)
                s1 = tl.load(sb + 1, mask=valid, other=0.0)
                s2 = tl.load(sb + 2, mask=valid, other=0.0)
                s3 = tl.load(sb + 3, mask=valid, other=0.0)
                s = _dot(q0, tl.trans(c0), DOT_F32) * s0[None, :] + _dot(q1, tl.trans(c1), DOT_F32) * s1[None, :]
                s += _dot(q2, tl.trans(c2), DOT_F32) * s2[None, :] + _dot(q3, tl.trans(c3), DOT_F32) * s3[None, :]
                p, alpha, m_i, l_i = _softmax_step(s * sm_scale_log2, valid, m_i, l_i)
                a0 = a0 * alpha[:, None] + _dot(p * s0[None, :], c0, DOT_F32)
                a1 = a1 * alpha[:, None] + _dot(p * s1[None, :], c1, DOT_F32)
                a2 = a2 * alpha[:, None] + _dot(p * s2[None, :], c2, DOT_F32)
                a3 = a3 * alpha[:, None] + _dot(p * s3[None, :], c3, DOT_F32)
        row = (t * SPLITS + sp) * H + h.to(tl.int64)
        _store_block(a0, l_i, m_i, t, h, hmask, c, out_ptr, stride_ot, stride_oh, part_o_ptr, row, SPLITS)
        _store_block(a1, l_i, m_i, t, h, hmask, c + 128, out_ptr, stride_ot, stride_oh, part_o_ptr, row, SPLITS)
        _store_block(a2, l_i, m_i, t, h, hmask, c + 256, out_ptr, stride_ot, stride_oh, part_o_ptr, row, SPLITS)
        _store_block(a3, l_i, m_i, t, h, hmask, c + 384, out_ptr, stride_ot, stride_oh, part_o_ptr, row, SPLITS)
        _store_ml(l_i, m_i, t, h, hmask, lse_ptr, part_ml_ptr, row, H, SPLITS)

    @triton.jit
    def _rtx_sparse_mla_fp4_kernel(q_ptr, stride_qt, stride_qh, kv_ptr, n_rows, idx_ptr, stride_it, K,
                                   out_ptr, stride_ot, stride_oh, lse_ptr, part_o_ptr, part_ml_ptr, sm_scale_log2,
                                   H: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
                                   SPLITS: tl.constexpr, TILES_PER_SPLIT, DOT_F32: tl.constexpr,
                                   NUM_STAGES: tl.constexpr):
        """288-byte E2M1 rows. Even/odd split: byte i holds latent elements 2i (low nibble) and 2i+1 (high), so the
        low nibbles form the even-column half and the high nibbles the odd half; q and out use the same split (no
        interleave in registers). value(code) x E4M3 scale has <= 6 significant bits: exact in bf16."""
        t = tl.program_id(0).to(tl.int64)
        hb = tl.program_id(1)
        sp = tl.program_id(2)
        h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = h < H
        i = tl.arange(0, 256)
        qb = q_ptr + t * stride_qt + h.to(tl.int64)[:, None] * stride_qh + (2 * i)[None, :]
        qe = tl.load(qb, mask=hmask[:, None], other=0.0)
        qo = tl.load(qb + 1, mask=hmask[:, None], other=0.0)
        m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_H], tl.float32)
        ae = tl.zeros([BLOCK_H, 256], tl.float32)
        ao = tl.zeros([BLOCK_H, 256], tl.float32)
        n_tiles = tl.cdiv(K, BLOCK_N)
        tile_lo = sp * TILES_PER_SPLIT
        tile_hi = tl.minimum(tile_lo + TILES_PER_SPLIT, n_tiles)
        irow = idx_ptr + t * stride_it
        for tile in tl.range(tile_lo, tile_hi, num_stages=NUM_STAGES):
            j = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            slot = tl.load(irow + j, mask=j < K, other=-1).to(tl.int64)
            valid = (slot >= 0) & (slot < n_rows)
            slot = tl.where(valid, slot, 0)
            base = kv_ptr + tl.multiple_of(slot * 288, 16)[:, None]
            b = tl.load(base + i[None, :], mask=valid[:, None], other=0)
            sc8 = tl.load(base + 256 + (i // 8)[None, :], mask=valid[:, None], other=0).to(tl.float8e4nv, bitcast=True)
            if DOT_F32:
                bi = b.to(tl.int32)
                sc = sc8.to(tl.float32)
                ke = _e2m1_value(bi & 15) * sc
                ko = _e2m1_value((bi >> 4) & 15) * sc
            else:  # bf16 all the way: E2M1 -> bf16 bits by integer ops, x scale exact in bf16 (<= 6 significant bits)
                sc = sc8.to(tl.bfloat16)
                ke = _e2m1_bf16(b.to(tl.int16) & 15) * sc
                ko = _e2m1_bf16((b.to(tl.int16) >> 4) & 15) * sc
            s = _dot(qe, tl.trans(ke), DOT_F32) + _dot(qo, tl.trans(ko), DOT_F32)
            p, alpha, m_i, l_i = _softmax_step(s * sm_scale_log2, valid, m_i, l_i)
            ae = ae * alpha[:, None] + _dot(p, ke, DOT_F32)
            ao = ao * alpha[:, None] + _dot(p, ko, DOT_F32)
        row = (t * SPLITS + sp) * H + h.to(tl.int64)
        _store_block(ae, l_i, m_i, t, h, hmask, 2 * i, out_ptr, stride_ot, stride_oh, part_o_ptr, row, SPLITS)
        _store_block(ao, l_i, m_i, t, h, hmask, 2 * i + 1, out_ptr, stride_ot, stride_oh, part_o_ptr, row, SPLITS)
        _store_ml(l_i, m_i, t, h, hmask, lse_ptr, part_ml_ptr, row, H, SPLITS)

    @triton.jit
    def _rtx_combine_kernel(part_o_ptr, part_ml_ptr, out_ptr, stride_ot, stride_oh, lse_ptr, H: tl.constexpr,
                            SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1).to(tl.int64)
        s = tl.arange(0, SPLITS_P2)
        smask = s < SPLITS
        row = (t * SPLITS + s) * H + h
        m = tl.load(part_ml_ptr + row * 2, mask=smask, other=float("-inf"))
        l_ = tl.load(part_ml_ptr + row * 2 + 1, mask=smask, other=0.0)
        mmax = tl.max(m, axis=0)
        msafe = tl.where(mmax == float("-inf"), 0.0, mmax)
        w = tl.where(l_ > 0, tl.exp2(m - msafe), 0.0)
        ltot = tl.sum(l_ * w, axis=0)
        d = tl.arange(0, 512)
        acc = tl.load(part_o_ptr + row[:, None] * 512 + d[None, :], mask=smask[:, None], other=0.0)
        o = tl.sum(acc * w[:, None], axis=0)
        has = ltot > 0
        lsafe = tl.where(has, ltot, 1.0)
        tl.store(out_ptr + t * stride_ot + h * stride_oh + d, (o / lsafe).to(tl.bfloat16))
        tl.store(lse_ptr + t * H + h, tl.where(has, msafe + tl.log2(lsafe), -1e30))

    @triton.jit
    def _rtx_merge_kernel(oa_ptr, stride_at, stride_ah, la_ptr, ob_ptr, stride_bt, stride_bh, lb_ptr,
                          H: tl.constexpr):
        """(oa, la) <- merge((oa, la), (ob, lb)); LSE log2-domain, <= -5e29 means empty."""
        t = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1).to(tl.int64)
        la = tl.load(la_ptr + t * H + h)
        lb = tl.load(lb_ptr + t * H + h)
        m = tl.maximum(la, lb)
        wa = tl.where(la <= -5e29, 0.0, tl.exp2(la - m))
        wb = tl.where(lb <= -5e29, 0.0, tl.exp2(lb - m))
        tot = wa + wb
        has = tot > 0
        safe = tl.where(has, tot, 1.0)
        d = tl.arange(0, 512)
        oa = tl.load(oa_ptr + t * stride_at + h * stride_ah + d).to(tl.float32)
        ob = tl.load(ob_ptr + t * stride_bt + h * stride_bh + d).to(tl.float32)
        o = (oa * wa + ob * wb) / safe
        tl.store(oa_ptr + t * stride_at + h * stride_ah + d, o.to(tl.bfloat16))
        tl.store(la_ptr + t * H + h, tl.where(has, m + tl.log2(safe), -1e30))

    @triton.jit
    def _nsa656_quant_write_kernel(x_ptr, stride_x, loc_ptr, dst_u8_ptr, dst_f32_ptr, n_rows, POW2: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        loc = tl.load(loc_ptr + t).to(tl.int64)
        ok = (loc > 0) & (loc < n_rows)  # loc 0 = CUDA-graph padding slot: never written, stays zero
        loc = tl.where(ok, loc, 0)
        x = tl.load(x_ptr + t * stride_x + tl.arange(0, 512)).to(tl.float32)
        xb = tl.reshape(x, [4, 128])
        amax = tl.max(tl.abs(xb), axis=1)
        s = tl.where(amax > 0, amax / 448.0, 1.0)
        if POW2:  # round the scale up to a power of two by exponent bits (exact)
            bits = s.to(tl.int32, bitcast=True)
            e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
            s = (e << 23).to(tl.float32, bitcast=True)
        inv = 1.0 / s
        codes = tl.clamp(xb * inv[:, None], -448.0, 448.0).to(tl.float8e4nv)
        row = dst_u8_ptr + loc * 656
        tl.store(row + tl.arange(0, 512), tl.reshape(codes, [512]).to(tl.uint8, bitcast=True),
                 mask=ok & (tl.arange(0, 512) >= 0))
        tl.store(dst_f32_ptr + loc * 164 + 128 + tl.arange(0, 4), s, mask=ok & (tl.arange(0, 4) >= 0))
        tl.store(row + 528 + tl.arange(0, 128), tl.zeros([128], tl.uint8), mask=ok & (tl.arange(0, 128) >= 0))


# ------------------------------------------------------------------------------------------------ wrappers


class Workspace:
    """Grow-only fp32 scratch for split-K partials. Pass a slice of the DSA backend's workspace_buffer (uint8) to
    stay inside memory the pool sizing already accounted for; CUDA-graph safe once sized (no reallocation in replay)."""

    def __init__(self, buf: torch.Tensor | None = None):
        self.buf = buf

    def get(self, n_floats: int, device) -> torch.Tensor:
        need = n_floats * 4
        if self.buf is None or self.buf.numel() * self.buf.element_size() < need:
            self.buf = torch.empty(need, dtype=torch.uint8, device=device)
        return self.buf.view(torch.uint8)[:need].view(torch.float32)


_default_ws = Workspace()


def sparse_mla(q: torch.Tensor, kv_buffer: torch.Tensor, idx: torch.Tensor, sm_scale: float, fmt: int,
               out: torch.Tensor | None = None, lse: torch.Tensor | None = None, splits: int | None = None,
               block_n: int = 32, block_h: int = 16, num_warps: int = 8, num_stages: int = 2,
               workspace: Workspace | None = None, num_sms: int | None = None):
    """Sparse MLA over a DSA pool buffer. q [T, H, 512] (last dim contiguous), idx [T, K] int32 (last dim contiguous).
    Returns (out [T, H, 512] bf16, lse [T, H] fp32 log2 domain)."""
    assert HAVE_TRITON
    t, h, dq = q.shape
    assert dq == D and q.stride(-1) == 1, (q.shape, q.stride())
    assert idx.dim() == 2 and idx.shape[0] == t and idx.stride(-1) == 1 and idx.dtype == torch.int32
    rows = pool_rows(kv_buffer, fmt)
    k = idx.shape[1]
    if out is None:
        out = torch.empty(t, h, D, dtype=torch.bfloat16, device=q.device)
    if lse is None:
        lse = torch.empty(t, h, dtype=torch.float32, device=q.device)
    if t == 0:
        return out, lse
    assert out.stride(-1) == 1 and lse.is_contiguous()
    kv_f32 = rows.view(torch.float32) if fmt in FP8_FMTS else rows
    if num_sms is None:
        num_sms = device_sms(q.device)
    n_tiles = max(1, triton_cdiv(k, block_n))
    if splits is None:
        splits = choose_splits(t, h, k, block_h, block_n, num_sms)
    splits = max(1, min(int(splits), n_tiles))
    tps = triton_cdiv(n_tiles, splits)
    splits = triton_cdiv(n_tiles, tps)  # no empty split
    hb = triton_cdiv(h, block_h)
    dot_f32 = interpret_mode()
    if splits == 1:
        part_o = part_ml = lse  # unused
    else:
        ws = workspace or _default_ws
        buf = ws.get(t * splits * h * (D + 2), q.device)
        part_o = buf[: t * splits * h * D]
        part_ml = buf[t * splits * h * D:]
    if fmt == FMT_FP4_288:
        _rtx_sparse_mla_fp4_kernel[(t, hb, splits)](
            q, q.stride(0), q.stride(1), rows, rows.shape[0], idx, idx.stride(0), k,
            out, out.stride(0), out.stride(1), lse, part_o, part_ml, float(sm_scale) * LOG2E,
            H=h, BLOCK_H=block_h, BLOCK_N=block_n, SPLITS=splits, TILES_PER_SPLIT=tps,
            DOT_F32=dot_f32, NUM_STAGES=num_stages, num_warps=num_warps)
    else:
        _rtx_sparse_mla_kernel[(t, hb, splits)](
            q, q.stride(0), q.stride(1), rows, kv_f32, rows.shape[0], idx, idx.stride(0), k,
            out, out.stride(0), out.stride(1), lse, part_o, part_ml, float(sm_scale) * LOG2E,
            H=h, FMT=fmt, BLOCK_H=block_h, BLOCK_N=block_n, SPLITS=splits, TILES_PER_SPLIT=tps,
            DOT_F32=dot_f32, NUM_STAGES=num_stages, ROW_B=ROW_BYTES[fmt], num_warps=num_warps)
    if splits > 1:
        _rtx_combine_kernel[(t, h)](part_o, part_ml, out, out.stride(0), out.stride(1), lse, H=h, SPLITS=splits,
                                    SPLITS_P2=triton.next_power_of_2(splits), num_warps=4)
    return out, lse


def merge_attn(out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor) -> None:
    """In place: (out_a, lse_a) = merge of the two partial attentions (log2 LSE)."""
    t, h, dd = out_a.shape
    assert dd == D and out_b.shape == out_a.shape and lse_a.is_contiguous() and lse_b.is_contiguous()
    if t == 0:
        return
    _rtx_merge_kernel[(t, h)](out_a, out_a.stride(0), out_a.stride(1), lse_a, out_b, out_b.stride(0),
                              out_b.stride(1), lse_b, H=h, num_warps=4)


def nsa656_quant_write(kv_buffer: torch.Tensor, loc: torch.Tensor, latent: torch.Tensor, pow2: bool = False) -> None:
    """latent [T, 512] (bf16/fp16/fp32) -> 656-byte rows at pool[loc[t]] (loc <= 0 or out of range: skipped).
    pow2: power-of-two scales (see nsa656_quantize_ref)."""
    rows = pool_rows(kv_buffer, FMT_FP8_656)
    x = latent.reshape(latent.shape[0], D)
    if x.shape[0] == 0:
        return
    if x.stride(-1) != 1:
        x = x.contiguous()
    assert loc.numel() == x.shape[0], (loc.shape, x.shape)
    loc = loc.reshape(-1)
    _nsa656_quant_write_kernel[(x.shape[0],)](x, x.stride(0), loc, rows, rows.view(torch.float32), rows.shape[0],
                                              POW2=bool(pow2), num_warps=4)


def bytes_per_token(fmt: int, layers: int = 11, indexer_bytes: float = 0.0) -> float:
    """DSA cache bytes per token (all DSA layers) for a row format, plus the indexer cache if given."""
    return layers * ROW_BYTES[fmt] + indexer_bytes


def decode_bytes_per_query(fmt: int, k: int = 2051, layers: int = 11) -> float:
    """Derived bytes one decode query token reads from the DSA rows (all layers), top-k rows once per head block."""
    return layers * k * ROW_BYTES[fmt]


def roofline_us(fmt: int, n_query: int, hbm_tb_s: float = 1.792, k: int = 2051, layers: int = 11) -> float:
    """Derived byte floor (us) of the DSA row reads for n_query query tokens per step on one GPU (GDDR7 1.792 TB/s
    = RTX PRO 6000 spec 28 Gbps x 512 bit)."""
    return decode_bytes_per_query(fmt, k, layers) * n_query / (hbm_tb_s * 1e12) * 1e6


if __name__ == "__main__":  # quick derived table
    for f in (FMT_BF16, FMT_FP8_656, FMT_FP4_288):
        print(FMT_NAMES[f], ROW_BYTES[f], "B/row;", f"{roofline_us(f, 64):.0f} us/step at 64 decode tokens,",
              f"{roofline_us(f, 256):.0f} us at 256 (MTP3 verify x64)")
    print("sm120 smem, BLOCK_H 16 x BLOCK_N 32:", (16 * 512 * 2 + 2 * 32 * 512 * 2) // 1024, "KB of 99")
    _ = math
