"""4-bit (NVFP4-style) KV cache rows for GLM-5.3's DSA latent cache: layout, Triton kernels, torch reference.

Served GLM-5.3 (SGLang v0.5.20, SM100, `trtllm` DSA roles) stores one 576-byte FP8 row per token per layer:
512 E4M3 latent (nope) values followed by 64 E4M3 rope values. This module defines the 352-byte row that replaces it:

    [0, 256)    256 bytes: 512 E2M1 nibbles of the latent, element 2i in the low nibble of byte i
    [256, 288)   32 bytes: one E4M3 scale per block of 16 latent values (block b covers elements 16b .. 16b+15)
    [288, 352)   64 bytes: the rope part, unchanged E4M3

Quantization (per block of 16): scale = round_e4m3(clamp(amax / 6, 2^-9, 448)); code = e2m1(x / scale) with the
midpoints 0.25 0.75 1.25 1.75 2.5 3.5 5 rounding down on a tie and magnitudes above 6 saturating; the sign bit is
set only on a nonzero magnitude (no negative zero). Dequantization
is exactly value(code) * scale, then a cast to E4M3 (for the FP8 scratch the TRT-LLM op reads) or bf16.

Kernels (Triton, one row = one token; every shape is a power of two so tiles reshape without reordering):
  quant_write     bf16 (or fp8) nope [T, 512] + fp8 (or bf16) rope [T, 64] -> pool[loc[t]] (the pool write door)
  dequant_rows    pool rows at idx[r] (-1 skipped) -> dst[r] in the 576-byte FP8 layout, or bf16 [R, 576]
  dequant_seq     every position p < seq_len[b] of request b: pool[page_table[b, p]] -> dst[base[b] + p] and
                  slot_to_pos[page_table[b, p]] = base[b] + p (the prefill sequence scratch)
  remap_table     out[t, k] = slot_to_pos[table[t, k]] (or -1)
  dequant_rows_dedup  selected rows with duplicates (a prefill chunk's tokens or a request's draft tokens pick
                  mostly the same rows): a stamp array (atomic_min of the flat index per slot) elects one owner per
                  slot, the owner list is compacted (count + cumsum + in-block scan) into COMPACT scratch rows, only
                  owners are dequantized, the table is remapped to those rows, the stamps are cleared; fixed-shape
                  passes, CUDA-graph safe; the scratch needs only as many rows as distinct slots (<= the context)
  sparse_mla      fused sparse MLA attention over the FP4 pool: q [T, H, 576] fp8, table [T, K] of pool slots,
                  out [T, H, 512] bf16; dequantizes each selected row in-kernel (no scratch), flash-decoding split
                  over the selected rows plus a combine kernel
Dequantization in the kernels goes through a 256-entry byte LUT (both nibbles -> exact bf16 bits) and one multiply.

The torch reference (`quantize_rows_ref`, `dequantize_rows_ref`, `sparse_mla_ref`) runs on CPU and is what the
tests compare the kernels against bit for bit (quant/dequant) or within tolerance (attention).

GLM-5.3 numbers: kv_lora_rank 512, rope 64, 64 local heads, top-2048; 78 layers -> 352 * 78 = 27,456 B/token
versus 44,928 (derived from the layout, before the indexer cache which is unchanged).
"""
from __future__ import annotations

import torch

DIM_NOPE = 512
DIM_ROPE = 64
DIM_QK = DIM_NOPE + DIM_ROPE  # 576
BLOCK = 16
N_BLOCKS = DIM_NOPE // BLOCK  # 32
NOPE_PACKED_BYTES = DIM_NOPE // 2  # 256
SCALE_OFF = NOPE_PACKED_BYTES  # 256
ROPE_OFF = SCALE_OFF + N_BLOCKS  # 288
ROW_BYTES = ROPE_OFF + DIM_ROPE  # 352
FP8_ROW_BYTES = DIM_QK  # 576, the raw layout TRT-LLM reads
E2M1_MAX = 6.0
SCALE_MIN = 2.0 ** -9  # smallest E4M3 subnormal
SCALE_MAX = 448.0
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_LUTS: dict = {}


def byte_lut(device) -> torch.Tensor:
    """int32 [256]: for byte b, bf16 bits of value(high nibble) << 16 | bf16 bits of value(low nibble). Every E2M1
    value is exact in bf16, so (lut << 16) and (lut & 0xFFFF0000) reinterpreted as fp32 are the exact pair."""
    key = str(device)
    lut = _LUTS.get(key)
    if lut is None:
        codes = torch.arange(256)
        mag = E2M1_VALUES[codes & 7]
        lo = torch.where(((codes & 8) != 0) & (mag != 0), -mag, mag)
        hmag = E2M1_VALUES[(codes >> 4) & 7]
        hi = torch.where(((codes >> 4) & 8 != 0) & (hmag != 0), -hmag, hmag)
        lo_bits = lo.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        hi_bits = hi.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        lut = ((hi_bits << 16) | lo_bits).to(torch.int32).to(device)
        _LUTS[key] = lut
    return lut

# ------------------------------------------------------------------------------------------------ torch reference


def _e2m1_codes_ref(y: torch.Tensor) -> torch.Tensor:
    """Magnitude codes 0..7 for y >= 0 (fp32): strict midpoint comparisons, ties round down, saturate at 6."""
    c = torch.zeros_like(y, dtype=torch.int32)
    for m in E2M1_MIDPOINTS.tolist():
        c += (y > m).to(torch.int32)
    return c


def quantize_rows_ref(nope: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """nope [T, 512] (bf16/fp32/fp8), rope [T, 64] (bf16/fp32/fp8) -> uint8 rows [T, 352]. CPU or GPU."""
    t = nope.shape[0]
    x = nope.float().reshape(t, N_BLOCKS, BLOCK)
    amax = x.abs().amax(dim=-1)
    s = (amax / E2M1_MAX).clamp(min=SCALE_MIN, max=SCALE_MAX)
    s8 = s.to(torch.float8_e4m3fn)
    s_eff = s8.float()
    inv = torch.where(s_eff > 0, 1.0 / s_eff, torch.zeros_like(s_eff))
    y = (x.abs() * inv.unsqueeze(-1)).clamp(max=E2M1_MAX)
    codes = _e2m1_codes_ref(y)
    codes = codes | (((x < 0) & (codes > 0)).to(torch.int32) << 3)  # no negative zero: code 8 is never written
    codes = codes.reshape(t, DIM_NOPE)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    out = torch.empty(t, ROW_BYTES, dtype=torch.uint8, device=nope.device)
    out[:, :SCALE_OFF] = packed
    out[:, SCALE_OFF:ROPE_OFF] = s8.view(torch.uint8)
    out[:, ROPE_OFF:] = rope.to(torch.float8_e4m3fn).view(torch.uint8)
    return out


def dequantize_rows_ref(rows: torch.Tensor, out_dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    """uint8 rows [R, 352] -> [R, 576] in out_dtype (fp8 = the TRT-LLM raw layout; bf16/fp32 for references)."""
    r = rows.shape[0]
    packed = rows[:, :SCALE_OFF].to(torch.int32)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(r, DIM_NOPE)
    mag = E2M1_VALUES.to(rows.device)[codes & 7]
    val = torch.where(((codes & 8) != 0) & (mag != 0), -mag, mag)
    s = rows[:, SCALE_OFF:ROPE_OFF].view(torch.float8_e4m3fn).float()
    nope = (val.reshape(r, N_BLOCKS, BLOCK) * s.unsqueeze(-1)).reshape(r, DIM_NOPE)
    rope = rows[:, ROPE_OFF:].view(torch.float8_e4m3fn).float()
    out = torch.cat([nope, rope], dim=-1)
    return out.to(out_dtype)


def sparse_mla_ref(q: torch.Tensor, rows: torch.Tensor, table: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """fp32 reference: q [T, H, 576] (any float dtype), rows [N, 352] uint8 pool, table [T, K] slots (-1 masked).
    Returns [T, H, 512] fp32. Rows with no valid slot give zeros."""
    t, h, _ = q.shape
    kv = dequantize_rows_ref(rows, torch.float32)  # [N, 576]
    out = torch.zeros(t, h, DIM_NOPE, dtype=torch.float32, device=q.device)
    qf = q.float()
    for i in range(t):
        idx = table[i].long()
        valid = idx >= 0
        if not bool(valid.any()):
            continue
        k = kv[idx[valid]]  # [k, 576]
        s = qf[i] @ k.T * sm_scale  # [H, k]
        p = torch.softmax(s, dim=-1)
        out[i] = p @ k[:, :DIM_NOPE]
    return out


# ------------------------------------------------------------------------------------------------ Triton kernels

try:  # the CPU-only host runs the references and the CPU tests without triton
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _decode_pairs(lut_ptr, b):
        """bytes [..] int32 -> (low-nibble values, high-nibble values) fp32, exact, via the 256-entry byte LUT."""
        pv = tl.load(lut_ptr + b)
        lo = (pv << 16).to(tl.float32, bitcast=True)
        hi = ((pv >> 16) << 16).to(tl.float32, bitcast=True)
        return lo, hi

    @triton.jit
    def _e2m1_encode(x, inv):
        """x fp32, inv = 1/scale (0 for a zero scale) -> codes 0..15 (sign in bit 3), ties round down, saturate."""
        y = tl.minimum(tl.abs(x) * inv, 6.0)
        c = (y > 0.25).to(tl.int32) + (y > 0.75).to(tl.int32) + (y > 1.25).to(tl.int32) + (y > 1.75).to(tl.int32)
        c += (y > 2.5).to(tl.int32) + (y > 3.5).to(tl.int32) + (y > 5.0).to(tl.int32)
        return c | tl.where((x < 0) & (c > 0), 8, 0)

    @triton.jit
    def _kv4_quant_write_kernel(nope_ptr, nope_stride, rope_ptr, rope_stride, loc_ptr, dst_ptr, n_pool,
                                NOPE_FP8: tl.constexpr, ROPE_FP8: tl.constexpr):
        t = tl.program_id(0)
        loc = tl.load(loc_ptr + t).to(tl.int64)
        ok = (loc >= 0) & (loc < n_pool)
        loc = tl.where(ok, loc, 0)
        offs = tl.arange(0, 512)
        if NOPE_FP8:
            x = tl.load(nope_ptr + t.to(tl.int64) * nope_stride + offs).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            x = tl.load(nope_ptr + t.to(tl.int64) * nope_stride + offs).to(tl.float32)
        xb = tl.reshape(x, [32, 16])
        amax = tl.max(tl.abs(xb), axis=1)  # [32]
        s = tl.minimum(tl.maximum(amax / 6.0, 0.001953125), 448.0)
        s8 = s.to(tl.float8e4nv)
        s_eff = s8.to(tl.float32)
        inv = tl.where(s_eff > 0, 1.0 / s_eff, 0.0)
        inv_b = tl.broadcast_to(inv[:, None], [32, 16])
        codes = _e2m1_encode(xb, inv_b)  # [32, 16] int32
        pairs = tl.reshape(codes, [256, 2])
        lo, hi = tl.split(pairs)
        byte = (lo | (hi << 4)).to(tl.uint8)
        row = dst_ptr + tl.multiple_of(loc * 352, 16)
        poffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 256), 16), 256)
        tl.store(row + poffs, byte, mask=ok & (poffs >= 0))
        soffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 16), 32)
        tl.store(row + 256 + soffs, s8.to(tl.uint8, bitcast=True), mask=ok & (soffs >= 0))
        roffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
        if ROPE_FP8:
            r8 = tl.load(rope_ptr + t.to(tl.int64) * rope_stride + roffs)
        else:
            r8 = tl.load(rope_ptr + t.to(tl.int64) * rope_stride + roffs).to(tl.float32).to(tl.float8e4nv).to(
                tl.uint8, bitcast=True)
        tl.store(row + 288 + roffs, r8, mask=ok & (roffs >= 0))

    @triton.jit
    def _kv4_dequant_row_tile(src_ptr, lut_ptr, slot, valid, dst_ptr, dst_row, OUT_BF16: tl.constexpr):
        """slot, valid, dst_row: [R] vectors. Writes 576 values per valid row (fp8 bytes or bf16)."""
        # rows are 352 B (16 x 22) and 576 B (16 x 36): every row start is 16-byte aligned, which the hints below
        # tell Triton so the byte loads and stores vectorize (measured: 3x on the dequant gather without them)
        row_off = tl.multiple_of(tl.where(valid, slot, 0).to(tl.int64) * 352, 16)
        src = src_ptr + row_off[:, None]
        pair = tl.max_contiguous(tl.multiple_of(tl.arange(0, 256), 16), 256)
        b = tl.load(src + pair[None, :], mask=valid[:, None], other=0).to(tl.int32)  # [R, 256]
        soffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 16), 32)
        sc = tl.load(src + 256 + soffs[None, :], mask=valid[:, None], other=0).to(
            tl.float8e4nv, bitcast=True).to(tl.float32)  # [R, 32]
        R: tl.constexpr = b.shape[0]
        sc_pair = tl.reshape(tl.broadcast_to(sc[:, :, None], [R, 32, 8]), [R, 256])
        lo, hi = _decode_pairs(lut_ptr, b)
        v = tl.reshape(tl.join(lo * sc_pair, hi * sc_pair), [R, 512])
        roffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
        rope = tl.load(src + 288 + roffs[None, :], mask=valid[:, None], other=0)
        dst = dst_ptr + tl.multiple_of(dst_row.to(tl.int64) * 576, 16)[:, None]
        doffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 16), 512)
        if OUT_BF16:
            tl.store(dst + doffs[None, :], v.to(tl.bfloat16), mask=valid[:, None])
            tl.store(dst + 512 + roffs[None, :],
                     rope.to(tl.float8e4nv, bitcast=True).to(tl.float32).to(tl.bfloat16), mask=valid[:, None])
        else:
            tl.store(dst + doffs[None, :], v.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=valid[:, None])
            tl.store(dst + 512 + roffs[None, :], rope, mask=valid[:, None])

    @triton.jit
    def _kv4_dequant_rows_kernel(src_ptr, lut_ptr, idx_ptr, n_rows, n_pool, dst_ptr, OUT_BF16: tl.constexpr,
                                 BLOCK_R: tl.constexpr):
        r = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < n_rows
        slot = tl.load(idx_ptr + r, mask=rmask, other=-1).to(tl.int64)
        valid = rmask & (slot >= 0) & (slot < n_pool)
        _kv4_dequant_row_tile(src_ptr, lut_ptr, slot, valid, dst_ptr, r, OUT_BF16)

    @triton.jit
    def _kv4_dequant_seq_kernel(src_ptr, lut_ptr, table_ptr, table_stride, seq_lens_ptr, base_ptr, max_len, n_pool,
                                dst_ptr, slot_to_pos_ptr, BLOCK_R: tl.constexpr, OUT_BF16: tl.constexpr = False):
        b = tl.program_id(0)
        p = tl.program_id(1).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        n = tl.load(seq_lens_ptr + b).to(tl.int64)
        base = tl.load(base_ptr + b).to(tl.int64)
        pmask = p < tl.minimum(n, max_len)
        slot = tl.load(table_ptr + b.to(tl.int64) * table_stride + p, mask=pmask, other=-1).to(tl.int64)
        valid = pmask & (slot >= 0) & (slot < n_pool)
        dst_row = base + p
        _kv4_dequant_row_tile(src_ptr, lut_ptr, slot, valid, dst_ptr, dst_row, OUT_BF16)
        tl.store(slot_to_pos_ptr + tl.where(valid, slot, 0), dst_row.to(tl.int32), mask=valid)

    @triton.jit
    def _kv4_remap_table_kernel(table_ptr, n, slot_to_pos_ptr, n_pool, out_ptr, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        s = tl.load(table_ptr + offs, mask=m, other=-1)
        ok = m & (s >= 0) & (s < n_pool)
        pos = tl.load(slot_to_pos_ptr + tl.where(ok, s, 0).to(tl.int64), mask=ok, other=-1)
        tl.store(out_ptr + offs, tl.where(ok, pos, -1), mask=m)

    @triton.jit
    def _kv4_stamp_kernel(table_ptr, n, n_pool, stamp_ptr, BLOCK: tl.constexpr):
        """stamp[slot] = min(stamp[slot], flat index): the first table entry naming a slot owns its scratch row."""
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        s = tl.load(table_ptr + offs, mask=m, other=-1)
        ok = m & (s >= 0) & (s < n_pool)
        tl.atomic_min(stamp_ptr + tl.where(ok, s, 0).to(tl.int64), offs.to(tl.int32), mask=ok)

    @triton.jit
    def _kv4_owner_count_kernel(table_ptr, n, n_pool, stamp_ptr, counts_ptr, BLOCK: tl.constexpr):
        """counts[block] = number of entries in the block that own their slot (stamp[slot] == flat index)."""
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        s = tl.load(table_ptr + offs, mask=m, other=-1)
        ok = m & (s >= 0) & (s < n_pool)
        owner = tl.load(stamp_ptr + tl.where(ok, s, 0).to(tl.int64), mask=ok, other=-1)
        is_owner = ok & (owner == offs.to(tl.int32))
        tl.store(counts_ptr + pid, tl.sum(is_owner.to(tl.int32), axis=0))

    @triton.jit
    def _kv4_owner_compact_kernel(table_ptr, n, n_pool, stamp_ptr, incl_ptr, owners_ptr, pos_ptr, cap,
                                  BLOCK: tl.constexpr):
        """owners[j] = flat index of the j-th owning entry (packed; incl = inclusive cumsum of counts) and
        pos[flat index] = j for owners with j < cap (their scratch row); owners past the cap get pos -1."""
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        s = tl.load(table_ptr + offs, mask=m, other=-1)
        ok = m & (s >= 0) & (s < n_pool)
        owner = tl.load(stamp_ptr + tl.where(ok, s, 0).to(tl.int64), mask=ok, other=-1)
        is_owner = (ok & (owner == offs.to(tl.int32))).to(tl.int32)
        base = tl.load(incl_ptr + pid) - tl.sum(is_owner, axis=0)
        j = base + tl.cumsum(is_owner, axis=0) - is_owner
        tl.store(owners_ptr + j.to(tl.int64), offs.to(tl.int32), mask=is_owner != 0)
        tl.store(pos_ptr + offs, tl.where(j < cap, j, -1), mask=is_owner != 0)

    @triton.jit
    def _kv4_dequant_owner_kernel(src_ptr, lut_ptr, table_ptr, owners_ptr, n_owners_ptr, cap, n_pool, dst_ptr,
                                  BLOCK_R: tl.constexpr, OUT_BF16: tl.constexpr = False):
        """dst[j] = dequant(pool[table[owners[j]]]) for j < min(n_owners, cap): the compact scratch."""
        n_owners = tl.minimum(tl.load(n_owners_ptr), cap)
        if tl.program_id(0) * BLOCK_R >= n_owners:
            return
        j = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = j < n_owners
        fi = tl.load(owners_ptr + j, mask=m, other=0).to(tl.int64)
        slot = tl.load(table_ptr + fi, mask=m, other=-1).to(tl.int64)
        valid = m & (slot >= 0) & (slot < n_pool)
        _kv4_dequant_row_tile(src_ptr, lut_ptr, slot, valid, dst_ptr, j, OUT_BF16)

    @triton.jit
    def _kv4_remap_stamp_kernel(table_ptr, n, n_pool, stamp_ptr, pos_ptr, out_ptr, BLOCK: tl.constexpr):
        """out[i] = pos[stamp[table[i]]] (the owner's compact scratch row, -1 past the cap) for valid slots."""
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        s = tl.load(table_ptr + offs, mask=m, other=-1)
        ok = m & (s >= 0) & (s < n_pool)
        owner = tl.load(stamp_ptr + tl.where(ok, s, 0).to(tl.int64), mask=ok, other=0)
        row = tl.load(pos_ptr + tl.where(ok, owner, 0).to(tl.int64), mask=ok, other=-1)
        tl.store(out_ptr + offs, tl.where(ok, row, -1), mask=m)

    @triton.jit
    def _kv4_unstamp_kernel(table_ptr, n, n_pool, stamp_ptr, BLOCK: tl.constexpr):
        """stamp[slot] = INT32_MAX for every valid entry: the stamp array is clean again for the next call."""
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        s = tl.load(table_ptr + offs, mask=m, other=-1)
        ok = m & (s >= 0) & (s < n_pool)
        tl.store(stamp_ptr + tl.where(ok, s, 0).to(tl.int64), tl.full([BLOCK], 2147483647, tl.int32), mask=ok)

    @triton.jit
    def _kv4_identity_table_kernel(table_ptr, n, out_ptr, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        s = tl.load(table_ptr + offs, mask=m, other=-1)
        tl.store(out_ptr + offs, tl.where(s >= 0, offs.to(tl.int32), -1), mask=m)

    @triton.jit
    def _kv4_sparse_mla_kernel(q_ptr, q_stride_t, q_stride_h, kv_ptr, lut_ptr, n_pool, table_ptr, table_stride, K,
                               out_ptr, out_stride_t, out_stride_h, part_o_ptr, part_ml_ptr, sm_scale_log2,
                               H: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr,
                               KS: tl.constexpr):
        """One program = one query token x BLOCK_H heads x one split of KS selected rows (flash-decoding). Each
        block of BLOCK_N rows is dequantized once (byte LUT -> exact fp32, x E4M3 scale -> bf16) and used for both
        QK (576 wide) and PV (512 wide). Masked (-1) slots score -inf. SPLITS == 1 writes the normalized output;
        otherwise the unnormalized fp32 accumulator and (m, l) go to the partial buffers for _kv4_combine_kernel."""
        t = tl.program_id(0).to(tl.int64)
        hg = tl.program_id(1)
        sp = tl.program_id(2)
        h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = h < H
        d_nope = tl.arange(0, 512)
        d_rope = tl.arange(0, 64)
        qbase = q_ptr + t * q_stride_t + h.to(tl.int64)[:, None] * q_stride_h
        q_nope = tl.load(qbase + d_nope[None, :], mask=hmask[:, None], other=0).to(
            tl.float8e4nv, bitcast=True).to(tl.bfloat16)  # [BH, 512]
        q_rope = tl.load(qbase + 512 + d_rope[None, :], mask=hmask[:, None], other=0).to(
            tl.float8e4nv, bitcast=True).to(tl.bfloat16)  # [BH, 64]

        m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_H], tl.float32)
        acc = tl.zeros([BLOCK_H, 512], tl.float32)
        pair = tl.max_contiguous(tl.multiple_of(tl.arange(0, 256), 16), 256)
        trow = table_ptr + t * table_stride
        k_end = tl.minimum((sp + 1) * KS, K)
        for start in range(sp * KS, k_end, BLOCK_N):
            j = start + tl.arange(0, BLOCK_N)
            slot = tl.load(trow + j, mask=j < k_end, other=-1).to(tl.int64)
            valid = (slot >= 0) & (slot < n_pool)
            src = kv_ptr + tl.multiple_of(tl.where(valid, slot, 0) * 352, 16)[:, None]
            b = tl.load(src + pair[None, :], mask=valid[:, None], other=0).to(tl.int32)  # [BN, 256]
            sc = tl.load(src + 256 + tl.arange(0, 32)[None, :], mask=valid[:, None], other=0).to(
                tl.float8e4nv, bitcast=True).to(tl.float32)  # [BN, 32]
            sc_pair = tl.reshape(tl.broadcast_to(sc[:, :, None], [BLOCK_N, 32, 8]), [BLOCK_N, 256])
            lo, hi = _decode_pairs(lut_ptr, b)
            k_nope = tl.reshape(tl.join(lo * sc_pair, hi * sc_pair), [BLOCK_N, 512]).to(tl.bfloat16)
            k_rope = tl.load(src + 288 + d_rope[None, :], mask=valid[:, None], other=0).to(
                tl.float8e4nv, bitcast=True).to(tl.bfloat16)  # [BN, 64]
            s = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))  # [BH, BN] fp32
            s = s * sm_scale_log2
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k_nope)
            m_i = m_new
        if SPLITS == 1:
            l_safe = tl.where(l_i > 0, l_i, 1.0)
            o = acc / l_safe[:, None]
            obase = out_ptr + t * out_stride_t + h.to(tl.int64)[:, None] * out_stride_h
            tl.store(obase + d_nope[None, :], o.to(tl.bfloat16), mask=hmask[:, None])
        else:
            pbase = part_o_ptr + ((t * SPLITS + sp) * H + h.to(tl.int64))[:, None] * 512
            tl.store(pbase + d_nope[None, :], acc, mask=hmask[:, None])
            mlbase = part_ml_ptr + ((t * SPLITS + sp) * H + h.to(tl.int64)) * 2
            tl.store(mlbase, m_i, mask=hmask)
            tl.store(mlbase + 1, l_i, mask=hmask)

    @triton.jit
    def _kv4_combine_kernel(part_o_ptr, part_ml_ptr, out_ptr, out_stride_t, out_stride_h,
                            H: tl.constexpr, BLOCK_H: tl.constexpr, SPLITS: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = h < H
        d = tl.arange(0, 512)
        m_all = tl.full([BLOCK_H], float("-inf"), tl.float32)
        for sp in tl.static_range(SPLITS):
            ml = part_ml_ptr + ((t * SPLITS + sp) * H + h.to(tl.int64)) * 2
            m_all = tl.maximum(m_all, tl.load(ml, mask=hmask, other=float("-inf")))
        m_safe = tl.where(m_all == float("-inf"), 0.0, m_all)
        l_all = tl.zeros([BLOCK_H], tl.float32)
        acc = tl.zeros([BLOCK_H, 512], tl.float32)
        for sp in tl.static_range(SPLITS):
            ml = part_ml_ptr + ((t * SPLITS + sp) * H + h.to(tl.int64)) * 2
            m_s = tl.load(ml, mask=hmask, other=float("-inf"))
            l_s = tl.load(ml + 1, mask=hmask, other=0.0)
            w = tl.exp2(m_s - m_safe)
            pbase = part_o_ptr + ((t * SPLITS + sp) * H + h.to(tl.int64))[:, None] * 512
            o_s = tl.load(pbase + d[None, :], mask=hmask[:, None], other=0.0)
            l_all += w * l_s
            acc += w[:, None] * o_s
        l_safe = tl.where(l_all > 0, l_all, 1.0)
        obase = out_ptr + t * out_stride_t + h.to(tl.int64)[:, None] * out_stride_h
        tl.store(obase + d[None, :], (acc / l_safe[:, None]).to(tl.bfloat16), mask=hmask[:, None])


# ------------------------------------------------------------------------------------------------ Python wrappers


def pool_rows(kv_buffer: torch.Tensor) -> torch.Tensor:
    """A pool buffer [N, 1, 352] (uint8 or an fp8 view of it) as uint8 [N, 352]."""
    b = kv_buffer if kv_buffer.dtype == torch.uint8 else kv_buffer.view(torch.uint8)
    return b.reshape(b.shape[0], -1)


def quant_write(kv_buffer: torch.Tensor, loc: torch.Tensor, nope: torch.Tensor, rope: torch.Tensor) -> None:
    """Quantize nope [T, 512] (bf16 or fp8) with rope [T, 64] (fp8 or bf16) into pool rows kv_buffer[loc[t]]."""
    rows = pool_rows(kv_buffer)
    nope = nope.reshape(nope.shape[0], DIM_NOPE)
    rope = rope.reshape(rope.shape[0], DIM_ROPE)
    t = nope.shape[0]
    if t == 0:
        return
    assert rope.shape[0] == t and loc.numel() == t, (nope.shape, rope.shape, loc.shape)
    if not nope.is_contiguous():
        nope = nope.contiguous()
    if not rope.is_contiguous():
        rope = rope.contiguous()
    nope_fp8 = nope.dtype == torch.float8_e4m3fn
    rope_fp8 = rope.dtype == torch.float8_e4m3fn
    assert nope_fp8 or nope.dtype in (torch.bfloat16, torch.float16, torch.float32), nope.dtype
    assert rope_fp8 or rope.dtype in (torch.bfloat16, torch.float16, torch.float32), rope.dtype
    _kv4_quant_write_kernel[(t,)](
        nope.view(torch.uint8) if nope_fp8 else nope, nope.stride(0),
        rope.view(torch.uint8) if rope_fp8 else rope, rope.stride(0),
        loc, rows, rows.shape[0], NOPE_FP8=nope_fp8, ROPE_FP8=rope_fp8, num_warps=4)


def dequant_rows(kv_buffer: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, block_r: int = 16,
                 num_warps: int = 4) -> None:
    """dst[r] = dequant(pool[idx[r]]) for idx[r] >= 0. dst: [R, 576] float8_e4m3fn/uint8 (raw layout) or bf16."""
    rows = pool_rows(kv_buffer)
    idx = idx.reshape(-1)
    n = idx.numel()
    if n == 0:
        return
    assert dst.shape[0] >= n and dst.shape[-1] == DIM_QK and dst.is_contiguous(), (dst.shape, n)
    out_bf16 = dst.dtype == torch.bfloat16
    d = dst if out_bf16 else dst.view(torch.uint8)
    _kv4_dequant_rows_kernel[(triton.cdiv(n, block_r),)](rows, byte_lut(rows.device), idx, n, rows.shape[0], d,
                                                        OUT_BF16=out_bf16, BLOCK_R=block_r, num_warps=num_warps)


def dequant_seq(kv_buffer: torch.Tensor, page_table: torch.Tensor, seq_lens: torch.Tensor, bases: torch.Tensor,
                max_len: int, dst: torch.Tensor, slot_to_pos: torch.Tensor, block_r: int = 16,
                num_warps: int = 4) -> None:
    """For each request b and position p < min(seq_lens[b], max_len): dst[bases[b] + p] = dequant(pool[table[b, p]])
    and slot_to_pos[table[b, p]] = bases[b] + p. dst [rows, 576] fp8/uint8 (raw layout) or bf16 (exact values, the
    Hopper bf16 scratch); slot_to_pos int32 [pool size]."""
    rows = pool_rows(kv_buffer)
    b = page_table.shape[0]
    if b == 0 or max_len == 0:
        return
    assert page_table.dtype == torch.int32 and dst.shape[-1] == DIM_QK and dst.is_contiguous()
    assert slot_to_pos.dtype == torch.int32 and slot_to_pos.numel() >= rows.shape[0]
    out_bf16 = dst.dtype == torch.bfloat16
    _kv4_dequant_seq_kernel[(b, triton.cdiv(max_len, block_r))](
        rows, byte_lut(rows.device), page_table, page_table.stride(0), seq_lens, bases, max_len, rows.shape[0],
        dst if out_bf16 else _raw_u8(dst), slot_to_pos, BLOCK_R=block_r, OUT_BF16=out_bf16,
        num_warps=num_warps)


def _raw_u8(dst: torch.Tensor) -> torch.Tensor:
    """An FP8 (or uint8) raw-layout scratch as uint8; anything else is a caller bug."""
    assert dst.dtype in (torch.uint8, torch.float8_e4m3fn), dst.dtype
    return dst if dst.dtype == torch.uint8 else dst.view(torch.uint8)


def remap_table(table: torch.Tensor, slot_to_pos: torch.Tensor, n_pool: int, out: torch.Tensor) -> None:
    """out[t, k] = slot_to_pos[table[t, k]] for valid slots, -1 elsewhere (same shape, int32, contiguous)."""
    n = table.numel()
    if n == 0:
        return
    assert table.is_contiguous() and out.is_contiguous() and out.shape == table.shape
    block = 1024
    _kv4_remap_table_kernel[(triton.cdiv(n, block),)](table, n, slot_to_pos, n_pool, out, BLOCK=block, num_warps=4)


def identity_table(table: torch.Tensor, out: torch.Tensor) -> None:
    """out[t, k] = t * K + k where table[t, k] >= 0, else -1: the block table for a scratch gathered in table order."""
    n = table.numel()
    if n == 0:
        return
    assert table.is_contiguous() and out.is_contiguous() and out.shape == table.shape
    block = 1024
    _kv4_identity_table_kernel[(triton.cdiv(n, block),)](table, n, out, BLOCK=block, num_warps=4)


STAMP_CLEAN = 2147483647


def dequant_rows_dedup(kv_buffer: torch.Tensor, table: torch.Tensor, stamp: torch.Tensor, dst: torch.Tensor,
                       out: torch.Tensor, block_r: int = 8, num_warps: int = 4, work: tuple | None = None) -> torch.Tensor:
    """Like dequant_rows(table) + identity_table, but every DISTINCT slot named by `table` [T, K] is dequantized once
    into a compact scratch row j (in flat-index order of its first entry), and every entry naming it is remapped:
    out[t, k] = j or -1. `stamp` is an int32 [pool size] array holding STAMP_CLEAN between calls. Passes (fixed
    shapes, no host sync: CUDA-graph safe): stamp (atomic_min), count owners per 1024 entries, cumsum, compact the
    owner list + pos[flat] = j, dequantize the owners only (programs past the owner count exit at once), remap,
    unstamp. The scratch needs only min(distinct rows, dst rows) rows: distinct rows never exceed the batch's total
    context, so a 2048-token prefill chunk (4M entries) fits a context-sized scratch. Owners past dst.shape[0] are
    dropped (their entries become -1): the caller sizes dst >= the largest context (`overflow()` reports drops).
    `work` = (counts [>= T*K/1024], owners [>= T*K], pos [>= T*K]) int32 buffers, else allocated here.
    Returns the inclusive cumsum tensor (its last element = number of distinct rows, on the device)."""
    rows = pool_rows(kv_buffer)
    n = table.numel()
    assert table.is_contiguous() and out.is_contiguous() and out.shape == table.shape and table.dtype == torch.int32
    assert stamp.dtype == torch.int32 and stamp.numel() >= rows.shape[0]
    flat = table.view(-1)
    n_pool = rows.shape[0]
    block = 1024
    nblk = max(1, triton.cdiv(n, block))
    if work is None:
        counts = torch.empty(nblk, dtype=torch.int32, device=table.device)
        owners = torch.empty(max(n, 1), dtype=torch.int32, device=table.device)
        pos = torch.empty(max(n, 1), dtype=torch.int32, device=table.device)
    else:
        counts, owners, pos = work
        assert counts.numel() >= nblk and owners.numel() >= n and pos.numel() >= n
        counts = counts[:nblk]
    if n == 0:
        return torch.zeros(1, dtype=torch.int32, device=table.device)
    cap = dst.shape[0]
    assert dst.shape[-1] == DIM_QK and dst.is_contiguous(), dst.shape
    out_bf16 = dst.dtype == torch.bfloat16
    _kv4_stamp_kernel[(nblk,)](flat, n, n_pool, stamp, BLOCK=block, num_warps=4)
    _kv4_owner_count_kernel[(nblk,)](flat, n, n_pool, stamp, counts, BLOCK=block, num_warps=4)
    incl = torch.cumsum(counts, 0, dtype=torch.int32)
    _kv4_owner_compact_kernel[(nblk,)](flat, n, n_pool, stamp, incl, owners, pos, cap, BLOCK=block, num_warps=4)
    _kv4_dequant_owner_kernel[(triton.cdiv(min(n, cap), block_r),)](
        rows, byte_lut(rows.device), flat, owners, incl[nblk - 1:], cap, n_pool, dst if out_bf16 else _raw_u8(dst),
        BLOCK_R=block_r, OUT_BF16=out_bf16, num_warps=num_warps)
    _kv4_remap_stamp_kernel[(nblk,)](flat, n, n_pool, stamp, pos, out, BLOCK=block, num_warps=4)
    _kv4_unstamp_kernel[(nblk,)](flat, n, n_pool, stamp, BLOCK=block, num_warps=4)
    return incl[nblk - 1:]


def sparse_mla(q: torch.Tensor, kv_buffer: torch.Tensor, table: torch.Tensor, sm_scale: float,
               out: torch.Tensor | None = None, block_h: int = 32, block_n: int = 64, num_warps: int = 8,
               num_stages: int = 2, splits: int = 8, workspace: tuple | None = None) -> torch.Tensor:
    """Fused sparse MLA over the FP4 pool. q [T, H, 576] fp8 (or bf16), table [T, K] int32 slots (-1 masked).
    splits > 1 runs flash-decoding over the K selected rows (partials fp32 [T, S, H, 512] + (m, l) [T, S, H, 2],
    allocated here unless `workspace` supplies them). Returns out [T, H, 512] bf16."""
    rows = pool_rows(kv_buffer)
    t, h, dqk = q.shape
    assert dqk == DIM_QK, q.shape
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
    if not q.is_contiguous():
        q = q.contiguous()
    if out is None:
        out = torch.empty(t, h, DIM_NOPE, dtype=torch.bfloat16, device=q.device)
    if t == 0:
        return out
    assert table.dtype == torch.int32 and table.is_contiguous() and table.shape[0] == t
    # block_h 64 (whole GLM head set in one program) compiles under Triton 3.7.1 but faults with a misaligned
    # address on B200 (measured); 16 and 32 are correct and are what the timing table compares
    assert block_h in (16, 32) and block_n in (16, 32, 64), (block_h, block_n)
    k = table.shape[1]
    splits = max(1, min(splits, triton.cdiv(k, block_n)))
    ks = triton.cdiv(triton.cdiv(k, splits), block_n) * block_n
    splits = triton.cdiv(k, ks)
    sm_scale_log2 = float(sm_scale) * 1.4426950408889634
    lut = byte_lut(rows.device)
    grid = (t, triton.cdiv(h, block_h), splits)
    if splits == 1:
        _kv4_sparse_mla_kernel[grid](q.view(torch.uint8), q.stride(0), q.stride(1), rows, lut, rows.shape[0], table,
                                     table.stride(0), k, out, out.stride(0), out.stride(1), out, out, sm_scale_log2,
                                     H=h, BLOCK_H=block_h, BLOCK_N=block_n, SPLITS=1, KS=ks, num_warps=num_warps,
                                     num_stages=num_stages)
        return out
    if workspace is None:
        part_o = torch.empty(t, splits, h, DIM_NOPE, dtype=torch.float32, device=q.device)
        part_ml = torch.empty(t, splits, h, 2, dtype=torch.float32, device=q.device)
    else:
        part_o, part_ml = workspace
        assert part_o.shape[0] >= t and part_o.shape[1] == splits and part_o.shape[2] == h, part_o.shape
        assert part_ml.shape[0] >= t and part_ml.shape[1] == splits and part_ml.shape[2] == h, part_ml.shape
    _kv4_sparse_mla_kernel[grid](q.view(torch.uint8), q.stride(0), q.stride(1), rows, lut, rows.shape[0], table,
                                 table.stride(0), k, out, out.stride(0), out.stride(1), part_o, part_ml,
                                 sm_scale_log2, H=h, BLOCK_H=block_h, BLOCK_N=block_n, SPLITS=splits, KS=ks,
                                 num_warps=num_warps, num_stages=num_stages)
    _kv4_combine_kernel[(t, triton.cdiv(h, block_h))](part_o, part_ml, out, out.stride(0), out.stride(1), H=h,
                                                      BLOCK_H=block_h, SPLITS=splits, num_warps=4)
    return out
