"""Mixed FP8 + FP4 KV rows for GLM-5.3's DSA latent cache: the kernels that let one engine hold both precisions.

Row formats are those of glm_kv4.py: the 576-byte FP8 raw row (512 E4M3 latent + 64 E4M3 rope) the TRT-LLM sparse
MLA op reads, and the 352-byte FP4 row (256 packed E2M1 + 32 E4M3 block scales + 64 E4M3 rope). This module adds
the pieces that address both formats from one unified slot space:

    slot s < B            FP8 row s of the FP8 pool (B = N8 + page_size; page 0 is the dummy page)
    slot s >= B           FP4 row (s - N8) of the FP4 pool (FP4 rows [0, page_size) are the FP4 dummy page)

Kernels (Triton; every row start is 16-byte aligned: 352 = 16 x 22, 576 = 16 x 36):
  write_mixed        the pool write door: per token, FP8 bytes into the FP8 pool or an FP4 quantization (from the
                     bf16 latent when available, else from the FP8 bytes) into the FP4 pool, chosen by slot
  redirect_gather    decode / verify / draft-extend read path: table entries [T, K] of unified slots -> a table the
                     TRT-LLM op reads against the FP8 pool view; FP4 entries are dequantized into a fixed FP8 scratch
                     (row = flat entry index) and rewritten to the scratch row; FP8 entries copy through (fixed
                     shapes, no host sync: CUDA-graph safe)
  seq_dequant        prefill read path: for the requests of a batch that hold FP4 rows, every FP4 position of the
                     request's sequence is dequantized once into the scratch and slot_to_pos records where
  redirect_seq       rewrites a prefill table [T, K] through slot_to_pos (FP8 entries copy through)
  promote_rows       requantize FP8 rows -> FP4 rows for all layers in one launch (promotion of a request's pages)
  move_rows          move_kv_cache for the mixed pool: any of the four format combinations per row, all layers
                     (same-format moves copy bytes; FP8 -> FP4 quantizes; FP4 -> FP8 dequantizes to E4M3)
  gather_bf16        rows at unified slots -> bf16 [T, 576] (the dense MHA prefill path reads cached prefixes)
  index_move         token-granular move of the DSA indexer cache (page-major buffer), all layers

Torch references (CPU) for each: write_mixed_ref, redirect_gather_ref, promote_ref, move_ref, seq_ref.
"""
from __future__ import annotations

import torch

try:  # installed next to glm_kv4.py under sglang/srt/layers/attention/dsa/
    from sglang.srt.layers.attention.dsa import glm_kv4 as kv4
except Exception:  # repository checkout / CPU host
    try:
        from operations.kernels import glm_kv4 as kv4  # type: ignore
    except Exception:
        import glm_kv4 as kv4  # type: ignore

DIM_NOPE = kv4.DIM_NOPE
DIM_ROPE = kv4.DIM_ROPE
DIM_QK = kv4.DIM_QK
ROW_BYTES = kv4.ROW_BYTES  # 352
FP8_ROW_BYTES = kv4.FP8_ROW_BYTES  # 576
INDEX_K_BYTES = 128
INDEX_S_BYTES = 4
HAVE_TRITON = kv4.HAVE_TRITON

if HAVE_TRITON:
    import triton
    import triton.language as tl

    _e2m1_encode = kv4._e2m1_encode
    _decode_pairs = kv4._decode_pairs
    _kv4_dequant_row_tile = kv4._kv4_dequant_row_tile

    @triton.jit
    def _quant_store_fp4(x, rope8, dst_row_ptr, ok):
        """x [512] fp32 latent, rope8 [64] uint8 (E4M3 bytes) -> the 352-byte FP4 row at dst_row_ptr (masked by ok)."""
        xb = tl.reshape(x, [32, 16])
        amax = tl.max(tl.abs(xb), axis=1)
        s = tl.minimum(tl.maximum(amax / 6.0, 0.001953125), 448.0)
        s8 = s.to(tl.float8e4nv)
        s_eff = s8.to(tl.float32)
        inv = tl.where(s_eff > 0, 1.0 / s_eff, 0.0)
        codes = _e2m1_encode(xb, tl.broadcast_to(inv[:, None], [32, 16]))
        lo, hi = tl.split(tl.reshape(codes, [256, 2]))
        byte = (lo | (hi << 4)).to(tl.uint8)
        poffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 256), 16), 256)
        tl.store(dst_row_ptr + poffs, byte, mask=ok & (poffs >= 0))
        soffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 16), 32)
        tl.store(dst_row_ptr + 256 + soffs, s8.to(tl.uint8, bitcast=True), mask=ok & (soffs >= 0))
        roffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
        tl.store(dst_row_ptr + 288 + roffs, rope8, mask=ok & (roffs >= 0))

    @triton.jit
    def _load_fp4_row(src_row_ptr, lut_ptr, ok):
        """One 352-byte FP4 row -> (latent [512] fp32 exact E2M1 x E4M3 scale, rope [64] uint8 E4M3 bytes)."""
        pair = tl.max_contiguous(tl.multiple_of(tl.arange(0, 256), 16), 256)
        b = tl.load(src_row_ptr + pair, mask=ok & (pair >= 0), other=0).to(tl.int32)
        soffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 16), 32)
        sc = tl.load(src_row_ptr + 256 + soffs, mask=ok & (soffs >= 0), other=0).to(
            tl.float8e4nv, bitcast=True).to(tl.float32)
        sc_pair = tl.reshape(tl.broadcast_to(sc[:, None], [32, 8]), [256])
        lo, hi = _decode_pairs(lut_ptr, b)
        x = tl.reshape(tl.join(lo * sc_pair, hi * sc_pair), [512])
        roffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
        rope8 = tl.load(src_row_ptr + 288 + roffs, mask=ok & (roffs >= 0), other=0)
        return x, rope8

    @triton.jit
    def _kvmix_write_kernel(nope8_ptr, nope8_stride, rope8_ptr, rope8_stride, bf16_ptr, bf16_stride, loc_ptr,
                            fp8_rows_ptr, boundary, fp4_rows_ptr, n_fp4_rows, n8, HAVE_BF16: tl.constexpr):
        t = tl.program_id(0)
        slot = tl.load(loc_ptr + t).to(tl.int64)
        is8 = (slot > 0) & (slot < boundary)  # slot 0 is the reserved dummy (stock skips it too)
        row4 = slot - n8
        is4 = (slot >= boundary) & (row4 >= 0) & (row4 < n_fp4_rows)
        offs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 16), 512)
        roffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
        n8b = tl.load(nope8_ptr + t.to(tl.int64) * nope8_stride + offs)
        r8b = tl.load(rope8_ptr + t.to(tl.int64) * rope8_stride + roffs)
        dst8 = fp8_rows_ptr + tl.multiple_of(tl.where(is8, slot, 0) * 576, 16)
        tl.store(dst8 + offs, n8b, mask=is8 & (offs >= 0))
        tl.store(dst8 + 512 + roffs, r8b, mask=is8 & (roffs >= 0))
        if HAVE_BF16:
            x = tl.load(bf16_ptr + t.to(tl.int64) * bf16_stride + offs).to(tl.float32)
        else:
            x = n8b.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        dst4 = fp4_rows_ptr + tl.multiple_of(tl.where(is4, row4, 0) * 352, 16)
        _quant_store_fp4(x, r8b, dst4, is4)

    @triton.jit
    def _kvmix_redirect_gather_kernel(table_ptr, n, out_ptr, fp4_rows_ptr, lut_ptr, n_fp4_rows, n8, boundary,
                                      scratch_ptr, scratch_base, BLOCK_R: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = i < n
        e = tl.load(table_ptr + i, mask=m, other=-1).to(tl.int64)
        row4 = e - n8
        is4 = m & (e >= boundary) & (row4 < n_fp4_rows)
        _kv4_dequant_row_tile(fp4_rows_ptr, lut_ptr, tl.where(is4, row4, 0), is4, scratch_ptr, i, False)
        keep = m & (e >= 0) & (e < boundary)
        out = tl.where(is4, scratch_base + i, tl.where(keep, e, -1))
        tl.store(out_ptr + i, out.to(tl.int32), mask=m)

    @triton.jit
    def _kvmix_seq_dequant_kernel(fp4_rows_ptr, lut_ptr, table_ptr, table_stride, req_ptr, seq_lens_ptr, base_ptr,
                                  max_len, n_fp4_rows, n8, boundary, scratch_ptr, slot_to_pos_ptr,
                                  BLOCK_R: tl.constexpr):
        b = tl.load(req_ptr + tl.program_id(0)).to(tl.int64)
        p = tl.program_id(1).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        n = tl.load(seq_lens_ptr + b).to(tl.int64)
        base = tl.load(base_ptr + tl.program_id(0)).to(tl.int64)
        pmask = p < tl.minimum(n, max_len)
        e = tl.load(table_ptr + b * table_stride + p, mask=pmask, other=-1).to(tl.int64)
        row4 = e - n8
        is4 = pmask & (e >= boundary) & (row4 < n_fp4_rows)
        dst_row = base + p
        _kv4_dequant_row_tile(fp4_rows_ptr, lut_ptr, tl.where(is4, row4, 0), is4, scratch_ptr, dst_row, False)
        tl.store(slot_to_pos_ptr + tl.where(is4, e, 0), dst_row.to(tl.int32), mask=is4)

    @triton.jit
    def _kvmix_redirect_seq_kernel(table_ptr, n, slot_to_pos_ptr, n_slots, boundary, scratch_base, out_ptr,
                                   BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        e = tl.load(table_ptr + i, mask=m, other=-1).to(tl.int64)
        is4 = m & (e >= boundary) & (e < n_slots)
        pos = tl.load(slot_to_pos_ptr + tl.where(is4, e, 0), mask=is4, other=-1)
        keep = m & (e >= 0) & (e < boundary)
        out = tl.where(is4, tl.where(pos >= 0, scratch_base + pos, -1), tl.where(keep, e, -1))
        tl.store(out_ptr + i, out.to(tl.int32), mask=m)

    @triton.jit
    def _kvmix_promote_kernel(fp8_all_ptr, stride8, fp4_all_ptr, stride4, src_ptr, dst_ptr, n8, boundary,
                              n_fp4_rows, n):
        i = tl.program_id(0).to(tl.int64)
        l = tl.program_id(1).to(tl.int64)
        src = tl.load(src_ptr + i).to(tl.int64)
        dst = tl.load(dst_ptr + i).to(tl.int64)
        row4 = dst - n8
        ok = (i < n) & (src >= 0) & (src < boundary) & (dst >= boundary) & (row4 < n_fp4_rows)
        offs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 16), 512)
        roffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
        srow = fp8_all_ptr + tl.multiple_of((l * stride8 + tl.where(ok, src, 0)) * 576, 16)
        x = tl.load(srow + offs, mask=ok & (offs >= 0), other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        r8 = tl.load(srow + 512 + roffs, mask=ok & (roffs >= 0), other=0)
        drow = fp4_all_ptr + tl.multiple_of((l * stride4 + tl.where(ok, row4, 0)) * 352, 16)
        _quant_store_fp4(x, r8, drow, ok)

    @triton.jit
    def _kvmix_move_kernel(fp8_all_ptr, stride8, fp4_all_ptr, stride4, lut_ptr, tgt_ptr, src_ptr, n8, boundary,
                           n_fp4_rows, n):
        i = tl.program_id(0).to(tl.int64)
        l = tl.program_id(1).to(tl.int64)
        src = tl.load(src_ptr + i).to(tl.int64)
        tgt = tl.load(tgt_ptr + i).to(tl.int64)
        m = i < n
        s8 = m & (src >= 0) & (src < boundary)
        s4 = m & (src >= boundary) & (src - n8 < n_fp4_rows)
        t8 = m & (tgt > 0) & (tgt < boundary)
        t4 = m & (tgt >= boundary) & (tgt - n8 < n_fp4_rows)
        offs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 16), 512)
        roffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
        srow8 = fp8_all_ptr + tl.multiple_of((l * stride8 + tl.where(s8, src, 0)) * 576, 16)
        x8 = tl.load(srow8 + offs, mask=s8 & (offs >= 0), other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        r8 = tl.load(srow8 + 512 + roffs, mask=s8 & (roffs >= 0), other=0)
        srow4 = fp4_all_ptr + tl.multiple_of((l * stride4 + tl.where(s4, src - n8, 0)) * 352, 16)
        x4, r4 = _load_fp4_row(srow4, lut_ptr, s4)
        x = tl.where(s8, x8, x4)
        r = tl.where(s8, r8, r4)
        drow8 = fp8_all_ptr + tl.multiple_of((l * stride8 + tl.where(t8, tgt, 0)) * 576, 16)
        tl.store(drow8 + offs, x.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=t8 & (offs >= 0))
        tl.store(drow8 + 512 + roffs, r, mask=t8 & (roffs >= 0))
        drow4 = fp4_all_ptr + tl.multiple_of((l * stride4 + tl.where(t4, tgt - n8, 0)) * 352, 16)
        _quant_store_fp4(x, r, drow4, t4 & s8)
        # FP4 -> FP4 moves copy the 352 bytes (requantizing a dequantized row is not the identity)
        boffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 16), 512)
        raw = tl.load(srow4 + boffs, mask=s4 & (boffs < 352), other=0)
        tl.store(drow4 + boffs, raw, mask=t4 & s4 & (boffs < 352))

    @triton.jit
    def _kvmix_gather_bf16_kernel(fp8_rows_ptr, fp4_rows_ptr, lut_ptr, loc_ptr, n, n8, boundary, n_fp4_rows, out_ptr):
        """out[i] = the 576 values of row loc[i] (either format) as bf16 (nope 512 | rope 64)."""
        i = tl.program_id(0).to(tl.int64)
        slot = tl.load(loc_ptr + i).to(tl.int64)
        m = i < n
        s8 = m & (slot >= 0) & (slot < boundary)
        s4 = m & (slot >= boundary) & (slot - n8 < n_fp4_rows)
        offs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 16), 512)
        roffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 64), 16), 64)
        srow8 = fp8_rows_ptr + tl.multiple_of(tl.where(s8, slot, 0) * 576, 16)
        x8 = tl.load(srow8 + offs, mask=s8 & (offs >= 0), other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        r8 = tl.load(srow8 + 512 + roffs, mask=s8 & (roffs >= 0), other=0)
        srow4 = fp4_rows_ptr + tl.multiple_of(tl.where(s4, slot - n8, 0) * 352, 16)
        x4, r4 = _load_fp4_row(srow4, lut_ptr, s4)
        x = tl.where(s8, x8, x4)
        r = tl.where(s8, r8, r4).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        orow = out_ptr + i * 576
        tl.store(orow + offs, x.to(tl.bfloat16), mask=m & (offs >= 0))
        tl.store(orow + 512 + roffs, r.to(tl.bfloat16), mask=m & (roffs >= 0))

    @triton.jit
    def _kvmix_index_move_kernel(buf_ptr, page_bytes, tgt_ptr, src_ptr, n, PAGE: tl.constexpr):
        i = tl.program_id(0).to(tl.int64)
        src = tl.load(src_ptr + i).to(tl.int64)
        tgt = tl.load(tgt_ptr + i).to(tl.int64)
        ok = (i < n) & (src >= 0) & (tgt > 0)
        koffs = tl.arange(0, 128)
        soffs = tl.arange(0, 4)
        sp = tl.where(ok, src, 0)
        tp = tl.where(ok, tgt, 0)
        sk = buf_ptr + (sp // PAGE) * page_bytes + (sp % PAGE) * 128
        tk = buf_ptr + (tp // PAGE) * page_bytes + (tp % PAGE) * 128
        k = tl.load(sk + koffs, mask=ok & (koffs >= 0), other=0)
        tl.store(tk + koffs, k, mask=ok & (koffs >= 0))
        ss = buf_ptr + (sp // PAGE) * page_bytes + PAGE * 128 + (sp % PAGE) * 4
        ts = buf_ptr + (tp // PAGE) * page_bytes + PAGE * 128 + (tp % PAGE) * 4
        s = tl.load(ss + soffs, mask=ok & (soffs >= 0), other=0)
        tl.store(ts + soffs, s, mask=ok & (soffs >= 0))


# ------------------------------------------------------------------------------------------------ helpers


def rows2d(buf: torch.Tensor, row_bytes: int) -> torch.Tensor:
    """A pool buffer [N, 1, D] (uint8 or fp8 view) as contiguous uint8 [N, row_bytes]."""
    b = buf if buf.dtype == torch.uint8 else buf.view(torch.uint8)
    b = b.reshape(b.shape[0], -1)
    assert b.shape[1] == row_bytes and b.is_contiguous(), (b.shape, row_bytes)
    return b


class Layout:
    """Unified slot space of one mixed pool: FP8 slots [0, boundary), FP4 slots [boundary, n_slots)."""

    def __init__(self, n8: int, n4: int, page_size: int):
        assert n8 % page_size == 0 and n4 % page_size == 0, (n8, n4, page_size)
        self.n8, self.n4, self.page_size = n8, n4, page_size
        self.boundary = n8 + page_size  # first FP4 slot
        self.n_slots = n8 + n4 + page_size
        self.fp8_rows = n8 + page_size  # rows of one layer's FP8 buffer (page 0 is the dummy page)
        self.fp4_rows = n4 + page_size  # rows of one layer's FP4 buffer (rows [0, page) never allocated)
        self.fp8_pages = n8 // page_size  # allocatable FP8 pages 1 .. fp8_pages
        self.fp4_pages = n4 // page_size  # allocatable FP4 pages, unified ids fp8_pages + 1 .. fp8_pages + fp4_pages

    def is_fp4_slot(self, slot):
        return slot >= self.boundary

    def fp4_row(self, slot):
        return slot - self.n8

    def fp4_page_to_unified(self, native_page):
        return native_page + self.fp8_pages

    def unified_page_to_fp4(self, page):
        return page - self.fp8_pages

    def describe(self) -> str:
        return (f"FP8 slots [1, {self.boundary}) = {self.n8} tokens, FP4 slots [{self.boundary}, {self.n_slots}) = "
                f"{self.n4} tokens, page {self.page_size}")


# ------------------------------------------------------------------------------------------------ wrappers


def write_mixed(lay: Layout, fp8_rows: torch.Tensor, fp4_rows: torch.Tensor, loc: torch.Tensor,
                nope_fp8: torch.Tensor, rope_fp8: torch.Tensor, nope_bf16: torch.Tensor | None) -> None:
    """Pool write door. fp8_rows [fp8_rows, 576], fp4_rows [fp4_rows, 352] (uint8 or fp8 views of one layer),
    loc [T] unified slots, nope_fp8 [T, 512] fp8, rope_fp8 [T, 64] fp8, nope_bf16 [T, 512] bf16 or None."""
    t = loc.numel()
    if t == 0:
        return
    r8 = rows2d(fp8_rows, FP8_ROW_BYTES)
    r4 = rows2d(fp4_rows, ROW_BYTES)
    n8 = nope_fp8.reshape(t, DIM_NOPE)
    ro = rope_fp8.reshape(t, DIM_ROPE)
    assert n8.dtype == torch.float8_e4m3fn and ro.dtype == torch.float8_e4m3fn, (n8.dtype, ro.dtype)
    n8 = n8.contiguous().view(torch.uint8)
    ro = ro.contiguous().view(torch.uint8)
    have = nope_bf16 is not None and nope_bf16.shape[0] == t
    if have:
        bf = nope_bf16.reshape(t, DIM_NOPE)
        assert bf.dtype in (torch.bfloat16, torch.float16, torch.float32), bf.dtype
        bf = bf.contiguous()
    else:
        bf = n8
    _kvmix_write_kernel[(t,)](n8, n8.stride(0), ro, ro.stride(0), bf, bf.stride(0), loc.reshape(-1), r8,
                              lay.boundary, r4, r4.shape[0], lay.n8, HAVE_BF16=have, num_warps=4)


def redirect_gather(lay: Layout, table: torch.Tensor, out: torch.Tensor, fp4_rows: torch.Tensor,
                    scratch: torch.Tensor, scratch_base: int, block_r: int = 16, num_warps: int = 4) -> None:
    """out[i] = table[i] for FP8 entries, scratch_base + i for FP4 entries (dequantized into scratch[i]), -1 else.
    table, out: int32, same shape, contiguous, numel <= scratch rows. scratch: [S, 576] fp8/uint8."""
    n = table.numel()
    if n == 0:
        return
    assert table.dtype == torch.int32 and out.shape == table.shape and table.is_contiguous() and out.is_contiguous()
    r4 = rows2d(fp4_rows, ROW_BYTES)
    sc = rows2d(scratch, FP8_ROW_BYTES)
    assert sc.shape[0] >= n, (sc.shape, n)
    _kvmix_redirect_gather_kernel[(triton.cdiv(n, block_r),)](
        table.view(-1), n, out.view(-1), r4, kv4.byte_lut(r4.device), r4.shape[0], lay.n8, lay.boundary, sc,
        scratch_base, BLOCK_R=block_r, num_warps=num_warps)


def seq_dequant(lay: Layout, fp4_rows: torch.Tensor, page_table: torch.Tensor, req_ids: torch.Tensor,
                seq_lens: torch.Tensor, bases: torch.Tensor, max_len: int, scratch: torch.Tensor,
                slot_to_pos: torch.Tensor, block_r: int = 16, num_warps: int = 4) -> None:
    """For each request b = req_ids[j] and position p < min(seq_lens[b], max_len) whose slot page_table[b, p] is FP4:
    scratch[bases[j] + p] = dequant(row) and slot_to_pos[slot] = bases[j] + p. page_table [bs, max_len] int32."""
    nb = req_ids.numel()
    if nb == 0 or max_len == 0:
        return
    assert page_table.dtype == torch.int32 and page_table.dim() == 2
    r4 = rows2d(fp4_rows, ROW_BYTES)
    sc = rows2d(scratch, FP8_ROW_BYTES)
    assert slot_to_pos.dtype == torch.int32 and slot_to_pos.numel() >= lay.n_slots
    _kvmix_seq_dequant_kernel[(nb, triton.cdiv(max_len, block_r))](
        r4, kv4.byte_lut(r4.device), page_table, page_table.stride(0), req_ids, seq_lens, bases, max_len,
        r4.shape[0], lay.n8, lay.boundary, sc, slot_to_pos, BLOCK_R=block_r, num_warps=num_warps)


def redirect_seq(lay: Layout, table: torch.Tensor, slot_to_pos: torch.Tensor, scratch_base: int,
                 out: torch.Tensor) -> None:
    """out[i] = table[i] (FP8), scratch_base + slot_to_pos[table[i]] (FP4), -1 (padding)."""
    n = table.numel()
    if n == 0:
        return
    assert table.dtype == torch.int32 and out.shape == table.shape and table.is_contiguous() and out.is_contiguous()
    block = 1024
    _kvmix_redirect_seq_kernel[(triton.cdiv(n, block),)](table.view(-1), n, slot_to_pos, lay.n_slots, lay.boundary,
                                                         scratch_base, out.view(-1), BLOCK=block, num_warps=4)


def promote_rows(lay: Layout, fp8_all: torch.Tensor, fp4_all: torch.Tensor, layers: int, src: torch.Tensor,
                 dst: torch.Tensor) -> None:
    """fp4[l][dst[i] - n8] = quantize(fp8[l][src[i]]) for every layer l. fp8_all: [layers * fp8_rows (+ scratch), 576]
    rows, fp4_all: [layers * fp4_rows, 352] rows (layer l at rows [l * rows, (l + 1) * rows))."""
    n = src.numel()
    if n == 0:
        return
    assert dst.numel() == n
    r8 = rows2d(fp8_all, FP8_ROW_BYTES)
    r4 = rows2d(fp4_all, ROW_BYTES)
    assert r8.shape[0] >= layers * lay.fp8_rows and r4.shape[0] == layers * lay.fp4_rows, (r8.shape, r4.shape)
    _kvmix_promote_kernel[(n, layers)](r8, lay.fp8_rows, r4, lay.fp4_rows, src.reshape(-1), dst.reshape(-1), lay.n8,
                                       lay.boundary, lay.fp4_rows, n, num_warps=4)


def move_rows(lay: Layout, fp8_all: torch.Tensor, fp4_all: torch.Tensor, layers: int, tgt: torch.Tensor,
              src: torch.Tensor) -> None:
    """row[tgt[i]] = row[src[i]] across formats for every layer (FP4 -> FP8 dequantizes, FP8 -> FP4 quantizes)."""
    n = src.numel()
    if n == 0:
        return
    assert tgt.numel() == n
    r8 = rows2d(fp8_all, FP8_ROW_BYTES)
    r4 = rows2d(fp4_all, ROW_BYTES)
    _kvmix_move_kernel[(n, layers)](r8, lay.fp8_rows, r4, lay.fp4_rows, kv4.byte_lut(r8.device), tgt.reshape(-1),
                                    src.reshape(-1), lay.n8, lay.boundary, lay.fp4_rows, n, num_warps=4)


def gather_bf16(lay: Layout, fp8_rows: torch.Tensor, fp4_rows: torch.Tensor, loc: torch.Tensor,
                out: torch.Tensor) -> None:
    """out [T, 576] bf16 = rows at unified slots loc (either format), for the dense MHA prefill path."""
    n = loc.numel()
    if n == 0:
        return
    r8 = rows2d(fp8_rows, FP8_ROW_BYTES)
    r4 = rows2d(fp4_rows, ROW_BYTES)
    assert out.dtype == torch.bfloat16 and out.shape == (n, DIM_QK) and out.is_contiguous()
    _kvmix_gather_bf16_kernel[(n,)](r8, r4, kv4.byte_lut(r8.device), loc.reshape(-1), n, lay.n8, lay.boundary,
                                    r4.shape[0], out, num_warps=4)


def gather_bf16_ref(lay: Layout, fp8_rows: torch.Tensor, fp4_rows: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    r8 = rows2d(fp8_rows, FP8_ROW_BYTES)
    r4 = rows2d(fp4_rows, ROW_BYTES)
    out = torch.zeros(loc.numel(), DIM_QK, dtype=torch.bfloat16)
    for i, s in enumerate(loc.tolist()):
        if 0 <= s < lay.boundary:
            out[i] = r8[s].view(torch.float8_e4m3fn).float().to(torch.bfloat16)
        elif s >= lay.boundary:
            out[i] = kv4.dequantize_rows_ref(r4[s - lay.n8: s - lay.n8 + 1], torch.float32)[0].to(torch.bfloat16)
    return out


def index_move(index_buf: torch.Tensor, page_size: int, tgt: torch.Tensor, src: torch.Tensor) -> None:
    """Token-granular move inside one layer's page-major DSA indexer buffer [pages, page_size * 132] uint8."""
    n = src.numel()
    if n == 0 or index_buf.shape[0] == 0:
        return
    assert index_buf.dtype == torch.uint8 and index_buf.is_contiguous()
    assert index_buf.shape[1] == page_size * (INDEX_K_BYTES + INDEX_S_BYTES)
    _kvmix_index_move_kernel[(n,)](index_buf, index_buf.shape[1], tgt.reshape(-1), src.reshape(-1), n,
                                   PAGE=page_size, num_warps=1)


# ------------------------------------------------------------------------------------------------ references (CPU)


def write_mixed_ref(lay: Layout, fp8_rows: torch.Tensor, fp4_rows: torch.Tensor, loc: torch.Tensor,
                    nope_fp8: torch.Tensor, rope_fp8: torch.Tensor, nope_bf16: torch.Tensor | None) -> None:
    r8 = rows2d(fp8_rows, FP8_ROW_BYTES)
    r4 = rows2d(fp4_rows, ROW_BYTES)
    t = loc.numel()
    n8 = nope_fp8.reshape(t, DIM_NOPE)
    ro = rope_fp8.reshape(t, DIM_ROPE)
    src = nope_bf16.reshape(t, DIM_NOPE) if nope_bf16 is not None else n8
    q4 = kv4.quantize_rows_ref(src, ro)
    for i in range(t):
        s = int(loc[i])
        if 0 < s < lay.boundary:
            r8[s, :DIM_NOPE] = n8[i].view(torch.uint8)
            r8[s, DIM_NOPE:] = ro[i].view(torch.uint8)
        elif lay.boundary <= s < lay.n_slots:
            r4[s - lay.n8] = q4[i]


def redirect_gather_ref(lay: Layout, table: torch.Tensor, fp4_rows: torch.Tensor, scratch_base: int):
    """Returns (out table int32, scratch rows dict {flat index: 576-byte fp8 row})."""
    r4 = rows2d(fp4_rows, ROW_BYTES)
    flat = table.reshape(-1).to(torch.int64)
    out = torch.full_like(flat, -1, dtype=torch.int32)
    rows = {}
    for i, e in enumerate(flat.tolist()):
        if 0 <= e < lay.boundary:
            out[i] = e
        elif lay.boundary <= e < lay.n_slots:
            out[i] = scratch_base + i
            rows[i] = kv4.dequantize_rows_ref(r4[e - lay.n8: e - lay.n8 + 1])[0]
    return out.reshape(table.shape), rows


def promote_ref(lay: Layout, fp8_all: torch.Tensor, fp4_all: torch.Tensor, layers: int, src: torch.Tensor,
                dst: torch.Tensor) -> None:
    r8 = rows2d(fp8_all, FP8_ROW_BYTES)
    r4 = rows2d(fp4_all, ROW_BYTES)
    for l in range(layers):
        rows = r8[l * lay.fp8_rows + src.to(torch.int64)]
        q = kv4.quantize_rows_ref(rows[:, :DIM_NOPE].view(torch.float8_e4m3fn),
                                  rows[:, DIM_NOPE:].view(torch.float8_e4m3fn))
        r4[l * lay.fp4_rows + dst.to(torch.int64) - lay.n8] = q


def move_ref(lay: Layout, fp8_all: torch.Tensor, fp4_all: torch.Tensor, layers: int, tgt: torch.Tensor,
             src: torch.Tensor) -> None:
    r8 = rows2d(fp8_all, FP8_ROW_BYTES)
    r4 = rows2d(fp4_all, ROW_BYTES)
    for l in range(layers):
        for t_, s_ in zip(tgt.tolist(), src.tolist()):
            if s_ < lay.boundary:
                vals = r8[l * lay.fp8_rows + s_].view(torch.float8_e4m3fn).float()[None]
            else:
                vals = kv4.dequantize_rows_ref(r4[l * lay.fp4_rows + s_ - lay.n8][None], torch.float32)
            if 0 < t_ < lay.boundary:
                r8[l * lay.fp8_rows + t_] = vals[0].to(torch.float8_e4m3fn).view(torch.uint8)
            elif t_ >= lay.boundary and s_ >= lay.boundary:
                r4[l * lay.fp4_rows + t_ - lay.n8] = r4[l * lay.fp4_rows + s_ - lay.n8].clone()
            elif t_ >= lay.boundary:
                r4[l * lay.fp4_rows + t_ - lay.n8] = kv4.quantize_rows_ref(vals[:, :DIM_NOPE], vals[:, DIM_NOPE:])[0]


def seq_ref(lay: Layout, fp4_rows: torch.Tensor, page_table: torch.Tensor, req_ids, seq_lens, bases, table,
            scratch_base: int):
    """Returns (redirected table, {scratch row: fp8 row}) for the prefill path."""
    r4 = rows2d(fp4_rows, ROW_BYTES)
    slot_to_pos = {}
    rows = {}
    for j, b in enumerate(req_ids.tolist()):
        for p in range(int(seq_lens[b])):
            e = int(page_table[b, p])
            if e >= lay.boundary:
                slot_to_pos[e] = int(bases[j]) + p
                rows[int(bases[j]) + p] = kv4.dequantize_rows_ref(r4[e - lay.n8: e - lay.n8 + 1])[0]
    flat = table.reshape(-1).tolist()
    out = torch.full((len(flat),), -1, dtype=torch.int32)
    for i, e in enumerate(flat):
        if 0 <= e < lay.boundary:
            out[i] = e
        elif e >= lay.boundary:
            out[i] = scratch_base + slot_to_pos[e]
    return out.reshape(table.shape), rows
