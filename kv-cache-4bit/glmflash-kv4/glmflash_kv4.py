"""4-bit KV row for GLM-5.3-Flash's DSA latent cache: the rope-free variant of operations/kernels/glm_kv4.py.

Flash's 11 DSA layers are MLA with kv_lora_rank 512 and qk_rope_head_dim 0 (`mla_use_nope`, source_code config.json),
so the served FP8 row (trtllm roles, SM100) is 512 bytes of E4M3 latent and nothing else. The FP4 row is:

    [0, 256)    256 bytes: 512 E2M1 nibbles of the latent, element 2i in the low nibble of byte i
    [256, 288)   32 bytes: one E4M3 scale per block of 16 latent values

288 B/token/layer against 512 (1.78x latent tokens per byte; 11 layers: 3,168 vs 5,632 B/token before the indexer
cache, which is unchanged). Quantization is glm_kv4's exactly (per-16 E4M3 scale = amax/6 clamped to [2^-9, 448],
E2M1 codes with ties down, saturation at 6, no negative zero), so glm_kv4's byte LUT, encoder and decoder are reused;
only the row geometry differs. The torch reference is CPU-tested (test_glmflash_kv4.py); the Triton kernels below are
glm_kv4's with the rope tail removed (the geometry-free kernels: stamp / count / compact / remap / unstamp / identity /
combine, are glm_kv4's own). GPU tests: test_glmflash_kv4_gpu.py, NOT YET RUN (no B200 available 2026-09-29): treat
every kernel here as unverified until that file passes on a box.

Port checklist (what changed against glm_kv4.py; every literal named here):
  quant_write     drop the rope pointer/stride and the 64-wide rope store at row offset 288; row stride 352 -> 288
  dequant tile    src row stride 352 -> 288; dst FP8 row stride 576 -> 512; drop the 64-byte rope copy at src 288 /
                  dst 512 (both the fp8 and the bf16 output paths)
  dequant_seq / dequant_rows / dequant_rows_dedup / owner kernels   only the two strides above
  sparse_mla      q row 576 -> 512 (no q_rope load at 512, no rope dot), k tile from 288-byte rows, out 512 unchanged
  engine (glm_kv4_engine.py)   kv dim 352 -> 288 in the kv_cache_configurator hook; scratch rows 576 -> 512 bytes;
                  attach() must accept qk_rope_head_dim == 0 (glm_kv4 asserts the GLM-5.3 geometry) and pass
                  `prepare_trtllm_nope_sparse_metadata` lens like dsa_backend._forward_trtllm does for rope 0
  hooks           the anchors in glm_kv4_hooks.py are on DeepseekSparseAttnBackend and kv_cache_configurator, the
                  same classes Flash uses (Glm5Next DSA layers are DeepseekV2AttentionMLA); the stash hook keys on
                  `merge_query = q_rope is not None and self.qk_rope_head_dim > 0`, which is False on Flash: the
                  q path is nope-only, so the bf16 latent stash still applies but no rope stash exists
Pays when (derived, research/glmflash.md): the node KV pool binds, i.e. above ~160 resident 70K-token sessions at
TP4 with fp8 KV + ratio 0.3 (11.3M tokens); below that the fp8 pairing already holds every session and 4-bit only
adds the dequant gather cost (kv4glm measured +2.1-2.2x per attention call on GLM-5.3; on Flash attention is 11 of
45 layers, so ~1/7 of that share).
"""
from __future__ import annotations

import torch

try:  # installed next to glm_kv4.py inside sglang/srt/layers/attention/dsa/ by glmflash_kv4_hooks.py
    from . import glm_kv4
except ImportError:  # the repo checkout (operations/kernels on sys.path)
    import glm_kv4
# same numerics and the geometry-free kernels, re-exported
BLOCK, E2M1_MAX, E2M1_MIDPOINTS, E2M1_VALUES = glm_kv4.BLOCK, glm_kv4.E2M1_MAX, glm_kv4.E2M1_MIDPOINTS, glm_kv4.E2M1_VALUES
HAVE_TRITON, SCALE_MAX, SCALE_MIN, STAMP_CLEAN = glm_kv4.HAVE_TRITON, glm_kv4.SCALE_MAX, glm_kv4.SCALE_MIN, glm_kv4.STAMP_CLEAN
_e2m1_codes_ref, byte_lut, identity_table, remap_table = (glm_kv4._e2m1_codes_ref, glm_kv4.byte_lut,
                                                          glm_kv4.identity_table, glm_kv4.remap_table)

DIM_NOPE = 512
DIM_ROPE = 0
DIM_QK = DIM_NOPE  # 512: the raw FP8 row TRT-LLM reads on Flash
N_BLOCKS = DIM_NOPE // BLOCK  # 32
NOPE_PACKED_BYTES = DIM_NOPE // 2  # 256
SCALE_OFF = NOPE_PACKED_BYTES  # 256
ROW_BYTES = SCALE_OFF + N_BLOCKS  # 288
FP8_ROW_BYTES = DIM_QK  # 512
LAYERS_DSA = 11


def bytes_per_token(layers: int = LAYERS_DSA, indexer_bytes: int = 0) -> dict:
    """Derived per-token bytes of the DSA cache in the three formats (indexer cache added as given)."""
    return {"bf16": layers * 2 * DIM_QK + indexer_bytes, "fp8": layers * FP8_ROW_BYTES + indexer_bytes,
            "fp4": layers * ROW_BYTES + indexer_bytes}


def quantize_rows_ref(nope: torch.Tensor) -> torch.Tensor:
    """nope [T, 512] (bf16/fp32/fp8) -> uint8 rows [T, 288]. CPU or GPU. Same math as glm_kv4.quantize_rows_ref."""
    t = nope.shape[0]
    x = nope.float().reshape(t, N_BLOCKS, BLOCK)
    amax = x.abs().amax(dim=-1)
    s8 = (amax / E2M1_MAX).clamp(min=SCALE_MIN, max=SCALE_MAX).to(torch.float8_e4m3fn)
    s_eff = s8.float()
    inv = torch.where(s_eff > 0, 1.0 / s_eff, torch.zeros_like(s_eff))
    y = (x.abs() * inv.unsqueeze(-1)).clamp(max=E2M1_MAX)
    codes = _e2m1_codes_ref(y)
    codes = (codes | (((x < 0) & (codes > 0)).to(torch.int32) << 3)).reshape(t, DIM_NOPE)
    out = torch.empty(t, ROW_BYTES, dtype=torch.uint8, device=nope.device)
    out[:, :SCALE_OFF] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    out[:, SCALE_OFF:] = s8.view(torch.uint8)
    return out


def dequantize_rows_ref(rows: torch.Tensor, out_dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    """uint8 rows [R, 288] -> [R, 512] in out_dtype (fp8 = the TRT-LLM raw layout on Flash)."""
    r = rows.shape[0]
    packed = rows[:, :SCALE_OFF].to(torch.int32)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(r, DIM_NOPE)
    mag = E2M1_VALUES.to(rows.device)[codes & 7]
    val = torch.where(((codes & 8) != 0) & (mag != 0), -mag, mag)
    s = rows[:, SCALE_OFF:].view(torch.float8_e4m3fn).float()
    return (val.reshape(r, N_BLOCKS, BLOCK) * s.unsqueeze(-1)).reshape(r, DIM_NOPE).to(out_dtype)


def sparse_mla_ref(q: torch.Tensor, rows: torch.Tensor, table: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """fp32 reference: q [T, H, 512], rows [N, 288] uint8, table [T, K] slots (-1 masked) -> [T, H, 512]."""
    t, h, _ = q.shape
    kv = dequantize_rows_ref(rows, torch.float32)
    out = torch.zeros(t, h, DIM_NOPE, dtype=torch.float32, device=q.device)
    qf = q.float()
    for i in range(t):
        idx = table[i].long()
        valid = idx >= 0
        if not bool(valid.any()):
            continue
        k = kv[idx[valid]]
        p = torch.softmax(qf[i] @ k.T * sm_scale, dim=-1)
        out[i] = p @ k
    return out


# ------------------------------------------------------------------------------------------------ Triton kernels
# The same kernels as glm_kv4 with the rope tail removed: rows are 288 B (16 x 18), FP8 output rows 512 B (16 x 32),
# so every row start stays 16-byte aligned and the same vectorization hints hold.

if HAVE_TRITON:
    import triton
    import triton.language as tl
    _decode_pairs, _e2m1_encode = glm_kv4._decode_pairs, glm_kv4._e2m1_encode
    _kv4_combine_kernel, _kv4_owner_compact_kernel = glm_kv4._kv4_combine_kernel, glm_kv4._kv4_owner_compact_kernel
    _kv4_owner_count_kernel, _kv4_remap_stamp_kernel = glm_kv4._kv4_owner_count_kernel, glm_kv4._kv4_remap_stamp_kernel
    _kv4_stamp_kernel, _kv4_unstamp_kernel = glm_kv4._kv4_stamp_kernel, glm_kv4._kv4_unstamp_kernel

    @triton.jit
    def _fkv4_quant_write_kernel(nope_ptr, nope_stride, loc_ptr, dst_ptr, n_pool, NOPE_FP8: tl.constexpr):
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
        amax = tl.max(tl.abs(xb), axis=1)
        s = tl.minimum(tl.maximum(amax / 6.0, 0.001953125), 448.0)
        s8 = s.to(tl.float8e4nv)
        s_eff = s8.to(tl.float32)
        inv = tl.where(s_eff > 0, 1.0 / s_eff, 0.0)
        codes = _e2m1_encode(xb, tl.broadcast_to(inv[:, None], [32, 16]))
        lo, hi = tl.split(tl.reshape(codes, [256, 2]))
        byte = (lo | (hi << 4)).to(tl.uint8)
        row = dst_ptr + tl.multiple_of(loc * 288, 16)
        poffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 256), 16), 256)
        tl.store(row + poffs, byte, mask=ok & (poffs >= 0))
        soffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 16), 32)
        tl.store(row + 256 + soffs, s8.to(tl.uint8, bitcast=True), mask=ok & (soffs >= 0))

    @triton.jit
    def _fkv4_dequant_row_tile(src_ptr, lut_ptr, slot, valid, dst_ptr, dst_row, OUT_BF16: tl.constexpr):
        """slot, valid, dst_row: [R] vectors. Writes 512 values per valid row (fp8 bytes or bf16)."""
        row_off = tl.multiple_of(tl.where(valid, slot, 0).to(tl.int64) * 288, 16)
        src = src_ptr + row_off[:, None]
        pair = tl.max_contiguous(tl.multiple_of(tl.arange(0, 256), 16), 256)
        b = tl.load(src + pair[None, :], mask=valid[:, None], other=0).to(tl.int32)
        soffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 32), 16), 32)
        sc = tl.load(src + 256 + soffs[None, :], mask=valid[:, None], other=0).to(
            tl.float8e4nv, bitcast=True).to(tl.float32)
        R: tl.constexpr = b.shape[0]
        sc_pair = tl.reshape(tl.broadcast_to(sc[:, :, None], [R, 32, 8]), [R, 256])
        lo, hi = _decode_pairs(lut_ptr, b)
        v = tl.reshape(tl.join(lo * sc_pair, hi * sc_pair), [R, 512])
        dst = dst_ptr + tl.multiple_of(dst_row.to(tl.int64) * 512, 16)[:, None]
        doffs = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 16), 512)
        if OUT_BF16:
            tl.store(dst + doffs[None, :], v.to(tl.bfloat16), mask=valid[:, None])
        else:
            tl.store(dst + doffs[None, :], v.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=valid[:, None])

    @triton.jit
    def _fkv4_dequant_rows_kernel(src_ptr, lut_ptr, idx_ptr, n_rows, n_pool, dst_ptr, OUT_BF16: tl.constexpr,
                                  BLOCK_R: tl.constexpr):
        r = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < n_rows
        slot = tl.load(idx_ptr + r, mask=rmask, other=-1).to(tl.int64)
        valid = rmask & (slot >= 0) & (slot < n_pool)
        _fkv4_dequant_row_tile(src_ptr, lut_ptr, slot, valid, dst_ptr, r, OUT_BF16)

    @triton.jit
    def _fkv4_dequant_seq_kernel(src_ptr, lut_ptr, table_ptr, table_stride, seq_lens_ptr, base_ptr, max_len, n_pool,
                                 dst_ptr, slot_to_pos_ptr, BLOCK_R: tl.constexpr):
        b = tl.program_id(0)
        p = tl.program_id(1).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        n = tl.load(seq_lens_ptr + b).to(tl.int64)
        base = tl.load(base_ptr + b).to(tl.int64)
        pmask = p < tl.minimum(n, max_len)
        slot = tl.load(table_ptr + b.to(tl.int64) * table_stride + p, mask=pmask, other=-1).to(tl.int64)
        valid = pmask & (slot >= 0) & (slot < n_pool)
        dst_row = base + p
        _fkv4_dequant_row_tile(src_ptr, lut_ptr, slot, valid, dst_ptr, dst_row, False)
        tl.store(slot_to_pos_ptr + tl.where(valid, slot, 0), dst_row.to(tl.int32), mask=valid)

    @triton.jit
    def _fkv4_dequant_owner_kernel(src_ptr, lut_ptr, table_ptr, owners_ptr, n_owners_ptr, cap, n_pool, dst_ptr,
                                   BLOCK_R: tl.constexpr):
        n_owners = tl.minimum(tl.load(n_owners_ptr), cap)
        if tl.program_id(0) * BLOCK_R >= n_owners:
            return
        j = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = j < n_owners
        fi = tl.load(owners_ptr + j, mask=m, other=0).to(tl.int64)
        slot = tl.load(table_ptr + fi, mask=m, other=-1).to(tl.int64)
        valid = m & (slot >= 0) & (slot < n_pool)
        _fkv4_dequant_row_tile(src_ptr, lut_ptr, slot, valid, dst_ptr, j, False)

    @triton.jit
    def _fkv4_sparse_mla_kernel(q_ptr, q_stride_t, q_stride_h, kv_ptr, lut_ptr, n_pool, table_ptr, table_stride, K,
                                out_ptr, out_stride_t, out_stride_h, part_o_ptr, part_ml_ptr, sm_scale_log2,
                                H: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr,
                                KS: tl.constexpr):
        """glm_kv4's fused sparse MLA without the rope dot: q [T, H, 512] fp8, 288-byte rows, out [T, H, 512]."""
        t = tl.program_id(0).to(tl.int64)
        hg = tl.program_id(1)
        sp = tl.program_id(2)
        h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = h < H
        d_nope = tl.arange(0, 512)
        qbase = q_ptr + t * q_stride_t + h.to(tl.int64)[:, None] * q_stride_h
        q_nope = tl.load(qbase + d_nope[None, :], mask=hmask[:, None], other=0).to(
            tl.float8e4nv, bitcast=True).to(tl.bfloat16)
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
            src = kv_ptr + tl.multiple_of(tl.where(valid, slot, 0) * 288, 16)[:, None]
            b = tl.load(src + pair[None, :], mask=valid[:, None], other=0).to(tl.int32)
            sc = tl.load(src + 256 + tl.arange(0, 32)[None, :], mask=valid[:, None], other=0).to(
                tl.float8e4nv, bitcast=True).to(tl.float32)
            sc_pair = tl.reshape(tl.broadcast_to(sc[:, :, None], [BLOCK_N, 32, 8]), [BLOCK_N, 256])
            lo, hi = _decode_pairs(lut_ptr, b)
            k_nope = tl.reshape(tl.join(lo * sc_pair, hi * sc_pair), [BLOCK_N, 512]).to(tl.bfloat16)
            s = tl.dot(q_nope, tl.trans(k_nope)) * sm_scale_log2
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
            obase = out_ptr + t * out_stride_t + h.to(tl.int64)[:, None] * out_stride_h
            tl.store(obase + d_nope[None, :], (acc / l_safe[:, None]).to(tl.bfloat16), mask=hmask[:, None])
        else:
            pbase = part_o_ptr + ((t * SPLITS + sp) * H + h.to(tl.int64))[:, None] * 512
            tl.store(pbase + d_nope[None, :], acc, mask=hmask[:, None])
            mlbase = part_ml_ptr + ((t * SPLITS + sp) * H + h.to(tl.int64)) * 2
            tl.store(mlbase, m_i, mask=hmask)
            tl.store(mlbase + 1, l_i, mask=hmask)


# ------------------------------------------------------------------------------------------------ Python wrappers


def pool_rows(kv_buffer: torch.Tensor) -> torch.Tensor:
    """A pool buffer [N, 1, 288] (uint8 or an fp8 view of it) as uint8 [N, 288]."""
    b = kv_buffer if kv_buffer.dtype == torch.uint8 else kv_buffer.view(torch.uint8)
    return b.reshape(b.shape[0], -1)


def quant_write(kv_buffer: torch.Tensor, loc: torch.Tensor, nope: torch.Tensor) -> None:
    """Quantize nope [T, 512] (bf16 or fp8) into pool rows kv_buffer[loc[t]] (loc outside the pool: skipped)."""
    rows = pool_rows(kv_buffer)
    nope = nope.reshape(nope.shape[0], DIM_NOPE)
    t = nope.shape[0]
    if t == 0:
        return
    assert loc.numel() == t, (nope.shape, loc.shape)
    if not nope.is_contiguous():
        nope = nope.contiguous()
    nope_fp8 = nope.dtype == torch.float8_e4m3fn
    assert nope_fp8 or nope.dtype in (torch.bfloat16, torch.float16, torch.float32), nope.dtype
    _fkv4_quant_write_kernel[(t,)](nope.view(torch.uint8) if nope_fp8 else nope, nope.stride(0), loc, rows,
                                   rows.shape[0], NOPE_FP8=nope_fp8, num_warps=4)


def dequant_rows(kv_buffer: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, block_r: int = 16,
                 num_warps: int = 4) -> None:
    """dst[r] = dequant(pool[idx[r]]) for idx[r] >= 0. dst: [R, 512] float8_e4m3fn/uint8 (raw layout) or bf16."""
    rows = pool_rows(kv_buffer)
    idx = idx.reshape(-1)
    n = idx.numel()
    if n == 0:
        return
    assert dst.shape[0] >= n and dst.shape[-1] == DIM_QK and dst.is_contiguous(), (dst.shape, n)
    out_bf16 = dst.dtype == torch.bfloat16
    d = dst if out_bf16 else dst.view(torch.uint8)
    _fkv4_dequant_rows_kernel[(triton.cdiv(n, block_r),)](rows, byte_lut(rows.device), idx, n, rows.shape[0], d,
                                                         OUT_BF16=out_bf16, BLOCK_R=block_r, num_warps=num_warps)


def dequant_seq(kv_buffer: torch.Tensor, page_table: torch.Tensor, seq_lens: torch.Tensor, bases: torch.Tensor,
                max_len: int, dst: torch.Tensor, slot_to_pos: torch.Tensor, block_r: int = 16,
                num_warps: int = 4) -> None:
    """glm_kv4.dequant_seq on 288-byte rows into a [rows, 512] fp8 scratch."""
    rows = pool_rows(kv_buffer)
    b = page_table.shape[0]
    if b == 0 or max_len == 0:
        return
    assert page_table.dtype == torch.int32 and dst.shape[-1] == DIM_QK and dst.is_contiguous()
    assert slot_to_pos.dtype == torch.int32 and slot_to_pos.numel() >= rows.shape[0]
    _fkv4_dequant_seq_kernel[(b, triton.cdiv(max_len, block_r))](
        rows, byte_lut(rows.device), page_table, page_table.stride(0), seq_lens, bases, max_len, rows.shape[0],
        dst.view(torch.uint8) if dst.dtype != torch.uint8 else dst, slot_to_pos, BLOCK_R=block_r,
        num_warps=num_warps)


def dequant_rows_dedup(kv_buffer: torch.Tensor, table: torch.Tensor, stamp: torch.Tensor, dst: torch.Tensor,
                       out: torch.Tensor, block_r: int = 8, num_warps: int = 4, work: tuple | None = None) -> torch.Tensor:
    """glm_kv4.dequant_rows_dedup (same passes, same contracts) with the 288-byte-row owner dequant."""
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
    _kv4_stamp_kernel[(nblk,)](flat, n, n_pool, stamp, BLOCK=block, num_warps=4)
    _kv4_owner_count_kernel[(nblk,)](flat, n, n_pool, stamp, counts, BLOCK=block, num_warps=4)
    incl = torch.cumsum(counts, 0, dtype=torch.int32)
    _kv4_owner_compact_kernel[(nblk,)](flat, n, n_pool, stamp, incl, owners, pos, cap, BLOCK=block, num_warps=4)
    _fkv4_dequant_owner_kernel[(triton.cdiv(min(n, cap), block_r),)](
        rows, byte_lut(rows.device), flat, owners, incl[nblk - 1:], cap, n_pool, dst.view(torch.uint8),
        BLOCK_R=block_r, num_warps=num_warps)
    _kv4_remap_stamp_kernel[(nblk,)](flat, n, n_pool, stamp, pos, out, BLOCK=block, num_warps=4)
    _kv4_unstamp_kernel[(nblk,)](flat, n, n_pool, stamp, BLOCK=block, num_warps=4)
    return incl[nblk - 1:]


def sparse_mla(q: torch.Tensor, kv_buffer: torch.Tensor, table: torch.Tensor, sm_scale: float,
               out: torch.Tensor | None = None, block_h: int = 32, block_n: int = 64, num_warps: int = 8,
               num_stages: int = 2, splits: int = 8, workspace: tuple | None = None) -> torch.Tensor:
    """Fused sparse MLA over the FP4 pool: q [T, H, 512] fp8 (or bf16), table [T, K] int32 (-1 masked) -> [T, H, 512]
    bf16. Same flash-decoding split and combine as glm_kv4.sparse_mla."""
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
    assert block_h in (16, 32) and block_n in (16, 32, 64), (block_h, block_n)
    k = table.shape[1]
    splits = max(1, min(splits, triton.cdiv(k, block_n)))
    ks = triton.cdiv(triton.cdiv(k, splits), block_n) * block_n
    splits = triton.cdiv(k, ks)
    sm_scale_log2 = float(sm_scale) * 1.4426950408889634
    lut = byte_lut(rows.device)
    grid = (t, triton.cdiv(h, block_h), splits)
    if splits == 1:
        _fkv4_sparse_mla_kernel[grid](q.view(torch.uint8), q.stride(0), q.stride(1), rows, lut, rows.shape[0], table,
                                      table.stride(0), k, out, out.stride(0), out.stride(1), out, out, sm_scale_log2,
                                      H=h, BLOCK_H=block_h, BLOCK_N=block_n, SPLITS=1, KS=ks, num_warps=num_warps,
                                      num_stages=num_stages)
        return out
    if workspace is None:
        part_o = torch.empty(t, splits, h, DIM_NOPE, dtype=torch.float32, device=q.device)
        part_ml = torch.empty(t, splits, h, 2, dtype=torch.float32, device=q.device)
    else:
        part_o, part_ml = workspace
    _fkv4_sparse_mla_kernel[grid](q.view(torch.uint8), q.stride(0), q.stride(1), rows, lut, rows.shape[0], table,
                                  table.stride(0), k, out, out.stride(0), out.stride(1), part_o, part_ml,
                                  sm_scale_log2, H=h, BLOCK_H=block_h, BLOCK_N=block_n, SPLITS=splits, KS=ks,
                                  num_warps=num_warps, num_stages=num_stages)
    _kv4_combine_kernel[(t, triton.cdiv(h, block_h))](part_o, part_ml, out, out.stride(0), out.stride(1), H=h,
                                                      BLOCK_H=block_h, SPLITS=splits, num_warps=4)
    return out
