"""Marlin W4A16 MoE weights -> the cutlass_w4a8 layout, on the fly (k3amd, coordinator 2026-10-05: size-dispatched MoE
for the monolithic M layout).

Why: on the M layout (DP8 attention, TP8 MoE) one GLM-5.3 MoE layer is 576 MiB of int4 weights + 18 MiB of scales per
rank, 43.5 GiB for the 75 layers (derived from config.json: 256 experts, hidden 6144, inter 2048 / 8). Keeping a
Marlin copy next to the cutlass copy would take about the whole KV pool, so ONE layout is resident: Marlin (decode /
verify batches, -31..-35 % per call measured) and, for a batch above the dispatch size (prefill), this kernel writes
the layer's cutlass_w4a8 bytes into a one-layer scratch (594 MiB per rank) that the stock cutlass_w4a8_moe (+ the tuned
tiles) then reads.

Layouts (source):
  Marlin  gptq_marlin_moe_repack output int32 [E, K/16, 2N]; per expert tiles of 16 (k) x 64 (n), tile (kt, nt) at
          word (kt * N/64 + nt) * 128, word th*4 + warp of the tile holds the 8 nibbles
          vals[pack_idx[p]] at bits 4p, pack_idx = (0, 2, 4, 6, 1, 3, 5, 7), vals[i] = q(k=tc_row + off[i], n=cur_n),
          vals[4+i] = q(k=tc_row + off[i], n=cur_n + 8), off = (0, 1, 8, 9), tc_row = (th % 4) * 2, tc_col = th // 4,
          cur_n = warp * 16 + tc_col (sglang/kernels/jit/csrc/gemm/marlin/gptq_marlin_repack.cuh,
          gptq_marlin_repack_kernel without perm, num_bits 4); values uint4b8 (q + 8).
  cutlass int8 [E, N, K/2], two signed int4 per byte along K, low nibble = even k (w4afp8.py create_weights).
  scales  Marlin: marlin_permute_scales (marlin_utils.py:313) of [K/128, N] in groups of 64 columns by scale_perm;
          cutlass: interleave_scales (w4afp8.py:111) of [E, N, K/128] with alignment a -> [E, K/128/a, N a].

marlin_forward_reference is a direct numpy transcription of the CUDA repack loops; the Triton inverse is derived
algebraically, and the CPU tests check one against the other (operations/kernels/test_glm_marlin_inverse.py). The GPU
check against the real gptq_marlin_moe_repack runs in operations/modal/glm_moe_decode_bench.py --dispatch.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

TILE_K, TILE_N, TILE_WORDS = 16, 64, 128
PACK_IDX = (0, 2, 4, 6, 1, 3, 5, 7)
TC_OFFSETS = (0, 1, 8, 9)


def scale_perm() -> list:
    """marlin_utils.get_scale_perms()[0] (group_size < size_k)."""
    out = []
    for i in range(8):
        out.extend([i + 8 * j for j in range(8)])
    return out


def marlin_forward_reference(q: np.ndarray) -> np.ndarray:
    """q int [K, N] signed int4 values of ONE expert (k = reduction) -> Marlin int32 words [K/16 * N/64 * 128],
    transcribing gptq_marlin_repack_kernel (no perm, num_bits 4) loop by loop."""
    k_size, n_size = q.shape
    u = (q.astype(np.int64) + 8) & 0xF
    n_tiles = n_size // TILE_N
    out = np.zeros((k_size // TILE_K) * n_tiles * TILE_WORDS, dtype=np.uint32)
    for kt in range(k_size // TILE_K):
        for nt in range(n_tiles):
            base = (kt * n_tiles + nt) * TILE_WORDS
            for warp in range(4):
                for th in range(32):
                    tc_col, tc_row = th // 4, (th % 4) * 2
                    cur_n = warp * 16 + tc_col
                    vals = [0] * 8
                    for i in range(4):
                        k = kt * TILE_K + tc_row + TC_OFFSETS[i]
                        vals[i] = u[k, nt * TILE_N + cur_n]
                        vals[4 + i] = u[k, nt * TILE_N + cur_n + 8]
                    res = 0
                    for p in range(8):
                        res |= int(vals[PACK_IDX[p]]) << (p * 4)
                    out[base + th * 4 + warp] = res
    return out


def marlin_source(k: np.ndarray, n: np.ndarray, n_size: int):
    """(word, nibble) in an expert's Marlin words of element (k, n): the inverse of the loops above."""
    kt, k_in = k // TILE_K, k % TILE_K
    nt, n_in = n // TILE_N, n % TILE_N
    warp, r = n_in // 16, n_in % 16
    upper, tc_col = r // 8, r % 8
    ii = (k_in % 2) + 2 * (k_in // 8)
    tc_row = (k_in % 8) - (k_in % 2)
    th = tc_col * 4 + tc_row // 2
    j = ii + 4 * upper
    nib = j // 2 + 4 * (j % 2)
    word = (kt * (n_size // TILE_N) + nt) * TILE_WORDS + th * 4 + warp
    return word, nib


if HAVE_TRITON:
    @triton.jit
    def _marlin_to_cutlass_kernel(src, dst, N, K, words_per_expert, n_tiles, BLOCK_N: tl.constexpr,
                                  BLOCK_B: tl.constexpr):
        e = tl.program_id(0)
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
        b = tl.program_id(2) * BLOCK_B + tl.arange(0, BLOCK_B)[None, :]        # output byte along K/2
        mask = (n < N) & (b < K // 2)
        nt = n // 64
        n_in = n % 64
        warp = n_in // 16
        r = n_in % 16
        upper = r // 8
        tc_col = r % 8
        out = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.int32)
        for half in tl.static_range(2):
            k = 2 * b + half
            kt = k // 16
            k_in = k % 16
            ii = (k_in % 2) + 2 * (k_in // 8)
            tc_row = (k_in % 8) - (k_in % 2)
            th = tc_col * 4 + tc_row // 2
            j = ii + 4 * upper
            nib = j // 2 + 4 * (j % 2)
            word = (kt * n_tiles + nt) * 128 + th * 4 + warp
            w = tl.load(src + e.to(tl.int64) * words_per_expert + word, mask=mask, other=0)
            v = ((w >> (nib * 4)) & 0xF) - 8                                       # signed int4
            out = out | ((v & 0xF) << (4 * half))
        tl.store(dst + e.to(tl.int64) * N * (K // 2) + n * (K // 2) + b, out.to(tl.int8), mask=mask)


if HAVE_TRITON:
    @triton.jit
    def _marlin_words_kernel(src, dst32, lut, k_tiles, n_tiles, words_per_expert, out_row_words,
                             KT: tl.constexpr, FP8: tl.constexpr):
        """Word-parallel inverse (v2). Each Marlin word holds four WHOLE output byte pairs: for th = tc_col * 4 + j,
        nibbles p and p + 4 (p = 0..3) are q(k = 2j + 8 * (p % 2), n = n0 + 8 * (p // 2)) and the next k, with
        n0 = nt * 64 + warp * 16 + tc_col (vals[pack_idx] order of gptq_marlin_repack.cuh). The four words
        j = 0..3 of one (tc_col, warp) fill bytes 0..7 of their two rows' k-tile chunk, so the output is written as
        whole 32-bit words and no element is gathered from another word."""
        e = tl.program_id(0)
        nt = tl.program_id(1)
        kt0 = tl.program_id(2) * KT
        r = tl.arange(0, KT * 32)
        kt = kt0 + r // 32
        tc_col = (r // 4) % 8
        warp = r % 4
        ok = kt < k_tiles
        base = e.to(tl.int64) * words_per_expert + (kt.to(tl.int64) * n_tiles + nt) * 128 + tc_col * 16 + warp
        w0 = tl.load(src + base, mask=ok, other=0)
        w1 = tl.load(src + base + 4, mask=ok, other=0)
        w2 = tl.load(src + base + 8, mask=ok, other=0)
        w3 = tl.load(src + base + 12, mask=ok, other=0)
        n0 = nt * 64 + warp * 16 + tc_col
        row0 = (e.to(tl.int64) * (n_tiles * 64) + n0) * out_row_words
        row8 = row0 + 8 * out_row_words
        if FP8:
            # output FP8 [E, N, K]: a k-tile is 16 bytes = 4 words per row; slot p of word j -> bytes 2j, 2j + 1
            # (+ 8 for odd p) of row n0 (p < 2) or n0 + 8 (p >= 2)
            for p in tl.static_range(4):
                v0 = tl.load(lut + ((w0 >> (4 * p)) & 0xF)) | (tl.load(lut + ((w0 >> (16 + 4 * p)) & 0xF)) << 8)
                v1 = tl.load(lut + ((w1 >> (4 * p)) & 0xF)) | (tl.load(lut + ((w1 >> (16 + 4 * p)) & 0xF)) << 8)
                v2 = tl.load(lut + ((w2 >> (4 * p)) & 0xF)) | (tl.load(lut + ((w2 >> (16 + 4 * p)) & 0xF)) << 8)
                v3 = tl.load(lut + ((w3 >> (4 * p)) & 0xF)) | (tl.load(lut + ((w3 >> (16 + 4 * p)) & 0xF)) << 8)
                row = row0 if p < 2 else row8
                col = kt * 4 + (p % 2) * 2
                tl.store(dst32 + row + col, v0 | (v1 << 16), mask=ok)
                tl.store(dst32 + row + col + 1, v2 | (v3 << 16), mask=ok)
        else:
            # output int8 pairs [E, N, K/2]: a k-tile is 8 bytes = 2 words per row; signed nibble = u ^ 8
            for p in tl.static_range(4):
                b0 = (((w0 >> (4 * p)) & 0xF) | (((w0 >> (16 + 4 * p)) & 0xF) << 4)) ^ 0x88
                b1 = (((w1 >> (4 * p)) & 0xF) | (((w1 >> (16 + 4 * p)) & 0xF) << 4)) ^ 0x88
                b2 = (((w2 >> (4 * p)) & 0xF) | (((w2 >> (16 + 4 * p)) & 0xF) << 4)) ^ 0x88
                b3 = (((w3 >> (4 * p)) & 0xF) | (((w3 >> (16 + 4 * p)) & 0xF) << 4)) ^ 0x88
                row = row0 if p < 2 else row8
                tl.store(dst32 + row + kt * 2 + (p % 2), b0 | (b1 << 8) | (b2 << 16) | (b3 << 24), mask=ok)


if HAVE_TRITON:
    @triton.jit
    def _pairs_to_fp8_kernel(src32, dst32, lut_signed, n_words, BLOCK: tl.constexpr):
        """cutlass_w4a8 int8 pairs (low nibble = even k, two's complement) -> FP8 e4m3 bytes, 4 input bytes per
        word -> 8 output bytes (2 words)."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < n_words
        w = tl.load(src32 + i, mask=ok, other=0)
        lo = tl.zeros((BLOCK,), dtype=tl.int32)
        hi = tl.zeros((BLOCK,), dtype=tl.int32)
        for t in tl.static_range(2):
            v = tl.load(lut_signed + ((w >> (8 * t)) & 0xF)) | (tl.load(lut_signed + ((w >> (8 * t + 4)) & 0xF)) << 8)
            lo = lo | (v << (16 * t))
        for t in tl.static_range(2):
            v = tl.load(lut_signed + ((w >> (16 + 8 * t)) & 0xF)) | \
                (tl.load(lut_signed + ((w >> (16 + 8 * t + 4)) & 0xF)) << 8)
            hi = hi | (v << (16 * t))
        tl.store(dst32 + 2 * i, lo, mask=ok)
        tl.store(dst32 + 2 * i + 1, hi, mask=ok)


if HAVE_TRITON:
    @triton.jit
    def _marlin_chunk_kernel(src, dst32, k_tiles, n_tiles, words_per_expert, out_row_words, KT: tl.constexpr):
        """v3: one (n-tile, KT k-tiles) chunk per program. The chunk's Marlin words are loaded coalesced as
        [KT, tc_col 8, j 4, warp 4]; each word's four output byte slots p (= half + 2 * upper) are built, the four j
        words are merged into 32-bit output words, and the result is permuted to the output's own [64 rows,
        2 KT words] order, so the stores are row-contiguous (v2's per-lane row scatter ran at 0.55 TB/s)."""
        e = tl.program_id(0)
        nt = tl.program_id(1)
        kt0 = tl.program_id(2) * KT
        kt = kt0 + tl.arange(0, KT)[:, None, None, None]
        tc_col = tl.arange(0, 8)[None, :, None, None]
        j = tl.arange(0, 4)[None, None, :, None]
        warp = tl.arange(0, 4)[None, None, None, :]
        ok = kt < k_tiles
        w = tl.load(src + e.to(tl.int64) * words_per_expert + (kt.to(tl.int64) * n_tiles + nt) * 128
                    + tc_col * 16 + j * 4 + warp, mask=ok, other=0)                     # [KT, 8, 4, 4]
        b0 = (((w >> 0) & 0xF) | (((w >> 16) & 0xF) << 4)) ^ 0x88
        b1 = (((w >> 4) & 0xF) | (((w >> 20) & 0xF) << 4)) ^ 0x88
        b2 = (((w >> 8) & 0xF) | (((w >> 24) & 0xF) << 4)) ^ 0x88
        b3 = (((w >> 12) & 0xF) | (((w >> 28) & 0xF) << 4)) ^ 0x88
        sh = 8 * j
        v0 = tl.sum(b0 << sh, axis=2)                                                    # [KT, 8, 4]
        v1 = tl.sum(b1 << sh, axis=2)
        v2 = tl.sum(b2 << sh, axis=2)
        v3 = tl.sum(b3 << sh, axis=2)
        # slot p = half + 2 * upper: [KT, tc_col, warp, half, upper]
        v = tl.join(tl.join(v0, v1), tl.join(v2, v3))
        o = tl.reshape(tl.permute(v, (2, 4, 1, 0, 3)), (64, 2 * KT))                    # [(warp, upper, tc_col), (kt, half)]
        r = tl.arange(0, 64)[:, None]
        c = tl.arange(0, 2 * KT)[None, :]
        okc = (kt0 + c // 2) < k_tiles
        tl.store(dst32 + (e.to(tl.int64) * (n_tiles * 64) + nt * 64 + r) * out_row_words + kt0 * 2 + c, o, mask=okc)


def marlin_to_cutlass_weights_v3(w_marlin: torch.Tensor, n: int, k: int, out: torch.Tensor | None = None,
                                 kt_block: int = 16) -> torch.Tensor:
    e = w_marlin.shape[0]
    assert w_marlin.dtype == torch.int32 and w_marlin.is_contiguous()
    assert w_marlin.numel() == e * k * n // 8 and n % TILE_N == 0 and k % TILE_K == 0
    if out is None:
        out = torch.empty((e, n, k // 2), dtype=torch.int8, device=w_marlin.device)
    assert out.shape == (e, n, k // 2) and out.dtype == torch.int8 and out.is_contiguous()
    k_tiles = k // TILE_K
    grid = (e, n // TILE_N, triton.cdiv(k_tiles, kt_block))
    _marlin_chunk_kernel[grid](w_marlin, out.view(torch.int32), k_tiles, n // TILE_N, k * n // 8, k // 8, KT=kt_block)
    return out


_FP8_LUT: dict = {}
_FP8_LUT_SIGNED: dict = {}


def _fp8_lut_signed(device) -> torch.Tensor:
    """int32 [16]: the e4m3 byte of the two's-complement nibble (0..7 -> 0..7, 8..15 -> -8..-1)."""
    lut = _FP8_LUT_SIGNED.get(str(device))
    if lut is None:
        q = torch.tensor([x if x < 8 else x - 16 for x in range(16)], dtype=torch.float32)
        lut = _FP8_LUT_SIGNED[str(device)] = q.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32).to(device)
    return lut


def int4_pairs_to_fp8(w_packed: torch.Tensor, out: torch.Tensor | None = None, block: int = 1024) -> torch.Tensor:
    """int8 [..., K/2] cutlass_w4a8 pairs -> float8_e4m3fn [..., K] with the int4 values exactly (no scales)."""
    assert w_packed.dtype == torch.int8 and w_packed.is_contiguous() and w_packed.shape[-1] % 4 == 0
    if out is None:
        out = torch.empty((*w_packed.shape[:-1], w_packed.shape[-1] * 2), dtype=torch.float8_e4m3fn,
                          device=w_packed.device)
    assert out.dtype == torch.float8_e4m3fn and out.is_contiguous() and out.numel() == 2 * w_packed.numel()
    n_words = w_packed.numel() // 4
    _pairs_to_fp8_kernel[(triton.cdiv(n_words, block),)](w_packed.view(torch.int32), out.view(torch.int32),
                                                        _fp8_lut_signed(w_packed.device), n_words, BLOCK=block)
    return out


def _fp8_lut(device) -> torch.Tensor:
    """int32 [16]: the e4m3 byte of q = u - 8 for each uint4b8 nibble u (exact: -8..7 are e4m3 values)."""
    lut = _FP8_LUT.get(str(device))
    if lut is None:
        q = torch.arange(16, dtype=torch.float32) - 8
        lut = _FP8_LUT[str(device)] = q.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32).to(device)
    return lut


def _marlin_words(w_marlin: torch.Tensor, n: int, k: int, out: torch.Tensor, fp8: bool, kt_block: int = 16):
    e = w_marlin.shape[0]
    assert w_marlin.dtype == torch.int32 and w_marlin.is_contiguous()
    assert w_marlin.numel() == e * k * n // 8, (w_marlin.shape, n, k)
    assert out.is_contiguous() and n % TILE_N == 0 and k % TILE_K == 0
    k_tiles = k // TILE_K
    row_words = (k if fp8 else k // 2) // 4
    grid = (e, n // TILE_N, triton.cdiv(k_tiles, kt_block))
    _marlin_words_kernel[grid](w_marlin, out.view(torch.int32), _fp8_lut(w_marlin.device), k_tiles, n // TILE_N,
                               k * n // 8, row_words, KT=kt_block, FP8=fp8)
    return out


def marlin_to_cutlass_weights(w_marlin: torch.Tensor, n: int, k: int, out: torch.Tensor | None = None,
                              kt_block: int = 16) -> torch.Tensor:
    """int32 [E, K/16, 2N] Marlin -> int8 [E, N, K/2] cutlass_w4a8 (into `out` when given: the layer scratch).
    The v3 chunk kernel: measured on 1xH200 (moe-r3, glm_moe_decode_bench_h200_20261005-141235.json) 232.6 us for a
    TP8 w13 layer (384 MiB) and 111.5 us for w2, bit-exact; v2 took 1,458 / 806 us (row-scattered stores)."""
    return marlin_to_cutlass_weights_v3(w_marlin, n, k, out=out, kt_block=kt_block)


def marlin_to_cutlass_weights_v2(w_marlin: torch.Tensor, n: int, k: int, out: torch.Tensor | None = None,
                                 kt_block: int = 16) -> torch.Tensor:
    """The word-parallel v2 (store-bound, 0.55 TB/s); kept as a cross-check."""
    e = w_marlin.shape[0]
    if out is None:
        out = torch.empty((e, n, k // 2), dtype=torch.int8, device=w_marlin.device)
    assert out.shape == (e, n, k // 2) and out.dtype == torch.int8
    return _marlin_words(w_marlin, n, k, out, False, kt_block)


def marlin_to_fp8_weights(w_marlin: torch.Tensor, n: int, k: int, out: torch.Tensor | None = None,
                          kt_block: int = 16) -> torch.Tensor:
    """int32 [E, K/16, 2N] Marlin -> float8_e4m3fn [E, N, K] holding the int4 values exactly (scales not applied)."""
    e = w_marlin.shape[0]
    if out is None:
        out = torch.empty((e, n, k), dtype=torch.float8_e4m3fn, device=w_marlin.device)
    assert out.shape == (e, n, k) and out.dtype == torch.float8_e4m3fn
    _marlin_words(w_marlin, n, k, out.view(torch.uint8), True, kt_block)
    return out


def marlin_to_cutlass_weights_v1(w_marlin: torch.Tensor, n: int, k: int, out: torch.Tensor | None = None,
                                 block_n: int = 64, block_b: int = 64) -> torch.Tensor:
    """The element-gather v1 (2,453 us per TP8 layer on H200, moe-dispatch-2): kept as a cross-check."""
    e = w_marlin.shape[0]
    assert w_marlin.dtype == torch.int32 and w_marlin.is_contiguous()
    assert w_marlin.numel() == e * k * n // 8, (w_marlin.shape, n, k)
    if out is None:
        out = torch.empty((e, n, k // 2), dtype=torch.int8, device=w_marlin.device)
    assert out.shape == (e, n, k // 2) and out.dtype == torch.int8 and out.is_contiguous()
    grid = (e, triton.cdiv(n, block_n), triton.cdiv(k // 2, block_b))
    _marlin_to_cutlass_kernel[grid](w_marlin, out, n, k, k * n // 8, n // TILE_N, BLOCK_N=block_n, BLOCK_B=block_b)
    return out


_INV_SCALE_PERM: dict = {}


def marlin_to_raw_scales(s_marlin: torch.Tensor) -> torch.Tensor:
    """Marlin scales [E, K/128, N] (marlin_moe_permute_scales) -> [E, K/128, N] before the permutation
    (= scales_to_marlin_input of the checkpoint scales)."""
    dev = s_marlin.device
    inv = _INV_SCALE_PERM.get(dev)
    if inv is None:
        inv = _INV_SCALE_PERM[dev] = torch.as_tensor(np.argsort(np.array(scale_perm())), device=dev)
    e, g, n = s_marlin.shape
    return s_marlin.reshape(e, -1, 64)[:, :, inv].reshape(e, g, n)


def cutlass_scales(s_raw_gn: torch.Tensor, alignment: int) -> torch.Tensor:
    """[E, K/128, N] -> interleave_scales([E, N, K/128]) with the given alignment: [E, K/128/a, N a]."""
    e, g, n = s_raw_gn.shape
    s = s_raw_gn.permute(0, 2, 1)                                # [E, N, G]
    return s.reshape(e, n, g // alignment, alignment).permute(0, 2, 1, 3).reshape(e, g // alignment, n * alignment)


def stock_alignment(groups: int) -> int:
    """w4afp8.interleave_scales: 4 if the group count divides by 4, else 1."""
    return 4 if groups % 4 == 0 else 1


# ------------------------------------------------------------------------------------------------ size dispatch
DISPATCH_ENV = "NUMINOUS_W4A8_DISPATCH_M"


def dispatch_threshold() -> int:
    """MoE batch size (tokens into the layer; under DP attention the gathered batch) at or below which Marlin runs."""
    v = int(__import__("os").environ.get(DISPATCH_ENV, "0") or 0)
    if v <= 0:
        raise ValueError(f"NUMINOUS_W4A8_MARLIN=dispatch needs {DISPATCH_ENV}=<tokens> > 0, got {v}")
    return v


def tiles_alignment(n: int, k: int) -> int:
    """The scale interleave the cutlass path expects for the (N, K) GEMM: the tuned tiles' NUMINOUS_W4A8_TILES_ALIGN
    entry when set (glm_w4a8_tiles_hooks), else w4afp8.interleave_scales' stock rule."""
    for part in filter(None, (p.strip() for p in __import__("os").environ.get("NUMINOUS_W4A8_TILES_ALIGN", "").split(";"))):
        shape, a = part.split("=")
        if tuple(int(x) for x in shape.split("x")) == (n, k):
            return int(a)
    return stock_alignment(k // 128)


class DispatchScratch:
    """One layer's cutlass_w4a8 tensors, shared by every MoE layer of the same shape on a device (the layers run one
    after another). Allocated at weight load, before the KV pool is sized, so the pool accounts for it."""

    def __init__(self, e: int, n13: int, k13: int, n2: int, k2: int, device, dtype=torch.bfloat16):
        self.threshold = dispatch_threshold()
        self.shape = (e, n13, k13, n2, k2)
        self.a13, self.a2 = tiles_alignment(n13, k13), tiles_alignment(n2, k2)
        self.w13 = torch.empty((e, n13, k13 // 2), dtype=torch.int8, device=device)
        self.w2 = torch.empty((e, n2, k2 // 2), dtype=torch.int8, device=device)
        g13, g2 = k13 // 128, k2 // 128
        self.s13 = torch.empty((e, g13 // self.a13, n13 * self.a13), dtype=dtype, device=device)
        self.s2 = torch.empty((e, g2 // self.a2, n2 * self.a2), dtype=dtype, device=device)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w13, self.w2, self.s13, self.s2))

    def cutlass_weights(self, layer):
        """Rebuild this layer's cutlass_w4a8 weights and interleaved scales from its Marlin tensors (into the scratch)."""
        e, n13, k13, n2, k2 = self.shape
        marlin_to_cutlass_weights(layer.w13_weight, n13, k13, out=self.w13)
        marlin_to_cutlass_weights(layer.w2_weight, n2, k2, out=self.w2)
        self.s13.copy_(cutlass_scales(marlin_to_raw_scales(layer.w13_weight_scale_inv), self.a13))
        self.s2.copy_(cutlass_scales(marlin_to_raw_scales(layer.w2_weight_scale_inv), self.a2))
        return self.w13, self.w2, self.s13, self.s2


_SCRATCH: dict = {}


def dispatch_scratch(e: int, n13: int, k13: int, n2: int, k2: int, device) -> DispatchScratch:
    key = (str(device), e, n13, k13, n2, k2)
    s = _SCRATCH.get(key)
    if s is None:
        s = _SCRATCH[key] = DispatchScratch(e, n13, k13, n2, k2, device)
        import logging
        logging.getLogger("sglang.numinous").info(
            "numinous W4A8 size dispatch: Marlin for MoE batches <= %d tokens, cutlass_w4a8 (+tiles) above, from a "
            "%.0f MiB one-layer scratch (E %d, w13 %dx%d, w2 %dx%d, scale align %d/%d)", s.threshold,
            s.nbytes() / 2 ** 20, e, n13, k13, n2, k2, s.a13, s.a2)
    return s
