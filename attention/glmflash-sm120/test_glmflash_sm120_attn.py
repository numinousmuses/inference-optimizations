"""CPU tests for glmflash_sm120_attn (Triton interpreter; run with system python3 which has triton + numpy).

    TRITON_INTERPRET=1 python3 -m pytest -q operations/kernels/test_glmflash_sm120_attn.py
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest
import torch

import glmflash_sm120_attn as A

pytestmark = pytest.mark.skipif(not A.HAVE_TRITON, reason="triton not installed")


def _fp4_rows(latent: torch.Tensor) -> torch.Tensor:
    import glmflash_kv4
    return glmflash_kv4.quantize_rows_ref(latent)


def _pool(fmt: int, n: int, seed: int = 0):
    """Returns (pool buffer shaped like SGLang's [n, 1, row], fp32 dequantized reference rows [n, 512])."""
    g = torch.Generator().manual_seed(seed)
    lat = torch.randn(n, 512, generator=g) * 0.7
    if fmt == A.FMT_BF16:
        buf = lat.to(torch.bfloat16).reshape(n, 1, 512)
        return buf, buf.reshape(n, 512).float()
    if fmt == A.FMT_FP8_656:
        rows = A.nsa656_quantize_ref(lat)
        return rows.reshape(n, 1, 656), A.nsa656_dequant_ref(rows)
    rows = _fp4_rows(lat)
    return rows.reshape(n, 1, 288), A.fp4_288_dequant_ref(rows)


def _table(t: int, k: int, n: int, n_valid, seed: int = 1) -> torch.Tensor:
    """Compact kpool-style table: first n_valid[i] entries distinct slots in [1, n), rest -1."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.full((t, k), -1, dtype=torch.int32)
    for i in range(t):
        v = int(n_valid[i])
        idx[i, :v] = (torch.randperm(n - 1, generator=g)[:v] + 1).to(torch.int32)
    return idx


def _interp_cast_e4m3(y: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3 bytes through the Triton interpreter's own conversion. The interpreter's cast is NOT IEEE
    (measured 2026-09-30, triton 3.7.1: 7.9 -> 4.0, the mantissa carry into the exponent is lost for some binades;
    400 -> 416, ties away from even), so a kernel run under TRITON_INTERPRET is compared against this cast; the GPU
    test compares against torch's cast bit for bit."""
    import triton
    import triton.language as tl

    @triton.jit
    def k(x_ptr, y_ptr, N: tl.constexpr):
        o = tl.arange(0, N)
        tl.store(y_ptr + o, tl.load(x_ptr + o).to(tl.float8e4nv).to(tl.uint8, bitcast=True))

    flat = y.reshape(-1).contiguous()
    out = torch.zeros(flat.numel(), dtype=torch.uint8)
    for i in range(0, flat.numel(), 512):
        k[(1,)](flat[i:i + 512], out[i:i + 512], N=512)
    return out.reshape(y.shape)


def _nsa656_ref_interp(latent: torch.Tensor, pow2: bool = False) -> torch.Tensor:
    t = latent.shape[0]
    x = latent.float().reshape(t, 4, 128)
    amax = x.abs().amax(-1)
    s = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    if pow2:
        s = A.pow2_ceil_ref(s)
    y = (x * (1.0 / s).unsqueeze(-1)).clamp(-448.0, 448.0)
    out = torch.zeros(t, 656, dtype=torch.uint8)
    out[:, :512] = _interp_cast_e4m3(y).reshape(t, 512)
    out[:, 512:528] = s.contiguous().view(torch.uint8).reshape(t, 16)
    return out


# ------------------------------------------------------------------------------------------------ formats


def test_fp4_reference_matches_glmflash_kv4():
    import glmflash_kv4
    lat = torch.randn(9, 512)
    rows = glmflash_kv4.quantize_rows_ref(lat)
    assert torch.equal(A.fp4_288_dequant_ref(rows), glmflash_kv4.dequantize_rows_ref(rows, torch.float32))


def test_nsa656_layout_is_flashinfer_glm_nsa():
    lat = torch.randn(5, 512) * 3
    rows = A.nsa656_quantize_ref(lat)
    assert rows.shape == (5, 656)
    assert torch.all(rows[:, 528:] == 0), "rope bytes must be zero (q_rope = 0 x k_rope = 0)"
    s = rows[:, 512:528].contiguous().view(torch.float32)
    amax = lat.reshape(5, 4, 128).abs().amax(-1)
    assert torch.allclose(s, amax / 448.0)  # arbitrary fp32 scale, kv_scale_format="arbitrary_fp32"
    rel = (A.nsa656_dequant_ref(rows) - lat).norm() / lat.norm()
    assert rel < 0.04


def test_nsa656_zero_block_is_finite():
    lat = torch.randn(3, 512)
    lat[1, 128:256] = 0
    rows = A.nsa656_quantize_ref(lat)
    deq = A.nsa656_dequant_ref(rows)
    assert torch.isfinite(deq).all() and torch.all(deq[1, 128:256] == 0)


def test_e2m1_decode_kernel_matches_table():
    import triton
    import triton.language as tl

    @triton.jit
    def k(c_ptr, o_ptr):
        c = tl.load(c_ptr + tl.arange(0, 16))
        tl.store(o_ptr + tl.arange(0, 16), A._e2m1_value(c))

    c = torch.arange(16, dtype=torch.int32)
    o = torch.zeros(16)
    k[(1,)](c, o)
    mag = A.E2M1_VALUES[c & 7]
    ref = torch.where((c & 8) != 0, -mag, mag)
    assert torch.equal(o, ref)


# ------------------------------------------------------------------------------------------------ attention


@pytest.mark.parametrize("fmt", [A.FMT_BF16, A.FMT_FP8_656, A.FMT_FP4_288])
@pytest.mark.parametrize("splits", [1, 3])
def test_sparse_mla_matches_reference(fmt, splits):
    n, t, h, k = 300, 3, 16, 70
    buf, ref_rows = _pool(fmt, n)
    q = (torch.randn(t, h, 512) * 0.2).to(torch.bfloat16)
    idx = _table(t, k, n, [70, 37, 5])
    scale = 1.0 / (256 ** 0.5)
    out, lse = A.sparse_mla(q, buf, idx, scale, fmt, splits=splits, block_n=16)
    ro, rl = A.sparse_mla_ref(q.float(), ref_rows, idx, scale)
    err = (out.float() - ro).norm() / ro.norm()
    assert err < 1e-2, err
    assert torch.allclose(lse, rl, atol=1e-3, rtol=1e-4)


def test_sparse_mla_empty_row_and_head_padding():
    n, t, h, k = 64, 2, 12, 20  # 12 heads: a padded head block
    buf, ref_rows = _pool(A.FMT_FP8_656, n)
    q = torch.randn(t, h, 512).to(torch.bfloat16)
    idx = _table(t, k, n, [20, 0])
    out, lse = A.sparse_mla(q, buf, idx, 0.05, A.FMT_FP8_656, splits=2, block_n=16)
    assert torch.all(out[1] == 0) and torch.all(lse[1] == A.LSE_EMPTY)
    ro, rl = A.sparse_mla_ref(q.float(), ref_rows, idx, 0.05)
    assert (out[0].float() - ro[0]).norm() / ro[0].norm() < 1e-2


def test_sparse_mla_q_view_of_576_buffer_and_kpool_width():
    """q as the [:, :, :512] view of a zero-rope [T, H, 576] buffer (the FlashInfer operand) and a 2051-wide table
    whose last valid entries sit in the tail columns 2048..2050."""
    n, t, h, k = 2200, 1, 16, 2051
    buf, ref_rows = _pool(A.FMT_FP4_288, n, seed=3)
    q576 = torch.zeros(t, h, 576, dtype=torch.bfloat16)
    q576[:, :, :512] = torch.randn(t, h, 512).to(torch.bfloat16) * 0.1
    idx = _table(t, k, n, [2051])
    out, lse = A.sparse_mla(q576[:, :, :512], buf, idx, 0.0625, A.FMT_FP4_288, block_n=32)
    ro, rl = A.sparse_mla_ref(q576[:, :, :512].float(), ref_rows, idx, 0.0625)
    assert (out.float() - ro).norm() / ro.norm() < 1e-2
    # dropping the 3 tail columns must change the answer (the tail is attended)
    ro2, _ = A.sparse_mla_ref(q576[:, :, :512].float(), ref_rows, idx[:, :2048], 0.0625)
    assert (ro2 - ro).abs().max() > 0


def test_merge_attn_equals_single_pass():
    """tail + main merged (log2 LSE) == attention over the union: the no-JIT FlashInfer fallback's contract."""
    n, t, h = 400, 2, 16
    buf, ref_rows = _pool(A.FMT_FP8_656, n, seed=5)
    q = torch.randn(t, h, 512).to(torch.bfloat16) * 0.3
    idx = _table(t, 67, n, [67, 50])
    main = idx[:, :64].contiguous()
    tail = idx[:, 64:].contiguous()
    oa, la = A.sparse_mla(q, buf, main, 0.06, A.FMT_FP8_656, block_n=16)
    ob, lb = A.sparse_mla(q, buf, tail, 0.06, A.FMT_FP8_656, block_n=16)
    A.merge_attn(oa, la, ob, lb)
    ro, rl = A.sparse_mla_ref(q.float(), ref_rows, idx, 0.06)
    assert (oa.float() - ro).norm() / ro.norm() < 1e-2
    assert torch.allclose(la, rl, atol=1e-3)
    mo, ml = A.merge_ref(*A.sparse_mla_ref(q.float(), ref_rows, main, 0.06),
                         *A.sparse_mla_ref(q.float(), ref_rows, tail, 0.06))
    assert (mo - ro).abs().max() < 1e-4 and torch.allclose(ml, rl, atol=1e-4)


def test_nsa656_quant_write_kernel_bitexact_and_skips_slot0():
    n = 130
    buf = torch.zeros(n, 1, 656, dtype=torch.uint8)
    lat = (torch.randn(6, 512) * 2).to(torch.bfloat16)
    lat[2, :128] = 0
    loc = torch.tensor([0, 5, 64, 129, -1, 7], dtype=torch.int64)
    A.nsa656_quant_write(buf, loc, lat)
    ref = _nsa656_ref_interp(lat.float())
    rows = buf.reshape(n, 656)
    assert torch.all(rows[0] == 0), "slot 0 is the padding slot and must stay zero"
    for i, l in enumerate(loc.tolist()):
        if 0 < l < n:
            assert torch.equal(rows[l], ref[i]), i
    assert torch.isfinite(A.nsa656_dequant_ref(rows)).all()


def test_choose_splits():
    assert A.choose_splits(1024, 16, 2051) == 1  # prefill fills the GPU
    s1 = A.choose_splits(1, 16, 2051)
    assert 8 <= s1 <= 16  # 65 tiles of 32, >= 4 tiles per split
    assert A.choose_splits(64, 16, 2051) == 6
    assert A.choose_splits(64, 64, 2051) == 2


def test_derived_bytes():
    assert A.bytes_per_token(A.FMT_FP8_656) == 11 * 656
    assert abs(A.roofline_us(A.FMT_FP8_656, 64) - 64 * 11 * 2051 * 656 / 1.792e6) < 1e-6


def test_e2m1_bf16_bit_construction():
    import triton
    import triton.language as tl

    @triton.jit
    def k(c_ptr, o_ptr):
        c = tl.load(c_ptr + tl.arange(0, 16))
        tl.store(o_ptr + tl.arange(0, 16), A._e2m1_bf16(c).to(tl.float32))

    c = torch.arange(16, dtype=torch.int16)
    o = torch.zeros(16)
    k[(1,)](c, o)
    mag = A.E2M1_VALUES[c.long() & 7]
    assert torch.equal(o, torch.where((c & 8) != 0, -mag, mag))



def test_pow2_scales_are_powers_of_two_and_cover_amax():
    lat = torch.randn(50, 512) * torch.rand(50, 1) * 10
    rows = A.nsa656_quantize_ref(lat, pow2=True)
    s = rows[:, 512:528].contiguous().view(torch.float32)
    m, e = torch.frexp(s)
    assert torch.all(m == 0.5), "every scale is an exact power of two"
    amax = lat.reshape(50, 4, 128).abs().amax(-1)
    assert torch.all(s >= amax / 448.0) and torch.all(s < 2 * amax / 448.0)
    # e4m3 relative precision is scale-independent: the pow2 row's error is within ~1.2x of the arbitrary row's
    ea = (A.nsa656_dequant_ref(A.nsa656_quantize_ref(lat)) - lat).norm()
    ep = (A.nsa656_dequant_ref(rows) - lat).norm()
    assert ep < 1.25 * ea, (float(ep), float(ea))
    assert torch.equal(A.pow2_ceil_ref(torch.tensor([1.0, 1.5, 0.25, 3.0001, 448.0])),
                       torch.tensor([1.0, 2.0, 0.25, 4.0, 512.0]))


def test_nsa656_quant_write_pow2_kernel():
    n = 70
    buf = torch.zeros(n, 1, 656, dtype=torch.uint8)
    lat = (torch.randn(5, 512) * 3).to(torch.bfloat16)
    loc = torch.tensor([3, 9, 0, 41, 69])
    A.nsa656_quant_write(buf, loc, lat, pow2=True)
    ref = _nsa656_ref_interp(lat.float(), pow2=True)
    rows = buf.reshape(n, 656)
    for i, l in enumerate(loc.tolist()):
        if l > 0:
            assert torch.equal(rows[l], ref[i]), i
    assert torch.all(rows[0] == 0)
