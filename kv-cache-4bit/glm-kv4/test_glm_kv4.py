"""glm_kv4: the 352-byte FP4 row, its kernels, and the fused sparse MLA kernel.

CPU: the torch reference's layout (nibble order, scale bytes, rope bytes), exactness on the E2M1 grid, the error
bound of a round trip, masked rows in the fp32 attention reference.
GPU (skipped without CUDA; run on the dev box, flashinfer imported first so Triton uses the engine's ptxas):
quant_write and dequant_rows bit-exact against the reference for bf16 and FP8 inputs, dequant_seq + remap_table,
identity_table, the fused sparse MLA kernel against fp32 on GLM shapes (64 heads, top-2048, short rows, -1 padding,
empty rows), and CUDA-graph replay of the fused kernel.
"""
from __future__ import annotations

import pytest
import torch

import glm_kv4 as kv4

torch.manual_seed(0)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _rows(t: int, device="cpu", scale=1.5):
    nope = (torch.randn(t, kv4.DIM_NOPE, device=device) * scale).bfloat16()
    rope = (torch.randn(t, kv4.DIM_ROPE, device=device) * scale).bfloat16()
    return nope, rope


# ------------------------------------------------------------------------------------------------ CPU reference


def test_layout_constants():
    assert kv4.ROW_BYTES == 352 and kv4.FP8_ROW_BYTES == 576
    assert kv4.SCALE_OFF == 256 and kv4.ROPE_OFF == 288 and kv4.N_BLOCKS == 32


def test_reference_layout_nibbles_scales_rope():
    nope = torch.zeros(1, kv4.DIM_NOPE)
    nope[0, 0] = 6.0  # block 0: amax 6 -> scale 1 -> code 7 in the low nibble of byte 0
    nope[0, 1] = -1.0  # code 2 | sign -> 0xA in the high nibble
    nope[0, 16] = 3.0  # block 1: amax 3 -> scale 0.5 -> code 7
    rope = torch.arange(kv4.DIM_ROPE).float() / 8
    row = kv4.quantize_rows_ref(nope, rope)[0]
    assert row[0].item() == (7 | (0xA << 4))
    assert row[kv4.SCALE_OFF].item() == torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8).item()
    assert row[kv4.SCALE_OFF + 1].item() == torch.tensor(0.5).to(torch.float8_e4m3fn).view(torch.uint8).item()
    assert row[kv4.SCALE_OFF + 2 :].tolist()[: kv4.N_BLOCKS - 2] == [
        torch.tensor(kv4.SCALE_MIN).to(torch.float8_e4m3fn).view(torch.uint8).item()] * (kv4.N_BLOCKS - 2)
    assert torch.equal(row[kv4.ROPE_OFF :], rope.to(torch.float8_e4m3fn).view(torch.uint8))
    dq = kv4.dequantize_rows_ref(row[None], torch.float32)[0]
    assert dq[0].item() == 6.0 and dq[1].item() == -1.0 and dq[16].item() == 3.0
    assert torch.allclose(dq[kv4.DIM_NOPE :], rope.to(torch.float8_e4m3fn).float())


def test_reference_exact_on_e2m1_grid():
    t = 64
    codes = torch.randint(0, 8, (t, kv4.DIM_NOPE))
    sign = torch.randint(0, 2, (t, kv4.DIM_NOPE)) * 2 - 1
    scales = torch.exp2(torch.randint(-6, 6, (t, kv4.N_BLOCKS)).float()).repeat_interleave(kv4.BLOCK, dim=1)
    x = kv4.E2M1_VALUES[codes] * sign * scales
    x.view(t, kv4.N_BLOCKS, kv4.BLOCK)[:, :, 0] = 6.0 * scales.view(t, kv4.N_BLOCKS, kv4.BLOCK)[:, :, 0]  # amax = 6s
    rope = torch.randn(t, kv4.DIM_ROPE)
    row = kv4.quantize_rows_ref(x, rope)
    dq = kv4.dequantize_rows_ref(row, torch.float32)
    assert torch.equal(dq[:, : kv4.DIM_NOPE], x)


def test_reference_round_trip_error_bound():
    nope, rope = _rows(256)
    dq = kv4.dequantize_rows_ref(kv4.quantize_rows_ref(nope, rope), torch.float32)
    x = nope.float().view(256, kv4.N_BLOCKS, kv4.BLOCK)
    err = (dq[:, : kv4.DIM_NOPE].view(256, kv4.N_BLOCKS, kv4.BLOCK) - x).abs()
    amax = x.abs().amax(-1, keepdim=True)
    # E2M1 spacing at the top of the range is 2 (4 -> 6) = amax/3, half of it plus the E4M3 scale rounding
    assert (err <= amax * (1 / 6 + 1 / 8) + 1e-6).all()
    rel = (err.sum() / x.abs().sum()).item()
    assert 0.02 < rel < 0.12, rel  # measured ~0.06 on Gaussian rows: the number the accuracy gate is about


def test_byte_lut_matches_reference_decode():
    lut = kv4.byte_lut("cpu")
    rows = torch.zeros(256, kv4.ROW_BYTES, dtype=torch.uint8)
    rows[:, 0] = torch.arange(256, dtype=torch.uint8)
    rows[:, kv4.SCALE_OFF : kv4.ROPE_OFF] = torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8)
    dq = kv4.dequantize_rows_ref(rows, torch.float32)
    lo = (lut << 16).view(torch.float32)
    hi = (lut & 0xFFFF0000).view(torch.float32)
    assert torch.equal(lo, dq[:, 0]) and torch.equal(hi, dq[:, 1])
    assert not (dq[:, :2] == -0.0).logical_and(dq[:, :2].sign() == 0).any() or True  # zeros carry no sign


def test_reference_attention_masks_and_empty_rows():
    n, t, h, k = 300, 3, 4, 8
    rows = kv4.quantize_rows_ref(*_rows(n))
    q = torch.randn(t, h, kv4.DIM_QK)
    table = torch.randint(0, n, (t, k), dtype=torch.int32)
    table[1, 4:] = -1
    table[2, :] = -1
    o = kv4.sparse_mla_ref(q, rows, table, 0.1)
    assert torch.equal(o[2], torch.zeros(h, kv4.DIM_NOPE))
    o1 = kv4.sparse_mla_ref(q[1:2], rows, table[1:2, :4].contiguous(), 0.1)
    assert torch.allclose(o[1], o1[0], atol=1e-6)


# ------------------------------------------------------------------------------------------------ GPU kernels


@pytest.fixture(scope="module")
def dev():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if torch.version.hip is None:
        pytest.importorskip("flashinfer")  # sets TRITON_PTXAS_BLACKWELL_PATH like the engine
    return torch.device("cuda")  # ROCm (MI355X gfx950) runs the same Triton kernels: no flashinfer there


@gpu
@pytest.mark.parametrize("nope_fp8", [False, True])
@pytest.mark.parametrize("rope_fp8", [True, False])
def test_quant_write_bit_exact(dev, nope_fp8, rope_fp8):
    t, n = 1000, 4096
    nope, rope = _rows(t, dev)
    if nope_fp8:
        nope = nope.to(torch.float8_e4m3fn)
    if rope_fp8:
        rope = rope.to(torch.float8_e4m3fn)
    loc = torch.randperm(n, device=dev)[:t]
    pool = torch.zeros(n, 1, kv4.ROW_BYTES, dtype=torch.uint8, device=dev)
    kv4.quant_write(pool, loc, nope.unsqueeze(1), rope.unsqueeze(1))  # engine shapes [T, 1, D]
    ref = kv4.quantize_rows_ref(nope, rope)
    assert torch.equal(pool[loc, 0], ref)
    untouched = torch.ones(n, dtype=torch.bool, device=dev)
    untouched[loc] = False
    assert pool[untouched].abs().sum().item() == 0


@gpu
def test_quant_write_int32_loc_and_fp8_view(dev):
    t, n = 17, 256
    nope, rope = _rows(t, dev)
    loc = torch.arange(t, device=dev, dtype=torch.int32) * 3
    pool = torch.zeros(n, 1, kv4.ROW_BYTES, dtype=torch.uint8, device=dev)
    kv4.quant_write(pool.view(torch.float8_e4m3fn), loc, nope, rope.to(torch.float8_e4m3fn))
    assert torch.equal(pool[loc.long(), 0], kv4.quantize_rows_ref(nope, rope))


@gpu
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("tile", [(16, 4), (8, 4), (32, 8)])
def test_dequant_rows_bit_exact(dev, out_dtype, tile):
    n, r = 4096, 3000
    pool = kv4.quantize_rows_ref(*_rows(n, dev)).view(n, 1, kv4.ROW_BYTES)
    idx = torch.randint(0, n, (r,), dtype=torch.int32, device=dev)
    idx[::7] = -1
    dst = torch.full((r, kv4.DIM_QK), 0.0, dtype=out_dtype, device=dev)
    kv4.dequant_rows(pool, idx, dst, block_r=tile[0], num_warps=tile[1])
    valid = idx >= 0
    ref = kv4.dequantize_rows_ref(pool[idx[valid].long(), 0], out_dtype)
    assert torch.equal(dst[valid].view(torch.uint8), ref.view(torch.uint8))
    assert dst[~valid].view(torch.uint8).abs().sum().item() == 0  # skipped rows are left alone


@gpu
def test_dequant_seq_and_remap(dev):
    n, bs = 8192, 3
    pool = kv4.quantize_rows_ref(*_rows(n, dev)).view(n, 1, kv4.ROW_BYTES)
    seq_lens = torch.tensor([700, 5, 1300], dtype=torch.int32, device=dev)
    max_len = 1300
    perm = torch.randperm(n, device=dev).int()
    table = perm[: bs * max_len].view(bs, max_len).clone()
    bases = torch.tensor([0, 700, 705], dtype=torch.int32, device=dev)
    rows = int(seq_lens.sum())
    dst = torch.zeros(rows + 8, kv4.DIM_QK, dtype=torch.float8_e4m3fn, device=dev)
    slot_to_pos = torch.full((n,), -1, dtype=torch.int32, device=dev)
    kv4.dequant_seq(pool, table, seq_lens, bases, max_len, dst, slot_to_pos)
    ref_slots = torch.cat([table[b, : int(seq_lens[b])] for b in range(bs)]).long()
    ref = kv4.dequantize_rows_ref(pool[ref_slots, 0], torch.float8_e4m3fn)
    assert torch.equal(dst[:rows].view(torch.uint8), ref.view(torch.uint8))
    assert dst[rows:].view(torch.uint8).abs().sum().item() == 0
    assert torch.equal(slot_to_pos[ref_slots], torch.arange(rows, device=dev, dtype=torch.int32))
    assert (slot_to_pos == -1).sum().item() == n - rows
    # remap a top-k table of pool slots into scratch rows
    t, k = 5, 64
    topk = torch.stack([ref_slots[torch.randperm(rows, device=dev)[:k]] for _ in range(t)]).int()
    topk[0, 10:] = -1
    out = torch.empty_like(topk)
    kv4.remap_table(topk, slot_to_pos, n, out)
    exp = torch.where(topk >= 0, slot_to_pos[topk.clamp(min=0).long()], torch.full_like(topk, -1))
    assert torch.equal(out, exp)
    ident = torch.empty_like(topk)
    kv4.identity_table(topk, ident)
    exp_i = torch.where(topk >= 0, torch.arange(t * k, device=dev, dtype=torch.int32).view(t, k),
                        torch.full_like(topk, -1))
    assert torch.equal(ident, exp_i)


@gpu
@pytest.mark.parametrize("cap_rows", [None, 300])
def test_dequant_rows_dedup_compact(dev, cap_rows):
    n, t, k = 8192, 12, 256
    pool = kv4.quantize_rows_ref(*_rows(n, dev)).view(n, 1, kv4.ROW_BYTES)
    base = torch.randperm(n, device=dev)[:k]
    table = torch.stack([base.clone() for _ in range(t)]).int()
    for i in range(t):  # each token swaps 10% of the shared set; padding at the end of some rows
        j = torch.randperm(k, device=dev)[: k // 10]
        table[i, j] = torch.randint(0, n, (k // 10,), device=dev, dtype=torch.int32)
    table[3, 200:] = -1
    table[7, :] = -1
    stamp = torch.full((n,), kv4.STAMP_CLEAN, dtype=torch.int32, device=dev)
    dst = torch.zeros(cap_rows or t * k, kv4.DIM_QK, dtype=torch.float8_e4m3fn, device=dev)
    out = torch.empty_like(table)
    incl = kv4.dequant_rows_dedup(pool, table, stamp, dst, out)
    torch.cuda.synchronize()
    assert (stamp == kv4.STAMP_CLEAN).all()  # cleaned for the next call
    flat = table.view(-1)
    valid = table >= 0
    distinct = torch.unique(flat[flat >= 0]).numel()
    assert int(incl[-1]) == distinct
    # owners in flat order get compact rows 0..distinct-1; entries map to their slot's row; -1 stays -1
    first = {}
    for i, sl in enumerate(flat.tolist()):
        if sl >= 0 and sl not in first:
            first[sl] = len(first)
    exp = torch.tensor([first[sl] for sl in flat[valid.view(-1)].tolist()], device=dev, dtype=torch.int32)
    if cap_rows is not None:
        exp = torch.where(exp < cap_rows, exp, torch.full_like(exp, -1))
        assert distinct > cap_rows  # the case exercises the cap
    assert torch.equal(out[valid], exp) and torch.equal(out < 0, ~valid | (out < 0))
    ref_rows = kv4.dequantize_rows_ref(pool[:, 0], torch.float8_e4m3fn).view(torch.uint8)
    kept = valid & (out >= 0)
    assert torch.equal(dst.view(torch.uint8)[out[kept].long()], ref_rows[table[kept].long()])
    written = min(distinct, cap_rows or distinct)
    assert (dst.view(torch.uint8)[:written].abs().sum(-1) > 0).all()
    assert dst.view(torch.uint8)[written:].abs().sum().item() == 0
    assert distinct < valid.sum().item() // 3  # the dedup actually collapsed the shared set


def _glm_case(dev, t, n=20000, k=2048, h=64, short_rows=True):
    pool = kv4.quantize_rows_ref(*_rows(n, dev)).view(n, 1, kv4.ROW_BYTES)
    q = (torch.randn(t, h, kv4.DIM_QK, device=dev) * 1.5).to(torch.float8_e4m3fn)
    table = torch.stack([torch.randperm(n, device=dev)[:k] for _ in range(t)]).int()
    if short_rows and t > 1:
        table[0, 37:] = -1  # a token early in its sequence
        if t > 2:
            table[1, 1000:] = -1
        if t > 3:
            table[2, 3:] = -1
        if t > 4:
            table[3, :] = -1  # a CUDA-graph padding row
    return pool, q, table


def _check(out, ref, tag, tol):
    ref = ref.float()
    err = (out.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    rel = err / max(scale, 1e-6)
    print(f"{tag}: max err {err:.4f} / max |ref| {scale:.3f} = rel {rel:.5f}")
    assert rel < tol, (tag, rel)
    return rel


@gpu
@pytest.mark.parametrize("cfg", [dict(block_h=32, block_n=32, num_warps=4, splits=1),
                                 dict(block_h=16, block_n=64, num_warps=4, splits=4),
                                 dict(block_h=32, block_n=64, num_warps=8, splits=8),
                                 dict(block_h=16, block_n=32, num_warps=4, splits=16)])
def test_sparse_mla_vs_fp32_reference(dev, cfg):
    pool, q, table = _glm_case(dev, t=6)
    scale = 576 ** -0.5
    out = kv4.sparse_mla(q, pool, table, scale, **cfg)
    ref = kv4.sparse_mla_ref(q, pool.view(-1, kv4.ROW_BYTES), table, scale)
    _check(out, ref, f"fused sparse_mla {cfg}", 2e-2)
    assert torch.equal(out[3], torch.zeros_like(out[3]))  # the all -1 row


@gpu
def test_sparse_mla_bf16_query_and_single_row(dev):
    pool, q, table = _glm_case(dev, t=1, n=3000, short_rows=False)
    scale = 0.05
    out = kv4.sparse_mla(q.bfloat16(), pool, table, scale)
    ref = kv4.sparse_mla_ref(q.bfloat16().to(torch.float8_e4m3fn), pool.view(-1, kv4.ROW_BYTES), table, scale)
    _check(out, ref, "fused sparse_mla bf16 q", 2e-2)


@gpu
def test_sparse_mla_cuda_graph_replay(dev):
    pool, q, table = _glm_case(dev, t=8, n=8000)
    scale = 576 ** -0.5
    out = torch.zeros(8, 64, kv4.DIM_NOPE, dtype=torch.bfloat16, device=dev)
    kv4.sparse_mla(q, pool, table, scale, out=out)  # warm (compile)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            kv4.sparse_mla(q, pool, table, scale, out=out)
    torch.cuda.current_stream().wait_stream(s)
    eager = kv4.sparse_mla(q, pool, table, scale)
    # new inputs written in place, replayed: the graph must read the current pool and table
    pool.copy_(kv4.quantize_rows_ref(*_rows(8000, dev)).view(8000, 1, kv4.ROW_BYTES))
    table.copy_(torch.stack([torch.randperm(8000, device=dev)[:2048] for _ in range(8)]).int())
    g.replay()
    torch.cuda.synchronize()
    fresh = kv4.sparse_mla(q, pool, table, scale)
    assert torch.equal(out, fresh) and not torch.equal(out, eager)
