"""glmflash_kv4 GPU tests (skipped without CUDA; run on a B200 dev box, flashinfer imported first so Triton uses the
engine's ptxas): the rope-free 288-byte-row kernels against the torch reference, bit for bit where the reference is
exact (quant_write, dequant_rows, dequant_seq, dequant_rows_dedup) and within tolerance for the fused sparse MLA
(Flash shape: 64 heads, 512-wide q, top-2048), plus CUDA-graph replay of the fused kernel.

    python3 operations/devbox/devbox.py test <box> operations/kernels/test_glmflash_kv4_gpu.py
NOT YET RUN (2026-09-29: no B200 schedulable). Status: agents/status/glmflash.md.
"""
from __future__ import annotations

import pytest
import torch

import glmflash_kv4 as fk

torch.manual_seed(0)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def dev():
    pytest.importorskip("flashinfer")
    return torch.device("cuda")


def _latent(t, device, scale=1.5):
    return (torch.randn(t, fk.DIM_NOPE, device=device) * scale).bfloat16()


def _pool(n, dev):
    return fk.quantize_rows_ref(_latent(n, dev)).view(n, 1, fk.ROW_BYTES)


@gpu
@pytest.mark.parametrize("nope_fp8", [False, True])
def test_quant_write_bit_exact(dev, nope_fp8):
    t, n = 1000, 4096
    nope = _latent(t, dev)
    if nope_fp8:
        nope = nope.to(torch.float8_e4m3fn)
    loc = torch.randperm(n, device=dev)[:t]
    pool = torch.zeros(n, 1, fk.ROW_BYTES, dtype=torch.uint8, device=dev)
    fk.quant_write(pool, loc, nope.unsqueeze(1))
    assert torch.equal(pool[loc, 0], fk.quantize_rows_ref(nope))
    untouched = torch.ones(n, dtype=torch.bool, device=dev)
    untouched[loc] = False
    assert pool[untouched].abs().sum().item() == 0


@gpu
def test_quant_write_out_of_pool_loc_skipped(dev):
    t, n = 9, 64
    nope = _latent(t, dev)
    loc = torch.arange(t, device=dev, dtype=torch.int32) * 8
    loc[-1] = n + 5
    pool = torch.zeros(n, 1, fk.ROW_BYTES, dtype=torch.uint8, device=dev)
    fk.quant_write(pool.view(torch.float8_e4m3fn), loc, nope)
    assert torch.equal(pool[loc[:-1].long(), 0], fk.quantize_rows_ref(nope[:-1]))


@gpu
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("tile", [(16, 4), (8, 4), (32, 8)])
def test_dequant_rows_bit_exact(dev, out_dtype, tile):
    n, r = 4096, 3000
    pool = _pool(n, dev)
    idx = torch.randint(0, n, (r,), dtype=torch.int32, device=dev)
    idx[::7] = -1
    dst = torch.zeros(r, fk.DIM_QK, dtype=out_dtype, device=dev)
    fk.dequant_rows(pool, idx, dst, block_r=tile[0], num_warps=tile[1])
    valid = idx >= 0
    ref = fk.dequantize_rows_ref(pool[idx[valid].long(), 0], out_dtype)
    assert torch.equal(dst[valid].view(torch.uint8), ref.view(torch.uint8))
    assert dst[~valid].view(torch.uint8).abs().sum().item() == 0


@gpu
def test_dequant_seq_and_remap(dev):
    n, bs = 8192, 3
    pool = _pool(n, dev)
    seq_lens = torch.tensor([700, 5, 1300], dtype=torch.int32, device=dev)
    max_len = 1300
    table = torch.randperm(n, device=dev).int()[: bs * max_len].view(bs, max_len).clone()
    bases = torch.tensor([0, 700, 705], dtype=torch.int32, device=dev)
    rows = int(seq_lens.sum())
    dst = torch.zeros(rows + 8, fk.DIM_QK, dtype=torch.float8_e4m3fn, device=dev)
    slot_to_pos = torch.full((n,), -1, dtype=torch.int32, device=dev)
    fk.dequant_seq(pool, table, seq_lens, bases, max_len, dst, slot_to_pos)
    ref_slots = torch.cat([table[b, : int(seq_lens[b])] for b in range(bs)]).long()
    ref = fk.dequantize_rows_ref(pool[ref_slots, 0], torch.float8_e4m3fn)
    assert torch.equal(dst[:rows].view(torch.uint8), ref.view(torch.uint8))
    assert dst[rows:].view(torch.uint8).abs().sum().item() == 0
    assert torch.equal(slot_to_pos[ref_slots], torch.arange(rows, device=dev, dtype=torch.int32))
    t, k = 5, 64
    topk = torch.stack([ref_slots[torch.randperm(rows, device=dev)[:k]] for _ in range(t)]).int()
    topk[0, 10:] = -1
    out = torch.empty_like(topk)
    fk.remap_table(topk, slot_to_pos, n, out)
    exp = torch.where(topk >= 0, slot_to_pos[topk.clamp(min=0).long()], torch.full_like(topk, -1))
    assert torch.equal(out, exp)


@gpu
@pytest.mark.parametrize("cap_rows", [None, 300])
def test_dequant_rows_dedup_compact(dev, cap_rows):
    n, t, k = 8192, 12, 256
    pool = _pool(n, dev)
    base = torch.randperm(n, device=dev)[:k]
    table = torch.stack([base.clone() for _ in range(t)]).int()
    for i in range(t):
        j = torch.randperm(k, device=dev)[: k // 10]
        table[i, j] = torch.randint(0, n, (k // 10,), device=dev, dtype=torch.int32)
    table[3, 200:] = -1
    table[7, :] = -1
    stamp = torch.full((n,), fk.STAMP_CLEAN, dtype=torch.int32, device=dev)
    dst = torch.zeros(cap_rows or t * k, fk.DIM_QK, dtype=torch.float8_e4m3fn, device=dev)
    out = torch.empty_like(table)
    incl = fk.dequant_rows_dedup(pool, table, stamp, dst, out)
    torch.cuda.synchronize()
    assert (stamp == fk.STAMP_CLEAN).all()
    flat = table.view(-1)
    valid = table >= 0
    distinct = torch.unique(flat[flat >= 0]).numel()
    assert int(incl[-1]) == distinct
    first = {}
    for sl in flat.tolist():
        if sl >= 0 and sl not in first:
            first[sl] = len(first)
    exp = torch.tensor([first[sl] for sl in flat[valid.view(-1)].tolist()], device=dev, dtype=torch.int32)
    if cap_rows is not None:
        exp = torch.where(exp < cap_rows, exp, torch.full_like(exp, -1))
        assert distinct > cap_rows
    assert torch.equal(out[valid], exp)
    ref_rows = fk.dequantize_rows_ref(pool[:, 0], torch.float8_e4m3fn).view(torch.uint8)
    kept = valid & (out >= 0)
    assert torch.equal(dst.view(torch.uint8)[out[kept].long()], ref_rows[table[kept].long()])


def _flash_case(dev, t, n=20000, k=2048, h=64):
    pool = _pool(n, dev)
    q = (torch.randn(t, h, fk.DIM_QK, device=dev) * 1.5).to(torch.float8_e4m3fn)
    table = torch.stack([torch.randperm(n, device=dev)[:k] for _ in range(t)]).int()
    if t > 4:
        table[0, 37:] = -1
        table[1, 1000:] = -1
        table[2, 3:] = -1
        table[3, :] = -1
    return pool, q, table


def _check(out, ref, tag, tol):
    ref = ref.float()
    err = (out.float() - ref).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-6)
    print(f"{tag}: max err {err:.4f} rel {rel:.5f}")
    assert rel < tol, (tag, rel)


@gpu
@pytest.mark.parametrize("cfg", [dict(block_h=32, block_n=32, num_warps=4, splits=1),
                                 dict(block_h=16, block_n=64, num_warps=4, splits=4),
                                 dict(block_h=32, block_n=64, num_warps=8, splits=8)])
def test_sparse_mla_vs_fp32_reference(dev, cfg):
    pool, q, table = _flash_case(dev, t=6)
    scale = 512 ** -0.5
    out = fk.sparse_mla(q, pool, table, scale, **cfg)
    ref = fk.sparse_mla_ref(q, pool.view(-1, fk.ROW_BYTES), table, scale)
    _check(out, ref, f"fused sparse_mla {cfg}", 2e-2)
    assert torch.equal(out[3], torch.zeros_like(out[3]))


@gpu
def test_sparse_mla_cuda_graph_replay(dev):
    pool, q, table = _flash_case(dev, t=8, n=8000)
    scale = 512 ** -0.5
    out = torch.zeros(8, 64, fk.DIM_NOPE, dtype=torch.bfloat16, device=dev)
    fk.sparse_mla(q, pool, table, scale, out=out)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            fk.sparse_mla(q, pool, table, scale, out=out)
    torch.cuda.current_stream().wait_stream(s)
    pool.copy_(_pool(8000, dev))
    table.copy_(torch.stack([torch.randperm(8000, device=dev)[:2048] for _ in range(8)]).int())
    g.replay()
    torch.cuda.synchronize()
    eager = fk.sparse_mla(q, pool, table, scale)
    assert torch.equal(out, eager)
