"""Tests for operations/kernels/kv4pro_fp4.py (4-bit KV cells for DeepSeek V4 Pro).

CPU: format, rounding, round-trip error, reference attention vs a naive loop, pool pricing.
GPU (skipped without CUDA): Triton store bit-exact vs the reference, dequant gather, sparse decode vs the
fp32 reference on Pro shapes (SWA 128/192 + c4 1024 / c128 widths, holes, lengths, sink, lonely rows, both
rope formats, FP8 SWA pool), CUDA-graph replay equality, 4-bit vs FP8 accuracy on the same data.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import kv4pro_fp4 as K  # noqa: E402

CUDA = torch.cuda.is_available()
if CUDA:
    import flashinfer  # noqa: F401  (engine ptxas; see agents/CONTEXT.md lessons)

gpu = pytest.mark.skipif(not CUDA, reason="needs CUDA")


# ------------------------------------------------------------------------------------------------ CPU
def test_cell_bytes():
    assert K.cell_bytes("fp8") == 320 and K.cell_bytes("bf16") == 384
    with pytest.raises(ValueError):
        K.cell_bytes("fp4")


def test_mode_env(monkeypatch):
    monkeypatch.delenv("NUMINOUS_KV4PRO", raising=False)
    assert K.mode() == (False, False, "fp8")
    monkeypatch.setenv("NUMINOUS_KV4PRO", "c4")
    assert K.mode() == (False, True, "fp8")
    monkeypatch.setenv("NUMINOUS_KV4PRO", "all")
    monkeypatch.setenv("NUMINOUS_KV4PRO_ROPE", "bf16")
    assert K.mode() == (True, True, "bf16")
    assert K.priced_cell_bytes(584) == (384, 384, 584)  # c128 stays FP8
    monkeypatch.setenv("NUMINOUS_KV4PRO", "c4")
    assert K.priced_cell_bytes(584) == (584, 384, 584)
    monkeypatch.setenv("NUMINOUS_KV4PRO", "bogus")
    with pytest.raises(ValueError):
        K.mode()


def test_e2m1_rounding_ties_to_even():
    a = torch.tensor([0.0, 0.2, 0.25, 0.26, 0.75, 1.25, 1.3, 1.75, 2.5, 2.6, 3.5, 5.0, 5.1, 7.0])
    want = [0, 0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7]
    assert K._e2m1_code(a).tolist() == want


def test_exact_values_round_trip():
    # rows whose 16-blocks are (E2M1 grid) x (E4M3-representable scale) survive exactly
    g = torch.tensor(K._E2M1)
    torch.manual_seed(0)
    n = 8
    code = torch.randint(0, 8, (n, 448))
    sign = torch.where(torch.rand(n, 448) < 0.5, -1.0, 1.0)
    code[:, ::16] = 7  # each block holds the max so amax/6 is the scale exactly
    scale = torch.tensor([0.5, 1.0, 2.0, 0.125, 4.0, 0.75, 1.5, 3.0]).repeat_interleave(448).reshape(n, 448)
    nope = g[code] * sign * scale
    rope = torch.randn(n, 64).to(torch.bfloat16).float()
    x = torch.cat([nope, rope], 1)
    for rope_fmt in ("fp8", "bf16"):
        cells = K.quantize_ref(x, rope_fmt)
        y = K.dequantize_ref(cells, rope_fmt)
        assert torch.equal(y[:, :448], nope)
        if rope_fmt == "bf16":
            assert torch.equal(y[:, 448:], rope)
        else:
            assert torch.equal(y[:, 448:], rope.to(torch.float8_e4m3fn).float())


def test_layout_and_error():
    torch.manual_seed(1)
    x = torch.randn(256, 512) * 2
    x[3] = 0
    cells = K.quantize_ref(x)
    assert cells.shape == (256, 320) and cells.dtype == torch.uint8
    assert (cells[:, 252:256] == 0).all()  # pad the decode kernel relies on
    assert (cells[3] == 0).all()  # zero row -> all-zero cell (no -0 nibbles)
    y = K.dequantize_ref(cells)
    nope_rel = ((y[:, :448] - x[:, :448]).norm() / x[:, :448].norm()).item()
    rope_rel = ((y[:, 448:] - x[:, 448:]).norm() / x[:, 448:].norm()).item()
    # measured-by-test bounds for Gaussian rows: NVFP4 with the scale search ~0.088, E4M3 rope ~0.02-0.03
    assert 0.05 < nope_rel < 0.092, nope_rel
    assert rope_rel < 0.04, rope_rel


def test_fp8_paged_ref_round_trip():
    torch.manual_seed(2)
    x = torch.randn(300, 512)
    buf = K.quantize_fp8_paged_ref(x, 128)
    assert buf.shape == (3, K.fp8_cell_page_bytes(128))
    y = K.dequantize_fp8_paged_ref(buf, 128, 300)
    assert ((y - x).norm() / x.norm()).item() < 0.04


def _naive_decode(q, rows_list, idx_list, len_list, sink, scale):
    T, H, _ = q.shape
    out = torch.zeros(T, H, 512)
    for t in range(T):
        keys = []
        for rows, idx, ln in zip(rows_list, idx_list, len_list):
            W = idx.shape[1]
            L = W if ln is None else int(ln[t])
            for j in range(min(L, W)):
                if idx[t, j] >= 0:
                    keys.append(rows[idx[t, j]])
        for h in range(H):
            if not keys:
                continue
            kv = torch.stack(keys)
            s = kv @ q[t, h] * scale
            z = torch.cat([s, sink[h:h + 1]]) if sink is not None else s
            p = torch.exp(s - torch.logsumexp(z, 0))
            out[t, h] = p @ kv
    return out


def test_reference_decode_matches_naive_loop():
    torch.manual_seed(3)
    T, H = 5, 4
    swa_rows, ext_rows = torch.randn(40, 512), torch.randn(60, 512)
    q = torch.randn(T, H, 512) * 0.2
    swa_idx = torch.randint(0, 40, (T, 8))
    swa_idx[1, 2] = -1
    ext_idx = torch.randint(0, 60, (T, 16))
    ext_idx[2] = -1
    swa_len = torch.tensor([8, 5, 0, 3, 8])
    ext_len = torch.tensor([16, 16, 16, 0, 7])
    sink = torch.randn(H)
    swa_len2 = swa_len.clone()
    ext_len2 = ext_len.clone()
    out, lse = K.sparse_decode_ref(q, None, swa_idx, swa_len2, None, ext_idx, ext_len2, attn_sink=sink,
                                   softmax_scale=0.1, kv_rows=(swa_rows, ext_rows))
    ref = _naive_decode(q, [swa_rows, ext_rows], [swa_idx, ext_idx], [swa_len, ext_len], sink, 0.1)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    # row 2 has no SWA keys (len 0) and no valid ext keys: lonely -> zeros, lse -inf
    assert (out[2] == 0).all() and torch.isneginf(lse[2]).all()


def test_pricing_matches_configurator_and_measured_pools():
    fp8 = K.bytes_per_full_token(swa_cell=584, comp_cell=584)
    uni = K.bytes_per_full_token(swa_cell=512, comp_cell=512)
    # measured (agents/status/proattn.md, same node): 1,404,160 (584) vs 1,521,664 (512) tokens/rank = +8.37%
    assert abs(fp8 / uni - 1521664 / 1404160) < 0.003
    c4 = K.bytes_per_full_token(swa_cell=584, comp_cell=320, c128_cell=584)
    allc = K.bytes_per_full_token(swa_cell=320, comp_cell=320, c128_cell=584)
    assert 1.17 < fp8 / c4 < 1.19
    assert 1.37 < fp8 / allc < 1.40
    assert abs(fp8 - 12914 * 62 / 61) < 2
    # bf16 compress state (SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16) halves the 3,840 B/token of c4 + indexer state
    st16 = K.bytes_per_full_token(swa_cell=584, comp_cell=584, state_bytes=2)
    both = K.bytes_per_full_token(swa_cell=320, comp_cell=320, c128_cell=584, state_bytes=2)
    assert 1.16 < fp8 / st16 < 1.19
    assert 1.72 < fp8 / both < 1.76


# ------------------------------------------------------------------------------------------------ GPU
def _rand_rows(n, dev, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 512, generator=g) * 1.5
    x[:, 448:] *= 0.7
    return x.to(dev).to(torch.bfloat16)


@gpu
@pytest.mark.parametrize("rope", ["fp8", "bf16"])
def test_store_bit_exact(rope):
    dev = "cuda"
    x = _rand_rows(777, dev, 4)
    x[5] = 0
    x[6, :16] = 1e-6  # scale underflow block
    cell = K.cell_bytes(rope)
    cache = torch.full((1000, cell), 0xAB, dtype=torch.uint8, device=dev)
    loc = torch.randperm(1000, device=dev)[:777].to(torch.int32)
    loc[10] = -1  # skipped
    K.kv4_store(cache, loc, x, rope)
    ref = K.quantize_ref(x.float(), rope)
    keep = torch.arange(777, device=dev) != 10
    got = cache[loc[keep].long()]
    # the scale search compares fp32 block errors summed in different orders: an exact tie may break differently
    diff = (got != ref[keep]).any(1)
    assert diff.float().mean().item() < 2e-3, diff.sum().item()
    e_got = (K.dequantize_ref(got, rope) - x[keep].float()).pow(2).sum(1)
    e_ref = (K.dequantize_ref(ref[keep], rope) - x[keep].float()).pow(2).sum(1)
    assert torch.allclose(e_got, e_ref, rtol=1e-5, atol=1e-6)
    untouched = torch.ones(1000, dtype=torch.bool, device=dev)
    untouched[loc[keep].long()] = False
    assert (cache[untouched] == 0xAB).all()


@gpu
@pytest.mark.parametrize("rope", ["fp8", "bf16"])
def test_dequant_gather(rope):
    dev = "cuda"
    x = _rand_rows(500, dev, 5)
    cells = K.quantize_ref(x.float(), rope)
    ids = torch.randint(0, 500, (333,), device=dev, dtype=torch.int32)
    ids[7] = -1
    out = torch.full((400, 1, 512), 7.0, dtype=torch.bfloat16, device=dev)
    K.kv4_dequant_gather(cells, ids, rope, out=out[:333])
    ref = K.dequantize_ref(cells, rope)[ids.clamp(min=0).long()]
    ref[7] = 0
    assert torch.equal(out[:333, 0].float(), ref.to(torch.bfloat16).float())
    assert (out[333:] == 7).all()


def _decode_case(T, W_swa, W_ext, rope, swa_fp8, seed, H=128, n_swa_tok=2048, n_ext_tok=8192):
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(T, H, 512, generator=g) * 0.05).to(dev).to(torch.bfloat16)
    swa_x = _rand_rows(n_swa_tok, dev, seed + 1)
    ext_x = _rand_rows(n_ext_tok, dev, seed + 2)
    if swa_fp8:
        swa_cache = K.quantize_fp8_paged_ref(swa_x.float(), 128)
        swa_rows = K.dequantize_fp8_paged_ref(swa_cache, 128, n_swa_tok)
    else:
        swa_cache = K.quantize_ref(swa_x.float(), rope)
        swa_rows = K.dequantize_ref(swa_cache, rope)
    ext_cache = K.quantize_ref(ext_x.float(), rope)
    ext_rows = K.dequantize_ref(ext_cache, rope)
    swa_idx = torch.randint(0, n_swa_tok, (T, W_swa), generator=g).to(torch.int32)
    swa_len = torch.randint(1, W_swa + 1, (T,), generator=g).to(torch.int32)
    swa_idx[:, ::7] = -1
    ext_idx = ext_len = None
    if W_ext:
        ext_idx = torch.randint(0, n_ext_tok, (T, W_ext), generator=g).to(torch.int32)
        ext_len = torch.randint(0, W_ext + 1, (T,), generator=g).to(torch.int32)
        ext_idx[:, 5::11] = -1
        ext_idx = ext_idx.to(dev)
        ext_len = ext_len.to(dev)
    if T > 2:
        swa_len[1] = 0
        if W_ext:
            ext_len[1] = 0  # lonely row
    sink = (torch.randn(H, generator=g)).to(dev).float()
    return dict(q=q, swa_cache=swa_cache, swa_idx=swa_idx.to(dev), swa_len=swa_len.to(dev), ext_cache=ext_cache,
                ext_idx=ext_idx, ext_len=ext_len, sink=sink, swa_rows=swa_rows, ext_rows=ext_rows,
                x=(swa_x, ext_x))


def _run(c, rope, swa_fp8, **kw):
    return K.kv4_sparse_decode(c["q"], c["swa_cache"], c["swa_idx"], c["swa_len"], c["sink"], 512 ** -0.5,
                               c["ext_cache"] if c["ext_idx"] is not None else None, c["ext_idx"], c["ext_len"],
                               rope=rope, swa_fp8_page_size=128 if swa_fp8 else None, **kw)


def _ref(c, rows=None):
    rows = rows or (c["swa_rows"], c["ext_rows"])
    return K.sparse_decode_ref(c["q"].float(), None, c["swa_idx"], c["swa_len"], None, c["ext_idx"], c["ext_len"],
                               attn_sink=c["sink"], softmax_scale=512 ** -0.5, kv_rows=rows)


CASES = [  # (T, W_swa, W_ext): decode c0 / c4 / c128, DSpark verify width 192, larger batches
    (1, 128, 0), (6, 192, 1024), (6, 192, 448), (37, 128, 1024), (192, 192, 1024), (64, 128, 832),
]


@gpu
@pytest.mark.parametrize("T,W_swa,W_ext", CASES)
@pytest.mark.parametrize("rope,swa_fp8", [("fp8", False), ("bf16", False), ("fp8", True)])
def test_decode_vs_reference(T, W_swa, W_ext, rope, swa_fp8):
    c = _decode_case(T, W_swa, W_ext, rope, swa_fp8, seed=T * 7 + W_ext)
    out, lse = _run(c, rope, swa_fp8)
    ref, ref_lse = _ref(c)
    err = (out.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= 0.02 * scale + 1e-3, (err, scale)
    fin = torch.isfinite(ref_lse)
    assert torch.allclose(lse[fin], ref_lse[fin], atol=2e-2, rtol=1e-3)
    assert torch.isneginf(lse[~fin]).all()
    if T > 2 and W_ext:
        assert (out[1] == 0).all()


@gpu
def test_decode_split_invariance():
    c = _decode_case(6, 192, 1024, "fp8", False, seed=11)
    a, _ = _run(c, "fp8", False, nsplit=1)
    b, _ = _run(c, "fp8", False, nsplit=19)
    assert (a.float() - b.float()).abs().max().item() < 2e-2 * a.float().abs().max().item()


@gpu
def test_decode_cuda_graph_replay():
    c = _decode_case(16, 192, 1024, "fp8", False, seed=12)
    eager, _ = _run(c, "fp8", False)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run(c, "fp8", False)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, _ = _run(c, "fp8", False)
    # new indices / lengths written in place, then replay: the graph must read the updated metadata
    c2 = _decode_case(16, 192, 1024, "fp8", False, seed=13)
    c["swa_idx"].copy_(c2["swa_idx"])
    c["ext_idx"].copy_(c2["ext_idx"])
    c["swa_len"].copy_(c2["swa_len"])
    c["ext_len"].copy_(c2["ext_len"])
    g.replay()
    torch.cuda.synchronize()
    ref, _ = _ref(c)
    assert (out.float() - ref).abs().max().item() <= 0.02 * ref.abs().max().item() + 1e-3
    assert not torch.equal(out, eager)


@gpu
def test_graph_store_then_decode_ordering():
    """Production order in one captured graph: store the step's rows, then attend to them (no host sync)."""
    dev = "cuda"
    T = 8
    c = _decode_case(T, 128, 1024, "fp8", False, seed=14)
    new = _rand_rows(T, dev, 99)
    loc = torch.arange(T, device=dev, dtype=torch.int32) + 100
    c["swa_idx"][:, 0] = loc  # every row attends to its own new token
    c["swa_len"].clamp_(min=1)

    def step():
        K.kv4_store(c["swa_cache"], loc, new, "fp8")
        return _run(c, "fp8", False)[0]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    c["swa_cache"][100:108].zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    c["swa_cache"][100:108].zero_()
    g.replay()
    torch.cuda.synchronize()
    rows = K.dequantize_ref(c["swa_cache"], "fp8")
    ref, _ = _ref(c, rows=(rows, c["ext_rows"]))
    assert (out.float() - ref).abs().max().item() <= 0.02 * ref.abs().max().item() + 1e-3


@gpu
def test_accuracy_fp4_vs_fp8_same_data():
    """Output error vs the unquantized-KV fp32 reference: 4-bit cells vs the FP8 584-byte cell (informational bound)."""
    c = _decode_case(32, 192, 1024, "fp8", False, seed=15)
    true_rows = (c["x"][0].float(), c["x"][1].float())
    ref, _ = _ref(c, rows=true_rows)
    out4, _ = _run(c, "fp8", False)
    fp8_rows = (K.dequantize_fp8_paged_ref(K.quantize_fp8_paged_ref(true_rows[0], 128), 128, true_rows[0].shape[0]),
                K.dequantize_fp8_paged_ref(K.quantize_fp8_paged_ref(true_rows[1], 64), 64, true_rows[1].shape[0]))
    ref8, _ = _ref(c, rows=fp8_rows)
    e4 = ((out4.float() - ref).norm() / ref.norm()).item()
    e8 = ((ref8 - ref).norm() / ref.norm()).item()
    print(f"relative output error vs unquantized KV: fp4 {e4:.4f}  fp8 {e8:.4f}")
    assert e4 < 0.12 and e8 < e4
