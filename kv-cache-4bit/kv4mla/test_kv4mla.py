"""Tests for the kv4mla native FP4 sparse MLA decode kernel (GPU, SM100).

Reference: fp32 attention over the exactly dequantized 352-byte rows (glm_kv4.dequantize_rows_ref semantics) with the
FP8 q, i.e. the math the kernel implements; the kernel's only roundings are fp16 softmax probabilities and the bf16
output. Shapes are GLM-5.3's (64 heads, 576/512, top-2048). Covers decode batches 1..128, verify rows (tokens of a
request sharing most of their picks), -1 padding, slots beyond the pool, rows with no valid slot, split-KV on/off,
top-k not a multiple of 64, CUDA-graph capture/replay with new table contents, and bit-for-bit determinism.

    python3 -m pytest -q operations/kernels/kv4mla/test_kv4mla.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glm_kv4 as kv4  # noqa: E402

GPU = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
pytestmark = pytest.mark.skipif(not GPU, reason="needs an SM100 GPU")
H, K = 64, 2048
SCALE = 576 ** -0.5


def make_pool(n: int, seed: int = 0, device="cuda") -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    nope = (torch.randn(n, 512, device=device, generator=g) * 1.5).bfloat16()
    # a few outlier channels, as real latents have
    nope[:, ::97] *= 6
    rope = (torch.randn(n, 64, device=device, generator=g) * 1.5).to(torch.float8_e4m3fn)
    pool = torch.empty(n, 1, kv4.ROW_BYTES, dtype=torch.uint8, device=device)
    kv4.quant_write(pool, torch.arange(n, device=device), nope, rope)
    return pool.view(n, kv4.ROW_BYTES)


def ref_attention(q8: torch.Tensor, rows: torch.Tensor, table: torch.Tensor, scale: float, chunk: int = 32,
                  e4m3_rows: bool = False):
    """fp32 [T, H, 512] over exactly dequantized rows (or, e4m3_rows, rows rounded to E4M3 as the fp8 mode and kv4glm's
    scratch path store them); invalid slots (-1 or >= rows) masked; empty rows -> 0."""
    t = q8.shape[0]
    n = rows.shape[0]
    out = torch.zeros(t, H, 512, dtype=torch.float32, device=q8.device)
    for c0 in range(0, t, chunk):
        c1 = min(t, c0 + chunk)
        tab = table[c0:c1].long()
        valid = (tab >= 0) & (tab < n)
        dt = torch.float8_e4m3fn if e4m3_rows else torch.float32
        kv = kv4.dequantize_rows_ref(rows[tab.clamp(0, n - 1).reshape(-1)], dt).float().view(c1 - c0, -1, 576)
        s = torch.einsum("thd,tkd->thk", q8[c0:c1].float(), kv) * scale
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
        p = torch.softmax(s, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)
        out[c0:c1] = torch.einsum("thk,tkd->thd", p, kv[..., :512])
    return out


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


@pytest.fixture(scope="module", params=["fp8", "exact"])
def dec(request):
    import kv4mla
    return kv4mla.Decoder("cuda", mode=request.param)


@pytest.fixture(scope="module")
def pool():
    return make_pool(300_000)


def case(t: int, n: int, k: int = K, seed: int = 1, share: int = 1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q8 = (torch.randn(t, H, 576, device="cuda", generator=g) * 1.5).to(torch.float8_e4m3fn)
    table = torch.randint(0, n, (t, k), device="cuda", generator=g, dtype=torch.int32)
    if share > 1:  # verify: a request's draft tokens share 90% of their picks
        for r0 in range(0, t, share):
            for i in range(r0 + 1, min(t, r0 + share)):
                j = torch.randperm(k, device="cuda", generator=g)[: k // 10]
                table[i] = table[r0]
                table[i, j] = torch.randint(0, n, (k // 10,), device="cuda", generator=g, dtype=torch.int32)
    return q8, table


# exact mode: fp16 P + bf16 output rounding over exactly dequantized rows (kv4glm's fused exact kernel: 0.0020).
# fp8 mode: E4M3 P and E4M3 tiles; measured against fp32 over the E4M3-rounded rows (what it and the scratch path compute
# on), plus a loose bound against the exact FP4 rows (the E4M3 requantization itself: kv4glm measured 0.056 there).
TOL = {"exact": 0.006, "fp8": 0.03}
TOL_EXACT_ROWS_FP8 = 0.12


def check(dec, out, q8, pool, table):
    assert torch.isfinite(out.float()).all()
    if dec.mode == "exact":
        err = rel_err(out, ref_attention(q8, pool, table, SCALE))
    else:
        err = rel_err(out, ref_attention(q8, pool, table, SCALE, e4m3_rows=True))
        assert rel_err(out, ref_attention(q8, pool, table, SCALE)) < TOL_EXACT_ROWS_FP8
    assert err < TOL[dec.mode], (dec.mode, err)
    return err


@pytest.mark.parametrize("t", [1, 2, 8, 32, 48, 64, 96, 128, 160, 256])
def test_decode_vs_fp32(dec, pool, t):
    q8, table = case(t, pool.shape[0], seed=t)
    out = dec(q8, pool, table, SCALE)
    print(dec.mode, t, check(dec, out, q8, pool, table))


@pytest.mark.parametrize("bs,tokens", [(16, 4), (48, 4), (32, 6), (64, 6)])
def test_verify_rows_vs_fp32(dec, pool, bs, tokens):
    q8, table = case(bs * tokens, pool.shape[0], seed=bs * tokens, share=tokens)
    out = dec(q8, pool, table, SCALE)
    check(dec, out, q8, pool, table)


@pytest.mark.parametrize("split", [True, False])
def test_split_and_nosplit_agree(dec, pool, split):
    q8, table = case(40, pool.shape[0], seed=7)
    out = dec(q8, pool, table, SCALE, split=split)
    check(dec, out, q8, pool, table)


def test_padding_oob_and_empty_rows(dec, pool):
    n = pool.shape[0]
    q8, table = case(12, n, seed=11)
    table[0, 100:] = -1                   # short context: only 100 valid picks
    table[1, :] = -1                      # nothing valid -> zeros
    table[2, ::3] = n + 5                 # slots beyond the pool are masked, never read
    table[3, 1:] = -1                     # a single valid pick
    table[4, :64] = -1                    # whole first block invalid
    out = dec(q8, pool, table, SCALE)
    assert out[1].float().abs().max().item() == 0.0
    check(dec, out, q8, pool, table)


@pytest.mark.parametrize("k", [100, 640, 704, 1000])
def test_topk_not_2048(dec, pool, k):
    q8, table = case(24, pool.shape[0], k=k, seed=k)
    out = dec(q8, pool, table, SCALE)
    check(dec, out, q8, pool, table)


def test_duplicate_slots(dec, pool):
    """The same slot picked many times counts that many times (as the reference and TRT-LLM do)."""
    q8, table = case(8, pool.shape[0], seed=3)
    table[:, : K // 2] = table[:, :1]
    out = dec(q8, pool, table, SCALE)
    check(dec, out, q8, pool, table)


def test_deterministic(dec, pool):
    q8, table = case(64, pool.shape[0], seed=5)
    a = dec(q8, pool, table, SCALE).clone()
    b = dec(q8, pool, table, SCALE).clone()
    assert torch.equal(a, b)


def test_strided_q_and_out_views(dec, pool):
    """q as a view with a padded token stride (the engine slices q_all); out written into a slice."""
    q8, table = case(16, pool.shape[0], seed=9)
    big = torch.zeros(16, H, 640, dtype=torch.float8_e4m3fn, device="cuda")
    big[..., :576] = q8
    qv = big[..., :576]
    out = dec(qv, pool, table, SCALE)
    check(dec, out, q8, pool, table)


@pytest.mark.parametrize("t", [48, 128, 192])
def test_cuda_graph_replay_reads_live_table(dec, pool, t):
    n = pool.shape[0]
    q8, table = case(t, n, seed=21)
    out = torch.empty(t, H, 512, dtype=torch.bfloat16, device="cuda")
    dec.prepare(t, K)
    dec(q8, pool, table, SCALE, out=out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            dec(q8, pool, table, SCALE, out=out)
    torch.cuda.current_stream().wait_stream(s)
    for seed in (22, 23):
        q_new, t_new = case(t, n, seed=seed)
        q8.copy_(q_new)
        table.copy_(t_new)
        g.replay()
        torch.cuda.synchronize()
        eager = dec(q8, pool, table, SCALE)
        assert torch.equal(out, eager)
        check(dec, out, q8, pool, table)


def test_matches_fused_triton_and_scratch_semantics(dec, pool):
    """Same inputs as kv4glm's paths: the result agrees with its exact-dequant fused Triton kernel."""
    q8, table = case(8, pool.shape[0], seed=31)
    out = dec(q8, pool, table, SCALE)
    tri = kv4.sparse_mla(q8, pool.view(-1, 1, kv4.ROW_BYTES), table, SCALE)
    assert rel_err(out, tri.float()) < (2 * TOL["exact"] if dec.mode == "exact" else TOL_EXACT_ROWS_FP8)
