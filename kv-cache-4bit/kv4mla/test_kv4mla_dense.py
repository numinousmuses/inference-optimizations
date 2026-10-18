"""GPU tests for the k27 dense mode of the kv4mla kernel (DenseDecoder): dense MLA over a paged FP4 pool with
tokens x heads folded into the 64-row q tile and per-row causal limits.

    python3 -m pytest operations/kernels/kv4mla/test_kv4mla_dense.py -q      (SM100 box; import flashinfer first)

Checks: (1) fp8-mode kernel vs the fp32 reference over exactly dequantized rows, single token per sequence, ragged
lengths and page sizes 32 / 64 (the two the MLA backends use), with and without split-KV; (2) the verify fold: 4
tokens x 8 heads per sequence with row_limit = ctx + t + 1 equals 4 separate single-token calls whose tables end at
ctx + t + 1 (the kernel's own sparse mode as the second witness); (3) padded rows (>= n_tok x heads) are ignored;
(4) CUDA-graph capture and replay with changed seq_lens; (5) the sparse entry point is unchanged (sparse_decode over
an identity table equals dense_decode on the same rows).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import flashinfer  # noqa: F401  (sets TRITON_PTXAS_BLACKWELL_PATH as the engine does)
except ImportError:
    pass
import kv4mla  # noqa: E402
from test_kv4mla import make_pool  # noqa: E402

GPU = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
pytestmark = pytest.mark.skipif(not GPU, reason="needs an SM100 GPU")
H, DQ, DV = kv4mla.H_Q, kv4mla.D_QK, kv4mla.D_V
SCALE = DQ ** -0.5
# the fp8-MMA kernel's error against the exact reference (measured by the kv4mla agent: 0.052 vs exact FP4 rows,
# TRT-LLM FP8 0.009 vs its own rows); the dense mode shares the math
TOL = 0.08


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-6))


def make_paged(seq_lens: list[int], page_size: int, n_pool: int, seed: int = 0, device="cuda"):
    """A pool of n_pool rows and a page table that maps every sequence to distinct random pages."""
    rows = make_pool(n_pool, seed=seed, device=device)
    n_pages = n_pool // page_size
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    perm = torch.randperm(n_pages, generator=g)
    max_pages = max((n + page_size - 1) // page_size for n in seq_lens)
    table = torch.full((len(seq_lens), max_pages), -1, dtype=torch.int32)
    used = 0
    for i, n in enumerate(seq_lens):
        k = (n + page_size - 1) // page_size
        table[i, :k] = perm[used: used + k].to(torch.int32)
        used += k
    assert used <= n_pages
    return rows, table.to(device)


def rand_q(b: int, seed: int = 3, device="cuda") -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(b, H, DQ, device=device, generator=g) * 1.2).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("split", [False, True])
def test_dense_single_token_matches_reference(page_size, split):
    seq_lens = [1, 63, 64, 65, 700, 2049, 4096 + 17]
    rows, table = make_paged(seq_lens, page_size, n_pool=8192 + 4096)
    b = len(seq_lens)
    q = rand_q(b)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    dec = kv4mla.DenseDecoder("cuda")
    out = dec(q, rows, table, sl, page_size, max(seq_lens), None, SCALE, split=split)
    ref = kv4mla.dense_reference(q, rows, table, sl, page_size, None, SCALE)
    for i in range(b):
        assert _rel(out[i], ref[i]) < TOL, (i, seq_lens[i], _rel(out[i], ref[i]))


def test_dense_equals_sparse_identity_table():
    """Same rows through the sparse entry point (explicit slot table) and the dense one (page table)."""
    page_size = 64
    seq_lens = [640, 1500]
    rows, table = make_paged(seq_lens, page_size, n_pool=4096)
    b = len(seq_lens)
    q = rand_q(b, seed=5)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    dense = kv4mla.DenseDecoder("cuda")(q, rows, table, sl, page_size, max(seq_lens), None, SCALE, split=False)
    k = (max(seq_lens) + 63) // 64 * 64
    slots = torch.full((b, k), -1, dtype=torch.int32, device="cuda")
    for i, n in enumerate(seq_lens):
        pos = torch.arange(n, device="cuda")
        slots[i, :n] = (table[i, pos // page_size].long() * page_size + pos % page_size).to(torch.int32)
    sparse = kv4mla.Decoder("cuda", mode="fp8")(q, rows, slots, SCALE, split=False)
    assert _rel(dense, sparse) < 1e-2, _rel(dense, sparse)


@pytest.mark.parametrize("split", [False, True])
def test_verify_fold_matches_per_token_calls(split):
    """4 draft tokens x 8 heads folded in one tile with row_limit == 4 single-token calls at growing lengths."""
    page_size, heads, n_tok = 64, 8, 4
    ctx = [300, 1000, 2500]
    b = len(ctx)
    rows, table = make_paged([c + n_tok for c in ctx], page_size, n_pool=8192)
    g = torch.Generator(device="cuda").manual_seed(9)
    q_tok = (torch.randn(b, n_tok, heads, DQ, device="cuda", generator=g) * 1.2).to(torch.float8_e4m3fn)
    q = torch.zeros(b, H, DQ, dtype=torch.float8_e4m3fn, device="cuda")
    q[:, : n_tok * heads] = q_tok.reshape(b, n_tok * heads, DQ)
    sl = torch.tensor([c + n_tok for c in ctx], dtype=torch.int32, device="cuda")
    limit = torch.zeros(b, H, dtype=torch.int32, device="cuda")
    for t in range(n_tok):
        limit[:, t * heads:(t + 1) * heads] = (sl + t + 1 - n_tok)[:, None]
    dec = kv4mla.DenseDecoder("cuda")
    out = dec(q, rows, table, sl, page_size, int(sl.max()), limit, SCALE, split=split)
    # witness 1: fp32 reference with the same limits
    ref = kv4mla.dense_reference(q, rows, table, sl, page_size, limit, SCALE)
    for i in range(b):
        assert _rel(out[i, : n_tok * heads], ref[i, : n_tok * heads]) < TOL
    # witness 2: one kernel call per token over ctx + t + 1 positions (no row_limit), heads padded to 64
    for t in range(n_tok):
        q1 = torch.zeros(b, H, DQ, dtype=torch.float8_e4m3fn, device="cuda")
        q1[:, :heads] = q_tok[:, t]
        sl_t = (sl - n_tok + t + 1).contiguous()
        o1 = dec(q1, rows, table, sl_t, page_size, int(sl_t.max()), None, SCALE, split=split)
        assert _rel(out[:, t * heads:(t + 1) * heads], o1[:, :heads]) < 2e-2, (t, _rel(out[:, t * heads:(t + 1) * heads], o1[:, :heads]))


def test_row_limit_zero_rows_and_padding_are_finite():
    page_size = 64
    rows, table = make_paged([500], page_size, n_pool=2048)
    q = rand_q(1)
    sl = torch.tensor([500], dtype=torch.int32, device="cuda")
    limit = torch.full((1, H), 500, dtype=torch.int32, device="cuda")
    limit[0, 40:] = 0   # rows with nothing to attend
    out = kv4mla.DenseDecoder("cuda")(q, rows, table, sl, page_size, 500, limit, SCALE, split=False)
    assert torch.isfinite(out.float()).all()
    assert float(out[0, 40:].float().abs().max()) == 0.0
    ref = kv4mla.dense_reference(q, rows, table, sl, page_size, limit, SCALE)
    assert _rel(out[0, :40], ref[0, :40]) < TOL


def test_graph_replay_with_new_lengths():
    page_size = 64
    seq_lens = [900, 1700]
    rows, table = make_paged(seq_lens, page_size, n_pool=8192)
    b = len(seq_lens)
    q = rand_q(b, seed=11)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    dec = kv4mla.DenseDecoder("cuda")
    dec.prepare(b)
    out = torch.empty(b, H, DV, dtype=torch.bfloat16, device="cuda")
    max_len = 4096   # the graph's block-loop bound: any length up to it replays
    dec(q, rows, table, sl, page_size, max_len, None, SCALE, out=out, split=True)   # warm-up (JIT, buffers)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        dec(q, rows, table, sl, page_size, max_len, None, SCALE, out=out, split=True)
    g.replay()
    torch.cuda.synchronize()
    ref = kv4mla.dense_reference(q, rows, table, sl, page_size, None, SCALE)
    assert all(_rel(out[i], ref[i]) < TOL for i in range(b))
    # shorter sequences on replay: the kernel must follow seq_lens, not the captured values
    sl.copy_(torch.tensor([333, 64], dtype=torch.int32, device="cuda"))
    g.replay()
    torch.cuda.synchronize()
    ref = kv4mla.dense_reference(q, rows, table, sl, page_size, None, SCALE)
    assert all(_rel(out[i], ref[i]) < TOL for i in range(b))


def test_sparse_mode_unchanged():
    """The sparse entry point still matches its reference (the kernel edit is inert with page_block_size 1)."""
    from test_kv4mla import ref_attention
    rows = make_pool(6000, seed=2)
    t, k = 4, 2048
    g = torch.Generator(device="cuda").manual_seed(4)
    table = torch.randint(0, 6000, (t, k), generator=g, device="cuda", dtype=torch.int32)
    q = rand_q(t, seed=7)
    out = kv4mla.Decoder("cuda", mode="fp8")(q, rows, table, SCALE, split=False)
    ref = ref_attention(q, rows, table, SCALE)
    assert _rel(out, ref) < TOL
