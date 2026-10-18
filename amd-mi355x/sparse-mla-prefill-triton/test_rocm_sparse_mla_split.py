"""CPU (Triton interpreter) checks of the split-K sparse MLA decode kernels in rocm_sparse_mla_prefill.py."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

import rocm_sparse_mla_prefill as M  # noqa: E402


def _case(t=3, h=16, k=256, slots=2000, seed=0, masked=0.1):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(t, h, 576, generator=g) * 0.3).to(torch.bfloat16)
    kv = (torch.randn(slots, 1, 576, generator=g) * 0.5).to(torch.float8_e4m3fn)
    idx = torch.randint(0, slots, (t, 1, k), generator=g, dtype=torch.int32)
    idx[torch.rand(t, 1, k, generator=g) < masked] = -1
    return q, kv, idx


def _ref(q, kv, idx, sm):
    """fp32 sparse attention over the valid picks (q and KV through FP8 as the kernels see them)."""
    t, h, d = q.shape
    q8 = q.to(torch.float8_e4m3fn).float()
    k = kv.view(kv.shape[0], d).float()
    out = torch.zeros(t, h, 512)
    for i in range(t):
        sel = idx[i, 0]
        sel = sel[sel >= 0].long()
        keys = k[sel]
        s = (q8[i] @ keys.T) * sm
        p = torch.softmax(s, -1)
        out[i] = p @ keys[:, :512]
    return out


@pytest.mark.parametrize("split", [1, 2, 4])
def test_prefetch_matches_plain_kernel(split):
    q, kv, idx = _case()
    sm = 576 ** -0.5
    a = M.sparse_mla_split_fp8(q, kv, idx, sm, 512, split=split, block_n=64).float()
    b = M.sparse_mla_split_fp8(q, kv, idx, sm, 512, split=split, block_n=64, prefetch=True).float()
    assert torch.equal(a, b)


def test_prefetch_close_to_reference():
    q, kv, idx = _case(t=2, h=16, k=128)
    sm = 576 ** -0.5
    o = M.sparse_mla_split_fp8(q, kv, idx, sm, 512, split=2, block_n=64, prefetch=True)[0].float()
    r = _ref(q, kv, idx, sm)
    assert (o - r).abs().max().item() / r.abs().max().item() < 0.08     # FP8 P quantization (the TileLang scheme)


def test_prefetch_all_masked_block_and_row():
    q, kv, idx = _case(t=2, h=16, k=128, masked=0.0)
    idx[0, 0, 64:] = -1          # second block of row 0 fully masked
    idx[1, 0, :] = -1            # row 1 fully masked -> zeros, no NaN
    sm = 576 ** -0.5
    a = M.sparse_mla_split_fp8(q, kv, idx, sm, 512, split=1, block_n=64).float()
    b = M.sparse_mla_split_fp8(q, kv, idx, sm, 512, split=1, block_n=64, prefetch=True).float()
    assert torch.equal(a, b) and not torch.isnan(b).any()


def _nan_slot0(kv):
    kv.view(torch.uint8)[0].fill_(0x7F)          # e4m3fn NaN bytes in slot 0 (padded graph rows write their KV there)
    return kv


@pytest.mark.parametrize("prefetch", [False, True])
def test_masked_picks_never_read_slot0_nan(prefetch):
    """k3b300 2026-10-03 20:13 (GLM-5.3 Value idle-rank garbage): an invalid (-1) pick must not load slot 0. p = 0 there,
    but the PV dot computes 0 * K, which is NaN when slot 0 holds NaN bytes. With idle DP ranks, every graph row is
    padding and writes its KV to slot 0."""
    q, kv, idx = _case(t=3, h=16, k=128, masked=0.3)
    idx[idx == 0] = 1                            # slot 0 is never a valid pick here
    kv = _nan_slot0(kv)
    sm = 576 ** -0.5
    o = M.sparse_mla_split_fp8(q, kv, idx, sm, 512, split=2, block_n=64, prefetch=prefetch)[0].float()
    assert not torch.isnan(o).any()
    r = _ref(q, kv, idx, sm)
    assert (o - r).abs().max().item() / r.abs().max().item() < 0.08


def test_prefill_kernel_masked_picks_never_read_slot0_nan():
    q, kv, idx = _case(t=3, h=16, k=128, masked=0.3)
    idx[idx == 0] = 1
    kv = _nan_slot0(kv)
    sm = 576 ** -0.5
    o = M.sparse_mla_prefill_fp8(q, kv, idx, sm, 512, block_n=64)[0].float()
    assert not torch.isnan(o).any()
    r = _ref(q, kv, idx, sm)
    assert (o - r).abs().max().item() / r.abs().max().item() < 0.08
