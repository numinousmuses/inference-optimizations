"""CPU tests for the D1c fused FP8 KV store (rtx_kv_store.py) under the Triton interpreter, against the stock chain
k.clone().div_(s).to(float8_e4m3fn) -> k_cache[loc] = ... (memory_pool.py:2563-2569 + _set_kv_buffer_impl)."""
from __future__ import annotations

import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

import rtx_kv_store as S  # noqa: E402

BF = torch.bfloat16


def _stock(k, v, kc, vc, loc, ks, vs, skip=0):
    """clone / div_ / to(fp8), then store_cache's scatter, which skips reserved_skip_index (default slot 0)."""
    kk, vv = k.clone(), v.clone()
    if ks is not None:
        kk.div_(ks)
    if vs is not None:
        vv.div_(vs)
    keep = loc != skip
    kc[loc[keep]] = kk.to(torch.float8_e4m3fn).view(torch.uint8)[keep]
    vc[loc[keep]] = vv.to(torch.float8_e4m3fn).view(torch.uint8)[keep]


@pytest.mark.parametrize("scale", [None, 1.0, 0.375, 0.3])
@pytest.mark.parametrize("hk,dk,hv,dv", [(4, 64, 4, 64), (2, 128, 2, 96)])
def test_matches_stock_chain_on_strided_qkv_views(scale, hk, dk, hv, dv):
    g = torch.Generator().manual_seed(0)
    n, slots = 7, 23
    qkv = (torch.randn(n, 3 * 128 + hk * dk + hv * dv, generator=g) * 6).to(BF)   # K/V as column slices
    k = qkv[:, 384:384 + hk * dk].view(n, hk, dk)
    v = qkv[:, 384 + hk * dk:].view(n, hv, dv)
    assert not k.is_contiguous()
    loc = torch.randperm(slots, generator=g)[:n]
    loc[3] = 0                                         # the reserved padding slot: stock store_cache skips it
    kc0 = torch.randint(0, 255, (slots, hk, dk), generator=g, dtype=torch.uint8)
    vc0 = torch.randint(0, 255, (slots, hv, dv), generator=g, dtype=torch.uint8)
    kc_ref, vc_ref = kc0.clone(), vc0.clone()
    _stock(k, v, kc_ref, vc_ref, loc, scale, scale)
    k_before = k.clone()
    kc, vc = kc0.clone(), vc0.clone()
    S.store_fp8(k, v, kc.view(torch.float8_e4m3fn), vc, loc, scale, scale)
    assert torch.equal(kc, kc_ref) and torch.equal(vc, vc_ref)    # written rows equal, other rows untouched
    assert torch.equal(k, k_before)                                # inputs never mutated (clones are skippable)


def test_overflow_saturates_where_stock_makes_nan():
    k = torch.tensor([[[500.0, -600.0, 448.0, 1.0]]], dtype=BF)
    kc = torch.zeros(2, 1, 4, dtype=torch.uint8)
    vc = torch.zeros(2, 1, 4, dtype=torch.uint8)
    S.store_fp8(k, k.clone(), kc, vc, torch.tensor([1]), None, None)
    assert kc[1].view(torch.float8_e4m3fn).float().flatten().tolist() == [448.0, -448.0, 448.0, 1.0]


def test_scale_arg_keeps_the_fp32_scalar_like_torch_opmath():
    is_ptr, t, f = S._scale_arg(0.3)
    assert not is_ptr and t is None and f == float(torch.tensor(0.3, dtype=torch.float32))
    assert S._scale_arg(None) == (False, None, 1.0)
    assert S._scale_arg(torch.tensor(2.0))[2] == 2.0


def test_try_store_declines_without_cuda_or_when_off(monkeypatch):
    class Pool:
        pass
    monkeypatch.setattr(S, "ON", True)
    k = torch.zeros(2, 1, 4, dtype=BF)
    assert S.try_store(Pool(), 0, torch.tensor([0, 1]), k, k, None, None, None) is False
    assert S.skip_clone(Pool()) is False             # not an exact MHATokenToKVPool (sglang absent here)
    monkeypatch.setattr(S, "ON", False)
    assert S.skip_clone(Pool()) is False


def test_reserved_slot_skip_matches_store_cache_and_can_be_disabled():
    """bench_rtx_7aa9ede3 flagged a byte mismatch at T=288 only: randperm placed slot 0 in loc. store_cache skips
    reserved_skip_index (default 0); the fused store now does the same, and -1 writes it."""
    k = torch.ones(2, 1, 4, dtype=BF)
    kc = torch.zeros(3, 1, 4, dtype=torch.uint8)
    vc = torch.zeros(3, 1, 4, dtype=torch.uint8)
    S.store_fp8(k, k.clone(), kc, vc, torch.tensor([0, 2]))
    assert int(kc[0].sum()) == 0 and int(kc[2].sum()) > 0
    S.store_fp8(k, k.clone(), kc, vc, torch.tensor([0, 2]), reserved_skip_index=-1)
    assert int(kc[0].sum()) > 0
