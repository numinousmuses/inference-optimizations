"""CPU tests for D1d fused q/k/v norm + neox rope (rtx_qkv_norm_rope.py) under the Triton interpreter, against the
stock order (gemma_qkv_rmsnorm bf16 store, then the neox rope in fp32 with the fp32 cos_sin_cache)."""
from __future__ import annotations

import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

import rtx_qkv_norm_rope as R  # noqa: E402

BF = torch.bfloat16


def _cache(d, base, rope_angles=None, max_pos=64):
    """sglang cos_sin_cache layout [max_pos, d]: cos for d/2 freqs, then sin. rope_angles < d/2 zero-pads the
    inverse frequencies like Gemma4RotaryEmbedding (proportional, partial rotary)."""
    ra = d // 2 if rope_angles is None else rope_angles
    inv = 1.0 / (base ** (torch.arange(0, 2 * ra, 2, dtype=torch.float) / d))
    inv = torch.cat([inv, torch.zeros(d // 2 - ra)])
    t = torch.arange(max_pos, dtype=torch.float)
    f = torch.outer(t, inv)
    return torch.cat([f.cos(), f.sin()], -1)


def _qkv(m, hq, hkv, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    qkv = (torch.randn(m, (hq + 2 * hkv) * d, generator=g) * 2).to(BF)
    q, k, v = qkv.split([hq * d, hkv * d, hkv * d], dim=-1)
    qw = (1 + 0.1 * torch.randn(d, generator=g)).to(BF)
    kw = (1 + 0.1 * torch.randn(d, generator=g)).to(BF)
    return qkv, q, k, v, qw, kw


def _ulps(a, b):
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


@pytest.mark.parametrize("hq,hkv,d,base,ra", [(4, 2, 256, 1e4, None), (4, 1, 512, 1e6, 64)])
def test_matches_stock_order_in_place_on_the_qkv_buffer(hq, hkv, d, base, ra):
    m = 5
    qkv, q, k, v, qw, kw = _qkv(m, hq, hkv, d)
    cs = _cache(d, base, ra)
    pos = torch.tensor([0, 3, 17, 63, 5])
    rq, rk, rv = R.ref(q.clone(), k.clone(), v.clone(), qw, kw, cs, pos, hq, hkv, d, 1e-6)
    R.norm_rope(q, k, v, qw, kw, cs, pos, hq, hkv, d, 1e-6)       # writes through the strided views
    for got, want in ((q, rq), (k, rk), (v, rv)):
        u = _ulps(got.contiguous(), want.contiguous())
        assert int(u.max()) <= 1 and float((u > 0).float().mean()) < 2e-3
    if ra is not None:                                             # nope dims: cos 1 / sin 0 -> plain norm output
        assert torch.equal(cs[:, ra: d // 2], torch.ones_like(cs[:, ra: d // 2]))


def test_kv_shared_layer_normalizes_and_ropes_q_only():
    m, hq, hkv, d = 3, 4, 2, 256
    qkv, q, k, v, qw, kw = _qkv(m, hq, hkv, d, seed=1)
    k0, v0 = k.clone(), v.clone()
    cs = _cache(d, 1e4)
    pos = torch.tensor([1, 2, 9])
    rq, _, _ = R.ref(q.clone(), None, None, qw, None, cs, pos, hq, hkv, d, 1e-6)
    R.norm_rope(q, None, None, qw, None, cs, pos, hq, hkv, d, 1e-6)
    assert int(_ulps(q.contiguous(), rq.contiguous()).max()) <= 1
    assert torch.equal(k, k0) and torch.equal(v, v0)               # untouched


def test_hook_body_declines_off_device():
    import types
    attn = types.SimpleNamespace(rotary_emb=types.SimpleNamespace(cos_sin_cache=_cache(256, 1e4), is_neox_style=True,
                                                                   rotary_dim=256, head_size=256), head_dim=256)
    q = torch.zeros(2, 1024, dtype=BF)
    R.ON = True
    try:
        assert R.try_qkv_norm_rope(attn, torch.zeros(2, dtype=torch.long), q, None, None) is None   # CPU tensor
    finally:
        R.ON = False
