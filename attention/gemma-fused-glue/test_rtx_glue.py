"""CPU tests for the Gemma D1 glue chains (rtx_glue.py) under the Triton interpreter.

The interpreter truncates fp32 -> bf16/fp8 casts, so the kernel runs with EMU=1 (bit-exact round-to-nearest-even
emulation, the semantics of the GPU's cvt.rn instructions). The norms are compared with the stock formulas op for op:
fp32 reduction order may differ, which allows a rare 1-ulp bf16 flip. The quant is compared bit-exactly against the
stock per_token_group_quant semantics applied to the kernel's own bf16 output.

    TRITON_INTERPRET=1 python3 -m pytest -q operations/kernels/gemma31/test_rtx_glue.py
"""
from __future__ import annotations

import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

import rtx_glue as G  # noqa: E402

BF = torch.bfloat16


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    # bf16 bit patterns are monotone within a sign; compare same-sign magnitudes
    return (ia - ib).abs()


def _close_bf16(got: torch.Tensor, ref: torch.Tensor, max_frac: float = 2e-3) -> None:
    d = _ulps(got, ref)
    assert int(d.max()) <= 1, f"max ulp diff {int(d.max())}"
    assert float((d > 0).float().mean()) <= max_frac, f"{float((d > 0).float().mean()):.4%} elements differ by 1 ulp"


def _inputs(m: int, n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    h = (torch.randn(m, n, generator=g) * 2.0).to(BF)
    r = (torch.randn(m, n, generator=g) * 4.0).to(BF)
    if m > 2:
        h[1].zero_()                      # all-zero row: rsqrt(eps), amax clamp 1e-10
        r[2] *= 300.0                     # large residual row
    w1 = (1.0 + 0.2 * torch.randn(n, generator=g)).to(BF)
    w2 = (1.0 + 0.2 * torch.randn(n, generator=g)).to(BF)
    return h, r, w1, w2


def test_emulated_casts_match_torch_rne():
    import triton
    import triton.language as tl

    @triton.jit
    def k(X, YB, YQ, N, BLOCK: tl.constexpr):
        c = tl.arange(0, BLOCK)
        x = tl.load(X + c, mask=c < N, other=0.0)
        tl.store(YB + c, G._to_bf16(x, True), mask=c < N)
        tl.store(YQ + c, G._to_e4m3(tl.minimum(x, 448.0), True), mask=c < N)

    g = torch.Generator().manual_seed(1)
    x = torch.cat([torch.randn(20000, generator=g) * 50, torch.randn(4000, generator=g) * 0.01,
                   torch.tensor([1.0625, 1.1875, 17.0, 19.0, 300.0, 440.0, -1.0625, 0.0, -0.0, 2.0 ** -10,
                                 3 * 2.0 ** -10, 448.0, -448.0, -448.2])])
    x = x.clamp(-448.0, 1e4)
    # exact bf16 ties, normal range (the interpreter flushes fp32 subnormals; activations never sit at 1e-40)
    ties = (((torch.arange(0, 2000, dtype=torch.int32) + 0x3C00) << 16) | 0x8000).view(torch.float32)
    x = torch.cat([x, ties[torch.isfinite(ties) & (ties.abs() < 448)]])
    yb = torch.empty(x.numel(), dtype=BF)
    yq = torch.empty(x.numel(), dtype=torch.float8_e4m3fn)
    k[(1,)](x, yb, yq.view(torch.uint8), x.numel(), BLOCK=triton.next_power_of_2(x.numel()))
    assert torch.equal(yb.view(torch.int16), x.to(BF).view(torch.int16))
    ref_q = torch.minimum(x, torch.tensor(448.0)).clamp_min(-448.0).to(torch.float8_e4m3fn)
    assert torch.equal(yq.view(torch.uint8), ref_q.view(torch.uint8))


@pytest.mark.parametrize("m,n,quant", [(5, 5376, True), (3, 1024, False), (1, 5376, True)])
def test_post_attn_chain(m, n, quant):
    h, r, w1, w2 = _inputs(m, n)
    eps1, eps2 = 1e-6, 1e-6
    h1_ref = G.ref_rmsnorm(h, w1, eps1)
    out_ref, res_ref = G.ref_fused_add_rmsnorm(h1_ref, r, w2, eps2)
    res = r.clone()
    out, res_got, q, s = G.post_attn_chain(h, res, w1, eps1, w2, eps2, quant)
    assert res_got.data_ptr() == res.data_ptr()          # in place, like fused_add_rmsnorm
    _close_bf16(res_got, res_ref)
    _close_bf16(out, out_ref)
    if quant:
        q_ref, s_ref = G.ref_group_quant(out)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        assert torch.equal(s, s_ref)
        assert s.stride() == (1, (m + 3) // 4 * 4) and s.shape == (m, n // 128)   # column-major, TMA-aligned
    else:
        assert q is None and s is None


@pytest.mark.parametrize("m,n,quant", [(5, 5376, True), (3, 1024, False)])
def test_cross_layer_chain(m, n, quant):
    h, r, w1, w2 = _inputs(m, n, seed=3)
    scalar = torch.tensor([0.8125], dtype=BF)
    eps1, eps2 = 1e-6, 1e-6
    rout_ref = G.ref_rmsnorm_residual_scalar(h, w1, r, scalar, eps1)
    out_ref = G.ref_rmsnorm(rout_ref, w2, eps2)
    rout, out, q, s = G.cross_layer_chain(h, r, w1, eps1, scalar, w2, eps2, quant)
    _close_bf16(rout, rout_ref)
    _close_bf16(out, G.ref_rmsnorm(rout, w2, eps2))     # the next norm reads the stored bf16 layer output
    _close_bf16(out, out_ref)
    if quant:
        q_ref, s_ref = G.ref_group_quant(out)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        assert torch.equal(s, s_ref)


def test_prequant_stash_rules():
    x = torch.zeros(4, 256, dtype=BF)
    q, s = G.alloc_prequant(4, 256, "cpu")
    G.set_prequant(x, q, s)
    assert G.take_prequant(x, [128, 64]) is None              # wrong block
    assert G.take_prequant(x[:2], [128, 128]) is None         # other shape
    assert G.take_prequant(torch.zeros_like(x), [128, 128]) is None   # other buffer
    got = G.take_prequant(x, [128, 128])
    assert got is not None and got[0] is q and got[1] is s
    assert G.take_prequant(x, [128, 128]) is None             # once
    G.set_prequant(x, q, s)
    G.clear_prequant()
    assert G.take_prequant(x, [128, 128]) is None


def test_prenormed_stash_identity_and_version():
    a, out = torch.ones(2, 128, dtype=BF), torch.zeros(2, 128, dtype=BF)
    G.set_prenormed(a, out, None, None)
    assert G.take_prenormed(a.clone()) is None                # identity
    G.set_prenormed(a, out, None, None)
    a.add_(1)                                                 # in-place change after the stash
    assert G.take_prenormed(a) is None
    G.set_prenormed(a, out, None, None)
    assert G.take_prenormed(a) is out
    assert G.take_prenormed(a) is None                        # single use
    q, s = G.alloc_prequant(2, 128, "cpu")
    G.set_prenormed(a, out, q, s)
    assert G.take_prenormed(a) is out
    assert G.take_prequant(out, [128, 128]) == (q, s)         # the prequant follows the prenormed input
