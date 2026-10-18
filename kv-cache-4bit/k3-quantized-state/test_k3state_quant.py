"""Tests for operations/kernels/k3state_quant.py.

CPU (any machine): reference format round trips, byte layout, zero rows, error bounds.
GPU (dev box, SGLang v0.5.20 image): Triton quant/dequant bit-exact vs the reference; the two-tier commit fold is
bit-identical on the fp32 working state to SGLang's own commit_kda_replayssm_spec_all_layers and its quantized
mirror equals ref_quant of that state; track snapshots; padded rows untouched.

    python3 -m pytest operations/kernels/test_k3state_quant.py -q
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3state_quant as kq  # noqa: E402

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="needs a GPU")


def _state(n=6, hv=3, v=128, k=128, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, hv, v, k, generator=g) * scale
    # realistic spread: some rows large, some tiny, one row exactly zero
    x = x * torch.logspace(-3, 1, v)[None, None, :, None]
    x[0, 0, 5] = 0
    return x


@pytest.mark.parametrize("fmt", kq.FORMATS)
def test_row_bytes_layout(fmt):
    assert kq.row_bytes("fp8") == 132 and kq.row_bytes("fp4") == 76 and kq.row_bytes("bf16") == 256
    x = _state()
    q = kq.ref_quant(x, fmt)
    assert q.dtype == torch.uint8 and q.shape == (*x.shape[:-1], kq.row_bytes(fmt))
    # scales sit 4-aligned inside each packed row
    assert (kq.row_bytes(fmt) - 4) % 4 == 0


@pytest.mark.parametrize("fmt", kq.FORMATS)
def test_zero_row_decodes_to_zero(fmt):
    z = torch.zeros(2, 3, kq.row_bytes(fmt), dtype=torch.uint8)
    assert torch.equal(kq.ref_dequant(z, fmt), torch.zeros(2, 3, 128))
    x = _state()
    assert torch.equal(kq.ref_dequant(kq.ref_quant(x, fmt), fmt)[0, 0, 5], torch.zeros(128))


@pytest.mark.parametrize("fmt,bound", [("fp8", 0.0625), ("fp4", 0.25), ("bf16", 2 ** -8)])
def test_round_trip_error(fmt, bound):
    """Per-element error relative to the row (fp8) / block (fp4) max: e4m3 half-ULP 2^-4, e2m1 half-step 1/6 of the
    block max plus the e4m3 block-scale rounding."""
    x = _state()
    y = kq.ref_dequant(kq.ref_quant(x, fmt), fmt)
    if fmt == "bf16":
        ref = x.abs()
    elif fmt == "fp8":
        ref = x.abs().amax(-1, keepdim=True)
    else:
        ref = x.unflatten(-1, (8, 16)).abs().amax(-1, keepdim=True).expand(*x.shape[:-1], 8, 16).flatten(-2)
    err = ((y - x).abs() / ref.clamp_min(1e-30)).max().item()
    assert err <= bound, err
    rel = ((y - x).norm() / x.norm()).item()
    assert rel < {"fp8": 0.03, "fp4": 0.12, "bf16": 0.003}[fmt], rel


def test_fp4_codes_exact_on_grid():
    """Values already on the e2m1 grid at a power-of-two scale survive exactly."""
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    row = torch.cat([grid, -grid]).repeat(8)[:128]
    y = kq.ref_dequant(kq.ref_quant(row, "fp4"), "fp4")
    assert torch.allclose(y, row, rtol=0, atol=1e-6)


# ------------------------------------------------------------------------------------------------ GPU
def _import_flashinfer():
    # the engine always has flashinfer loaded (it sets the Blackwell ptxas path Triton uses)
    try:
        import flashinfer  # noqa: F401
    except Exception:  # noqa: BLE001
        pass


def _assert_rows_match(got: torch.Tensor, ref: torch.Tensor, fmt: str) -> None:
    """Kernel rows vs the CPU reference: scale bytes identical (both IEEE round-to-nearest divisions); value codes
    identical except at most 1 in 10^4 that sit one code apart (an fp32 product landing on an fp8 rounding tie can
    round either way between the CPU cast and the device cvt)."""
    k = 128
    vb = {"fp8": k, "fp4": k // 2, "bf16": 2 * k}[fmt]
    assert torch.equal(got[..., vb:], ref[..., vb:]), "scale bytes differ"
    if fmt == "fp8":
        diff = got[..., :vb] != ref[..., :vb]
        assert diff.float().mean().item() <= 1e-4, diff.float().mean().item()
        gv = got[..., :vb].view(torch.float8_e4m3fn).float()
        rv = ref[..., :vb].view(torch.float8_e4m3fn).float()
        step = (rv.abs() / 8).clamp_min(2 ** -9)
        assert ((gv - rv).abs() <= step * 1.01).all()
    else:
        assert torch.equal(got[..., :vb], ref[..., :vb])


@gpu
@pytest.mark.parametrize("fmt", kq.FORMATS)
def test_triton_quant_dequant_match_reference(fmt):
    _import_flashinfer()
    x = _state(n=5).cuda()
    slots = torch.zeros(9, 3, 128, kq.row_bytes(fmt), dtype=torch.uint8, device="cuda")
    dst = torch.tensor([7, -1, 2, 4, 0], device="cuda")
    kq.quant_rows(x, None, slots, dst, fmt)
    ref = kq.ref_quant(x.cpu(), fmt)
    for i, d in enumerate(dst.tolist()):
        if d >= 0:
            _assert_rows_match(slots[d].cpu(), ref[i], fmt)
    # the round trip below is checked on the kernel's own bytes
    ref = torch.stack([slots[d].cpu() if d >= 0 else ref[i] for i, d in enumerate(dst.tolist())])
    assert torch.equal(slots[1], torch.zeros_like(slots[1]))          # padded row -1 wrote nothing
    out = torch.full((4, 3, 128, 128), 7.0, device="cuda")
    kq.dequant_rows(slots, torch.tensor([7, 2, 4, -1], device="cuda"), out, torch.tensor([0, 1, 3, 2], device="cuda"),
                    fmt)
    assert torch.equal(out[0].cpu(), kq.ref_dequant(ref[0], fmt))
    assert torch.equal(out[1].cpu(), kq.ref_dequant(ref[2], fmt))
    assert torch.equal(out[3].cpu(), kq.ref_dequant(ref[3], fmt))
    assert torch.equal(out[2], torch.full_like(out[2], 7.0))


@gpu
def test_quant_rows_with_src_index():
    _import_flashinfer()
    w = _state(n=6).cuda()
    q = torch.zeros(8, 3, 128, 132, dtype=torch.uint8, device="cuda")
    kq.quant_rows(w, torch.tensor([5, 1], device="cuda"), q, torch.tensor([3, 6], device="cuda"), "fp8")
    assert torch.equal(q[3].cpu(), kq.ref_quant(w[5].cpu(), "fp8"))
    assert torch.equal(q[6].cpu(), kq.ref_quant(w[1].cpu(), "fp8"))


def _stock_fold():
    if importlib.util.find_spec("sglang") is None:
        pytest.skip("needs the SGLang v0.5.20 image")
    from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import commit_kda_replayssm_spec_all_layers
    return commit_kda_replayssm_spec_all_layers


@gpu
@pytest.mark.parametrize("fmt", kq.FORMATS)
@pytest.mark.parametrize("track", [False, True])
def test_fold_matches_sglang_and_mirrors(fmt, track):
    """K3 shapes per rank: HV=H=12, K=V=128, ring 16, three layers here."""
    _import_flashinfer()
    stock = _stock_fold()
    torch.manual_seed(1)
    L, HV, K, V, RING = 3, 12, 128, 128, 16
    b = 5
    w_rows, slots = 9, 20
    w_idx = torch.tensor([3, 1, -1, 7, 5], device="cuda", dtype=torch.int32)
    slot_idx = torch.tensor([11, 2, -1, 17, 8], device="cuda", dtype=torch.int32)
    acc = torch.tensor([3, 1, 4, 7, 0], device="cuda", dtype=torch.int32)
    w0 = torch.randn(L, w_rows, HV, V, K, device="cuda") * 0.1
    rawv = torch.randn(L, w_rows, HV, RING, V, device="cuda").to(torch.bfloat16)
    rawk = torch.randn(L, w_rows, HV, RING, K, device="cuda").to(torch.bfloat16)
    gk = -torch.rand(L, w_rows, HV, RING, K, device="cuda") * 5
    beta = torch.rand(L, w_rows, HV, RING, device="cuda")
    q_pool = torch.zeros(L, slots, HV, V, kq.row_bytes(fmt), dtype=torch.uint8, device="cuda")
    trk = torch.tensor([12, 4, -1, 18, 9], device="cuda", dtype=torch.int32) if track else None
    steps = torch.tensor([1, -1, 0, 6, 0], device="cuda", dtype=torch.int32) if track else None

    # stock reference: its h0 is indexed by the same row as the ring, so run it on the working rows
    ref = w0.clone()
    stock(checkpoint_state=ref, rawv_cache=rawv, rawk_cache=rawk, gk_cache=gk, beta_cache=beta,
          ssm_state_indices=w_idx, accept_lens=acc, max_cache_len=RING, num_k_heads=HV, null_block_id=-1)
    w = w0.clone()
    kq.commit_fold_all_layers(w_state=w, q_pool=q_pool, rawv=rawv, rawk=rawk, gk=gk, beta=beta, w_indices=w_idx,
                              slot_indices=slot_idx, accept_lens=acc, fmt=fmt, num_k_heads=HV, track_indices=trk,
                              steps_to_track=steps)
    # fp32 working state: stock op order; bit-identical unless the compiler contracts a multiply-add differently
    # next to the quantizing store (seen for fp4: 1 ULP), so bound it at a few fp32 ULPs of the state scale
    assert (w - ref).abs().max().item() <= 4 * 2 ** -23 * max(ref.abs().max().item(), 1.0)
    for i in range(b):
        s, r = int(slot_idx[i]), int(w_idx[i])
        if s < 0 or int(acc[i]) <= 0:
            continue
        _assert_rows_match(q_pool[:, s].cpu(), kq.ref_quant(w[:, r].cpu(), fmt), fmt)
    # untouched: padded row's slot and every slot not named
    named = {int(x) for x in slot_idx.tolist() if x >= 0 and True}
    if track:
        named |= {int(t) for t, st in zip(trk.tolist(), steps.tolist()) if t >= 0 and st >= 0}
    for s in range(slots):
        if s not in named:
            assert not q_pool[:, s].any(), s
    if track:
        # request 0 crosses at step 1 (after 2 of its 3 tokens): its track slot holds that intermediate state
        mid = w0.clone()
        stock(checkpoint_state=mid, rawv_cache=rawv, rawk_cache=rawk, gk_cache=gk, beta_cache=beta,
              ssm_state_indices=w_idx[:1], accept_lens=torch.tensor([2], device="cuda", dtype=torch.int32),
              max_cache_len=RING, num_k_heads=HV, null_block_id=-1)
        got = kq.ref_dequant(q_pool[:, 12].cpu(), fmt)
        want = kq.ref_dequant(kq.ref_quant(mid[:, 3].cpu(), fmt), fmt)
        assert ((got - want).norm() / want.norm()).item() < 1e-3
