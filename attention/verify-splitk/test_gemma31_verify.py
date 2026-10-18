"""GQA-packed split-K verify (Gemma 4 SWA + full layers): exactness vs the extend_attention semantics, interpreter."""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gemma31_verify as V  # noqa: E402


def _case(g, prefix, hq, hkv, d, n_ext, fp8=False, pool=512):
    kb = torch.randn(pool, hkv, d, generator=g)
    vb = torch.randn(pool, hkv, d, generator=g)
    ks = vs = 1.0
    if fp8:
        ks, vs = 0.02, 0.03
        kb = (kb / ks).clamp(-448, 448).to(torch.float8_e4m3fn)
        vb = (vb / vs).clamp(-448, 448).to(torch.float8_e4m3fn)
    perm = torch.randperm(pool, generator=g)
    locs, off = [], 0
    for n in prefix:
        locs.append(perm[off:off + n])
        off += n
    bs = len(prefix)
    q = torch.randn(bs * n_ext, hq, d, generator=g)
    ke = torch.randn(bs * n_ext, hkv, d, generator=g)
    ve = torch.randn(bs * n_ext, hkv, d, generator=g)
    qo = torch.arange(0, bs * n_ext + 1, n_ext, dtype=torch.int32)
    kvp = torch.tensor([0] + list(torch.tensor(prefix).cumsum(0)), dtype=torch.int32)
    kvi = torch.cat([x for x in locs if x.numel()] or [torch.zeros(1, dtype=torch.long)]).to(torch.int32)
    return q, ke, ve, kb, vb, qo, kvp, kvi, ks, vs


@pytest.mark.parametrize("hq,hkv,window,splits,fp8", [
    (8, 4, 0, 2, False),      # SWA-like group 2, no window
    (8, 4, 24, 3, True),      # SWA-like group 2, window smaller than the prefix, FP8 pool
    (8, 1, 0, 1, False),      # full-like group 8
    (8, 1, 0, 4, True),       # full-like group 8, FP8 pool, 4 splits
])
def test_verify_gqa_matches_extend_semantics(hq, hkv, window, splits, fp8):
    g = torch.Generator().manual_seed(7)
    d, n_ext = 64, 6
    prefix = [0, 5, 40, 33]
    q, ke, ve, kb, vb, qo, kvp, kvi, ks, vs = _case(g, prefix, hq, hkv, d, n_ext, fp8)
    o = torch.empty_like(q)
    V.verify_gqa(q, ke, ve, kb, vb, o, qo, kvp, kvi, n_ext, 0.125, ks, vs, window=window, n_splits=splits)
    ref = V.verify_ref(q, ke, ve, kb, vb, qo, kvp, kvi, n_ext, 0.125, ks, vs, window=window)
    assert torch.allclose(o, ref, rtol=1e-4, atol=1e-5), (o - ref).abs().max()


def test_choose_splits_is_static_and_bounded():
    assert V.choose_splits(32 * 1400, 32, 16, 1024) in (1, 2, 4)
    assert V.choose_splits(1 * 100000, 1, 4, 0) == 16
    assert V.choose_splits(0, 4, 4, 0) == 1


@pytest.mark.parametrize("hq,hkv,window,fp8", [(8, 4, 20, False), (8, 4, 7, True), (4, 1, 1000, False),
                                              (8, 1, 0, True), (8, 1, 0, False)])     # window 0: full-layer prefill
def test_extend_gqa_window_matches_reference(hq, hkv, window, fp8):
    g = torch.Generator().manual_seed(8)
    d = 64
    prefix = [0, 9, 30]                     # window lists (already <= window on the server; may exceed here)
    ext = [37, 5, 18]
    q, ke, ve, kb, vb, qo, kvp, kvi, ks, vs = _case(g, prefix, hq, hkv, d, 1, fp8)
    n = sum(ext)
    q = torch.randn(n, hq, d, generator=g)
    ke = torch.randn(n, hkv, d, generator=g)
    ve = torch.randn(n, hkv, d, generator=g)
    qo = torch.tensor([0] + list(torch.tensor(ext).cumsum(0)), dtype=torch.int32)
    o = torch.empty_like(q)
    V.extend_gqa_window(q, ke, ve, kb, vb, o, qo, kvp, kvi, max(ext), 0.125, ks, vs, window=window)
    ref = V.extend_window_ref(q, ke, ve, kb, vb, qo, kvp, kvi, 0.125, ks, vs, window)
    assert torch.allclose(o, ref, rtol=1e-4, atol=1e-5), (o - ref).abs().max()



@pytest.mark.parametrize("d_split", [2, 4])
def test_extend_gqa_split_d(d_split):
    """split-D (each program accumulates a slice of the output dims, recomputing q.k) equals the reference."""
    g = torch.Generator().manual_seed(9)
    d, hq, hkv = 64, 8, 1
    prefix, ext = [0, 11], [20, 9]
    q, ke, ve, kb, vb, qo, kvp, kvi, ks, vs = _case(g, prefix, hq, hkv, d, 1, True)
    n = sum(ext)
    q = torch.randn(n, hq, d, generator=g)
    ke = torch.randn(n, hkv, d, generator=g)
    ve = torch.randn(n, hkv, d, generator=g)
    qo = torch.tensor([0] + list(torch.tensor(ext).cumsum(0)), dtype=torch.int32)
    o = torch.empty_like(q)
    V.extend_gqa_window(q, ke, ve, kb, vb, o, qo, kvp, kvi, max(ext), 0.125, ks, vs, window=0, d_split=d_split)
    ref = V.extend_window_ref(q, ke, ve, kb, vb, qo, kvp, kvi, 0.125, ks, vs, 0)
    assert torch.allclose(o, ref, rtol=1e-4, atol=1e-5), (o - ref).abs().max()


def test_extend_fa4_glue_assembles_prefix_and_halves():
    """FA4 is GPU-only; a torch stand-in with FA4's varlen + bottom-right causal semantics checks the glue: prefix
    gathered and dequantized in front of the new K/V, cu_seqlens, the two V halves, and the output written."""
    g = torch.Generator().manual_seed(10)
    d, hq, hkv = 64, 8, 1
    prefix, ext = [0, 7, 19], [5, 9, 3]
    q, ke, ve, kb, vb, qo, kvp, kvi, ks, vs = _case(g, prefix, hq, hkv, d, 1, True)
    n = sum(ext)
    q = torch.randn(n, hq, d, generator=g)
    ke = torch.randn(n, hkv, d, generator=g)
    ve = torch.randn(n, hkv, d, generator=g)
    qo = torch.tensor([0] + list(torch.tensor(ext).cumsum(0)), dtype=torch.int32)

    def fake_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, pack_gqa,
                   window_size=(-1, -1)):
        out = torch.empty(q.shape[0], q.shape[1], v.shape[-1])
        gq = q.shape[1] // k.shape[1]
        for b in range(cu_seqlens_q.shape[0] - 1):
            s, e = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
            a, z = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
            kk = k[a:z].repeat_interleave(gq, 1).float()
            vv = v[a:z].repeat_interleave(gq, 1).float()
            sc = torch.einsum("qhd,khd->hqk", q[s:e].float(), kk) * softmax_scale
            off = (z - a) - (e - s)                                   # bottom-right causal
            mask = torch.arange(z - a)[None, :] <= (torch.arange(e - s)[:, None] + off)
            if window_size[0] >= 0:
                mask &= (torch.arange(e - s)[:, None] + off - torch.arange(z - a)[None, :]) <= window_size[0]
            sc = sc.masked_fill(~mask[None], float("-inf"))
            out[s:e] = torch.einsum("hqk,khd->qhd", torch.softmax(sc, -1), vv)
        return out

    o = torch.empty_like(q)
    V.extend_fa4_hd512(q, ke, ve, kb, vb, o, qo, kvp, kvi, 0.125, ks, vs, flash=fake_flash)
    ref = V.extend_window_ref(q, ke, ve, kb, vb, qo, kvp, kvi, 0.125, ks, vs, 0)
    assert torch.allclose(o, ref, rtol=1e-4, atol=1e-4), (o - ref).abs().max()
    # SWA: window list prefix + window_size=(W, 0) == extend_attention's rule q - k <= W
    for w in (4, 15):
        o2 = torch.empty_like(q)
        V.extend_fa4_hd512(q, ke, ve, kb, vb, o2, qo, kvp, kvi, 0.125, ks, vs, flash=fake_flash, window=w)
        ref2 = V.extend_window_ref(q, ke, ve, kb, vb, qo, kvp, kvi, 0.125, ks, vs, w)
        assert torch.allclose(o2, ref2, rtol=1e-4, atol=1e-4), (w, (o2 - ref2).abs().max())


@pytest.mark.parametrize("window", [0, 9])
def test_extend_image_mask_kmax(window):
    """Image batches: an image span [b, e] is bidirectional, causal elsewhere (gemma4_mm.py:368-388). kmax_from_mask
    recovers k <= kmax(q) from the dense mask, and the kernel equals masked attention (window included on SWA)."""
    g = torch.Generator().manual_seed(11)
    d, hq, hkv = 64, 8, 4
    prefix, ext = [0, 6], [20, 12]
    q, ke, ve, kb, vb, qo, kvp, kvi, ks, vs = _case(g, prefix, hq, hkv, d, 1, False)
    n = sum(ext)
    q = torch.randn(n, hq, d, generator=g)
    ke = torch.randn(n, hkv, d, generator=g)
    ve = torch.randn(n, hkv, d, generator=g)
    qo = torch.tensor([0] + list(torch.tensor(ext).cumsum(0)), dtype=torch.int32)
    spans = [[(3, 8), (12, 15)], [(2, 9)]]                  # image spans per request (extend coordinates)
    masks, kmax_ref = [], []
    for b, (p, e) in enumerate(zip(prefix, ext)):
        km = torch.arange(e)
        for lo, hi in spans[b]:
            km[lo:hi + 1] = hi
        kmax_ref.append(km)
        m = torch.zeros(e, p + e, dtype=torch.bool)
        m[:, :p] = True
        m[:, p:] = torch.arange(e)[None, :] <= km[:, None]
        masks.append(m.flatten())
    cm = torch.cat(masks)
    mi = torch.tensor([0, masks[0].numel(), masks[0].numel() + masks[1].numel()], dtype=torch.int64)
    kmax = V.kmax_from_mask(cm, mi, qo, kvp)
    assert torch.equal(kmax.long(), torch.cat(kmax_ref))
    o = torch.empty_like(q)
    V.extend_gqa_window(q, ke, ve, kb, vb, o, qo, kvp, kvi, max(ext), 0.125, ks, vs, window=window, kmax=kmax)
    # reference: dense mask (+ the window rule q - k <= W over absolute positions)
    for b, (p, e) in enumerate(zip(prefix, ext)):
        s0 = int(qo[b])
        loc = kvi[int(kvp[b]):int(kvp[b + 1])].long()
        kk = torch.cat([kb[loc].float() * ks, ke[s0:s0 + e]], 0).repeat_interleave(hq // hkv, 1)
        vv = torch.cat([vb[loc].float() * vs, ve[s0:s0 + e]], 0).repeat_interleave(hq // hkv, 1)
        sc = torch.einsum("qhd,khd->hqk", q[s0:s0 + e], kk) * 0.125
        m = masks[b].view(e, p + e).clone()
        if window > 0:
            qpos = p + torch.arange(e)[:, None]
            kpos = torch.arange(p + e)[None, :]
            m &= (qpos - kpos) <= window
        sc = sc.masked_fill(~m[None], float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", torch.softmax(sc, -1), vv)
        assert torch.allclose(o[s0:s0 + e], ref, rtol=1e-4, atol=1e-5), (b, (o[s0:s0 + e] - ref).abs().max())
    # a mask that is not 'prefix + 0..kmax' is refused (stock keeps it)
    bad = cm.clone()
    bad[5] = ~bad[5]
    assert V.kmax_from_mask(bad, mi, qo, kvp) is None



def test_fa4_plan_from_host_lengths_matches_the_sync_path():
    """Serving path: the per-forward plan from host lengths (window-clipped on SWA) gives the same output as the
    per-layer sync path, and fa4_plan caches per forward and layer kind."""
    import types
    g = torch.Generator().manual_seed(21)
    d, hq, hkv = 64, 8, 4
    prefix, ext = [5, 0, 13], [7, 4, 9]
    q, ke, ve, kb, vb, qo, kvp, kvi, ks, vs = _case(g, prefix, hq, hkv, d, 1, False)
    n = sum(ext)
    q = torch.randn(n, hq, d, generator=g)
    ke = torch.randn(n, hkv, d, generator=g)
    ve = torch.randn(n, hkv, d, generator=g)
    qo = torch.tensor([0] + list(torch.tensor(ext).cumsum(0)), dtype=torch.int32)

    def fake_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, pack_gqa,
                   window_size=(-1, -1)):
        out = torch.empty(q.shape[0], q.shape[1], v.shape[-1])
        for b in range(cu_seqlens_q.shape[0] - 1):
            s, e = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
            a, z = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
            kk = k[a:z].repeat_interleave(q.shape[1] // k.shape[1], 1)
            vv = v[a:z].repeat_interleave(q.shape[1] // v.shape[1], 1)
            sc = torch.einsum("qhd,khd->hqk", q[s:e], kk) * softmax_scale
            off = (z - a) - (e - s)
            mask = torch.arange(z - a)[None, :] <= (torch.arange(e - s)[:, None] + off)
            if window_size[0] >= 0:
                mask &= (torch.arange(e - s)[:, None] + off - torch.arange(z - a)[None, :]) <= window_size[0]
            sc = sc.masked_fill(~mask[None], float("-inf"))
            out[s:e] = torch.einsum("hqk,khd->qhd", torch.softmax(sc, -1), vv)
        return out

    fb = types.SimpleNamespace(extend_seq_lens_cpu=ext, extend_prefix_lens_cpu=prefix)
    plan_f = V.fa4_plan(fb, types.SimpleNamespace(sliding_window_size=0), kvp, kvi, False)
    assert V.fa4_plan(fb, None, kvp, kvi, False) is plan_f            # cached per forward
    o1, o2 = torch.empty_like(q), torch.empty_like(q)
    V.extend_fa4_hd512(q, ke, ve, kb, vb, o1, qo, kvp, kvi, 0.125, ks, vs, flash=fake_flash)
    V.extend_fa4_hd512(q, ke, ve, kb, vb, o2, qo, kvp, kvi, 0.125, ks, vs, flash=fake_flash, plan=plan_f)
    assert torch.allclose(o1, o2, rtol=1e-5, atol=1e-6)
    ref = V.extend_window_ref(q, ke, ve, kb, vb, qo, kvp, kvi, 0.125, ks, vs, 0)
    assert torch.allclose(o2, ref, rtol=1e-4, atol=1e-4)
    # SWA: the backend's window list holds the last min(prefix, S) prefix tokens (triton_backend.py:2530-2534)
    S = 6
    wl = [min(p, S) for p in prefix]
    wkvp = torch.tensor([0] + list(torch.tensor(wl).cumsum(0)), dtype=torch.int32)
    wkvi = torch.cat([kvi[int(kvp[b + 1]) - wl[b]:int(kvp[b + 1])] for b in range(len(prefix))])
    fb2 = types.SimpleNamespace(extend_seq_lens_cpu=ext, extend_prefix_lens_cpu=prefix)
    plan_w = V.fa4_plan(fb2, types.SimpleNamespace(sliding_window_size=S), wkvp, wkvi, True)
    o3 = torch.empty_like(q)
    V.extend_fa4_hd512(q, ke, ve, kb, vb, o3, qo, wkvp, wkvi, 0.125, ks, vs, flash=fake_flash, window=S, plan=plan_w)
    ref_w = V.extend_window_ref(q, ke, ve, kb, vb, qo, kvp, kvi, 0.125, ks, vs, S)
    assert torch.allclose(o3, ref_w, rtol=1e-4, atol=1e-4), (o3 - ref_w).abs().max()
    # no host lengths -> None (caller uses the Triton kernel)
    assert V.fa4_plan(types.SimpleNamespace(), None, kvp, kvi, False) is None
