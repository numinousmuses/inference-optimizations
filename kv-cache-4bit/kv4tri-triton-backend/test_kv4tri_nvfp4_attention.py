"""CPU tests (Triton interpreter) for kv4tri_nvfp4_attention: FP4 bit decoding, decode (split-KV, GQA groups that are
not powers of two, empty requests, logit cap, global scales) and extend (prefix + new tokens, causal, custom mask,
sliding window with SGLang's `q - kv <= W` rule, DV slices, head_dim 512) against float32 references computed from
the dequantized pool."""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

import kv4tri_nvfp4_attention as k4  # noqa: E402


def test_quantize_roundtrip_matches_e2m1_grid():
    torch.manual_seed(0)
    x = torch.randn(7, 3, 64) * 3
    packed, sf = k4.nvfp4_quantize_ref(x)
    assert packed.shape == (7, 3, 32) and sf.shape == (7, 3, 4) and packed.dtype == torch.uint8
    y = k4.nvfp4_dequantize_ref(packed, sf)
    # E2M1 relative step is at most 1/3 of a block's max after the E4M3 scale rounding
    blk = x.reshape(-1, 16)
    err = (y.reshape(-1, 16) - blk).abs().amax(-1)
    assert torch.all(err <= blk.abs().amax(-1) * 0.2 + 1e-6)


def test_fp4_pair_decodes_every_code_and_scale():
    # every byte value against every E4M3 scale byte that the quantizer can produce (0x00..0x7E, both signs)
    import triton
    import triton.language as tl

    @triton.jit
    def _k(P, S, LO, HI, N: tl.constexpr):
        o = tl.arange(0, N)
        lo, hi = k4._fp4_pair(tl.load(P + o), tl.load(S + o))
        tl.store(LO + o, lo)
        tl.store(HI + o, hi)

    p = torch.arange(256, dtype=torch.int32).repeat_interleave(64).to(torch.uint8)
    s = torch.cat([torch.arange(0, 0x7F), torch.arange(0x80, 0xFF)]).to(torch.uint8)
    s = s[torch.arange(p.numel()) % s.numel()]
    lo = torch.empty(p.numel())
    hi = torch.empty(p.numel())
    _k[(1,)](p, s, lo, hi, N=p.numel())
    vals = torch.tensor(k4.E2M1_VALUES)
    sc = s.view(torch.float8_e4m3fn).float()
    pl, ph = p.long() & 0xF, p.long() >> 4
    ref_lo = vals[pl & 7] * torch.where((pl & 8) != 0, -1.0, 1.0) * sc
    ref_hi = vals[ph & 7] * torch.where((ph & 8) != 0, -1.0, 1.0) * sc
    assert torch.equal(lo, ref_lo) and torch.equal(hi, ref_hi)


def _pool(slots, KVH, D, gs=1.0, seed=0):
    torch.manual_seed(seed)
    k = torch.randn(slots, KVH, D) * 1.5
    v = torch.randn(slots, KVH, D) * 0.7
    kp, ks = k4.nvfp4_quantize_ref(k, gs)
    vp, vs = k4.nvfp4_quantize_ref(v, gs)
    return kp, ks, vp, vs, k4.nvfp4_dequantize_ref(kp, ks, gs), k4.nvfp4_dequantize_ref(vp, vs, gs)


def _decode_case(HQ, KVH, D, lens, splits, gs=1.0, logit_cap=0.0, seed=0):
    slots = sum(lens) + 16
    kp, ks, vp, vs, kd, vd = _pool(slots, KVH, D, gs, seed)
    B = len(lens)
    q = torch.randn(B, HQ, D).to(torch.bfloat16)
    kv_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), dtype=torch.int32)
    kv_indices = torch.randperm(slots)[: sum(lens)].to(torch.int64)
    MS = max(splits)
    att = torch.full((B + 1, HQ, MS + 1, D), float("nan"))
    lse = torch.full((B + 1, HQ, MS + 1), float("nan"))
    nsp = torch.tensor(splits + [1], dtype=torch.int32)
    o = torch.empty(B, HQ, D, dtype=torch.bfloat16)
    k4.nvfp4_decode_fwd(q, kp, vp, ks, vs, o, kv_indptr, kv_indices, att, lse, nsp, MS, D ** -0.5, gs, gs,
                        logit_cap, block_n=16, num_stages=1)
    ref = k4.decode_reference(q, kd, vd, kv_indptr, kv_indices, D ** -0.5, logit_cap)
    return o.float(), ref


@pytest.mark.parametrize("HQ,KVH", [(12, 2), (8, 1), (4, 2)])   # G = 6 (Qwen 3.8), 8 (Qwen 3.6 / Gemma global), 2
def test_decode_matches_reference(HQ, KVH):
    o, ref = _decode_case(HQ, KVH, 64, lens=[70, 1, 33], splits=[3, 1, 4])
    assert (o - ref).abs().max() < 0.02


def test_decode_global_scale_and_logit_cap():
    o, ref = _decode_case(8, 2, 64, lens=[45, 20], splits=[2, 2], gs=0.25, logit_cap=5.0, seed=4)
    assert (o - ref).abs().max() < 0.02


def test_decode_empty_request_writes_zeros():
    o, _ = _decode_case(4, 2, 32, lens=[0, 9], splits=[2, 2])
    assert torch.all(o[0] == 0)


def _extend_case(HQ, KVH, D, prefixes, news, seed=0, gs=1.0):
    torch.manual_seed(seed)
    P_tot, T_tot = sum(prefixes), sum(news)
    kp, ks, vp, vs, kd, vd = _pool(P_tot + 8, KVH, D, gs, seed)
    q = torch.randn(T_tot, HQ, D).to(torch.bfloat16)
    ke = (torch.randn(T_tot, KVH, D) * 1.5).to(torch.bfloat16)
    ve = (torch.randn(T_tot, KVH, D) * 0.7).to(torch.bfloat16)
    qo = torch.tensor([0] + list(torch.cumsum(torch.tensor(news), 0)), dtype=torch.int32)
    kvp = torch.tensor([0] + list(torch.cumsum(torch.tensor(prefixes), 0)), dtype=torch.int32)
    idx = torch.randperm(P_tot + 8)[:P_tot].to(torch.int64)
    return q, ke, ve, kp, ks, vp, vs, kd, vd, qo, kvp, idx, max(news)


@pytest.mark.parametrize("window", [-1, 20])
@pytest.mark.parametrize("dv", [None, 32])
def test_extend_matches_reference(window, dv):
    HQ, KVH, D = 12, 2, 64
    q, ke, ve, kp, ks, vp, vs, kd, vd, qo, kvp, idx, mx = _extend_case(HQ, KVH, D, [70, 5, 0], [19, 33, 7])
    o = torch.empty_like(q)
    k4.nvfp4_extend_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, mx, D ** -0.5, sliding_window_size=window,
                        block_t=4, block_n=16, dk=32, dv=dv, num_stages=1)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, sliding_window_size=window)
    assert (o.float() - ref).abs().max() < 0.03


def test_extend_global_scale_prefix_only():
    q, ke, ve, kp, ks, vp, vs, kd, vd, qo, kvp, idx, mx = _extend_case(4, 1, 32, [40], [9], seed=3, gs=0.5)
    o = torch.empty_like(q)
    k4.nvfp4_extend_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, mx, 32 ** -0.5, k_scale=0.5, v_scale=0.5,
                        block_t=8, block_n=16, dk=32, num_stages=1)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, 32 ** -0.5)
    assert (o.float() - ref).abs().max() < 0.03


def test_extend_custom_mask_new_tokens_only():
    # EAGLE target-verify style: a custom mask over [q_len, prefix + q_len]; the prefix part is ignored
    HQ, KVH, D = 8, 2, 32
    prefixes, news = [30, 12], [4, 4]
    q, ke, ve, kp, ks, vp, vs, kd, vd, qo, kvp, idx, mx = _extend_case(HQ, KVH, D, prefixes, news, seed=7)
    masks, mptr = [], [0]
    g = torch.Generator().manual_seed(1)
    for P, Tn in zip(prefixes, news):
        m = torch.rand(Tn, P + Tn, generator=g) > 0.4
        m[:, :P] = False                      # must be ignored by the kernel (skip_prefix_custom_mask)
        m[torch.arange(Tn), P + torch.arange(Tn)] = True
        masks.append(m.flatten())
        mptr.append(mptr[-1] + m.numel())
    cm = torch.cat(masks)
    mi = torch.tensor(mptr[:-1], dtype=torch.int64)
    o = torch.empty_like(q)
    k4.nvfp4_extend_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, mx, D ** -0.5, custom_mask=cm, mask_indptr=mi,
                        block_t=4, block_n=16, dk=32, num_stages=1)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, custom_mask=cm, mask_indptr=mi)
    assert (o.float() - ref).abs().max() < 0.03


def test_extend_head_dim_512_dv_split():
    # Gemma 4 global layers: 16 query heads on 2 KV heads, head_dim 512 (scaled down: 8 on 1, D 128, dv slice 64)
    q, ke, ve, kp, ks, vp, vs, kd, vd, qo, kvp, idx, mx = _extend_case(8, 1, 128, [37], [10], seed=9)
    o = torch.empty_like(q)
    k4.nvfp4_extend_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, mx, 128 ** -0.5, block_t=2, block_n=16, dk=64,
                        dv=64, num_stages=1)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, 128 ** -0.5)
    assert (o.float() - ref).abs().max() < 0.03


def test_gather_dequant_bf16_exact():
    kp, ks, vp, vs, kd, vd = _pool(50, 3, 64, gs=0.5, seed=12)
    idx = torch.tensor([7, 0, 49, 3, 3, 21], dtype=torch.int64)
    out = torch.empty(8, 3, 64, dtype=torch.bfloat16)
    k4.nvfp4_gather_dequant(kp, ks, idx, out, 0.5, block_t=4)
    assert torch.equal(out[:6].float(), kd[idx])


@pytest.mark.parametrize("splits", [1, 3])
@pytest.mark.parametrize("window", [-1, 20])
def test_verify_matches_reference(splits, window):
    HQ, KVH, D = 12, 2, 64          # G = 6 -> up to 8 tokens per request
    q, ke, ve, kp, ks, vp, vs, kd, vd, qo, kvp, idx, mx = _extend_case(HQ, KVH, D, [70, 0, 33], [4, 8, 1], seed=21)
    o = torch.empty_like(q)
    k4.nvfp4_verify_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, mx, D ** -0.5, sliding_window_size=window,
                        num_splits=splits, block_n=16, dk=32, num_stages=1)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, sliding_window_size=window)
    assert (o.float() - ref).abs().max() < 0.03


def test_verify_custom_mask_and_global_scale():
    HQ, KVH, D = 8, 2, 32
    prefixes, news = [40, 17], [4, 4]
    q, ke, ve, kp, ks, vp, vs, kd, vd, qo, kvp, idx, mx = _extend_case(HQ, KVH, D, prefixes, news, seed=5, gs=0.25)
    masks, mptr = [], [0]
    g = torch.Generator().manual_seed(3)
    for P, Tn in zip(prefixes, news):
        m = torch.rand(Tn, P + Tn, generator=g) > 0.5
        m[torch.arange(Tn), P + torch.arange(Tn)] = True
        masks.append(m.flatten())
        mptr.append(mptr[-1] + m.numel())
    cm = torch.cat(masks)
    mi = torch.tensor(mptr[:-1], dtype=torch.int64)
    o = torch.empty_like(q)
    k4.nvfp4_verify_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, mx, D ** -0.5, k_scale=0.25, v_scale=0.25,
                        custom_mask=cm, mask_indptr=mi, num_splits=2, block_n=16, dk=32, num_stages=1)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, custom_mask=cm, mask_indptr=mi)
    assert (o.float() - ref).abs().max() < 0.03


def _mask_case(prefixes_full, news, window, seed):
    """Gemma-4-image-style custom mask over the FULL sequence (causal + a bidirectional block), with the prefix
    windowed as SGLang does on sliding layers (last `window` prefix tokens, offsets = full - windowed)."""
    g = torch.Generator().manual_seed(seed)
    pw = [min(p, window) if window > 0 else p for p in prefixes_full]
    offs = [pf - p for pf, p in zip(prefixes_full, pw)]
    masks, mptr = [], [0]
    for pf, Tn in zip(prefixes_full, news):
        m = torch.ones(Tn, pf + Tn, dtype=torch.bool).tril(diagonal=pf)
        a = int(torch.randint(0, max(1, Tn - 2), (1,), generator=g))
        m[a:a + 3, pf + a:pf + a + 3] = True         # a bidirectional "image" span among the new tokens
        masks.append(m.flatten())
        mptr.append(mptr[-1] + m.numel())
    return pw, torch.cat(masks), torch.tensor(mptr[:-1], dtype=torch.int64), torch.tensor(offs, dtype=torch.int32)


@pytest.mark.parametrize("fn", ["extend", "verify"])
@pytest.mark.parametrize("window", [-1, 20])
def test_custom_mask_with_window_offsets_and_int64_indptr(fn, window):
    HQ, KVH, D = 4, 2, 32
    prefixes_full, news = [50, 7, 0], [8, 6, 5]
    pw, cm, mi, offs = _mask_case(prefixes_full, news, window, seed=4)
    q, ke, ve, kp, ks, vp, vs, kd, vd, qo, kvp, idx, mx = _extend_case(HQ, KVH, D, pw, news, seed=6)
    qo, kvp = qo.to(torch.int64), kvp.to(torch.int64)     # SGLang hands int64 indptrs on some paths
    woff = offs if window > 0 else None
    o = torch.empty_like(q)
    if fn == "extend":
        k4.nvfp4_extend_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, mx, D ** -0.5, sliding_window_size=window,
                            custom_mask=cm, mask_indptr=mi, window_kv_offsets=woff, block_t=4, block_n=16, dk=32,
                            num_stages=1)
    else:
        k4.nvfp4_verify_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, mx, D ** -0.5, sliding_window_size=window,
                            custom_mask=cm, mask_indptr=mi, window_kv_offsets=woff, num_splits=2, block_n=16, dk=32,
                            num_stages=1)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, sliding_window_size=window,
                              custom_mask=cm, mask_indptr=mi, window_kv_offsets=woff)
    assert (o.float() - ref).abs().max() < 0.03


@pytest.mark.parametrize("window", [-1, 20])
def test_split_extend_token_blocks(window):
    # incremental prefill over a long prefix: more new tokens than one 64-row block (G = 6 -> 8 tokens per block)
    HQ, KVH, D = 12, 2, 64
    q, ke, ve, kp, ks, vp, vs, kd, vd, qo, kvp, idx, mx = _extend_case(HQ, KVH, D, [90, 0, 40], [27, 9, 3], seed=31)
    o = torch.empty_like(q)
    k4.nvfp4_verify_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, mx, D ** -0.5, sliding_window_size=window,
                        num_splits=3, block_n=16, dk=32, num_stages=1)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, sliding_window_size=window)
    assert (o.float() - ref).abs().max() < 0.03


def test_dq_groups_cover_every_request_once():
    import kv4tri_backend as kb
    assert kb._dq_groups([100, 100, 100], 250) == [(0, 2), (2, 3)]
    assert kb._dq_groups([400, 10, 10], 250) == [(0, 1), (1, 3)]
    assert kb._dq_groups([], 250) == [(0, 0)]
    assert kb._dq_groups([5, 0, 7], 250) == [(0, 3)]
