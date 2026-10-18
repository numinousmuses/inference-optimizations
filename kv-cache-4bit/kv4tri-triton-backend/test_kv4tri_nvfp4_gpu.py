"""Box test (RTX PRO 6000 / any SM100-SM120 with SGLang v0.5.20): the kv4tri NVFP4 kernels read SGLang's own NVFP4
pools correctly. K/V are written through the pool's set_kv_buffer (FlashInfer nvfp4_kv_quantize, the production
store path) and read by the kernels with no host synchronization in between; references come from SGLang's own
dequantizer (NVFP4KVQuantizeUtil.dequantize).

    cd /root/work && python3 -m pytest -q operations/kernels/test_kv4tri_nvfp4_gpu.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

flashinfer = pytest.importorskip("flashinfer")  # engine ptxas (CONTEXT lessons)
import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv4tri_nvfp4_attention as k4  # noqa: E402

DEV = "cuda"


def _mha_pool(size, H, D, layers=1):
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import get_kv_cache_quant_method
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    qm = get_kv_cache_quant_method("nvfp4", num_layers=layers, device=DEV)
    return MHATokenToKVPool(size=size, page_size=1, dtype=torch.float4_e2m1fn_x2, head_num=H, head_dim=D,
                            layer_num=layers, device=DEV, enable_memory_saver=False, quant_method=qm)


def _sgl_dequant(pool, layer_id, gs=1.0):
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    kp, vp, ks, vs = pool.get_raw_kv_buffer(layer_id)
    g = torch.tensor([gs], dtype=torch.float32, device=DEV)
    # bf16 is exact here: an E2M1 value times an E4M3 scale needs at most 5 significant bits (global scales in the
    # tests are powers of two); FlashInfer's dequantizer has no float32 output
    kd = NVFP4KVQuantizeUtil.dequantize(kp, ks, g, dtype=torch.bfloat16).float()
    vd = NVFP4KVQuantizeUtil.dequantize(vp, vs, g, dtype=torch.bfloat16).float()
    return kp, vp, ks, vs, kd, vd


def _fill(pool, slots, H, D, layer_id=0, k_scale=None, seed=0):
    torch.manual_seed(seed)
    k = torch.randn(slots, H, D, device=DEV, dtype=torch.bfloat16) * 2
    v = torch.randn(slots, H, D, device=DEV, dtype=torch.bfloat16) * 0.5
    loc = torch.arange(1, slots + 1, device=DEV)
    layer = types.SimpleNamespace(layer_id=layer_id)
    pool.set_kv_buffer(layer, loc, k, v, k_scale, k_scale)
    return k, v


def test_reference_dequantizer_matches_sglang():
    pool = _mha_pool(512, 4, 256)
    _fill(pool, 400, 4, 256)
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0)
    assert torch.equal(k4.nvfp4_dequantize_ref(kp, ks.view(torch.uint8)), kd)
    assert torch.equal(k4.nvfp4_dequantize_ref(vp, vs.view(torch.uint8)), vd)


@pytest.mark.parametrize("HQ,KVH,D", [(24, 4, 256), (16, 2, 256), (16, 8, 256), (16, 2, 512)])
def test_decode_on_sglang_pool(HQ, KVH, D):
    lens = [3000, 1, 777, 64]
    slots = sum(lens)
    pool = _mha_pool(slots + 64, KVH, D)
    _fill(pool, slots, KVH, D)
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0)
    B, MS = len(lens), 8
    q = torch.randn(B, HQ, D, device=DEV, dtype=torch.bfloat16)
    kv_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), dtype=torch.int32, device=DEV)
    kv_indices = (torch.randperm(slots, device=DEV) + 1).to(torch.int64)
    att = torch.empty(B, HQ, MS, D, device=DEV)
    lse = torch.empty(B, HQ, MS, device=DEV)
    nsp = torch.tensor([8, 1, 5, 2], dtype=torch.int32, device=DEV)
    o = torch.empty(B, HQ, D, device=DEV, dtype=torch.bfloat16)
    k4.nvfp4_decode_fwd(q, kp, vp, ks, vs, o, kv_indptr, kv_indices, att, lse, nsp, MS, D ** -0.5)
    ref = k4.decode_reference(q, kd, vd, kv_indptr, kv_indices, D ** -0.5)
    assert (o.float() - ref).abs().max() < 0.02


def test_global_scale_through_the_store_path():
    # a layer k_scale (e.g. a checkpoint's calibrated FP8 KV scale) becomes the NVFP4 global scale at store time;
    # the kernel must apply the same value
    pool = _mha_pool(2048, 2, 256)
    _fill(pool, 1500, 2, 256, k_scale=torch.tensor([0.125], device=DEV))
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0, gs=0.125)
    q = torch.randn(1, 16, 256, device=DEV, dtype=torch.bfloat16)
    kv_indptr = torch.tensor([0, 1500], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(1, 1501, device=DEV)
    att = torch.empty(1, 16, 4, 256, device=DEV)
    lse = torch.empty(1, 16, 4, device=DEV)
    o = torch.empty(1, 16, 256, device=DEV, dtype=torch.bfloat16)
    k4.nvfp4_decode_fwd(q, kp, vp, ks, vs, o, kv_indptr, kv_indices, att, lse,
                        torch.tensor([4], dtype=torch.int32, device=DEV), 4, 256 ** -0.5, 0.125, 0.125)
    ref = k4.decode_reference(q, kd, vd, kv_indptr, kv_indices, 256 ** -0.5)
    assert (o.float() - ref).abs().max() < 0.02


@pytest.mark.parametrize("HQ,KVH,D,window", [(24, 4, 256, -1), (16, 2, 256, -1), (16, 8, 256, 1023),
                                             (16, 2, 512, -1)])
def test_extend_on_sglang_pool(HQ, KVH, D, window):
    prefixes, news = [2500, 0, 300], [200, 64, 1]
    slots = sum(prefixes)
    pool = _mha_pool(slots + 64, KVH, D)
    _fill(pool, slots, KVH, D, seed=2)
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0)
    T = sum(news)
    q = torch.randn(T, HQ, D, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16) * 2
    ve = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16) * 0.5
    qo = torch.tensor([0] + list(torch.cumsum(torch.tensor(news), 0)), dtype=torch.int32, device=DEV)
    kvp = torch.tensor([0] + list(torch.cumsum(torch.tensor(prefixes), 0)), dtype=torch.int32, device=DEV)
    idx = (torch.randperm(slots, device=DEV) + 1).to(torch.int64)
    o = torch.empty_like(q)
    k4.nvfp4_extend_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, max(news), D ** -0.5,
                        sliding_window_size=window)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, sliding_window_size=window)
    assert (o.float() - ref).abs().max() < 0.03


def test_extend_custom_mask_verify_shape():
    # EAGLE target verify (Qwen): 4 draft tokens per request over long prefixes, chain mask
    HQ, KVH, D = 24, 4, 256
    prefixes, news = [4000, 900], [4, 4]
    slots = sum(prefixes)
    pool = _mha_pool(slots + 64, KVH, D)
    _fill(pool, slots, KVH, D, seed=3)
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0)
    T = sum(news)
    q = torch.randn(T, HQ, D, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16)
    qo = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEV)
    kvp = torch.tensor([0, prefixes[0], slots], dtype=torch.int32, device=DEV)
    idx = (torch.arange(slots, device=DEV) + 1).to(torch.int64)
    masks, mptr = [], [0]
    for P in prefixes:
        m = torch.ones(4, P + 4, dtype=torch.bool, device=DEV)
        m[:, P:] = torch.tril(torch.ones(4, 4, dtype=torch.bool, device=DEV))
        masks.append(m.flatten())
        mptr.append(mptr[-1] + m.numel())
    cm = torch.cat(masks)
    mi = torch.tensor(mptr[:-1], dtype=torch.int64, device=DEV)
    o = torch.empty_like(q)
    k4.nvfp4_extend_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, 4, D ** -0.5, custom_mask=cm, mask_indptr=mi)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, custom_mask=cm, mask_indptr=mi)
    assert (o.float() - ref).abs().max() < 0.03


def test_decode_cuda_graph_capture_and_replay():
    HQ, KVH, D = 24, 4, 256
    lens = [1000, 2000]
    slots = sum(lens)
    pool = _mha_pool(slots + 64, KVH, D)
    _fill(pool, slots, KVH, D, seed=5)
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0)
    q = torch.randn(2, HQ, D, device=DEV, dtype=torch.bfloat16)
    kv_indptr = torch.tensor([0, 1000, 3000], dtype=torch.int32, device=DEV)
    kv_indices = (torch.arange(slots, device=DEV) + 1).to(torch.int64)
    att = torch.empty(2, HQ, 8, D, device=DEV)
    lse = torch.empty(2, HQ, 8, device=DEV)
    nsp = torch.tensor([8, 8], dtype=torch.int32, device=DEV)
    o = torch.empty(2, HQ, D, device=DEV, dtype=torch.bfloat16)
    fn = lambda: k4.nvfp4_decode_fwd(q, kp, vp, ks, vs, o, kv_indptr, kv_indices, att, lse, nsp, 8, D ** -0.5)  # noqa: E731
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    q.copy_(torch.randn_like(q))
    g.replay()
    ref = k4.decode_reference(q, kd, vd, kv_indptr, kv_indices, D ** -0.5)
    assert (o.float() - ref).abs().max() < 0.02


@pytest.mark.parametrize("HQ,KVH,D,qlen", [(24, 4, 256, 4), (16, 2, 256, 5), (16, 8, 256, 1), (16, 2, 512, 2)])
def test_verify_on_sglang_pool(HQ, KVH, D, qlen):
    prefixes = [3000, 0, 777, 64]
    slots = sum(prefixes)
    pool = _mha_pool(slots + 64, KVH, D)
    _fill(pool, slots, KVH, D, seed=8)
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0)
    B = len(prefixes)
    T = B * qlen
    q = torch.randn(T, HQ, D, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16) * 2
    ve = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16) * 0.5
    qo = torch.arange(0, (B + 1) * qlen, qlen, dtype=torch.int32, device=DEV)
    kvp = torch.tensor([0] + list(torch.cumsum(torch.tensor(prefixes), 0)), dtype=torch.int32, device=DEV)
    idx = (torch.randperm(slots, device=DEV) + 1).to(torch.int64)
    masks, mptr = [], [0]
    for P in prefixes:
        m = torch.ones(qlen, P + qlen, dtype=torch.bool, device=DEV)
        m[:, P:] = torch.tril(torch.ones(qlen, qlen, dtype=torch.bool, device=DEV))
        masks.append(m.flatten())
        mptr.append(mptr[-1] + m.numel())
    cm = torch.cat(masks)
    mi = torch.tensor(mptr[:-1], dtype=torch.int64, device=DEV)
    o = torch.empty_like(q)
    k4.nvfp4_verify_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, qlen, D ** -0.5, custom_mask=cm, mask_indptr=mi)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, custom_mask=cm, mask_indptr=mi)
    assert (o.float() - ref).abs().max() < 0.03


def test_verify_cuda_graph_replay():
    HQ, KVH, D, qlen = 16, 2, 256, 5
    prefixes = [2000, 900]
    slots = sum(prefixes)
    pool = _mha_pool(slots + 64, KVH, D)
    _fill(pool, slots, KVH, D, seed=9)
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0)
    T = 2 * qlen
    q = torch.randn(T, HQ, D, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16)
    qo = torch.tensor([0, qlen, 2 * qlen], dtype=torch.int32, device=DEV)
    kvp = torch.tensor([0, 2000, slots], dtype=torch.int32, device=DEV)
    idx = (torch.arange(slots, device=DEV) + 1).to(torch.int64)
    o = torch.empty_like(q)
    fn = lambda: k4.nvfp4_verify_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, qlen, D ** -0.5)  # noqa: E731
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    q.copy_(torch.randn_like(q))
    g.replay()
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5)
    assert (o.float() - ref).abs().max() < 0.03


def test_gather_dequant_matches_sglang():
    pool = _mha_pool(4096, 4, 256)
    _fill(pool, 3000, 4, 256, seed=10)
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0)
    idx = (torch.randperm(3000, device=DEV) + 1)[:2500]
    out = torch.empty(2500, 4, 256, device=DEV, dtype=torch.bfloat16)
    k4.nvfp4_gather_dequant(kp, ks, idx, out)
    assert torch.equal(out.float(), kd[idx])


@pytest.mark.parametrize("fn", ["extend", "verify"])
@pytest.mark.parametrize("window", [-1, 1023])
def test_image_mask_window_offsets_int64(fn, window):
    """Gemma 4 image prefill on the Triton backend always carries a custom mask over the full sequence; on sliding
    layers window_kv_offsets place the windowed prefix. int64 indptrs as SGLang passes them (this combination hit a
    Triton compile error in the verify kernel at the first gate boot)."""
    HQ, KVH, D = 16, 8, 256
    prefixes_full, news = [3000, 200, 0], [24, 6, 5]
    pw = [min(p, window) if window > 0 else p for p in prefixes_full]
    offs = torch.tensor([pf - p for pf, p in zip(prefixes_full, pw)], dtype=torch.int32, device=DEV)
    masks, mptr = [], [0]
    for pf, Tn in zip(prefixes_full, news):
        m = torch.ones(Tn, pf + Tn, dtype=torch.bool, device=DEV).tril(diagonal=pf)
        m[1:4, pf + 1:pf + 4] = True
        masks.append(m.flatten())
        mptr.append(mptr[-1] + m.numel())
    cm = torch.cat(masks)
    mi = torch.tensor(mptr[:-1], dtype=torch.int64, device=DEV)
    slots = sum(pw)
    pool = _mha_pool(slots + 64, KVH, D)
    _fill(pool, slots, KVH, D, seed=11)
    kp, vp, ks, vs, kd, vd = _sgl_dequant(pool, 0)
    T = sum(news)
    q = torch.randn(T, HQ, D, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(T, KVH, D, device=DEV, dtype=torch.bfloat16)
    qo = torch.tensor([0] + list(torch.cumsum(torch.tensor(news), 0)), dtype=torch.int64, device=DEV)
    kvp = torch.tensor([0] + list(torch.cumsum(torch.tensor(pw), 0)), dtype=torch.int64, device=DEV)
    idx = (torch.randperm(slots, device=DEV) + 1).to(torch.int64)
    woff = offs if window > 0 else None
    o = torch.empty_like(q)
    if fn == "extend":
        k4.nvfp4_extend_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, max(news), D ** -0.5,
                            sliding_window_size=window, custom_mask=cm, mask_indptr=mi, window_kv_offsets=woff)
    else:
        assert max(news) <= k4.verify_max_len(HQ // KVH)  # G = 2 -> up to 32 tokens
        k4.nvfp4_verify_fwd(q, ke, ve, o, kp, vp, ks, vs, qo, kvp, idx, max(news), D ** -0.5,
                            sliding_window_size=window, custom_mask=cm, mask_indptr=mi, window_kv_offsets=woff)
    ref = k4.extend_reference(q, ke, ve, kd, vd, qo, kvp, idx, D ** -0.5, sliding_window_size=window,
                              custom_mask=cm, mask_indptr=mi, window_kv_offsets=woff)
    assert (o.float() - ref).abs().max() < 0.03
