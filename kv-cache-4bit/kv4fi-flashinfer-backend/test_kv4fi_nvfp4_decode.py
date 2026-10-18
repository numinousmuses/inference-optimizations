"""GPU tests for kv4fi_nvfp4_decode (run on a Blackwell box: `devbox.py test <box> operations/kernels/test_kv4fi_nvfp4_decode.py`).

Checks, against an fp32 torch reference that decodes the same bytes:
- decode (QL=1) at batch 1 and 8 for the Qwen 3.8 (24q/4kv x 256) and Qwen 3.6 (16q/2kv x 256) shapes, mixed
  lengths up to 40K, both scale-expansion paths;
- verify (QL=4): causal chain without a mask, and a random tree mask packed like FlashInfer's segment_packbits;
- byte format equals FlashInfer's nvfp4_kv_quantize / nvfp4_kv_dequantize (the pool's writer);
- CUDA-graph replay with different lengths and indices than at capture (graph safety).
Imports flashinfer first so Triton uses the same ptxas as the engine (lesson in agents/CONTEXT.md).
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

try:
    import flashinfer  # noqa: F401  (sets TRITON_PTXAS_BLACKWELL_PATH like the engine)
except ImportError:  # local CPU host
    flashinfer = None

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kv4fi_nvfp4_decode as K  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel tests")
DEV = "cuda"
D = 256


def quantize_ref(x: torch.Tensor, gscale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [..., D] -> (fp4 bytes [..., D/2], e4m3 scale bytes [..., D/16]); nearest E2M1, like FlashInfer."""
    xf = x.float()
    blocks = xf.view(*xf.shape[:-1], -1, K.SF_BLOCK)
    amax = blocks.abs().amax(-1, keepdim=True)
    sf = (amax / (6.0 * gscale)).to(torch.float8_e4m3fn)
    sf_f = sf.float()
    scaled = torch.where(sf_f > 0, blocks / (sf_f * gscale), torch.zeros_like(blocks))
    lut = K.E2M1_VALUES[:8].to(x.device)
    idx = (scaled.abs()[..., None] - lut).abs().argmin(-1)  # nearest magnitude code 0..7
    code = idx + 8 * (scaled < 0)
    code = code.flatten(-2).to(torch.uint8)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed, sf.view(torch.uint8).flatten(-2)


def make_pool(total: int, hk: int, gk: float, gv: float, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = torch.randn((total, hk, D), device=DEV, generator=g, dtype=torch.float32) * 3
    v = torch.randn((total, hk, D), device=DEV, generator=g, dtype=torch.float32)
    k_fp4, k_sf = quantize_ref(k.bfloat16(), gk)
    v_fp4, v_sf = quantize_ref(v.bfloat16(), gv)
    return k_fp4, k_sf, v_fp4, v_sf


def make_batch(lens: list[int], total_slots: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    kv_indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
    kv_indices = torch.randperm(total_slots, device=DEV, generator=g)[: int(kv_indptr[-1])].to(torch.int32)
    return kv_indptr, kv_indices


def check(out, ref, tol=2.5e-2):
    out, ref = out.float(), ref.float()
    err = (out - ref).norm() / ref.norm().clamp_min(1e-6)
    assert torch.isfinite(out).all()
    assert err < tol, f"rel err {err:.4f}"
    return float(err)


@pytest.mark.parametrize("hq,hk", [(24, 4), (16, 2)])
@pytest.mark.parametrize("lens", [[777], [1, 5, 300, 4096, 129, 2048, 33, 1000]])
@pytest.mark.parametrize("use_cvt", [True, False])
def test_decode_matches_reference(hq, hk, lens, use_cvt):
    total = 8192
    gk, gv = 1.7, 0.9
    k_fp4, k_sf, v_fp4, v_sf = make_pool(total, hk, gk, gv, 1)
    kv_indptr, kv_indices = make_batch(lens, total, 2)
    q = torch.randn((len(lens), hq, D), device=DEV, dtype=torch.bfloat16)
    sm = D**-0.5
    if use_cvt and not K.hardware_e2m1_cvt_available():
        pytest.skip("no hardware e2m1 cvt on this GPU")
    out = K.nvfp4_attention(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, gk, gv, sm, use_cvt=use_cvt)
    ref = K.nvfp4_attention_ref(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, gk, gv, sm)
    check(out, ref)


def test_decode_long_context_many_splits():
    hq, hk = 24, 4
    total = 45000
    k_fp4, k_sf, v_fp4, v_sf = make_pool(total, hk, 1.0, 1.0, 3)
    kv_indptr, kv_indices = make_batch([40000, 3], total, 4)
    q = torch.randn((2, hq, D), device=DEV, dtype=torch.bfloat16)
    for splits in (1, 4, 32):
        out = K.nvfp4_attention(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.0, 1.0, D**-0.5,
                                num_kv_splits=splits)
        ref = K.nvfp4_attention_ref(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.0, 1.0, D**-0.5)
        check(out, ref)


@pytest.mark.parametrize("hq,hk", [(24, 4), (16, 2)])
def test_verify_causal_chain(hq, hk):
    ql, lens = 4, [12, 700, 64, 2500, 4, 4, 900, 100]  # kv_len includes the ql draft tokens
    total = 8192
    k_fp4, k_sf, v_fp4, v_sf = make_pool(total, hk, 1.3, 1.1, 5)
    kv_indptr, kv_indices = make_batch(lens, total, 6)
    q = torch.randn((len(lens) * ql, hq, D), device=DEV, dtype=torch.bfloat16)
    out = K.nvfp4_attention(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.3, 1.1, D**-0.5, ql=ql)
    ref = K.nvfp4_attention_ref(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.3, 1.1, D**-0.5, ql=ql)
    check(out, ref)


def test_verify_packed_tree_mask():
    hq, hk, ql = 24, 4, 5
    lens = [37, 1500, 8, 611]
    total = 4096
    k_fp4, k_sf, v_fp4, v_sf = make_pool(total, hk, 1.0, 1.0, 7)
    kv_indptr, kv_indices = make_batch(lens, total, 8)
    g = torch.Generator(device=DEV).manual_seed(9)
    allowed = []
    for L in lens:
        m = torch.ones((ql, L), dtype=torch.bool, device=DEV)
        # random tree: each draft row drops a random subset of the other draft tokens; prefix always visible
        m[:, L - ql :] = torch.rand((ql, ql), device=DEV, generator=g) > 0.5
        m[torch.arange(ql), L - ql + torch.arange(ql)] = True
        allowed.append(m)
    mask, mask_indptr = K.pack_mask_ref(allowed)
    q = torch.randn((len(lens) * ql, hq, D), device=DEV, dtype=torch.bfloat16)
    out = K.nvfp4_attention(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.0, 1.0, D**-0.5, ql=ql,
                            mask=mask, mask_indptr=mask_indptr)
    ref = K.nvfp4_attention_ref(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.0, 1.0, D**-0.5, ql=ql,
                                mask=mask, mask_indptr=mask_indptr)
    check(out, ref)
    # the unpacked mask round-trips
    for b, L in enumerate(lens):
        assert torch.equal(K.unpack_mask_ref(mask, mask_indptr, b, ql, L), allowed[b])


@pytest.mark.skipif(flashinfer is None, reason="needs flashinfer (box)")
def test_byte_format_matches_flashinfer():
    from flashinfer import nvfp4_kv_dequantize, nvfp4_kv_quantize
    cap = torch.cuda.get_device_capability()
    if cap[0] not in (10, 12):
        pytest.skip("nvfp4_kv_quantize needs SM100/SM120")
    x = (torch.randn((512, D), device=DEV) * 4).bfloat16()
    gs = torch.tensor([1.5], device=DEV)
    fp4, sf = nvfp4_kv_quantize(x, gs)
    fi = nvfp4_kv_dequantize(fp4.view(torch.uint8), sf.view(torch.uint8), gs, output_dtype=torch.bfloat16)
    ours = K.dequant_nvfp4_ref(fp4.view(torch.uint8), sf.view(torch.uint8), 1.5).bfloat16()
    assert torch.equal(fi, ours), (fi - ours).abs().max()
    # and the kernel on FlashInfer-produced bytes agrees with the reference
    hk = 4
    k_fp4 = fp4.view(torch.uint8).view(-1, hk, D // 2).contiguous()
    k_sf = sf.view(torch.uint8).view(-1, hk, D // 16).contiguous()
    n = k_fp4.shape[0]
    kv_indptr = torch.tensor([0, n], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(n, dtype=torch.int32, device=DEV)
    q = torch.randn((1, 24, D), device=DEV, dtype=torch.bfloat16)
    out = K.nvfp4_attention(q, k_fp4, k_sf, k_fp4, k_sf, kv_indptr, kv_indices, 1.5, 1.5, D**-0.5)
    ref = K.nvfp4_attention_ref(q, k_fp4, k_sf, k_fp4, k_sf, kv_indptr, kv_indices, 1.5, 1.5, D**-0.5)
    check(out, ref)


def test_cuda_graph_replay_with_new_lengths():
    hq, hk, bs = 24, 4, 8
    total = 20000
    k_fp4, k_sf, v_fp4, v_sf = make_pool(total, hk, 1.0, 1.0, 11)
    maxtok = 17000
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEV)
    kv_indices = torch.zeros(maxtok, dtype=torch.int32, device=DEV)
    q = torch.randn((bs, hq, D), device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    splits = K.choose_num_kv_splits(bs, hk)

    def fill(lens, seed):
        ip, ix = make_batch(lens, total, seed)
        kv_indptr.copy_(ip)
        kv_indices[: ix.numel()].copy_(ix)

    fill([1] * bs, 12)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):  # warm up / compile
            K.nvfp4_attention(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.0, 1.0, D**-0.5,
                              num_kv_splits=splits, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        K.nvfp4_attention(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.0, 1.0, D**-0.5,
                          num_kv_splits=splits, out=out)
    for lens, seed in (([3000, 1, 900, 12000, 7, 64, 33, 5], 13), ([2] * bs, 14), ([1900] * bs, 15)):
        fill(lens, seed)
        q.copy_(torch.randn_like(q))
        graph.replay()
        torch.cuda.synchronize()
        ref = K.nvfp4_attention_ref(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.0, 1.0, D**-0.5)
        check(out, ref)


def test_choose_num_kv_splits_is_shape_only():
    a = K.choose_num_kv_splits(1, 4, sm_count=188)
    b = K.choose_num_kv_splits(8, 4, sm_count=188)
    c = K.choose_num_kv_splits(64, 4, sm_count=188)
    assert a >= b >= c >= 1 and a <= 128
    assert all((x & (x - 1)) == 0 for x in (a, b, c))


def test_ragged_qo_noncausal_with_lse():
    """Eager draft-extend paged part: variable query counts per request, full visibility, base-2 lse."""
    hq, hk = 24, 4
    lens = [300, 17, 1200, 5]
    qlens = [3, 1, 4, 2]
    total = 4096
    k_fp4, k_sf, v_fp4, v_sf = make_pool(total, hk, 1.2, 0.8, 21)
    kv_indptr, kv_indices = make_batch(lens, total, 22)
    qo_indptr = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
    q = torch.randn((sum(qlens), hq, D), device=DEV, dtype=torch.bfloat16)
    out, lse = K.nvfp4_attention(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.2, 0.8, D**-0.5, ql=4,
                                 qo_indptr=qo_indptr, causal=False, return_lse=True)
    ref, lse_ref = K.nvfp4_attention_ref(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.2, 0.8, D**-0.5,
                                         ql=4, qo_indptr=qo_indptr, causal=False, return_lse=True)
    check(out, ref)
    assert torch.allclose(lse, lse_ref, atol=2e-2, rtol=1e-3), (lse - lse_ref).abs().max()
    # causal ragged too
    out, lse = K.nvfp4_attention(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.2, 0.8, D**-0.5, ql=4,
                                 qo_indptr=qo_indptr, causal=True, return_lse=True)
    ref, lse_ref = K.nvfp4_attention_ref(q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.2, 0.8, D**-0.5,
                                         ql=4, qo_indptr=qo_indptr, causal=True, return_lse=True)
    check(out, ref)
    assert torch.allclose(lse, lse_ref, atol=2e-2, rtol=1e-3), (lse - lse_ref).abs().max()


@pytest.mark.skipif(flashinfer is None, reason="needs flashinfer (box)")
def test_lse_convention_matches_flashinfer():
    """FlashInfer's forward_return_lse is base 2; merge_state in the ragged prefill path expects that base."""
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    hq, hk, L = 24, 4, 333
    k = torch.randn((L, 1, hk, D), device=DEV, dtype=torch.bfloat16)
    v = torch.randn((L, 1, hk, D), device=DEV, dtype=torch.bfloat16)
    q = torch.randn((2, hq, D), device=DEV, dtype=torch.bfloat16)
    ws = torch.empty(64 << 20, dtype=torch.uint8, device=DEV)
    w = BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    qo_indptr = torch.tensor([0, 2], dtype=torch.int32, device=DEV)
    kv_indptr = torch.tensor([0, L], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(L, dtype=torch.int32, device=DEV)
    last = torch.ones(1, dtype=torch.int32, device=DEV)
    w.plan(qo_indptr, kv_indptr, kv_indices, last, hq, hk, D, 1, causal=False, q_data_type=torch.bfloat16,
           kv_data_type=torch.bfloat16)
    _, lse_fi = w.run_return_lse(q, (k, v))
    kk = k.view(L, hk, D).float().repeat_interleave(hq // hk, dim=1)
    s = torch.einsum("thd,lhd->thl", q.float(), kk) * D**-0.5
    lse2 = torch.logsumexp(s, -1) / math.log(2.0)
    assert torch.allclose(lse_fi.float(), lse2, atol=3e-2, rtol=1e-3), (lse_fi.float() - lse2).abs().max()
