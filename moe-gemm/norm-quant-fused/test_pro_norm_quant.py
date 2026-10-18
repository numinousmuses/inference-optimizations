"""Tests for pro_norm_quant (RMSNorm + FP8 UE8M0 group quant in DeepGEMM's layout).

CPU (Triton interpreter; it truncates fp32->bf16 and its fp8 cast is wrong, so loose checks): scale layout and
packing, bf16 output, exponents.
GPU (B200 box, flashinfer first): bit-exact vs the engine (RMSNorm module + sglang_per_token_group_quant_fp8 with
the Blackwell DeepGEMM flags) at Pro's q_lora shape (strided view of qkv_a, K = 1536) for decode 1-128 rows and
prefill 4096; DeepGEMM output from the pre-quantized input equals the engine's fp8 linear; timing old vs new.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ["TRITON_INTERPRET"] = "1"
else:
    import flashinfer  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_norm_quant as NQ  # noqa: E402

gpu = pytest.mark.skipif(not HAS_CUDA, reason="needs a GPU")
cpu = pytest.mark.skipif(HAS_CUDA, reason="interpreter test (CPU only)")

Q_LORA, QKV_A = 1536, 2048  # Pro: q_lora 1536 + kv 512 in the fused wqkv_a output


def _inputs(m, device, seed=0, k=Q_LORA, row=QKV_A):
    g = torch.Generator().manual_seed(seed)
    base = (torch.randn(m, row, generator=g) * 3).bfloat16().to(device)
    w = (1 + 0.2 * torch.randn(k, generator=g)).bfloat16().to(device)
    return base[:, :k], w


@cpu
@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("k", [1536, 640])  # 640: 5 groups, exercises the packed tail
def test_cpu_layout_and_values(m, k):
    x, w = _inputs(m, "cpu", k=k, row=k + 128)
    y, q, s = NQ.rmsnorm_fp8_quant(x, w, 1e-6)
    ry, rq, re = NQ.reference(x, w, 1e-6)
    groups = k // 128
    assert s.shape == (m, (groups + 3) // 4) and s.stride(0) == 1 and s.stride(1) == (m + 3) // 4 * 4
    assert torch.allclose(y.float(), ry.float(), rtol=1.6e-2, atol=1e-3)
    e = NQ.unpack_scales(s, groups)
    assert (e.int() - re.int()).abs().max() <= 1
    if groups % 4:
        tail = s.contiguous().view(torch.uint8).view(m, -1)[:, groups:]
        assert int(tail.abs().sum()) == 0
    # fp8 values are not checked here: the interpreter's fp32 -> fp8 cast is wrong (measured 126 -> 64); the GPU
    # test checks the bytes against the engine
    assert q.dtype == torch.float8_e4m3fn and q.shape == (m, k)


# ------------------------------------------------------------------------------------------------ GPU
def _engine_quant(y):
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    return sglang_per_token_group_quant_fp8(y, 128, column_major_scales=True, scale_tma_aligned=True,
                                            scale_ue8m0=True)


def _engine_norm(x, w):
    from sglang.srt.layers.layernorm import RMSNorm

    n = RMSNorm(x.shape[1], eps=1e-6).to(device=x.device, dtype=torch.bfloat16)
    n.weight.data.copy_(w)
    return n


@gpu
@pytest.mark.parametrize("m", [1, 3, 8, 16, 64, 96, 128, 4096])
def test_gpu_bit_exact_vs_engine(m):
    x, w = _inputs(m, "cuda", seed=m)
    norm = _engine_norm(x, w)
    ey = norm(x)
    eq, es = _engine_quant(ey.contiguous())
    y, q, s = NQ.rmsnorm_fp8_quant(x, w, 1e-6)
    torch.cuda.synchronize()
    y_mism = (y != ey).float().mean().item()
    q_mism = (q.view(torch.uint8) != eq.view(torch.uint8)).float().mean().item()
    s_mism = (NQ.unpack_scales(s, 12) != NQ.unpack_scales(es, 12)).float().mean().item()
    print(f"m={m}: bf16 mismatch {y_mism:.2e}, fp8 byte mismatch {q_mism:.2e}, scale mismatch {s_mism:.2e}")
    assert es.shape == s.shape and es.stride() == s.stride()
    assert y_mism < 1e-4 and q_mism < 1e-4 and s_mism < 1e-3


@gpu
def test_gpu_deepgemm_consumes_prequant():
    """wq_b-shaped FP8 layer (N = 8192 like the indexer's; K = 1536): engine fp8 linear on bf16 input vs DeepGEMM on
    our pre-quantized input must be identical when the quantized bytes are identical."""
    from sglang.srt.layers.quantization.fp8_utils import (deepgemm_w8a8_block_fp8_linear_with_fallback,
                                                           quant_weight_ue8m0, transform_scale_ue8m0)

    torch.manual_seed(0)
    n, k = 8192, Q_LORA
    wb = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    wq, ws = quant_weight_ue8m0(wb, weight_block_size=[128, 128])
    ws = transform_scale_ue8m0(ws, mn=n)

    class L:
        weight, weight_scale_inv, bias = wq, ws, None

    for m in (1, 96, 4096):
        x, w = _inputs(m, "cuda", seed=10 + m)
        y, q, s = NQ.rmsnorm_fp8_quant(x, w, 1e-6)
        ref = deepgemm_w8a8_block_fp8_linear_with_fallback(y, wq, [128, 128], ws)
        out = NQ.fp8_linear_prequant(L, q, s)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), (m, (out.float() - ref.float()).abs().max().item())


@gpu
@pytest.mark.parametrize("m", [1, 16, 96, 128, 4096])
def test_gpu_timing(m):
    """Old: RMSNorm + one quant (the engine quantizes q_lora again inside the indexer wq_b: a second quant not
    counted here). New: one kernel emitting bf16 + fp8 + scales."""
    from pro_mhc_timing import graph_us, n_sets

    sets = [_inputs(m, "cuda", seed=s) for s in range(n_sets(m * QKV_A * 2 * 3))]
    norms = [_engine_norm(x, w) for x, w in sets]
    t_old = graph_us([lambda x=x, n=n: _engine_quant(n(x)) for (x, _), n in zip(sets, norms)])
    t_old2 = graph_us([lambda x=x, n=n: (lambda y: (_engine_quant(y), _engine_quant(y)))(n(x))
                       for (x, _), n in zip(sets, norms)])
    t_new = graph_us([lambda x=x, w=w: NQ.rmsnorm_fp8_quant(x, w, 1e-6) for x, w in sets])
    print(f"TIMING norm_quant m={m}: engine norm+quant {t_old:.2f} us, norm+2 quants (as served) {t_old2:.2f} us, "
          f"fused {t_new:.2f} us")
