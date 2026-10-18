"""GPU tests (RTX PRO 6000 / SM120) for glmflash_sm120_attn + glmflash_sm120_fi. NOT RUN YET (no GPU as of
2026-09-30); research/rtx-flash/attention.md section 8 is the plan that runs them.

    cd /root/work && python3 -m pytest -q -x operations/kernels/test_glmflash_sm120_gpu.py 2>&1 | tee /root/rtxattn_gpu.log

flashinfer is imported first (it sets TRITON_PTXAS_BLACKWELL_PATH; the engine always has it loaded).
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest
import torch

try:
    import flashinfer  # noqa: F401  (must precede triton compilation, see agents/CONTEXT.md lessons)
except ImportError:
    flashinfer = None

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glmflash_sm120_attn as A  # noqa: E402
import glmflash_sm120_fi as FI  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12, reason="needs an SM120 GPU")
DEV = "cuda"
SCALE = 1.0 / (256 ** 0.5)  # Flash: qk_nope_head_dim 256 -> softmax scale 1/16 (absorbed MLA keeps the model's scale)


def _pool(fmt: int, n: int, seed: int = 0, pow2: bool = False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    lat = torch.randn(n, 512, generator=g, device=DEV) * 0.7
    if fmt == A.FMT_BF16:
        b = lat.to(torch.bfloat16)
        return b.reshape(n, 1, 512), b.float()
    if fmt == A.FMT_FP8_656:
        rows = torch.zeros(n, 1, 656, dtype=torch.uint8, device=DEV)
        loc = torch.arange(n, device=DEV)
        A.nsa656_quant_write(rows, loc, lat.to(torch.bfloat16), pow2=pow2)
        return rows, A.nsa656_dequant_ref(rows.reshape(n, 656))
    import glmflash_kv4
    rows = glmflash_kv4.quantize_rows_ref(lat)
    rows[0] = 0
    return rows.reshape(n, 1, 288), A.fp4_288_dequant_ref(rows)


def _table(t: int, n: int, k: int = 2051, seed: int = 1, short_every: int = 3):
    """kpool-style compact tables: full 2051 rows, and every `short_every`-th row a short context (-1 tail)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = torch.randint(1, n, (t, k), generator=g, device=DEV, dtype=torch.int32)
    for i in range(0, t, short_every):
        v = int(torch.randint(1, k, (1,), generator=g, device=DEV))
        idx[i, v:] = -1
    return idx


def _ref(q, rows_f32, idx):
    out = torch.empty(q.shape, dtype=torch.float32, device=DEV)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=DEV)
    for i in range(0, q.shape[0], 64):  # chunked fp32 reference on the GPU
        o, l_ = A.sparse_mla_ref(q[i:i + 64].float(), rows_f32, idx[i:i + 64], SCALE)
        out[i:i + 64], lse[i:i + 64] = o, l_
    return out, lse


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


# ------------------------------------------------------------------------------------------------ writer


@pytest.mark.parametrize("pow2", [False, True])
def test_nsa656_writer_bitexact_vs_torch(pow2):
    n = 4096
    lat = (torch.randn(1000, 512, device=DEV) * 3).to(torch.bfloat16)
    lat[7, 128:256] = 0
    loc = torch.randperm(n - 1, device=DEV)[:1000] + 1
    loc[5] = 0
    rows = torch.zeros(n, 1, 656, dtype=torch.uint8, device=DEV)
    A.nsa656_quant_write(rows, loc, lat, pow2=pow2)
    ref = A.nsa656_quantize_ref(lat.float(), pow2=pow2)
    r = rows.reshape(n, 656)
    ok = loc > 0
    assert torch.equal(r[loc[ok]], ref[ok])
    assert torch.all(r[0] == 0)
    assert torch.isfinite(A.nsa656_dequant_ref(r)).all()


# ------------------------------------------------------------------------------------------------ Triton kernel


@pytest.mark.parametrize("fmt", [A.FMT_BF16, A.FMT_FP8_656, A.FMT_FP4_288])
@pytest.mark.parametrize("t,h", [(1, 16), (8, 16), (64, 16), (256, 16), (1024, 16), (1, 64), (64, 64), (512, 64)])
def test_triton_sparse_mla(fmt, t, h):
    n = 200_000
    buf, rows = _pool(fmt, n)
    q = (torch.randn(t, h, 512, device=DEV) * 0.1).to(torch.bfloat16)
    idx = _table(t, n)
    out, lse = A.sparse_mla(q, buf, idx, SCALE, fmt)
    ro, rl = _ref(q, rows, idx)
    assert _rel(out, ro) < 8e-3, _rel(out, ro)  # bf16 P and bf16 operands; fp32 accumulate
    assert torch.allclose(lse, rl, atol=2e-2)


def test_triton_q_view_and_tiles():
    n = 50_000
    buf, rows = _pool(A.FMT_FP8_656, n)
    q576 = torch.zeros(16, 16, 576, dtype=torch.bfloat16, device=DEV)
    q576[..., :512] = (torch.randn(16, 16, 512, device=DEV) * 0.1).to(torch.bfloat16)
    idx = _table(16, n)
    ro, _ = _ref(q576[..., :512], rows, idx)
    for bn, nw, ns in ((16, 8, 2), (32, 8, 2), (32, 4, 2), (64, 8, 1)):
        out, _ = A.sparse_mla(q576[..., :512], buf, idx, SCALE, A.FMT_FP8_656, block_n=bn, num_warps=nw,
                              num_stages=ns)
        assert _rel(out, ro) < 8e-3, (bn, nw, ns)


# ------------------------------------------------------------------------------------------------ FlashInfer


@pytest.fixture(scope="module")
def fi_module():
    return FI.build_module()


def _fi_case(t, h, n=200_000, seed=0, pow2=False):
    buf, rows = _pool(A.FMT_FP8_656, n, seed, pow2=pow2)
    q = (torch.randn(t, h, 512, device=DEV) * 0.1).to(torch.bfloat16)
    idx = _table(t, n, seed=seed + 1)
    return buf, rows, q, idx


@pytest.mark.parametrize("t,h", [(1, 16), (16, 16), (64, 16), (65, 16), (512, 16), (4096, 16),
                                 (1, 64), (64, 64), (256, 64), (2048, 64), (8, 32), (128, 32)])
def test_fi_jit_2112(fi_module, t, h):
    buf, rows, q, idx = _fi_case(t, h)
    ws = torch.zeros(384 << 20, dtype=torch.uint8, device=DEV)
    out, lse = FI.FlashInferGlm5Next("jit", workspace=ws, module=fi_module).forward(q, buf, idx, SCALE)
    ro, rl = _ref(q, rows, idx)
    e = _rel(out, ro)
    # FP8 compute mode (Q and P quantized to E4M3 in-kernel): kv4mla measured the TRT-LLM FP8 op at 0.009 on similar rows
    assert e < 3e-2, e
    assert torch.isfinite(out).all()
    ot, _ = A.sparse_mla(q, buf, idx, SCALE, A.FMT_FP8_656)
    assert _rel(out, ot) < 3e-2


@pytest.mark.parametrize("t,h", [(1, 16), (64, 16), (300, 16), (8, 64), (512, 64)])
def test_fi_stock_plus_tail_equals_jit(fi_module, t, h):
    buf, rows, q, idx = _fi_case(t, h, seed=7)
    ws = torch.zeros(384 << 20, dtype=torch.uint8, device=DEV)
    oj, lj = FI.FlashInferGlm5Next("jit", workspace=ws, module=fi_module).forward(q, buf, idx, SCALE)
    os_, ls = FI.FlashInferGlm5Next("stock", workspace=ws).forward(q, buf, idx, SCALE)
    ro, _ = _ref(q, rows, idx)
    assert _rel(os_, ro) < 3e-2 and _rel(oj, ro) < 3e-2
    assert torch.allclose(lj, ls, atol=5e-2)


def test_fi_all_minus_one_rows_are_zero_not_nan(fi_module):
    buf, rows, q, idx = _fi_case(8, 16)
    idx[3] = -1
    out, lse = FI.FlashInferGlm5Next("jit", workspace=None, module=fi_module).forward(q, buf, idx, SCALE)
    assert torch.isfinite(out).all() and torch.all(out[3] == 0)


def test_fi_decode_cuda_graph(fi_module):
    buf, rows, q, idx = _fi_case(32, 16)
    ws = torch.zeros(384 << 20, dtype=torch.uint8, device=DEV)
    fi = FI.FlashInferGlm5Next("jit", workspace=ws, module=fi_module)
    out = torch.empty(32, 16, 512, dtype=torch.bfloat16, device=DEV)
    fi.forward(q, buf, idx, SCALE, out=out)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            fi.forward(q, buf, idx, SCALE, out=out)
    q.copy_((torch.randn_like(q.float()) * 0.1).to(torch.bfloat16))
    g.replay()
    torch.cuda.synchronize()
    ro, _ = _ref(q, rows, idx)
    assert _rel(out, ro) < 3e-2


def test_triton_cuda_graph_split():
    buf, rows = _pool(A.FMT_FP4_288, 100_000)
    q = (torch.randn(4, 16, 512, device=DEV) * 0.1).to(torch.bfloat16)
    idx = _table(4, 100_000)
    ws = A.Workspace(torch.zeros(64 << 20, dtype=torch.uint8, device=DEV))
    out = torch.empty(4, 16, 512, dtype=torch.bfloat16, device=DEV)
    lse = torch.empty(4, 16, dtype=torch.float32, device=DEV)
    A.sparse_mla(q, buf, idx, SCALE, A.FMT_FP4_288, out=out, lse=lse, workspace=ws)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        A.sparse_mla(q, buf, idx, SCALE, A.FMT_FP4_288, out=out, lse=lse, workspace=ws)
    q.copy_((torch.randn_like(q.float()) * 0.1).to(torch.bfloat16))
    g.replay()
    torch.cuda.synchronize()
    ro, _ = _ref(q, rows, idx)
    assert _rel(out, ro) < 8e-3


# ---- pow2 rows -> FlashInfer DSV3_2 kernels (research/rtx-flash/attention.md section 12) ----


@pytest.mark.parametrize("t,h", [(1, 16), (64, 16), (65, 16), (4096, 16), (1, 64), (64, 64), (2048, 64)])
def test_fi_jit_2112_pow2_dsv3_2(fi_module, t, h):
    buf, rows, q, idx = _fi_case(t, h, pow2=True)
    ws = torch.zeros(384 << 20, dtype=torch.uint8, device=DEV)
    fi = FI.FlashInferGlm5Next("jit", workspace=ws, module=fi_module, pow2=True)
    assert fi.model_type == FI.MODEL_TYPE_DSV3_2
    out, lse = fi.forward(q, buf, idx, SCALE)
    ro, rl = _ref(q, rows, idx)
    assert _rel(out, ro) < 3e-2, _rel(out, ro)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("t,h", [(8, 16), (300, 64)])
def test_fi_stock_pow2_plus_tail(t, h):
    buf, rows, q, idx = _fi_case(t, h, seed=3, pow2=True)
    out, _ = FI.FlashInferGlm5Next("stock", pow2=True).forward(q, buf, idx, SCALE)
    ro, _ = _ref(q, rows, idx)
    assert _rel(out, ro) < 3e-2
