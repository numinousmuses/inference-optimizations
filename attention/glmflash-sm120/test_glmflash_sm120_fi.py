"""CPU tests for glmflash_sm120_fi: source patches against FlashInfer 0.6.18's csrc (vendored fixture) and the call
plumbing with the FlashInfer module replaced by a torch emulation of the GLM_NSA contract.

    TRITON_INTERPRET=1 python3 -m pytest -q operations/kernels/test_glmflash_sm120_fi.py
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest
import torch

import glmflash_sm120_attn as A
import glmflash_sm120_fi as FI

CSRC = HERE / "testdata" / "flashinfer_0_6_18_csrc"


def test_fixture_is_the_pinned_wheel():
    for line in (CSRC / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((CSRC / name).read_bytes()).hexdigest() == digest, name


def _balanced(text: str) -> bool:
    body = re.sub(r"//[^\n]*", "", text)
    return body.count("{") == body.count("}") and body.count("(") == body.count(")")


@pytest.mark.parametrize("trim", [True, False])
def test_patched_sources(trim):
    src = FI.patched_sources(CSRC, trim=trim)
    assert set(src) == set(FI.SOURCES)
    for n, t in src.items():
        assert _balanced(t), n
    dec = src["sparse_mla_sm120_decode_dsv3_2.cu"]
    pre = src["sparse_mla_sm120_prefill.cu"]
    for h in FI.HEADS:
        assert (f"DSV3_2_DISPATCH_MT(ModelType::GLM_NSA, {h}, 2112)" in dec) or (f"DSV3_2_DISPATCH({h}, 2112)" in dec)
    # the widened prefill template is the stock one with TK for 2048 and is reached for topk == 2112
    assert "inline bool dispatch_v32_tk(" in pre and "dispatch_v32_tk<MT, 2112>(" in pre
    tk = pre[pre.index("inline bool dispatch_v32_tk("):pre.index("inline bool dispatch_v32(")]
    assert "2048" not in tk and tk.count("TK") == 4  # the 4 literal sites
    assert "template <ModelType MT, int TK>\ninline bool dispatch_v32_tk(" in pre
    assert "launch_prefill_sg<MT, ComputeMode::FP8, 16, TK, 64>" in tk
    assert "launch_prefill_mg<MT, ComputeMode::FP8, NH, TK, 64>" in tk
    dsv4 = src["sparse_mla_sm120_decode_dsv4.cu"]
    if trim:
        assert "DSV4_DISPATCH(" not in dsv4.split("#define DSV4_DISPATCH")[1].split("#undef DSV4_DISPATCH")[0] \
            .split("\\\n  }\n", 1)[1]
        assert "dispatch_dsv4_single" not in pre and "dispatch_dsv4_dual" not in pre
        # trimmed = the two 656-B row types (DSV3_2 pow2 + GLM_NSA arbitrary) only, no DSv4
        assert "return dispatch_v32<ModelType::DSV3_2>(" in pre and "DSV4" not in pre.split("bool sparse_mla_prefill_dispatch(")[1]
    else:
        assert "DSV4_DISPATCH(128, 1024)" in dsv4 and "dispatch_dsv4_single(" in pre
        assert "DSV3_2_DISPATCH(128, 2048)" in dec
    # the binding still exports the three entry points the runtime calls
    b = src["sparse_mla_sm120_jit_binding.cu"]
    for f in ("sparse_mla_sm120_paged_attention", "sparse_mla_sm120_decode_dsv3_2"):
        assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({f}," in b


def test_patch_refuses_unknown_source():
    text = (CSRC / "sparse_mla_sm120_decode_dsv3_2.cu").read_text().replace("  DSV3_2_DISPATCH(128, 2048)\n", "")
    with pytest.raises(FI.PatchError):
        FI.patch_decode_dsv3_2(text, trim=True)


# ------------------------------------------------------------------------------------------------ plumbing


def _emulate(q576, kv3, idx, out, lse, sm_scale, width):
    """Torch emulation of the GLM_NSA contract: 656-byte rows, q . [nope | rope], V = nope, idx < 0 masked."""
    assert q576.shape[-1] == 576 and q576.is_contiguous() and idx.shape[1] == width and idx.dtype == torch.int32
    rows = kv3.reshape(-1, 656)
    nope = A.nsa656_dequant_ref(rows)
    rope = rows[:, 528:].contiguous().view(torch.bfloat16).float()
    kfull = torch.cat([nope, rope], dim=-1)
    t, h, _ = q576.shape
    for i in range(t):
        sel = idx[i].long()
        sel = sel[sel >= 0]
        if sel.numel() == 0:
            out[i] = 0
            lse[i] = A.LSE_EMPTY
            continue
        s = (q576[i].float() @ kfull[sel].T) * sm_scale * A.LOG2E
        m = s.amax(-1, keepdim=True)
        p = torch.exp2(s - m)
        l_ = p.sum(-1, keepdim=True)
        out[i] = ((p / l_) @ nope[sel]).to(out.dtype)
        lse[i] = (m + torch.log2(l_)).squeeze(-1)


class FakeModule:
    def __init__(self):
        self.calls = []
        self.model_types = []

    def sparse_mla_sm120_decode_dsv3_2(self, q, kv, idx, mid_out, mid_lse, out, lse, splits, sm_scale, topk_length,
                                       attn_sink, model_type, cpb):
        assert model_type in (FI.MODEL_TYPE_GLM_NSA, FI.MODEL_TYPE_DSV3_2) and splits == 33 and cpb == -1
        assert q.shape[0] <= 64
        self.model_types.append(model_type)
        assert mid_out.shape == (q.shape[0], q.shape[1], 33, 512) and mid_lse.shape == (q.shape[0], q.shape[1], 33)
        self.calls.append("decode")
        _emulate(q, kv, idx, out, lse, sm_scale, 2112)

    def sparse_mla_sm120_paged_attention(self, q, kv, idx, out, lse, sm_scale, model_type, *rest):
        assert model_type in (FI.MODEL_TYPE_GLM_NSA, FI.MODEL_TYPE_DSV3_2) and all(r is None for r in rest)
        assert q.shape[0] > 64
        self.model_types.append(model_type)
        self.calls.append("prefill")
        _emulate(q, kv, idx, out, lse, sm_scale, 2112)


def fake_stock(q, kv, idx, out, lse, sm_scale, *, kv_scale_format, mid_out=None, mid_lse=None):
    assert kv_scale_format in ("arbitrary_fp32", "pow2_fp32") and idx.shape[1] == 2048
    assert (mid_out is not None) == (q.shape[0] <= 64)
    _emulate(q, kv, idx, out, lse, sm_scale, 2048)


def _setup(t, seed=0, n=2304, h=16, pow2=False):
    g = torch.Generator().manual_seed(seed)
    lat = torch.randn(n, 512, generator=g)
    rows = A.nsa656_quantize_ref(lat, pow2=pow2)
    rows[0] = 0
    q = (torch.randn(t, h, 512, generator=g) * 0.1).to(torch.bfloat16)
    idx = torch.full((t, 2051), -1, dtype=torch.int32)
    for i in range(t):
        v = [2051, 2049, 700, 0][i % 4]
        idx[i, :v] = (torch.randperm(n - 1, generator=g)[:v] + 1).to(torch.int32)
    return rows.reshape(n, 1, 656), q, idx


@pytest.mark.parametrize("t", [3, 66])
def test_jit_plumbing_matches_reference(t):
    kv, q, idx = _setup(t)
    mod = FakeModule()
    fi = FI.FlashInferGlm5Next("jit", workspace=torch.zeros(64 << 20, dtype=torch.uint8), module=mod)
    out, lse = fi.forward(q[:, :, :512], kv, idx, 0.0625)
    assert mod.calls == (["decode"] if t <= 64 else ["prefill"])
    ro, rl = A.sparse_mla_ref(q.float(), A.nsa656_dequant_ref(kv.reshape(-1, 656)), idx, 0.0625)
    assert (out.float() - ro).norm() / ro.norm() < 1e-2
    assert torch.allclose(lse, rl, atol=1e-3)


def test_stock_fallback_main_plus_tail_merge():
    kv, q, idx = _setup(4, seed=2)
    fi = FI.FlashInferGlm5Next("stock", stock_fn=fake_stock)
    out, lse = fi.forward(q, kv, idx, 0.0625)
    ro, rl = A.sparse_mla_ref(q.float(), A.nsa656_dequant_ref(kv.reshape(-1, 656)), idx, 0.0625)
    assert (out.float() - ro).norm() / ro.norm() < 1e-2
    assert torch.allclose(lse, rl, atol=1e-3)
    assert torch.all(lse[3] == A.LSE_EMPTY) and torch.all(out[3] == 0)


def test_zero_rope_row_is_exact_for_the_rope_mma():
    """The q rope half is zero and the row rope bytes are zero: the 576-wide score equals the 512-wide one."""
    kv, q, idx = _setup(2, seed=4)
    rows = kv.reshape(-1, 656)
    assert torch.all(rows[:, 528:] == 0)
    q576 = FI.pad_q576(q)
    assert q576.shape == (2, 16, 576) and torch.all(q576[..., 512:] == 0) and q576.is_contiguous()


def test_views_and_scratch():
    kv = torch.zeros(128, 1, 656, dtype=torch.uint8)
    v = FI.kv_pages_view(kv)
    assert v.shape == (2, 64, 656) and v.data_ptr() == kv.data_ptr()
    with pytest.raises(AssertionError):
        FI.kv_pages_view(torch.zeros(100, 1, 656, dtype=torch.uint8))
    ws = torch.zeros(1 << 22, dtype=torch.uint8)
    mo, ml = FI.decode_scratch(ws, 4, 16, 33, "cpu")
    assert mo.data_ptr() == ws.data_ptr() and ml.data_ptr() % 16 == 0
    assert mo.shape == (4, 16, 33, 512) and ml.shape == (4, 16, 33)
    ip = FI.pad_idx(torch.zeros(3, 2051, dtype=torch.int32), 2112)
    assert ip.shape == (3, 2112) and torch.all(ip[:, 2051:] == -1)


def test_heads_supported():
    kv, q, idx = _setup(1, h=12)
    with pytest.raises(AssertionError):
        FI.FlashInferGlm5Next("jit", module=FakeModule()).forward(q, kv, idx, 0.1)


@pytest.mark.parametrize("t", [3, 66])
def test_jit_pow2_rows_use_dsv3_2_kernels(t):
    kv, q, idx = _setup(t, pow2=True)
    mod = FakeModule()
    fi = FI.FlashInferGlm5Next("jit", module=mod, pow2=True)
    out, lse = fi.forward(q, kv, idx, 0.0625)
    assert mod.model_types == [FI.MODEL_TYPE_DSV3_2] and fi.kv_scale_format == "pow2_fp32"
    ro, rl = A.sparse_mla_ref(q.float(), A.nsa656_dequant_ref(kv.reshape(-1, 656)), idx, 0.0625)
    assert (out.float() - ro).norm() / ro.norm() < 1e-2


def test_trimmed_build_has_dsv3_2():
    src = FI.patched_sources(CSRC, trim=True)
    dec, pre = src["sparse_mla_sm120_decode_dsv3_2.cu"], src["sparse_mla_sm120_prefill.cu"]
    for h in FI.HEADS:
        for k in (2048, 2112):
            assert f"DSV3_2_DISPATCH_MT(ModelType::DSV3_2, {h}, {k})" in dec
            assert f"DSV3_2_DISPATCH_MT(ModelType::GLM_NSA, {h}, {k})" in dec
    assert "return dispatch_v32<ModelType::DSV3_2>(" in pre and "return dispatch_v32<ModelType::GLM_NSA>(" in pre
