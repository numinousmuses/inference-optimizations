"""CPU tests for gemma_extend_sm90_hooks.py (Hopper port of the GQA extend kernel, env-gated; agent rtxv41).

The patch is applied on top of gemma_extend_hooks over the stock v0.5.20 extend_attention.py (git bytes). The helper
semantics are run with mocked capabilities. The H100 tile configs are run through Triton's CPU interpreter at
Gemma 4 31B's global-layer shape (32 query heads, 4 KV heads, head_dim 512, FP8 cached prefix) against the float32
reference.
    pytest -p no:cacheprovider operations/patches/test_gemma_extend_sm90_hooks.py
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "operations" / "kernels"))
import gemma_extend_sm90_hooks as hk  # noqa: E402

SGL = pathlib.Path(os.environ.get("SGLANG_FORK", "/home/ubuntu/sglang-fork"))
REF = "94602c9c2b"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("sgl") / "sglang"
    f = root / hk.PATH
    f.parent.mkdir(parents=True)
    f.write_text(subprocess.run(["git", "-C", str(SGL), "show", f"{REF}:python/sglang/{hk.PATH}"],
                                capture_output=True, text=True, check=True).stdout)
    base = _load("gemma_extend_hooks", HERE / "gemma_extend_hooks.py")
    base.install_modules(str(root), check=False)
    base.apply(str(root))
    return root


def test_refuses_without_the_base_hook(tmp_path):
    root = tmp_path / "sglang"
    f = root / hk.PATH
    f.parent.mkdir(parents=True)
    f.write_text("# stock\n")
    with pytest.raises(SystemExit, match="run gemma_extend_hooks.py first"):
        hk.apply(str(root))


def test_apply_compile_idempotent(tree):
    assert "anchors ok" in hk.apply(str(tree), check=True)[0]
    assert "applied" in hk.apply(str(tree))[0]
    src = (tree / hk.PATH).read_text()
    compile(src, "extend_attention.py", "exec")
    assert src.count(hk.MARKER) == 2
    assert "already applied" in hk.apply(str(tree))[0]


def _helpers(tree, cap):
    text = (tree / hk.PATH).read_text()
    if hk.MARKER not in text:
        text = hk.patch_source(text)
    block = text[text.index("# numinous-hook:gqa_helpers"):]
    ns = {"_is_cuda": True, "CUDA_CAPABILITY": cap}
    exec(compile(block, "<helpers>", "exec"), ns)  # noqa: S102
    return ns


@pytest.mark.parametrize("cap,env,expect", [
    ((9, 0), {}, False),                                         # Hopper: off unless asked
    ((9, 0), {"SGLANG_NUMINOUS_EXTEND_GQA_SM90": "1"}, True),
    ((9, 0), {"SGLANG_NUMINOUS_EXTEND_GQA_SM90": "1", "SGLANG_NUMINOUS_EXTEND_GQA": "0"}, False),
    ((12, 0), {}, True),                                         # SM120 unchanged
    ((10, 0), {"SGLANG_NUMINOUS_EXTEND_GQA_SM90": "1"}, False),  # Blackwell DC never
])
def test_enable_gate(tree, monkeypatch, cap, env, expect):
    for k in ("SGLANG_NUMINOUS_EXTEND_GQA_SM90", "SGLANG_NUMINOUS_EXTEND_GQA"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert _helpers(tree, cap)["_numinous_gqa_enabled"]() is expect


def test_sm90_tiles_and_sm120_unchanged(tree, monkeypatch):
    monkeypatch.delenv("SGLANG_NUMINOUS_GQA_CFG", raising=False)
    h90 = _helpers(tree, (9, 0))["_numinous_gqa_config"]
    assert h90(512, 8) == dict(block_t=8, block_n=128, dk=64, dv=256, num_warps=8, num_stages=2)
    assert h90(256, 2) == dict(block_t=32, block_n=64, dk=64, dv=256, num_warps=8, num_stages=2)
    h120 = _helpers(tree, (12, 0))["_numinous_gqa_config"]
    assert h120(512, 8) == dict(block_t=8, block_n=128, dk=64, dv=256, num_warps=8, num_stages=3)
    assert h120(256, 2)["num_stages"] == 1
    monkeypatch.setenv("SGLANG_NUMINOUS_GQA_CFG", "num_stages=1")
    assert h90(512, 8)["num_stages"] == 1


def test_sm90_tiles_fit_hopper_shared_memory():
    """K + V tiles per pipeline stage, BF16 (the new-token K/V are BF16; the FP8 prefix is half), times stages,
    plus the query tile (block_t x group rows x dk) and the fp32 accumulator spill margin: under 227 KB."""
    for hd, group, cfg in ((512, 8, dict(block_t=8, block_n=128, dk=64, dv=256, num_stages=2)),
                           (256, 2, dict(block_t=32, block_n=64, dk=64, dv=256, num_stages=2))):
        per_stage = cfg["block_n"] * cfg["dk"] * 2 + cfg["block_n"] * cfg["dv"] * 2
        q_tile = cfg["block_t"] * group * cfg["dk"] * 2
        assert per_stage * cfg["num_stages"] + q_tile < 227 * 1024, (hd, group)


def test_h100_tiles_match_reference_at_the_gemma31_global_shape():
    """Gemma 4 31B full-attention layer at TP1: 32 query heads, 4 KV heads (group 8), head_dim 512, FP8 prefix,
    two requests with different prefix / new lengths, partial tiles. Run with the exact SM90 config."""
    import torch
    import extend_attention_gqa as eg
    torch.manual_seed(3)
    H, KVH, D = 32, 4, 512
    prefixes, news = [37, 0], [11, 5]
    P, T = sum(prefixes), sum(news)
    kb = (torch.randn(P + 4, KVH, D) * 0.5).to(torch.float8_e4m3fn)
    vb = (torch.randn(P + 4, KVH, D) * 0.5).to(torch.float8_e4m3fn)
    q = torch.randn(T, H, D).to(torch.bfloat16)
    ke = (torch.randn(T, KVH, D) * 0.5).to(torch.bfloat16)
    ve = (torch.randn(T, KVH, D) * 0.5).to(torch.bfloat16)
    qo = torch.tensor([0, news[0], T], dtype=torch.int32)
    kvp = torch.tensor([0, prefixes[0], P], dtype=torch.int32)
    idx = torch.randperm(P + 4)[:P].to(torch.int64)
    o = torch.empty_like(q)
    cfg = dict(block_t=8, block_n=128, dk=64, dv=256, num_warps=8, num_stages=2)
    eg.extend_attention_gqa(q, ke, ve, o, kb, vb, qo, kvp, idx, max(news), D ** -0.5, 1.0, 1.0, -1, **cfg)
    ref = eg.reference(q, ke, ve, kb, vb, qo, kvp, idx, D ** -0.5, 1.0, 1.0, sliding_window_size=-1)
    assert (o.float() - ref).abs().max() < 0.05


def test_preflight():
    import types
    ok = types.SimpleNamespace(env={"SGLANG_NUMINOUS_EXTEND_GQA_SM90": "1"}, device={"sm_major": 9},
                               args={"attention_backend": "triton"})
    assert hk.numinous_preflight(ok) == []
    assert hk.numinous_preflight(types.SimpleNamespace(env={}, device={}, args={})) == []
    bad = types.SimpleNamespace(env={"SGLANG_NUMINOUS_EXTEND_GQA_SM90": "1"}, device={"sm_major": 10},
                                args={"attention_backend": "fa3"})
    probs = hk.numinous_preflight(bad)
    assert any("non-SM90" in p for p in probs) and any("triton extend kernel only" in p for p in probs)
