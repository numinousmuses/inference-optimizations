"""CPU test for gemma_trtmoe_hooks.py: applies the patcher to a copy of gemma4_causal.py (the installed sglang
when present, else the v0.5.20 file from the local sglang-fork checkout), checks anchors, idempotence, syntax,
the inserted call sites, and the helper's decisions with mocked backends. On a box with sglang it also imports
the patched module.

    pytest operations/patches/test_gemma_trtmoe_hooks.py -v
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.expanduser("~/sglang-fork")


def _hooks():
    spec = importlib.util.spec_from_file_location("gemma_trtmoe_hooks", os.path.join(HERE, "gemma_trtmoe_hooks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _source() -> str:
    """A pristine (unpatched) gemma4_causal.py: the installed one unless the patcher already ran on it, else the
    v0.5.20 file from the local sglang-fork checkout."""
    spec = importlib.util.find_spec("sglang")
    if spec and spec.submodule_search_locations:
        p = os.path.join(list(spec.submodule_search_locations)[0], "srt/models/gemma4_causal.py")
        if os.path.isfile(p):
            src = open(p).read()
            if "# numinous-hook:" not in src:
                return src
            pristine = p + ".numinous.orig"
            if os.path.isfile(pristine):
                return open(pristine).read()
    if os.path.isdir(FORK):
        r = subprocess.run(["git", "show", "v0.5.20:python/sglang/srt/models/gemma4_causal.py"], cwd=FORK,
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
    pytest.skip("no pristine gemma4_causal.py (installed copy already patched, no .numinous.orig, no sglang-fork)")


@pytest.fixture
def root():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "srt/models"))
        with open(os.path.join(d, "srt/models/gemma4_causal.py"), "w") as f:
            f.write(_source())
        yield d


def test_check_then_apply_then_idempotent(root):
    h = _hooks()
    assert all("anchor ok" in x for x in h.apply(root, check=True))
    lines = h.apply(root)
    assert all("applied" in x for x in lines) and len(lines) == 3
    assert all("already applied" in x for x in h.apply(root))
    src = open(os.path.join(root, "srt/models/gemma4_causal.py")).read()
    compile(src, "gemma4_causal.py", "exec")
    # the TopK gets the output_format right after the custom routing function
    assert "custom_routing_function=routing_function,\n            # numinous-hook:trtmoe_routing\n" \
           "            output_format=_numinous_gemma4_topk_output_format(),\n        )" in src
    # the dump keeps the MoE input and fires after the experts
    i_in = src.index("_numinous_moe_in = hidden_states")
    i_topk = src.index("topk_output = self.topk(hidden_states, router_logits)")
    i_exp = src.index("hidden_states = self.experts(hidden_states, topk_output)")
    i_dump = src.index("_numinous_moe_dump(self, _numinous_moe_in, router_logits, topk_output, hidden_states)")
    assert i_in < i_topk < i_exp < i_dump < src.index("return hidden_states.view(num_tokens, hidden_dim)")


def test_missing_anchor_fails_loudly(root):
    h = _hooks()
    p = os.path.join(root, "srt/models/gemma4_causal.py")
    open(p, "w").write(open(p).read().replace("custom_routing_function=routing_function,", "custom_routing_function=rf,"))
    with pytest.raises(SystemExit):
        h.apply(root)


def _helper_module(root):
    """Exec only the appended helpers (no sglang import needed) into a fresh module."""
    h = _hooks()
    h.apply(root)
    src = open(os.path.join(root, "srt/models/gemma4_causal.py")).read()
    helpers = src[src.index("# numinous-hook:trtmoe_helpers"):]
    mod = types.ModuleType("helpers")
    exec(compile(helpers, "helpers", "exec"), mod.__dict__)
    return mod


def test_topk_format_decision(root, monkeypatch):
    mod = _helper_module(root)
    fmt = types.SimpleNamespace(STANDARD="STANDARD")

    def fake(backend_is_trtllm):
        moe = types.ModuleType("sglang.srt.layers.moe")
        moe.get_moe_runner_backend = lambda: types.SimpleNamespace(is_flashinfer_trtllm=lambda: backend_is_trtllm)
        topk = types.ModuleType("sglang.srt.layers.moe.topk")
        topk.TopKOutputFormat = fmt
        pkg = types.ModuleType("sglang")
        srt = types.ModuleType("sglang.srt")
        layers = types.ModuleType("sglang.srt.layers")
        for name, m in (("sglang", pkg), ("sglang.srt", srt), ("sglang.srt.layers", layers),
                        ("sglang.srt.layers.moe", moe), ("sglang.srt.layers.moe.topk", topk)):
            monkeypatch.setitem(sys.modules, name, m)

    monkeypatch.delenv("SGLANG_NUMINOUS_GEMMA_TRTMOE_ROUTING", raising=False)
    fake(True)
    assert mod._numinous_gemma4_topk_output_format() == "STANDARD"
    fake(False)
    assert mod._numinous_gemma4_topk_output_format() is None      # cutlass / marlin / cutedsl: stock behaviour
    fake(True)
    monkeypatch.setenv("SGLANG_NUMINOUS_GEMMA_TRTMOE_ROUTING", "0")
    assert mod._numinous_gemma4_topk_output_format() is None      # the probe's stock-trtllm arm


def test_dump_inert_without_env(root, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    mod = _helper_module(root)
    moe = types.SimpleNamespace(layer_id=0, per_expert_scale=torch.ones(4))
    x = torch.zeros(16, 8)
    monkeypatch.delenv("NUMINOUS_MOE_DUMP", raising=False)
    mod._numinous_moe_dump(moe, x, x, types.SimpleNamespace(), x)
    monkeypatch.setenv("NUMINOUS_MOE_DUMP", str(tmp_path))
    monkeypatch.delenv("NUMINOUS_MOE_DUMP_LAYERS", raising=False)
    mod._numinous_moe_dump(types.SimpleNamespace(layer_id=3, per_expert_scale=torch.ones(4)), x, x, None, x)
    assert not list(tmp_path.iterdir())                            # layers outside the list never dump
    topk = types.SimpleNamespace(topk_ids=torch.zeros(16, 2, dtype=torch.int32), topk_weights=torch.ones(16, 2))
    for _ in range(6):
        mod._numinous_moe_dump(moe, x, x, topk, x)
    files = sorted(tmp_path.iterdir())
    assert len(files) == 4                                          # capped
    rec = torch.load(files[0])
    assert rec["topk_ids"].shape == (16, 2) and rec["topk_format"] == "SimpleNamespace" and rec["layer"] == 0
    monkeypatch.setenv("NUMINOUS_MOE_DUMP_LAYERS", "0,7")
    mod._numinous_moe_dump(types.SimpleNamespace(layer_id=7, per_expert_scale=torch.ones(4)), x, x, topk, x)
    assert (tmp_path / "moe_l7_0.pt").exists()


@pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="needs the engine package")
def test_patched_module_imports(root):
    h = _hooks()
    h.apply(root)
    spec = importlib.util.spec_from_file_location("gemma4_patched", os.path.join(root, "srt/models/gemma4_causal.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod._numinous_gemma4_topk_output_format) and callable(mod._numinous_moe_dump)
