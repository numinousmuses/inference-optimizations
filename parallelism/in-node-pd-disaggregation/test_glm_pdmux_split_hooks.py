"""CPU tests for glm_pdmux_split_hooks.py (amdglm): the anchor resolves once on SGLang v0.5.20 and the patched file
compiles; the patched DeepseekV2ForCausalLM.forward_split_prefill, run as splits [0,a) + [a,b) + [b,n), reproduces the
stock layer loop exactly (hidden / residual / the shared DSA top-k carried across splits) on fake layers; refused
shapes raise."""
from __future__ import annotations

import ast
import contextlib
import os
import pathlib
import shutil
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm_pdmux_split_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python")) / "sglang"
needs_src = pytest.mark.skipif(not (SRC / H.DSV2).exists(), reason="SGLang v0.5.20 source absent")


def _copy(root):
    pkg = root / "sglang"
    (pkg / H.DSV2).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / H.DSV2, pkg / H.DSV2)
    return pkg


@needs_src
def test_apply_once_compile_idempotent(tmp_path):
    pkg = _copy(tmp_path)
    assert H.apply(str(pkg), check=True) == ["split_prefill: anchor ok"]
    assert H.apply(str(pkg)) == [f"split_prefill: applied to {H.DSV2}"]
    assert H.apply(str(pkg)) == ["split_prefill: already applied"]
    compile((pkg / H.DSV2).read_text(), H.DSV2, "exec")


def _method_source(pkg) -> str:
    src = (pkg / H.DSV2).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2ForCausalLM":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward_split_prefill":
                    seg = ast.get_source_segment(src, item)
                    import textwrap
                    return textwrap.dedent("    " + seg) if not seg.startswith("def") else seg
    raise AssertionError("forward_split_prefill not found in DeepseekV2ForCausalLM")


class _TopKShare:
    """Stand-in for IndexTopKShareState: keeps the last non-None top-k (the stock update shares across layers)."""

    def __init__(self, fb, topk):
        self.topk_indices = topk

    def update(self, topk):
        if topk is not None:
            self.topk_indices = topk


class _Bump:
    def __init__(self, buffer_size, dtype, device):
        self.size = buffer_size


class _Ctx:
    def maybe_input_scattered(self, fb):
        return contextlib.nullcontext()


class _Recorder:
    def with_current_layer(self, i):
        return contextlib.nullcontext()


def _namespace(torch):
    return {"torch": torch, "Tuple": tuple, "ForwardBatch": object, "IndexTopKShareState": _TopKShare,
            "BumpAllocator": _Bump, "get_attn_tp_context": lambda: _Ctx(),
            "get_global_expert_distribution_recorder": lambda: _Recorder()}


def _fake_model(torch, n_layers, freq=4):
    class Layer:
        def __init__(self, i):
            self.i = i

        def __call__(self, positions, h, fb, residual, zero_alloc, gemm_alloc, scaling, prev_topk_indices,
                     captured_last_layer_outputs, next_full_attention_layer_id):
            assert scaling is None and captured_last_layer_outputs is None and zero_alloc.size >= 2
            # DSA: every freq-th layer computes a top-k, the others reuse prev (index_topk_freq)
            topk = (h.sum(-1, keepdim=True) * 0 + self.i).long() if self.i % freq == 0 else None
            used = topk if topk is not None else prev_topk_indices
            res = h if residual is None else residual + h
            out = torch.tanh(h * 1.01 + 0.1 * self.i) + 0.001 * used.float()
            return out, res, topk

    class Norm:
        def __call__(self, h, residual=None):
            y = h * 2.0 if residual is None else (h + residual) * 2.0
            return y if residual is None else (y, None)

    m = types.SimpleNamespace(
        layers=[Layer(i) for i in range(n_layers)], embed_tokens=lambda ids: ids.float().unsqueeze(-1).repeat(1, 4),
        norm=Norm(), pp_group=types.SimpleNamespace(is_first_rank=True, is_last_rank=True), layers_to_capture=[],
        llama_4_scaling_config=None, next_full_attention_layer_id={}, gemm_output_zero_allocator_size=0)
    return m


def _fake_self(torch, model, n_layers):
    return types.SimpleNamespace(
        model=model, capture_aux_hidden_states=False, config=types.SimpleNamespace(num_hidden_layers=n_layers),
        lm_head="head", logits_processor=lambda ids, h, head, fb, aux: ("logits", h.clone()))


def _fb():
    return types.SimpleNamespace(can_run_tbo=False, hidden_states=None, residual=None, model_specific_states=None,
                                 forward_mode=types.SimpleNamespace(is_idle=lambda: False))


@needs_src
def test_splits_equal_one_pass(tmp_path):
    torch = pytest.importorskip("torch")
    pkg = _copy(tmp_path)
    H.apply(str(pkg))
    ns = _namespace(torch)
    exec(_method_source(pkg), ns)
    fn = ns["forward_split_prefill"]
    n = 11
    ids = torch.arange(7)
    pos = torch.arange(7)
    full_self = _fake_self(torch, _fake_model(torch, n), n)
    fb_full = _fb()
    ref = fn(full_self, ids, pos, fb_full, (0, n))
    for cuts in ([0, 3, 7, 11], [0, 1, 2, 11], [0, 4, 8, 11], [0, 5, 11]):
        s = _fake_self(torch, _fake_model(torch, n), n)
        fb = _fb()
        outs = [fn(s, ids, pos, fb, (a, b)) for a, b in zip(cuts, cuts[1:])]
        assert all(o is None for o in outs[:-1])
        assert outs[-1][0] == "logits" and torch.equal(outs[-1][1], ref[1]), cuts
        assert torch.equal(fb.model_specific_states["dsa_topk"], fb_full.model_specific_states["dsa_topk"])


@needs_src
def test_refused_shapes_raise(tmp_path):
    torch = pytest.importorskip("torch")
    pkg = _copy(tmp_path)
    H.apply(str(pkg))
    ns = _namespace(torch)
    exec(_method_source(pkg), ns)
    fn = ns["forward_split_prefill"]
    ids = torch.arange(3)
    for mutate in (lambda s, fb: setattr(fb, "can_run_tbo", True),
                   lambda s, fb: setattr(s, "capture_aux_hidden_states", True),
                   lambda s, fb: setattr(s.model, "llama_4_scaling_config", {"beta": 1}),
                   lambda s, fb: setattr(s.model.pp_group, "is_last_rank", False)):
        s = _fake_self(torch, _fake_model(torch, 4), 4)
        fb = _fb()
        mutate(s, fb)
        with pytest.raises(NotImplementedError):
            fn(s, ids, ids, fb, (0, 4))
