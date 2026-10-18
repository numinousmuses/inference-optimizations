"""CPU tests of rtxsys D1 (gemma31_glue hooks + rtx_glue chains).

- The hook applies to SGLang v0.5.20: every anchor once, both files compile, idempotent, module installed.
- The PATCHED Gemma4DecoderLayer.forward, cut from the patched gemma4_causal.py and run on a 4-layer fake stack
  under the Triton interpreter, gives the same hidden states with NUMINOUS_GEMMA_GLUE on and off.
  - The fake linears quantize like the stock DeepGEMM linear, or consume the stashed (q, s) the way the fp8_utils
    entry hook does.
  - The stashes hand over exactly as designed: input_layernorm runs only in layer 0; post_attention_layernorm,
    pre_feedforward_layernorm and the stock layer-output kernel run only where the chains decline; nothing is
    left in a stash.
- MoE / PLE layers and empty batches stay stock; the preflight flags inert setups.
"""
from __future__ import annotations

import os
import shutil
import sys
import types

os.environ["TRITON_INTERPRET"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "kernels", "gemma31"))
import gemma31_glue_hooks as gh  # noqa: E402
import rtx_glue as G  # noqa: E402

SRC = os.path.expanduser("~/.cache/numinous/sgl-v0520/python/sglang")
need_tree = pytest.mark.skipif(not os.path.isdir(SRC), reason="needs operations/hopper/restore_cpu_preflight.sh")
BF = torch.bfloat16


@need_tree
def test_hook_applies_installs_and_is_idempotent(tmp_path):
    root = tmp_path / "sglang"
    for sub in ("srt/models", "srt/layers"):
        shutil.copytree(os.path.join(SRC, sub), root / sub, ignore=shutil.ignore_patterns("__pycache__"))
    out = gh.apply(str(root))
    assert "installed srt/layers/numinous_gemma_glue.py" in out
    assert sum("applied" in x for x in out) == len(gh.HOOKS)
    again = gh.apply(str(root))
    assert all("already applied" in x for x in again if not x.startswith("installed"))
    g = (root / gh.GEMMA).read_text()
    f = (root / gh.FP8U).read_text()
    assert g.count(gh.MARKER) == 5 and f.count(gh.MARKER) == 2
    compile(g, "gemma4_causal.py", "exec")
    compile(f, "fp8_utils.py", "exec")


@need_tree
def test_prequant_anchor_is_disjoint_from_dp_fp8_passthrough(tmp_path):
    """Both hooks patch deepgemm_w8a8_block_fp8_linear_with_fallback; they must apply in either order."""
    import dp_fp8_passthrough_hooks as pt
    for order in ("pt-first", "glue-first"):
        root = tmp_path / order / "sglang"
        for sub in ("srt/models", "srt/layers"):
            shutil.copytree(os.path.join(SRC, sub), root / sub, ignore=shutil.ignore_patterns("__pycache__"))
        if order == "pt-first":
            pt.apply(str(root))
            gh.apply(str(root))
        else:
            gh.apply(str(root))
            pt.apply(str(root))
        f = (root / gh.FP8U).read_text()
        assert gh.MARKER in f and pt.TAG in f
        compile(f, "fp8_utils.py", "exec")


# --------------------------------------------------------------------------------------- patched forward, fake stack
def _patched_forward():
    src = open(os.path.join(SRC, gh.GEMMA)).read()
    for rel, _name, anchor, text in gh.HOOKS:
        if rel == gh.GEMMA:
            src = src.replace(anchor, text.replace("{marker}", gh.MARKER), 1)
    cls = src.index("class Gemma4DecoderLayer(")
    start = src.index("    def forward(", cls)
    end = src.index("\nclass Gemma4TextModel(", start)
    body = src[start:end]
    # emulate the CUDA branch of the stock condition (the fakes run on CPU tensors)
    body = body.replace("(hidden_states.is_cuda or hidden_states.is_xpu)", "True")
    body = "\n".join(line[4:] if line.startswith("    ") else line for line in body.split("\n"))
    calls = {"stock_out": 0}

    def gemma_rmsnorm_residual_scalar(x, w, res, scalar, eps):
        calls["stock_out"] += 1
        return G.ref_rmsnorm_residual_scalar(x, w, res, scalar, eps)

    ns = {"torch": torch, "Optional": __import__("typing").Optional, "ForwardBatch": object, "_ngg": G,
          "gemma_rmsnorm_residual_scalar": gemma_rmsnorm_residual_scalar,
          "gemma_dual_rmsnorm_residual_scalar": None}
    exec(compile(body, "patched_gemma4_forward", "exec"), ns)
    return ns["forward"], calls


class Norm:
    def __init__(self, w, name, calls):
        self.weight = types.SimpleNamespace(data=w)
        self.variance_epsilon = 1e-6
        self.name, self.calls = name, calls

    def __call__(self, x, residual=None):
        self.calls[self.name] = self.calls.get(self.name, 0) + 1
        if residual is None:
            return G.ref_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return G.ref_fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)


def _dequant(q, s):
    m, k = q.shape
    return (q.float().view(m, k // 128, 128) * s[..., None]).view(m, k)


class Linear:
    """A block-FP8 linear: consumes a stashed (q, s) like the fp8_utils entry hook, else quantizes like the stock
    per_token_group_quant; then a bf16-output GEMM on the dequantized input."""

    def __init__(self, w, calls, name):
        self.w, self.calls, self.name = w, calls, name

    def __call__(self, x):
        pq = G.take_prequant(x, [128, 128])
        if pq is not None:
            self.calls[self.name + "_prequant"] = self.calls.get(self.name + "_prequant", 0) + 1
            q, s = pq
            rq, rs = G.ref_group_quant(x)                # what the stock linear would have quantized
            assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8)) and torch.equal(s, rs)
        else:
            q, s = G.ref_group_quant(x)
        return (_dequant(q, s) @ self.w.float().t()).to(BF)


class Attn:
    def __init__(self, w, calls):
        self.qkv_proj = Linear(w, calls, "qkv")

    def __call__(self, positions, hidden_states, forward_batch):
        return self.qkv_proj(hidden_states)


class MLP:
    def __init__(self, w, calls):
        self.gate_up_proj = Linear(w, calls, "gate_up")

    def __call__(self, x):
        return self.gate_up_proj(x)


class Layer:
    pass


def _stack(n_layers, n, calls, seed=0):
    g = torch.Generator().manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        L = Layer()
        L.enable_moe_block, L.moe, L.has_ple = False, None, False
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"):
            setattr(L, name, Norm((1.0 + 0.2 * torch.randn(n, generator=g)).to(BF), name, calls))
        L.self_attn = Attn((torch.randn(n, n, generator=g) / n ** 0.5).to(BF), calls)
        L.mlp = MLP((torch.randn(n, n, generator=g) / n ** 0.5).to(BF), calls)
        L.layer_scalar = torch.tensor([0.75], dtype=BF)
        layers.append(L)
    G.link_layers_full(layers, 0, n_layers)
    return layers


def _run(forward, layers, h):
    for L in layers:
        h = forward(L, None, h, None, None)[0]
    return h


def _ref_prequant(out):
    q, s = G.alloc_prequant(out.shape[0], out.shape[1], out.device)
    rq, rs = G.ref_group_quant(out)
    q.copy_(rq)
    s.copy_(rs)
    return q, s


def _ref_post_attn_chain(h, residual, w1, eps1, w2, eps2, quant):
    out, res_new = G.ref_fused_add_rmsnorm(G.ref_rmsnorm(h, w1, eps1), residual, w2, eps2)
    residual.copy_(res_new)
    q, s = _ref_prequant(out) if quant else (None, None)
    return out, residual, q, s


def _ref_cross_layer_chain(h, residual, w1, eps1, scalar, w2, eps2, quant):
    rout = G.ref_rmsnorm_residual_scalar(h, w1, residual, scalar, eps1)
    out = G.ref_rmsnorm(rout, w2, eps2)
    q, s = _ref_prequant(out) if quant else (None, None)
    return rout, out, q, s


@need_tree
@pytest.mark.parametrize("chains", ["torch-reference", "triton-kernel"])
def test_patched_forward_matches_stock_and_hands_over(monkeypatch, chains):
    """torch-reference: the chains computed with the very functions the fake stock norms use, so ON must equal OFF
    bit for bit (pure plumbing). triton-kernel: the real kernels under the interpreter, within reduction noise."""
    forward, out_calls = _patched_forward()
    if chains == "torch-reference":
        monkeypatch.setattr(G, "post_attn_chain", _ref_post_attn_chain)
        monkeypatch.setattr(G, "cross_layer_chain", _ref_cross_layer_chain)
    monkeypatch.setattr(G, "_linear_prequant_ok", lambda lin: lin is not None)
    monkeypatch.setattr(G, "_act_ok", lambda h, r: (h.dim() == 2 and h.dtype == BF and h.is_contiguous()
                                                    and r.shape == h.shape and h.shape[0] > 0))
    n, m, nl = 256, 5, 4
    h0 = (torch.randn(m, n, generator=torch.Generator().manual_seed(7)) * 3).to(BF)

    calls_off: dict = {}
    monkeypatch.setattr(G, "ON", False)
    ref = _run(forward, _stack(nl, n, calls_off), h0.clone())
    stock_out_off = out_calls["stock_out"]

    calls_on: dict = {}
    monkeypatch.setattr(G, "ON", True)
    layers = _stack(nl, n, calls_on)
    out_calls["stock_out"] = 0
    got = _run(forward, layers, h0.clone())

    assert calls_off["input_layernorm"] == nl and calls_off["post_attention_layernorm"] == nl
    assert calls_off["pre_feedforward_layernorm"] == nl and stock_out_off == nl
    assert calls_on.get("input_layernorm", 0) == 1                 # layers 1..3 take the prenormed input
    assert calls_on.get("post_attention_layernorm", 0) == 0 and calls_on.get("pre_feedforward_layernorm", 0) == 0
    assert out_calls["stock_out"] == 1                             # the last layer has no local successor
    assert calls_on.get("qkv_prequant", 0) == nl - 1 and calls_on.get("gate_up_prequant", 0) == nl
    assert G._pq is None and G._pre is None                        # nothing left behind
    if chains == "torch-reference":
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    else:
        # The kernel's fp32 reduction order and rsqrt rounding differ from the torch reference norms: about one
        # 1-ulp bf16 flip per 10^5 elements (test_rtx_glue.py). A flip that lands on a group max moves that
        # group's FP8 scale, and four layers of GEMMs spread it. Measured 5.4e-3.
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel < 1e-2, f"ON vs OFF relative error {rel:.2e}"


def test_moe_ple_and_empty_stay_stock(monkeypatch):
    monkeypatch.setattr(G, "ON", True)
    monkeypatch.setattr(G, "_linear_prequant_ok", lambda lin: True)
    calls: dict = {}
    layers = _stack(2, 256, calls)
    layers[0].enable_moe_block = True
    h = torch.zeros(3, 256, dtype=BF)
    assert G.layer_post_attn(layers[0], h, h.clone()) is None
    layers[1].has_ple = True
    assert G.layer_output(layers[1], h, h.clone()) is None
    fresh = _stack(2, 256, calls)
    e = torch.zeros(0, 256, dtype=BF)
    assert G.layer_post_attn(fresh[0], e, e.clone()) is None          # empty batch (and CPU tensors) -> stock


def test_preflight_flags():
    ctx = types.SimpleNamespace(env={gh.ENV: "1"}, args={"fp8_gemm_backend": "triton"},
                                hf_config=types.SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
                                device={"sku": "H100", "sm_major": 9, "is_hip": False})
    out = gh.numinous_preflight(ctx)
    assert any("Gemma4DecoderLayer only" in x for x in out) and any("DeepGEMM linear" in x for x in out)
    ctx.env = {gh.ENV: "1", "NUMINOUS_GEMMA_NORM_QUANT": "1"}
    assert any("clashes with NUMINOUS_GEMMA_NORM_QUANT" in x for x in gh.numinous_preflight(ctx))
    ctx.env = {}
    assert gh.numinous_preflight(ctx) == []


@need_tree
def test_g1b_and_d1_cannot_both_apply(tmp_path):
    """k3amd's G1b patches the same two anchors; the composed arm must carry one of them (preflight refuses both)."""
    import gemma31_norm_quant_hooks as nq
    root = tmp_path / "sglang"
    for sub in ("srt/models", "srt/layers"):
        shutil.copytree(os.path.join(SRC, sub), root / sub, ignore=shutil.ignore_patterns("__pycache__"))
    nq.apply(str(root))
    with pytest.raises(SystemExit, match="input anchor found 0 times"):
        gh.apply(str(root))
