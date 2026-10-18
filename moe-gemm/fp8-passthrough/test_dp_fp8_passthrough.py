"""CPU tests for dp_fp8_passthrough (hooks + module). Tree: $SGLANG_SRC or ~/.cache/numinous/sgl-v0520/python."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import dp_fp8_passthrough_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", pathlib.Path.home() / ".cache/numinous/sgl-v0520/python")) / "sglang"
FILES = sorted({h[1] for h in H.HOOKS})


def _tree(tmp_path):
    if not SRC.is_dir():
        pytest.skip(f"no SGLang tree at {SRC}")
    root = tmp_path / "sglang"
    for rel in FILES + ["kernels/ops/quantization/fp8_kernel.py"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / rel, root / rel)
    return root


def _module(on: bool):
    os.environ["NUMINOUS_DP_FP8_PASSTHROUGH"] = "1" if on else "0"
    spec = importlib.util.spec_from_file_location(f"nfp_{int(on)}", REPO / H.MODULE_SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _quant(x):
    """sglang_per_token_group_quant_fp8(x, 128) semantics: s = max(amax, eps)/448 per group, q = e4m3(x/s)."""
    g = x.float().view(x.shape[0], -1, 128)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    q = (g / s).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(x.shape), s.squeeze(-1)


def test_anchors_apply_compile_and_are_idempotent(tmp_path):
    root = _tree(tmp_path)
    assert all("anchor(s) ok" in ln for ln in H.apply(str(root), check=True))
    out = H.apply(str(root))
    assert any("installed" in ln for ln in out) and sum("applied" in ln for ln in out) == len(FILES)
    assert all("already applied" in ln for ln in H.apply(str(root), install_module=False))
    dp = (root / "srt/layers/dp_attention.py").read_text()
    # clear sits in _dp_gather (every gather path), stash right after the fp8 gather's dequant
    i_g = dp.index("def _dp_gather(")
    assert dp.index("_nfp.clear()", i_g) < dp.index("def dp_gather_partial(")
    i_f = dp.index("def _dp_gather_via_all_gatherv_fp8(")
    assert i_f < dp.index("_nfp.stash(global_tokens, gq, gs)") < dp.index("def _dp_gather_via_all_gatherv(")
    fu = (root / "srt/layers/quantization/fp8_utils.py").read_text()
    i_l = fu.index("def deepgemm_w8a8_block_fp8_linear_with_fallback(")
    assert i_l < fu.index("_nfp.for_linear(input_2d, block_size[1])") < fu.index("def _unpack_ue8m0_scale_for_triton(")
    cm = (root / "srt/layers/moe/cutlass_moe.py").read_text()
    i_c = cm.index("def cutlass_fused_experts_fp8(")
    assert i_c < cm.index("_nfp.for_routed(a)")
    dg = (root / "srt/layers/moe/moe_runner/deep_gemm.py").read_text()
    i_d = dg.index("def pre_permute_standard_to_deep_gemm(")
    assert i_d < dg.index("_nfp.for_routed(hidden_states)") < dg.index("ep_scatter(", i_d)
    assert "block_k == 128 and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0" in dg
    assert (root / H.MODULE_DST).is_file()


def test_linear_scale_layout_matches_the_stock_tma_layout(tmp_path):
    """for_linear's scale buffer copies the formula in create_per_token_group_quant_fp8_output_scale."""
    root = _tree(tmp_path)
    src = (root / "kernels/ops/quantization/fp8_kernel.py").read_text()
    assert "aligned_size = (x_shape[-2] + 3) // 4 * 4" in src
    assert "x_shape[:-2] + (x_shape[-1] // group_size, aligned_size)" in src
    assert ".transpose(-1, -2)[..., : x_shape[-2], :]" in src
    m = _module(True)
    rows, hidden = 4097, 4096
    x = torch.randn(rows, hidden).to(torch.bfloat16)
    q, s = _quant(x)
    m.stash(x, q.view(torch.uint8), s.contiguous())
    lq, ls = m.for_linear(x, 128)
    assert lq.dtype == torch.float8_e4m3fn and lq.shape == (rows, hidden)
    assert ls.shape == (rows, hidden // 128) and ls.stride() == (1, (rows + 3) // 4 * 4)
    assert torch.equal(ls, s)


def test_single_use_identity_and_env_guards():
    m = _module(True)
    x = torch.randn(64, 256).to(torch.bfloat16)
    q, s = _quant(x)
    m.stash(x, q.view(torch.uint8), s.contiguous())
    other = x.clone()
    assert m.for_routed(other) is None                       # different buffer
    assert m.for_linear(x, 64) is None                       # different group
    assert m.for_linear(x[:32], 128) is None                 # different rows
    rq, rs = m.for_routed(x)
    assert torch.equal(rq.view(torch.uint8), q.view(torch.uint8)) and torch.equal(rs, s)
    assert m.for_routed(x) is None                           # once per gather
    assert m.for_linear(x, 128) is not None and m.for_linear(x, 128) is None
    m.stash(x, q.view(torch.uint8), s.contiguous())
    m.clear()
    assert m.for_routed(x) is None and m.for_linear(x, 128) is None
    off = _module(False)
    off.stash(x, q.view(torch.uint8), s.contiguous())
    assert off.for_routed(x) is None and off.for_linear(x, 128) is None


def test_passthrough_equals_requantizing_the_dequantized_buffer():
    """The stock path requantizes bf16(q*s); the gathered (q, s) is what it recomputes."""
    torch.manual_seed(0)
    x = (torch.randn(2048, 4096) * torch.logspace(-2, 1, 4096)).to(torch.bfloat16)
    q, s = _quant(x)
    deq = (q.float().view(2048, -1, 128) * s.unsqueeze(-1)).view(2048, 4096).to(torch.bfloat16)
    q2, s2 = _quant(deq)
    same = (q2.view(torch.uint8) == q.view(torch.uint8)).float().mean().item()
    rel = ((s2 - s).abs() / s).max().item()
    assert same > 0.9999, same
    assert rel <= 2.0 ** -8, rel


# ------------------------------------------------------------------------------------------- FX1 regression (2026-10-06)
def _payload(m, rows, hidden):
    x = (torch.randn(rows, hidden) * 3).to(torch.bfloat16)
    q, s = _quant(x)
    m.stash(x, q.view(torch.uint8), s.contiguous())
    return x


def test_fx1_stale_moe_stash_never_reaches_the_attention_qkv_proj():
    """FX1: in a GLM-5.3-Flash MoE layer nothing consumes the "linear" kind, so the MoE input stash lingered. The next
    layer's KDA qkv_proj input on the 1-token health prefill had the same data_ptr/shape, and for_linear took it.
    The attention-boundary clear (LayerCommunicator.prepare_attn) now drops it first."""
    m = _module(True)
    buf = _payload(m, 1, 512)                         # 1 token on DP0, other ranks idle: global == local rows
    assert m.for_routed(buf) is not None              # the routed MoE consumer takes "moe"; "linear" stays unused
    alias = buf.view(1, 512)                          # the next layer's attention input at the same address
    assert m._take(alias, "probe") is not None        # the bug: without a lifetime bound a later linear matches
    m.clear()                                         # what the patched prepare_attn does at its top
    assert m.for_linear(alias, 128) is None           # the attention qkv_proj quantizes for itself


def test_malformed_payload_is_declined_never_raises():
    """FX1 raised IndexError on a scale tensor without a second dim. Any payload whose q/s are not
    [rows, hidden] / [rows, hidden/128] fp32 is refused at stash time and again at take time."""
    m = _module(True)
    x = torch.zeros(4, 256, dtype=torch.bfloat16)
    q = torch.zeros(4, 256, dtype=torch.uint8)
    m.stash(x, q, torch.zeros(4 * 2, dtype=torch.float32))         # 1-D scales
    assert m._st is None and m.for_linear(x, 128) is None and m.for_routed(x) is None
    m.stash(x, q, torch.zeros(4, 3, dtype=torch.float32))           # wrong group count
    assert m._st is None
    m.stash(x, q, torch.zeros(4, 2, dtype=torch.float32))           # well-formed
    assert m._st is not None
    m._st["s"] = torch.zeros(8, dtype=torch.float32)                # whatever corrupts it later
    assert m.for_linear(x, 128) is None and m._st is None


def test_patched_tree_clears_before_every_attention_and_every_forward(tmp_path):
    root = _tree(tmp_path)
    H.apply(str(root))
    cm = (root / "srt/layers/communicator.py").read_text()
    i = cm.index("    def prepare_attn(\n")
    body = cm[i:cm.index("\n    def ", i + 10)]
    assert body.index("_nfp.clear()") < body.index("if self._sp_variant is not None:")
    dp = (root / "srt/layers/dp_attention.py").read_text()
    j = dp.index("def set_dp_buffer_len(")
    assert dp.index("_nfp.clear()", j) < dp.index("_DpGatheredBufferWrapper.set_dp_buffer_len(", j)


def test_real_glm5_next_kda_qkv_proj_runs_after_prepare_attn():
    """The FX1 path in v0.5.20 glm5_next.py: Glm5NextDecoderLayer.forward calls layer_communicator.prepare_attn
    (now clearing the stash) before self.self_attn. The KDA attention's forward reaches qkv_proj only through
    forward_qkvbfg / forward_qkvbfg_fused, both inside self_attn."""
    import ast
    src_path = SRC / "srt/models/glm5_next.py"
    if not src_path.is_file():
        pytest.skip("no glm5_next.py")
    src = src_path.read_text()
    tree = ast.parse(src)
    layer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextDecoderLayer")
    fwd = next(n for n in layer.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    calls = [ast.unparse(n.func) for n in ast.walk(fwd) if isinstance(n, ast.Call)]
    order = [c for c in calls if c in ("self.layer_communicator.prepare_attn", "self.self_attn")]
    assert order[:2] == ["self.layer_communicator.prepare_attn", "self.self_attn"]
    seg = ast.get_source_segment(src, fwd)
    assert seg.index("self.layer_communicator.prepare_attn(") < seg.index("self.self_attn(")
    qkv_users = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and "self.qkv_proj(hidden_states)" in (ast.get_source_segment(src, n) or "")]
    assert "forward_qkvbfg" in qkv_users


@pytest.mark.parametrize("registry,key", [("hopper_registry.json", "glm-5.3-flash/standard@8xH200-FX1"),
                                          ("hopper_glm_registry.json", "glm-5.3/standard@8xH200-GX1")])
def test_fixed_hook_composes_with_the_final_arm_series(tmp_path, registry, key):
    """Every operations/patches/*_hooks.py of the FX1 / GX1 cells, in registry order, on a full v0.5.20 copy, with
    this (fixed) dp_fp8_passthrough: all apply, the touched files compile, the clears are in place."""
    import importlib
    import json
    if not SRC.is_dir():
        pytest.skip(f"no SGLang tree at {SRC}")
    d = json.loads((REPO / "configs" / registry).read_text())
    cells = d["cells"] if isinstance(d, dict) and "cells" in d else d
    cell = next((c for c in cells if isinstance(c, dict) and c.get("key") == key), None)
    if cell is None:
        pytest.skip(f"{key} not in {registry}")
    files = [f for f in cell["config"]["hooks"]["files"] if f.startswith("operations/patches/")
             and f.endswith("_hooks.py") and not f.endswith("sglang_numinous_hooks.py")]
    assert "operations/patches/dp_fp8_passthrough_hooks.py" in files
    root = tmp_path / "sglang"
    shutil.copytree(SRC, root, ignore=shutil.ignore_patterns("__pycache__"))
    for f in files:
        importlib.import_module(pathlib.Path(f).stem).apply(str(root))
    for rel in ("srt/layers/communicator.py", "srt/layers/dp_attention.py", "srt/layers/quantization/fp8_utils.py"):
        text = (root / rel).read_text()
        compile(text, rel, "exec")
        assert H.TAG in text
    assert (root / "srt/layers/dp_attention.py").read_text().count("_nfp.clear()") == 2
