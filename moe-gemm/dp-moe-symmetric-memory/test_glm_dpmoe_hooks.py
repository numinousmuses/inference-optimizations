"""The GLM fused DP-MoE hook patcher against the SGLang actually installed (dev boxes; skips without sglang).

Copies the installed package, installs numinous_dp_moe_nvfp4 / numinous_dp_moe / symm_collectives next to the hooks
the way the overlay does, applies both hooks, then in a clean interpreter imports the patched model modules
(deepseek_v2, glm4_moe, deepseek_nextn) and the installed module, checks where the hooks landed (inside
DeepseekV2DecoderLayer.forward before prepare_mlp, with every name they read already bound; inside
DeepseekV2ForCausalLM.load_weights after do_load_weights), and resolves every symbol the module imports lazily.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import glm_dpmoe_hooks as hooks  # noqa: E402

sglang_spec = importlib.util.find_spec("sglang")
pytestmark = pytest.mark.skipif(sglang_spec is None, reason="needs an installed sglang (dev box)")

MODULES = {
    "operations/kernels/numinous_dp_moe_nvfp4.py": "srt/layers/moe/numinous_dp_moe_nvfp4.py",
    "operations/kernels/numinous_dp_moe.py": "srt/layers/moe/numinous_dp_moe.py",
    "operations/kernels/symm_collectives.py": "srt/layers/moe/symm_collectives.py",
}

# (module, attribute): everything numinous_dp_moe_nvfp4 imports inside functions
LAZY = [
    ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
    ("flashinfer.fused_moe", "trtllm_fp4_block_scale_routed_moe"),
    ("sglang.srt.distributed", "get_tp_group"),
    ("sglang.srt.layers.communicator", "CommunicateSummableTensorPairFn"),
    ("sglang.srt.layers.communicator", "CommunicateWithAllReduceAndLayerNormFn"),
    ("sglang.srt.layers.communicator", "get_attn_tp_context"),
    ("sglang.srt.layers.dp_attention", "get_attention_dp_rank"),
    ("sglang.srt.layers.dp_attention", "get_attention_dp_size"),
    ("sglang.srt.layers.dp_attention", "get_global_dp_buffer_len"),
    ("sglang.srt.layers.dp_attention", "get_local_dp_buffer_len"),
    ("sglang.srt.layers.dp_attention", "is_dp_attention_enabled"),
    ("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm", "get_activation_type"),
    ("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm", "quantize_hidden_states_fp4"),
    ("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm", "trtllm_moe_enable_pdl"),
    ("sglang.srt.layers.moe.topk", "select_experts"),
    ("sglang.srt.layers.moe.utils", "RoutingMethodType"),
    ("sglang.srt.layers.quantization.modelopt_quant", "ModelOptNvFp4FusedMoEMethod"),
    ("sglang.srt.models.deepseek_v2", "DeepseekV2MoE"),
    ("sglang.srt.runtime_context", "get_exec"),
    ("sglang.srt.runtime_context", "get_forward"),
    ("sglang.srt.runtime_context", "get_parallel"),
    ("sglang.srt.utils.common", "next_power_of_2"),
]


@pytest.fixture(scope="module")
def patched(tmp_path_factory) -> pathlib.Path:
    """A patched copy of the installed sglang package under <tmp>/sglang."""
    src = list(sglang_spec.submodule_search_locations)[0]
    base = tmp_path_factory.mktemp("sgl")
    dst = base / "sglang"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    for local, rel in MODULES.items():
        shutil.copy(ROOT / local, dst / rel)
    hooks.apply(str(dst))
    return base


def test_every_anchor_resolves_once_on_the_installed_sglang():
    lines = hooks.apply(list(sglang_spec.submodule_search_locations)[0], check=True)
    assert len(lines) == len(hooks.HOOKS)
    assert all(line.endswith(("anchor ok", "already applied")) for line in lines), lines


def test_apply_is_idempotent(patched):
    again = hooks.apply(str(patched / "sglang"))
    assert all(line.endswith("already applied") for line in again), again


def test_hooks_land_in_the_right_functions(patched):
    """AST of the patched deepseek_v2.py: the layer hook sits in DeepseekV2DecoderLayer.forward before the
    prepare_mlp call and every name it reads is assigned earlier in that function; the prepare hook follows
    do_load_weights in DeepseekV2ForCausalLM.load_weights."""
    import ast
    tree = ast.parse((patched / "sglang/srt/models/deepseek_v2.py").read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    def method(cls, name):
        return next(n for n in classes[cls].body if isinstance(n, ast.FunctionDef) and n.name == name)

    fwd = method("DeepseekV2DecoderLayer", "forward")
    src = ast.get_source_segment((patched / "sglang/srt/models/deepseek_v2.py").read_text(), fwd)
    hook_at, prep_at = src.index("_ndm4.forward_layer("), src.index("self.layer_communicator.prepare_mlp(")
    assert hook_at < prep_at
    bound = {a.arg for a in fwd.args.args}
    hook_line = fwd.lineno + src[:hook_at].count("\n")
    for node in ast.walk(fwd):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.lineno < hook_line:
            bound.add(node.id)
    for name in ("self", "hidden_states", "residual", "forward_batch", "topk_indices"):
        assert name in bound, f"{name} is not bound before the hook"
    lw = ast.get_source_segment((patched / "sglang/srt/models/deepseek_v2.py").read_text(),
                                method("DeepseekV2ForCausalLM", "load_weights"))
    assert lw.index("self.do_load_weights(weights, is_nextn)") < lw.index("_ndm4.prepare(self)")


def test_patched_modules_import_in_a_clean_interpreter(patched):
    code = ("import sglang, sys, inspect; assert sglang.__file__.startswith(sys.argv[1]), sglang.__file__\n"
            "import sglang.srt.models.deepseek_v2 as m\n"
            "import sglang.srt.models.glm4_moe as g, sglang.srt.models.deepseek_nextn as nx\n"
            "src = inspect.getsource(m)\n"
            "assert 'numinous-hook:glm_dpmoe' in src and 'numinous-hook:glm_prepare' in src\n"
            "assert issubclass(g.GlmMoeDsaForCausalLM, m.DeepseekV2ForCausalLM)\n"
            "assert g.GlmMoeDsaForCausalLM.load_weights is m.DeepseekV2ForCausalLM.load_weights\n"
            "from sglang.srt.layers.moe import numinous_dp_moe_nvfp4 as n4, numinous_dp_moe as n\n"
            "import sglang.srt.layers.moe.symm_collectives as sc\n"
            "assert n4.layout is n.layout and n4.phase_pull is n.phase_pull and n4.push_payloads is sc.push_payloads\n"
            "print('ok')")
    env = {**os.environ, "PYTHONPATH": str(patched)}
    r = subprocess.run([sys.executable, "-c", code, str(patched)], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


def test_lazy_imports_resolve_in_server_import_order():
    """In a clean interpreter, after the model module (importing the MoE runner standalone is circular)."""
    code = ("import importlib, json\n"
            "import sglang.srt.models.deepseek_v2\n"
            f"lazy = {LAZY!r}\n"
            "missing = [f'{m}.{a}' for m, a in lazy if not hasattr(importlib.import_module(m), a)]\n"
            "print(json.dumps(missing))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip().splitlines()[-1] == "[]", r.stdout[-2000:]


def test_non_moe_layer_is_declined_with_one_log_line(patched):
    """A dense decoder layer (GLM layers 0-2: DeepseekV2MLP) is declined statically, logged once, cached."""
    code = ("import logging, types\n"
            "import sglang.srt.models.deepseek_v2\n"
            "from sglang.srt.layers.moe import numinous_dp_moe_nvfp4 as n4\n"
            "rec = []\n"
            "h = logging.Handler(); h.emit = lambda r: rec.append(r.getMessage())\n"
            "logging.getLogger('sglang.numinous').addHandler(h)\n"
            "layer = types.SimpleNamespace(mlp=types.SimpleNamespace())\n"
            "fb = types.SimpleNamespace(dp_padding_mode=None)\n"
            "assert n4.covers(layer, fb) is False and n4.covers(layer, fb) is False\n"
            "assert 'not DeepseekV2MoE' in layer._numinous_ndm4_static, layer._numinous_ndm4_static\n"
            "assert sum('numinous_dp_moe_nvfp4 declined' in m for m in rec) == 1, rec\n"
            "print('ok')")
    env = {**os.environ, "PYTHONPATH": str(patched)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]
