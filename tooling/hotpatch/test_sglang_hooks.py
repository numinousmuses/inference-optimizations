"""The anchored hook patcher against the SGLang actually installed (runs on the dev boxes; skips without sglang).

Copies the installed package, installs our modules and tables next to the hooks the way gate_arms' overlay dicts do,
applies every hook, then imports the patched modules in a clean interpreter, and resolves every symbol our modules
import lazily (the v0.5.18 -> v0.5.20 moves broke several of them once).
"""
from __future__ import annotations

import importlib
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

import sglang_numinous_hooks as hooks  # noqa: E402

sglang_spec = importlib.util.find_spec("sglang")
pytestmark = pytest.mark.skipif(sglang_spec is None, reason="needs an installed sglang (dev box)")

MODULES = {
    "operations/kernels/numinous_dp_moe.py": "srt/layers/moe/numinous_dp_moe.py",
    "operations/kernels/symm_collectives.py": "srt/layers/moe/symm_collectives.py",
    "operations/kernels/numinous_moe_tactics.py": "srt/layers/moe/numinous_moe_tactics.py",
}

# (module, attribute): everything numinous_dp_moe / numinous_moe_tactics import inside functions
LAZY = [
    ("flashinfer.fused_moe", "trtllm_fp4_block_scale_routed_moe"),
    ("flashinfer.autotuner", "AutoTuner"),
    ("flashinfer.fused_moe.core", "MoeRunnerInputs"),
    ("sglang.srt.distributed", "get_tp_group"),
    ("sglang.srt.layers.dp_attention", "get_attention_dp_rank"),
    ("sglang.srt.layers.dp_attention", "get_attention_dp_size"),
    ("sglang.srt.layers.dp_attention", "get_global_dp_buffer_len"),
    ("sglang.srt.layers.dp_attention", "get_local_dp_buffer_len"),
    ("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm", "trtllm_moe_enable_pdl"),
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_quantize"),
    ("sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe", "Mxfp4FlashinferTrtllmMoEMethod"),
    ("sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe", "RoutingMethodType"),
    ("sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe", "_MXFP8_QUANTIZE_BACKEND"),
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
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / local, dst / rel)
    shutil.copytree(ROOT / "operations/kernels/moe_tactics", dst / "srt/layers/moe/moe_tactics", dirs_exist_ok=True)
    hooks.apply(str(dst), [])
    return base


def test_every_anchor_resolves_once_on_the_installed_sglang():
    lines = hooks.apply(list(sglang_spec.submodule_search_locations)[0], [], check=True)
    assert all(line.endswith(("anchor ok", "already applied")) for line in lines), lines
    assert len(lines) == len(hooks.HOOKS)


def test_apply_is_idempotent(patched):
    again = hooks.apply(str(patched / "sglang"), [])
    assert all(line.endswith("already applied") for line in again), again


def test_patched_modules_import_in_a_clean_interpreter(patched):
    code = ("import sglang, sys; assert sglang.__file__.startswith(sys.argv[1]), sglang.__file__\n"
            "import sglang.srt.models.deepseek_v4 as m, inspect\n"
            "assert 'numinous-hook:v4_dpmoe' in inspect.getsource(m)\n"
            "import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm\n"
            "import sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe\n"
            "import sglang.srt.layers.moe.numinous_dp_moe, sglang.srt.layers.moe.numinous_moe_tactics\n"
            "print('ok')")
    env = {**os.environ, "PYTHONPATH": str(patched)}
    r = subprocess.run([sys.executable, "-c", code, str(patched)], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


def test_lazy_imports_resolve_in_server_import_order():
    """In a clean interpreter, after the model module (importing the MoE runner standalone is circular)."""
    code = ("import importlib, json, sys\n"
            "import sglang.srt.models.deepseek_v4\n"
            f"lazy = {LAZY!r}\n"
            "missing = [f'{m}.{a}' for m, a in lazy if not hasattr(importlib.import_module(m), a)]\n"
            "print(json.dumps(missing))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip().splitlines()[-1] == "[]", r.stdout[-2000:]


def test_routed_moe_takes_top_k_unpacked_or_packing_exists():
    """numinous_dp_moe passes (ids, weights) on FlashInfer >= 0.6.18 and packs with PackTopkIds before it."""
    import inspect

    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe as f

    unpacked = "Tuple" in str(inspect.signature(f).parameters["topk_ids"].annotation)
    packer = importlib.util.find_spec("sglang.kernels.ops.moe.pack_topk_ids") is not None
    assert unpacked or packer
