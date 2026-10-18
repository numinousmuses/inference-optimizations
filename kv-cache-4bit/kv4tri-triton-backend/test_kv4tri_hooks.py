"""Box test for kv4tri_hooks.py (needs the SGLang v0.5.20 image + a GPU for the glue check).

  1. check mode: every anchor resolves exactly once in the installed package;
  2. a full copy of the installed sglang package is patched (twice: the second run must be a no-op), every patched
     module imports from the copy, and the installed modules are untouched;
  3. operations/kernels/kv4tri_glue_check.py runs against the patched copy: real HybridLinearKVPool / SWAKVPool with
     NVFP4, the real TritonAttnBackend.forward_decode / forward_extend with the hooks, float references.

    python3 operations/devbox/devbox.py test rtx-qwen38 operations/patches/test_kv4tri_hooks.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _hooks():
    spec = importlib.util.spec_from_file_location("kv4tri_hooks", os.path.join(_HERE, "kv4tri_hooks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _installed_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("sglang not installed")
    return list(spec.submodule_search_locations)[0]


def test_check_mode_anchors_resolve():
    lines = _hooks().apply(_installed_root(), check=True)
    assert len(lines) == len(_hooks().HOOKS)
    assert all(line.endswith("anchor ok") or line.endswith("already applied") for line in lines), lines


@pytest.fixture(scope="module")
def patched_copy():
    root = _installed_root()
    tmp = tempfile.mkdtemp(prefix="kv4tri_sgl_")
    dst = os.path.join(tmp, "sglang")
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
    h = _hooks()
    h.install_modules(dst, check=False)
    first = h.apply(dst)
    second = h.apply(dst)
    yield tmp, first, second
    shutil.rmtree(tmp, ignore_errors=True)


def test_apply_is_complete_and_idempotent(patched_copy):
    _, first, second = patched_copy
    assert all(": applied to " in line for line in first), first
    assert all(line.endswith("already applied") for line in second), second


def test_installed_package_untouched():
    root = _installed_root()
    for h in _hooks().HOOKS:
        with open(os.path.join(root, h["path"])) as f:
            assert f"# numinous-hook:{h['name']}\n" not in f.read(), f"{h['name']} found in the installed package"


def _run_in_copy(tmp, args, timeout=900):
    env = dict(os.environ, PYTHONPATH=tmp + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True, timeout=timeout,
                          cwd=_REPO)


def test_patched_modules_import(patched_copy):
    tmp, _, _ = patched_copy
    code = ("import sglang, os\n"
            "assert sglang.__file__.startswith(%r), sglang.__file__\n"
            "import sglang.srt.layers.attention.triton_backend as tb\n"
            "import sglang.srt.mem_cache.memory_pool, sglang.srt.mem_cache.swa_memory_pool\n"
            "import sglang.srt.mem_cache.kv_cache_configurator, sglang.srt.model_executor.pool_configurator\n"
            "import sglang.srt.layers.attention.numinous_kv4tri as nk4\n"
            "assert nk4._k.__name__ == 'sglang.kernels.ops.attention.numinous_kv4tri_attention', nk4._k.__name__\n"
            "import inspect\n"
            "assert 'numinous-hook:triton_decode' in inspect.getsource(tb.TritonAttnBackend.forward_decode)\n"
            "assert 'numinous-hook:triton_extend' in inspect.getsource(tb.TritonAttnBackend.forward_extend)\n"
            "print('ok')\n") % tmp
    r = _run_in_copy(tmp, ["-c", code], timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


def test_glue_end_to_end_on_gpu(patched_copy):
    tmp, _, _ = patched_copy
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("needs a GPU")
    except ImportError:
        pytest.skip("needs torch")
    r = _run_in_copy(tmp, [os.path.join(_REPO, "operations/kernels/kv4tri_glue_check.py")])
    rows = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    bad = [row for row in rows if not row["ok"]]
    assert r.returncode == 0 and rows and not bad, (bad, r.stderr[-3000:])
