"""Box tests for deepgemm_megamoe_hooks.py (skip without an installed sglang).

1. Every anchor resolves exactly once on the installed SGLang; apply is idempotent.
2. The patched mega_moe.py imports in a clean interpreter and `_numinous_megamoe_balanced_rows` returns the
   documented row counts (off by default; capture mode, CP prefill, unknown or mismatched DP counts -> own count;
   otherwise max(own, ceil(mean)), capped).
3. With 8 GPUs: the bench drives the kernel with the hook's padding on a mixed step and checks the output of every
   rank against the dense reference (numerical equivalence of the padded run) and that it is not slower.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import deepgemm_megamoe_hooks as hooks  # noqa: E402

sglang_spec = importlib.util.find_spec("sglang")
pytestmark = pytest.mark.skipif(sglang_spec is None, reason="needs an installed sglang (dev box)")


@pytest.fixture(scope="module")
def patched(tmp_path_factory) -> pathlib.Path:
    src = list(sglang_spec.submodule_search_locations)[0]
    base = tmp_path_factory.mktemp("sgl")
    shutil.copytree(src, base / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
    hooks.apply(str(base / "sglang"))
    return base


def test_every_anchor_resolves_once_on_the_installed_sglang():
    lines = hooks.apply(list(sglang_spec.submodule_search_locations)[0], check=True)
    assert all(line.endswith(("anchor ok", "already applied")) for line in lines), lines
    assert len(lines) == len(hooks.HOOKS)


def test_apply_is_idempotent(patched):
    again = hooks.apply(str(patched / "sglang"))
    assert all(line.endswith("already applied") for line in again), again


HELPER_CHECK = r'''
import sys, types
import sglang
assert sglang.__file__.startswith(sys.argv[1]), sglang.__file__
import sglang.srt.layers.moe.mega_moe as m
src = open(m.__file__).read()
assert src.count("# numinous-hook:megamoe_balance\n") == 1
# the call sits after SGLang's y allocation and before the kernel call
assert src.index("_numinous_megamoe_balanced_rows(num_tokens") < src.index("deep_gemm.fp8_fp4_mega_moe(")
g = types.SimpleNamespace(size=lambda: 8)
state = {"counts": [4096, 0, 0, 0, 0, 0, 0, 0], "capture": False, "cp": False}
m.get_dp_global_num_tokens = lambda: state["counts"]
m.get_is_capture_mode = lambda: state["capture"]
m.is_dsa_enable_prefill_cp = lambda: state["cp"]
f = m._numinous_megamoe_balanced_rows
m._NUMINOUS_MEGAMOE_BALANCE = False
assert f(0, 4096, g) == 0                                   # off
m._NUMINOUS_MEGAMOE_BALANCE = True
assert f(0, 4096, g) == 512                                 # ceil(4096 / 8)
assert f(4096, 4096, g) == 4096                             # never below own
state["counts"] = [4096, 96, 96, 96, 96, 96, 96, 96]
assert f(96, 4096, g) == 596                                # ceil(4768 / 8)
assert f(96, 300, g) == 300                                 # capped at the symmetric buffer size
state["capture"] = True
assert f(96, 4096, g) == 96                                 # CUDA-graph capture: untouched
state["capture"] = False; state["cp"] = True
assert f(96, 4096, g) == 96                                 # CP prefill: untouched
state["cp"] = False; state["counts"] = None
assert f(96, 4096, g) == 96                                 # counts unknown
state["counts"] = [10, 20, 30, 40]
assert f(10, 4096, g) == 10                                 # DP group != EP group
print("ok")
'''


def test_patched_helper_in_a_clean_interpreter(patched):
    env = dict(os.environ, PYTHONPATH=str(patched))
    r = subprocess.run([sys.executable, "-c", HELPER_CHECK, str(patched)], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-2000:] + r.stderr[-4000:]


def _gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 0


@pytest.mark.skipif(_gpus() < 8, reason="needs 8 GPUs")
def test_padded_mixed_step_matches_reference_and_is_not_slower(tmp_path):
    rows = {}
    for policy in ("none", "mean"):
        out = tmp_path / f"{policy}.json"
        cmd = ["torchrun", "--nproc-per-node", "8", str(ROOT / "operations/kernels/megamoe_bench.py"),
               "--mma", "fp8xfp4", "--tokens-by-rank", "4096,0,0,0,96,96,96,96", "--pad-policy", policy,
               "--graph-steps", "3", "--replays", "5", "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        rows[policy] = json.load(open(out))["results"][0]
    for policy, row in rows.items():
        assert row["diff_max"] is not None and 0 <= row["diff_max"] < 1e-3, (policy, row)
    assert rows["mean"]["block_m_by_rank"][1] > rows["none"]["block_m_by_rank"][1]
    assert rows["mean"]["total_us"] < rows["none"]["total_us"] * 1.02, rows
