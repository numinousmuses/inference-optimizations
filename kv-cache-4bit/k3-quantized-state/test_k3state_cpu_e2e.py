"""Two-tier KDA state (k3state) end to end vs stock SGLang on both trees: v0.5.20 (B300 image) and main 172b1b4825
(ROCm MI355X image). The runner is operations/patches/k3state_cpu_e2e.py.

Always: the recorded CPU-interpreter results in research/k3amd/e2e/*.json meet the criteria below.
K3STATE_E2E=1: re-run all six (tree x format) in the Triton interpreter (~6 min each, in parallel).
On a GPU box (V3 of research/k3amd.md) the same runner with --device cuda uses the real kernels; there the mirror
criterion tightens to the reference quantizer's bound (the interpreter's fp8 / bf16 casts round differently).

Criteria: verify outputs, committed working state, decode, prefill (fresh and resumed from the quantized pool) all
bit-identical to stock (max diff 0.0); conv states identical; the pool mirror decodes to W within the format's bound;
the ReplaySSM ring is sized by request rows (spec + 2) instead of mamba slots (size + 1).
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVIDENCE = os.path.join(ROOT, "research", "k3amd", "e2e")
# max |dequant(mirror) - W| / row amax. CPU interpreter bounds (its casts round differently) / reference bounds.
MIRROR_BOUND_INTERP = {"fp8": 0.1429 + 1e-3, "fp4": 0.1667 + 1e-3, "bf16": 0.0078 + 1e-4}
MIRROR_BOUND_HW = {"fp8": 0.0357 + 1e-3, "fp4": 0.1667 + 1e-3, "bf16": 0.0039 + 1e-4}


def check(res: dict, hw: bool = False) -> list[str]:
    bad = [k for k, v in res.items() if k.endswith("maxdiff") and v != 0.0]
    bad += [k for k, v in res.items() if k.endswith("_equal") and v is not True]
    bound = (MIRROR_BOUND_HW if hw else MIRROR_BOUND_INTERP)[res["fmt"]]
    bad += [k for k, v in res.items() if k.endswith("mirror_rel_err") and v > bound]
    if res["ring_rows"] != [11, 8]:           # pool size 10 + 1 stock; spec 6 + 2 two-tier
        bad.append(f"ring_rows {res['ring_rows']}")
    if res["pool_dtype"] != "torch.uint8" or res["engine_mode"] != res["fmt"]:
        bad.append("pool not quantized")
    return bad


def test_recorded_results():
    files = sorted(glob.glob(os.path.join(EVIDENCE, "*.json")))
    got = {(json.load(open(f))["rev"], json.load(open(f))["fmt"]) for f in files}
    assert got >= {(r, f) for r in ("172b1b4825", "v0.5.20") for f in ("fp8", "fp4", "bf16")}, got
    for f in files:
        res = json.load(open(f))
        assert check(res, hw=res.get("device") == "cuda") == [], (f, check(res))


@pytest.mark.skipif(os.environ.get("K3STATE_E2E") != "1", reason="set K3STATE_E2E=1 (~6 min per case, interpreter)")
@pytest.mark.parametrize("rev", ["172b1b4825", "v0.5.20"])
@pytest.mark.parametrize("fmt", ["fp8", "fp4", "bf16"])
def test_rerun_interpreter(rev, fmt, tmp_path):
    out = tmp_path / "r.json"
    r = subprocess.run([sys.executable, os.path.join(HERE, "k3state_cpu_e2e.py"), "--rev", rev, "--fmt", fmt,
                        "--json", str(out)], env={**os.environ, "TRITON_INTERPRET": "1"}, capture_output=True,
                       text=True, timeout=3600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert check(json.load(open(out))) == []
