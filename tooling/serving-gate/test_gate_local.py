"""gate_local's engine preparation on a real box: reset, apply an arm's patch series and overlays, reset again.

Runs where sglang is an editable git checkout and numinous-kernels (with .git) is at GATE_KERNELS; skips elsewhere.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KERNELS = pathlib.Path(os.environ.get("GATE_KERNELS", "/root/numinous-kernels"))
ready = importlib.util.find_spec("sglang") is not None and (KERNELS / ".git").exists() \
    and importlib.util.find_spec("modal") is not None
pytestmark = pytest.mark.skipif(not ready, reason="needs an editable sglang checkout, numinous-kernels and modal")


@pytest.fixture(scope="module")
def gl():
    import gate_local
    return gate_local


def _dirty(repo) -> str:
    return subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "python"], capture_output=True,
                          text=True).stdout.strip()


def test_prepare_then_reset_leaves_a_clean_checkout(gl):
    from gate_arms import ARMS
    pkg = gl.engine_root()
    repo = gl.checkout_of(pkg)
    gl.reset_engine(repo)
    assert _dirty(repo) == ""
    arm = ARMS["glm_v20_k"]
    steps = gl.apply_patches(repo, pkg, arm["patches"]) + gl.apply_overlays(pkg, arm["overlay"])
    assert len(steps) == 5 + len(arm["overlay"])
    assert (pkg / "srt/managers/decode_burst.py").exists()
    assert "numinous-hook:tactics_runner" in (pkg / "srt/layers/moe/moe_runner/flashinfer_trtllm.py").read_text()
    assert (pkg / "srt/layers/moe/numinous_moe_tactics.py").exists()
    gl.reset_engine(repo)
    assert _dirty(repo) == ""
    assert not (pkg / "srt/managers/decode_burst.py").exists()


def test_pro_overlay_set_installs_the_fused_path_hooks(gl):
    from gate_arms import ARMS
    pkg = gl.engine_root()
    repo = gl.checkout_of(pkg)
    gl.reset_engine(repo)
    try:
        gl.apply_overlays(pkg, ARMS["v20_k"]["overlay"])
        src = (pkg / "srt/models/deepseek_v4.py").read_text()
        assert "numinous-hook:v4_dpmoe" in src and "numinous-hook:v4_prepare" in src
        r = subprocess.run([sys.executable, "-c", "import sglang.srt.models.deepseek_v4"], capture_output=True,
                           text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
    finally:
        gl.reset_engine(repo)
