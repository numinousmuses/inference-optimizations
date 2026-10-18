"""rocm_quark_fused_hooks: anchors resolve on the SGLang v0.5.20 source, the patched files parse, re-application is a
no-op, and a missing anchor fails loudly. The box run is the functional proof (GLM-5.3 Quark MXFP4 on 8x MI355X:
batch-1 greedy correct, EAGLE accept 2.87 / 2.88 / 2.76 at 1 / 4 / 8 running requests; agents/status/amd.md)."""
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import rocm_quark_fused_hooks as hooks  # noqa: E402

FORK = "/home/ubuntu/sglang-fork"


@pytest.fixture(scope="module")
def v0520_root(tmp_path_factory):
    if not os.path.exists(os.path.join(FORK, ".git")):
        pytest.skip("sglang fork not available")
    out = tmp_path_factory.mktemp("sgl")
    archive = subprocess.run(["git", "-C", FORK, "archive", "v0.5.20", "python/sglang"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(out)], input=archive.stdout, check=True)
    return str(out / "python" / "sglang")


def test_check_resolves_every_anchor(v0520_root):
    lines = hooks.apply(v0520_root, [], check=True)
    assert len(lines) == len(hooks.HOOKS)
    assert all(line.endswith("anchor ok") for line in lines)


def test_apply_parses_and_is_idempotent(v0520_root, tmp_path):
    root = str(tmp_path / "sglang")
    shutil.copytree(v0520_root, root)
    first = hooks.apply(root, [])
    assert all("applied to" in line for line in first)
    for h in hooks.HOOKS:
        src = open(os.path.join(root, h.path)).read()
        assert h.marker in src
        compile(src, h.path, "exec")
    second = hooks.apply(root, [])
    assert all(line.endswith("already applied") for line in second)


def test_missing_anchor_fails_loudly(v0520_root, tmp_path):
    root = str(tmp_path / "sglang")
    shutil.copytree(v0520_root, root)
    h = hooks.HOOKS[0]
    path = os.path.join(root, h.path)
    open(path, "w").write(open(path).read().replace(h.anchor, "# anchor removed\n"))
    with pytest.raises(SystemExit):
        hooks.apply(root, [h.name])
