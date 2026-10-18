"""Box test for glm_tp1_shared_hooks.py: apply the TP1 shared-expert fix to a copy of the installed SGLang and check
the result; plus a CPU simulation of the reduction semantics the fix relies on.

    python3 -m pytest operations/patches/test_glm_tp1_shared_hooks.py -q     # on a box with SGLang installed
    (the semantics test runs anywhere; the patch tests skip without an installed sglang)
"""
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import glm_tp1_shared_hooks as hooks  # noqa: E402


def _sglang_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("no installed sglang")
    return pathlib.Path(list(spec.submodule_search_locations)[0])


@pytest.fixture()
def pkg_copy(tmp_path):
    src = _sglang_root() / "srt/models/deepseek_v2.py"
    dst = tmp_path / "srt/models/deepseek_v2.py"
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)
    return tmp_path


def test_apply_replaces_both_sites_and_compiles(pkg_copy):
    out = hooks.apply(str(pkg_copy))
    assert sorted(o.split(":")[0] for o in out) == ["tp1_shared_once_dual_stream", "tp1_shared_once_normal"]
    text = (pkg_copy / "srt/models/deepseek_v2.py").read_text()
    assert text.count(hooks.MARKER) == 2
    # the unguarded adds are gone from both sites
    assert "        if self._shared_expert_tp1:\n            final_hidden_states += shared_output\n" not in text
    assert ("        if shared_output is not None and self._shared_expert_tp1:\n"
            "            final_hidden_states += shared_output\n") not in text
    assert text.count("get_parallel().tp_rank != 0") == 2
    compile(text, "deepseek_v2.py", "exec")


def test_apply_is_idempotent(pkg_copy):
    hooks.apply(str(pkg_copy))
    before = (pkg_copy / "srt/models/deepseek_v2.py").read_text()
    out = hooks.apply(str(pkg_copy))
    assert all("already applied" in o for o in out)
    assert (pkg_copy / "srt/models/deepseek_v2.py").read_text() == before


def test_missing_anchor_fails_loudly(tmp_path):
    bad = tmp_path / "srt/models/deepseek_v2.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("def forward():\n    pass\n")
    with pytest.raises(SystemExit):
        hooks.apply(str(tmp_path))


def test_cli_check_mode_changes_nothing(pkg_copy):
    before = (pkg_copy / "srt/models/deepseek_v2.py").read_text()
    r = subprocess.run([sys.executable, str(HERE / "glm_tp1_shared_hooks.py"), "--root", str(pkg_copy), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "anchor ok" in r.stdout
    assert (pkg_copy / "srt/models/deepseek_v2.py").read_text() == before


def test_reduction_semantics_rank0_add_is_exact():
    """The fix's arithmetic: partials + replicated shared added on rank 0 only, then the deferred reduction (sum
    over ranks; scatter is row-selection and does not change per-row sums) equals full routed sum + shared once.
    The stock every-rank add counts shared tp_size times."""
    torch.manual_seed(0)
    world, rows, hidden = 8, 16, 32
    partials = [torch.randn(rows, hidden, dtype=torch.float64) for _ in range(world)]
    shared = torch.randn(rows, hidden, dtype=torch.float64)
    want = sum(partials) + shared
    fixed = sum(p + (shared if r == 0 else 0) for r, p in enumerate(partials))
    stock = sum(p + shared for p in partials)
    torch.testing.assert_close(fixed, want)
    assert (stock - want).abs().max() > 1.0                  # the bug: shared counted world times
    torch.testing.assert_close(stock, want + (world - 1) * shared)
