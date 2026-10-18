"""CPU tests of the rtxsys D2-b hook: it applies to SGLang v0.5.20 frozen_kv_mtp_worker_v2.py (anchors once,
compiles, idempotent, installs the module), and the preflight flags inert setups."""
from __future__ import annotations

import os
import shutil
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gemma31_draft_top1_hooks as dh  # noqa: E402

SRC = os.path.expanduser("~/.cache/numinous/sgl-v0520/python/sglang")
need_tree = pytest.mark.skipif(not os.path.isdir(SRC), reason="needs operations/hopper/restore_cpu_preflight.sh")


@need_tree
def test_hook_applies_installs_and_is_idempotent(tmp_path):
    root = tmp_path / "sglang"
    for sub in ("srt/speculative", "srt/layers"):
        shutil.copytree(os.path.join(SRC, sub), root / sub, ignore=shutil.ignore_patterns("__pycache__"))
    for sub in ("srt/models",):
        shutil.copytree(os.path.join(SRC, sub), root / sub, ignore=shutil.ignore_patterns("__pycache__"))
    out = dh.apply(str(root))
    assert out == ["installed srt/layers/numinous_draft_top1.py", f"seed: applied to {dh.PATH}",
                   f"step: applied to {dh.PATH}", f"head: applied to {dh.MTP}"]
    assert (root / "srt/layers/numinous_draft_top1.py").exists()
    again = dh.apply(str(root))
    assert again[1:] == ["seed: already applied", "step: already applied", "head: already applied"]
    src = (root / dh.PATH).read_text()
    assert src.count(dh.MARKER) == 2
    compile(src, "frozen_kv_mtp_worker_v2.py", "exec")
    mtp = (root / dh.MTP).read_text()
    assert mtp.count(dh.MARKER) == 1
    compile(mtp, "gemma4_mtp.py", "exec")
    # the seed hook's import precedes the step hook's use of _ndt in the same function
    assert src.index("(seed)") < src.index("(step)")


def test_preflight_flags():
    ctx = types.SimpleNamespace(env={dh.ENV: "1"}, args={"speculative_algorithm": "EAGLE3", "speculative_eagle_topk": 4})
    out = dh.numinous_preflight(ctx)
    assert any("frozen-KV MTP draft only" in x for x in out) and any("topk == 1 only" in x for x in out)
    ctx.args = {"speculative_algorithm": "NEXTN", "speculative_eagle_topk": 1}
    assert dh.numinous_preflight(ctx) == []
    ctx.env = {dh.ENV: "1", "NUMINOUS_DRAFT_HEAD_FP8": "1"}
    assert any("without NUMINOUS_DRAFT_HEAD_FUSED" in x for x in dh.numinous_preflight(ctx))
    ctx.env = {"NUMINOUS_DRAFT_HEAD_FUSED": "1"}
    assert any("needs NUMINOUS_DRAFT_TOP1" in x for x in dh.numinous_preflight(ctx))
    ctx.env = {}
    assert dh.numinous_preflight(ctx) == []
