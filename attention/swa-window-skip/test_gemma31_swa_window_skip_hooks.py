"""SWA window loop-start: the skipped tiles are exactly tiles with no visible key for any row (so the output is
bit-identical), anchors/compile, and composition with rtxv41's gemma_extend hooks on the same file."""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gemma31_swa_window_skip_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python/sglang"))


@pytest.mark.parametrize("prefix,w,bm,bn", [(0, 1024, 64, 32), (3000, 1024, 64, 64), (100, 16, 16, 16),
                                            (0, 7, 32, 16), (511, 512, 128, 32)])
def test_skipped_tiles_have_no_visible_key(prefix, w, bm, bn):
    ext = 2048
    for block_m in range(0, ext // bm):
        q_rows = [prefix + block_m * bm + m for m in range(bm)]
        # prefix loop (kv_id = position in the prefix list, q_id = prefix + extend row)
        lo = max(0, prefix + block_m * bm - w) // bn * bn
        for kv in range(0, min(lo, prefix)):
            assert not any(q <= kv + w for q in q_rows), (prefix, block_m, kv)
        assert lo == 0 or any(q <= lo + bn - 1 + w for q in q_rows) or lo >= prefix
        # extend loop (q_id = extend row, kv_id = extend position)
        lo_e = max(0, block_m * bm - w) // bn * bn
        for kv in range(0, lo_e):
            assert not any(block_m * bm + m <= kv + w for m in range(bm)), (block_m, kv)


def test_apply_compile_and_compose(tmp_path):
    if not (SRC / H.REL).exists():
        pytest.skip("SGLang v0.5.20 source absent")
    root = tmp_path / "sglang"
    (root / H.REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / H.REL, root / H.REL)
    for hook in ("gemma_extend_hooks.py", "gemma_extend_sm90_hooks.py"):
        if (HERE / hook).exists():
            r = subprocess.run([sys.executable, str(HERE / hook), "--root", str(root)], capture_output=True,
                               text=True, cwd=HERE.parents[1])
            assert r.returncode == 0, (hook, r.stderr[-600:])
    assert all("anchor ok" in x for x in H.apply(str(root), check=True))
    assert all("applied to" in x for x in H.apply(str(root)))
    assert all("already applied" in x for x in H.apply(str(root)))
    text = (root / H.REL).read_text()
    compile(text, H.REL, "exec")
    assert "for start_n in range(prefix_lo, prefix_end, BLOCK_N_PREFIX):" in text
    assert "for start_n in range(extend_lo, extend_end, BLOCK_N):" in text
