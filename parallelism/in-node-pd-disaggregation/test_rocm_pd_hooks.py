"""rocm_pd_hooks + rocm_mori_xfer_stats. The anchor test needs the SGLang tree (MI355X box); the stats test runs
anywhere.

    python3 -m pytest -q operations/patches/test_rocm_pd_hooks.py
"""
import importlib.util
import os
import shutil
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import rocm_mori_xfer_stats as xs  # noqa: E402
import rocm_pd_hooks as hooks  # noqa: E402

HAVE_SGLANG = importlib.util.find_spec("sglang") is not None


@pytest.mark.skipif(not HAVE_SGLANG, reason="needs the SGLang tree")
def test_anchors_apply_idempotent(tmp_path):
    root = tmp_path / "sglang"
    shutil.copytree(hooks.sglang_root(), root, ignore=shutil.ignore_patterns("__pycache__"))
    chk = hooks.apply(str(root), check=True)
    assert chk == ["mori_xgmi: anchor ok", "mori_stats: anchor ok", "pd_draft_probs: anchor ok"], chk
    first = hooks.apply(str(root))
    assert sum("applied to" in l for l in first) == 3 and first[-1].startswith("installed"), first
    again = hooks.apply(str(root), check=True)
    assert all(l.endswith("already applied") for l in again), again
    for rel in (hooks.CONN, hooks.EAGLE, hooks.STATS_MODULE[1]):
        compile(open(root / rel).read(), rel, "exec")


def test_stats_window(monkeypatch, caplog):
    monkeypatch.setattr(xs, "_PERIOD", 0.0)
    lines = []
    monkeypatch.setattr(xs.logger, "info", lines.append)

    def one(nbytes):
        xs.chunk_begin()
        xs.planned([nbytes // 2, nbytes - nbytes // 2])
        xs.chunk_submitted()
        xs.chunk_end()

    ts = [threading.Thread(target=one, args=(10**6,)) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(lines) == 4 and all(l.startswith("[numinous mori]") for l in lines)
    assert "GB=0.00" in lines[0] and "items=2" in lines[0] and "item_KB=500.0" in lines[0]
    # planned() outside a chunk is ignored rather than raising
    xs.planned([1, 2, 3])
