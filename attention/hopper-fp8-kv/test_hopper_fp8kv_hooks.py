"""CPU tests for operations/patches/hopper_fp8kv_hooks.py (rtxsys). The tree tests need a v0.5.20 source tree
(SGL_TREE, default /tmp/sgl0520/python/sglang) and are skipped without one."""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hopper_fp8kv_hooks as hk  # noqa: E402

TREE = pathlib.Path(os.environ.get("SGL_TREE", "/tmp/sgl0520/python/sglang"))
if not (TREE / hk.BACKEND).exists():   # the persistent copy (the 2026-10-05 host crash wiped /tmp)
    TREE = pathlib.Path(os.path.expanduser("~/.cache/numinous/sgl-v0520/python/sglang"))
need_tree = pytest.mark.skipif(not (TREE / hk.BACKEND).exists(), reason="no v0.5.20 tree")


def test_padded_width():
    assert hk.padded_width(2048, 4) == 2112      # GLM-5.3-Flash: 2048 + 3 tail -> next multiple of 64
    assert hk.padded_width(2048, 1) == 2048      # no kpool: unchanged
    assert hk.padded_width(1024, 2) == 1088
    assert all(hk.padded_width(2048, k) % 64 == 0 for k in range(1, 9))


def test_pad_semantics():
    """The hook pads with F.pad(value=-1) on int32 rows: dtype kept, contiguous, valid prefix untouched."""
    rows = torch.arange(2051, dtype=torch.int32).repeat(3, 1).unsqueeze(1)
    rows[1, 0, 1000:] = -1
    w = hk.padded_width(2048, 4)
    out = torch.nn.functional.pad(rows, (0, w - rows.shape[-1]), value=-1)
    assert out.dtype == torch.int32 and out.is_contiguous() and out.shape == (3, 1, w)
    assert torch.equal(out[..., :2051], rows) and (out[..., 2051:] == -1).all()


@need_tree
def test_apply_check_idempotent(tmp_path):
    dst = tmp_path / "sglang"
    shutil.copytree(TREE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    assert [ln for ln in hk.apply(str(dst), check=True) if "anchor ok" in ln].__len__() == len(hk.HOOKS)
    first = hk.apply(str(dst))
    assert sum("applied to" in ln for ln in first) == len(hk.HOOKS)
    second = hk.apply(str(dst))
    assert all("already applied" in ln for ln in second)
    src = (dst / hk.BACKEND).read_text()
    assert src.count("# numinous-hook:hopper_fp8kv_pad") == 1 and src.count("# numinous-hook:hopper_fp8kv_meta") == 1
    assert "flashmla_kv" in (dst / hk.KPOOL).read_text().split("numinous-hook:hopper_fp8kv_kpool")[1][:200]
    # inert without the env: the stock assert still guards the non-kpool path
    assert "indices.shape[-1] == self.dsa_index_topk" in src


@need_tree
def test_composes_with_rtxflash_hooks(tmp_path):
    """The Hopper Flash cell runs rtxflash_hooks.py first (FLASH_HOOKS); ours must still apply after it."""
    dst = tmp_path / "sglang"
    shutil.copytree(TREE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    r = subprocess.run([sys.executable, str(HERE / "rtxflash_hooks.py"), "--root", str(dst)], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = hk.apply(str(dst))
    assert sum("applied to" in ln for ln in lines) == len(hk.HOOKS)


@need_tree
def test_composes_with_rtxattn_engine_hooks(tmp_path):
    """The flash-h200-fp8kv container also carries rtxflash_attn_hooks.py (rtxattn's engine): same kpool anchor, both
    conditions must survive, OR-ed."""
    dst = tmp_path / "sglang"
    shutil.copytree(TREE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    r = subprocess.run([sys.executable, str(HERE / "rtxflash_attn_hooks.py"), "--root", str(dst)], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = hk.apply(str(dst))
    assert sum("applied to" in ln for ln in lines) == len(hk.HOOKS), lines
    assert not any("anchor ok" in ln for ln in lines)   # verify_registry.hook_commands refuses those
    src = (dst / hk.KPOOL).read_text()
    assert "numinous-hook:hopper_fp8kv_kpool" in src and "numinous-hook:rtxattn_kpool" in src


@need_tree
def test_slice_branch_keeps_stock_call(tmp_path):
    """The slice hook wraps the stock call: the stock (graph-captured) path is intact in the else branch, the sliced
    branch only runs with the env on, outside capture, above the row threshold."""
    dst = tmp_path / "sglang"
    shutil.copytree(TREE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    hk.apply(str(dst))
    src = (dst / hk.BACKEND).read_text()
    body = src.split("# numinous-hook:hopper_fp8kv_slice")[1].split("if target_q_heads != num_q_heads")[0]
    assert "torch.cuda.is_current_stream_capturing()" in body and "NUMINOUS_HOPPER_FP8KV_SLICE" in body
    assert "tile_scheduler_metadata=metadata.flashmla_metadata.flashmla_metadata" in body.split("else:")[-1]


def test_xform_withdrawn():
    """hopper_fp8kv_xform is out of the series: it turned an assertion into XID 31 read faults on the PD prefill
    (arm F relaunch 2, 2026-10-05). The stock 2048-only assert guards the PAGED transform again."""
    assert "hopper_fp8kv_xform" not in [h.name for h in hk.HOOKS]
    assert hk.XFORM not in hk.FILES and "hopper_fp8kv_xform" in hk.WITHDRAWN
