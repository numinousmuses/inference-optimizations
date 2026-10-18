"""CPU tests for marlin_ep_block_hooks. Tree: $SGLANG_SRC or ~/.cache/numinous/sgl-v0520/python."""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import textwrap

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marlin_ep_block_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", pathlib.Path.home() / ".cache/numinous/sgl-v0520/python")) / "sglang"


def _patched(tmp_path):
    if not (SRC / H.PATH).is_file():
        pytest.skip(f"no SGLang tree at {SRC}")
    root = tmp_path / "sglang"
    (root / H.PATH).parent.mkdir(parents=True)
    shutil.copy(SRC / H.PATH, root / H.PATH)
    assert H.apply(str(root), check=True) == [f"{H.NAME}: anchor ok in {H.PATH}"]
    assert H.apply(str(root)) == [f"{H.NAME}: applied to {H.PATH}"]
    assert H.apply(str(root)) == [f"{H.NAME}: already applied"]
    return (root / H.PATH).read_text()


def _pick(text, M, topk, E, global_num_experts, ep, env):
    """Run the patched block-pick lines exactly as they sit in fused_marlin_moe."""
    i = text.index(H.TAG)
    j = text.index("            break\n", i) + len("            break\n")
    body = textwrap.dedent(text[text.rfind("\n", 0, i) + 1:j])
    os.environ["NUMINOUS_MARLIN_EP_BLOCK"] = env
    ns = {"M": M, "topk": topk, "E": E, "global_num_experts": global_num_experts,
          "expert_map": object() if ep else None}
    exec(compile(body, "pick", "exec"), ns)
    return ns["block_size_m"]


def test_anchor_sits_before_the_align_and_compiles(tmp_path):
    text = _patched(tmp_path)
    fn = text[text.index("def fused_marlin_moe("):]
    assert fn.index(H.TAG) < fn.index("moe_align_block_size(") and fn.index(H.TAG) < fn.index(
        "if global_num_experts == -1:")


def test_ep_uses_the_global_count_and_matches_the_measured_best(tmp_path):
    text = _patched(tmp_path)
    # GLM-5.3 EP8: 32 local of 256; measured best block per m (moe_decode_bench_h200.json)
    for m, best in ((48, 8), (96, 8), (144, 8), (192, 8), (288, 16), (576, 32)):
        assert _pick(text, m, 8, 32, 256, ep=True, env="1") == best, m
    # EP4: 64 local of 256
    for m, best in ((144, 8), (288, 16), (576, 32)):
        assert _pick(text, m, 8, 64, 256, ep=True, env="1") == best, m
    # stock (env unset) reproduces the measured over-pick
    assert _pick(text, 144, 8, 32, 256, ep=True, env="0") == 48
    assert _pick(text, 576, 8, 32, 256, ep=True, env="0") == 64


def test_tp_and_unset_global_keep_the_stock_pick(tmp_path):
    text = _patched(tmp_path)
    for m in (24, 96, 192, 288, 480, 576):                          # Flash TP4: 288 experts, no expert map
        assert _pick(text, m, 8, 288, -1, ep=False, env="1") == _pick(text, m, 8, 288, -1, ep=False, env="0")
    assert _pick(text, 144, 8, 32, -1, ep=True, env="1") == 48     # global count unknown (-1): stock
