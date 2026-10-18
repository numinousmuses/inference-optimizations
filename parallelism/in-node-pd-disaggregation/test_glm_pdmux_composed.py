"""Composition test (final exam, amdglm): the DP-lockstep engine fix (pdmux under DP attention) on the FULL composed
GLM-5.3 tree of arm GX0 (glm_hopper_cells.arm_gx0: M6 + FP8 passthrough + lazy dequant + w4a8dg + pdmux; every hook
applied exactly as the boot applies them, hopper_cells.patched_src):
  - every S1-S5 edit and rtxv41's lane hooks are present together with every M6 hook, and the patched files compile;
  - the installed numinous_pdmux_dp module is this repo's file;
  - the composed DeepseekV2ForCausalLM.forward_split_prefill still reproduces the one-pass layer loop over splits
    (fake layers; the other hooks' edits to deepseek_v2.py included);
  - attach-time: glm_pdmux_dp_hooks.numinous_preflight is clean on GX0's own flags and env, and refuses GX1's (pacing
    + mixed chunk + delayer + overlap)."""
from __future__ import annotations

import functools
import pathlib
import py_compile
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hopper"))
import glm_pdmux_dp_hooks as D  # noqa: E402
import glm_pdmux_split_hooks as S  # noqa: E402
import test_glm_pdmux_split_hooks as TS  # noqa: E402


@functools.lru_cache(maxsize=1)
def _composed():
    import glm_hopper_cells as g
    import hopper_cells as hc
    if not pathlib.Path(hc.SGL_SRC).is_dir():
        pytest.skip("needs operations/hopper/restore_cpu_preflight.sh (CPU v0.5.20 tree)")
    cell = g.arm_gx0()
    root = pathlib.Path(hc.patched_src(cell["config"])) / "sglang"
    return cell, root


def _args(flags: list) -> types.SimpleNamespace:
    def has(f):
        return f in flags

    def val(f, d=None):
        return flags[flags.index(f) + 1] if f in flags else d
    return types.SimpleNamespace(
        enable_pdmux=has("--enable-pdmux"), enable_dp_attention=has("--enable-dp-attention"),
        disable_overlap_schedule=has("--disable-overlap-schedule"), enable_mixed_chunk=has("--enable-mixed-chunk"),
        enable_prefill_delayer=has("--enable-prefill-delayer"), chunked_prefill_size=int(val("--chunked-prefill-size", -1)),
        enable_two_batch_overlap=has("--enable-two-batch-overlap"), pp_size=1)


def test_every_edit_present_and_compiles():
    cell, root = _composed()
    runs = [r[0] for r in cell["config"]["hooks"]["run"]]
    for h in ("operations/patches/glm_pdmux_lane_hooks.py", "operations/patches/glm_pdmux_split_hooks.py",
              "operations/patches/glm_pdmux_dp_hooks.py", "operations/patches/glm_absorb_q_fold_hooks.py",
              "operations/patches/glm_w4a8_tiles_hooks.py", "operations/patches/glm_kv4_sm90_hooks.py"):
        assert h in runs, h
    for _name, rel, _anchor, text, _already in D.HOOKS:
        assert text in (root / rel).read_text() or (_already and _already in (root / rel).read_text()), _name
    assert S.SPLIT_TEXT in (root / S.DSV2).read_text()
    assert "_numinous_dp_execute_extend" in (root / D.ER).read_text()
    assert "numinous" in (root / "srt/distributed/parallel_state.py").read_text()       # rtxv41's lane twins
    for rel in (D.MIX, D.VAL, D.TW, D.MR, D.ER, D.MODULE_DST, S.DSV2, "srt/distributed/parallel_state.py"):
        py_compile.compile(str(root / rel), doraise=True)
    assert (root / D.MODULE_DST).read_text() == D.MODULE_SRC.read_text()


def test_composed_split_equals_one_pass():
    torch = pytest.importorskip("torch")
    _cell, root = _composed()
    ns = TS._namespace(torch)
    exec(TS._method_source(root), ns)
    fn = ns["forward_split_prefill"]
    n = 9
    ids = torch.arange(5)
    ref = fn(TS._fake_self(torch, TS._fake_model(torch, n), n), ids, ids, TS._fb(), (0, n))
    s = TS._fake_self(torch, TS._fake_model(torch, n), n)
    fb = TS._fb()
    outs = [fn(s, ids, ids, fb, r) for r in ((0, 4), (4, 8), (8, 9))]
    assert outs[0] is None and outs[1] is None and torch.equal(outs[2][1], ref[1])


def test_attach_preflight_on_the_final_cells():
    import glm_hopper_cells as g
    cell, _root = _composed()
    cfg = cell["config"]
    assert D.numinous_preflight(types.SimpleNamespace(env=cfg["env"], args=_args(cfg["flags"]))) == []
    gx1 = next(c for c in g.final_chain() if c["key"].endswith("-GX1"))["config"]
    probs = D.numinous_preflight(types.SimpleNamespace(env={**gx1["env"], D.ENV: "1"}, args=_args(gx1["flags"])))
    assert len(probs) >= 3          # no pdmux flag, overlap on, mixed chunk / delayer / pacing on
