"""Composition test: glm_pdmux_lane_hooks on top of the full GLM-5.3 M6 hook stack (agent rtxv41).

Builds the tree the cell boots (hopper_cells.patched_src: the hopper-glm-5.3 series, then every M6 hook in order at its
pinned ref, then this hook) and checks:
  - every hook still applies (patched_src raises otherwise) and every patched file compiles;
  - the lane hook landed, and re-running it on the composed tree is a no-op;
  - the module-level buffer dicts the lane switch swaps exist under those exact names in the composed tree. A rename in
    stock or in rtxattn's router piggyback would otherwise make the swap silently skip them;
  - the attach-time preflight is clean for the M6 server args with --enable-pdmux.
    pytest -p no:cacheprovider operations/patches/test_glm_pdmux_lane_composition.py      (~1-2 min: builds the tree)
"""
from __future__ import annotations

import ast
import copy
import pathlib
import subprocess
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "operations" / "hopper"))
import glm_pdmux_lane_hooks as hk  # noqa: E402

HOOK = "operations/patches/glm_pdmux_lane_hooks.py"


def _hook_ref() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", HOOK], capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture(scope="module")
def composed():
    hc = pytest.importorskip("hopper_cells")
    if not pathlib.Path(hc.SGL_SRC).exists():
        pytest.skip(f"stock v0.5.20 tree missing: {hc.SGL_SRC}")
    import glm_hopper_cells as g
    cell = g.arm_m6()
    cfg = copy.deepcopy(cell["config"])
    h = cfg["hooks"]
    ref = _hook_ref()
    # the M6 files must exist at the lane hook's ref (it is newer than M6's pin)
    for f in h["files"]:
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{ref}:{f}"], check=True)
    cfg["hooks"] = hc.hooks_block(h["run"] + [[HOOK]], h["files"] + [HOOK], h["series"], h.get("install"))
    cfg["hooks"]["ref"] = ref
    tree = pathlib.Path(hc.patched_src(cfg)) / "sglang"
    return types.SimpleNamespace(tree=tree, cell=cell, cfg=cfg)


def test_lane_hook_applied_and_idempotent(composed):
    for rel in (hk.PS, hk.W4):
        f = composed.tree / rel
        src = f.read_text()
        assert hk.MARKER in src, rel
        compile(src, str(f), "exec")
    w4 = (composed.tree / hk.W4).read_text()
    assert "numinous-hook:glm_w4a8_tiles" in w4          # k3amd's tiles hook and this one share w4afp8.py
    assert "*_numinous_w4a8_lane_meta(self)," in w4
    assert all("already applied" in m for m in hk.apply(str(composed.tree), check=True))


def test_w4a8_metadata_flows_into_every_kernel_of_the_composed_runner(composed):
    """The composed cutlass_w4a8_moe (src2dst + tiles + w4a8dg + shared post-reorder hooks) must take the routing
    metadata only from its arguments, so the per-lane triple reaches every GEMM."""
    src = (composed.tree / "srt/layers/moe/cutlass_w4a8_moe.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cutlass_w4a8_moe")
    params = {a.arg for a in fn.args.args}
    assert {"expert_offsets", "problem_sizes1", "problem_sizes2"} <= params
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute) and n.attr in ("expert_offsets", "problem_sizes1", "problem_sizes2"):
            raise AssertionError(f"composed runner reads .{n.attr} from an object, not its argument")


def test_every_patched_file_compiles(composed):
    marked = [p for p in composed.tree.rglob("*.py") if "numinous" in p.name or "numinous-hook" in p.read_text(
        errors="ignore")]
    assert len(marked) >= 10, len(marked)
    for p in marked:
        compile(p.read_text(), str(p), "exec")


def _module_dicts(path: pathlib.Path) -> set:
    out = set()
    for n in ast.parse(path.read_text()).body:
        tgt, val = None, None
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            tgt, val = n.targets[0], n.value
        elif isinstance(n, ast.AnnAssign):
            tgt, val = n.target, n.value
        if isinstance(tgt, ast.Name) and isinstance(val, ast.Dict):
            out.add(tgt.id)
    return out


def test_swapped_dicts_exist_in_the_composed_tree(composed):
    for mod, attr in _lane_dicts():
        rel = mod.removeprefix("sglang.").replace(".", "/") + ".py"
        path = composed.tree / rel
        assert path.exists(), f"{mod} is not in the composed tree ({rel})"
        assert attr in _module_dicts(path), f"{mod}.{attr} is no longer a module-level dict"


def _lane_dicts():
    tree = ast.parse(hk.TAIL)
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_NUMINOUS_LANE_DICTS":
            return ast.literal_eval(n.value)
    raise AssertionError("_NUMINOUS_LANE_DICTS not found in the hook tail")


def test_preflight_clean_for_m6_with_pdmux(composed):
    args = {"tp_size": 8, "dp_size": 8, "enable_dp_attention": True, "ep_size": 8, "moe_a2a_backend": "none",
            "pp_size": 1, "enable_two_batch_overlap": False, "enable_pdmux": True}
    flags = composed.cfg["flags"]
    assert flags[flags.index("--ep-size") + 1] == "8" and flags[flags.index("--dp") + 1] == "8"
    assert hk.numinous_preflight(types.SimpleNamespace(args=args, env=dict(composed.cfg["env"]))) == []
