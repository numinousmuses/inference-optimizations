"""Anchored hooks: GLM-5.3-Flash (Glm5Next) mHC TileLang JIT stall fix + per-step scheduler trace, SGLang v0.5.20.

    python3 rtx_mhc_jit_hooks.py --root /sgl-workspace/sglang/python/sglang          # apply
    python3 rtx_mhc_jit_hooks.py --root /tmp/sgl-v0520-wt/python/sglang --check      # verify anchors only

Measured cause (rtxsys run 8, analysis research/gpu-evidence/rtxv41_glmstall/README.md): each 11.6 s global
prefill stall was one rank compiling mhc_pre_big_fuse_with_norm_tilelang for a new n_splits bucket. Policy and
rationale: operations/kernels/numinous_mhc_jit.py.

Hooks (all inert unless their env is set):
  glm5next_mhc_prewarm  models/glm5_next.py Glm5NextForConditionalGeneration.load_weights: right after its
                        `if run_post:` call of the mixin's post_load_weights (the real load path), prewarm every
                        n_splits bucket once per process (NUMINOUS_MHC_PREWARM=1).
  mhc_nsplit_pow2       kernels/ops/layernorm/mhc.py _compute_num_split_for_mhc_pre: round the split count down
                        to a power of two (NUMINOUS_MHC_NSPLIT_POW2=1): ~24 -> 6-7 compile buckets on 188 SMs.
  step_trace            managers/scheduler.py (appended): wrap the scheduler loop phases and write per-step host
                        timestamps per DP rank (NUMINOUS_STEP_TRACE=<dir>).
Modules installed: srt/layers/numinous_mhc_jit.py, srt/managers/numinous_step_trace.py (copies of
operations/kernels/numinous_mhc_jit.py and numinous_step_trace.py).

Anchors verified on v0.5.20 94602c9 with and without the numinous-kernels burst series 0001-0005 (which edits
scheduler.py; this file only appends to it).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(HERE, "..", "kernels")
MODULES = {
    "srt/layers/numinous_mhc_jit.py": os.path.join(KERNELS, "numinous_mhc_jit.py"),
    "srt/managers/numinous_step_trace.py": os.path.join(KERNELS, "numinous_step_trace.py"),
}

PREWARM_TEXT = '''        # numinous-hook:glm5next_mhc_prewarm -- compile every mHC n_splits bucket at load (rtxv41)
        import os as _numinous_os

        if not is_nextn and _numinous_os.environ.get("NUMINOUS_MHC_PREWARM", "0") == "1":
            from sglang.srt.layers import numinous_mhc_jit as _numinous_mhc_jit

            _numinous_mhc_jit.prewarm_glm5next(self)
'''

NSPLIT_OLD = "    return max(1, min(n_sms // max(grid_size, 1), num_block_k // 4))\n"
NSPLIT_NEW = '''    # numinous-hook:mhc_nsplit_pow2 -- power-of-two split buckets (rtxv41): ~24 -> 7 TileLang compiles
    _n = max(1, min(n_sms // max(grid_size, 1), num_block_k // 4))
    import os as _numinous_os

    if _numinous_os.environ.get("NUMINOUS_MHC_NSPLIT_POW2", "0") == "1":
        _n = 1 << (_n.bit_length() - 1)
    return _n
'''

TRACE_TEXT = '''

# numinous-hook:step_trace -- per-step scheduler host timestamps (rtxv41; NUMINOUS_STEP_TRACE=<dir>)
if os.environ.get("NUMINOUS_STEP_TRACE"):
    from sglang.srt.managers import numinous_step_trace as _numinous_step_trace

    _numinous_step_trace.install(Scheduler)
'''


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str
    mode: str = "after"  # after | replace | append

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


HOOKS = [
    # Anchored in load_weights, which calls the MIXIN's post_load_weights directly (glm5_next.py `if run_post:`):
    # Glm5NextForConditionalGeneration.post_load_weights itself is NOT on the load path (measured: the first anchor,
    # in that method, never ran in rtxsys's composed run g156o, 2026-10-01).
    Hook("glm5next_mhc_prewarm", "srt/models/glm5_next.py",
         "        if run_post:\n"
         "            DeepseekV2WeightLoaderMixin.post_load_weights(\n"
         "                self, is_nextn=is_nextn, weight_names=weight_names\n"
         "            )\n", PREWARM_TEXT),
    Hook("mhc_nsplit_pow2", "kernels/ops/layernorm/mhc.py", NSPLIT_OLD, NSPLIT_NEW, "replace"),
    Hook("step_trace", "srt/managers/scheduler.py", "\nclass Scheduler(", TRACE_TEXT, "append"),
]


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def _write(path: str, text: str) -> None:
    compile(text, path, "exec")
    tmp = f"{path}.numinous.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def apply(root: str, check: bool = False) -> list[str]:
    out, missing, pending = [], [], {}
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = pending.get(path) or open(path).read()
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        if h.mode == "after":
            new = src.replace(h.anchor, h.anchor + h.text, 1)
        elif h.mode == "replace":
            new = src.replace(h.anchor, h.text, 1)
        else:  # append: the anchor only proves we are patching the right file
            if "\nimport os\n" not in src:
                missing.append(f"{h.name}: {h.path} does not import os at module level")
                continue
            new = src.rstrip("\n") + "\n" + h.text
        pending[path] = new
        out.append(f"{h.name}: anchor ok" if check else f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("rtx mhc jit hooks: " + "; ".join(missing))
    for rel, src_path in MODULES.items():
        dst = os.path.join(root, rel)
        want = open(src_path).read()
        cur = open(dst).read() if os.path.exists(dst) else None
        if check:
            out.append(f"module: {'up to date' if cur == want else 'would write'} {rel}")
        elif cur != want:
            _write(dst, want)
            out.append(f"module: wrote {rel}")
    if not check:
        for path, text in pending.items():
            _write(path, text)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
