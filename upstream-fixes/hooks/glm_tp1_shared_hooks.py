"""Fix SGLANG_SHARED_EXPERT_TP1 under DP attention in SGLang v0.5.20 (srt/models/deepseek_v2.py), in the anchored
style of sglang_numinous_hooks.py, except these hooks REPLACE their anchor (the buggy add cannot be fixed by pure
insertion): each hook names its file, an exact unique anchor and the replacement; a missing or repeated anchor fails
loudly; hooks are idempotent (marker line checked first); a patched file must compile.

The bug (measured on 8xB200, GLM-5.3 NVFP4, --tp 8 --dp 8 --enable-dp-attention, SGLANG_SHARED_EXPERT_TP1=1: the
target answers garbage from token 1 at temperature 0; without the env the same server answers "17*23 = 391"):
DeepseekV2MoE.forward_normal and forward_normal_dual_stream add the replicated (TP1) shared expert output after the
post-experts all-reduce. Under DP attention every MAX_LEN step skips that all-reduce
(should_skip_post_experts_all_reduce: the LayerCommunicator reduce-scatters / all-reduces the TP partials later), so
the add lands on each rank's PARTIAL routed sum and the deferred reduction then counts the shared output tp_size
times.

The fix: when the post-experts all-reduce was skipped, add the shared output on TP rank 0 only; the deferred
reduction sums rank contributions exactly once, so the result is exact under both the deferred all-reduce and the
reduce-scatter. When the all-reduce ran (it is outside the reduction), keep the stock every-rank add. The decision
uses the same should_skip_post_experts_all_reduce() the skip itself uses, so the two can never disagree; it is
per-layer-forward Python, identical on every rank, and constant at CUDA-graph capture.

    python3 glm_tp1_shared_hooks.py                           # patch the installed sglang package
    python3 glm_tp1_shared_hooks.py --root /tmp/sgl --check   # report which anchors resolve, change nothing
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sglang_numinous_hooks import sglang_root  # noqa: E402

MARKER = "numinous-hook:glm_tp1_shared_once"

_GUARD = '''            # {marker}
            # The all-reduce above was skipped on this step (the communicator
            # reduces the TP partials later): every rank holds a PARTIAL sum,
            # and adding the replicated shared output on every rank counts it
            # tp_size times after the deferred reduction. Add it on TP rank 0
            # only; exact under both the deferred all-reduce and the
            # reduce-scatter. When the all-reduce ran it is outside any later
            # reduction and the every-rank add is correct.
            if not (
                self.tp_size > 1
                and should_skip_post_experts_all_reduce(is_tp_path=True)
                and get_parallel().tp_rank != 0
            ):
                final_hidden_states += shared_output
'''.format(marker=MARKER)

_COMMENT = (
    "        # TP1 shared experts are replicated, so add them after all-reduce to\n"
    "        # avoid summing the same shared output once per TP rank.\n"
)


@dataclass(frozen=True)
class Replace:
    name: str
    path: str
    anchor: str
    text: str


HOOKS = [
    Replace(
        "tp1_shared_once_dual_stream", "srt/models/deepseek_v2.py",
        _COMMENT
        + "        if self._shared_expert_tp1:\n"
        + "            final_hidden_states += shared_output\n",
        _COMMENT
        + "        if self._shared_expert_tp1 and shared_output is not None:\n"
        + _GUARD,
    ),
    Replace(
        "tp1_shared_once_normal", "srt/models/deepseek_v2.py",
        _COMMENT
        + "        if shared_output is not None and self._shared_expert_tp1:\n"
        + "            final_hidden_states += shared_output\n",
        _COMMENT
        + "        if shared_output is not None and self._shared_expert_tp1:\n"
        + _GUARD,
    ),
]


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook; SystemExit if an anchor is missing or repeated. All hooks of a file are applied to
    its text together and the file is written once (never half-patched)."""
    out, missing, files = [], [], {}
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = files.get(path)
        if src is None:
            src = files[path] = open(path).read()
        if f"{MARKER} ({h.name})" in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        files[path] = src.replace(h.anchor, h.text.replace(MARKER, f"{MARKER} ({h.name})"), 1)
        out.append(f"{h.name}: {'anchor ok' if check else 'applied to ' + h.path}")
    if missing:
        raise SystemExit("glm_tp1_shared hooks: " + "; ".join(missing))
    if not check:
        for path, new in files.items():
            if new == open(path).read():
                continue
            compile(new, path, "exec")  # never leave a file that does not parse
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
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
