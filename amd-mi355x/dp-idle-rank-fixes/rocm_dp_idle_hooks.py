#!/usr/bin/env python3
"""DP attention + EAGLE on ROCm: a step with idle DP ranks runs eagerly on every rank. Overlay patcher, inert unless
NUMINOUS_DP_IDLE_EAGER=1.

Bug (measured 2026-10-03, amdglm and k3b300 bisects on mi355x8b, 20261002 image, GLM-5.3 Quark): with DP attention
and EAGLE, an almost empty server (1-2 requests on 4 DP ranks, i.e. idle ranks) decodes garbage after a correct first
token ('17!!!!', production node_oAB5lafZ; '39<random multilingual>' in chat).
  plain DP4 (no EAGLE): correct | DP4 + EAGLE (stock, none of our hooks): garbage | Fast v2 (TP attention): correct
  DP4 + EAGLE --disable-cuda-graph (eager): correct | + --disable-cuda-graph-padding: correct
  + SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1: garbage | + --disable-overlap-schedule: garbage
  + extra kernels in every graph (graph NaN locator): correct (masked)
  k3b300: dp_gather hook / gatherv / burst / one-hot / sparse decode / lm-head flags off: all garbage
Under load (every DP rank has requests) every config is correct.

Hook (dpidle_sync): where the scheduler min-reduces the per-rank graph votes (MLPSyncBatchInfo, every step, identical
tp0_info on every rank), a step in which some DP ranks have tokens and some have none runs eagerly on ALL ranks
(can_run_decode_cuda_graph and can_run_draft_cuda_graph False): the measured-correct --disable-cuda-graph behaviour,
only for those steps. Every rank takes the same decision, so the collectives stay aligned. (A first version made only
the idle rank eager inside the graph runners: the active ranks replayed graphs, the collectives misaligned and the
server warm-up deadlocked, measured 18:11.) Cost: steps with an idle rank run eagerly (low load only: fewer running
requests than DP ranks or an unbalanced queue); steps with every rank busy keep their graphs.

    python3 rocm_dp_idle_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import sys

ENV = "NUMINOUS_DP_IDLE_EAGER"

_SYNC_ANCHOR = "        self.can_run_draft_cuda_graph = bool(tp0_info_cpu[:, 8].min())\n"
_SYNC_REPL = (_SYNC_ANCHOR
              + f'        if __import__("os").environ.get("{ENV}") == "1" and 0 in self.global_num_tokens '
                "and max(self.global_num_tokens) > 0:  # numinous-hook:dpidle_sync\n"
              + "            self.can_run_decode_cuda_graph = False\n"
              + "            self.can_run_draft_cuda_graph = False\n")

HOOKS = [
    ("dpidle_sync", "srt/managers/scheduler_components/dp_attn.py", _SYNC_ANCHOR, _SYNC_REPL),
]


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, check: bool = False) -> list[str]:
    out, missing = [], []
    for name, rel, anchor, repl in HOOKS:
        path = os.path.join(root, rel)
        src = open(path).read()
        if f"numinous-hook:{name}" in src:
            out.append(f"{name}: already applied")
            continue
        n = src.count(anchor)
        if n != 1:
            missing.append(f"{name}: anchor found {n} times in {rel}")
            continue
        if check:
            out.append(f"{name}: anchor ok")
            continue
        new = src.replace(anchor, repl, 1)
        compile(new, path, "exec")
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"{name}: applied to {rel}")
    if missing:
        raise SystemExit("rocm dp idle hooks: " + "; ".join(missing))
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
