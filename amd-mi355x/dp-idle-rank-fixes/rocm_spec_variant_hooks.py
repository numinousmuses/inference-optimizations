#!/usr/bin/env python3
"""EAGLE draft graphs under DP attention: every DP rank replays the same sampling variant. Overlay patcher, inert unless
NUMINOUS_DP_NO_GREEDY_VARIANT=1.

Measured (amdglm 2026-10-03/04, mi355x8b, 20261002 image, GLM-5.3 Quark, TP4 x DP4 attention + EAGLE, empty server):
garbage / '!!!!' with the AITER custom all-reduce and with sglang's own CustomAllreduce (SGLANG_USE_AITER_AR=0),
correct with --disable-custom-all-reduce (RCCL). Both custom all-reduces pair each CUDA graph's input buffer across ranks
by registration order (IPC handles registered per captured graph), so they are only correct when every rank replays
the graph with the same key; RCCL has no such pairing.

Source (1330e437e1 eagle_draft_cuda_graph_runner): ROCm turns on rejection sampling for EAGLE, and the draft-decode graph
key carries a sampling variant: `_sampling_variant` returns "greedy" when `forward_batch.sampling_info.is_all_greedy`.
`is_all_greedy = all(top_k <= 1 for r in reqs)` is True for an EMPTY batch, so an idle DP rank replays the greedy
draft graph while an active rank serving a sampled request (temperature > 0, the chat default) replays the non-greedy
graph: same collectives, different registered buffers -> the custom all-reduce reads the wrong peer buffers.

Hook spvar_draft (srt/speculative/eagle_draft_cuda_graph_runner.py, _sampling_variant): under DP attention
(require_mlp_tp_gather) always replay the default (non-greedy) variant, so every rank keys the same graph; greedy
batches then take the proposal-distribution path they would take without the variant optimisation (exact outputs).

    python3 rocm_spec_variant_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import sys

ENV = "NUMINOUS_DP_NO_GREEDY_VARIANT"
TARGET = "srt/speculative/eagle_draft_cuda_graph_runner.py"
ANCHOR = ("    def _sampling_variant(self, forward_batch: ForwardBatch) -> Optional[str]:\n"
          "        # Under rejection sampling an all-greedy batch replays the graph that\n")
REPL = ("    def _sampling_variant(self, forward_batch: ForwardBatch) -> Optional[str]:\n"
        f'        if __import__("os").environ.get("{ENV}") == "1" and getattr(self, "require_mlp_tp_gather", False):'
        "  # numinous-hook:spvar_draft\n"
        "            return None\n"
        "        # Under rejection sampling an all-greedy batch replays the graph that\n")


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, check: bool = False) -> list[str]:
    path = os.path.join(root, TARGET)
    src = open(path).read()
    if "numinous-hook:spvar_draft" in src:
        return ["spvar_draft: already applied"]
    n = src.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"rocm spec variant hooks: spvar_draft anchor found {n} times in {TARGET}")
    if check:
        return ["spvar_draft: anchor ok"]
    new = src.replace(ANCHOR, REPL, 1)
    compile(new, path, "exec")
    tmp = f"{path}.numinous.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(new)
    os.replace(tmp, path)
    return ["spvar_draft: applied"]


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
