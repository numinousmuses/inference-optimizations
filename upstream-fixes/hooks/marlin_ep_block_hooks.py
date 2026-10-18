"""Anchored hook: pick fused_marlin_moe's M-block from the GLOBAL expert count under expert parallelism (rtxsys,
decode-MoE owner, 2026-10-05). Tree: SGLang v0.5.20 (IMG_V0520), srt/layers/moe/fused_moe_triton/fused_marlin_moe.py.

    python3 marlin_ep_block_hooks.py [--root <sglang pkg dir>] [--check]

The bug (source_code, fused_marlin_moe.py:240-242):
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break
- E = w1.shape[0] is the rank's LOCAL experts. M * topk counts every (token, expert) pair, including the -1 ids of
  other ranks' experts (the standard dispatcher's local mapping, moe_runner/marlin.py:184).
- Under EP8 that is 8x the real tokens per local expert, so the block is 8x too large.
- Marlin's per-GEMM time is linear in the block (thread_m_blocks = block/16 MMA tiles per weight tile).

Measured, 1xH200 graph replay (research/gpu-evidence/rtxsys/moe_decode_bench_h200.json), GLM-5.3 int4 g128 (us per
MoE call, stock pick -> pick from the global count):

| layout | m 144 | m 192 | m 288 | m 576 |
|---|---|---|---|---|
| EP8 | 441 (bs48) -> 255 (bs8) | 535 (bs64) -> 269 (bs8) | 552 (bs64) -> 298 (bs16) | 599 (bs64) -> 413 (bs32) |
| EP4 | 662 (bs32) -> 451 (bs8) | - | 856 (bs48) -> 521 (bs16) | 1079 (bs64) -> 740 (bs32) |

TP layouts (expert_map None) keep the stock pick byte for byte. That pick matched the best block within 5% on
Flash TP4 / GLM TP8.

The fix uses the expected local pairs per local expert, M * topk / global_num_experts (uniform-routing
expectation). That needs no device->host sync, so it stays CUDA-graph safe. Env NUMINOUS_MARLIN_EP_BLOCK=1.
Unset: stock.
"""
from __future__ import annotations

import argparse
import os
import py_compile
import sys

PATH = "srt/layers/moe/fused_moe_triton/fused_marlin_moe.py"
NAME = "marlin_ep_block"
TAG = f"# numinous-hook:{NAME}"
ANCHOR = ("    for block_size_m in [8, 16, 32, 48, 64]:\n"
          "        if M * topk / E / block_size_m < 0.9:\n"
          "            break\n")
TEXT = (f"    {TAG} (rtxsys): under EP, E counts LOCAL experts while M * topk counts every pair\n"
        "    _nm_e = E\n"
        "    if (\n"
        "        expert_map is not None\n"
        "        and global_num_experts > E\n"
        '        and __import__("os").environ.get("NUMINOUS_MARLIN_EP_BLOCK", "0") == "1"\n'
        "    ):\n"
        "        _nm_e = global_num_experts\n"
        "    for block_size_m in [8, 16, 32, 48, 64]:\n"
        "        if M * topk / _nm_e / block_size_m < 0.9:\n"
        "            break\n")


def apply(root: str, check: bool = False) -> list[str]:
    f = os.path.join(root, PATH)
    src = open(f).read()
    if TAG in src:
        return [f"{NAME}: already applied"]
    n = src.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"{NAME}: anchor found {n} times in {PATH}")
    if check:
        return [f"{NAME}: anchor ok in {PATH}"]
    open(f, "w").write(src.replace(ANCHOR, TEXT))
    py_compile.compile(f, doraise=True)
    return [f"{NAME}: applied to {PATH}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = a.root
    if root is None:
        import importlib.util
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("sglang is not importable")
        root = list(spec.submodule_search_locations)[0]
    for ln in apply(root, a.check):
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
