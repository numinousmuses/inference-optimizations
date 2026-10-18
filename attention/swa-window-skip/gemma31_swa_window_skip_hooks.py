"""Sliding-window prefill/extend: start the stock extend kernel's KV loops at the window instead of at 0 (rtxattn,
Gemma kernel program A3 item "check prefill skips out-of-window tiles").

Measured (rtxv41, findings rtxv41_gemma31.md section 9): an 8K prefill chunk spends 311 ms in the 50 SWA layers (6.2
ms per layer, 24x the 13.0 ms floor) with window 1024.
Source (v0.5.20 kernels/ops/attention/extend_attention.py, _fwd_kernel):
  - The prefix loop (:495) and the extend loop (:675) both run from start_n = 0.
  - With a window they only SKIP_TILE a fully masked tile (:520-523, :711-714) after building its mask and reducing
    it. At 8K x window 1024 that is ~7 of every 8 iterations doing mask work for nothing.
Fix: for SLIDING_WINDOW_SIZE > 0 without a custom mask, begin each loop at the first tile the block's first query
row can see. Window rule (:513-519, :705-710): visible iff q_id <= kv_id + W.
  prefix_lo = floor((prefix_len + block_m * BLOCK_M - W) / BLOCK_N_PREFIX) * BLOCK_N_PREFIX, clamped at 0
  extend_lo = floor((block_m * BLOCK_M - W) / BLOCK_N) * BLOCK_N, clamped at 0
Every skipped tile is one SKIP_TILE would have skipped, so the tiles that run are the same, in the same order:
bit-identical output. The hook applies to every sliding-window model through this kernel; it is included only in the
cells that list it.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import sys

MARKER = "numinous-hook:gemma31_swa_window_skip"
REL = "kernels/ops/attention/extend_attention.py"
HOOKS = [
    ("prefix",
     "    prefix_end = 0 if SKIP_PREFIX else cur_seq_len_prefix\n"
     "    for start_n in range(0, prefix_end, BLOCK_N_PREFIX):\n",
     "    prefix_end = 0 if SKIP_PREFIX else cur_seq_len_prefix\n"
     "    prefix_lo = 0  # {marker}: first tile the block's first row can see\n"
     "    if SLIDING_WINDOW_SIZE > 0 and not USE_CUSTOM_MASK:\n"
     "        prefix_lo = (\n"
     "            tl.maximum(0, cur_seq_len_prefix + cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE)\n"
     "            // BLOCK_N_PREFIX\n"
     "            * BLOCK_N_PREFIX\n"
     "        )\n"
     "    for start_n in range(prefix_lo, prefix_end, BLOCK_N_PREFIX):\n"),
    ("extend",
     "    extend_end = 0 if SKIP_EXTEND else cur_block_m_end\n"
     "    for start_n in range(0, extend_end, BLOCK_N):\n",
     "    extend_end = 0 if SKIP_EXTEND else cur_block_m_end\n"
     "    extend_lo = 0  # {marker}: first tile the block's first row can see\n"
     "    if SLIDING_WINDOW_SIZE > 0 and not USE_CUSTOM_MASK:\n"
     "        extend_lo = tl.maximum(0, cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE) // BLOCK_N * BLOCK_N\n"
     "    for start_n in range(extend_lo, extend_end, BLOCK_N):\n"),
]


def apply(root: str, check: bool = False) -> list[str]:
    path = pathlib.Path(root) / REL
    text = path.read_text()
    out = []
    for name, old, new in HOOKS:
        tag = f"{MARKER} ({name})"
        if tag in text:
            out.append(f"{name}: already applied")
            continue
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"{name}: anchor found {n} times in {REL} (need exactly 1)")
        if check:
            out.append(f"{name}: anchor ok in {REL}")
            continue
        text = text.replace(old, new.replace("{marker}", tag))
        out.append(f"{name}: applied to {REL}")
    if not check and any("applied to" in o for o in out):
        tmp = path.with_suffix(path.suffix + f".numinous.{os.getpid()}")
        tmp.write_text(text)
        py_compile.compile(str(tmp), doraise=True)
        os.replace(tmp, path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
