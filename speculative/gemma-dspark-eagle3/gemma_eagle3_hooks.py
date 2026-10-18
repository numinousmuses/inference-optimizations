"""Anchored patcher: honour `norm_before_residual` in SGLang v0.5.20's EAGLE3 draft layer (srt/models/llama_eagle3.py).

Why: the RedHat Gemma-4-26B-A4B EAGLE3 speculator (RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3, vLLM
"speculators" packaging) was trained with `norm_before_residual: true`: the input layer takes the residual AFTER
hidden_norm (vLLM llama_eagle3 does the same when the flag is set). SGLang always takes it BEFORE, so the draft runs a
different function than it was trained as. That cannot change outputs (the target verifies every token) but lowers
acceptance. With the flag absent/false the patched code is identical to stock.

    python3 operations/patches/gemma_eagle3_hooks.py              # patch the installed sglang
    python3 operations/patches/gemma_eagle3_hooks.py --check      # verify anchors only
    python3 operations/patches/gemma_eagle3_hooks.py --root DIR   # patch a copy (DIR = the sglang package dir)

Test: operations/patches/test_gemma_eagle3_hooks.py. Missing anchors fail loudly (exit 1).
"""
from __future__ import annotations

import argparse
import os
import sys

TARGET = "srt/models/llama_eagle3.py"
MARK = "numinous-hook:eagle3_norm_before_residual"

# (anchor, replacement) pairs; each anchor must occur exactly once
EDITS = [
    ("        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)\n",
     "        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)\n"
     f"        # {MARK}\n"
     "        self._norm_before_residual = bool(getattr(config, \"norm_before_residual\", False))\n"),
    ("            residual = hidden_states\n"
     "            hidden_states = self.hidden_norm(hidden_states)\n",
     f"            # {MARK}: speculators/vLLM semantics when norm_before_residual is set\n"
     "            if self._norm_before_residual:\n"
     "                hidden_states = self.hidden_norm(hidden_states)\n"
     "                residual = hidden_states\n"
     "            else:\n"
     "                residual = hidden_states\n"
     "                hidden_states = self.hidden_norm(hidden_states)\n"),
]


def sglang_root() -> str:
    import importlib.util
    return list(importlib.util.find_spec("sglang").submodule_search_locations)[0]


def apply(src: str) -> str:
    if MARK in src:
        return src
    for anchor, repl in EDITS:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"gemma_eagle3_hooks: anchor found {n} times (need 1) in {TARGET}:\n{anchor}")
        src = src.replace(anchor, repl)
    return src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    path = os.path.join(a.root or sglang_root(), TARGET)
    src = open(path).read()
    new = apply(src)
    compile(new, path, "exec")
    if a.check:
        print(f"gemma_eagle3_hooks: anchors ok in {path}" + (" (already patched)" if new == src else ""))
        return 0
    if new != src:
        open(path, "w").write(new)
    print(f"gemma_eagle3_hooks: patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
