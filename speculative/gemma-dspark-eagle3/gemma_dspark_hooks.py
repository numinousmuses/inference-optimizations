"""Anchored patcher: let SGLang v0.5.20's DSpark draft run on a Gemma target (srt/models/dspark.py).

DSpark attaches the TARGET's embedding and lm_head modules to the draft (attach_shared_modules). On Gemma 4 the tied
lm_head is `Gemma3TextScaledWordEmbedding` (a plain nn.Embedding that multiplies by embed_scale=sqrt(hidden) in
forward), which has neither `org_vocab_size` nor `quant_method`, so the draft crashes at boot (measured:
AttributeError ... 'org_vocab_size'). Fixes:
  0. attach_shared_modules gives such a head org_vocab_size = its rows (the draft sampler reads it too);
  1. vocab crop falls back to the head's weight rows when `org_vocab_size` is absent (TP=1: identical result);
  2. project_through_lm_head treats a head without `quant_method` as unquantized (plain matmul, as stock does);
  3. NUMINOUS_DSPARK_EMBED=unscaled makes forward_embed use the raw embedding table instead of the target's scaled
     module (speculators-trained drafts ship their own unscaled table); default "native" = stock behaviour.
For non-Gemma targets every branch is the stock code path.

    python3 operations/patches/gemma_dspark_hooks.py [--check] [--root DIR]
Test: operations/patches/test_gemma_dspark_hooks.py. Missing anchors fail loudly.
"""
from __future__ import annotations

import argparse
import os
import sys

TARGET = "srt/models/dspark.py"
MARK = "numinous-hook:gemma_dspark"

EDITS = [
    ("            self.embed_tokens = embed_tokens\n"
     "        self.lm_head = lm_head\n",
     "            self.embed_tokens = embed_tokens\n"
     "        self.lm_head = lm_head\n"
     f"        # {MARK}: Gemma's tied head lacks org_vocab_size, read by the draft sampler\n"
     "        # (speculative/dspark_components/dspark_draft_sampler.py) as well as below\n"
     "        if not hasattr(lm_head, \"org_vocab_size\") and hasattr(lm_head, \"weight\"):\n"
     "            lm_head.org_vocab_size = int(lm_head.weight.shape[0])\n"),
    ("    return full_logits[..., : int(lm_head.org_vocab_size)]\n",
     f"    # {MARK}: Gemma's tied head (scaled nn.Embedding) has no org_vocab_size\n"
     "    return full_logits[..., : int(getattr(lm_head, \"org_vocab_size\", None) or lm_head.weight.shape[0])]\n"),
    ("    quant_method = lm_head.quant_method\n"
     "    if should_apply_lm_head_quant_method(lm_head, quant_method):\n",
     f"    # {MARK}: a head without quant_method (Gemma's tied embedding) is unquantized\n"
     "    quant_method = getattr(lm_head, \"quant_method\", None)\n"
     "    if quant_method is not None and should_apply_lm_head_quant_method(lm_head, quant_method):\n"),
    ("        # model exposes forward_embed).\n"
     "        return self.embed_tokens(input_ids)\n",
     "        # model exposes forward_embed).\n"
     f"        # {MARK}: NUMINOUS_DSPARK_EMBED=unscaled -> raw table (Gemma's shared module scales in forward)\n"
     "        if (__import__(\"os\").environ.get(\"NUMINOUS_DSPARK_EMBED\", \"native\") == \"unscaled\"\n"
     "                and getattr(self.embed_tokens, \"embed_scale\", None) is not None):\n"
     "            return torch.nn.functional.embedding(input_ids, self.embed_tokens.weight)\n"
     "        return self.embed_tokens(input_ids)\n"),
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
            raise SystemExit(f"gemma_dspark_hooks: anchor found {n} times (need 1) in {TARGET}:\n{anchor}")
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
        print(f"gemma_dspark_hooks: anchors ok in {path}" + (" (already patched)" if new == src else ""))
        return 0
    if new != src:
        open(path, "w").write(new)
    print(f"gemma_dspark_hooks: patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
