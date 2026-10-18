"""Anchored hook for SGLang v0.5.20: let the expert-distribution recorder see MegaMoE layers (DeepSeek V4 Pro).

Why (measured, rtxattn 2026-10-03, B300:8 run 6): with --expert-distribution-recorder-mode stat on the MegaMoE path
MoE calls raised "RuntimeError: Index tensor must have the same number of dimensions as self tensor" in
eplb/expert_distribution.py _SelectExpertsSinglePassGatherer.on_select_experts. The target model's layer loop
(deepseek_v4.py DeepseekV4Model.forward) enters the recorder's with_current_layer(i), but the DSpark draft's MoE layers
run outside it, so the gatherer indexes its [layers, experts] buffer with layer_idx=None (a 3-D view) and
scatter_add_ rejects the 1-D index.

Hook `megamoe_recorder_layer`: forward_mega_moe enters with_current_layer(moe.layer_id) around its router top-k call
ONLY when no layer is set (the draft path). Entering unconditionally nests inside the target loop's context and trips
Withable's `assert self._value is None` (measured, run 7, 2026-10-03 11:27). The no-op recorder (recorder off) has no
_current_layer_idx, so the hook takes the nullcontext branch and is inert in every serving launch.

    python3 megamoe_eprec_hooks.py                 # patch the installed sglang
    python3 megamoe_eprec_hooks.py --root /tmp/sgl --check
"""
from __future__ import annotations

import argparse
import os
import sys

PATH = "srt/layers/moe/mega_moe.py"
MARK = "# numinous-hook:megamoe_recorder_layer"
ANCHOR_ENTER = "        topk_output = moe.topk(\n"
ANCHOR_EXIT = "        topk_ids = topk_output.topk_ids\n"
ENTER = ("        " + MARK + "\n"
         "        from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder as _numinous_rec\n"
         "        _numinous_r = _numinous_rec()\n"
         "        _numinous_rec_ctx = (_numinous_r.with_current_layer(moe.layer_id)\n"
         "                             if getattr(getattr(_numinous_r, '_current_layer_idx', None), 'value', 0) is None\n"
         "                             else nullcontext())\n"
         "        _numinous_rec_ctx.__enter__()\n")
EXIT = "        _numinous_rec_ctx.__exit__(None, None, None)\n"


def apply(text: str) -> str:
    if MARK in text:
        return text
    if text.count(ANCHOR_ENTER) != 1 or text.count(ANCHOR_EXIT) != 1:
        raise SystemExit(f"megamoe_recorder_layer: anchors not found exactly once in {PATH}")
    text = text.replace(ANCHOR_ENTER, ENTER + ANCHOR_ENTER)
    return text.replace(ANCHOR_EXIT, ANCHOR_EXIT + EXIT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = a.root
    if not root:
        import sglang
        root = os.path.dirname(sglang.__file__)
    p = os.path.join(root, PATH)
    src = open(p).read()
    out = apply(src)
    if a.check:
        compile(out, p, "exec")
        print("megamoe_recorder_layer: " + ("already applied" if out == src else "anchor ok"))
        return 0
    if out != src:
        open(p, "w").write(out)
    print("megamoe_recorder_layer: applied to " + PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
