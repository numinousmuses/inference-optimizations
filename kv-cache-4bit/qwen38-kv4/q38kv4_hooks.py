"""q38kv4_hooks: checkpoint FP8 KV scales must not be applied to an unscaled (bf16/fp16/fp32) KV cache.

Bug (measured, RTX PRO 6000 gate q38kv4-bf16, 2026-09-29): unsloth/Qwen3.8-27B-NVFP4 with `--kv-cache-dtype
bfloat16` scores GSM8K 0.005 (control FP8 KV 0.875), teacher-forced KL 2.39, top-1 0.424.
Cause (source_code, SGLang v0.5.20):
  * layers/quantization/kv_cache.py BaseKVCacheMethod.process_weights_after_loading sets layer.k_scale_float /
    v_scale_float from the checkpoint (FP8-convention per-tensor scales, 0.02-0.04 on Qwen3.8) whatever the KV dtype;
    the KV dtype is only resolved later (model_runner.configure_kv_cache_dtype runs after load_model).
  * the pool divides K/V by the scale only when the cache dtype differs from the input dtype (the FP8 case), so a
    bf16 cache holds unscaled K/V;
  * readers multiply by the scale anyway: FlashInfer prefill/decode pass k_scale=layer.k_scale_float,
    v_scale=layer.v_scale_float (sm_scale x k_scale, output x v_scale regardless of the KV dtype).
  => logits x~0.03 and outputs x~0.025 in every full-attention layer, prefill and decode.
Same mechanism as the NVFP4 dequant-workspace bug (kv4trt_hooks edit 8), which is why both land at KL ~2.39.

Fix (one anchored edit, model_executor/model_runner.py configure_kv_cache_dtype, right after the runner's KV dtype
is resolved and the weights are loaded): when the resolved KV cache is an unquantized float dtype, reset every
attention layer's per-tensor KV scales to 1.0 (tensor and float forms). FP8 caches keep the scales (they are the
store/read descale), FP4 caches keep them (NVFP4KVCacheMethod.load_scales_from_model uses them as global scales).

    python3 operations/patches/q38kv4_hooks.py [--root <sglang package dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import sys

TARGET = "srt/model_executor/model_runner.py"
MARK = "numinous-hook:q38kv4"
ANCHOR = ("            else get_model().kv_cache_dtype\n"
          "        )\n")
INSERT = (f"        # {MARK}: checkpoint FP8 KV scales describe an FP8 cache stored as k / k_scale; an unquantized\n"
          "        # cache stores k as is, but readers still multiply by k_scale_float -> neutralize them here.\n"
          "        if self.kv_cache_dtype in (torch.bfloat16, torch.float16, torch.float32) and getattr(\n"
          "            self, \"model\", None\n"
          "        ) is not None:\n"
          "            from sglang.srt.layers.quantization.q38kv4_scale_reset import reset_unquantized_kv_scales\n"
          "            reset_unquantized_kv_scales(self.model)\n")
MODULE_DST = "srt/layers/quantization/q38kv4_scale_reset.py"
MODULE = '''"""Installed by operations/patches/q38kv4_hooks.py: reset checkpoint FP8 KV scales for unquantized KV caches."""
import logging

import torch

logger = logging.getLogger(__name__)


def reset_unquantized_kv_scales(model) -> int:
    """Set k/v scales to 1.0 on every module carrying checkpoint KV scales; returns how many modules changed."""
    n = 0
    for m in model.modules():
        kf, vf = getattr(m, "k_scale_float", None), getattr(m, "v_scale_float", None)
        if kf is None and vf is None:
            continue
        if (kf in (None, 1.0)) and (vf in (None, 1.0)):
            continue
        m.k_scale_float = 1.0
        m.v_scale_float = 1.0
        for name in ("k_scale", "v_scale"):
            t = getattr(m, name, None)
            if isinstance(t, torch.Tensor):
                with torch.no_grad():
                    t.fill_(1.0)
        n += 1
    if n:
        logger.warning("q38kv4: unquantized KV cache: reset checkpoint FP8 KV scales to 1.0 on %d attention layers", n)
    return n
'''


def sglang_root() -> str:
    import importlib.util
    return list(importlib.util.find_spec("sglang").submodule_search_locations)[0]


def apply(src: str) -> str:
    if MARK in src:
        return src
    n = src.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"q38kv4_hooks: anchor found {n} times (need 1) in {TARGET}")
    return src.replace(ANCHOR, ANCHOR + INSERT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = a.root or sglang_root()
    path = os.path.join(root, TARGET)
    src = open(path).read()
    new = apply(src)
    compile(new, path, "exec")
    compile(MODULE, MODULE_DST, "exec")
    if a.check:
        print(f"q38kv4_hooks: anchors ok in {path}" + (" (already patched)" if new == src else ""))
        return 0
    if new != src:
        open(path, "w").write(new)
        print(f"q38kv4_hooks: patched {path}")
    else:
        print(f"q38kv4_hooks: already patched {path}")
    open(os.path.join(root, MODULE_DST), "w").write(MODULE)
    print(f"q38kv4_hooks: wrote {os.path.join(root, MODULE_DST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
