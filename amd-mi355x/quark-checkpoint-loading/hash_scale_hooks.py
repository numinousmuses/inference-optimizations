"""Anchored hook: DeepSeek-V4 hash-routed MoE layers on AITER apply routed_scaling_factor (amdpro, 2026-10-04).
Tree: SGLang 1330e437e1 (lmsysorg/sglang-rocm v0.5.21-rocm724-mi35x-20261002), srt/models/deepseek_v2.py.

    python3 hash_scale_hooks.py [--root <sglang pkg dir>] [--check]

Bug (measured, claim amdpro.pro-mi355x.hash-layer-scaling-bug): on the AITER MoE path DeepseekV2MoE.forward never
multiplies the routed output by routed_scaling_factor (it assumes the top-k weights carry it); the DSV4 override sets
apply_routed_scaling_factor_on_output=True for the standard TopK when _use_aiter, but the HashTopK used by the first
num_hash_layers (3 for V4 Pro) keeps self.experts.should_fuse_routed_scaling_factor_in_topk (False on AITER). Result:
layers 0-2 run their routed experts at 1/2.5 of the intended weight. Per-module dumps vs 8xB200: identical expert ids,
MI355X weights sum 1.0 vs 2.5, routed output rel diff 0.600 = 1 - 1/2.5; every MI355X arm sits at KL 0.031 vs the B200
reference (B300 0.021) and the T=0 primes sanity prompt flips to "1, 2, 3, 5, 7".

Fix: the same override for HashTopK when _use_aiter and no fused shared expert (HashTopK rejects the on-output scaling
with a fused shared slot). Default on; NUMINOUS_HASH_SCALE_FIX=0 restores the stock behaviour for A/B.
"""
from __future__ import annotations

import argparse
import os
import sys

PATH = "srt/models/deepseek_v2.py"
NAME = "hash_scale_fix"
ANCHOR = ("                routed_scaling_factor=self.routed_scaling_factor,\n"
          "                apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,\n"
          "                layer_id=self.layer_id,\n"
          "            )\n"
          "        else:\n"
          "            # Default: grouped noaux_tc top-k.")
TEXT = ("                routed_scaling_factor=self.routed_scaling_factor,\n"
        "                # numinous-hook:hash_scale_fix (amdpro): the AITER MoE forward never applies routed_scaling_factor,\n"
        "                # so the hash layers' top-k must carry it, as the DSV4 TopK override below already does\n"
        "                apply_routed_scaling_factor_on_output=(\n"
        "                    True\n"
        "                    if (\n"
        "                        _use_aiter\n"
        "                        and self.num_fused_shared_experts == 0\n"
        "                        and __import__(\"os\").environ.get(\"NUMINOUS_HASH_SCALE_FIX\", \"1\") != \"0\"\n"
        "                    )\n"
        "                    else self.experts.should_fuse_routed_scaling_factor_in_topk\n"
        "                ),\n"
        "                layer_id=self.layer_id,\n"
        "            )\n"
        "        else:\n"
        "            # Default: grouped noaux_tc top-k.")


def apply(root: str, check: bool = False) -> list[str]:
    f = os.path.join(root, PATH)
    src = open(f).read()
    if f"numinous-hook:{NAME}" in src:
        return [f"{NAME}: already applied"]
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"{NAME}: anchor found {src.count(ANCHOR)} times in {PATH}")
    if check:
        return [f"{NAME}: anchor ok in {PATH}"]
    open(f, "w").write(src.replace(ANCHOR, TEXT))
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
