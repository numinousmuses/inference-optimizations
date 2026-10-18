"""Anchored hooks: deterministic MoE-side GEMMs for DeepSeek-V4 on ROCm (amdpro, 2026-10-04).
Tree: SGLang 1330e437e1 (lmsysorg/sglang-rocm v0.5.21-rocm724-mi35x-20261002).

    python3 det_gemm_hooks.py [--root <sglang pkg dir>] [--check]

Why (measured on mi355x8): identical T=0 requests gave different first-token distributions (the production sanity
gate's "first five primes" flips between "1, 2, 3, 5, 7", "2, 3, ..." and "The first five ..."). Tensor dumps with
module inputs: layer-0 mlp.gate and mlp.shared_experts.gate_up_proj get bit-identical inputs and return outputs that
differ by 1 bf16 ulp; 61 layers amplify that to logit differences of ~20. In isolation (gemm_determinism.py) the router
GEMM through aiter tgemm.mm differs on 19/19 calls at M = 15 / 64 / 1024 while F.linear on the same inputs is bitwise
deterministic. The shared-expert GEMM at M <= 128 takes the MXFP8 "ptpc" path (aiter.gemm_a8w8_bpreshuffle); above it
takes the block-FP8 GEMM that the (deterministic) attention linears use.

Hooks (each inert unless its env is "1"):
  det_router  layers/rocm_linear_utils.py aiter_dsv3_router_gemm: NUMINOUS_DET_ROUTER=1 -> F.linear (bf16 out, as before)
  det_ptpc    layers/quantization/fp8_utils.py aiter_w8a8_block_fp8_linear: NUMINOUS_DET_PTPC=1 -> skip the ptpc
              decode path, use the block-FP8 GEMM at every M
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str


HOOKS = [
    Hook("det_router", "srt/layers/rocm_linear_utils.py",
         "    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)\n",
         "    # numinous-hook:det_router (amdpro): tgemm.mm's default solution here is not bitwise reproducible\n"
         "    if __import__(\"os\").environ.get(\"NUMINOUS_DET_ROUTER\") == \"1\":\n"
         "        return torch.nn.functional.linear(hidden_states, weight.detach())\n"
         "    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)\n"),
    Hook("det_ptpc", "srt/layers/quantization/fp8_utils.py",
         "            ptpc_weight is not None\n"
         "            and input_2d.shape[0] <= MXFP8_DENSE_PTPC_DECODE_MAX_M\n",
         "            ptpc_weight is not None\n"
         "            and input_2d.shape[0] <= MXFP8_DENSE_PTPC_DECODE_MAX_M\n"
         "            # numinous-hook:det_ptpc (amdpro): NUMINOUS_DET_PTPC=1 keeps the block-FP8 GEMM at every M\n"
         "            and __import__(\"os\").environ.get(\"NUMINOUS_DET_PTPC\") != \"1\"\n"),
]


def apply(root: str, check: bool = False) -> list[str]:
    out = []
    for hk in HOOKS:
        f = os.path.join(root, hk.path)
        if not os.path.exists(f):
            raise SystemExit(f"{hk.name}: {hk.path} missing")
        src = open(f).read()
        if f"numinous-hook:{hk.name}" in src:
            out.append(f"{hk.name}: already applied")
            continue
        if src.count(hk.anchor) != 1:
            raise SystemExit(f"{hk.name}: anchor found {src.count(hk.anchor)} times in {hk.path}")
        if check:
            out.append(f"{hk.name}: anchor ok in {hk.path}")
            continue
        open(f, "w").write(src.replace(hk.anchor, hk.text))
        out.append(f"{hk.name}: applied to {hk.path}")
    return out


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
