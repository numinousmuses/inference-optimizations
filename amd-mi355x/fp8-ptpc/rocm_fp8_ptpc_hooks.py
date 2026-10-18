"""Anchored hooks: serve block-FP8 dense linears through AITER's ptpc FP8 GEMM (k3amd for amdflash).
Tree: SGLang 1330e437e1 (lmsysorg/sglang-rocm v0.5.21-rocm724-mi35x-20261002). Inert unless
NUMINOUS_FP8_BLOCK_TO_PTPC=1 / NUMINOUS_PTPC_MAX_M are set. Logic: operations/kernels/numinous_fp8_ptpc.py
(installed as srt/layers/quantization/numinous_fp8_ptpc.py).

    python3 rocm_fp8_ptpc_hooks.py [--root <sglang pkg dir>] [--check]

Hooks:
  ptpc_attach     quantization/fp8.py process_weights_after_loading_block_quant: after the block weight / scales are
                  final and before the bpreshuffle, build the rowwise copy (aiter + gfx950 + aiter_w8a8_block_fp8_linear
                  only; layers that keep the plain layout are skipped)
  ptpc_max_m      quantization/fp8_utils.py aiter_w8a8_block_fp8_linear: the ptpc branch's M bound reads
                  NUMINOUS_PTPC_MAX_M (default: upstream's MXFP8_DENSE_PTPC_DECODE_MAX_M)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass

F8 = "srt/layers/quantization/fp8.py"
FU = "srt/layers/quantization/fp8_utils.py"
HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = {"srt/layers/quantization/numinous_fp8_ptpc.py": os.path.join(HERE, "..", "kernels", "numinous_fp8_ptpc.py")}
_M = "__import__('sglang.srt.layers.quantization.numinous_fp8_ptpc', fromlist=['x'])"


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str


HOOKS = [
    Hook("ptpc_attach", F8,
         "        layer.weight.data = weight.data\n"
         "        layer.weight_scale_inv.data = weight_scale.data\n"
         "        if self.block_fp8_as_mxfp8:\n",
         "        layer.weight.data = weight.data\n"
         "        layer.weight_scale_inv.data = weight_scale.data\n"
         "        # numinous-hook:ptpc_attach (k3amd): rowwise-FP8 copy for the ptpc GEMM (before the bpreshuffle)\n"
         f"        if ({_M}.enabled() and _use_aiter_bpreshuffle_gfx95\n"
         "                and self.w8a8_block_fp8_linear is aiter_w8a8_block_fp8_linear):\n"
         f"            {_M}.attach(layer, tuple(self.weight_block_size or (128, 128)))\n"
         "        if self.block_fp8_as_mxfp8:\n"),
    Hook("ptpc_max_m", FU,
         "            and input_2d.shape[0] <= MXFP8_DENSE_PTPC_DECODE_MAX_M\n",
         "            # numinous-hook:ptpc_max_m (k3amd): M bound from NUMINOUS_PTPC_MAX_M\n"
         f"            and input_2d.shape[0] <= {_M}.max_m(MXFP8_DENSE_PTPC_DECODE_MAX_M)\n"),
]


def sglang_root() -> str:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, check: bool = False) -> list[str]:
    out, pending = [], {}
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = pending.get(path) or open(path).read()
        if f"numinous-hook:{h.name}" in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            raise SystemExit(f"{h.name}: anchor found {n} times in {h.path}")
        if check:
            out.append(f"{h.name}: anchor ok (not applied)")
            continue
        pending[path] = src.replace(h.anchor, h.text, 1)
        out.append(f"{h.name}: applied to {h.path}")
    for path, src in pending.items():
        compile(src, path, "exec")
        open(path, "w").write(src)
    if not check:
        for dst, srcf in MODULES.items():
            shutil.copy(srcf, os.path.join(root, dst))
            out.append(f"module: wrote {dst}")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    lines = apply(a.root or sglang_root(), a.check)
    print("\n".join(lines))
    return 1 if any("not applied" in ln for ln in lines) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
