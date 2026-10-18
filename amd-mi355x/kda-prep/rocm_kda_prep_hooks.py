"""Anchored hooks: one-pass KDA prefill q/k/v split + q/k L2 norm (k3amd for amdflash).
Tree: SGLang 1330e437e1 (lmsysorg/sglang-rocm v0.5.21-rocm724-mi35x-20261002). Inert unless NUMINOUS_KDA_PREP=1.
Kernel and rationale: operations/kernels/numinous_kda_prep.py (installed as
srt/layers/attention/linear/numinous_kda_prep.py).

    python3 rocm_kda_prep_hooks.py [--root <sglang pkg dir>] [--check]

Hooks:
  kda_prep_split     layers/attention/linear/kda_backend.py forward_extend: q, k, v from the fused kernel (contiguous,
                     q/k normalized) instead of qkv.split views
  kda_prep_flag      the dispatcher.extend call passes qk_prenormed
  kda_prep_l2        layers/attention/linear/kernels/kda_triton.py TritonKDAKernel.extend: use_qk_l2norm_in_kernel is
                     False when the inputs arrive normalized
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass

KB = "srt/layers/attention/linear/kda_backend.py"
KT = "srt/layers/attention/linear/kernels/kda_triton.py"
HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = {"srt/layers/attention/linear/numinous_kda_prep.py": os.path.join(HERE, "..", "kernels", "numinous_kda_prep.py")}
_M = "__import__('sglang.srt.layers.attention.linear.numinous_kda_prep', fromlist=['x'])"


def module_sha16() -> str:
    """amd_gate keys the overlay on the patcher bytes; the test pins this so a kernel edit forces a rebuild."""
    return hashlib.sha256(open(MODULES["srt/layers/attention/linear/numinous_kda_prep.py"], "rb").read()).hexdigest()[:16]


MODULE_SHA256 = "71a36a87629a4157"


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str


HOOKS = [
    Hook("kda_prep_split", KB,
         "        qkv = self._convolve_prefill(\n"
         "            layer, forward_batch, mixed_qkv, conv_states, cache_indices\n"
         "        )\n"
         "        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)\n",
         "        qkv = self._convolve_prefill(\n"
         "            layer, forward_batch, mixed_qkv, conv_states, cache_indices\n"
         "        )\n"
         "        # numinous-hook:kda_prep_split (k3amd): one pass split + q/k L2 norm, contiguous outputs\n"
         "        _nm_prenormed = False\n"
         f"        if ({_M}.enabled() and layer.q_dim == layer.k_dim == layer.v_dim\n"
         "                and layer.head_q_dim == layer.head_k_dim == layer.head_v_dim and qkv.stride(-1) == 1):\n"
         f"            q, k, v = {_M}.split_l2norm(qkv, layer.q_dim, layer.k_dim, layer.v_dim, layer.head_k_dim)\n"
         "            _nm_prenormed = True\n"
         "        else:\n"
         "            q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)\n"),
    Hook("kda_prep_flag", KB,
         "            is_spec_decode=forward_batch.forward_mode.is_draft_extend_v2(),\n",
         "            is_spec_decode=forward_batch.forward_mode.is_draft_extend_v2(),\n"
         "            qk_prenormed=_nm_prenormed,  # numinous-hook:kda_prep_flag (k3amd)\n"),
    Hook("kda_prep_l2", KT,
         "            initial_state_indices=cache_indices,\n"
         "            use_qk_l2norm_in_kernel=True,\n"
         "            cu_seqlens=query_start_loc,\n",
         "            initial_state_indices=cache_indices,\n"
         "            # numinous-hook:kda_prep_l2 (k3amd): skip the l2norm when q / k arrive normalized\n"
         "            use_qk_l2norm_in_kernel=not kwargs.get(\"qk_prenormed\", False),\n"
         "            cu_seqlens=query_start_loc,\n"),
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
