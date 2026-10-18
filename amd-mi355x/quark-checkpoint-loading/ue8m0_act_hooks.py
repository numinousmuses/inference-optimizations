"""Anchored hooks: DeepSeek-V4 dense FP8 activations quantized on the checkpoint's ue8m0 grid on ROCm (amdpro,
2026-10-04). Tree: SGLang 1330e437e1 (lmsysorg/sglang-rocm v0.5.21-rocm724-mi35x-20261002).

    python3 ue8m0_act_hooks.py [--root <sglang pkg dir>] [--check]

Why (measured): the T=0 primes sanity prompt centres on P("1") ~0.5 on MI355X vs P("The") 0.64 / P("1") 0.11 on
8xB200 v0.5.20 (bit-identical over 12 requests). Per-module dumps vs the B200 reference: embeddings identical, the
first platform difference is layer-0 self_attn.wqkv_a (rel 1.5e-2 with zero MI355X noise), i.e. the first FP8 dense
GEMM. The checkpoint's quantization_config is {fmt e4m3, weight_block_size [128,128], scale_fmt "ue8m0"} and DeepSeek's
reference act_quant rounds every activation scale up to a power of two (fast_round_scale); the B200 path (DeepGEMM)
does the same. AITER's per-1x128 quant (aiter_per1x128_quant and the fused RMSNorm+quant) uses exact fp32 scales
(amax / 448): a different FP8 grid than the one the model was built for, at every dense projection.

Runtime NUMINOUS_ACT_UE8M0=1 (inert otherwise):
  ue8m0_linear   layers/quantization/fp8_utils.py aiter_w8a8_block_fp8_linear: bf16 inputs are quantized per 1x128
                 with scale = 2^ceil(log2(amax / 448)) (amax floored at 1e-4, as the reference) in torch
  ue8m0_norm     models/deepseek_v4.py _fused_rmsnorm_fp8_quant: returns the bf16 RMSNorm output (reference formula)
                 as the "quantized operand", so the projection quantizes it through ue8m0_linear
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


UE8M0_FN = (
    "\n\n# numinous-hook:ue8m0_quant_fn (amdpro): reference (DeepSeek act_quant, scale_fmt ue8m0) 1x128 FP8 quant\n"
    "def _nm_ue8m0_per1x128_quant(x):\n"
    "    m, k = x.shape\n"
    "    xg = x.float().view(m, k // 128, 128)\n"
    "    amax = xg.abs().amax(-1).clamp(min=1e-4)\n"
    "    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))\n"
    "    q = (xg / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)\n"
    "    return q.view(m, k), s.contiguous()\n"
)

HOOKS = [
    Hook("ue8m0_linear", "srt/layers/quantization/fp8_utils.py",
         "        q_input, x_scale = aiter_per1x128_quant(\n"
         "            input_2d,\n"
         "            quant_dtype=aiter.dtypes.fp8,\n"
         "            transpose_scale=emit_bpreshuffle_scale,\n"
         "        )\n",
         "        # numinous-hook:ue8m0_linear (amdpro): NUMINOUS_ACT_UE8M0=1 -> the checkpoint's ue8m0 activation grid\n"
         "        if __import__(\"os\").environ.get(\"NUMINOUS_ACT_UE8M0\") == \"1\":\n"
         "            q_input, x_scale = _nm_ue8m0_per1x128_quant(input_2d)\n"
         "            emit_bpreshuffle_scale = False\n"
         "        else:\n"
         "            q_input, x_scale = aiter_per1x128_quant(\n"
         "                input_2d,\n"
         "                quant_dtype=aiter.dtypes.fp8,\n"
         "                transpose_scale=emit_bpreshuffle_scale,\n"
         "            )\n"),
    Hook("ue8m0_quant_fn", "srt/layers/quantization/fp8_utils.py",
         "def aiter_w8a8_block_fp8_linear(\n",
         UE8M0_FN.lstrip("\n") + "\n\ndef aiter_w8a8_block_fp8_linear(\n"),
    Hook("ue8m0_norm", "srt/models/deepseek_v4.py",
         "def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):\n",
         "def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):\n"
         "    # numinous-hook:ue8m0_norm (amdpro): with NUMINOUS_ACT_UE8M0=1 hand the projection the bf16 norm output\n"
         "    if __import__(\"os\").environ.get(\"NUMINOUS_ACT_UE8M0\") == \"1\":\n"
         "        _x = hidden_states.float()\n"
         "        _x = _x * torch.rsqrt(_x.square().mean(-1, keepdim=True) + eps)\n"
         "        _y = (weight.float() * _x).to(hidden_states.dtype)\n"
         "        return _y, _y\n"),
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
