"""Anchored hooks: feed the DP gather's FP8 payload to the shared-expert / dense DeepGEMM linear and the routed cutlass
MoE instead of requantizing (rtxsys, kernel program 2026-10-05). Tree: SGLang v0.5.20 (IMG_V0520).

    python3 dp_fp8_passthrough_hooks.py [--root <sglang pkg dir>] [--check]

The why, the exactness argument and the safety rules are in operations/kernels/dp_fp8_passthrough.py, installed as
srt/layers/numinous_fp8_passthrough.py. Measured target, F4 prefill TP0 per MoE layer: shared-expert quant 54 us +
routed quant 48 us after the FP8 AllGatherV's 49 us dequant.

Anchors:
- dp_attention.py:
  - clear: top of `_dp_gather`, so every gather path drops the old payload;
  - stash: after the dequant in `_dp_gather_via_all_gatherv_fp8`;
  - fwd_clear: top of `set_dp_buffer_len` (every forward start).
- communicator.py attn_clear: top of `LayerCommunicator.prepare_attn`, so no stash survives into an attention block.
  This is the FX1 fix: the KDA qkv_proj took a stale MoE stash.
- fp8_utils.py linear: the DeepGEMM w8a8 linear quant (`deepgemm_w8a8_block_fp8_linear_with_fallback`, non-MUSA,
  non-UE8M0 only).
- cutlass_moe.py routed: the activation quant in `cutlass_fused_experts_fp8`, the non-mxfp8 branch.
- moe_runner/deep_gemm.py deepgemm: the unpermuted activation quant in `pre_permute_standard_to_deep_gemm`
  (deep_gemm.py:1081-1092; row-major group-128 on SM90, the same format as the routed consumer). This is the prefill
  role's MoE input when it runs --moe-runner-backend deep_gemm (F7-b2). Saves the routed requant (48 us/layer in the F4
  profile, ~-2 ms per 4,096-token step there).
Env NUMINOUS_DP_FP8_PASSTHROUGH=1 (prefill role, with SGLANG_ENABLE_DP_GATHER_FP8=1). Unset: the module's entry points
return None and the stock quant runs.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
NAME = "dp_fp8_passthrough"
TAG = f"# numinous-hook:{NAME}"
MODULE_SRC = "operations/kernels/dp_fp8_passthrough.py"
MODULE_DST = "srt/layers/numinous_fp8_passthrough.py"
IMP = "from sglang.srt.layers import numinous_fp8_passthrough as _nfp"

# (name, rel path, anchor, replacement)
HOOKS = [
    ("clear", "srt/layers/dp_attention.py",
     "    is_partial: bool,\n):\n    _note_dp_gather_in_prefill_graph()\n",
     "    is_partial: bool,\n):\n    _note_dp_gather_in_prefill_graph()\n"
     f"    {IMP}  {TAG}\n"
     "    _nfp.clear()\n"),
    ("fwd_clear", "srt/layers/dp_attention.py",
     "    global_num_tokens_gpu: Optional[torch.Tensor] = None,\n):\n    _DpGatheredBufferWrapper.set_dp_buffer_len(\n",
     "    global_num_tokens_gpu: Optional[torch.Tensor] = None,\n):\n"
     f"    {IMP}  {TAG}\n"
     "    _nfp.clear()\n"
     "    _DpGatheredBufferWrapper.set_dp_buffer_len(\n"),
    ("attn_clear", "srt/layers/communicator.py",
     "        quant_format: str = \"\",\n"
     "        post_residual_addition: Optional[torch.Tensor] = None,\n"
     "    ):\n"
     "        # residual is None marks the first decoder layer, where the SP region\n",
     "        quant_format: str = \"\",\n"
     "        post_residual_addition: Optional[torch.Tensor] = None,\n"
     "    ):\n"
     f"        {IMP}  {TAG}\n"
     "        _nfp.clear()   # no DP-gather payload may reach an attention projection (FX1 fix)\n"
     "        # residual is None marks the first decoder layer, where the SP region\n"),
    ("stash", "srt/layers/dp_attention.py",
     "        GROUP=_DP_GATHER_FP8_GROUP,\n        BLOCK=2048,\n    )\n",
     "        GROUP=_DP_GATHER_FP8_GROUP,\n        BLOCK=2048,\n    )\n"
     f"    {IMP}  {TAG}\n"
     "    _nfp.stash(global_tokens, gq, gs)\n"),
    ("linear", "srt/layers/quantization/fp8_utils.py",
     "    if not _is_musa:\n"
     "        q_input, x_scale = sglang_per_token_group_quant_fp8(\n"
     "            input_2d,\n"
     "            block_size[1],\n"
     "            column_major_scales=True,\n"
     "            scale_tma_aligned=True,\n"
     "            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,\n"
     "        )\n",
     f"    _nfp_pre = None  {TAG}\n"
     "    if not _is_musa and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:\n"
     f"        {IMP}\n"
     "        _nfp_pre = _nfp.for_linear(input_2d, block_size[1])\n"
     "    if _nfp_pre is not None:\n"
     "        q_input, x_scale = _nfp_pre\n"
     "    elif not _is_musa:\n"
     "        q_input, x_scale = sglang_per_token_group_quant_fp8(\n"
     "            input_2d,\n"
     "            block_size[1],\n"
     "            column_major_scales=True,\n"
     "            scale_tma_aligned=True,\n"
     "            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,\n"
     "        )\n"),
    ("deepgemm", "srt/layers/moe/moe_runner/deep_gemm.py",
     "        packed_input_source, packed_input_source_scale = (\n"
     "            sglang_per_token_group_quant_fp8(\n"
     "                hidden_states,\n"
     "                block_k,\n"
     "                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,\n"
     "                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,\n"
     "                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,\n"
     "            )\n"
     "        )\n",
     f"        _nfp_pre = None  {TAG}\n"
     "        if block_k == 128 and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:\n"
     f"            {IMP}\n"
     "            _nfp_pre = _nfp.for_routed(hidden_states)   # row-major group-128 (q, s), as below on SM90\n"
     "        if _nfp_pre is not None:\n"
     "            packed_input_source, packed_input_source_scale = _nfp_pre\n"
     "        else:\n"
     "            packed_input_source, packed_input_source_scale = (\n"
     "                sglang_per_token_group_quant_fp8(\n"
     "                    hidden_states,\n"
     "                    block_k,\n"
     "                    column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,\n"
     "                    scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,\n"
     "                    scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,\n"
     "                )\n"
     "            )\n"),
    ("routed", "srt/layers/moe/cutlass_moe.py",
     "        a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128)\n",
     f"        {IMP}  {TAG}\n"
     "        _nfp_pre = _nfp.for_routed(a)\n"
     "        if _nfp_pre is not None:\n"
     "            a_q, a1_scale = _nfp_pre\n"
     "        else:\n"
     "            a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128)\n"),
]


def apply(root: str, check: bool = False, install_module: bool = True) -> list[str]:
    out = []
    by_file: dict[str, list] = {}
    for h in HOOKS:
        by_file.setdefault(h[1], []).append(h)
    for rel, hooks in by_file.items():
        f = os.path.join(root, rel)
        src = open(f).read()
        if TAG in src:
            out.append(f"{NAME}: already applied in {rel}")
            continue
        for name, _, anchor, _ in hooks:
            n = src.count(anchor)
            if n != 1:
                raise SystemExit(f"{NAME}.{name}: anchor found {n} times in {rel}")
        if check:
            out.append(f"{NAME}: {len(hooks)} anchor(s) ok in {rel}")
            continue
        for _, _, anchor, text in hooks:
            src = src.replace(anchor, text)
        open(f, "w").write(src)
        py_compile.compile(f, doraise=True)
        out.append(f"{NAME}: applied {','.join(h[0] for h in hooks)} to {rel}")
    if install_module and not check:
        dst = os.path.join(root, MODULE_DST)
        shutil.copy(REPO / MODULE_SRC, dst)
        py_compile.compile(dst, doraise=True)
        out.append(f"{NAME}: installed {MODULE_DST}")
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
