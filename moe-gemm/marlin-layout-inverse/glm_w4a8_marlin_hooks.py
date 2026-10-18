"""Serve W4AFP8 (int4 g128) MoE experts through Marlin W4A16 (fused_marlin_moe) instead of cutlass_w4a8, opt-in per
server with NUMINOUS_W4A8_MARLIN=1 (k3amd, GLM MoE GEMM owner; coordinator 2026-10-05).

Why (measured): the P2 decode-role profile puts cutlass_w4a8_moe_mm at 40.8 ms of a 70 ms EAGLE cycle (160 calls,
258.5 us median; w13 at 65% of HBM, w2 K=512 at 33%; rtxattn, weights-cache:/verify_logs/prof_P2_20261005-051443).
The decode MoE is weight-bandwidth bound and needs no FP8 activations; Marlin reads the same int4 bytes. The prefill
role keeps cutlass W4A8 (FP8 MMA, 1.40x per-rank prefill vs Marlin W4A16, measured 2026-10-04), so set the env on the
decode role of an in-box PD cell only. The winner between the two is decided by operations/kernels/
glm_moe_decode_bench.py on 1x H200 at the served shapes, including a numerics check of the conversion.

Hooks (anchored replace on SGLang v0.5.20 srt/layers/quantization/w4afp8.py; a missing or repeated anchor fails
loudly; idempotent by marker; the patched file must compile):
  1. W4AFp8MoEMethod.process_weights_after_loading: with the env set, convert w13/w2 (int8 pairs [E, N, K/2]) and their
     fp32 group scales to Marlin layout (glm_w4a8_marlin.w4afp8_to_marlin: GPTQ uint4b8 int32 + gptq_marlin_moe_repack +
     marlin_moe_permute_scales), REPLACE the cutlass tensors (no extra expert memory), mark the layer, skip the
     cutlass scale interleave.
  2. W4AFp8MoEMethod.apply: a marked layer runs fused_marlin_moe (bf16 activations, routed_scaling_factor as served).
The converter module is installed next to w4afp8.py (srt/layers/quantization/glm_w4a8_marlin.py).

    python3 glm_w4a8_marlin_hooks.py                     # patch the installed sglang package
    python3 glm_w4a8_marlin_hooks.py --root /tmp/sgl --check
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sglang_numinous_hooks import sglang_root  # noqa: E402

MARKER = "numinous-hook:glm_w4a8_marlin"
MODULE = ("glm_w4a8_marlin.py", "srt/layers/quantization/glm_w4a8_marlin.py")
# NUMINOUS_W4A8_MARLIN=dispatch: the Marlin -> cutlass rebuild for batches above NUMINOUS_W4A8_DISPATCH_M
INVERSE = ("../kernels/glm_marlin_inverse.py", "srt/layers/quantization/glm_marlin_inverse.py")


@dataclass(frozen=True)
class Replace:
    name: str
    path: str
    anchor: str
    text: str


HOOKS = [
    Replace(
        "convert", "srt/layers/quantization/w4afp8.py",
        "    def process_weights_after_loading(self, layer: Module) -> None:\n"
        "        dtype = torch.bfloat16\n"
        "        device = layer.w2_weight.device\n",
        "    def process_weights_after_loading(self, layer: Module) -> None:\n"
        "        dtype = torch.bfloat16\n"
        "        device = layer.w2_weight.device\n"
        "        import os as _os  # {marker}\n"
        "\n"
        "        _nm_mode = _os.environ.get(\"NUMINOUS_W4A8_MARLIN\")\n"
        "        if _nm_mode in (\"1\", \"dispatch\") and layer.w13_weight.dtype == torch.int8:\n"
        "            # the same int4 g128 bytes in Marlin layout, replacing the cutlass tensors\n"
        "            from sglang.srt.layers.quantization.glm_w4a8_marlin import w4afp8_to_marlin\n"
        "\n"
        "            _nm_e, _nm_n13, _nm_k13 = layer.w13_weight.shape[0], layer.w13_weight.shape[1], 2 * layer.w13_weight.shape[2]\n"
        "            _nm_n2, _nm_k2 = layer.w2_weight.shape[1], 2 * layer.w2_weight.shape[2]\n"
        "            w13, s13 = w4afp8_to_marlin(layer.w13_weight.data, layer.w13_weight_scale_inv.data, dtype=dtype)\n"
        "            layer.w13_weight = Parameter(w13, requires_grad=False)\n"
        "            layer.w13_weight_scale_inv = Parameter(s13, requires_grad=False)\n"
        "            w2, s2 = w4afp8_to_marlin(layer.w2_weight.data, layer.w2_weight_scale_inv.data, dtype=dtype)\n"
        "            layer.w2_weight = Parameter(w2, requires_grad=False)\n"
        "            layer.w2_weight_scale_inv = Parameter(s2, requires_grad=False)\n"
        "            layer.numinous_marlin = True\n"
        "            if _nm_mode == \"dispatch\":\n"
        "                # batches above NUMINOUS_W4A8_DISPATCH_M run cutlass_w4a8 from a shared one-layer scratch rebuilt\n"
        "                # from these Marlin tensors (glm_marlin_inverse); its input scales are processed as below\n"
        "                from sglang.srt.layers.quantization.glm_marlin_inverse import dispatch_scratch\n"
        "\n"
        "                layer.numinous_dispatch = dispatch_scratch(_nm_e, _nm_n13, _nm_k13, _nm_n2, _nm_k2, device)\n"
        "                layer.w13_input_scale = Parameter(torch.tensor(\n"
        "                    [layer.w13_input_scale.max().to(torch.float32).item()], dtype=torch.float32, device=device),\n"
        "                    requires_grad=False)\n"
        "                layer.w2_input_scale = Parameter(torch.tensor(\n"
        "                    [layer.w2_input_scale.max().to(torch.float32).item()], dtype=torch.float32, device=device),\n"
        "                    requires_grad=False)\n"
        "            torch.cuda.empty_cache()\n"
        "            return\n",
    ),
    Replace(
        "apply", "srt/layers/quantization/w4afp8.py",
        "        x = dispatch_output.hidden_states\n"
        "        topk_output = dispatch_output.topk_output\n"
        "        topk_weights, topk_ids, _ = topk_output\n"
        "\n"
        "        output = cutlass_w4a8_moe(\n",
        "        x = dispatch_output.hidden_states\n"
        "        topk_output = dispatch_output.topk_output\n"
        "        topk_weights, topk_ids, _ = topk_output\n"
        "\n"
        "        _nm_d = getattr(layer, \"numinous_dispatch\", None)  # {marker}\n"
        "        if _nm_d is not None and x.shape[0] > _nm_d.threshold:\n"
        "            # size dispatch: a prefill-sized batch runs cutlass_w4a8 (+ tuned tiles) on this layer's cutlass\n"
        "            # tensors, rebuilt from the resident Marlin copy into the shared scratch\n"
        "            _nm_w13, _nm_w2, _nm_s13, _nm_s2 = _nm_d.cutlass_weights(layer)\n"
        "            output = cutlass_w4a8_moe(\n"
        "                x, _nm_w13, _nm_w2, _nm_s13, _nm_s2, topk_weights, topk_ids,\n"
        "                self.a_strides1, self.b_strides1, self.c_strides1, self.a_strides2, self.b_strides2,\n"
        "                self.c_strides2, self.s_strides13, self.s_strides2, self.expert_offsets, self.problem_sizes1,\n"
        "                self.problem_sizes2, layer.w13_input_scale, layer.w2_input_scale,\n"
        "                routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,\n"
        "            )\n"
        "            return StandardCombineInput(hidden_states=output)\n"
        "        if getattr(layer, \"numinous_marlin\", False):\n"
        "            from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe\n"
        "\n"
        "            # gating_output is only shape-checked by fused_marlin_moe (fused_marlin_moe.py:195)\n"
        "            # under EP the dispatcher hands LOCAL ids (-1 = another rank's expert); Marlin then needs the\n"
        "            # mapping and the global expert count, as SGLang's Marlin runner passes them (moe_runner/marlin.py)\n"
        "            _nm_map = getattr(getattr(layer, \"dispatcher\", None), \"local_expert_mapping\", None)\n"
        "            output = fused_marlin_moe(\n"
        "                x, layer.w13_weight, layer.w2_weight, layer.w13_weight_scale_inv,\n"
        "                layer.w2_weight_scale_inv, topk_weights, topk_weights, topk_ids,\n"
        "                global_num_experts=(layer.dispatcher.num_experts if _nm_map is not None\n"
        "                                    else layer.w13_weight.shape[0]),\n"
        "                expert_map=_nm_map, num_bits=4, is_k_full=True,\n"
        "                routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,\n"
        "            )\n"
        "            return StandardCombineInput(hidden_states=output)\n"
        "\n"
        "        output = cutlass_w4a8_moe(\n",
    ),
]


def numinous_preflight(ctx) -> list:
    """CPU preflight (operations/hopper/hook_preflight.py): NUMINOUS_W4A8_MARLIN=1 converts W4AFP8 (int8-packed) MoE
    weights at load and runs fused_marlin_moe. DeepEP on Marlin is a boot error (measured), so an a2a backend is refused;
    a non-W4AFP8 checkpoint leaves the flag inert. EP with a2a none is correct with the dispatcher's local ids (moe-r5:
    rel 0.0052 vs fp32 under EP8; the round-1 illegal access came from passing GLOBAL ids) but slower than cutlass +
    tiles from m 96 at EP8 shapes (437 vs 266 us at m 144)."""
    mode = ctx.env.get("NUMINOUS_W4A8_MARLIN")
    if mode not in ("1", "dispatch"):
        return [f"NUMINOUS_W4A8_MARLIN={mode!r}: expected 1 (Marlin everywhere) or dispatch"] if mode else []
    out = []
    if mode == "dispatch":
        try:
            m = int(ctx.env.get("NUMINOUS_W4A8_DISPATCH_M", "0") or 0)
        except ValueError:
            m = 0
        if m <= 0:
            out.append("NUMINOUS_W4A8_MARLIN=dispatch needs NUMINOUS_W4A8_DISPATCH_M=<tokens> > 0 (the Marlin / cutlass "
                       "crossover); the server would refuse at load")
    quant = str(ctx.args.get("quantization") or getattr(ctx, "checkpoint_quant", "") or "")
    if quant and quant != "w4afp8":
        out.append(f"NUMINOUS_W4A8_MARLIN=1 on a {quant} server: only W4AFP8 int4 weights are converted (inert)")
    a2a = str(ctx.args.get("moe_a2a_backend") or "none")
    if a2a != "none":
        out.append(f"NUMINOUS_W4A8_MARLIN=1 with moe_a2a_backend {a2a}: DeepEP on Marlin is a boot error (measured)")
    return out


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook; SystemExit if an anchor is missing or repeated. The converter module is installed
    first; all hooks of a file are applied to its text together and the file is written once."""
    out, missing, files = [], [], {}
    dst = os.path.join(root, MODULE[1])
    if not check:
        shutil.copy(os.path.join(HERE, MODULE[0]), dst)
        out.append(f"installed {MODULE[1]}")
        inv = os.path.normpath(os.path.join(HERE, INVERSE[0]))
        if os.path.exists(inv):       # staged only by cells that dispatch (hopper_cells.MARLIN_DISPATCH files)
            shutil.copy(inv, os.path.join(root, INVERSE[1]))
            out.append(f"installed {INVERSE[1]}")
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = files.get(path)
        if src is None:
            src = files[path] = open(path).read()
        tag = f"{MARKER} ({h.name})"
        if tag in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        files[path] = src.replace(h.anchor, h.text.replace("{marker}", tag), 1)
        out.append(f"{h.name}: {'anchor ok' if check else 'applied to ' + h.path}")
    if missing:
        raise SystemExit("glm_w4a8_marlin hooks: " + "; ".join(missing))
    if not check:
        for path, new in files.items():
            if new == open(path).read():
                continue
            compile(new, path, "exec")
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
