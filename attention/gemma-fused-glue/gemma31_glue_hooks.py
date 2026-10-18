"""rtxsys D1 hook (Gemma 4 31B on H100, kernel program 2026-10-06): decode/verify glue fusion.

The dense Gemma4DecoderLayer (gemma4_causal.py:657-756, v0.5.20) runs five glue kernels per layer around its FP8
GEMMs: input_layernorm, the qkv_proj input quant, post_attention_layernorm, pre_feedforward_layernorm(h, residual),
the gate_up input quant, plus gemma_rmsnorm_residual_scalar at the end. With NUMINOUS_GEMMA_GLUE=1 they become two
Triton kernels per layer (operations/kernels/gemma31/rtx_glue.py, installed as srt/layers/numinous_gemma_glue.py):
  - post-attn chain: post_attention_layernorm -> + residual -> pre_feedforward_layernorm -> FP8 (q, s) for gate_up;
  - cross-layer chain: post_ff_norm -> + residual -> x layer_scalar -> NEXT layer's input_layernorm -> FP8 (q, s)
    for the next qkv_proj.
The (q, s) pairs enter deepgemm_w8a8_block_fp8_linear_with_fallback's pre-quantized branch (fp8_utils.py:1062-1088).

Hooks (anchored replace, v0.5.20, idempotent, the patched files must compile):
  gemma4_causal.py
    1. import: the glue module.
    2. input: take the prenormed input (cross-layer chain) instead of input_layernorm; clear the prequant stash
       after self_attn; run the post-attn chain instead of post_attention_layernorm when eligible.
    3. dense: skip pre_feedforward_layernorm when the chain ran; clear the prequant stash after mlp.
    4. output: run the cross-layer chain instead of gemma_rmsnorm_residual_scalar when the layer has a local
       successor.
    5. link: Gemma4TextModel.__init__ links each layer to its successor's input_layernorm / qkv_proj.
  fp8_utils.py
    6. prequant: at deepgemm_w8a8_block_fp8_linear_with_fallback's entry, a stashed (q, s) for this exact input
       becomes (input, input_scale). This anchor is disjoint from dp_fp8_passthrough's.
Env: NUMINOUS_GEMMA_GLUE=1 (chains), NUMINOUS_GEMMA_GLUE_QUANT=0 keeps the chains but leaves the quant to the linear.

    python3 gemma31_glue_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sglang_numinous_hooks import sglang_root  # noqa: E402

MARKER = "numinous-hook:gemma31_glue"
ENV = "NUMINOUS_GEMMA_GLUE"
INSTALL = [("../kernels/gemma31/rtx_glue.py", "srt/layers/numinous_gemma_glue.py")]
CELL_INSTALL = [("operations/kernels/gemma31/rtx_glue.py", "srt/layers/numinous_gemma_glue.py")]

GEMMA = "srt/models/gemma4_causal.py"
FP8U = "srt/layers/quantization/fp8_utils.py"

HOOKS = [
    (GEMMA, "import",
     "from sglang.srt.layers.layernorm import Gemma4RMSNorm, RMSNorm\n",
     "from sglang.srt.layers.layernorm import Gemma4RMSNorm, RMSNorm\n"
     "from sglang.srt.layers import numinous_gemma_glue as _ngg  # {marker} (import)\n"),
    (GEMMA, "input",
     "        residual = hidden_states\n\n"
     "        # Apply input layernorm\n"
     "        hidden_states = self.input_layernorm(hidden_states)\n"
     "        hidden_states = self.self_attn(\n"
     "            positions=positions,\n"
     "            hidden_states=hidden_states,\n"
     "            forward_batch=forward_batch,\n"
     "        )\n"
     "        hidden_states = self.post_attention_layernorm(hidden_states)\n",
     "        residual = hidden_states\n\n"
     "        # Apply input layernorm\n"
     "        _ngg_pre = _ngg.take_prenormed(hidden_states) if _ngg.ON else None  # {marker} (input)\n"
     "        if _ngg_pre is not None:\n"
     "            hidden_states = _ngg_pre\n"
     "        else:\n"
     "            hidden_states = self.input_layernorm(hidden_states)\n"
     "        hidden_states = self.self_attn(\n"
     "            positions=positions,\n"
     "            hidden_states=hidden_states,\n"
     "            forward_batch=forward_batch,\n"
     "        )\n"
     "        _ngg_pa = None\n"
     "        if _ngg.ON:\n"
     "            _ngg.clear_prequant()\n"
     "            _ngg_pa = _ngg.layer_post_attn(self, hidden_states, residual)\n"
     "        if _ngg_pa is not None:\n"
     "            hidden_states, residual = _ngg_pa\n"
     "        else:\n"
     "            hidden_states = self.post_attention_layernorm(hidden_states)\n"),
    (GEMMA, "dense",
     "        else:\n"
     "            # Fuse: hidden_states + residual -> residual; pre_ff_norm(residual) -> hidden_states\n"
     "            hidden_states, residual = self.pre_feedforward_layernorm(\n"
     "                hidden_states, residual\n"
     "            )\n"
     "            hidden_states = self.mlp(hidden_states)\n",
     "        else:\n"
     "            # Fuse: hidden_states + residual -> residual; pre_ff_norm(residual) -> hidden_states\n"
     "            if _ngg_pa is None:  # {marker} (dense)\n"
     "                hidden_states, residual = self.pre_feedforward_layernorm(\n"
     "                    hidden_states, residual\n"
     "                )\n"
     "            hidden_states = self.mlp(hidden_states)\n"
     "            if _ngg.ON:\n"
     "                _ngg.clear_prequant()\n"),
    (GEMMA, "output",
     "            # Fused: (post_ff_norm(h) + residual) * layer_scalar in one kernel\n"
     "            norm = self.post_feedforward_layernorm\n"
     "            hidden_states = gemma_rmsnorm_residual_scalar(\n",
     "            # Fused: (post_ff_norm(h) + residual) * layer_scalar in one kernel\n"
     "            norm = self.post_feedforward_layernorm\n"
     "            _ngg_out = _ngg.layer_output(self, hidden_states, residual) if _ngg.ON else None  # {marker} (output)\n"
     "            if _ngg_out is not None:\n"
     "                return _ngg_out, None\n"
     "            hidden_states = gemma_rmsnorm_residual_scalar(\n"),
    (GEMMA, "link",
     "            self.norm = PPMissingLayer()\n"
     "        self.layers_to_capture = []\n"
     "        self.post_init()\n",
     "            self.norm = PPMissingLayer()\n"
     "        self.layers_to_capture = []\n"
     "        _ngg.link_layers_full(self.layers, self.start_layer, self.end_layer)  # {marker} (link)\n"
     "        self.post_init()\n"),
    (FP8U, "import",
     "from sglang.srt.layers import deep_gemm_wrapper\n",
     "from sglang.srt.layers import deep_gemm_wrapper\n"
     "from sglang.srt.layers import numinous_gemma_glue as _ngg  # {marker} (import)\n"),
    (FP8U, "prequant",
     ") -> torch.Tensor:\n"
     "    if input_scale is not None:\n"
     "        # Pre-quantized activation (SGLANG_OPT_MOE_QUANT_ONCE): ``input`` is\n",
     ") -> torch.Tensor:\n"
     "    if input_scale is None and _ngg.ON:  # {marker} (prequant)\n"
     "        _ngg_pq = _ngg.take_prequant(input, block_size)\n"
     "        if _ngg_pq is not None:\n"
     "            input, input_scale = _ngg_pq\n"
     "    if input_scale is not None:\n"
     "        # Pre-quantized activation (SGLANG_OPT_MOE_QUANT_ONCE): ``input`` is\n"),
]


def numinous_preflight(ctx) -> list:
    """Where NUMINOUS_GEMMA_GLUE=1 is inert or partly inert (the hook falls back silently)."""
    if (ctx.env or {}).get(ENV) != "1":
        return []
    out, a = [], ctx.args or {}
    arch = " ".join(getattr(ctx.hf_config, "architectures", None) or []) if ctx.hf_config else ""
    if arch and "Gemma4" not in arch:
        out.append(f"{ENV}=1 on {arch}: the hook patches Gemma4DecoderLayer only, inert")
    dev = getattr(ctx, "device", None) or {}
    if dev and dev.get("is_hip"):
        out.append(f"{ENV}=1 on {dev.get('sku')}: the chains are CUDA Triton kernels; ROCm untested, inert by design")
    if dev and int(dev.get("sm_major", 9)) >= 10:
        out.append(f"{ENV}=1 on SM{dev.get('sm_major')}x: DeepGEMM uses UE8M0 scales there, so the chains run but the "
                   "fused FP8 quant stays off")
    backend = str(a.get("fp8_gemm_runner_backend") or a.get("fp8_gemm_backend") or "auto")
    if backend not in ("auto", "deep_gemm"):
        out.append(f"{ENV}=1 with FP8 GEMM backend {backend}: the fused quant needs the DeepGEMM linear; chains only")
    if (ctx.env or {}).get("NUMINOUS_GEMMA_NORM_QUANT") == "1":
        out.append(f"{ENV}=1 clashes with NUMINOUS_GEMMA_NORM_QUANT=1 (k3amd G1b, gemma31_norm_quant_hooks): both "
                   "patch the input-norm and pre-FFN anchors of gemma4_causal.py, so the second hook cannot apply. "
                   "D1 subsumes G1b (same two quant passes plus post-attn norm and the layer-output kernel)")
    return out


def apply(root: str, check: bool = False) -> list:
    out = []
    if not check:
        for rel, dst in INSTALL:
            src = os.path.normpath(os.path.join(HERE, rel))
            shutil.copy(src, os.path.join(root, dst))
            out.append(f"installed {dst}")
    for path_rel in (GEMMA, FP8U):
        path = os.path.join(root, path_rel)
        src = open(path).read()
        new = src
        for rel, name, anchor, text in HOOKS:
            if rel != path_rel:
                continue
            tag = f"{MARKER} ({name})"
            if tag in new:
                out.append(f"{path_rel} {name}: already applied")
                continue
            n = new.count(anchor)
            if n != 1:
                raise SystemExit(f"gemma31_glue hooks: {name} anchor found {n} times in {path_rel}")
            new = new.replace(anchor, text.replace("{marker}", MARKER), 1)
            out.append(f"{path_rel} {name}: {'anchor ok' if check else 'applied'}")
        if not check and new != src:
            compile(new, path, "exec")
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
