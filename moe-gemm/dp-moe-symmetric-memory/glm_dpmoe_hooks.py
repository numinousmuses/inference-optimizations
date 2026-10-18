"""Insert the GLM-5.3 fused DP-MoE hooks (NVFP4 payload) into an installed SGLang at named anchors, in the style of
sglang_numinous_hooks.py: each hook names its file, an exact anchor and the text inserted after it; a missing or
repeated anchor fails loudly; hooks are idempotent (marker line checked first); a patched file must compile.

    python3 glm_dpmoe_hooks.py                           # patch the installed sglang package
    python3 glm_dpmoe_hooks.py --root /tmp/sgl --check   # report which anchors resolve, change nothing

Hooks (SGLang v0.5.20, srt/models/deepseek_v2.py):
  glm_dpmoe    DeepseekV2DecoderLayer.forward, before prepare_mlp: when numinous_dp_moe_nvfp4 covers the layer and
               step, do the post-attention add + RMSNorm on local rows, run the fused MoE block and return, so the
               DP gather (prepare_mlp) and the DP scatter (postprocess_layer) never run.
  glm_prepare  DeepseekV2ForCausalLM.load_weights (the target and the NextN draft both load through it): allocate
               the symmetric buffers at load, before CUDA-graph capture.

Install next to them (the overlay): numinous_dp_moe_nvfp4.py, numinous_dp_moe.py and symm_collectives.py under
srt/layers/moe/ (the NVFP4 module reuses the Pro module's buffers / push / pull by import).
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sglang_numinous_hooks import Hook, sglang_root  # noqa: E402

GLM_DPMOE = '''
        # numinous-hook:glm_dpmoe
        # Numinous fused DP-MoE, NVFP4 payload (NUMINOUS_DP_MOE=1): add + norm, route and NVFP4-quantize local rows,
        # push them to every peer, routed experts, pull-reduce own rows; the DP gather and scatter are skipped
        # (sglang/srt/layers/moe/numinous_dp_moe_nvfp4.py). Declined layers and steps continue below unchanged.
        from sglang.srt.layers.moe import numinous_dp_moe_nvfp4 as _ndm4

        if _ndm4.enabled() and _ndm4.covers(self, forward_batch):
            return _ndm4.forward_layer(self, hidden_states, residual, forward_batch, topk_indices)
'''

GLM_PREPARE = '''
        # numinous-hook:glm_prepare
        # Numinous fused DP-MoE (NVFP4): symmetric buffers now, before CUDA-graph capture (every rank loads together)
        from sglang.srt.layers.moe import numinous_dp_moe_nvfp4 as _ndm4

        if _ndm4.enabled():
            _ndm4.prepare(self)
'''

HOOKS = [
    Hook("glm_dpmoe", "glm_dpmoe", "srt/models/deepseek_v2.py",
         "        maybe_prefetch_next_full_attention_kv(\n"
         "            forward_batch, next_full_attention_layer_id\n"
         "        )\n", GLM_DPMOE),
    Hook("glm_prepare", "glm_dpmoe", "srt/models/deepseek_v2.py",
         "        self.do_load_weights(weights, is_nextn)\n", GLM_PREPARE.lstrip("\n")),
]


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook; SystemExit if an anchor is missing or repeated. All hooks of a file are applied
    to its text together and the file is written once (never half-patched)."""
    out, missing, files = [], [], {}
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = files.get(path)
        if src is None:
            src = files[path] = open(path).read()
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        files[path] = src.replace(h.anchor, h.anchor + h.text, 1)
        out.append(f"{h.name}: {'anchor ok' if check else 'applied to ' + h.path}")
    if missing:
        raise SystemExit("glm_dpmoe hooks: " + "; ".join(missing))
    if not check:
        for path, new in files.items():
            if new == open(path).read():
                continue
            compile(new, path, "exec")  # never leave a file that does not parse
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
