"""Anchored patcher: Gemma 4 routing under the TRT-LLM FP4 MoE runner (SGLang v0.5.20, SM100). Style follows
operations/patches/gemma_extend_hooks.py.

Root cause (source_code, v0.5.20; measured on B200 2026-09-29 as an accuracy FAIL, top-1 agreement 0.864 / KL 0.44
against a deterministic control, GSM8K unchanged):
  Gemma4MoE (srt/models/gemma4_causal.py) routes with a custom_routing_function: softmax over all 128 experts,
  top-8, renormalise (= softmax of the top-8 logits), times the checkpoint's router.per_expert_scale. Its TopK has
  no output_format and its FusedMoE no routing_method_type. Under `--moe-runner-backend flashinfer_trtllm`,
  TopK.forward_cuda returns the BYPASSED format (srt/layers/moe/topk.py), so the custom routing never runs and
  trtllm_fp4_block_scale_moe routes the raw logits itself with RoutingMethodType.Default (0: softmax over all
  experts -> top-k, NO renormalisation) and no per-expert scale. The chosen experts are the same (argmax of the same
  logits), so GSM8K survives; every expert weight is scaled by sum(top-8 softmax) (< 1, token dependent) and the
  per-expert scale is dropped, so the output distribution moves (KL 0.44).

Fix (hook `trtmoe_routing`): when the MoE runner backend is flashinfer_trtllm, build Gemma4MoE's TopK with
output_format=STANDARD. The custom routing then runs (fused kernel, ids + fp32 weights) and the runner takes its
existing routed path, trtllm_fp4_block_scale_routed_moe((topk_ids, topk_weights), ...), which consumes the
weights as given (FlashInfer 0.6.18 accepts the unpacked tuple). Nothing changes for any other backend or model.
SGLANG_NUMINOUS_GEMMA_TRTMOE_ROUTING=0 disables the fix (the probe's "stock trtllm" arm).

Probe support (hook `moe_dump`, inert unless NUMINOUS_MOE_DUMP=<dir>): Gemma4MoE.forward of the layers in
NUMINOUS_MOE_DUMP_LAYERS (default "0") saves its input hidden states, router logits, the TopK output (ids/weights
when STANDARD) and the MoE output for forwards with >= 8 tokens, into <dir>/moe_l<layer>_<n>.pt (first 4 per
layer after the directory is emptied). Consumed by
operations/modal/gemma_trtmoe_probe.py, which compares cutlass / trtllm / trtllm-fixed outputs on identical
inputs against a dequantised fp32 reference of layer 0's experts under both routings.

    python3 operations/patches/gemma_trtmoe_hooks.py            # patch installed sglang
    python3 operations/patches/gemma_trtmoe_hooks.py --check    # anchors only
    python3 operations/patches/gemma_trtmoe_hooks.py --root /tmp/sgl
Test: operations/patches/test_gemma_trtmoe_hooks.py (applies to a copy of gemma4_causal.py, compiles, checks
the inserted call sites; on a box it also imports the patched module).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

GEMMA4_PATH = "srt/models/gemma4_causal.py"

_TOPK_FORMAT = """\
            # numinous-hook:trtmoe_routing
            output_format=_numinous_gemma4_topk_output_format(),
"""

_DUMP_IN = """\
        # numinous-hook:moe_dump
        _numinous_moe_in = hidden_states
"""

_DUMP_OUT = """\
        _numinous_moe_dump(self, _numinous_moe_in, router_logits, topk_output, hidden_states)
"""

_HELPERS = """

# numinous-hook:trtmoe_helpers
# ---- numinous Gemma 4 TRT-LLM MoE routing helpers (operations/patches/gemma_trtmoe_hooks.py) --------------------
def _numinous_gemma4_topk_output_format():
    \"\"\"STANDARD under the flashinfer_trtllm MoE runner, so Gemma 4's custom routing (softmax over all experts,
    top-k, renormalise, per-expert scale) runs and the runner takes the routed kernel with those weights; None
    (stock behaviour) everywhere else. SGLANG_NUMINOUS_GEMMA_TRTMOE_ROUTING=0 restores stock (kernel routing with
    RoutingMethodType.Default: no renormalisation, no per-expert scale).\"\"\"
    import os as _os
    if _os.environ.get("SGLANG_NUMINOUS_GEMMA_TRTMOE_ROUTING", "1") == "0":
        return None
    try:
        from sglang.srt.layers.moe import get_moe_runner_backend as _get_backend
        from sglang.srt.layers.moe.topk import TopKOutputFormat as _Fmt
        if _get_backend().is_flashinfer_trtllm():
            return _Fmt.STANDARD
    except Exception:  # noqa: BLE001  (no runtime context: stock behaviour)
        return None
    return None


_NUMINOUS_MOE_DUMPS = 0


def _numinous_moe_dump(moe, hidden_in, router_logits, topk_output, out):
    \"\"\"Probe only: NUMINOUS_MOE_DUMP=<dir> saves layer 0's MoE input/routing/output for forwards of >= 8 tokens
    (first 4 after the directory is emptied). Never raises into the forward.\"\"\"
    import os as _os
    d = _os.environ.get("NUMINOUS_MOE_DUMP")
    if not d or hidden_in.shape[0] < 8:
        return
    layers = [int(x) for x in _os.environ.get("NUMINOUS_MOE_DUMP_LAYERS", "0").split(",") if x.strip()]
    if moe.layer_id not in layers:
        return
    global _NUMINOUS_MOE_DUMPS
    try:
        import torch as _torch
        _os.makedirs(d, exist_ok=True)
        existing = [f for f in _os.listdir(d) if f.startswith(f"moe_l{moe.layer_id}_")]
        if len(existing) >= 4:
            return
        _NUMINOUS_MOE_DUMPS += 1
        rec = {"hidden_in": hidden_in.detach().cpu(), "router_logits": router_logits.detach().cpu(),
               "out": out.detach().cpu(), "per_expert_scale": moe.per_expert_scale.detach().float().cpu(),
               "topk_format": type(topk_output).__name__, "layer": moe.layer_id}
        ids = getattr(topk_output, "topk_ids", None)
        w = getattr(topk_output, "topk_weights", None)
        if ids is not None and w is not None:
            rec["topk_ids"] = ids.detach().cpu()
            rec["topk_weights"] = w.detach().float().cpu()
        _torch.save(rec, _os.path.join(d, f"moe_l{moe.layer_id}_{len(existing)}.pt"))
    except Exception as _e:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).warning("numinous moe dump failed: %s", _e)
# ---- end numinous Gemma 4 TRT-LLM MoE routing helpers ------------------------------------------------------------
"""

HOOKS = [
    {"name": "trtmoe_routing", "path": GEMMA4_PATH,
     "anchor": "            custom_routing_function=routing_function,\n", "text": _TOPK_FORMAT},
    {"name": "moe_dump", "path": GEMMA4_PATH,
     "anchor": "        num_tokens, hidden_dim = hidden_states.shape\n        topk_output = self.topk(hidden_states, router_logits)\n",
     "text": None},   # two-part insert, see apply()
    {"name": "trtmoe_helpers", "path": GEMMA4_PATH, "anchor": "", "text": _HELPERS},
]
_DUMP_OUT_ANCHOR = "        hidden_states = self.experts(hidden_states, topk_output)\n"


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook. SystemExit when an anchor is missing or not unique."""
    out, missing = [], []
    for h in HOOKS:
        marker = f"# numinous-hook:{h['name']}"
        path = os.path.join(root, h["path"])
        src = open(path).read()
        if marker in src:
            out.append(f"{h['name']}: already applied")
            continue
        anchors = [h["anchor"]] if h["anchor"] else []
        if h["name"] == "moe_dump":
            anchors.append(_DUMP_OUT_ANCHOR)
        bad = [a for a in anchors if src.count(a) != 1]
        if bad:
            missing.append(f"{h['name']}: anchor found {src.count(bad[0])} times in {h['path']}")
            continue
        if check:
            out.append(f"{h['name']}: anchor ok")
            continue
        if h["name"] == "moe_dump":
            # keep the MoE input: the forward reassigns hidden_states with the experts' output
            new = src.replace(h["anchor"], _DUMP_IN + h["anchor"], 1)
            new = new.replace(_DUMP_OUT_ANCHOR, _DUMP_OUT_ANCHOR + _DUMP_OUT, 1)
        elif h["anchor"]:
            new = src.replace(h["anchor"], h["anchor"] + h["text"], 1)
        else:
            new = src.rstrip("\n") + "\n" + h["text"]
        compile(new, path, "exec")
        if not os.path.exists(path + ".numinous.orig"):
            shutil.copyfile(path, path + ".numinous.orig")   # pristine copy for the test / a manual revert
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"{h['name']}: applied to {h['path']}")
    if missing:
        raise SystemExit("gemma_trtmoe_hooks: " + "; ".join(missing))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="", help="sglang package root (default: installed)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
