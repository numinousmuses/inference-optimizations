"""Insert Numinous hooks into an installed SGLang at named anchors, instead of shipping whole-file copies that break
on every release. Each hook names its file, an exact anchor, and the lines to insert; a missing anchor fails the
image build. Hooks are idempotent (a marker line is checked first).

    python3 sglang_numinous_hooks.py pro_dpmoe tactics         # patch the installed sglang package
    python3 sglang_numinous_hooks.py --root /tmp/sgl --check   # report which anchors resolve, change nothing

The modules the hooks import (numinous_dp_moe, symm_collectives, numinous_moe_tactics, tables) are installed next to
them by the overlay dict in gate_arms.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Hook:
    name: str
    group: str
    path: str  # relative to the sglang package directory
    anchor: str  # exact text; the insertion goes after it ("" = end of file)
    text: str

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


DPMOE_V4 = '''
        # numinous-hook:v4_dpmoe
        # Numinous fused DP-MoE (NUMINOUS_DP_MOE=1): route, quantize and run the shared expert on local rows, push
        # the MXFP8 rows + top-k to every peer, pull-reduce own rows (sglang/srt/layers/moe/numinous_dp_moe.py).
        if _use_tp_moe_gather:
            from sglang.srt.layers.moe import numinous_dp_moe as _ndm

            if _ndm.enabled():
                if get_parallel().attn_tp_size == 1 and get_parallel().tp_size == get_parallel().attn_dp_size:
                    if _ndm.covers(self.mlp, forward_batch):
                        return _ndm.forward(self.mlp, hidden_states, forward_batch, input_ids)
                else:
                    _ndm.log_once("layout", "numinous_dp_moe declined: needs attention TP 1 and TP size == DP size")'''

PREPARE_V4 = '''
        # numinous-hook:v4_prepare
        # Numinous fused DP-MoE: symmetric buffers now, before CUDA-graph capture (every rank loads together)
        from sglang.srt.layers.moe import numinous_dp_moe as _ndm

        if _ndm.enabled():
            # the top-k width the MoE layers route with: routed experts plus a fused shared column, if any
            _ndm.prepare(self.config.hidden_size,
                         self.config.num_experts_per_tok + getattr(self, "num_fused_shared_experts", 0))'''


def _tactics(name: str) -> str:
    return f'''


# numinous-hook:{name}
# Numinous: serve measured tactics for FlashInfer's TRT-LLM FP4 MoE kernel (DeepSeek V4 Pro MXFP4, GLM-5.3 NVFP4).
# With autotune off FlashInfer's fallback takes the smallest tile of its candidate set; the tables in
# sglang/srt/layers/moe/moe_tactics/ hold the fastest tactic per token count per GPU (moe_tactic_sweep.py).
try:
    from sglang.srt.layers.moe import numinous_moe_tactics as _numinous_moe_tactics

    _numinous_moe_tactics.install()
except Exception as _e:  # noqa: BLE001
    logger.warning("numinous_moe_tactics not installed: %s", _e)
'''


HOOKS = [
    Hook("v4_dpmoe", "pro_dpmoe", "srt/models/deepseek_v4.py",
         "        _use_tp_moe_gather = (\n"
         "            not _use_cp\n"
         "            and get_parallel().attn_dp_size > 1\n"
         "            and get_moe_a2a_backend().is_none()\n"
         "        )\n", DPMOE_V4.lstrip("\n") + "\n"),
    Hook("v4_prepare", "pro_dpmoe", "srt/models/deepseek_v4.py",
         "        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)\n",
         PREPARE_V4.lstrip("\n") + "\n"),
    Hook("tactics_runner", "tactics", "srt/layers/moe/moe_runner/flashinfer_trtllm.py", "",
         _tactics("tactics_runner")),
    Hook("tactics_mxfp4", "tactics", "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py", "",
         _tactics("tactics_mxfp4")),
]


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, groups: list[str], check: bool = False) -> list[str]:
    """Returns one status line per selected hook; raises SystemExit if an anchor is missing."""
    out, missing = [], []
    for h in HOOKS:
        if groups and h.group not in groups and h.name not in groups:
            continue
        path = os.path.join(root, h.path)
        src = open(path).read()
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        if h.anchor:
            n = src.count(h.anchor)
            if n != 1:
                missing.append(f"{h.name}: anchor found {n} times in {h.path}")
                continue
            new = src.replace(h.anchor, h.anchor + h.text, 1)
        else:
            new = src.rstrip("\n") + "\n" + h.text
        if check:
            out.append(f"{h.name}: anchor ok")
            continue
        compile(new, path, "exec")  # never leave a file that does not parse
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("numinous hooks: " + "; ".join(missing))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", help="hook groups or names (default: all)")
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.groups, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
