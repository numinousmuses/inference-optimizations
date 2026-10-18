"""Anchored hook: a hybrid-SSM idle DP rank with ZERO padded tokens stays IDLE instead of becoming an empty
TARGET_VERIFY batch (rtxsys, 2026-10-04; coordinator 13:50, Flash x_dpa4 step 1). Tree: SGLang 1330e437e1
(lmsysorg/sglang-rocm v0.5.21-rocm724-mi35x-20261002).

Root cause (source_code, forward_batch_info.ForwardBatch.prepare_mlp_sync_batch): for hybrid-SSM models (GLM-5.3-Flash:
KDA layers) with a target spec_info, an IDLE batch is rewritten to TARGET_VERIFY, with batch_size =
num_tokens // num_tokens_per_req, so that idle ranks replay the verify layout. Under CUDA graphs the DP padding is
MAX_LEN, so num_tokens > 0 and the idle rank runs a padded dummy verify. In eager mode with SUM_LEN padding, an
idle rank's num_tokens is 0, so it gets a TARGET_VERIFY batch with 0 requests and 0 tokens, which takes the extend
path. Every op that cannot take M = 0 then crashes in turn (measured by amdflash on mi355x8b,
flash_x_dpa4e_1002):
  #1 dsa_backend.init_forward_metadata seq_lens_cpu.max() on an empty tensor;
  #2 the ptpc FP8 linear ("HIP invalid configuration argument");
  #3 the bf16 qkv linear in glm5_next forward_qkvbfg (AITER "Unsupported N value: 64,4096,0").
Fix: rewrite only when num_tokens > 0. A 0-token idle rank keeps ForwardMode.IDLE, the same path non-hybrid DP
models take, where the attention and KDA layers return early on is_idle() and the MoE / DP collectives run with
0 rows. The graph path (num_tokens > 0) is unchanged. NUMINOUS_DP_EMPTY_IDLE=0 restores stock.

    python3 rocm_dp_empty_idle_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

PATH = "srt/model_executor/forward_batch_info.py"
NAME = "dp_empty_idle"
ANCHOR = ("                if self.forward_mode.is_idle():\n"
          "                    self._original_forward_mode = self.forward_mode\n"
          "                    self.forward_mode = ForwardMode.TARGET_VERIFY\n")
TEXT = ("                if self.forward_mode.is_idle() and (  # numinous-hook:dp_empty_idle (rtxsys)\n"
        "                    num_tokens > 0\n"
        "                    or __import__(\"os\").environ.get(\"NUMINOUS_DP_EMPTY_IDLE\", \"1\") != \"1\"\n"
        "                ):\n"
        "                    self._original_forward_mode = self.forward_mode\n"
        "                    self.forward_mode = ForwardMode.TARGET_VERIFY\n")


def sglang_root() -> str:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def patch_source(src: str) -> tuple[str, str]:
    if f"numinous-hook:{NAME}" in src:
        return src, f"{NAME}: already applied"
    n = src.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"{NAME}: anchor found {n} times in {PATH}")
    return src.replace(ANCHOR, TEXT, 1), f"{NAME}: applied to {PATH}"


def apply(root: str, check: bool = False) -> list[str]:
    path = os.path.join(root, PATH)
    src = open(path).read()
    new, msg = patch_source(src)
    if check:
        return [msg.replace("applied to", "anchor ok in") if new != src else msg]
    if new != src:
        compile(new, path, "exec")
        open(path, "w").write(new)
    return [msg]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    print("\n".join(apply(a.root or sglang_root(), a.check)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
