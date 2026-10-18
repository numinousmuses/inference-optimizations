"""Hopper (SM90) port of the Numinous GQA-packed extend attention for Gemma 4, env-gated (agent rtxv41, 2026-10-05).

Layered on gemma_extend_hooks.py. That hook installs numinous_extend_gqa.py and the dispatch block into SGLang
v0.5.20's extend_attention_fwd, gated to SM120 with SM120 tiles. This hook only edits the two helpers that hook
appended. It must run after gemma_extend_hooks; it refuses otherwise.
  1. `_numinous_gqa_enabled()` additionally returns True on SM90 when SGLANG_NUMINOUS_EXTEND_GQA_SM90=1 (default
     off). SM120 behaviour is unchanged, as is SGLANG_NUMINOUS_EXTEND_GQA=0 (off everywhere).
  2. `_numinous_gqa_config()` gets an SM90 table. H100 has 227 KB of shared memory per block versus SM120's ~99 KB,
     so the pipelined tiles that SM120 could not fit are used, inside 227 KB even with BF16 K/V tiles:
       - head_dim >= 512 or group >= 4: block_t 8, block_n 128, dk 64, dv 256, 8 warps, 2 stages
         (per stage: K 128x64 + V 128x256 BF16 = 80 KB, x2 = 160 KB);
       - else (group 2): block_t 32, block_n 64, dk 64, dv = head_dim, 8 warps, 2 stages.
     SGLANG_NUMINOUS_GQA_CFG overrides keys, as on SM120.
Where it engages on Gemma 4 31B (source_code, v0.5.20 triton_backend.py:1671-1682):
  - the 10 full-attention layers (head_dim 512, 32 query / 4 KV heads = group 8, no window, page 1);
  - the sliding layers pass window_kv_offsets under the hybrid-SWA pool, which the dispatch guard excludes, so they
    stay on the stock kernel. That kernel already skips out-of-window tiles (extend_attention.py:514-527, 706-716).
Gain (derived): SM120 measured 2.72x the stock Triton kernel on the 512-dim global layers (rtx-gemma). On H100 the
global layers are ~7% of a 2.3K-token prefill at ~150 TFLOPS attention, so 2.7x is ~+4.5% prefill (+3% warm at the
OpenRouter mix), and ~+12% prefill at 8K prompts. Untuned on H100: a compile or launch failure inside the dispatch's
try/except falls back to stock and latches _NUMINOUS_GQA_FAILED, so it can never take the server down.

    python3 gemma_extend_sm90_hooks.py [--root <sglang pkg>] [--check]
"""
from __future__ import annotations

import argparse
import os
import sys

NAME = "gemma_extend_sm90"
MARKER = f"numinous-hook:{NAME}"
PATH = "kernels/ops/attention/extend_attention.py"

ENABLED_ANCHOR = "    return _is_cuda and CUDA_CAPABILITY[0] == 12\n"
ENABLED_TEXT = (f"    # {MARKER}: Hopper only when explicitly enabled (untuned until measured on an H100)\n"
                "    if _is_cuda and CUDA_CAPABILITY[0] == 9:\n"
                "        return _os.environ.get(\"SGLANG_NUMINOUS_EXTEND_GQA_SM90\", \"0\") == \"1\"\n"
                "    return _is_cuda and CUDA_CAPABILITY[0] == 12\n")

CONFIG_ANCHOR = ("    import os as _os\n"
                 "    if head_dim >= 512:\n"
                 "        cfg = dict(block_t=8, block_n=128, dk=64, dv=256, num_warps=8, num_stages=3)\n")
CONFIG_TEXT = ("    import os as _os\n"
               f"    if CUDA_CAPABILITY[0] == 9:  # {MARKER}: H100 table (227 KB smem: 2-stage pipelines fit)\n"
               "        if head_dim >= 512 or group >= 4:\n"
               "            cfg = dict(block_t=8, block_n=128, dk=64, dv=min(256, head_dim), num_warps=8, num_stages=2)\n"
               "        else:\n"
               "            cfg = dict(block_t=32, block_n=64, dk=64, dv=head_dim, num_warps=8, num_stages=2)\n"
               "    elif head_dim >= 512:\n"
               "        cfg = dict(block_t=8, block_n=128, dk=64, dv=256, num_warps=8, num_stages=3)\n")
PREREQ = "# numinous-hook:gqa_helpers"


def patch_source(src: str) -> str:
    if PREREQ not in src:
        raise SystemExit(f"{NAME}: run gemma_extend_hooks.py first ({PREREQ} not found in {PATH})")
    for anchor in (ENABLED_ANCHOR, CONFIG_ANCHOR):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"{NAME}: anchor found {n} times in {PATH}: {anchor[:70]!r}")
    return src.replace(ENABLED_ANCHOR, ENABLED_TEXT, 1).replace(CONFIG_ANCHOR, CONFIG_TEXT, 1)


def apply(root: str, check: bool = False) -> list[str]:
    f = os.path.join(root, PATH)
    src = open(f).read()
    if MARKER in src:
        return [f"{NAME}: already applied"]
    out = patch_source(src)
    compile(out, f, "exec")
    if check:
        return [f"{NAME}: anchors ok in {PATH}"]
    tmp = f"{f}.numinous.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(out)
    os.replace(tmp, f)
    return [f"{NAME}: applied to {PATH} (SM90 gate SGLANG_NUMINOUS_EXTEND_GQA_SM90=1, H100 tiles)"]


def numinous_preflight(ctx) -> list:
    """With SGLANG_NUMINOUS_EXTEND_GQA_SM90=1 the port only acts with the triton attention backend on SM90."""
    env = dict(getattr(ctx, "env", None) or {})
    if env.get("SGLANG_NUMINOUS_EXTEND_GQA_SM90") != "1":
        return []
    probs = []
    dev = dict(getattr(ctx, "device", None) or {})
    if dev and int(dev.get("sm_major") or 0) != 9:
        probs.append("SGLANG_NUMINOUS_EXTEND_GQA_SM90=1 on a non-SM90 device: inert")
    args = dict(getattr(ctx, "args", None) or {})
    backends = {args.get("attention_backend"), args.get("prefill_attention_backend")} - {None}
    if backends and "triton" not in backends:
        probs.append(f"SGLANG_NUMINOUS_EXTEND_GQA_SM90=1 but the prefill attention backend is {sorted(backends)}: "
                     "the port patches the triton extend kernel only (inert)")
    return probs


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
