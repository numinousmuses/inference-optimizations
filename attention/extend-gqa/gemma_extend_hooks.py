"""Anchored patcher that installs the Numinous GQA-packed extend-attention kernel for Gemma 4 on SM120
(RTX PRO 6000) into SGLang v0.5.20. Style follows operations/patches/sglang_numinous_hooks.py.

Background: Gemma 4's global attention layers have head_dim=512 and only 2 KV heads (16 Q heads per KV head).
The stock v0.5.20 Triton extend kernel runs one program per query head, reading each KV tile 8x. On SM120 this
measured at 75 TFLOPS (20% of peak) at 64k sessions (research/gpu-evidence/dc_arch). The GQA-packed kernel
(operations/kernels/extend_attention_gqa.py) runs one program per (request, KV head, token-block), reading each
KV tile once for all 8 query heads (2.7x on the global layers, 8.7x on the window-limited sliding layers).

This patcher:
  1. Copies numinous_extend_gqa.py into sglang/kernels/ops/attention/numinous_extend_gqa.py (MODULES).
  2. Inserts a dispatch block into extend_attention_fwd after kv_group_num is resolved, before the AMD-specific
     FP8 zero-prefix specialization. A Python-level exception falls back to stock (never kills the server).
  3. Appends three module-level helpers (_NUMINOUS_GQA_FAILED, _numinous_gqa_enabled, _numinous_gqa_config)
     at the end of extend_attention.py via an empty anchor ("" = end of file per the hooks convention).

Usage:
    python3 operations/patches/gemma_extend_hooks.py                  # patch installed sglang
    python3 operations/patches/gemma_extend_hooks.py --check          # check anchors, change nothing
    python3 operations/patches/gemma_extend_hooks.py --root /tmp/sgl  # patch a copy

The test (operations/patches/test_gemma_extend_hooks.py) applies the patcher to a working-tree copy and
imports the result (CPU interpreter) to verify syntax and dispatch entry.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

# kernel source lives next to the extend kernel in operations/kernels/
_HERE = os.path.dirname(os.path.abspath(__file__))
_GQA_SRC = os.path.join(_HERE, "..", "kernels", "extend_attention_gqa.py")

# install path inside the sglang package (relative to sglang root)
MODULES = {_GQA_SRC: "kernels/ops/attention/numinous_extend_gqa.py"}

_DISPATCH = """\
    # numinous-hook:gqa_dispatch
    # Numinous GQA-packed extend-attention for SM120 (Gemma 4 global hd=512, 8x GQA ratio; sliding hd=256).
    # One program per (request, KV-head, token-block): each KV tile read once for all query heads in the group.
    # Falls back to the stock kernel on any Python-level failure; SGLANG_NUMINOUS_EXTEND_GQA=0 disables globally.
    if (
        _numinous_gqa_enabled()
        and q_extend.shape[-1] >= 256
        and q_extend.shape[-1] == v_extend.shape[-1]
        and is_causal
        and custom_mask is None
        and sinks is None
        and not logit_cap
        and score_mod is None
        and lse_extend is None
        and not skip_prefix
        and not skip_extend
        and page_size == 1
        and window_kv_offsets is None
        and (xai_temperature_len is None or xai_temperature_len <= 0)
        and k_buffer.ndim == 3
    ):
        try:
            from sglang.kernels.ops.attention.numinous_extend_gqa import (
                extend_attention_gqa as _numinous_extend_gqa,
            )

            _numinous_extend_gqa(
                q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
                qo_indptr, kv_indptr, kv_indices,
                int(max_len_extend),
                sm_scale if sm_scale is not None else 1.0 / (q_extend.shape[-1] ** 0.5),
                float(k_scale) if k_scale is not None else 1.0,
                float(v_scale) if v_scale is not None else 1.0,
                int(sliding_window_size) if sliding_window_size is not None else -1,
                **_numinous_gqa_config(q_extend.shape[-1], q_extend.shape[1] // k_extend.shape[1]),
            )
            return
        except Exception as _numinous_e:  # noqa: BLE001
            global _NUMINOUS_GQA_FAILED
            _NUMINOUS_GQA_FAILED = True
            import logging as _numinous_logging
            _numinous_logging.getLogger(__name__).exception(
                "numinous GQA extend kernel failed, falling back to stock: %s", _numinous_e
            )
"""

_HELPERS = """

# numinous-hook:gqa_helpers
# ---- numinous GQA extend helpers (gemma_extend_hooks.py) -------------------------------------------------------
_NUMINOUS_GQA_FAILED = False


def _numinous_gqa_enabled() -> bool:
    import os as _os
    if _NUMINOUS_GQA_FAILED or _os.environ.get("SGLANG_NUMINOUS_EXTEND_GQA", "1") == "0":
        return False
    return _is_cuda and CUDA_CAPABILITY[0] == 12


def _numinous_gqa_config(head_dim: int, group: int) -> dict:
    \"\"\"Tile parameters measured on RTX PRO 6000 (SM120).

    head_dim >= 512 (Gemma 4 global layers, 8 query heads per KV head):
        CUDA-graph timed: 2.72x stock at b=4 req, 2.79x at b=8 (measured, rtx-gemma).
    group >= 4 (Qwen 3.8 full attention, 6 query heads per KV head):
        1.18x FlashInfer, 1.9x stock Triton (measured, rtx-qwen38).
    other (Gemma 4 sliding, 2 query heads, window 1024):
        8.7x stock (stock reads outside the window; measured, rtx-gemma).
    SGLANG_NUMINOUS_GQA_CFG='k=v,...' overrides individual keys.
    \"\"\"
    import os as _os
    if head_dim >= 512:
        cfg = dict(block_t=8, block_n=128, dk=64, dv=256, num_warps=8, num_stages=3)
    elif group >= 4:
        cfg = dict(block_t=8, block_n=128, dk=64, dv=head_dim, num_warps=8, num_stages=3)
    else:
        cfg = dict(block_t=32, block_n=64, dk=64, dv=head_dim, num_warps=8, num_stages=1)
    for kv in filter(None, _os.environ.get("SGLANG_NUMINOUS_GQA_CFG", "").split(",")):
        k, v = kv.split("=")
        cfg[k.strip()] = int(v)
    return cfg
# ---- end numinous GQA extend helpers ---------------------------------------------------------------------------
"""

# numinous-hook:gqa_dispatch inserts into extend_attention_fwd right after kv_group_num is resolved
# (before the AMD-specific zero_prefix_shape block); anchor appears exactly once in v0.5.20.
# numinous-hook:gqa_helpers appends module-level state/helpers at end of file (anchor = "").
EXTEND_ATTN_PATH = "kernels/ops/attention/extend_attention.py"

HOOKS = [
    {
        "name": "gqa_dispatch",
        "path": EXTEND_ATTN_PATH,
        "anchor": "    kv_group_num = q_extend.shape[1] // k_extend.shape[1]\n",
        "text": _DISPATCH,
    },
    {
        "name": "gqa_helpers",
        "path": EXTEND_ATTN_PATH,
        "anchor": "",  # end of file
        "text": _HELPERS,
    },
]


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang not importable")
    return list(spec.submodule_search_locations)[0]


def install_modules(root: str, check: bool) -> list[str]:
    """Copy MODULES source files into their target locations inside the sglang package."""
    out = []
    for src, rel in MODULES.items():
        if not os.path.isabs(src):
            src = os.path.join(_HERE, src)
        dst = os.path.join(root, rel)
        if check:
            out.append(f"module {rel}: src exists={os.path.isfile(src)}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        content = open(src, "rb").read()
        with open(dst, "wb") as f:
            f.write(content)
        out.append(f"module {rel}: installed from {src}")
    return out


def apply(root: str, check: bool = False) -> list[str]:
    """Returns one status line per hook. Raises SystemExit if an anchor is missing or non-unique."""
    out = []
    missing = []
    for h in HOOKS:
        marker = f"# numinous-hook:{h['name']}"
        path = os.path.join(root, h["path"])
        src = open(path).read()
        if marker in src:
            out.append(f"{h['name']}: already applied")
            continue
        anchor = h["anchor"]
        if anchor:
            n = src.count(anchor)
            if n != 1:
                missing.append(f"{h['name']}: anchor found {n} times in {h['path']}")
                continue
        if check:
            out.append(f"{h['name']}: anchor ok")
            continue
        text_with_marker = h["text"].replace(
            f"    # numinous-hook:{h['name']}",
            f"    # numinous-hook:{h['name']}"
        )
        if anchor:
            new = src.replace(anchor, anchor + h["text"], 1)
        else:
            new = src.rstrip("\n") + "\n" + h["text"]
        compile(new, path, "exec")
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"{h['name']}: applied to {h['path']}")
    if missing:
        raise SystemExit("gemma_extend_hooks: " + "; ".join(missing))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="", help="sglang package root (default: installed)")
    ap.add_argument("--check", action="store_true", help="check anchors only, change nothing")
    a = ap.parse_args(argv)
    root = a.root or sglang_root()
    for line in install_modules(root, a.check):
        print(line)
    for line in apply(root, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
