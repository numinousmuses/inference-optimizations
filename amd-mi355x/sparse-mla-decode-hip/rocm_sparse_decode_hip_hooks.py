"""Anchored hook: route GLM-5.3 DSA DECODE / EAGLE-verify attention (64 heads, kv_lora 512 + rope 64, FP8 KV) on
gfx950 to rtxsys's hand-written HIP kernel (operations/kernels/rocm_sparse_mla_decode_hip.py), CUDA graphs included.
Overlay only; tree SGLang 1330e437e1 (20261002 ROCm image). Agent rtxsys, 2026-10-04 (coordinator 04:58 funding:
"rtxsys sparse-MLA gather").

Measured vs the shipped Triton path (split-K below 128 rows, one-pass from 128), mi355x8b GPU 4, CUDA graphs:
rows 12/24/48/72/96/128/192/288 = 0.91/0.91/0.83/0.76/0.74/0.55/0.84/0.68 x the time, same error vs an fp32
reference (research/gpu-evidence/rtxsys/amd/smla_hip_mi355x8b.json).

Inert unless NUMINOUS_SPARSE_DECODE_HIP=1. Routed calls (all must hold): HIP, kv float8_e4m3fn [slots, 1, 576],
tail_dim 64, d_v 512, 64 heads (DP attention; TP-sharded heads stay on the existing path), top-k a multiple of 256
and <= 2048, and fewer than NUMINOUS_SPARSE_DECODE_HIP_MAX_ROWS (384) rows OR inside graph capture.
Prefill-sized eager calls stay with amdglm's tl_sparse_prefill hook. Both anchor on the same line, and this block
runs first. The kernel JIT-compiles once per source hash (torch cpp_extension, file-locked) into
NUMINOUS_HIP_BUILD_DIR (default ~/.cache/numinous_hip). Split per row count: split_for_rows (override
NUMINOUS_SPARSE_DECODE_HIP_SPLIT_TABLE="rows:split,...").

    python3 rocm_sparse_decode_hip_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TL = "kernels/ops/attention/dsa/tilelang_kernel.py"
MODULE_SRC = os.path.join(HERE, "..", "kernels", "rocm_sparse_mla_decode_hip.py")
MODULE_DST = "kernels/ops/attention/dsa/numinous_sparse_mla_decode_hip.py"
NAME = "sparse_decode_hip"
# sha256[:16] of rocm_sparse_mla_decode_hip.py (the overlay is keyed on this patcher's bytes; test asserts the match)
KERNEL_SHA256 = "d4f7ec1104ee6583"
ANCHOR = "        is_fp8_kv = kv.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)\n        if is_fp8_kv:\n"
TEXT = ANCHOR + (
    "            # numinous-hook:sparse_decode_hip (rtxsys)\n"
    "            import os as _nm_sdh_os\n"
    "            if (_nm_sdh_os.environ.get(\"NUMINOUS_SPARSE_DECODE_HIP\") == \"1\" and tail_dim == 64 and d_v == 512\n"
    "                    and num_heads == 64 and kv.dtype == torch.float8_e4m3fn and kv.shape[1] == 1\n"
    "                    and topk % 256 == 0 and topk <= 2048\n"
    "                    and (torch.cuda.is_current_stream_capturing() or q.shape[0] < int(\n"
    "                        _nm_sdh_os.environ.get(\"NUMINOUS_SPARSE_DECODE_HIP_MAX_ROWS\", \"384\")))):\n"
    "                from sglang.kernels.ops.attention.dsa.numinous_sparse_mla_decode_hip import (\n"
    "                    sparse_mla_decode_hip as _nm_sdh)\n"
    "                return _nm_sdh(q, kv, indices, sm_scale, d_v)\n")


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
        raise SystemExit(f"{NAME}: anchor found {n} times in {TL}")
    return src.replace(ANCHOR, TEXT, 1), f"{NAME}: applied to {TL}"


def apply(root: str, check: bool = False) -> list[str]:
    path = os.path.join(root, TL)
    src = open(path).read()
    new, msg = patch_source(src)
    out = []
    if check:
        out.append(msg.replace("applied to", "anchor ok in") if new != src else msg)
    else:
        if new != src:
            compile(new, path, "exec")
            open(path, "w").write(new)
        out.append(msg)
    dst = os.path.join(root, MODULE_DST)
    same = os.path.exists(dst) and open(dst).read() == open(MODULE_SRC).read()
    if same:
        out.append(f"module: up to date {MODULE_DST}")
    elif check:
        out.append(f"module: MISSING or stale {MODULE_DST}")
    else:
        shutil.copy(MODULE_SRC, dst)
        out.append(f"installed {MODULE_DST}")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    lines = apply(a.root or sglang_root(), a.check)
    print("\n".join(lines))
    return 0 if not any("MISSING" in ln for ln in lines) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
