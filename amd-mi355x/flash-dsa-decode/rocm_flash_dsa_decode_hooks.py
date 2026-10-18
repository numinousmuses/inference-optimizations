"""Anchored hook: route GLM-5.3-Flash's rope-free DSA DECODE / EAGLE-verify attention on HIP + FP8 KV to rtxsys's
split-K Triton kernel (operations/kernels/rocm_flash_dsa_decode.py), CUDA graphs included. Overlay only; tree SGLang
1330e437e1. Agent rtxsys, 2026-10-04 (coordinator assignment 01:50: Flash decode per-request cost).

Inert unless NUMINOUS_FLASH_DSA_DECODE=1. Routed calls (all must hold): HIP, kv float8_e4m3fn [slots, 1, 512],
tail_dim 0 (rope-free), d_v 512, top-k a multiple of SPLIT x 64, and fewer than NUMINOUS_FLASH_SPARSE_PREFILL_MIN_ROWS
(256) rows OR inside graph capture. Prefill-sized eager calls stay with amdflash's flash_sparse_prefill hook, which
this hook leaves untouched: both anchor on the same line, this block runs first.
Split per row count: rocm_flash_dsa_decode.split_for_rows (table measured by rocm_flash_dsa_decode_bench.py on gfx950,
overridable with NUMINOUS_FLASH_DSA_DECODE_SPLIT_TABLE="rows:split,..."); head split NUMINOUS_FLASH_DSA_DECODE_HS (1).

    python3 rocm_flash_dsa_decode_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TL = "kernels/ops/attention/dsa/tilelang_kernel.py"
MODULE_SRC = os.path.join(HERE, "..", "kernels", "rocm_flash_dsa_decode.py")
MODULE_DST = "kernels/ops/attention/dsa/numinous_flash_dsa_decode.py"
NAME = "flash_dsa_decode"
# sha256[:16] of rocm_flash_dsa_decode.py: the overlay is keyed on this patcher's bytes, so a kernel edit must change
# this line too (test_rocm_flash_dsa_decode_hooks asserts the match)
KERNEL_SHA256 = "4bca83bb2b7ff295"
ANCHOR = "        is_fp8_kv = kv.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)\n        if is_fp8_kv:\n"
TEXT = ANCHOR + (
    "            # numinous-hook:flash_dsa_decode (rtxsys)\n"
    "            import os as _nm_fdd_os\n"
    "            if (_nm_fdd_os.environ.get(\"NUMINOUS_FLASH_DSA_DECODE\") == \"1\" and tail_dim == 0 and d_v == 512\n"
    "                    and kv.dtype == torch.float8_e4m3fn and kv.shape[1] == 1\n"
    "                    and indices.shape[-1] % 256 == 0\n"
    "                    and (torch.cuda.is_current_stream_capturing() or q.shape[0] < int(\n"
    "                        _nm_fdd_os.environ.get(\"NUMINOUS_FLASH_SPARSE_PREFILL_MIN_ROWS\", \"256\")))):\n"
    "                from sglang.kernels.ops.attention.dsa.numinous_flash_dsa_decode import (\n"
    "                    flash_sparse_decode_fp8 as _nm_fdd, split_for_rows as _nm_fdd_split)\n"
    "                return _nm_fdd(q, kv, indices, sm_scale, d_v, split=_nm_fdd_split(q.shape[0]),\n"
    "                               head_split=int(_nm_fdd_os.environ.get(\"NUMINOUS_FLASH_DSA_DECODE_HS\", \"1\")))\n")


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
