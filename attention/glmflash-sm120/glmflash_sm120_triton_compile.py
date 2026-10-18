"""CPU compile of glmflash_sm120_attn's Triton kernels for sm_120a (no GPU): shared memory, registers, spills per
config, using Triton's own ptxas-blackwell with the flags Triton passes (-v --regAllocOptLevel=2).

    python3 operations/kernels/glmflash_sm120_triton_compile.py [--out research/gpu-evidence/rtxattn_triton_sm120_compile.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import triton  # noqa: E402
from triton import knobs  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

import glmflash_sm120_attn as A  # noqa: E402

SIG = {"q_ptr": "*bf16", "stride_qt": "i32", "stride_qh": "i32", "kv_ptr": "*u8", "kv_f32_ptr": "*fp32",
       "n_rows": "i32", "idx_ptr": "*i32", "stride_it": "i32", "K": "i32", "out_ptr": "*bf16", "stride_ot": "i32",
       "stride_oh": "i32", "lse_ptr": "*fp32", "part_o_ptr": "*fp32", "part_ml_ptr": "*fp32", "sm_scale_log2": "fp32",
       "H": "constexpr", "FMT": "constexpr", "BLOCK_H": "constexpr", "BLOCK_N": "constexpr", "SPLITS": "constexpr",
       "TILES_PER_SPLIT": "i32", "DOT_F32": "constexpr", "NUM_STAGES": "constexpr"}
CONFIGS = ((16, 4, 2), (16, 8, 2), (32, 4, 2), (32, 8, 2), (32, 8, 3), (64, 4, 1), (64, 8, 1), (64, 8, 2))


def compile_one(fmt, bn, nw, ns, splits=4, heads=16):
    target = GPUTarget("cuda", 120, 32)
    s = dict(SIG)
    const = {"H": heads, "FMT": fmt, "BLOCK_H": 16, "BLOCK_N": bn, "SPLITS": splits, "DOT_F32": False,
             "NUM_STAGES": ns}
    if fmt == A.FMT_BF16:
        s["kv_ptr"] = s["kv_f32_ptr"] = "*bf16"
    if fmt == A.FMT_FP4_288:
        s = {k: v for k, v in s.items() if k not in ("kv_f32_ptr", "FMT")}
        const.pop("FMT")
        fn = A._rtx_sparse_mla_fp4_kernel
    else:
        fn = A._rtx_sparse_mla_kernel
    k = triton.compile(ASTSource(fn=fn, signature=s, constexprs=const), target=target,
                       options={"num_warps": nw, "num_stages": ns})
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(k.asm["ptx"])
    r = subprocess.run([knobs.nvidia.ptxas_blackwell.path, "-v", "--regAllocOptLevel=2", "--gpu-name=sm_120a",
                        f.name, "-o", "/dev/null"], capture_output=True, text=True)
    m = re.search(r"Used (\d+) registers", r.stderr)
    sp = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", r.stderr)
    return {"fmt": A.FMT_NAMES[fmt], "block_n": bn, "warps": nw, "stages": ns, "splits": splits,
            "smem_bytes": k.metadata.shared, "fits_99KB": k.metadata.shared <= 101376,
            "regs": int(m.group(1)) if m else None,
            "spill_store_load": [int(sp.group(1)), int(sp.group(2))] if sp else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for fmt in (A.FMT_BF16, A.FMT_FP8_656, A.FMT_FP4_288):
        for bn, nw, ns in CONFIGS:
            for splits in (1, 4):
                r = compile_one(fmt, bn, nw, ns, splits)
                print(json.dumps(r), flush=True)
                rows.append(r)
    doc = {"triton": triton.__version__, "ptxas": knobs.nvidia.ptxas_blackwell.version, "target": "sm_120a",
           "rows": rows}
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
