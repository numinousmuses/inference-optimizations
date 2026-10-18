"""CPU compile check of rtxsys's Gemma Triton kernels for SM90 (no GPU): Triton AOT to PTX, then ptxas -v for
registers / spills. torch 2.13.0 (the v0.5.20 pin) pairs with triton 3.7.x, the version this runs with.

    python3 operations/kernels/gemma31/rtx_aot_check.py [--json research/gpu-evidence/rtxsys/gemma31_rtx_kernels_ptxas_cpu.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

import rtx_draft_decode as DD  # noqa: E402
import rtx_qkv_norm_rope as QR  # noqa: E402
import rtx_draft_top1 as T1  # noqa: E402
import rtx_glue as G  # noqa: E402
import rtx_kv_store as KS  # noqa: E402

PTXAS = os.environ.get("PTXAS", os.path.expanduser("~/.cache/numinous/cudatk/nvidia/cuda_nvcc/bin/ptxas"))
GLUE = {"H": "*bf16", "R": "*bf16", "W1": "*bf16", "W2": "*bf16", "SC": "*bf16", "OUT": "*bf16", "ROUT": "*bf16",
        "Q": "*fp8e4nv", "QS": "*fp32", "stride_h": "i32", "stride_r": "i32", "stride_o": "i32", "stride_ro": "i32",
        "stride_q": "i32", "qs_m": "i32", "qs_g": "i32", "N": "i32", "NG": "i32", "eps1": "fp32", "eps2": "fp32",
        "CROSS": "constexpr", "QUANT": "constexpr", "BLOCK": "constexpr", "NGB": "constexpr", "GSZ": "constexpr",
        "EMU": "constexpr"}
KV = {"K": "*bf16", "V": "*bf16", "KC": "*u8", "VC": "*u8", "LOC": "*i64", "KSP": "*fp32", "VSP": "*fp32",
      "ks": "fp32", "vs": "fp32", "skn": "i32", "skh": "i32", "svn": "i32", "svh": "i32", "ROW_K": "i32",
      "ROW_V": "i32", "DK": "i32", "DV": "i32", "SKIP": "i64", "SCALE_PTR": "constexpr", "BLOCK": "constexpr",
      "EMU": "constexpr"}
TOP1P = {"X": "*fp32", "PM": "*fp32", "PI": "*i32", "PS": "*fp32", "stride_x": "i32", "V": "i32", "NSPLIT": "i32",
         "BLOCK": "constexpr"}
TOP1C = {"PM": "*fp32", "PI": "*i32", "PS": "*fp32", "P": "*fp32", "I": "*i64", "NSPLIT": "i32",
         "BLOCK_S": "constexpr"}

HEAD = {"Hd": "*bf16", "W": "*bf16", "WS": "*fp32", "PM": "*fp32", "PI": "*i32", "PS": "*fp32", "BS": "i32",
        "V": "i32", "K": "i32", "stride_h": "i32", "stride_w": "i32", "NT": "i32", "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "FP8": "constexpr", "EMU": "constexpr"}

DD1 = {"Q": "*bf16", "Kb": "*fp8e4nv", "Vb": "*fp8e4nv", "KV_INDPTR": "*i32", "KV_INDICES": "*i64", "OUT": "*fp32",
       "LSE": "*fp32", "qk_scale": "fp32", "stride_qb": "i32", "stride_qh": "i32", "stride_kt": "i32",
       "stride_kh": "i32", "stride_vt": "i32", "stride_vh": "i32", "stride_ob": "i32", "stride_oh": "i32",
       "stride_os": "i32", "stride_lb": "i32", "stride_lh": "i32", "NSPLIT": "i32", "GROUP": "constexpr",
       "D": "constexpr", "BLOCK_H": "constexpr", "BLOCK_N": "constexpr", "EMU": "constexpr"}
DD2 = {"OUT": "*fp32", "LSE": "*fp32", "O": "*bf16", "v_scale": "fp32", "stride_ob": "i32", "stride_oh": "i32",
       "stride_os": "i32", "stride_lb": "i32", "stride_lh": "i32", "stride_fb": "i32", "stride_fh": "i32",
       "NSPLIT": "i32", "D": "constexpr", "BLOCK_S": "constexpr"}

QNR = {"Q": "*bf16", "K": "*bf16", "V": "*bf16", "QW": "*bf16", "KW": "*bf16", "CS": "*fp32", "POS": "*i64",
       "sq": "i32", "sk": "i32", "sv": "i32", "eps": "fp32", "HQ": "constexpr", "HKV": "constexpr", "D": "constexpr",
       "HAS_KV": "constexpr", "EMU": "constexpr"}

CASES = [
    ("D1 post-attn chain, quant, hidden 5376", G._glue_chain_kernel, GLUE,
     {"CROSS": False, "QUANT": True, "BLOCK": 8192, "NGB": 64, "GSZ": 128, "EMU": False}, 16),
    ("D1 cross-layer chain, quant, hidden 5376", G._glue_chain_kernel, GLUE,
     {"CROSS": True, "QUANT": True, "BLOCK": 8192, "NGB": 64, "GSZ": 128, "EMU": False}, 16),
    ("D1 post-attn chain, assistant hidden 1024", G._glue_chain_kernel, GLUE,
     {"CROSS": False, "QUANT": False, "BLOCK": 1024, "NGB": 8, "GSZ": 128, "EMU": False}, 4),
    ("D1 cross-layer chain, assistant hidden 1024", G._glue_chain_kernel, GLUE,
     {"CROSS": True, "QUANT": False, "BLOCK": 1024, "NGB": 8, "GSZ": 128, "EMU": False}, 4),
    ("D1c KV store, scalar scale", KS._kv_store_kernel, KV, {"SCALE_PTR": False, "BLOCK": 1024, "EMU": False}, 4),
    ("D1c KV store, tensor scale", KS._kv_store_kernel, KV, {"SCALE_PTR": True, "BLOCK": 1024, "EMU": False}, 4),
    ("D2-b top-1 partial, fp32 logits", T1._top1_partial_kernel, TOP1P, {"BLOCK": 4096}, 8),
    ("D2-b top-1 partial, bf16 logits", T1._top1_partial_kernel, dict(TOP1P, X="*bf16"), {"BLOCK": 4096}, 8),
    ("D2-b top-1 combine", T1._top1_combine_kernel, TOP1C, {"BLOCK_S": 64}, 4),
    ("D2-b top-1 combine, 2048 head tiles", T1._top1_combine_kernel, TOP1C, {"BLOCK_S": 2048}, 8),
    ("D2-c fused head, bf16 weights, bs<=64", T1._head_top1_partial_kernel, HEAD,
     {"BLOCK_M": 64, "BLOCK_N": T1.HEAD_BN, "BLOCK_K": T1.HEAD_BK, "FP8": False, "EMU": False}, 8),
    ("D2-c fused head, fp8 weights, bs<=64", T1._head_top1_partial_kernel, dict(HEAD, W="*fp8e4nv"),
     {"BLOCK_M": 64, "BLOCK_N": T1.HEAD_BN, "BLOCK_K": T1.HEAD_BK, "FP8": True, "EMU": False}, 8),
    ("draft decode stage1, SWA hd256 group 2, fp8 KV", DD._dd_stage1, DD1,
     {"GROUP": 2, "D": 256, "BLOCK_H": 16, "BLOCK_N": 32, "EMU": False}, 4),
    ("draft decode stage1, full hd512 group 8, fp8 KV", DD._dd_stage1, DD1,
     {"GROUP": 8, "D": 512, "BLOCK_H": 16, "BLOCK_N": 32, "EMU": False}, 8),
    ("draft decode stage1, SWA hd256 group 2, bf16 KV", DD._dd_stage1, dict(DD1, Kb="*bf16", Vb="*bf16"),
     {"GROUP": 2, "D": 256, "BLOCK_H": 16, "BLOCK_N": 32, "EMU": False}, 8),
    ("draft decode stage2, hd256", DD._dd_stage2, DD2, {"D": 256, "BLOCK_S": 16}, 4),
    ("draft decode stage2, hd512", DD._dd_stage2, DD2, {"D": 512, "BLOCK_S": 16}, 4),
    ("D1d qkv norm+rope, SWA hd256 32/16", QR._qkv_norm_rope_kernel, QNR,
     {"HQ": 32, "HKV": 16, "D": 256, "HAS_KV": True, "EMU": False}, 4),
    ("D1d qkv norm+rope, full hd512 32/4", QR._qkv_norm_rope_kernel, QNR,
     {"HQ": 32, "HKV": 4, "D": 512, "HAS_KV": True, "EMU": False}, 8),
    ("D1d qkv norm+rope, kv-shared draft hd256", QR._qkv_norm_rope_kernel, QNR,
     {"HQ": 32, "HKV": 0, "D": 256, "HAS_KV": False, "EMU": False}, 4),
]


def run() -> list:
    tgt = GPUTarget("cuda", 90, 32)
    out = []
    for name, fn, sig, consts, nw in CASES:
        row = {"kernel": name, "num_warps": nw, "constexprs": consts}
        try:
            opts = {"num_warps": nw}
            if fn is T1._head_top1_partial_kernel:
                opts["num_stages"] = 4
            if fn is DD._dd_stage1:
                opts["num_stages"] = 4 if (consts["D"] == 256 and sig["Kb"] == "*bf16") else 3
            k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=tgt, options=opts)
            with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
                f.write(k.asm["ptx"])
            r = subprocess.run([PTXAS, "-arch=sm_90a", "-v", f.name, "-o", os.devnull], capture_output=True, text=True)
            txt = r.stderr + r.stdout
            m = re.search(r"Used (\d+) registers", txt)
            s = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", txt)
            row.update({"ok": r.returncode == 0, "regs": int(m.group(1)) if m else None,
                        "spill_stores": int(s.group(1)) if s else None, "spill_loads": int(s.group(2)) if s else None,
                        "shared": k.metadata.shared})
        except Exception as e:  # noqa: BLE001
            row.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"})
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    rows = run()
    for r in rows:
        print(json.dumps({k: r.get(k) for k in ("kernel", "ok", "regs", "spill_stores", "shared", "error") if k in r}))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"triton": triton.__version__, "target": "sm_90a", "rows": rows}, f, indent=1)
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
