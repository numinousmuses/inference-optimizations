"""CPU-side compile + resource check for rtx_w4_decode (no GPU; operator 2026-10-05: GPU runs are scarce).

NVRTC compiles rtx_w4_decode_kernels.cuh with every kernel instantiation the extension launches to sm_90a PTX; ptxas
assembles it with -v. Reported per kernel: registers, spill stores / loads, shared memory, stack. A spill or a
compile error shows up here instead of on the GPU box.

    python3 compile_check.py [--json out.json]

Toolkit: CUDA 12.8 wheels under ~/.cache/numinous/cudatk (nvidia-cuda-nvrtc-cu12, nvidia-cuda-nvcc-cu12 for ptxas,
nvidia-cuda-runtime-cu12 + nvidia-cuda-cccl-cu12 for headers), or $RTXW4_CUDATK.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
TK = pathlib.Path(os.environ.get("RTXW4_CUDATK", pathlib.Path.home() / ".cache/numinous/cudatk"))
HEADER = HERE / "rtx_w4_decode_kernels.cuh"

# (kernel, template args) the host launchers instantiate (rtx_w4_decode.cu)
INSTANCES = (
    [f"rtxw4::w4_kernel<{f}, {s}, {t}, {m}>" for f in (0, 1) for s in ("true", "false") for t, m in ((2, 4), (4, 2))]
    + [f"rtxw4::w4b_kernel<{f}, {s}, {t}, {m}, 4>" for f in (0, 1) for s in ("true", "false")
       for t, m in ((2, 4), (4, 2))]
    + [f"rtxw4::w4f8_kernel<{s}, 2, 2>" for s in ("true", "false")]
)
PARAM_TYPE = {"w4_kernel": "rtxw4::Params", "w4b_kernel": "rtxw4::Params", "w4f8_kernel": "rtxw4::ParamsF8"}


def _find(pattern: str) -> pathlib.Path:
    hits = sorted(TK.rglob(pattern))
    if not hits:
        raise SystemExit(f"{pattern} not found under {TK}")
    return hits[0]


def nvrtc_ptx(src: str, opts: list[str]) -> str:
    lib = ctypes.CDLL(str(_find("libnvrtc.so.12")))
    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"rtxw4_check.cu", 0, None, None)
    assert rc == 0, f"nvrtcCreateProgram {rc}"
    arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        raise RuntimeError(f"NVRTC compile failed ({rc}):\n{log.value.decode(errors='replace')[-6000:]}")
    lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
    ptx = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetPTX(prog, ptx)
    return ptx.value.decode()


def check() -> dict:
    lines = [f'#include "{HEADER.name}"']
    for inst in INSTANCES:
        name = inst.split("<")[0].split("::")[-1]
        lines.append(f"template __global__ void {inst}({PARAM_TYPE[name]});")
    src = "\n".join(lines) + "\n"
    incs = sorted({str(p.parent) for p in TK.rglob("cuda_bf16.h")} | {str(p.parent.parent) for p in TK.rglob("cstdint")
                                                                       if p.parent.name == "std"})
    opts = ["--gpu-architecture=sm_90a", "-std=c++17", "-default-device", "--use_fast_math=false",
            f"-I{HERE}"] + [f"-I{i}" for i in incs]
    opts = [o for o in opts if o != "--use_fast_math=false"]
    ptx = nvrtc_ptx(src, opts)
    with tempfile.TemporaryDirectory() as tmp:
        pf = pathlib.Path(tmp) / "k.ptx"
        pf.write_text(ptx)
        r = subprocess.run([str(_find("ptxas")), "-arch=sm_90a", "-v", str(pf), "-o", str(pathlib.Path(tmp) / "k.cubin")],
                           capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ptxas failed:\n{r.stderr[-6000:]}")
    out, cur = {}, None
    for ln in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", ln)
        if m:
            cur = m.group(1)
            out[cur] = {}
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", ln)
        if m:
            out[cur].update(stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", ln)
        if m:
            out[cur]["regs"] = int(m.group(1))
        m = re.search(r"(\d+) bytes smem", ln)
        if m:
            out[cur]["smem_static"] = int(m.group(1))
    demangled = {}
    for k, v in out.items():
        short = k
        for inst in INSTANCES:
            nm = inst.split("<")[0].split("::")[-1]
            if nm in k:
                short = k
        demangled[short] = v
    return {"ptx_bytes": len(ptx), "kernels": demangled}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--only", default="", help="substring filter on the instantiations (e.g. w4f8)")
    a = ap.parse_args()
    global INSTANCES
    if a.only:
        INSTANCES = [i for i in INSTANCES if a.only in i]
    res = check()
    txt = json.dumps(res, indent=1)
    if a.json:
        pathlib.Path(a.json).write_text(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
