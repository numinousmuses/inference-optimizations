"""CPU compile check of the widened FlashInfer SM120 module: nvcc -gencode sm_120a on every patched source, no GPU.

A CUDA 13.0 toolkit assembled from NVIDIA's pip wheels is enough (nvidia-cuda-nvcc, -crt, -runtime, -cccl, nvidia-nvvm
13.0.x unpacked into one root); FlashInfer headers + csrc from the 0.6.18 wheel; tvm-ffi headers from apache-tvm-ffi.

    python3 operations/kernels/glmflash_sm120_fi_compile.py --cuda /tmp/nvccw/x/nvidia/cu13 \
        --fi /tmp/kv4fi_fi_pkg/flashinfer --out /tmp/fi_sm120_build [--full] [--link]

Flags mirror flashinfer.jit (gen_jit_spec + cpp_ext.build_cuda_cflags) for major 12: -std=c++17 -O3 -DNDEBUG
-use_fast_math --expt-relaxed-constexpr -static-global-template-stub=false -gencode arch=compute_120a,code=sm_120a.
Writes <out>/compile_report.json (per file: seconds, ok, stderr tail, SASS kernel names with the 2112 instantiations).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glmflash_sm120_fi as FI  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", required=True)
    ap.add_argument("--fi", required=True, help="flashinfer package dir (has data/include, data/csrc, data/cccl)")
    ap.add_argument("--tvm-ffi", default=None)
    ap.add_argument("--out", default="/tmp/fi_sm120_build")
    ap.add_argument("--full", action="store_true", help="untrimmed build (every stock variant + 2112)")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--link", action="store_true", help="also link a shared object")
    a = ap.parse_args()
    cuda = pathlib.Path(a.cuda)
    fi = pathlib.Path(a.fi)
    tvm = a.tvm_ffi
    if tvm is None:
        import tvm_ffi
        tvm = str(pathlib.Path(tvm_ffi.__file__).parent / "include")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = FI.patched_sources(fi / "data" / "csrc", trim=not a.full)
    nvcc = str(cuda / "bin" / "nvcc")
    env = dict(os.environ, PATH=f"{cuda / 'bin'}:{os.environ['PATH']}", CUDA_HOME=str(cuda))
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-use_fast_math", "--expt-relaxed-constexpr",
             "-static-global-template-stub=false", "--compiler-options=-fPIC",
             "-gencode", "arch=compute_120a,code=sm_120a", "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16",
             "-DFLASHINFER_ENABLE_FP8_E4M3", "-DFLASHINFER_ENABLE_FP8_E5M2", "-DPy_LIMITED_API=0x03090000",
             f"-I{fi / 'data' / 'include'}", f"-I{fi / 'data' / 'csrc'}", f"-I{fi / 'data' / 'cccl' / 'libcudacxx' / 'include'}",
             f"-I{fi / 'data' / 'cccl' / 'cub'}", f"-I{fi / 'data' / 'cccl' / 'thrust'}", f"-I{tvm}",
             f"-I{fi / 'data' / 'spdlog' / 'include'}", "--threads=2"]

    def build(name: str) -> dict:
        p = out / name
        p.write_text(src[name])
        o = p.with_suffix(".o")
        t0 = time.time()
        r = subprocess.run([nvcc, *flags, "-c", str(p), "-o", str(o)], capture_output=True, text=True, env=env)
        rep = {"file": name, "seconds": round(time.time() - t0, 1), "ok": r.returncode == 0,
               "stderr_tail": r.stderr[-3000:]}
        if r.returncode == 0:
            sym = subprocess.run([str(cuda / "bin" / "cuobjdump"), "-symbols", str(o)], capture_output=True,
                                 text=True, env=env) if (cuda / "bin" / "cuobjdump").exists() else None
            names = subprocess.run(["nm", "-C", str(o)], capture_output=True, text=True).stdout
            rep["kernels_2112"] = sorted({ln.split(" ", 2)[-1][:160] for ln in names.splitlines()
                                          if "2112" in ln and "sparse_mla" in ln})[:40]
            rep["cuobjdump"] = bool(sym and sym.returncode == 0)
        print(json.dumps({k: v for k, v in rep.items() if k != "stderr_tail"}), flush=True)
        if not rep["ok"]:
            print(rep["stderr_tail"], flush=True)
        return rep

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        reps = list(ex.map(build, FI.SOURCES))
    ok = all(r["ok"] for r in reps)
    if ok and a.link:
        objs = [str((out / n).with_suffix(".o")) for n in FI.SOURCES]
        r = subprocess.run([nvcc, "-shared", "-gencode", "arch=compute_120a,code=sm_120a", f"-L{cuda / 'lib'}",
                            "-Xlinker", "--unresolved-symbols=ignore-all", *objs, "-o",
                            str(out / f"{FI.MODULE_NAME}.so")], capture_output=True, text=True, env=env)
        reps.append({"file": "link", "ok": r.returncode == 0, "stderr_tail": r.stderr[-2000:]})
        ok = ok and r.returncode == 0
    (out / "compile_report.json").write_text(json.dumps({"trim": not a.full, "ok": ok, "files": reps}, indent=1))
    print("COMPILE_OK" if ok else "COMPILE_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
