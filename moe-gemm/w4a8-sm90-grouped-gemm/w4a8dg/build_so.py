"""Build k3amd_w4a8dg.so (operations/kernels/w4a8dg) with nvcc for sm_90a on the machine it runs on: the production
v0.5.20 image (nvcc 13.0, torch 2.13 cu130), as a Modal CPU container (operations/modal/w4a8dg_build.py) or the
Daytona h200 box (operations/kernels/w4a8dg/box_bench.py build).

CUTLASS: sglang-kernel's pin 57e3cfb4 (cute + cutlass arch headers), downloaded once into a cache dir.

    python3 build_so.py <out_root>          # -> <out_root>/w4a8dg_<stamp>/k3amd_w4a8dg.so + build.json
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
CUTLASS = "https://github.com/NVIDIA/cutlass/archive/57e3cfb47a2d9e0d46eb6335c3dc411498efa198.tar.gz"
FLAGS = ["-std=c++17", "-O3", "-Xcompiler", "-fPIC", "-gencode=arch=compute_90a,code=sm_90a", "-DNDEBUG",
         "-DCUTE_USE_PACKED_TUPLE=1", "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "--expt-relaxed-constexpr",
         "--expt-extended-lambda", "-Xcompiler=-Wno-deprecated-declarations", "-Xcudafe=--diag_suppress=177",
         "-lineinfo", "-Xptxas=-v"]


def bundled_cutlass(roots=None) -> pathlib.Path | None:
    """A CUTLASS tree with the SM90 cute headers shipped inside the image (flashinfer / sgl_kernel / deep_gemm data
    dirs), for a box without internet: the dir whose include/cute/arch/mma_sm90_gmma.hpp exists."""
    import site
    roots = roots or [pathlib.Path(p) for p in site.getsitepackages() + ["/opt/sglang/lib/python3.12/site-packages",
                                                                          "/usr/local/lib/python3.12/dist-packages"]]
    for root in roots:
        if not root.is_dir():
            continue
        for hit in sorted(root.glob("**/include/cute/arch/mma_sm90_gmma.hpp")):
            base = hit.parents[3]
            if (base / "include/cutlass/arch/barrier.h").exists():
                return base
    return None


def cutlass_dir(cache: pathlib.Path) -> pathlib.Path:
    """The pinned CUTLASS checkout under cache (downloaded on first use; a bundled copy if the download fails)."""
    found = sorted(cache.glob("cutlass-57e3cfb4*"))
    if found:
        return found[0]
    cache.mkdir(parents=True, exist_ok=True)
    tgz = cache / "cutlass.tgz"
    try:
        urllib.request.urlretrieve(CUTLASS, tgz)
        subprocess.run(["tar", "xzf", str(tgz), "-C", str(cache)], check=True)
        return sorted(cache.glob("cutlass-57e3cfb4*"))[0]
    except Exception as ex:  # noqa: BLE001
        b = bundled_cutlass()
        if b is None:
            raise RuntimeError(f"no CUTLASS: download failed ({ex}) and no bundled tree with SM90 cute headers") from ex
        print(f"CUTLASS download failed ({type(ex).__name__}); using the bundled tree {b}", flush=True)
        return b


def ptxas_summary(text: str) -> list:
    """One summary per kernel: entry name, registers, spills, and ptxas performance warnings (C75xx)."""
    return [ln for ln in text.splitlines()
            if "Compiling entry" in ln or "registers" in ln or "spill" in ln or "C75" in ln]


def build(out_root: str, src: pathlib.Path = HERE, cache: str = "/root/k3amd_cache") -> dict:
    import torch
    from torch.utils import cpp_extension
    t0 = time.time()
    work = pathlib.Path(cache) / f"build_{os.getpid()}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cut = cutlass_dir(pathlib.Path(cache))
    inc = [f"-I{cut}/include", f"-I{cut}/tools/util/include", f"-I{src}"]
    inc += [f"-I{p}" for p in cpp_extension.include_paths(device_type="cuda")]
    inc += [f"-I{sysconfig.get_paths()['include']}"]
    defs = [f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", "-DTORCH_API_INCLUDE_EXTENSION_H",
            "-DTORCH_EXTENSION_NAME=k3amd_w4a8dg"]
    out = {"cutlass": CUTLASS, "src_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                                               for f in sorted(src.glob("*.cu*"))}}
    p = subprocess.run(["nvcc", "-c", str(src / "w4a8dg.cu"), "-o", str(work / "w4a8dg.o")] + FLAGS + inc + defs,
                       capture_output=True, text=True)
    out["compile_rc"] = p.returncode
    out["ptxas"] = ptxas_summary(p.stderr + p.stdout)
    if p.returncode:
        out["compile_error"] = (p.stderr or p.stdout)[-6000:]
        out["seconds"] = round(time.time() - t0, 1)
        return out
    lib = [f"-L{d}" for d in cpp_extension.library_paths(device_type="cuda")]
    p = subprocess.run(["g++", "-shared", "-o", str(work / "k3amd_w4a8dg.so"), str(work / "w4a8dg.o")] + lib
                       + ["-lc10", "-lc10_cuda", "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lcudart"],
                       capture_output=True, text=True)
    if p.returncode:
        out["link_error"] = p.stderr[-3000:]
        return out
    dst = pathlib.Path(out_root) / f"w4a8dg_{time.strftime('%Y%m%d-%H%M%S')}"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(work / "k3amd_w4a8dg.so", dst / "k3amd_w4a8dg.so")
    shutil.rmtree(work, ignore_errors=True)
    out["so"] = str(dst / "k3amd_w4a8dg.so")
    out["sha256"] = hashlib.sha256((dst / "k3amd_w4a8dg.so").read_bytes()).hexdigest()
    out["seconds"] = round(time.time() - t0, 1)
    (dst / "build.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = build(sys.argv[1] if len(sys.argv) > 1 else "/root/k3amd_ext")
    print(json.dumps({k: v for k, v in res.items() if k != "compile_error"}, indent=1))
    if res.get("compile_error"):
        print(res["compile_error"])
    sys.exit(0 if res.get("so") else 1)
