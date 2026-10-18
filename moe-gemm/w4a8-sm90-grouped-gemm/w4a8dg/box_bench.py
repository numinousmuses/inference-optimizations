"""k3amd GLM prefill MoE batch on the Daytona h200 box (coordinator 19:5x: Modal GPUs are out; devbox.py sync / run;
one booked window, nothing else on the box during it).

    python3 operations/devbox/devbox.py sync h200
    python3 operations/devbox/devbox.py push h200 /tmp/k3a_box/k3amd_ext /root/k3amd_ext   # production tiles .so
    python3 operations/devbox/devbox.py run h200 -- python3 operations/kernels/w4a8dg/box_bench.py all

Steps (idempotent; everything under /root/k3amd*; the installed sglang package is never modified, other agents share
the box):
  build   nvcc k3amd_w4a8dg.so for sm_90a (build_so.py) -> /root/k3amd_ext/w4a8dg_<stamp>/
  prep    private copy of the installed sglang at /root/k3amd_sgl/sglang with the bench's hooks applied
          (moe_src2dst_fused_hooks and glm_w4a8_tiles_hooks incl. the w4a8dg dispatch and the fused-silu GEMM1)
  gemm    per-GEMM microbench (glm_moe_decode_bench --prefill-mm): stock, served tiles, w4a8dg v3 / v4 configs and
          the v4 ablations, EP8 + EP4 shapes, m 2K-16K, rel vs fp32 and vs stock
  call    the whole cutlass_w4a8_moe call (--moe-call): stock, tiles, tiles + w4a8dg, + fused silu, same accuracy
  all     build, prep, gemm, call
Each bench step prints its JSON between K3A_JSON_BEGIN / K3A_JSON_END and keeps it in /root/k3amd_out/.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXT = pathlib.Path(os.environ.get("K3A_EXT", "/root/k3amd_ext"))
SGL = pathlib.Path(os.environ.get("K3A_SGL", "/root/k3amd_sgl"))
OUT = pathlib.Path(os.environ.get("K3A_OUT", "/root/k3amd_out"))
TILES_SO = EXT / "20261005-065859/k3amd_w4a8.so"
HOOKS = ["moe_src2dst_fused_hooks.py", "glm_w4a8_tiles_hooks.py"]
MS = os.environ.get("K3A_MS", "2048,4096,8192,16384")
GEMM_CFGS = os.environ.get("K3A_GEMM_CFGS", "20,40,41,42,43,44")
CALL_DG = os.environ.get("K3A_CALL_DG", "40,41")
CALL_SILU = os.environ.get("K3A_CALL_SILU", "50,51")


def sh(cmd: list, env: dict | None = None, timeout: int = 3600) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=timeout)


def installed_sglang() -> pathlib.Path:
    src = os.environ.get("K3A_SGLANG_SRC")
    if src:
        return pathlib.Path(src)
    p = subprocess.run([sys.executable, "-c", "import sglang, os; print(os.path.dirname(sglang.__file__))"],
                       capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    return pathlib.Path(p.stdout.strip())


def newest_so() -> pathlib.Path | None:
    found = sorted(glob.glob(str(EXT / "w4a8dg_*/k3amd_w4a8dg.so")))
    return pathlib.Path(found[-1]) if found else None


def step_build() -> dict:
    sys.path.insert(0, str(HERE))
    import build_so
    res = build_so.build(str(EXT), cache=str(EXT.parent / "k3amd_cache"))
    print(json.dumps({k: v for k, v in res.items() if k != "compile_error"}, indent=1), flush=True)
    if not res.get("so"):
        raise SystemExit(res.get("compile_error") or res.get("link_error") or "build failed")
    return res


def step_prep() -> dict:
    """/root/k3amd_sgl/sglang = the installed package + hooks. Rebuilt when the hook files changed."""
    sys.path.insert(0, str(REPO / "operations/patches"))
    stamp = hashlib.sha256(b"".join((REPO / "operations/patches" / h).read_bytes() for h in HOOKS)).hexdigest()[:16]
    dst = SGL / "sglang"
    mark = SGL / f".hooks_{stamp}"
    out = {"sglang_copy": str(dst), "hooks_stamp": stamp}
    if not mark.exists():
        shutil.rmtree(SGL, ignore_errors=True)
        src = installed_sglang()
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
        for h in HOOKS:
            r = sh([sys.executable, str(REPO / "operations/patches" / h), "--root", str(dst)])
            out[h] = (r.stdout + r.stderr)[-1500:]
            if r.returncode:
                raise SystemExit(f"{h} failed on the private sglang copy:\n{r.stdout}{r.stderr}")
        mark.write_text(json.dumps(out))
    else:
        out["reused"] = True
    tiles = hashlib.sha256(TILES_SO.read_bytes()).hexdigest() if TILES_SO.exists() else None
    import glm_w4a8_tiles_hooks as th
    if tiles != th.SO_SHA256:
        raise SystemExit(f"{TILES_SO}: sha256 {tiles} != the production tiles pin {th.SO_SHA256} (push it first)")
    out["tiles_so_sha256"] = tiles
    print(json.dumps(out, indent=1), flush=True)
    return out


def bench(name: str, args: list) -> dict:
    so = newest_so()
    if so is None:
        raise SystemExit("no k3amd_w4a8dg.so under /root/k3amd_ext: run the build step")
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / f"glm_moe_box_{name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    env = dict(os.environ, K3A_TILES_SO=str(TILES_SO),
               PYTHONPATH=":".join([str(SGL), str(REPO / "operations/kernels"), str(REPO / "operations/patches")]))
    cmd = [sys.executable, str(REPO / "operations/kernels/glm_moe_decode_bench.py"), "--json", str(js),
           "--cases", "none", "--w4a8dg", str(so)] + args
    t0 = time.time()
    r = sh(cmd, env=env, timeout=int(os.environ.get("K3A_BENCH_TIMEOUT", "1500")))
    res = json.loads(js.read_text()) if js.exists() else {"error": f"rc={r.returncode}"}
    res.update(step=name, so=str(so), so_sha256=hashlib.sha256(so.read_bytes()).hexdigest(),
               seconds=round(time.time() - t0, 1), stderr_tail=r.stderr[-3000:], host=os.uname().nodename)
    js.write_text(json.dumps(res, indent=1))
    print("K3A_JSON_BEGIN", name)
    print(json.dumps(res))
    print("K3A_JSON_END", name, flush=True)
    return res


def main(argv: list) -> int:
    steps = argv or ["all"]
    if steps == ["all"]:
        steps = ["build", "prep", "gemm", "call"]
    for s in steps:
        if s == "build":
            step_build()
        elif s == "prep":
            step_prep()
        elif s == "gemm":
            bench("gemm", ["--prefill-mm", MS, "--prefill-layouts", "ep8,ep4", "--w4a8dg-cfgs", GEMM_CFGS,
                           "--iters", "10"])
        elif s == "call":
            bench("call", ["--moe-call", MS, "--moe-call-layouts", "ep8,ep4", "--moe-call-dg", CALL_DG,
                           "--moe-call-silu", CALL_SILU, "--iters", "10"])
        else:
            raise SystemExit(f"unknown step {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
