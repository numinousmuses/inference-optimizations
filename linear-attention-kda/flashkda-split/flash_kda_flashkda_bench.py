"""FlashKDA (MoonshotAI CUTLASS, fully fused) vs SGLang's Triton chunk_kda at GLM-5.3-Flash's served prefill shapes, 1x H200
(amdflash kernel program 2026-10-05).

Why:
- source_code: Flash's KDA core costs intra 5.27 + inter_solve 5.24 + recompute_w_u 4.57 + delta_h 5.41 + gla_o 6.28
  + cumsum 2.34 = 29.1 ms per 4,096-token-per-rank prefill step (F4 TP0 profile, 34 layers).
- source_code: SGLang's FlashKDA backend caps sequences at 2,048 tokens (kda_flashkda.py:12-15, "Triton wins above").
- FlashKDA's own H20 table (BENCHMARK_H20.md) has T=8192, H=64 at 1.62 ms vs fla chunk_kda 3.17 (1.95x).
- Flash has H=64, K=V=128, gate_lower_bound -5 (config linear_attn_config), so FlashKDA's preconditions hold apart
  from the cap. This bench measures the cap at our shapes.

Both paths get RAW q/k/v/g/beta + A_log/dt_bias/lower_bound and fp32 pool states, as in production:
- Triton: chunk_kda(use_qk_l2norm_in_kernel=True, beta_is_raw=True), which is kda_flashkda._triton_fallback;
- FlashKDA: FlashKDAKernel._flashkda_extend, the production wrapper, called past the 2,048 cap.
FlashKDA is built from source in the container when it is missing (FLASH_KDA_CUDA_ARCHS=90a); the build time is
recorded.

    python3 flash_kda_flashkda_bench.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import torch

H, K = 64, 128
CASES = [
    (4096, [4096]), (4096, [2048, 2048]), (4096, [1300, 547, 2048, 201]),
    (8192, [8192]), (8192, [4096, 4096]), (8192, [1300, 547, 2048, 963, 271, 3063]),
]


def ensure_flash_kda() -> dict:
    info = {}
    try:
        import flash_kda  # noqa: F401
        info["preinstalled"] = True
        return info
    except ImportError:
        info["preinstalled"] = False
    t0 = time.time()
    src = "/tmp/fkda"
    cmds = [["git", "clone", "--recursive", "--depth", "1", "https://github.com/MoonshotAI/FlashKDA.git", src],
            [sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation", src]]
    env = {**os.environ, "FLASH_KDA_CUDA_ARCHS": "90a", "MAX_JOBS": "8"}
    for c in cmds:
        p = subprocess.run(c, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            info["build_error"] = (" ".join(c[:4]), p.returncode, p.stderr[-3000:], p.stdout[-2000:])
            return info
    rev = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    info.update({"build_s": round(time.time() - t0, 1), "flashkda_commit": rev})
    return info


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-12))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    res = {"device": torch.cuda.get_device_name(0), "H": H, "K": K, "build": ensure_flash_kda(), "cases": []}
    if "build_error" in res["build"]:
        json.dump(res, open(a.json, "w"), indent=1)
        return 0
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from sglang.srt.layers.attention.linear.kernels.kda_flashkda import FlashKDAKernel
    fk = FlashKDAKernel()
    g0 = torch.Generator(device="cuda").manual_seed(0)
    A_log = torch.log(torch.empty(H, device="cuda").uniform_(1, 16, generator=g0)).float()
    dt_bias = (torch.randn(H * K, device="cuda", generator=g0) * 0.1).float()
    for T, lens in CASES:
        r = {"T": T, "seq_lens": lens}
        try:
            n = len(lens)
            cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
            q = torch.randn(1, T, H, K, device="cuda", generator=g0).to(torch.bfloat16)
            k = torch.randn(1, T, H, K, device="cuda", generator=g0).to(torch.bfloat16)
            v = torch.randn(1, T, H, K, device="cuda", generator=g0).to(torch.bfloat16)
            g = torch.randn(1, T, H, K, device="cuda", generator=g0).to(torch.bfloat16)
            beta = torch.randn(1, T, H, device="cuda", generator=g0).to(torch.bfloat16)
            pool = (torch.randn(n + 4, H, K, K, device="cuda", generator=g0) * 0.05).float()
            idx = torch.arange(2, 2 + n, dtype=torch.int32, device="cuda")

            # Triton's chunk_kda can change its g input in place (debug run 2: results depended on call order), so
            # each path reads its own copy, refreshed from the originals before the accuracy call.
            gt, gf = g.clone(), g.clone()

            def run_t(states):
                return chunk_kda(q=q, k=k, v=v, g=gt, beta=beta, initial_state=states, initial_state_indices=idx,
                                 use_qk_l2norm_in_kernel=True, cu_seqlens=cu, A_log=A_log, dt_bias=dt_bias,
                                 lower_bound=-5.0, beta_is_raw=True)

            def run_f(states):
                return fk._flashkda_extend(q, k, v, gf, beta, ssm_states=states, cache_indices=idx,
                                           query_start_loc=cu, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
                                           beta_is_raw=True)
            st_t, st_f = pool.clone(), pool.clone()
            o_t = run_t(st_t)
            o_t = o_t[0] if isinstance(o_t, tuple) else o_t
            r["triton_changed_g"] = not torch.equal(gt, g)
            o_f = run_f(st_f)
            r["flashkda_changed_g"] = not torch.equal(gf, g)
            r["rel_err_out"] = rel(o_f.reshape(o_t.shape), o_t)
            r["rel_err_state"] = rel(st_f[idx.long()], st_t[idx.long()])
            bt, bf = pool.clone(), pool.clone()
            r["triton_ms"] = round(timed(lambda: run_t(bt)), 4)
            r["flashkda_ms"] = round(timed(lambda: run_f(bf)), 4)
            r["flashkda_vs_triton"] = round(r["flashkda_ms"] / r["triton_ms"], 3)
            r["saving_ms_per_step_x34"] = round((r["triton_ms"] - r["flashkda_ms"]) * 34, 2)
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            r["tb"] = traceback.format_exc()[-2500:]
        res["cases"].append(r)
        print(json.dumps({kk: vv for kk, vv in r.items() if kk != "tb"}), flush=True)
    json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
