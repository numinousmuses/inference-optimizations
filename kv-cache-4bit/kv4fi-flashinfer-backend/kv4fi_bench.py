"""Bench: kv4fi native NVFP4 decode/verify kernel vs FlashInfer FP8 decode/verify (the shipped path), CUDA-graph replay.

    python3 operations/kernels/kv4fi_bench.py [--out research/gpu-evidence/rtx/kv4fi_decode_bench.json]

Shapes: Qwen 3.8 full-attention layer (24 q / 4 kv x 256) and Qwen 3.6 (16 / 2 x 256); batch 1, 8, 16; kv length
8K, 32K, 100K per request; decode (QL=1) and EAGLE verify (QL=4, causal). Times are per call (one layer), median of
graph replays; the FP8 comparison uses FlashInfer BatchDecodeWithPagedKVCacheWrapper (tensor cores) with an
fp8_e4m3 KV cache and BatchPrefillWithPagedKVCacheWrapper for verify, planned with the same page-1 indices.
Also sweeps kernel tuning (BLOCK_N, warps, scale expansion) and records the best per shape.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import statistics
import sys
import time

import torch

import flashinfer  # noqa: F401  (ptxas path; and the FP8 comparison)
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kv4fi_nvfp4_decode as K  # noqa: E402

D = 256
DEV = "cuda"


def graph_time_us(fn, iters=20, reps=5) -> float:
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en) * 1000 / iters)
    return statistics.median(times)


def bench_shape(hq, hk, bs, kv_len, ql, tune):
    total = bs * kv_len + 64
    k_fp4 = torch.randint(0, 256, (total, hk, D // 2), dtype=torch.uint8, device=DEV)
    v_fp4 = torch.randint(0, 256, (total, hk, D // 2), dtype=torch.uint8, device=DEV)
    k_sf = torch.randint(0x30, 0x40, (total, hk, D // 16), dtype=torch.uint8, device=DEV)
    v_sf = torch.randint(0x30, 0x40, (total, hk, D // 16), dtype=torch.uint8, device=DEV)
    kv_indptr = torch.arange(0, (bs + 1) * kv_len, kv_len, dtype=torch.int32, device=DEV)
    kv_indices = torch.randperm(total, device=DEV)[: bs * kv_len].to(torch.int32)
    q = torch.randn((bs * ql, hq, D), device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    sm = D**-0.5
    res = {"hq": hq, "hk": hk, "bs": bs, "kv_len": kv_len, "ql": ql}

    # ours: sweep
    best = None
    auto_splits = K.choose_num_kv_splits(bs, hk)
    configs = (list(itertools.product([32, 64, 128], [4, 8], [True, False], [2, 3], [auto_splits, max(1, auto_splits // 2)]))
               if tune else [(64, 4, True, 2, auto_splits)])
    sweep = {}
    for block_n, warps, cvt, stages, splits in configs:
        key = f"bn{block_n}_w{warps}_cvt{int(cvt)}_st{stages}_sp{splits}"
        try:
            t = graph_time_us(lambda: K.nvfp4_attention(
                q, k_fp4, k_sf, v_fp4, v_sf, kv_indptr, kv_indices, 1.0, 1.0, sm, ql=ql, block_n=block_n,
                num_warps=warps, use_cvt=cvt, num_stages=stages, num_kv_splits=splits, out=out))
        except Exception as e:  # a config that does not compile for this shape
            sweep[key] = f"error: {type(e).__name__}: {str(e)[:120]}"
            torch.cuda.empty_cache()
            continue
        sweep[key] = round(t, 1)
        if best is None or t < best[0]:
            best = (t, block_n, warps, cvt, stages, splits)
    if best is None:
        res["kv4fi_error"] = next(iter(sweep.values()))
        return res
    res["kv4fi_us"] = round(best[0], 1)
    res["kv4fi_best"] = {"block_n": best[1], "num_warps": best[2], "use_cvt": best[3], "num_stages": best[4],
                         "num_kv_splits": best[5]}
    res["kv4fi_sweep"] = sweep
    fp4_bytes = bs * kv_len * hk * D * 2 * (0.5 + 1 / 16)
    res["kv4fi_eff_TBps"] = round(fp4_bytes / (best[0] * 1e-6) / 1e12, 2)

    # FlashInfer FP8 reference path (what fp8_e4m3 KV runs today)
    k8 = torch.randn((total, 1, hk, D), device=DEV).to(torch.float8_e4m3fn)
    v8 = torch.randn((total, 1, hk, D), device=DEV).to(torch.float8_e4m3fn)
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=DEV)
    last = torch.ones(bs, dtype=torch.int32, device=DEV)
    if ql == 1:
        w = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
        w.plan(kv_indptr, kv_indices, last, hq, hk, D, 1, data_type=torch.float8_e4m3fn, q_data_type=torch.bfloat16)
        fn = lambda: w.run(q, (k8, v8), k_scale=1.0, v_scale=1.0, out=out)  # noqa: E731
    else:
        w = BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        qo_indptr = torch.arange(0, (bs + 1) * ql, ql, dtype=torch.int32, device=DEV)
        w.plan(qo_indptr, kv_indptr, kv_indices, last, hq, hk, D, 1, causal=True,
               q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)
        fn = lambda: w.run(q, (k8, v8), k_scale=1.0, v_scale=1.0, out=out)  # noqa: E731
    t8 = graph_time_us(fn)
    res["flashinfer_fp8_us"] = round(t8, 1)
    res["flashinfer_fp8_eff_TBps"] = round(bs * kv_len * hk * D * 2 / (t8 * 1e-6) / 1e12, 2)
    res["ratio_kv4fi_over_fp8"] = round(best[0] / t8, 3)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/gpu-evidence/rtx/kv4fi_decode_bench.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    shapes = [(24, 4), (16, 2)]
    grid = [(1, 8192), (1, 32768), (1, 100000), (8, 8192), (8, 32768), (16, 32768), (16, 100000)]
    if a.quick:
        shapes, grid = [(24, 4)], [(1, 32768), (8, 32768)]
    rows = []
    for (hq, hk), (bs, L), ql in itertools.product(shapes, grid, (1, 4)):
        if bs * L > 1_700_000:
            continue  # more tokens than the FP8 pool holds on 96 GB
        r = bench_shape(hq, hk, bs, L, ql, tune=not a.no_tune)
        rows.append(r)
        torch.cuda.empty_cache()
        print(json.dumps({k: v for k, v in r.items() if k != "kv4fi_sweep"}), flush=True)
    meta = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "flashinfer": flashinfer.__version__,
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "unit": "us per layer call, CUDA-graph median",
            "label": "measured"}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
