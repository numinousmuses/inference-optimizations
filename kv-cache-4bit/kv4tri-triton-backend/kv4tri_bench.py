"""kv4tri kernel timing on one GPU (CUDA-graph replay, never eager loops): NVFP4-native Triton decode/extend vs the
stock SGLang Triton kernels on an FP8 pool (the `triton + fp8` arm) and FlashInfer FP8 decode (the Qwen control's
backend). Shapes = the RTX models' attention layers. Prints one JSON line per case.

    cd /root/work && python3 operations/kernels/kv4tri_bench.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import flashinfer  # noqa: F401  (engine ptxas)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv4tri_nvfp4_attention as k4  # noqa: E402

DEV = "cuda"


def graph_time_us(fn, reps: int = 20, iters: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000.0 / reps)
    return best


def _pools(slots, KVH, D):
    k = (torch.randn(slots, KVH, D, device=DEV) * 2).to(torch.bfloat16)
    v = torch.randn(slots, KVH, D, device=DEV).to(torch.bfloat16)
    from flashinfer import nvfp4_kv_quantize

    one = torch.ones(1, device=DEV)
    kp, ks = nvfp4_kv_quantize(k.view(-1, D), one)
    vp, vs = nvfp4_kv_quantize(v.view(-1, D), one)
    fp4 = (kp.view(slots, KVH, D // 2), vp.view(slots, KVH, D // 2), ks.view(torch.uint8).view(slots, KVH, D // 16),
           vs.view(torch.uint8).view(slots, KVH, D // 16))
    fp8 = (k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn))
    del k, v
    return fp4, fp8


def bench_decode(name, HQ, KVH, D, B, ctx, splits=8):
    slots = B * ctx + 1
    fp4, fp8 = _pools(slots, KVH, D)
    q = torch.randn(B, HQ, D, device=DEV, dtype=torch.bfloat16)
    kv_indptr = torch.arange(0, (B + 1) * ctx, ctx, dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(1, B * ctx + 1, device=DEV, dtype=torch.int64)
    att = torch.empty(B, HQ, splits, D, device=DEV)
    lse = torch.empty(B, HQ, splits, device=DEV)
    nsp = torch.full((B,), splits, dtype=torch.int32, device=DEV)
    o = torch.empty(B, HQ, D, device=DEV, dtype=torch.bfloat16)
    out = {"case": f"decode/{name}", "B": B, "ctx": ctx, "HQ": HQ, "KVH": KVH, "D": D}
    kvb_fp4 = B * ctx * KVH * 2 * (D // 2 + D // 16)
    kvb_fp8 = B * ctx * KVH * 2 * D
    out["fp4_us"] = graph_time_us(lambda: k4.nvfp4_decode_fwd(q, *fp4[:2], *fp4[2:], o, kv_indptr, kv_indices, att,
                                                               lse, nsp, splits, D ** -0.5))
    out["fp4_GBps"] = kvb_fp4 / out["fp4_us"] / 1e3
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

    out["triton_fp8_us"] = graph_time_us(lambda: decode_attention_fwd(q, fp8[0], fp8[1], o, kv_indptr, kv_indices, att,
                                                                      lse, nsp, splits, D ** -0.5, 1.0, 1.0))
    out["triton_fp8_GBps"] = kvb_fp8 / out["triton_fp8_us"] / 1e3
    try:
        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
        last = torch.ones(B, dtype=torch.int32, device=DEV)
        w.plan(kv_indptr, kv_indices.to(torch.int32), last, HQ, KVH, D, 1, q_data_type=torch.bfloat16,
               kv_data_type=torch.float8_e4m3fn)
        kc = fp8[0].view(slots, 1, KVH, D)
        vc = fp8[1].view(slots, 1, KVH, D)
        out["flashinfer_fp8_us"] = graph_time_us(lambda: w.run(q, (kc, vc)))
        out["flashinfer_fp8_GBps"] = kvb_fp8 / out["flashinfer_fp8_us"] / 1e3
    except Exception as e:  # noqa: BLE001
        out["flashinfer_fp8_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    del fp4, fp8
    torch.cuda.empty_cache()


def bench_extend(name, HQ, KVH, D, prefix, new, window=-1):
    slots = prefix + 1
    fp4, fp8 = _pools(slots, KVH, D)
    q = torch.randn(new, HQ, D, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(new, KVH, D, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(new, KVH, D, device=DEV, dtype=torch.bfloat16)
    o = torch.empty_like(q)
    qo = torch.tensor([0, new], dtype=torch.int32, device=DEV)
    p_eff = min(prefix, window) if window > 0 else prefix
    kvp = torch.tensor([0, p_eff], dtype=torch.int32, device=DEV)
    idx = torch.arange(1 + prefix - p_eff, prefix + 1, device=DEV, dtype=torch.int64)
    out = {"case": f"extend/{name}", "prefix": prefix, "new": new, "HQ": HQ, "KVH": KVH, "D": D, "window": window}
    out["fp4_us"] = graph_time_us(lambda: k4.nvfp4_extend_fwd(q, ke, ve, o, *fp4[:2], *fp4[2:], qo, kvp, idx, new,
                                                               D ** -0.5, sliding_window_size=window), reps=3, iters=3)
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

    out["triton_fp8_us"] = graph_time_us(lambda: extend_attention_fwd(q, ke, ve, o, fp8[0], fp8[1], qo, kvp, idx, None,
                                                                      True, None, new, 1.0, 1.0, D ** -0.5,
                                                                      sliding_window_size=window), reps=3, iters=3)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import extend_attention_gqa as eg

        cfg = (dict(block_t=8, block_n=128, dk=64, dv=256, num_warps=8, num_stages=3) if D >= 512 else
               dict(block_t=8, block_n=128, dk=64, dv=D, num_warps=8, num_stages=3) if HQ // KVH >= 4 else
               dict(block_t=32, block_n=64, dk=64, dv=D, num_warps=8, num_stages=1))
        out["gqa_overlay_fp8_us"] = graph_time_us(
            lambda: eg.extend_attention_gqa(q, ke, ve, o, fp8[0], fp8[1], qo, kvp, idx, new, D ** -0.5,
                                            sliding_window_size=window, **cfg), reps=3, iters=3)
    except Exception as e:  # noqa: BLE001
        out["gqa_overlay_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    del fp4, fp8
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="decode|extend")
    a = ap.parse_args()
    shapes = {"qwen38": (24, 4, 256), "qwen36": (16, 2, 256), "gemma_global": (16, 2, 512),
              "gemma_sliding": (16, 8, 256)}
    if a.only in ("", "decode"):
        for name, (hq, kvh, d) in shapes.items():
            ctxs = [1023] if name == "gemma_sliding" else ([8192, 104000] if not a.quick else [32768])
            for ctx in ctxs:
                for B in ([1, 16] if not a.quick else [16]):
                    bench_decode(name, hq, kvh, d, B, ctx)
    if a.only in ("", "extend"):
        for name, (hq, kvh, d) in shapes.items():
            w = 1023 if name == "gemma_sliding" else -1
            bench_extend(name, hq, kvh, d, prefix=65536 if not a.quick else 16384, new=4096, window=w)
