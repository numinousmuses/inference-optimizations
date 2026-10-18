"""kv4trt decode-attention microbench (one full-attention layer, CUDA-graph replay timing):
  - fi_fp8   : FlashInfer BatchDecodeWithPagedKVCacheWrapper, FP8 KV, page 1 (the shipped control's decode kernel)
  - xqa_fp8  : TRT-LLM MHA decode (XQA on SM120 / trtllm-gen on SM100), FP8 KV, page 64 (kv4trt_*_fp8)
  - xqa_fp4  : same kernel family, NVFP4 KV + FP8 block scales, page 64 (kv4trt_*_nvfp4)
  - fi_fp8_verify: FlashInfer paged prefill, causal, q_len = draft tokens, FP8 KV, page 1 (the control's verify)
  - xqa_{fp8,nvfp4}_spec : TRT-LLM MHA verify shape (q_len = draft tokens, causal mask) (the kv4trt hook's verify)
for the Qwen geometries at the sold ~104k context and the lane's concurrency. Per layer microseconds; the step cost is
x16 (Qwen3.8) / x10 (Qwen3.6) full-attention layers.

    python3 operations/kernels/kv4trt_decode_bench.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import flashinfer  # noqa: F401  (first: engine ptxas path)
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_kv4trt_xqa_nvfp4 import GEOMS, build_case, paged, xqa_causal_mask  # noqa: E402

WS = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def graph_time(fn, iters: int = 20, reps: int = 5) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / iters)
    return best


def trt(c: dict, qlen: int = 1):
    geom, page = c["geom"], c["page"]
    hkv, d = geom["hkv"], geom["d"]
    kw = {}
    if c["kv_dtype"] == "nvfp4":
        kv = (paged(c["k_store"], page, hkv, d // 2), paged(c["v_store"], page, hkv, d // 2))
        kw["kv_cache_sf"] = (paged(c["k_sf"], page, hkv, d // 16).view(torch.float8_e4m3fn),
                             paged(c["v_sf"], page, hkv, d // 16).view(torch.float8_e4m3fn))
    else:
        kv = (paged(c["k_store"], page, hkv, d), paged(c["v_store"], page, hkv, d))
    q = c["q"]
    if torch.cuda.get_device_capability()[0] == 10 and c["kv_dtype"] == "fp8":
        q = q.to(torch.float8_e4m3fn)
    if qlen > 1:
        kw["q_len_per_req"] = qlen
        if torch.cuda.get_device_capability()[0] in (9, 12):
            kw["mask"] = xqa_causal_mask(c["sl"].numel(), qlen, "cuda")
    return lambda: flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q, kv_cache=kv, workspace_buffer=WS, block_tables=c["bt"], seq_lens=c["sl"], max_seq_len=262144,
        bmm1_scale=c["ks"] / math.sqrt(d), bmm2_scale=c["vs"], window_left=-1, out_dtype=torch.bfloat16, **kw)


def fi_fp8(geom: dict, bs: int, ctx: int):
    hq, hkv, d = geom["hq"], geom["hkv"], geom["d"]
    n = bs * ctx
    k = (torch.randn(n, hkv, d, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    v = (torch.randn(n, hkv, d, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(WS[: 256 * 1024 * 1024], "NHD", use_tensor_cores=True)
    indptr = torch.arange(0, n + 1, ctx, dtype=torch.int32, device="cuda")
    idx = torch.randperm(n, device="cuda").int()
    last = torch.ones(bs, dtype=torch.int32, device="cuda")
    w.plan(indptr, idx, last, hq, hkv, d, 1, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)
    q = torch.randn(bs, hq, d, device="cuda").bfloat16()
    kv = (k.view(n, 1, hkv, d), v.view(n, 1, hkv, d))
    return lambda: w.run(q, kv, k_scale=1.0, v_scale=1.0)


def fi_fp8_verify(geom: dict, bs: int, ctx: int, qlen: int):
    """The shipped control's verify: FlashInfer paged prefill (causal), q_len = draft tokens, FP8 KV, page 1."""
    hq, hkv, d = geom["hq"], geom["hkv"], geom["d"]
    n = bs * ctx
    k = (torch.randn(n, hkv, d, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    v = (torch.randn(n, hkv, d, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(WS[: 256 * 1024 * 1024], "NHD")
    qo = torch.arange(0, bs * qlen + 1, qlen, dtype=torch.int32, device="cuda")
    indptr = torch.arange(0, n + 1, ctx, dtype=torch.int32, device="cuda")
    idx = torch.randperm(n, device="cuda").int()
    last = torch.ones(bs, dtype=torch.int32, device="cuda")
    w.plan(qo, indptr, idx, last, hq, hkv, d, 1, causal=True, q_data_type=torch.bfloat16,
           kv_data_type=torch.float8_e4m3fn)
    q = torch.randn(bs * qlen, hq, d, device="cuda").bfloat16()
    kv = (k.view(n, 1, hkv, d), v.view(n, 1, hkv, d))
    return lambda: w.run(q, kv, k_scale=1.0, v_scale=1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cases = [("qwen38", 16, 104_000, 4), ("qwen38", 24, 104_000, 4), ("qwen38", 1, 104_000, 4),
             ("qwen36", 44, 104_000, 5), ("qwen36", 64, 104_000, 5), ("qwen36", 1, 104_000, 5)]
    out = {"gpu": torch.cuda.get_device_name(), "sm": list(torch.cuda.get_device_capability()), "rows": []}
    for name, bs, ctx, qlen in cases:
        geom = GEOMS[name]
        row = {"geom": name, "bs": bs, "ctx": ctx}
        try:
            row["fi_fp8_us"] = round(graph_time(fi_fp8(geom, bs, ctx)), 1)
        except Exception as e:  # noqa: BLE001
            row["fi_fp8_err"] = f"{type(e).__name__}: {str(e)[:120]}"
        torch.cuda.empty_cache()
        try:
            row[f"fi_fp8_verify_q{qlen}_us"] = round(graph_time(fi_fp8_verify(geom, bs, ctx, qlen)), 1)
        except Exception as e:  # noqa: BLE001
            row["fi_fp8_verify_err"] = f"{type(e).__name__}: {str(e)[:160]}"
        torch.cuda.empty_cache()
        for kv in ("fp8", "nvfp4"):
            c = build_case(geom, [ctx] * bs, 64, kv, "ckpt_fp8", deq=False)
            for key in ("k", "v", "k_deq", "v_deq"):
                c.pop(key, None)
            torch.cuda.empty_cache()
            row[f"xqa_{kv}_us"] = round(graph_time(trt(c)), 1)
            g = torch.Generator(device="cuda").manual_seed(1)
            c["q"] = torch.randn(bs * qlen, geom["hq"], geom["d"], device="cuda", generator=g).bfloat16()
            row[f"xqa_{kv}_spec_q{qlen}_us"] = round(graph_time(trt(c, qlen)), 1)
            del c
            torch.cuda.empty_cache()
        print(json.dumps(row), flush=True)
        out["rows"].append(row)
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
