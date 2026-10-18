"""Per-layer timing: native FP4 sparse MLA decode (kv4mla) vs the served TRT-LLM FP8 op and kv4glm's FP4 scratch path.

CUDA-graph replay (N launches captured, replayed, events), GLM-5.3 shapes: 64 heads, top-2048, fp8 q [T, 64, 576], pool
slots random over a pool much larger than L2 (600k rows: 346 MB FP8 / 211 MB FP4 per layer). Rows T: decode batches and
verify rows (4 or 6 draft tokens per request, 90% shared picks inside a request, as kv4glm's bench models).

Bytes moved (derived): KV rows read = T x 2048 x row bytes (576 FP8, 352 FP4), plus q (T x 64 x 576) and out (T x 64 x
512 x 2). The split-KV partial outputs (fp32) are extra traffic of the kernel itself and are not counted.

    python3 bench_kv4mla.py [--pool 600000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import flashinfer  # noqa: F401  (the engine's ptxas for Triton)
import flashinfer.decode
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_kv4 as kv4  # noqa: E402
import kv4mla  # noqa: E402

H, K, CTX = 64, 2048, 65536
SCALE = 576 ** -0.5


def graph_time_us(fn, n_launch: int = 20, replays: int = 15) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(n_launch):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    times = []
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000 / n_launch)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=600_000)
    ap.add_argument("--json", default="")
    ap.add_argument("--rows", default="1,8,32,48,64,96,128,v192,v384,v288")
    ap.add_argument("--no-scratch", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    n = a.pool
    g = torch.Generator(device=dev).manual_seed(0)
    nope = (torch.randn(n, 512, device=dev, generator=g) * 1.5).bfloat16()
    rope = (torch.randn(n, 64, device=dev, generator=g) * 1.5).to(torch.float8_e4m3fn)
    pool8 = torch.empty(n, 576, dtype=torch.float8_e4m3fn, device=dev)
    pool8[:, :512] = nope.to(torch.float8_e4m3fn)
    pool8[:, 512:] = rope
    pool4 = torch.empty(n, 1, 352, dtype=torch.uint8, device=dev)
    kv4.quant_write(pool4, torch.arange(n, device=dev), nope, rope)
    del nope, rope
    ws = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    from sglang.srt.layers.attention.trtllm_mla_backend import make_persistent_multi_ctas_kv_counter_buffer
    counter = make_persistent_multi_ctas_kv_counter_buffer(dev, H, 8192)
    dec = kv4mla.Decoder(dev)
    res = []
    for spec in a.rows.split(","):
        verify = spec.startswith("v")
        rows = int(spec.lstrip("v"))
        per = 6 if (verify and rows == 288) else 4
        q8 = (torch.randn(rows, H, 576, device=dev, generator=g) * 1.5).to(torch.float8_e4m3fn)
        table = torch.randint(0, n, (rows, K), device=dev, generator=g, dtype=torch.int32)
        if verify:
            for r0 in range(0, rows, per):
                for i in range(r0 + 1, min(rows, r0 + per)):
                    j = torch.randperm(K, device=dev, generator=g)[: K // 10]
                    table[i] = table[r0]
                    table[i, j] = torch.randint(0, n, (K // 10,), device=dev, generator=g, dtype=torch.int32)
        seq_lens = torch.full((rows,), K, dtype=torch.int32, device=dev)
        out8 = torch.empty(rows, 1, H, 512, dtype=torch.bfloat16, device=dev)
        out4 = torch.empty(rows, H, 512, dtype=torch.bfloat16, device=dev)

        def trt(kv=pool8, tab=table, out=out8):
            flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=q8.view(rows, 1, H, 576), kv_cache=kv.view(-1, 1, 64, 576), workspace_buffer=ws,
                qk_nope_head_dim=192, kv_lora_rank=512, qk_rope_head_dim=64, block_tables=tab.unsqueeze(1),
                seq_lens=seq_lens, max_seq_len=CTX, sparse_mla_top_k=K, bmm1_scale=SCALE, backend="trtllm-gen",
                multi_ctas_kv_counter_buffer=counter, out=out)

        dec.prepare(rows, K)
        r = {"case": ("verify " if verify else "decode ") + str(rows), "rows": rows}
        r["fp8_trtllm_us"] = graph_time_us(trt)
        r["fp4_native_us"] = graph_time_us(lambda: dec(q8, pool4, table, SCALE, out=out4))
        r["fp4_native_nosplit_us"] = graph_time_us(lambda: dec(q8, pool4, table, SCALE, out=out4, split=False))
        if not a.no_scratch:
            scratch = torch.empty(rows * K, 576, dtype=torch.float8_e4m3fn, device=dev)
            ident = torch.empty_like(table)

            def sel():
                kv4.dequant_rows(pool4, table, scratch)
                kv4.identity_table(table, ident)
                trt(kv=scratch, tab=ident)
            r["fp4_scratch_us"] = graph_time_us(sel)
            del scratch
        # accuracy of the native kernel vs FP8-TRT-LLM on the same (unquantized-source) data, and vs its own reference
        dec(q8, pool4, table, SCALE, out=out4)
        trt()
        torch.cuda.synchronize()
        r["native_vs_fp8_maxrel"] = ((out4.float() - out8.float().view(rows, H, 512)).abs().max()
                                     / out8.float().abs().max()).item()
        kvb = rows * K * 352 + rows * H * 576 + rows * H * 512 * 2
        r["fp4_bytes_MB"] = kvb / 1e6
        r["fp8_bytes_MB"] = (rows * K * 576 + rows * H * 576 + rows * H * 512 * 2) / 1e6
        r["fp4_native_TBps"] = kvb / r["fp4_native_us"] / 1e6
        r["fp8_trtllm_TBps"] = r["fp8_bytes_MB"] * 1e6 / r["fp8_trtllm_us"] / 1e6
        r["native_over_fp8"] = r["fp4_native_us"] / r["fp8_trtllm_us"]
        res.append(r)
        print(f"{r['case']:<12} trt {r['fp8_trtllm_us']:7.1f}  native {r['fp4_native_us']:7.1f} "
              f"(nosplit {r['fp4_native_nosplit_us']:7.1f})  scratch {r.get('fp4_scratch_us', float('nan')):7.1f}  "
              f"ratio {r['native_over_fp8']:.2f}  TB/s fp4 {r['fp4_native_TBps']:.2f} fp8 {r['fp8_trtllm_TBps']:.2f}  "
              f"|fp4-fp8| {r['native_vs_fp8_maxrel']:.3f}", flush=True)
    print(f"(us per layer call, CUDA-graph replay medians; {torch.cuda.get_device_name()}, pool {n} rows)")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
