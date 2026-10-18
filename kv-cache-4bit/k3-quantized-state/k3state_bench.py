"""k3state per-step overhead on one GPU (CUDA-graph replay timing, no model load).

Per verify step the two-tier layout adds, for all 69 KDA layers at once: the conv-window gather into the request rows
(copy_rows_all_layers) and a commit fold that also writes the quantized mirror (k3state_quant.commit_fold_all_layers)
instead of SGLang's fp32 fold (commit_kda_replayssm_spec_all_layers). Prefill adds a dequant + quant of each
request's rows per layer. Shapes per rank at TP16 (6 heads) and TP8 (12 heads); accept length 4 of 8.

    python3 operations/kernels/k3state_bench.py --out /root/k3state_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3state_quant as kq  # noqa: E402


def graph_time_us(fn, n: int = 20) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (5 * n)


def bench(hv: int, bs: int, layers: int = 69, ring: int = 16, accept: int = 4) -> dict:
    from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import commit_kda_replayssm_spec_all_layers
    dev = "cuda"
    rows = bs + 2
    slots = 5 * bs + 1
    torch.manual_seed(0)
    w = torch.randn(layers, rows, hv, 128, 128, device=dev) * 0.05
    pool32 = torch.randn(layers, slots, hv, 128, 128, device=dev) * 0.05
    rawv = torch.randn(layers, rows, hv, ring, 128, device=dev).to(torch.bfloat16)
    rawk = torch.randn(layers, rows, hv, ring, 128, device=dev).to(torch.bfloat16)
    gk = -torch.rand(layers, rows, hv, ring, 128, device=dev)
    beta = torch.rand(layers, rows, hv, ring, device=dev)
    rawv_s = torch.randn(layers, slots, hv, ring, 128, device=dev).to(torch.bfloat16)
    rawk_s = torch.randn(layers, slots, hv, ring, 128, device=dev).to(torch.bfloat16)
    gk_s = -torch.rand(layers, slots, hv, ring, 128, device=dev)
    beta_s = torch.rand(layers, slots, hv, ring, device=dev)
    slot_idx = torch.randperm(slots - 1, device=dev)[:bs].to(torch.int32)
    row_idx = torch.arange(1, bs + 1, device=dev, dtype=torch.int32)
    acc = torch.full((bs,), accept, device=dev, dtype=torch.int32)
    conv = torch.randn(layers, slots, 3, 3 * hv * 128, device=dev).to(torch.bfloat16)
    cw = torch.zeros(layers, rows, 3, 3 * hv * 128, device=dev, dtype=torch.bfloat16)
    out = {"heads_per_rank": hv, "batch": bs, "layers": layers, "accept": accept}
    out["stock_fold_us"] = graph_time_us(lambda: commit_kda_replayssm_spec_all_layers(
        checkpoint_state=pool32, rawv_cache=rawv_s, rawk_cache=rawk_s, gk_cache=gk_s, beta_cache=beta_s,
        ssm_state_indices=slot_idx, accept_lens=acc, max_cache_len=ring, num_k_heads=hv, null_block_id=-1))
    out["conv_gather_us"] = graph_time_us(lambda: kq.copy_rows_all_layers(conv, slot_idx, cw, row_idx))
    for fmt in kq.FORMATS:
        q = torch.zeros(layers, slots, hv, 128, kq.row_bytes(fmt), dtype=torch.uint8, device=dev)
        out[f"{fmt}_fold_us"] = graph_time_us(lambda: kq.commit_fold_all_layers(
            w_state=w, q_pool=q, rawv=rawv, rawk=rawk, gk=gk, beta=beta, w_indices=row_idx, slot_indices=slot_idx,
            accept_lens=acc, fmt=fmt, num_k_heads=hv))
        out[f"{fmt}_prefill_rows_us_per_layer"] = graph_time_us(
            lambda: (kq.dequant_rows(q[0], slot_idx, w[0], row_idx, fmt),
                     kq.quant_rows(w[0], row_idx, q[0], slot_idx, fmt)))
    state_bytes = layers * bs * hv * 128 * 128 * 4
    out["stock_fold_GBps"] = round(2 * state_bytes / out["stock_fold_us"] / 1e3, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    try:
        import flashinfer  # noqa: F401  (the engine's ptxas path)
    except Exception:  # noqa: BLE001
        pass
    res = []
    for hv in (6, 12):
        for bs in (32, 64, 128):
            r = bench(hv, bs)
            print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            res.append(r)
            torch.cuda.empty_cache()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
