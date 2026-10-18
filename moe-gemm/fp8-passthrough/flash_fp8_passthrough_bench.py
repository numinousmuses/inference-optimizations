"""dp_fp8_passthrough on one H200 at GLM-5.3-Flash's prefill MoE input shape (rtxsys, kernel program 2026-10-05).

Lever: operations/patches/dp_fp8_passthrough_hooks.py + operations/kernels/dp_fp8_passthrough.py.
Served shape: the FP8 DP gather fills a [rows, 4096] bf16 buffer, rows = 4 ranks x 4,096 = 16,384 per step on F4.
- stock: the gather's dequant output is quantized twice:
  - shared-expert DeepGEMM linear quant: column-major TMA-aligned scales (fp8_utils.py:1112-1119);
  - routed cutlass quant: row-major (cutlass_moe.py:214).
- passthrough: the gathered (q, s) is reused. The only cost left is the scale copy into the TMA layout
  (dp_fp8_passthrough.for_linear).
Reports us per call (CUDA-graph replay) and accuracy:
- identical fp8 bytes and max scale relative diff, stock requant vs gathered;
- the DeepGEMM shared gate_up output (TP4 shard N=1024, K=4096, random block-FP8 weight) both ways, against an fp32
  reference.

    python3 flash_fp8_passthrough_bench.py --json out.json [--rows 16384,4096,32768] [--module /root/dp_fp8_passthrough.py]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os


def timed(fn, iters=50):
    import torch
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--rows", default="16384,4096,32768")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--module", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dp_fp8_passthrough.py"))
    a = ap.parse_args()
    os.environ["NUMINOUS_DP_FP8_PASSTHROUGH"] = "1"
    import torch
    from sglang.kernels.ops.quantization.fp8_kernel import (sglang_per_token_group_quant_fp8,
                                                            w8a8_block_fp8_matmul_deepgemm)
    spec = importlib.util.spec_from_file_location("nfp", a.module)
    nfp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nfp)
    H, N = a.hidden, a.n
    out = {"gpu": torch.cuda.get_device_name(), "hidden": H, "n": N, "rows": []}
    torch.manual_seed(0)
    w = torch.randn(N, H, device="cuda") * 0.02
    wb = w.view(N // 128, 128, H // 128, 128)
    ws = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)            # [N/128, K/128]
    wq = (wb / ws[:, None, :, None]).to(torch.float8_e4m3fn).view(N, H)
    w_deq = (wq.float().view(N // 128, 128, H // 128, 128) * ws[:, None, :, None]).view(N, H)
    for rows in [int(r) for r in a.rows.split(",")]:
        x = (torch.randn(rows, H, device="cuda") * torch.logspace(-2, 1, H, device="cuda")).to(torch.bfloat16)
        q, s = sglang_per_token_group_quant_fp8(x, 128)                   # the gather's quant (dp_attention.py:728)
        g = (q.float().view(rows, -1, 128) * s.view(rows, -1, 1)).view(rows, H).to(torch.bfloat16)  # its dequant
        gq = q.view(torch.uint8)
        hold = {}

        def stock():
            hold["l"] = sglang_per_token_group_quant_fp8(g, 128, column_major_scales=True, scale_tma_aligned=True)
            hold["r"] = sglang_per_token_group_quant_fp8(g, 128)

        def passthrough():
            nfp.stash(g, gq, s)
            hold["pl"] = nfp.for_linear(g, 128)
            hold["pr"] = nfp.for_routed(g)

        t_stock, t_pt = timed(stock), timed(passthrough)
        stock()
        passthrough()
        torch.cuda.synchronize()
        (q1, s1), (q2, s2) = hold["l"], hold["r"]
        (pq, ps), (rq, rs) = hold["pl"], hold["pr"]
        same = lambda u, v: float((u.view(torch.uint8) == v.view(torch.uint8)).float().mean())  # noqa: E731
        srel = lambda u, v: float(((u.float() - v.float()).abs() / v.float().abs().clamp_min(1e-30)).max())  # noqa
        o_stock = w8a8_block_fp8_matmul_deepgemm(q1, wq, s1, ws, [128, 128], output_dtype=torch.bfloat16)
        o_pt = w8a8_block_fp8_matmul_deepgemm(pq, wq, ps, ws, [128, 128], output_dtype=torch.bfloat16)
        ref = x.float() @ w_deq.t()
        rel = lambda o: float((o.float() - ref).norm() / ref.norm())  # noqa: E731
        row = {"rows": rows, "stock_two_quants_us": round(t_stock, 2), "passthrough_us": round(t_pt, 2),
               "saving_us": round(t_stock - t_pt, 2),
               "linear_same_fp8_bytes": same(q1, pq), "linear_scale_max_rel": srel(s1, ps),
               "linear_scale_stride_match": tuple(s1.stride()) == tuple(ps.stride()),
               "routed_same_fp8_bytes": same(q2, rq), "routed_scale_max_rel": srel(s2, rs),
               "deepgemm_out_max_abs_diff": float((o_stock.float() - o_pt.float()).abs().max()),
               "deepgemm_rel_l2_stock": round(rel(o_stock), 6), "deepgemm_rel_l2_passthrough": round(rel(o_pt), 6)}
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
