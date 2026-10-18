"""KDA small bf16 projections: separate (served) vs fused, on one H200 at GLM-5.3-Flash's shapes (rtxsys, kernel
program 2026-10-05; lever candidate "FP8 KDA projections" remainder).

Served path with a quant_config (glm5_next.py:340, do_fuse_qkvbfg False): the FP8 checkpoint leaves these bf16, so
each KDA layer runs five cuBLAS GEMMs beside the FP8 qkv/o:
- b_proj: H -> 64 (heads, attn_tp 1);
- f_a_proj, g_a_proj: H -> 128;
- f_b_proj, g_b_proj: 128 -> 8192.
F4 prefill TP0 measured them at ~87 us/layer (14 + 10.2 + 23 + 16.4 + 23).
Fused candidate, the unquantized model's own layout (forward_qkvbfg_fused):
- one GEMM H -> 64 + 128 + 128 (b | f_a | g_a);
- one batched GEMM [2, M, 128] x [2, 128, 8192] (f_b, g_b).
Reports us per KDA layer under graph replay for M = prefill tokens and decode verify tokens, plus the max abs diff
fused vs separate (expect 0: same products, different launch grouping).

    python3 flash_kda_smallproj_bench.py --json out.json [--m 4096,8192,24,48,96,120,144]
"""
from __future__ import annotations

import argparse
import json


def timed(fn, iters=100):
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
    ap.add_argument("--m", default="4096,8192,24,48,96,120,144")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--head-dim", type=int, default=128)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    H, NH, D = a.hidden, a.heads, a.head_dim
    P = NH * D
    torch.manual_seed(0)
    bf = torch.bfloat16
    wb, wfa, wga = (torch.randn(n, H, device="cuda", dtype=bf) * 0.02 for n in (NH, D, D))
    wfb, wgb = (torch.randn(P, D, device="cuda", dtype=bf) * 0.05 for _ in range(2))
    w_a = torch.cat([wb, wfa, wga], 0).contiguous()                  # [NH + 2D, H]
    w_b = torch.stack([wfb.t(), wgb.t()], 0).contiguous()            # [2, D, P]
    out = {"gpu": torch.cuda.get_device_name(), "hidden": H, "heads": NH, "head_dim": D, "m": []}
    for m in [int(x) for x in a.m.split(",")]:
        x = torch.randn(m, H, device="cuda", dtype=bf)
        res = {}

        def separate():
            res["b"] = F.linear(x, wb)
            res["f"] = F.linear(F.linear(x, wfa), wfb)
            res["g"] = F.linear(F.linear(x, wga), wgb)

        def fused():
            y = F.linear(x, w_a)
            b, fg = y.split([NH, 2 * D], dim=-1)
            fgb = torch.bmm(fg.view(m, 2, D).transpose(0, 1), w_b)       # [2, m, P]
            res["fb"], res["ff"], res["fg"] = b, fgb[0], fgb[1]

        t_sep, t_fus = timed(separate), timed(fused)
        separate()
        fused()
        torch.cuda.synchronize()
        diff = max(float((res["b"].float() - res["fb"].float()).abs().max()),
                   float((res["f"].float() - res["ff"].float()).abs().max()),
                   float((res["g"].float() - res["fg"].float()).abs().max()))
        row = {"m": m, "separate_us": round(t_sep, 2), "fused_us": round(t_fus, 2),
               "saving_us": round(t_sep - t_fus, 2), "max_abs_diff": diff,
               "per_step_ms_34_layers": round((t_sep - t_fus) * 34 / 1e3, 3)}
        out["m"].append(row)
        print(json.dumps(row), flush=True)
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
