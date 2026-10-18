"""A/B of DeepGEMM MQA-logits header variants (agent glmidx): correctness vs a torch reference and CUDA-graph timing at
the GLM shape (32 heads x 128, 2048 prefill rows, causal context) for FP8 (fp32 and bf16 logits), FP4, and the paged
decode / verify kernels. Run once per header variant (fresh process each: DeepGEMM JIT-compiles the header).
    python3 ab_mqa_logits.py --tag stock|hoist --out /root/ab_<tag>.json
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
try:
    import flashinfer  # noqa: F401
except Exception:
    pass
import deep_gemm
from operations.kernels import glm_fp4idx as g
from operations.kernels.glm_fp4idx_bench import graph_time_us

H, D = 32, 128


def check(gen):
    """Small ragged case vs torch: FP8 path and FP4 path, max rel error on valid entries."""
    rows, ctx = 64, 5000
    k = torch.randn(ctx, D, device="cuda", generator=gen)
    s = (k.abs().amax(-1) / 448).contiguous()
    k8 = (k / s[:, None]).to(torch.float8_e4m3fn)
    q = torch.randn(rows, H, D, device="cuda", generator=gen)
    q8 = (q / (q.abs().amax(-1, keepdim=True) / 448)).to(torch.float8_e4m3fn)
    w = torch.randn(rows, H, device="cuda", generator=gen)
    ks = torch.randint(0, 1000, (rows,), device="cuda", dtype=torch.int32)
    ke = torch.randint(3000, ctx + 1, (rows,), device="cuda", dtype=torch.int32)
    ref = g.logits_ref(q8.float(), k8.float() * s[:, None], w)
    got = deep_gemm.fp8_mqa_logits(q8, (k8, s), w, ks, ke, clean_logits=False)
    k4, sk4 = g.quant_q(k.view(-1, 1, D)); q4, s4 = g.quant_q(q8)
    kd = g.dequant_ref(k4.view(torch.uint8).view(ctx, 64).cpu(),
                       torch.stack([(sk4.view(ctx).cpu().to(torch.int64) >> (8 * j)) & 0xFF for j in range(4)], -1)
                       .to(torch.uint8)).cuda()
    qd = g.dequant_ref(q4.view(torch.uint8).view(-1, 64).cpu(),
                       torch.stack([(s4.view(-1).cpu().to(torch.int64) >> (8 * j)) & 0xFF for j in range(4)], -1)
                       .to(torch.uint8)).cuda().view(rows, H, D)
    ref4 = g.logits_ref(qd, kd, w)
    got4 = deep_gemm.fp8_fp4_mqa_logits((q4, s4), (k4.view(ctx, 64), sk4.view(ctx)), w, ks, ke, clean_logits=False)
    mask = (torch.arange(ctx, device="cuda")[None] >= ks[:, None]) & (torch.arange(ctx, device="cuda")[None] < ke[:, None])
    e8 = ((got - ref).abs() * mask).max() / (ref.abs() * mask).max()
    e4 = ((got4 - ref4).abs() * mask).max() / (ref4.abs() * mask).max()
    return {"fp8_max_rel": e8.item(), "fp4_max_rel": e4.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    res = {"tag": a.tag, "check": check(gen), "ragged": []}
    print(json.dumps(res["check"]), flush=True)
    rows = 2048
    for ctx in (131072, 524288):
        k = torch.randn(ctx, D, device="cuda", generator=gen)
        s = (k.abs().amax(-1) / 448).contiguous()
        k8 = (k / s[:, None]).to(torch.float8_e4m3fn)
        k4, sk4 = g.quant_q(k.view(-1, 1, D)); k4, sk4 = k4.view(ctx, 64), sk4.view(ctx)
        del k
        q = torch.randn(rows, H, D, device="cuda", generator=gen)
        q8 = (q / (q.abs().amax(-1, keepdim=True) / 448)).to(torch.float8_e4m3fn)
        q4, s4 = g.quant_q(q8)
        w = torch.rand(rows, H, device="cuda", generator=gen)
        wb = w.to(torch.bfloat16)
        ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
        ke = torch.arange(ctx - rows + 1, ctx + 1, dtype=torch.int32, device="cuda")
        r = {"ctx": ctx, "rows": rows,
             "fp8_us": graph_time_us(lambda: deep_gemm.fp8_mqa_logits(q8, (k8, s), w, ks, ke, clean_logits=False), 2),
             "fp8_bf16_us": graph_time_us(lambda: deep_gemm.fp8_fp4_mqa_logits(
                 q8, (k8, s), wb, ks, ke, clean_logits=False, logits_dtype=torch.bfloat16), 2),
             "fp4_us": graph_time_us(lambda: deep_gemm.fp8_fp4_mqa_logits(
                 (q4, s4), (k4, sk4), w, ks, ke, clean_logits=False), 2)}
        res["ragged"].append(r)
        print(json.dumps(r), flush=True)
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
