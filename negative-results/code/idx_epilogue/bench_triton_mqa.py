"""Triton MQA-logits prototype vs DeepGEMM fp8_mqa_logits at the GLM prefill shape (agent glmidx)."""
import json, os, sys, itertools
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import flashinfer  # noqa: F401
import deep_gemm
from operations.kernels.glm_fp4idx_bench import graph_time_us
from operations.kernels.idx_epilogue.triton_mqa_logits import mqa_logits

H, D = 32, 128
gen = torch.Generator(device="cuda").manual_seed(0)
rows, ctx = int(os.environ.get("ROWS", 2048)), int(os.environ.get("CTX", 131072))
k = torch.randn(ctx, D, device="cuda", generator=gen)
s = (k.abs().amax(-1) / 448).contiguous()
k8 = (k / s[:, None]).to(torch.float8_e4m3fn)
q = torch.randn(rows, H, D, device="cuda", generator=gen)
q8 = (q / (q.abs().amax(-1, keepdim=True) / 448)).to(torch.float8_e4m3fn)
w = torch.rand(rows, H, device="cuda", generator=gen)
ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
ke = torch.full((rows,), ctx, dtype=torch.int32, device="cuda")
ref = deep_gemm.fp8_mqa_logits(q8, (k8, s), w, ks, ke, clean_logits=False)
print(json.dumps({"deepgemm_us": graph_time_us(lambda: deep_gemm.fp8_mqa_logits(q8, (k8, s), w, ks, ke, clean_logits=False), 2)}), flush=True)
configs = [dict(bk=128, bq=4, rows_per_prog=256, num_warps=4, num_stages=2, store=st, ws=ws)
           for st in (True, False) for ws in (False, True)]
for c in configs:
    try:
        out = mqa_logits(q8, w, k8, s, ks, ke, **c)
        err = ((out - ref).abs().max() / ref.abs().max()).item() if c.get('store', True) else None
        us = graph_time_us(lambda: mqa_logits(q8, w, k8, s, ks, ke, **c), 2)
        print(json.dumps({**c, "us": round(us, 1), "max_rel_err": err}), flush=True)
    except Exception as e:
        print(json.dumps({**c, "error": repr(e)[:200]}), flush=True)
