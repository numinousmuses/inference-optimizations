"""Fused exact indexer top-k (patched DeepGEMM candidate epilogue) vs stock DeepGEMM logits + top-k v2 (agent glmidx).
Run with the header patched (patch_mqa_cand.py). Correctness: index sets identical to the stock path's exact top-k on
the same scores (ties aside). CUDA-graph timing is not possible for the fused path's host-side threshold sizing, so
both paths are timed eagerly with CUDA events over 10 calls after warm-up (same method for both)."""
import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import flashinfer  # noqa: F401
import deep_gemm
from operations.kernels.idx_epilogue.fused_topk import indexer_topk_fused, Workspace

H, D = 32, 128


def ev_time(fn, n=10):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / n


def stock(q8, w, k8, s, ks, ke):
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
    lg = deep_gemm.fp8_mqa_logits(q8, (k8, s), w, ks, ke, clean_logits=False)
    rows = q8.shape[0]
    o = torch.empty(rows, 2048, dtype=torch.int32, device="cuda")
    topk_transform_ragged_v2(lg, (ke - ks).to(torch.int32), out_offsets=torch.zeros(rows, dtype=torch.int32, device="cuda"),
                             out_indices=o, row_starts=ks)
    return lg, o


gen = torch.Generator(device="cuda").manual_seed(0)
out = []
for ctx in [int(x) for x in os.environ.get("CTXS", "131072,262144,524288,655360").split(",")]:
    rows = 2048
    k = torch.randn(ctx, D, device="cuda", generator=gen)
    # realistic-ish structure: a low-rank shared direction so some keys score high for many rows
    k += 0.5 * torch.randn(1, D, device="cuda", generator=gen) * torch.randn(ctx, 1, device="cuda", generator=gen)
    s = (k.abs().amax(-1) / 448).contiguous()
    k8 = (k / s[:, None]).to(torch.float8_e4m3fn)
    del k
    q = torch.randn(rows, H, D, device="cuda", generator=gen)
    q8 = (q / (q.abs().amax(-1, keepdim=True) / 448)).to(torch.float8_e4m3fn)
    w = torch.rand(rows, H, device="cuda", generator=gen)
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = torch.arange(ctx - rows + 1, ctx + 1, dtype=torch.int32, device="cuda")
    ws = Workspace(rows, ctx, "cuda")
    lg, o_stock = stock(q8, w, k8, s, ks, ke)
    idx, ok, st = indexer_topk_fused(q8, w, k8, s, ks, ke, ws=ws, return_stats=True)
    # correctness: overlap of index sets per row (exact up to ties)
    ov = []
    for r in (0, 1, rows // 2, rows - 1):
        a = set(o_stock[r].tolist()) - {-1}
        b = set(idx[r].tolist()) - {-1}
        ov.append(len(a & b) / max(1, len(a)))
    t_stock = ev_time(lambda: stock(q8, w, k8, s, ks, ke))
    t_logits = ev_time(lambda: deep_gemm.fp8_mqa_logits(q8, (k8, s), w, ks, ke, clean_logits=False))
    t_fused = ev_time(lambda: indexer_topk_fused(q8, w, k8, s, ks, ke, ws=ws))
    r = {"ctx": ctx, "rows": rows, "stock_logits_topk_us": round(t_stock, 1), "stock_logits_only_us": round(t_logits, 1),
         "fused_us": round(t_fused, 1), "set_overlap_min": min(ov), **st, "ok_rows": int(ok.sum())}
    out.append(r)
    print(json.dumps(r), flush=True)
    del lg
    torch.cuda.empty_cache()
json.dump(out, open("/root/glmidx_fused_topk.json", "w"), indent=1)
