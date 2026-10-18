"""End-to-end check of flash_sparse_prefill.maybe_forward against the hooked decode path, 1x H200, served shapes
(amdflash 2026-10-05). It uses a real-size FP8 528-byte pool (3.16M rows, the F4 prefill pool) and three requests of
30K / 6K / 60K tokens at scattered pool locations; the page table carries stale garbage past each request's length.
There are 4,096 / 8,192 query rows (half on request 0, a quarter each on 1 and 2). Each row has 2,051 top-k pool
locations from its own request.

  D  rtxsys hopper_fp8kv path: per 2,048-row slice get_mla_metadata + flash_mla_with_kvcache over the whole pool + cat
  P  maybe_forward:
     - first layer of a forward: builds the inverse map;
     - later layers: hit the per-forward cache;
     - every layer: compact dequant + remap + flash_mla_sparse_fwd.
Accuracy: P vs D; the Triton dequant vs the torch reference (bitwise on the GPU).

    python3 flash_sparse_prefill_e2e_bench.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
import types

import torch

os.environ["NUMINOUS_FLASH_SPARSE_PREFILL"] = "1"
import flash_sparse_prefill as fsp  # noqa: E402

H, D, POOL, TOPK, PADW = 64, 512, 3_158_336, 2051, 2112
LENS = (30_000, 6_000, 60_000)


def timed(fn, iters=10, warmup=3):
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        out = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, out


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-12))


def make_pool(rows, g):
    x = torch.randn(rows, 4, 128, device="cuda", generator=g, dtype=torch.float32) * 0.5
    scale = x.abs().amax(-1).clamp(min=1e-8) / 448.0
    q8 = (x / scale[..., None]).to(torch.float8_e4m3fn).view(rows, 512)
    pool = torch.empty(rows, 528, dtype=torch.uint8, device="cuda")
    pool[:, :512] = q8.view(torch.uint8)
    pool[:, 512:] = scale.contiguous().view(torch.uint8).view(rows, 16)
    del x, q8
    return pool


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    res = {"device": torch.cuda.get_device_name(0), "cases": []}
    g = torch.Generator(device="cuda").manual_seed(0)
    pool = make_pool(POOL, g)
    perm = torch.randperm(POOL, device="cuda", generator=g)[:sum(LENS)].to(torch.int32)
    lmax = max(LENS)
    md_pt = torch.randint(0, POOL, (len(LENS), lmax), device="cuda", generator=g, dtype=torch.int32)   # stale garbage
    req_locs, off = [], 0
    for i, n in enumerate(LENS):
        md_pt[i, :n] = perm[off:off + n]
        req_locs.append(perm[off:off + n])
        off += n
    cu = torch.tensor([0, LENS[0], LENS[0] + LENS[1], sum(LENS)], dtype=torch.int32, device="cuda")
    locs = perm
    # dequant exactness on the GPU
    probe = locs[:4096].long()
    res["dequant_bitwise"] = bool(torch.equal(fsp.dequant_528_rows(pool, probe), fsp.dequant_528_ref(pool, probe)))
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
    k_cache = pool.view(torch.float8_e4m3fn).view(POOL // 64, 64, 1, 528)
    for s_q in (4096, 8192):
        r = {"s_q": s_q}
        try:
            q = (torch.randn(s_q, H, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
            share = (s_q // 2, s_q // 4, s_q - s_q // 2 - s_q // 4)
            topk = torch.cat([rl[torch.randint(0, rl.numel(), (m, TOPK), device="cuda", generator=g)]
                              for rl, m in zip(req_locs, share)], 0)                     # pool locations
            topk[:, -3:][torch.rand(s_q, 3, device="cuda", generator=g) < 0.3] = -1      # kpool-tail style -1s
            topk_pad = torch.nn.functional.pad(topk, (0, PADW - TOPK), value=-1)

            def run_d():
                outs = []
                for s0 in range(0, s_q, 2048):
                    s1 = min(s_q, s0 + 2048)
                    ones = torch.ones(s1 - s0, dtype=torch.int32, device="cuda")
                    meta, splits = get_mla_metadata(cache_seqlens=ones, num_q_tokens_per_head_k=H, num_heads_k=1,
                                                    num_heads_q=H, is_fp8_kvcache=True, topk=PADW)
                    o, _ = flash_mla_with_kvcache(q=q[s0:s1].unsqueeze(1), k_cache=k_cache, cache_seqlens=ones,
                                                  head_dim_v=D, tile_scheduler_metadata=meta, num_splits=splits,
                                                  softmax_scale=D ** -0.5,
                                                  indices=topk_pad[s0:s1].unsqueeze(1).contiguous(),
                                                  block_table=torch.empty((s1 - s0, 0), dtype=torch.int32,
                                                                          device="cuda"),
                                                  is_fp8_kvcache=True)
                    outs.append(o)
                return torch.cat(outs, 0)

            be = types.SimpleNamespace(kv_cache_dim=528, dsa_kv_cache_store_fp8=True, real_page_size=64)
            layer = types.SimpleNamespace(tp_q_head_num=H)

            def run_p(fresh: bool):
                md = types.SimpleNamespace(page_table_1=md_pt, seq_lens_sum=sum(LENS), cu_seqlens_k=cu)
                if not fresh:
                    md._numinous_nsp = run_p.cache
                o = fsp.maybe_forward(be, q.unsqueeze(1), pool, D, D ** -0.5, layer, md, topk)
                run_p.cache = md._numinous_nsp
                return o
            run_p.cache = None
            exp = types.SimpleNamespace(page_table_1=torch.zeros(s_q, 8, dtype=torch.int32, device="cuda"),
                                        seq_lens_sum=8 * s_q, cu_seqlens_k=torch.zeros(s_q + 1, device="cuda"))
            r["expanded_table_falls_back"] = fsp.maybe_forward(be, q.unsqueeze(1), pool, D, D ** -0.5, layer, exp,
                                                               topk) is None
            ms_d, o_d = timed(run_d)
            ms_p1, o_p = timed(lambda: run_p(True))
            ms_p, o_p2 = timed(lambda: run_p(False))
            r.update({"D_ms": round(ms_d, 4), "P_first_layer_ms": round(ms_p1, 4), "P_steady_layer_ms": round(ms_p, 4),
                      "P_vs_D_steady": round(ms_p / ms_d, 3),
                      "saving_ms_per_step_x12": round((ms_d - (ms_p1 + 11 * ms_p) / 12) * 12, 2),
                      "rel_err_P_vs_D": rel(o_p.squeeze(1), o_d.squeeze(1)),
                      "P_shape": list(o_p.shape), "D_shape": list(o_d.shape)})
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            r["tb"] = traceback.format_exc()[-2000:]
        res["cases"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "tb"}), flush=True)
    json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
