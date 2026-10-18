#!/usr/bin/env python3
"""Split-factor / kernel sweep for the Triton sparse MLA decode at verify row counts (amdglm 2026-10-03): which of
sparse_mla_split_fp8(split=2/4/8/16) or sparse_mla_prefill_fp8(head_split=1/2/4) is fastest per row count, CUDA-graph
timed, error vs TileLang. Feeds the row -> kernel table in rocm_prefill_hooks.tl_sparse_prefill (NUMINOUS_SPARSE_DECODE).

    HIP_VISIBLE_DEVICES=0 python3 rocm_dsa_split_sweep.py [--rows 12,24,48,72,96,144,192,288]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rocm_dsa_decode_bench import DV, D, graph_time_us, make  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="12,24,48,72,96,144,192,288")
    ap.add_argument("--json", default="")
    ap.add_argument("--heads", type=int, default=64, help="local heads (64 = DP attention, 16 = TP4 attention)")
    ap.add_argument("--wide", action="store_true", help="also head-split / stages / warps / block_n variants")
    a = ap.parse_args()
    from rocm_sparse_mla_prefill import sparse_mla_prefill_fp8, sparse_mla_split_fp8
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    sm = D ** -0.5
    res = []
    for rows in [int(x) for x in a.rows.split(",")]:
        q, kv, idx = make(rows, a.heads)
        ref = tilelang_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=sm, d_v=DV).float()
        amax = ref.abs().max().item()
        row = {"rows": rows, "heads": a.heads}
        cands = {f"split{s}": (lambda s=s: sparse_mla_split_fp8(q, kv, idx, sm, DV, split=s)) for s in (2, 4, 8, 16)}
        cands.update({f"one_hs{h}": (lambda h=h: sparse_mla_prefill_fp8(q, kv, idx, sm, DV, head_split=h))
                      for h in (1, 2, 4)})
        # software double-buffered variant (rtxsys 2026-10-03): next block's gather issued before this block's math
        cands.update({f"split{s}_pf": (lambda s=s: sparse_mla_split_fp8(q, kv, idx, sm, DV, split=s, prefetch=True))
                      for s in (1, 2, 4, 8, 16)})
        if a.wide:
            # split x head-split x pipeline variants (smaller per-program accumulators -> more programs resident)
            for sp in (2, 4, 8):
                for hs in (2, 4):
                    cands[f"split{sp}_hs{hs}"] = (lambda sp=sp, hs=hs: sparse_mla_split_fp8(
                        q, kv, idx, sm, DV, split=sp, head_split=hs))
                for st in (1, 2, 3):
                    cands[f"split{sp}_st{st}"] = (lambda sp=sp, st=st: sparse_mla_split_fp8(
                        q, kv, idx, sm, DV, split=sp, num_stages=st))
                cands[f"split{sp}_w8"] = (lambda sp=sp: sparse_mla_split_fp8(q, kv, idx, sm, DV, split=sp,
                                                                              num_warps=8))
                cands[f"split{sp}_bn32"] = (lambda sp=sp: sparse_mla_split_fp8(q, kv, idx, sm, DV, split=sp,
                                                                                block_n=32))
        for name, fn in cands.items():
            try:
                o = fn().float()
                row[name] = graph_time_us(fn)
                row[name + "_rel"] = round((o - ref).abs().max().item() / amax, 4)
            except Exception as e:  # noqa: BLE001
                row[name] = f"{type(e).__name__}: {str(e)[:80]}"
        best = min((k for k in row if k.startswith(("split", "one")) and not k.endswith("_rel")
                    and isinstance(row[k], float)), key=lambda k: row[k])
        row["best"] = best
        print(json.dumps(row), flush=True)
        res.append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
