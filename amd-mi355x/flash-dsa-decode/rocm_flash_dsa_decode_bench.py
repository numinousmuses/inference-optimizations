#!/usr/bin/env python3
"""Flash DSA decode / verify attention on gfx950: stock TileLang tilelang_sparse_fwd vs rtxsys's split-K Triton kernel
(rocm_flash_dsa_decode.py) vs amdflash's one-program-per-row kernel (flash_sparse_prefill.py), CUDA-graph timed, at
Flash's shape: 64 heads (TP1), rope-free 512-dim FP8 KV, top-2048, verify rows = requests x 8 (MTP 7/1/8).

    HIP_VISIBLE_DEVICES=<g> bash operations/amdpro/chroot_exec.sh /root/img/20261002-rocm724 -- \\
        python3 operations/kernels/rocm_flash_dsa_decode_bench.py --json out.json
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "amdflash"))
from rocm_fp8_smallm_bench import graph_us  # noqa: E402
from rocm_flash_dsa_decode import flash_sparse_decode_fp8  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="64,128,256,384")
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2_000_000)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    try:
        from flash_sparse_prefill import flash_sparse_prefill_fp8
    except ImportError:
        flash_sparse_prefill_fp8 = None
    g = torch.Generator().manual_seed(0)
    kv = (torch.randn(a.slots, 1, 512, generator=g) * 0.5).to(torch.float8_e4m3fn).cuda()
    sm = 512 ** -0.5
    rows_out = []
    for t in (int(x) for x in a.rows.split(",")):
        q = (torch.randn(t, a.heads, 512, generator=g) * 0.3).to(torch.bfloat16).cuda()
        idx = torch.randint(1, a.slots, (t, 1, 2048), generator=g, dtype=torch.int32)
        idx[:, :, 2000:] = -1
        idx = idx.cuda()
        ref = tilelang_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=sm, d_v=512).float()
        amax = ref.abs().max().item()
        row = {"rows": t, "heads": a.heads, "tilelang_us": graph_us(
            lambda: tilelang_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=sm, d_v=512), iters=20)}
        cands = {f"split{s}_hs{h}": (lambda s=s, h=h: flash_sparse_decode_fp8(q, kv, idx, sm, split=s, head_split=h))
                 for s in (1, 2, 4, 8, 16) for h in (1, 2, 4)}
        if flash_sparse_prefill_fp8 is not None:
            cands["prefill_row"] = lambda: flash_sparse_prefill_fp8(q, kv, idx, sm, 512)
        for name, fn in cands.items():
            try:
                o = fn().float()
                row[name + "_us"] = graph_us(fn, iters=20)
                row[name + "_rel"] = round((o - ref).abs().max().item() / amax, 4)
            except Exception as e:  # noqa: BLE001
                row[name + "_us"] = f"{type(e).__name__}: {str(e)[:80]}"
        best = min((v, k) for k, v in row.items() if k.endswith("_us") and isinstance(v, float) and k != "tilelang_us")
        row["best"] = best[1]
        row["best_vs_tilelang"] = round(row["tilelang_us"] / best[0], 2)
        print(json.dumps({k: v for k, v in row.items() if k in ("rows", "tilelang_us", "best", "best_vs_tilelang")
                          or k == best[1] or k == best[1].replace("_us", "_rel")}), flush=True)
        rows_out.append(row)
    if a.json:
        json.dump(rows_out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
