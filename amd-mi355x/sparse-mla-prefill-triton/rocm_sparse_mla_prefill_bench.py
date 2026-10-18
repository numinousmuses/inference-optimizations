#!/usr/bin/env python3
"""Sparse-MLA prefill kernel study on one MI355X: the stock HIP TileLang path (tilelang_sparse_fwd, FP8 KV -> the
decode-shaped `sparse_mla_fwd_decode_partial_fp8` with a fixed 16-head tile + combine) against variants that tile more
heads per block, at the served DP-attention shape (64 local heads, top-2048, d_v 512 + rope 64, FP8 raw KV).

Why (profile P3, 2026-09-29): on the TP4xDP4 replica the TileLang `main_kernel` is 37% of prefill GPU time. With
h_per_block = 16 a 64-head token runs 4 blocks that each gather the same 2048 KV rows (4x the KV bytes) and issue
M=16 MFMAs. Variants are built from the engine's own kernel source with only `h_per_block` changed, so correctness
is checked against the stock output (same math, different tiling) and a torch fp32 reference.

    HIP_VISIBLE_DEVICES=3 python3 rocm_sparse_mla_prefill_bench.py [--tokens 4096] [--heads 64]
"""
from __future__ import annotations

import argparse
import inspect
import pathlib
import json
import sys
import time

import torch


def build_variant(mod, h_per_block: int, policy: str | None = None):
    """A copy of the engine's tilelang_kernel module (written to a file: TileLang parses kernel source through
    inspect) whose FP8 partial kernel uses `h_per_block` heads per block; returns the copy's tilelang_sparse_fwd."""
    import importlib.util
    full = open(mod.__file__).read()
    assert full.count("    h_per_block = 16\n") == 1
    start = full.index("def sparse_mla_fwd_decode_partial_fp8(")
    end = full.index("\ndef tilelang_sparse_fwd(", start)
    body = full[start:end].replace("    h_per_block = 16\n", f"    h_per_block = {h_per_block}\n")
    if policy:
        # one warp policy for every GEMM of the kernel: with M > 16 the default policy and the tail GEMM's FullCol
        # infer different fragment layouts for acc_s (TileLang InternalError "Get different layout for acc_s")
        body = body.replace(", policy=T.GemmWarpPolicy.FullCol", "")
        body = body.replace("transpose_B=True, clear_accum=False)", f"transpose_B=True, clear_accum=False, policy={policy})")
        body = body.replace("transpose_B=True, clear_accum=True)", f"transpose_B=True, clear_accum=True, policy={policy})")
        body = body.replace("                        transpose_B=True,\n                        clear_accum=True,\n",
                            f"                        transpose_B=True,\n                        clear_accum=True,\n"
                            f"                        policy={policy},\n")
        body = body.replace("                        transpose_B=True,\n                        policy=T.GemmWarpPolicy.FullCol,\n",
                            f"                        transpose_B=True,\n                        policy={policy},\n")
        body = body.replace("sv_tile, clear_accum=True)", f"sv_tile, clear_accum=True, policy={policy})")
    path = f"/tmp/tl_variant_h{h_per_block}_{(policy or 'default').split('.')[-1]}.py"
    open(path, "w").write(full[:start] + body + full[end:])
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.tilelang_sparse_fwd


def run(mod, partial_factory, q, kv, idx, sm_scale, d_v, inner_iter=None):
    num_heads, dim = q.shape[1], q.shape[2]
    tail = dim - d_v
    topk = idx.shape[-1]
    block_I = 64
    ni = topk // block_I
    ii = inner_iter or mod._pick_inner_iter(q.shape[0], ni, 256, 2)
    k_partial = partial_factory(num_heads, d_v, tail, topk, sm_scale=sm_scale, block_I=block_I, inner_iter=ii,
                                threads=256)
    po, pl = k_partial(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0))
    n_groups = ni // ii
    k_comb = mod.sparse_mla_fwd_decode_combine(num_heads, d_v, n_groups * block_I, head_per_block=4,
                                               block_I=block_I, threads=256)
    return k_comb(po, pl)


def timed(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--slots", type=int, default=200_000)
    ap.add_argument("--tile-variants", action="store_true", help="also build the 32-head-tile TileLang variant")
    ap.add_argument("--decode", action="store_true",
                    help="decode/verify row counts (12, 48, 192): stock vs the split-K Triton kernel (split 4/8/16)")
    a = ap.parse_args()
    torch.manual_seed(0)
    from sglang.kernels.ops.attention.dsa import tilelang_kernel as mod
    fp8 = torch.float8_e4m3fn
    T, H, D, DV, K = a.tokens, a.heads, 576, 512, 2048
    kv = (torch.randn(a.slots, 1, D, device="cuda") * 0.5).to(fp8)
    q = (torch.randn(T, H, D, device="cuda") * 0.3).to(fp8)
    # prefill-like indices: tokens of one request pick mostly the same rows
    base = torch.randint(0, a.slots, (K,), device="cuda", dtype=torch.int32)
    idx = base.repeat(T, 1)
    swap = torch.rand(T, K, device="cuda") < 0.1
    idx[swap] = torch.randint(0, a.slots, (int(swap.sum()),), device="cuda", dtype=torch.int32)
    idx = idx.unsqueeze(1).contiguous()
    sm = D ** -0.5
    if a.decode:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from rocm_sparse_mla_prefill import sparse_mla_split_fp8, sparse_mla_prefill_fp8
        rows = []
        for R in (12, 48, 192):
            qq, ii = q[:R].contiguous(), idx[:R].contiguous()
            st = lambda: mod.tilelang_sparse_fwd(q=qq, kv=kv, indices=ii, sm_scale=sm, d_v=DV)  # noqa: E731
            ref = st()
            # fp32 reference on the same FP8 values (q rounded to FP8 as both kernels do)
            q8 = qq.float()
            kf = kv.view(kv.shape[0], D).float()
            ref32 = []
            for rr in range(R):
                sel = ii[rr, 0][ii[rr, 0] >= 0].long()
                kk = kf[sel]
                pr = torch.softmax((q8[rr] @ kk.T) * sm, -1)
                ref32.append(pr @ kk[:, :DV])
            ref32 = torch.stack(ref32).unsqueeze(0)
            rel32 = lambda o: round(((o.float() - ref32).abs().max() / ref32.abs().max()).item(), 5)  # noqa: E731
            r = {"rows": R, "stock_us": round(timed(st, 50) * 1e3, 1), "stock_rel32": rel32(ref)}
            f1 = lambda: sparse_mla_prefill_fp8(qq, kv, ii, sm, DV)  # noqa: E731
            r["one_program_per_row_us"] = round(timed(f1, 50) * 1e3, 1)
            r["one_program_rel32"] = rel32(f1())
            for sp in (4, 8, 16):
                f = lambda: sparse_mla_split_fp8(qq, kv, ii, sm, DV, split=sp)  # noqa: E731
                out = f()
                r[f"split{sp}_us"] = round(timed(f, 50) * 1e3, 1)
                r[f"split{sp}_rel32"] = rel32(out)
            rows.append(r)
            print(json.dumps(r), flush=True)
        json.dump(rows, open("/root/sparse_mla_decode_bench.json", "w"), indent=1)
        return 0
    stock = lambda: mod.tilelang_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=sm, d_v=DV)  # noqa: E731
    ref = stock()
    rows = [{"variant": "stock (h16, engine path)", "ms": round(timed(stock), 2)}]
    flops = T * H * K * (D + DV) * 2
    for hb, pol in (((32, "T.GemmWarpPolicy.FullCol"),) if "--tile-variants" in sys.argv else ()):
        if H % hb:
            continue
        try:
            f = build_variant(mod, hb, pol)
            call = lambda: f(q=q, kv=kv, indices=idx, sm_scale=sm, d_v=DV)  # noqa: E731
            out = call()
            err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
            ms = timed(call)
            rows.append({"variant": f"h_per_block={hb} {pol}", "ms": round(ms, 2), "rel_vs_stock": round(err, 5),
                         "tflops": round(flops / ms / 1e9, 1)})
        except Exception as e:  # noqa: BLE001
            rows.append({"variant": f"h_per_block={hb} {pol}", "error": repr(e)[:300]})
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rocm_sparse_mla_prefill import sparse_mla_prefill_fp8
    # (block_n, warps, stages, waves_per_eu, kpack, nonkdim, head_split); production default: stages 3 (third row)
    sweep = [(64, 4, 1, 0, 2, 16, 1), (64, 4, 2, 0, 2, 16, 1), (64, 4, 3, 0, 2, 16, 1), (64, 4, 4, 0, 2, 16, 1), (32, 4, 2, 0, 2, 16, 1),
             (64, 4, 1, 0, 2, 16, 2), (64, 4, 2, 0, 2, 16, 2), (32, 4, 1, 0, 2, 16, 2), (64, 8, 2, 0, 2, 16, 1),
             (128, 8, 1, 0, 2, 16, 1), (64, 4, 1, 2, 2, 16, 2), (64, 4, 1, 0, 2, 16, 4)]
    if "--full-sweep" in sys.argv:
        sweep += [(64, 4, 1, 0, 0, 0, 1), (64, 4, 1, 0, 2, 0, 1), (64, 4, 1, 0, 0, 16, 1), (64, 4, 1, 0, 0, 32, 1),
                  (128, 4, 1, 0, 0, 0, 1), (64, 4, 1, 1, 0, 16, 1)]
    for bn, nw, ns, wpe, kp, nk, hs in sweep:
        try:
            call = lambda: sparse_mla_prefill_fp8(q, kv, idx, sm, DV, block_n=bn, num_warps=nw, num_stages=ns,  # noqa: E731
                                                  waves_per_eu=wpe, kpack=kp, nonkdim=nk, head_split=hs)
            out = call()
            err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
            ms = timed(call)
            rows.append({"variant": f"triton prefill bn{bn} w{nw} s{ns} wpe{wpe} kp{kp} nk{nk} hs{hs}", "ms": round(ms, 2), "rel_vs_stock": round(err, 5),
                         "tflops": round(flops / ms / 1e9, 1)})
        except Exception as e:  # noqa: BLE001
            rows.append({"variant": f"triton prefill bn{bn} w{nw} s{ns} wpe{wpe} kp{kp} nk{nk} hs{hs}", "error": repr(e)[:300]})
    rows[0]["tflops"] = round(flops / rows[0]["ms"] / 1e9, 1)
    for r in rows:
        print(json.dumps(r), flush=True)
    json.dump({"tokens": T, "heads": H, "rows": rows}, open("/root/sparse_mla_prefill_bench.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
