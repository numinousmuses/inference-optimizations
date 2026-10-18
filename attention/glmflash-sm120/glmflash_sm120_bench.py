"""Microbenchmark (RTX PRO 6000 / SM120) of Flash's DSA attention kernels at served shapes. CUDA-graph replay timing
(capture `reps` launches, replay, CUDA events), never eager loops. NOT RUN YET (no GPU).

    python3 operations/kernels/glmflash_sm120_bench.py --out /root/rtxattn_bench.json            # ~3 min
    python3 operations/kernels/glmflash_sm120_bench.py --tiles --out /root/rtxattn_tiles.json    # Triton tile sweep

Pool: one DSA layer, 2.4M rows (the TP4 bf16 pool size, research/flash-rtx-plan.md): 2.5 GB bf16 / 1.6 GB fp8 /
0.7 GB fp4, far above the 128 MB L2, random page placement. Tables: kpool-shaped (512 groups of 4 consecutive slots +
3 tail slots = 2051, compact). Verify rows: 4 draft tokens per request share 90% of their groups.
Per row: us per layer call, derived byte floor (T x 2051 x row bytes / 1.792 TB/s; head blocks re-read via L2 not
counted), achieved GB/s, TFLOP/s (T x H x 2051 x 1024 x 2).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

try:
    import flashinfer  # noqa: F401  (sets TRITON_PTXAS_BLACKWELL_PATH first)
except ImportError:
    flashinfer = None

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glmflash_sm120_attn as A  # noqa: E402
import glmflash_sm120_fi as FI  # noqa: E402

DEV = "cuda"
N_POOL = 2_400_000
K = 2051
HBM = 1.792e12
SCALE = 1 / 16


def kpool_table(t: int, per_req: int = 1, share: float = 0.9, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_groups = N_POOL // 4
    reqs = (t + per_req - 1) // per_req
    base = torch.randint(1, n_groups, (reqs, 512), generator=g, device=DEV)
    grp = base.repeat_interleave(per_req, 0)[:t].clone()
    if per_req > 1:
        swap = torch.rand(grp.shape, generator=g, device=DEV) > share
        grp[swap] = torch.randint(1, n_groups, (int(swap.sum()),), generator=g, device=DEV)
    idx = (grp.unsqueeze(-1) * 4 + torch.arange(4, device=DEV)).reshape(t, 2048)
    tail = torch.randint(4, N_POOL - 4, (t, 1), generator=g, device=DEV) + torch.arange(3, device=DEV)
    return torch.cat([idx, tail], 1).to(torch.int32).contiguous()


def request_table(t: int, ctx: int, seed: int = 0) -> torch.Tensor:
    """Prefill-realistic table: all T query tokens belong to ONE request whose context is `ctx` tokens (ctx/4 pooled
    groups at random pool positions); each token picks its own 512 groups from that request's groups + a 3-token tail.
    The chunk's union is ctx rows (31K x 656 B = 20 MB: L2-resident), unlike kpool_table's independent rows."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_req_groups = ctx // 4
    req_groups = torch.randperm(N_POOL // 4 - 1, generator=g, device=DEV)[:n_req_groups] + 1
    pick = torch.argsort(torch.rand(t, n_req_groups, generator=g, device=DEV), dim=1)[:, :512]
    grp = req_groups[pick]
    idx = (grp.unsqueeze(-1) * 4 + torch.arange(4, device=DEV)).reshape(t, 2048)
    tail = (req_groups[-1] * 4 + torch.arange(3, device=DEV)).expand(t, 3)
    return torch.cat([idx, tail], 1).to(torch.int32).contiguous()


def pools():
    lat_chunk = 1 << 18
    out = {}
    bf = torch.empty(N_POOL, 1, 512, dtype=torch.bfloat16, device=DEV)
    f8 = torch.zeros(N_POOL, 1, 656, dtype=torch.uint8, device=DEV)
    f8p = torch.zeros(N_POOL, 1, 656, dtype=torch.uint8, device=DEV)  # pow2-scale rows (FlashInfer DSV3_2)
    f4 = torch.empty(N_POOL, 1, 288, dtype=torch.uint8, device=DEV)
    import glmflash_kv4
    for s in range(0, N_POOL, lat_chunk):
        e = min(s + lat_chunk, N_POOL)
        lat = (torch.randn(e - s, 512, device=DEV) * 0.7).to(torch.bfloat16)
        bf[s:e, 0] = lat
        loc = torch.arange(s, e, device=DEV)
        A.nsa656_quant_write(f8, loc, lat)
        A.nsa656_quant_write(f8p, loc, lat, pow2=True)
        glmflash_kv4.quant_write(f4, loc, lat)
    out[A.FMT_BF16], out[A.FMT_FP8_656], out[A.FMT_FP4_288] = bf, f8, f4
    out["fp8_pow2"] = f8p
    return out


def graph_time(fn, reps: int = 20, iters: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
    best = float("inf")
    for _ in range(iters):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) * 1e3 / reps)
    return best


SHAPES = ([("decode", t, 1) for t in (1, 8, 16, 32, 64)] + [("verify", t, 4) for t in (64, 128, 256)]
          + [("prefill", t, 1) for t in (1024, 4096, 8192)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/rtxattn_bench.json")
    ap.add_argument("--heads", default="16,64")
    ap.add_argument("--tiles", action="store_true")
    ap.add_argument("--prefill-ctx", type=int, default=0,
                    help="supplementary: prefill shapes only, tables from one request of this context length")
    a = ap.parse_args()
    t0 = time.time()
    P = pools()
    ws_bytes = torch.zeros(384 << 20, dtype=torch.uint8, device=DEV)
    ws = A.Workspace(ws_bytes)
    try:
        mod = FI.build_module()
    except Exception as e:  # recorded, the Triton rows still run
        mod = None
        print("FI build failed:", e, flush=True)
    rows = []
    heads = [int(x) for x in a.heads.split(",")]
    shapes = [s_ for s_ in SHAPES if s_[0] == "prefill"] if a.prefill_ctx else SHAPES
    for h in heads:
        for phase, t, per_req in shapes:
            idx = request_table(t, a.prefill_ctx) if a.prefill_ctx else kpool_table(t, per_req)
            q = (torch.randn(t, h, 512, device=DEV) * 0.1).to(torch.bfloat16)
            out = torch.empty(t, h, 512, dtype=torch.bfloat16, device=DEV)
            lse = torch.empty(t, h, dtype=torch.float32, device=DEV)
            cands = []
            if a.tiles:
                for f in (A.FMT_BF16, A.FMT_FP8_656, A.FMT_FP4_288):
                    for bn, nw, ns in ((16, 4, 2), (16, 8, 2), (32, 8, 2), (32, 8, 3), (64, 8, 1)):
                        cands.append((f"triton_{A.FMT_NAMES[f]}_bn{bn}_w{nw}_s{ns}", f, dict(block_n=bn, num_warps=nw,
                                                                                         num_stages=ns)))
            else:
                for f in (A.FMT_BF16, A.FMT_FP8_656, A.FMT_FP4_288):
                    cands.append((f"triton_{A.FMT_NAMES[f]}", f, {}))
            for name, f, kw in cands:
                fn = (lambda f=f, kw=kw: A.sparse_mla(q, P[f], idx, SCALE, f, out=out, lse=lse, workspace=ws, **kw))
                rows.append(_row(name, f, phase, t, h, fn))
            if not a.tiles and mod is not None and h in FI.HEADS:
                for mode in ("jit", "stock"):
                    for pow2 in (False, True):
                        fi = FI.FlashInferGlm5Next(mode, workspace=ws_bytes, module=mod, pow2=pow2)
                        kv = P["fp8_pow2"] if pow2 else P[A.FMT_FP8_656]
                        fn = (lambda fi=fi, kv=kv: fi.forward(q, kv, idx, SCALE, out=out))
                        name = f"flashinfer_{mode}_fp8_656" + ("_pow2_dsv3_2" if pow2 else "")
                        rows.append(_row(name, A.FMT_FP8_656, phase, t, h, fn))
            print(f"[{time.time() - t0:.0f}s] h={h} {phase} T={t} done", flush=True)
    doc = {"gpu": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(0).multi_processor_count,
           "pool_rows": N_POOL, "k": K, "rows": rows}
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=1))
    print("wrote", a.out)


def _row(name, fmt, phase, t, h, fn):
    try:
        us = graph_time(fn)
        err = None
    except Exception as e:
        us, err = None, str(e)[:300]
    floor = t * K * A.ROW_BYTES[fmt] / HBM * 1e6
    r = {"kernel": name, "phase": phase, "T": t, "H": h, "us": us, "floor_us": round(floor, 2), "error": err}
    if us:
        r["x_floor"] = round(us / floor, 2)
        r["GBps"] = round(t * K * A.ROW_BYTES[fmt] / (us * 1e-6) / 1e9, 1)
        r["TFLOPs"] = round(t * h * K * 1024 * 2 / (us * 1e-6) / 1e12, 1)
    print(json.dumps(r), flush=True)
    return r


if __name__ == "__main__":
    main()
