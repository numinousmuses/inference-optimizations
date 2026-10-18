"""1xH100 microbench: Gemma 4 31B TARGET_VERIFY attention, stock v0.5.20 extend_attention_fwd vs verify_gqa (rtxattn).

What the server runs today (triton_backend.py forward_extend, target verify, topk 1). For each layer it calls
extend_attention_fwd with:
  - the chain custom mask over [6 x (seq_len + 6)] per request (init_forward_metadata :940-945);
  - is_causal True;
  - on SWA layers: the window kv list (window_kv_indptr / indices), sliding_window_size 1024 and window_kv_offsets
    = seq_len - window list length;
  - FP8 KV with k/v scales.
Shapes (operations/kernels/gemma31_roofline.py):
  - SWA: 32 q / 16 kv heads, hd 256, window 1024;
  - full: 32 q / 4 kv heads, hd 512.
Sessions 32 / 48 x 6 draft tokens; context 1.4K (the measured OR mix), 8K, 32K.
Timing: CUDA graphs, 10 calls per graph, replayed 20x, median per call. Accuracy: max |diff| and rel L2 of the
verify_gqa output vs stock. Output JSON: --json.

    python3 operations/kernels/gemma31_verify_bench.py --json /root/g31_verify_bench.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gemma31_verify as V  # noqa: E402


def timeit_graph(fn, calls=10, replays=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    ts = []
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / calls)
    return statistics.median(ts)


def build(bs, ctx, hq, hkv, d, window, n_ext=6, dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    w_list = min(ctx, window) if window > 0 else ctx           # kv list length per request (window list on SWA)
    pool = bs * w_list + 16
    ks, vs = 0.02, 0.02
    kb = (torch.randn(pool, hkv, d, device=dev, generator=g) / ks).clamp(-448, 448).to(torch.float8_e4m3fn)
    vb = (torch.randn(pool, hkv, d, device=dev, generator=g) / vs).clamp(-448, 448).to(torch.float8_e4m3fn)
    kvi = torch.randperm(pool, device=dev, generator=g)[:bs * w_list].to(torch.int32)
    kvp = torch.arange(0, bs * w_list + 1, w_list, device=dev, dtype=torch.int32)
    qo = torch.arange(0, bs * n_ext + 1, n_ext, device=dev, dtype=torch.int32)
    q = torch.randn(bs * n_ext, hq, d, device=dev, generator=g).to(torch.bfloat16)
    ke = torch.randn(bs * n_ext, hkv, d, device=dev, generator=g).to(torch.bfloat16)
    ve = torch.randn(bs * n_ext, hkv, d, device=dev, generator=g).to(torch.bfloat16)
    # the server's chain mask over the FULL sequence: [n_ext x (ctx + n_ext)] per request
    m = torch.ones(n_ext, ctx + n_ext, dtype=torch.bool, device=dev)
    m[:, ctx:] = torch.tril(torch.ones(n_ext, n_ext, dtype=torch.bool, device=dev))
    custom_mask = m.flatten().repeat(bs)
    mask_indptr = torch.arange(0, bs * m.numel() + 1, m.numel(), device=dev, dtype=torch.int64)
    window_kv_offsets = torch.full((bs,), ctx - w_list, device=dev, dtype=torch.int32) if window > 0 else None
    return dict(q=q, ke=ke, ve=ve, kb=kb, vb=vb, qo=qo, kvp=kvp, kvi=kvi, ks=ks, vs=vs, cm=custom_mask,
                mi=mask_indptr, wko=window_kv_offsets, n_ext=n_ext)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--ctx", default="1400,8192,32768")
    ap.add_argument("--sessions", default="32,48")
    ap.add_argument("--sweep", action="store_true", help="full layer, ctx 1400, 32 sessions: launch-config sweep")
    a = ap.parse_args(argv)
    if a.sweep:
        t = build(32, 1400, 32, 4, 512, 0)
        o = torch.empty_like(t["q"])
        out = []
        for nw in (4, 8):
            for bn in (16, 32, 64):
                for ns in (1, 2, 3):
                    for sp in (None, 2, 4, 8):
                        def f():
                            V.verify_gqa(t["q"], t["ke"], t["ve"], t["kb"], t["vb"], o, t["qo"], t["kvp"], t["kvi"],
                                         t["n_ext"], 512 ** -0.5, t["ks"], t["vs"], 0, n_splits=sp, block_n=bn,
                                         num_warps=nw, num_stages=ns)
                        try:
                            us = round(timeit_graph(f), 1)
                        except Exception as e:  # noqa: BLE001
                            us = f"{type(e).__name__}"
                        out.append({"num_warps": nw, "block_n": bn, "num_stages": ns, "splits": sp, "us": us})
                        print(json.dumps(out[-1]), flush=True)
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
        return 0
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
    res = {"device": torch.cuda.get_device_name(), "rows": []}
    layers = {"swa": dict(hq=32, hkv=16, d=256, window=1024, n_layers=50),
              "full": dict(hq=32, hkv=4, d=512, window=0, n_layers=10)}
    for bs in [int(x) for x in a.sessions.split(",")]:
        for ctx in [int(x) for x in a.ctx.split(",")]:
            for name, L in layers.items():
                t = build(bs, ctx, L["hq"], L["hkv"], L["d"], L["window"])
                o_s = torch.empty_like(t["q"])
                o_n = torch.empty_like(t["q"])
                sm = L["d"] ** -0.5

                def stock():
                    extend_attention_fwd(t["q"], t["ke"], t["ve"], o_s, t["kb"], t["vb"], t["qo"], t["kvp"], t["kvi"],
                                         t["cm"], True, t["mi"], t["n_ext"], t["ks"], t["vs"], sm, logit_cap=0.0,
                                         sliding_window_size=L["window"] if L["window"] > 0 else -1,
                                         window_kv_offsets=t["wko"])

                def ours():
                    V.verify_gqa(t["q"], t["ke"], t["ve"], t["kb"], t["vb"], o_n, t["qo"], t["kvp"], t["kvi"],
                                 t["n_ext"], sm, t["ks"], t["vs"], window=L["window"])

                row = {"layer": name, "sessions": bs, "ctx": ctx}
                try:
                    stock()
                    ours()
                    torch.cuda.synchronize()
                    diff = (o_n.float() - o_s.float())
                    row["max_abs_diff"] = round(diff.abs().max().item(), 6)
                    row["rel_l2"] = round((diff.norm() / o_s.float().norm()).item(), 6)
                    # both against an fp32 reference on the first 2 requests (stock casts q and p to the FP8 cache
                    # dtype before its dots, so ours vs stock alone does not say who is closer)
                    n2 = 2 * t["n_ext"]
                    sub = {k2: t[k2] for k2 in ("qo", "kvp")}
                    ref = V.verify_ref(t["q"][:n2].cpu(), t["ke"][:n2].cpu(), t["ve"][:n2].cpu(), t["kb"].cpu(),
                                       t["vb"].cpu(), sub["qo"][:3].cpu(), sub["kvp"][:3].cpu(), t["kvi"].cpu(),
                                       t["n_ext"], sm, t["ks"], t["vs"], window=L["window"]).to(o_s.device)
                    for tag, o in (("stock", o_s), ("ours", o_n)):
                        e = o[:n2].float() - ref
                        row[f"{tag}_rel_l2_vs_fp32"] = round((e.norm() / ref.norm()).item(), 6)
                    row["stock_us"] = round(timeit_graph(stock), 1)
                    row["ours_us"] = round(timeit_graph(ours), 1)
                    row["speedup"] = round(row["stock_us"] / row["ours_us"], 2)
                    row["stock_ms_all_layers"] = round(row["stock_us"] * L["n_layers"] / 1000, 2)
                    row["ours_ms_all_layers"] = round(row["ours_us"] * L["n_layers"] / 1000, 2)
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(json.dumps(row), flush=True)
                res["rows"].append(row)
                del t
                torch.cuda.empty_cache()
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
