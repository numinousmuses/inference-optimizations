"""SM120 NVFP4 MoE grouped-GEMM tactic sweep at GLM-5.3-Flash decode token counts (rtxv41, row 6, 2026-10-03).

Why: rtxsys's v4 decode profile (Daytona box 998e8c3f, n=64) spends 35 ms of a ~66 ms cycle in the MoE, ~31 ms of it
in the two grouped GEMMs. At decode T the EP4 shard (72 local experts x 2048 x 4096, NVFP4) measured 1.08-1.16x its
byte floor (system.md V2, autotuned). This script tells whether any FlashInfer CUTLASS tactic pair beats the autotuner's
choice at the shapes the engine runs. If one does, the recipe pins it; if none does, the GEMMs are at the kernel
ceiling and the remaining cycle has to come from elsewhere.

For each T (engine-wide tokens entering the MoE per decode step = running requests x draft tokens, every rank sees all
T under the FP4 all-gather; the verify and the draft forwards both count):
  1. autotuned baseline: flashinfer's autotune(True) pass, then the median of `cutlass_fused_moe` (what the engine does);
  2. every GEMM1 tactic with GEMM2 default, then every GEMM2 tactic with the best GEMM1 (profile_ids=[g1, g2]);
  3. the byte floor at the measured HBM copy bandwidth, and the share of HBM each reaches.
Writes one JSON row per (T, variant) to --out and a summary with the best pair per T and the per-cycle saving
modeled as 42 MoE layers x (baseline - best) x the number of MoE forwards per cycle at that T.

    python3 operations/kernels/moe_tactic_sweep.py --out /root/moe_tactics.json            # on one RTX PRO 6000
    python3 operations/kernels/moe_tactic_sweep.py --tokens 64,256 --max-tactics 8 --out /tmp/x.json   # quick

Needs a FlashInfer whose cutlass_fused_moe takes `profile_ids` (absolute indices into the runner's combined tactic
list). Without it the script records the autotuned baseline only and says so.
"""
from __future__ import annotations

import argparse
import inspect
import json
import time

H, INTER, E_LOCAL, E_GLOBAL, TOPK, EP = 4096, 2048, 72, 288, 8, 4
MOE_LAYERS = 42


def bench(fn, iters: int = 40, warmup: int = 8) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def hbm_gbs() -> float:
    import torch
    n = 2 << 30
    a = torch.empty(n, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a), iters=10, warmup=2)
    return 2 * n / (ms * 1e-3) / 1e9


def make_case(T: int, seed: int = 0):
    """The EP4 shard SGLang hands FlashInfer (moe_runner/flashinfer_cutlass.py, fp4): random finite NVFP4 weights."""
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=g) * 0.1
    ids = torch.stack([torch.randperm(E_GLOBAL, device="cuda", generator=g)[:TOPK] for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, TOPK, dtype=torch.float32, device="cuda", generator=g)
    out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
    w13 = torch.randint(0, 256, (E_LOCAL, 2 * INTER, H // 2), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (E_LOCAL, H, INTER // 2), dtype=torch.uint8, device="cuda", generator=g)
    sfv = torch.tensor([0.25, 0.5, 1.0, 2.0], device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    w13_sf = sfv[torch.randint(0, 4, (E_LOCAL, 2 * INTER, H // 16), device="cuda", generator=g)].view(torch.float8_e4m3fn)
    w2_sf = sfv[torch.randint(0, 4, (E_LOCAL, H, INTER // 16), device="cuda", generator=g)].view(torch.float8_e4m3fn)
    one = torch.ones((), dtype=torch.float32, device="cuda")
    alpha = torch.full((E_LOCAL,), 1e-3, dtype=torch.float32, device="cuda")
    qs = [one, w13_sf.view(torch.int32), alpha, one, w2_sf.view(torch.int32), alpha]
    local = ids < E_LOCAL
    touched = int(torch.unique(ids[local]).numel())
    wbytes = E_LOCAL * (2 * INTER * H // 2 + H * INTER // 2) * (1 + 1 / 8) * touched / E_LOCAL
    return dict(x=x, ids=ids, w=w, out=out, fc1=w13.view(torch.long), fc2=w2.view(torch.long), qs=qs,
                touched=touched, wbytes=wbytes)


def caller(c: dict, T: int, profile_ids=None):
    from flashinfer.fused_moe import cutlass_fused_moe
    import torch
    kw = dict(output=c["out"], input=c["x"], token_selected_experts=c["ids"], token_final_scales=c["w"],
              fc1_expert_weights=c["fc1"], fc2_expert_weights=c["fc2"], output_dtype=torch.bfloat16, input_sf=None,
              quant_scales=c["qs"], ep_size=EP, ep_rank=0, tp_size=1, tp_rank=0,
              tune_max_num_tokens=1 << max(0, (T - 1).bit_length()))
    if profile_ids is not None:
        kw["profile_ids"] = list(profile_ids)
    return lambda: cutlass_fused_moe(**kw)[0]


def supports_profile_ids() -> bool:
    from flashinfer.fused_moe import cutlass_fused_moe
    try:
        return "profile_ids" in inspect.signature(cutlass_fused_moe).parameters
    except (TypeError, ValueError):
        return False


def sweep_T(T: int, bw: float, max_tactics: int, rows: list) -> dict:
    import torch
    from flashinfer.autotuner import autotune
    c = make_case(T)
    floor_ms = c["wbytes"] / (bw * 1e9) * 1e3
    base = caller(c, T)
    with autotune(True):
        base()
    torch.cuda.synchronize()
    base_ms = bench(base)
    rows.append({"T": T, "variant": "autotuned", "ms": base_ms, "floor_ms": floor_ms, "hbm_share": floor_ms / base_ms})
    best = {"pair": None, "ms": base_ms}
    if not supports_profile_ids():
        rows.append({"T": T, "variant": "sweep", "error": "this flashinfer has no profile_ids; baseline only"})
        return {"T": T, "autotuned_ms": base_ms, "best_ms": base_ms, "best_pair": None, "floor_ms": floor_ms}

    def try_pair(g1: int, g2: int):
        try:
            ms = bench(caller(c, T, (g1, g2)), iters=20, warmup=3)
        except Exception as e:  # noqa: BLE001 -- invalid tactic for this shape: recorded, skipped
            rows.append({"T": T, "variant": f"g1={g1},g2={g2}", "error": f"{type(e).__name__}: {e}"[:200]})
            return None
        rows.append({"T": T, "variant": f"g1={g1},g2={g2}", "ms": ms, "floor_ms": floor_ms, "hbm_share": floor_ms / ms})
        return ms

    g1_best, g1_ms, misses = -1, float("inf"), 0
    for g1 in range(max_tactics):
        ms = try_pair(g1, -1)
        if ms is None:
            misses += 1
            if misses >= 3 and g1_best >= 0:
                break
            continue
        if ms < g1_ms:
            g1_best, g1_ms = g1, ms
    misses = 0
    for g2 in range(max_tactics):
        ms = try_pair(g1_best, g2)
        if ms is None:
            misses += 1
            if misses >= 3:
                break
            continue
        if ms < best["ms"]:
            best = {"pair": [g1_best, g2], "ms": ms}
    return {"T": T, "autotuned_ms": base_ms, "best_ms": best["ms"], "best_pair": best["pair"], "floor_ms": floor_ms,
            "saving_ms_per_layer": base_ms - best["ms"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="32,64,128,192,256,384,512")
    ap.add_argument("--max-tactics", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    assert torch.cuda.is_available(), "needs one RTX PRO 6000"
    rows: list = []
    t0 = time.time()
    bw = hbm_gbs()
    rows.append({"variant": "hbm_copy", "gb_s": bw, "device": torch.cuda.get_device_name(0),
                 "profile_ids": supports_profile_ids()})
    summary = [sweep_T(int(T), bw, a.max_tactics, rows) for T in a.tokens.split(",") if T]
    # per cycle: the verify forward sees T = running x draft tokens; the draft steps see T = running; only the verify
    # shapes are swept here, so the cycle saving is 42 layers x the verify-T saving (draft MTP layer: one layer)
    for s in summary:
        s["cycle_saving_ms_if_verify_T"] = round(MOE_LAYERS * (s.get("saving_ms_per_layer") or 0.0), 3)
    rep = {"shape": {"H": H, "inter": INTER, "E_local": E_LOCAL, "E_global": E_GLOBAL, "topk": TOPK, "ep": EP},
           "seconds": round(time.time() - t0), "summary": summary, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
