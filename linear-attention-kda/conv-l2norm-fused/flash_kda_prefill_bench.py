"""GLM-5.3-Flash KDA prefill microbench at the exact served shapes (amdflash, Hopper kernel program 2026-10-05).

Runs INSIDE the production v0.5.20 image on one H200 (launcher: operations/modal/flash_kda_prefill_bench.py). Each KDA
layer on the PD prefill role (DP4 attention: all 64 heads per rank, head dim 128, 34 KDA layers) runs this chain per
extend (layers/attention/linear/kda_backend.py:843-930, kernels/kda_triton.py:220-255, fla/kda.py:1083-1255):

    causal_conv1d_fn(mixed_qkv.T)  ->  split q/k/v (strided)  ->  l2norm(q.contiguous()), l2norm(k.contiguous()),
    v.contiguous()  ->  kda_gate_chunk_cumsum  ->  chunk_kda_fwd_intra (intra_sub_chunk, inter_solve_fused,
    recompute_w_u)  ->  chunk_gated_delta_rule_fwd_h  ->  chunk_gla_fwd_o_gk

F4 prefill profile (TP0, 4,096 tokens per rank per step, ms per step over 34 layers; flash_F4 attribution): conv1d
5.46, direct_copy 8.32 (3 per KDA layer), l2norm 2.19, gate cumsum 2.34, intra 5.27, inter_solve 5.24, recompute_w_u
4.57, delta_h 5.41, gla_o 6.28.

Measured here per case: CUDA-event time of each stage of the stock chain (ms per layer and x34 per step), and the
variants under test, each with its accuracy against the stock chain:
  V1  chunk_kda with FUSE_DIAGONAL + FUSE_RECOMPUTE forced on (stock gates them on B*NT*H <= 256, kda.py:1141)
  V2  fused conv1d + SiLU + per-head l2norm + token-major q/k/v store (flash_kda_conv_l2norm.py), replacing conv1d,
      the three .contiguous() copies and the two l2norm kernels

    python3 flash_kda_prefill_bench.py --json out.json [--cases c1,c2] [--iters 20]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback

import torch

H, D, LAYERS = 64, 128, 34
QDIM = KDIM = VDIM = H * D
CONV_W = 4
LOWER_BOUND = -5.0          # linear_attn_config.gate_lower_bound

CASES = {   # name -> (sequence lengths of this rank's extend, every sequence continues a cached prefix?)
    "c1_4096x1": ([4096], True),
    "c2_4096x2": ([2900, 1196], True),
    "c3_8192x1": ([8192], True),
    "c4_8192x4": ([2048, 2048, 2048, 2048], True),
    "c5_4096x1_cold": ([4096], False),
}


def make_inputs(lens: list[int], warm: bool, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    n = sum(lens)
    dim = QDIM + KDIM + VDIM
    t = {}
    t["mixed_qkv"] = (torch.randn(n, dim, device=device, generator=g) * 0.5).to(torch.bfloat16)
    t["conv_w"] = (torch.randn(dim, CONV_W, device=device, generator=g) * 0.3).to(torch.bfloat16)
    t["conv_b"] = None
    nseq = len(lens)
    slots = 8
    t["conv_states"] = (torch.randn(slots, CONV_W - 1, dim, device=device, generator=g) * 0.5).to(torch.bfloat16)
    t["ssm_states"] = (torch.randn(slots, H, D, D, device=device, generator=g) * 0.05).to(torch.float32)
    t["cache_indices"] = torch.arange(1, nseq + 1, device=device, dtype=torch.int32)
    t["query_start_loc"] = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), device=device,
                                        dtype=torch.int32)
    t["has_initial_state"] = torch.full((nseq,), bool(warm), device=device, dtype=torch.bool)
    t["seq_lens_cpu"] = list(lens)
    t["a"] = (torch.randn(1, n, H * D, device=device, generator=g)).to(torch.bfloat16)        # raw gate per channel
    t["b"] = (torch.randn(1, n, H, device=device, generator=g)).to(torch.bfloat16)            # raw beta
    t["A_log"] = (torch.rand(H, device=device, generator=g) * 2.0).to(torch.float32)
    t["dt_bias"] = (torch.randn(H * D, device=device, generator=g) * 0.5).to(torch.float32)
    return t


def stock_front(t, conv_states):
    """conv1d -> split -> unflatten (kda_backend.py:843-857). Returns strided q, k, v views."""
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    qkv = causal_conv1d_fn(t["mixed_qkv"].transpose(0, 1), t["conv_w"], t["conv_b"], activation="silu",
                           conv_states=conv_states.transpose(-1, -2), has_initial_state=t["has_initial_state"],
                           cache_indices=t["cache_indices"], query_start_loc=t["query_start_loc"],
                           seq_lens_cpu=t["seq_lens_cpu"]).transpose(0, 1)
    q, k, v = qkv.split([QDIM, KDIM, VDIM], dim=-1)
    return (q.unflatten(-1, (-1, D)).unsqueeze(0), k.unflatten(-1, (-1, D)).unsqueeze(0),
            v.unflatten(-1, (-1, D)).unsqueeze(0))


def stock_core(q, k, v, t, ssm_states, fuse: bool | None = None):
    """chunk_kda as kda_triton.extend calls it (l2norm in kernel, raw beta, A_log gate)."""
    import sglang.kernels.ops.attention.fla.kda as kda
    if fuse is None:
        return kda.chunk_kda(q=q, k=k, v=v, g=t["a"].unflatten(-1, (-1, D)), beta=t["b"], initial_state=ssm_states,
                             initial_state_indices=t["cache_indices"], use_qk_l2norm_in_kernel=True,
                             cu_seqlens=t["query_start_loc"].long(), A_log=t["A_log"], dt_bias=t["dt_bias"],
                             lower_bound=LOWER_BOUND, beta_is_raw=True)
    # V1: the same chain with FUSE_DIAGONAL / FUSE_RECOMPUTE forced (patch the grid heuristic for this call)
    orig = kda.chunk_kda_fwd_intra

    def forced(*a, **kw):
        kw["fuse_diagonal"] = fuse
        kw["fuse_recompute"] = fuse
        return orig(*a, **kw)
    kda.chunk_kda_fwd_intra = forced
    try:
        return kda.chunk_kda(q=q, k=k, v=v, g=t["a"].unflatten(-1, (-1, D)), beta=t["b"], initial_state=ssm_states,
                             initial_state_indices=t["cache_indices"], use_qk_l2norm_in_kernel=True,
                             cu_seqlens=t["query_start_loc"].long(), A_log=t["A_log"], dt_bias=t["dt_bias"],
                             lower_bound=LOWER_BOUND, beta_is_raw=True)
    finally:
        kda.chunk_kda_fwd_intra = orig


def timed(fn, iters: int, warmup: int = 3) -> tuple[float, object]:
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


def kernel_table(fn, iters: int = 5) -> dict:
    """Per-kernel GPU time (ms per call of fn) from torch.profiler."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    agg = {}
    for ev in prof.key_averages():
        if ev.device_type is not None and str(ev.device_type).endswith("CUDA") and ev.self_device_time_total > 0:
            agg[ev.key[:90]] = round(ev.self_device_time_total / 1000.0 / iters, 4)
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-12))


def run_case(name: str, lens: list[int], warm: bool, iters: int) -> dict:
    t = make_inputs(lens, warm)
    res = {"case": name, "lens": lens, "warm": warm, "tokens": sum(lens)}
    cs0, ss0 = t["conv_states"].clone(), t["ssm_states"].clone()

    def front():
        return stock_front(t, cs0.clone())

    def copies(q, k, v):
        return q.contiguous(), k.contiguous(), v.contiguous()
    ms_front, (q, k, v) = timed(front, iters)
    ms_copies, _ = timed(lambda: copies(q, k, v), iters)
    ms_core, o_ref = timed(lambda: stock_core(q, k, v, t, ss0.clone()), iters)
    res["stock_ms_per_layer"] = {"conv1d": round(ms_front, 4), "contiguous_copies": round(ms_copies, 4),
                                 "chunk_kda_incl_l2norm_and_copies": round(ms_core, 4)}
    res["stock_ms_per_step_x34"] = {k_: round(v_ * LAYERS, 2) for k_, v_ in res["stock_ms_per_layer"].items()}
    res["stock_kernels_per_layer_ms"] = kernel_table(lambda: stock_core(q, k, v, t, ss0.clone()))
    res["front_kernels_per_layer_ms"] = kernel_table(front)
    # V1: forced fusion
    try:
        ms_v1, o_v1 = timed(lambda: stock_core(q, k, v, t, ss0.clone(), fuse=True), iters)
        res["V1_fused_intra"] = {"ms_per_layer": round(ms_v1, 4), "vs_stock": round(ms_v1 / ms_core, 3),
                                 "rel_err_o": rel_err(o_v1, o_ref), "max_abs_o": float((o_v1.float() - o_ref.float()).abs().max()),
                                 "kernels": kernel_table(lambda: stock_core(q, k, v, t, ss0.clone(), fuse=True))}
    except Exception as e:  # noqa: BLE001
        res["V1_fused_intra"] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "tb": traceback.format_exc()[-1500:]}
    # V2: fused conv + l2norm + token-major store
    try:
        import flash_kda_conv_l2norm as fk
        cs_ref, cs_v2 = cs0.clone(), cs0.clone()
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
        qs, ks, vs = stock_front(t, cs_ref)
        q_ref, k_ref = l2norm_fwd(qs.contiguous()), l2norm_fwd(ks.contiguous())   # the stock norm kernel itself

        def v2():
            return fk.conv_l2norm_qkv(t["mixed_qkv"], t["conv_w"], t["conv_b"], cs_v2_run[0], t["cache_indices"],
                                      t["has_initial_state"], t["query_start_loc"], H, D,
                                      seq_lens=t["seq_lens_cpu"])
        cs_v2_run = [cs_v2]
        q2, k2, v2_ = v2()
        acc = {"rel_err_q": rel_err(q2, q_ref), "rel_err_k": rel_err(k2, k_ref), "rel_err_v": rel_err(v2_, vs),
               "conv_state_max_abs": float((cs_v2.float() - cs_ref.float()).abs().max())}
        cs_v2_run = [cs0.clone()]
        ms_v2, _ = timed(v2, iters)
        # config sweep (block rows x warps) on the same inputs; accuracy is config-independent math
        sweep = {}
        for bt in (16, 32, 64):
            for nw in (2, 4, 8):
                try:
                    ms, _ = timed(lambda bt=bt, nw=nw: fk.conv_l2norm_qkv(
                        t["mixed_qkv"], t["conv_w"], t["conv_b"], cs_v2_run[0], t["cache_indices"],
                        t["has_initial_state"], t["query_start_loc"], H, D, seq_lens=t["seq_lens_cpu"], bt=bt,
                        num_warps=nw), iters)
                    sweep[f"bt{bt}_w{nw}"] = round(ms, 4)
                except Exception as e:  # noqa: BLE001
                    sweep[f"bt{bt}_w{nw}"] = f"{type(e).__name__}"
        for seg in (32, 64, 128, 256):
            for nw in (1, 2, 4):
                try:
                    ms, _ = timed(lambda seg=seg, nw=nw: fk.conv_l2norm_qkv(
                        t["mixed_qkv"], t["conv_w"], t["conv_b"], cs_v2_run[0], t["cache_indices"],
                        t["has_initial_state"], t["query_start_loc"], H, D, seq_lens=t["seq_lens_cpu"],
                        num_warps=nw, rowloop_seg=seg), iters)
                    sweep[f"row{seg}_w{nw}"] = round(ms, 4)
                except Exception as e:  # noqa: BLE001
                    sweep[f"row{seg}_w{nw}"] = f"{type(e).__name__}: {str(e)[:120]}"
        # row-loop accuracy vs the stock chain (fresh state copy)
        try:
            cs_r = cs0.clone()
            qr, kr, vr = fk.conv_l2norm_qkv(t["mixed_qkv"], t["conv_w"], t["conv_b"], cs_r, t["cache_indices"],
                                            t["has_initial_state"], t["query_start_loc"], H, D,
                                            seq_lens=t["seq_lens_cpu"], rowloop_seg=64, num_warps=2)
            acc["rowloop_rel_err_q"] = rel_err(qr, q_ref)
            acc["rowloop_rel_err_v"] = rel_err(vr, vs)
            acc["rowloop_conv_state_max_abs"] = float((cs_r.float() - cs_ref.float()).abs().max())
        except Exception as e:  # noqa: BLE001
            acc["rowloop_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        best = min((v_, k_) for k_, v_ in sweep.items() if isinstance(v_, float))
        # the stock work V2 replaces: conv1d + 3 copies + 2 l2norm kernels
        l2 = sum(v_ for k_, v_ in res["stock_kernels_per_layer_ms"].items() if "l2norm" in k_)
        base = ms_front + ms_copies + l2
        res["V2_conv_l2norm"] = {"ms_per_layer": round(ms_v2, 4), "replaces_ms_per_layer": round(base, 4),
                                 "saving_ms_per_step_x34": round((base - ms_v2) * LAYERS, 2), **acc,
                                 "sweep_ms_per_layer": sweep, "best": {"cfg": best[1], "ms": best[0],
                                 "saving_ms_per_step_x34": round((base - best[0]) * LAYERS, 2)},
                                 "kernels": kernel_table(v2)}
    except Exception as e:  # noqa: BLE001
        res["V2_conv_l2norm"] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "tb": traceback.format_exc()[-1500:]}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args(argv)
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cases": []}
    try:
        import sglang
        out["sglang"] = getattr(sglang, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        out["sglang"] = f"import failed: {e}"
    for name in a.cases.split(","):
        lens, warm = CASES[name]
        t0 = time.time()
        try:
            r = run_case(name, lens, warm, a.iters)
        except Exception as e:  # noqa: BLE001
            r = {"case": name, "error": f"{type(e).__name__}: {str(e)[:400]}", "tb": traceback.format_exc()[-3000:]}
        r["wall_s"] = round(time.time() - t0, 1)
        out["cases"].append(r)
        print(json.dumps({k: r.get(k) for k in ("case", "stock_ms_per_step_x34", "error")}), flush=True)
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
