"""Pro MoE layer, path A vs path B vs stock, at identical shapes and routing on 8 GPUs (torchrun).

Per MoE layer, from this GPU's bf16 hidden rows + its top-6 routing to this GPU's routed MoE output rows (scaled by
routed_scaling_factor 2.5), including every path's own dispatch and combine:

  A_fp8   MegaMoE fp8xfp4 (EP8, DeepGEMM 0.2.0): SGLang's JIT pre-dispatch + fused dispatch/GEMMs/combine + scale
  A_w4a4  MegaMoE mxf4xmxf4 (W4A4): DeepGEMM pre-dispatch + the same kernel
  B_tp8   numinous_dp_moe (operations/kernels): local MXFP8 quantize, push rows + top-k to every peer, barrier,
          FlashInfer TRT-LLM MXFP4 routed experts on the gathered rows (each GPU 1/8 of every expert), barrier,
          pull-reduce own rows with the scale. Decode = MAX_LEN block (the CUDA-graph layout), prefill = SUM_LEN.
  B_ep8   the same with EP8 experts (48 whole experts per GPU)
  C_ar    SGLang stock TP8 path: gather bf16 rows + top-k (NCCL all-gather at MAX_LEN, all-reduce over a zero-filled
          buffer at SUM_LEN, by SGLang's DP padding rule), MXFP8 quantize the gathered rows, the same TP8 experts,
          NCCL all-reduce of the full output, own rows x scale
  C_rs    as C_ar with reduce-scatter instead of all-reduce (SGLANG_DP_USE_REDUCE_SCATTER)

Not included in any path (identical or path-independent): router GEMM + top-k and the shared expert. Stock C runs
both on the gathered (8x) rows in the engine, so C here is a lower bound for stock (kernel map: router 15 us +
6 us per layer at 768 gathered rows on B200).

Timing: decode shapes by CUDA-graph replay (graph of --graph-steps calls, --replays replays, events), prefill and
mixed shapes eagerly with events (--iters calls after warm-up), all ranks released together after a barrier. Each
rank's time includes waiting for the others; the table reports the slowest rank. Checks: B_tp8 vs C_ar output (same
weights and rows) and A_fp8 vs the dense reference of megamoe_bench at decode 96.

    torchrun --nproc-per-node 8 operations/kernels/megamoe_path_compare.py --out /root/megamoe/compare_b200.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import flashinfer  # noqa: F401  (engine import order)
import torch
import torch.distributed as dist

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import megamoe_bench as mb  # noqa: E402

E, H, INTER, TOPK, SCALE = 384, 7168, 3072, 6, 2.5
PATHS = ("A_fp8", "A_w4a4", "B_tp8", "B_ep8", "C_ar", "C_rs")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--decode", default="16,32,96,128")
    p.add_argument("--prefill", default="2048,4096")
    p.add_argument("--mixed", default="4096,96,96,96,96,96,96,96")
    p.add_argument("--paths", default=",".join(PATHS))
    p.add_argument("--graph-steps", type=int, default=10)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--mixed-timing", default="eager", choices=("eager", "graph"),
                   help="mixed steps run eagerly in SGLang; 'graph' times them by replay to resolve small differences")
    p.add_argument("--tactics", action="store_true",
                   help="serve our measured FlashInfer MoE tactics (numinous_moe_tactics, served scope: CUDA-graph "
                        "calls only, i.e. decode) to paths B and C; default is FlashInfer's autotune-off fallback")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------------------ FlashInfer experts
def flashinfer_experts(n_loc: int, inter: int, off_loc: int, dev, seed: int):
    """fused_path_bench's expert call (numinous_dp_moe._experts_prequant, verbatim) with random MXFP4 weights."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from sglang.srt.layers.moe.utils import RoutingMethodType
    g = torch.Generator(device=dev).manual_seed(seed)
    w13 = torch.randint(0, 256, (n_loc, 2 * inter, H // 2), dtype=torch.uint8, device=dev, generator=g)
    w13_s = torch.randint(120, 126, (n_loc, 2 * inter, H // 32), dtype=torch.uint8, device=dev,
                          generator=g).view(torch.float8_e4m3fn)
    w2 = torch.randint(0, 256, (n_loc, H, inter // 2), dtype=torch.uint8, device=dev, generator=g)
    w2_s = torch.randint(120, 126, (n_loc, H, inter // 32), dtype=torch.uint8, device=dev,
                         generator=g).view(torch.float8_e4m3fn)
    ones = torch.ones(n_loc, dtype=torch.float32, device=dev)
    clamp = torch.full((n_loc,), 10.0, dtype=torch.float32, device=dev)

    def run(xq, xs_u8, ids, wts, out):
        rows = xq.shape[0]
        trtllm_fp4_block_scale_routed_moe(
            topk_ids=(ids.contiguous(), wts.contiguous()), routing_bias=None, hidden_states=xq,
            hidden_states_scale=xs_u8.view(torch.float8_e4m3fn).reshape(rows, -1),
            gemm1_weights=w13, gemm1_weights_scale=w13_s, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
            gemm1_clamp_limit=clamp, gemm2_weights=w2, gemm2_weights_scale=w2_s, gemm2_bias=None,
            output1_scale_scalar=ones, output1_scale_gate_scalar=ones, output2_scale_scalar=ones, num_experts=E,
            top_k=TOPK, n_group=1, topk_group=1, intermediate_size=inter, local_expert_offset=off_loc,
            local_num_experts=n_loc, routed_scaling_factor=1.0, routing_method_type=int(RoutingMethodType.TopK),
            do_finalize=True, tune_max_num_tokens=1 << max(0, rows - 1).bit_length(), output=out, enable_pdl=True)
    return run


def mxfp8(x):
    from flashinfer import mxfp8_quantize
    n = x.shape[0]
    if n == 0:
        return (torch.empty(0, H, dtype=torch.float8_e4m3fn, device=x.device),
                torch.empty(0, H // 32, dtype=torch.uint8, device=x.device))
    q, s = mxfp8_quantize(x, False, alignment=H)
    return q, s.view(torch.uint8).reshape(n, -1)


# ------------------------------------------------------------------------------------------------ the paths
class Paths:
    def __init__(self, rank, world, group, want):
        import numinous_dp_moe as ndm
        self.ndm, self.rank, self.world, self.group = ndm, rank, world, group
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.want = want
        self.mega = {}
        if any(p.startswith("A_") for p in want):
            weights = mb.make_weights(E // world, seed=rank)
            for p, mma in (("A_fp8", "fp8xfp4"), ("A_w4a4", "mxf4xmxf4")):
                if p in want:
                    self.mega[p] = mb.MegaCase(__import__("deep_gemm"), group, rank, world, mma, weights, 4096, 2)
            self.mega_weights = weights
        self.exp = {}
        if "B_tp8" in want or "C_ar" in want or "C_rs" in want:
            self.exp["tp8"] = flashinfer_experts(E, INTER // world, 0, self.dev, 100 + rank)
        if "B_ep8" in want:
            self.exp["ep8"] = flashinfer_experts(E // world, INTER, rank * (E // world), self.dev, 200 + rank)
        self.buf = ndm.buffers(group, 40960, H, TOPK, self.dev) if any(p.startswith("B_") for p in want) else None
        self.valids = {}

    # A: MegaMoE, rows padded to `rows_run` with no-route rows when balancing (numinous_megamoe_balance)
    def a_step(self, path, x, ids, wts, y, rows_run=None):
        case = self.mega[path]
        n = x.shape[0]
        case.pre(x, ids, wts)
        if rows_run is not None and rows_run > n:
            case.buf.topk_idx[n:rows_run].fill_(-1)
        case.moe(y)
        y[:n].mul_(SCALE)

    # B: numinous_dp_moe, MAX_LEN block (decode graphs) or SUM_LEN (counts)
    def b_step(self, layout, x, ids, wts, out, counts, block=None):
        ndm, buf, r = self.ndm, self.buf, self.rank
        n = x.shape[0]
        xq, xs = mxfp8(x)
        if block:
            if (block, n) not in self.valids:
                self.valids[(block, n)] = torch.tensor([n], dtype=torch.int64, device=self.dev)
            valid, row_off, rows = self.valids[(block, n)], r * block, self.world * block
        else:
            valid, row_off, rows = None, sum(counts[:r]), sum(counts)
        ndm.phase_push(ndm.LocalPrep(xq, xs, ids, wts, None), buf, row_off, multicast=valid is None)
        buf.barrier()
        ndm.phase_experts(buf, rows, self.exp[layout])
        buf.barrier()
        if n:
            ndm.phase_pull(buf, row_off, n, out, SCALE, None, valid)

    # C: SGLang stock TP8 semantics
    def c_step(self, reduce, x, ids, wts, out, counts, scratch):
        r, world = self.rank, self.world
        n, total, mx = x.shape[0], sum(counts), max(counts)
        max_len = total * 2 >= mx * world          # SGLang DpPaddingMode: MAX_LEN unless the step is very skewed
        if max_len:
            gx, gi, gw = scratch["gx"][:mx * world], scratch["gi"][:mx * world], scratch["gw"][:mx * world]
            if n < mx:  # padded rows route to expert 0 at weight 0 (as a fresh MAX_LEN buffer would)
                pad = mx - n
                x = torch.cat([x, x.new_zeros(pad, H)])
                ids = torch.cat([ids, ids.new_zeros(pad, TOPK)])
                wts = torch.cat([wts, wts.new_zeros(pad, TOPK)])
            dist.all_gather_into_tensor(gx, x.contiguous(), group=self.group)
            dist.all_gather_into_tensor(gi, ids.contiguous(), group=self.group)
            dist.all_gather_into_tensor(gw, wts.contiguous(), group=self.group)
            off, rows = r * mx, mx * world
        else:
            gx, gi, gw = scratch["gx"][:total], scratch["gi"][:total], scratch["gw"][:total]
            off, rows = sum(counts[:r]), total
            gx.zero_(), gi.zero_(), gw.zero_()
            gx[off:off + n].copy_(x), gi[off:off + n].copy_(ids), gw[off:off + n].copy_(wts)
            for t in (gx, gi, gw):
                dist.all_reduce(t, group=self.group)
        part = scratch["part"][:rows]
        xq, xs = mxfp8(gx)
        self.exp["tp8"](xq, xs, gi, gw, part)
        if reduce == "ar":
            dist.all_reduce(part, group=self.group)
            out[:n].copy_(part[off:off + n] * SCALE)
        else:
            if max_len:
                own = scratch["own"][:mx]
                dist.reduce_scatter_tensor(own, part, group=self.group)
                out[:n].copy_(own[:n] * SCALE)
            else:  # uneven: reduce-scatter-v is not in NCCL; SGLang all-reduces then slices here
                dist.all_reduce(part, group=self.group)
                out[:n].copy_(part[off:off + n] * SCALE)


# ------------------------------------------------------------------------------------------------ timing
def graph_time(fn, steps, replays, group):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier(group)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(steps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    dist.barrier(group)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(replays):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    del g
    return t0.elapsed_time(t1) * 1e3 / (steps * replays)


def eager_time(fn, iters, group):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def inputs(n, rank, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed * 1000 + rank)
    x = torch.randn(n, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
    scores = torch.randn(n, E, device=dev, generator=g)
    w, idx = torch.topk(scores, TOPK, dim=-1, sorted=False)   # uniform expert choice, as megamoe_bench
    return x, idx.to(torch.int32), torch.softmax(w, -1).float()


def main(args):
    rank, world, group = mb.init()
    dev = torch.device("cuda", torch.cuda.current_device())
    want = args.paths.split(",")
    tag = ""
    if args.tactics:
        import numinous_moe_tactics
        os.environ.setdefault("NUMINOUS_MOE_TACTICS_SCOPE", "graph")
        numinous_moe_tactics.install()
        tag = "+tactics"
    P = Paths(rank, world, group, want)
    max_rows = 40960
    scratch = dict(gx=torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev),
                   gi=torch.empty(max_rows, TOPK, dtype=torch.int32, device=dev),
                   gw=torch.empty(max_rows, TOPK, dtype=torch.float32, device=dev),
                   part=torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev),
                   own=torch.empty(4096, H, dtype=torch.bfloat16, device=dev))
    shapes = [("decode", [int(t)] * world) for t in args.decode.split(",") if t] + \
             [("prefill", [int(t)] * world) for t in args.prefill.split(",") if t]
    if args.mixed:
        shapes.append(("mixed", [int(t) for t in args.mixed.split(",")]))
    rows_out = []
    gpu = torch.cuda.get_device_name()
    for si, (kind, counts) in enumerate(shapes):
        n = counts[rank]
        x, ids, wts = inputs(n, rank, 17 + si, dev)
        label = f"{kind} {counts[0]}" if kind != "mixed" else "mixed " + "/".join(map(str, counts)) + (
            " (graph-timed)" if args.mixed_timing == "graph" else "")
        variants = []
        for p in want:
            if p.startswith("A_"):
                variants.append((p, None))
                if kind == "mixed":
                    variants.append((p + "+balance", -(-sum(counts) // world)))
            else:
                variants.append((p, None))
        outs = {}
        for name, bal in variants:
            base = name.split("+")[0]
            rows_run = max(n, bal) if bal else n
            y = torch.zeros(max(rows_run, 1), H, dtype=torch.bfloat16, device=dev)
            if base.startswith("A_"):
                fn = lambda base=base, y=y, rows_run=rows_run: P.a_step(base, x, ids, wts, y, rows_run)
            elif base.startswith("B_"):
                lay = base[2:]
                blk = counts[0] if kind == "decode" else None
                fn = lambda lay=lay, y=y, blk=blk: P.b_step(lay, x, ids, wts, y, counts, blk)
            else:
                red = base[2:]
                fn = lambda red=red, y=y: P.c_step(red, x, ids, wts, y, counts, scratch)
            try:
                if kind == "decode" or (kind == "mixed" and args.mixed_timing == "graph"):
                    us = graph_time(fn, args.graph_steps, args.replays, group)
                else:
                    us = eager_time(fn, args.iters, group)
                err = ""
            except Exception as exc:  # noqa: BLE001
                us, err = float("nan"), f"{type(exc).__name__}: {str(exc)[:200]}"
                torch.cuda.synchronize()
                dist.barrier(group)
            outs[name] = y[:n].clone()
            t = torch.tensor([us], dtype=torch.float64, device=dev)
            ts = [torch.empty_like(t) for _ in range(world)]
            dist.all_gather(ts, t)
            vals = [float(v) for v in torch.cat(ts).cpu()]
            if rank == 0:
                row = dict(gpu=gpu, shape=label, kind=kind, path=name + (tag if name[0] in "BC" else ""), us_max=round(max(vals), 1),
                           us_median=round(sorted(vals)[world // 2], 1), error=err)
                rows_out.append(row)
                print(json.dumps(row), flush=True)
        # checks (every rank computes, rank 0 prints the max)
        chk = {}
        if not args.no_check and "B_tp8" in outs and "C_ar" in outs and n:
            chk["B_tp8_vs_C_ar_rel"] = float((outs["B_tp8"].float() - outs["C_ar"].float()).norm()
                                             / outs["C_ar"].float().norm().clamp_min(1e-6))
        if not args.no_check and kind == "decode" and counts[0] == 96 and "A_fp8" in outs:
            ref = mb.reference(x, wts, ids.long(), *P.mega_weights, rank, world, "fp8")
            chk["A_fp8_vs_reference_diff"] = mb.calc_diff(outs["A_fp8"].float() / SCALE, ref)
        for k in sorted(chk):
            t = torch.tensor([chk[k]], dtype=torch.float64, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            if rank == 0:
                row = dict(gpu=gpu, shape=label, kind=kind, check=k, value=float(t))
                rows_out.append(row)
                print(json.dumps(row), flush=True)
    if rank == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), rows=rows_out, time=time.time()), f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
