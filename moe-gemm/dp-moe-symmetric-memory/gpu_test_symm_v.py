"""8-GPU unit test + timing of the vectorized symmetric-memory kernels (torchrun --nproc-per-node 8, dev box).

Correctness: push_payloads (unicast, multicast) delivers every rank's rows of four real-shaped payloads (MXFP8 rows,
e8m0 scales, top-k ids, top-k weights) to every GPU from uneven per-rank counts with an idle rank; pull_reduce_v
matches pull_reduce_rows_ptrs bit for bit (unicast, same fp32 peer order) and within bf16 rounding (multicast).
Timing: CUDA-graph replay for the decode block (96 rows per rank), events for a 4,096-row prefill; old vs new.
Exit code 1 on any mismatch.
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

import flashinfer  # noqa: F401  (as in the engine: it points Triton at the toolkit ptxas)
import torch.distributed._symmetric_memory as symm_mem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symm_collectives import (pull_reduce_rows_ptrs, pull_reduce_v, push_payloads,  # noqa: E402
                              push_rows_ptrs)

H, TOPK = 7168, 6


def alloc(name, shape, dtype):
    t = symm_mem.empty(shape, dtype=dtype, device="cuda")
    t.zero_()
    return t, symm_mem.rendezvous(t, name)


def timed(fn, graph: bool, iters: int = 50) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    else:
        run = fn
    run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main() -> int:
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    name = dist.group.WORLD.group_name
    cap = 4096 * world
    xq, hq = alloc(name, (cap, H), torch.uint8)
    xs, hs = alloc(name, (cap, H // 32), torch.uint8)
    ids, hi = alloc(name, (cap, TOPK), torch.int32)
    wts, hw = alloc(name, (cap, TOPK), torch.float32)
    part, hp = alloc(name, (cap, H), torch.bfloat16)
    torch.cuda.synchronize()
    hq.barrier(channel=0)
    fails = []

    def payloads(src_q, src_s, src_i, src_w):
        return [(src_q, hq.buffer_ptrs_dev, hq.multicast_ptr, H), (src_s, hs.buffer_ptrs_dev, hs.multicast_ptr, H // 32),
                (src_i, hi.buffer_ptrs_dev, 0, TOPK * 4), (src_w, hw.buffer_ptrs_dev, 0, TOPK * 4)]

    # ---- correctness: uneven SUM_LEN layout, rank 1 idle
    counts = [((r * 37) % 211) + 5 for r in range(world)]
    counts[1] = 0
    offs = [sum(counts[:r]) for r in range(world)]
    g = torch.Generator(device="cuda").manual_seed(1234)

    def rows_of(r):
        gg = torch.Generator(device="cuda").manual_seed(1000 + r)
        n = counts[r]
        return (torch.randint(0, 256, (n, H), dtype=torch.uint8, device="cuda", generator=gg),
                torch.randint(0, 256, (n, H // 32), dtype=torch.uint8, device="cuda", generator=gg),
                torch.randint(0, 384, (n, TOPK), dtype=torch.int32, device="cuda", generator=gg),
                torch.rand((n, TOPK), dtype=torch.float32, device="cuda", generator=gg))

    for mc in (False, True):
        for t in (xq, xs, ids, wts):
            t.zero_()
        torch.cuda.synchronize()
        hq.barrier(channel=0)
        mine = rows_of(rank)
        push_payloads(payloads(*mine), world, counts[rank], offs[rank], multicast=mc)
        hq.barrier(channel=0)          # no host sync in between: production orders the push by the barrier alone
        for r in range(world):
            ref = rows_of(r)
            o, n = offs[r], counts[r]
            for got, want, nm in ((xq, ref[0], "xq"), (xs, ref[1], "xs"), (ids, ref[2], "ids"), (wts, ref[3], "wts")):
                if not torch.equal(got[o:o + n], want):
                    fails.append(f"push mc={mc}: rank {rank} holds wrong {nm} rows of rank {r}")
    # ---- pull: partials rank-dependent; own rows with scale, shared, valid
    part.copy_((torch.randn(cap, H, device="cuda", generator=g) * (rank + 1)).to(torch.bfloat16))
    torch.cuda.synchronize()
    hp.barrier(channel=0)
    n_mine, off_mine = 300, 300 * rank
    shared = torch.randn(n_mine, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)).to(
        torch.bfloat16)
    valid = torch.tensor([250], dtype=torch.int32, device="cuda")
    ref = torch.empty(n_mine, H, dtype=torch.bfloat16, device="cuda")
    pull_reduce_rows_ptrs(hp.buffer_ptrs_dev, world, off_mine, n_mine, ref, scale=2.5, shared=shared, valid=valid)
    out_u = torch.empty_like(ref)
    pull_reduce_v(hp.buffer_ptrs_dev, world, off_mine, n_mine, out_u, scale=2.5, shared=shared, valid=valid)
    if not torch.equal(out_u, ref):
        d = (out_u.float() - ref.float()).abs().max().item()
        fails.append(f"pull unicast != reference kernel (max abs diff {d})")
    out_m = torch.empty_like(ref)
    pull_reduce_v(hp.multicast_ptr, world, off_mine, n_mine, out_m, scale=2.5, shared=shared, valid=valid,
                  multicast=True)
    rel = ((out_m.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    if rel > 0.02:
        fails.append(f"pull multicast rel err {rel:.4f}")
    if (out_m[250:] != 0).any() or (out_u[250:] != 0).any():
        fails.append("rows past valid are not zero")

    # ---- timing
    res = {}
    for label, n, graph in (("decode_96", 96, True), ("prefill_4096", 4096, False)):
        src = rows_of(rank) if False else (torch.randint(0, 256, (n, H), dtype=torch.uint8, device="cuda"),
                                           torch.randint(0, 256, (n, H // 32), dtype=torch.uint8, device="cuda"),
                                           torch.randint(0, 384, (n, TOPK), dtype=torch.int32, device="cuda"),
                                           torch.rand((n, TOPK), device="cuda"))
        off = n * rank
        out = torch.empty(n, H, dtype=torch.bfloat16, device="cuda")
        sh = torch.randn(n, H, device="cuda").to(torch.bfloat16)

        def old_push():
            for s_, h_, w_ in ((src[0], hq, H), (src[1], hs, H // 32), (src[2], hi, TOPK * 4), (src[3], hw, TOPK * 4)):
                push_rows_ptrs(s_, h_.buffer_ptrs_dev, world, w_, off)

        def bar():
            hq.barrier(channel=0)

        # stores are fire-and-forget (a kernel can end before its remote writes land): every push is timed with the
        # barrier that follows it in production, and the barrier alone is reported
        r = {
            "barrier": timed(bar, graph),
            "push_old": timed(lambda: (old_push(), bar()), graph),
            "push_v": timed(lambda: (push_payloads(payloads(*src), world, n, off), bar()), graph),
            "push_mc": timed(lambda: (push_payloads(payloads(*src), world, n, off, multicast=True), bar()), graph),
            "pull_old": timed(lambda: (pull_reduce_rows_ptrs(hp.buffer_ptrs_dev, world, off, n, out, 2.5, sh), bar()),
                              graph),
            "pull_v": timed(lambda: (pull_reduce_v(hp.buffer_ptrs_dev, world, off, n, out, 2.5, sh), bar()), graph),
            "pull_mc": timed(lambda: (pull_reduce_v(hp.multicast_ptr, world, off, n, out, 2.5, sh, multicast=True),
                                      bar()), graph),
        }
        # bytes each GPU receives: push = the other ranks' rows; pull = own rows from the other ranks
        push_in = (world - 1) * n * (H + H // 32 + 2 * TOPK * 4)
        pull_in = (world - 1) * n * H * 2
        r["push_in_GBps_v"] = round(push_in / r["push_v"] / 1e3, 1)
        r["push_in_GBps_mc"] = round(push_in / r["push_mc"] / 1e3, 1)
        r["pull_in_GBps_v"] = round(pull_in / r["pull_v"] / 1e3, 1)
        res[label] = {k: round(v, 1) for k, v in r.items()}
    bad = torch.tensor([len(fails)], device="cuda")
    dist.all_reduce(bad)
    if rank == 0:
        print(json.dumps({"gpu": torch.cuda.get_device_name(0), "world": world, "failures_all_ranks": int(bad.item()),
                          "timing_us_rank0": res}, indent=1), flush=True)
    for f in fails[:5]:
        print(f"[rank {rank}] FAIL {f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 1 if int(bad.item()) else 0


if __name__ == "__main__":
    sys.exit(main())
