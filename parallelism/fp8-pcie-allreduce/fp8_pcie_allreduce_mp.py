"""Multi-GPU gate for the quantized PCIe all-reduce (agent rtxv41). NOT runnable on one GPU; for the 4/8 x RTX box.

    torchrun --nproc-per-node 8 operations/kernels/fp8_pcie_allreduce_mp.py --out /root/quantar_mp.json

Per mode (q8, q8_rs, fp8, fp8_rs) and rows in {1, 16, 64, 192, 384} x 5120 (decode reductions of V4.1 at 64 users x
6 DSpark rows = 384):
  correctness  every rank's output bitwise identical (all_gather) and bitwise equal to reference_all_reduce of the
               gathered inputs (computed on rank 0);
  timing       CUDA-graph-replayed time of Fp8PcieAllReduce.all_reduce vs NCCL all_reduce vs the FlashInfer
               PCIe-IPC workspace (flashinfer >= 0.7.0), same tensor.
Preregistered gate (research/rtx-flash/v41.md section 5.7): q8 at 384 rows <= 0.65 x the FlashInfer PCIe-IPC bf16 time
and at 1 row <= 1.5 x it (latency floor: 5 launches incl. 2 barriers); correctness exact. Accuracy is a separate
served gate (GSM8K + logit KL vs the bf16 all-reduce).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8_pcie_allreduce as qar  # noqa: E402


def timed(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/quantar_mp.json")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--rows", default="1,16,64,192,384", help="comma list; prefill-chunk sizes e.g. 2048,8192")
    ap.add_argument("--modes", default=",".join(qar.MODES))
    ap.add_argument("--no-flashinfer", action="store_true")
    ap.add_argument("--pull", default="0", help="0 push, 1 pull (P2P reads), both: 0,1")
    a = ap.parse_args()
    rows_list = [int(r) for r in a.rows.split(",")]
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    group = dist.group.WORLD
    max_rows = max(rows_list)
    results = []
    fi_ws = None
    try:
        if a.no_flashinfer:
            raise RuntimeError("disabled by --no-flashinfer")
        from flashinfer.comm import PcieIpcAllReduceWorkspace

        fi_ws = PcieIpcAllReduceWorkspace(group=group, max_numel=max_rows * a.hidden, dtype=torch.bfloat16)
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print("FlashInfer PCIe-IPC unavailable:", e)
    for mode, pull in [(m, p) for m in a.modes.split(",") for p in a.pull.split(",")]:
      comm = qar.Fp8PcieAllReduce(group, dev, max_rows * a.hidden, mode=mode, pull=pull == "1")
      for rows in rows_list:
            g = torch.Generator().manual_seed(1000 * rank + rows)
            x = torch.randn(rows, a.hidden, generator=g).bfloat16().to(dev)
            out = comm.all_reduce(x)
            torch.cuda.synchronize()
            outs = [torch.empty_like(out) for _ in range(world)]
            dist.all_gather(outs, out)
            xs = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(xs, x)
            same = all(torch.equal(o, outs[0]) for o in outs)
            bitwise = None
            if rank == 0 and rows <= 1024:
                bitwise = bool(torch.equal(outs[0].cpu(), qar.reference_all_reduce([t.cpu() for t in xs], mode)))
            t_q = timed(lambda: comm.all_reduce(x))
            y = x.clone()
            t_nccl = timed(lambda: dist.all_reduce(y))
            t_fi = timed(lambda: fi_ws.all_reduce(x)) if fi_ws is not None else None
            if rank == 0:
                r = {"mode": mode, "pull": pull == "1", "rows": rows, "ranks_identical": same, "bitwise_vs_ref": bitwise,
                     "quant_us": t_q * 1e6, "nccl_us": t_nccl * 1e6, "flashinfer_ipc_us": t_fi and t_fi * 1e6}
                results.append(r)
                print(json.dumps(r))
    if rank == 0:
        json.dump(results, open(a.out, "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
