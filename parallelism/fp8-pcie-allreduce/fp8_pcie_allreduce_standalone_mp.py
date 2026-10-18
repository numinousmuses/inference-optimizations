"""torchrun check of StandaloneComm (the sglang pcie_ipc-slot adapter of the quantized all-reduce), agent rtxv41.

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 operations/kernels/fp8_pcie_allreduce_standalone_mp.py

Checks: lazy build on the first eligible tensor, supports() rules, ranks bitwise identical, relative error vs the NCCL
bf16 all-reduce, and CUDA-graph capture/replay of the adapter call (decode runs inside graphs).
"""
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8_pcie_allreduce as qar  # noqa: E402


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    comm = qar.StandaloneComm(dist.group.WORLD, dev, "q8")
    res = {}
    # outside a server _decode_width() has no decode config and returns its fallback width; probe it first
    probe = torch.zeros(8, 5120, device=dev, dtype=torch.bfloat16)
    assert comm.should_pcie_ipc_ar(probe)
    rows = min(384, comm.max_numel // 5120)
    res["max_rows"] = rows
    x32 = torch.randn(rows, 5120, device=dev, dtype=torch.float32, generator=torch.Generator(device=dev).manual_seed(rank))
    assert not comm.should_pcie_ipc_ar(x32)
    x = x32.bfloat16()
    assert comm.should_pcie_ipc_ar(x)
    out = comm.pcie_ipc_all_reduce(x)
    ref = x.clone()
    dist.all_reduce(ref)
    outs = [torch.empty_like(out) for _ in range(world)]
    dist.all_gather(outs, out)
    res["ranks_identical"] = all(torch.equal(o, outs[0]) for o in outs)
    res["rel_vs_nccl"] = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = comm.pcie_ipc_all_reduce(x)
    g.replay()
    torch.cuda.synchronize()
    res["graph_replay_equal"] = bool(torch.equal(y, out))
    res["too_big_rejected"] = not comm.should_pcie_ipc_ar(torch.zeros(comm.max_numel // 5120 + 8, 5120, device=dev,
                                                                       dtype=torch.bfloat16))
    if rank == 0:
        ok = res["ranks_identical"] and res["rel_vs_nccl"] < 0.015 and res["graph_replay_equal"] and res["too_big_rejected"]
        print(("PASS " if ok else "FAIL ") + json.dumps(res), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
