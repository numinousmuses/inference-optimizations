"""Tests for rocm_symm_collectives (portable symmetric-memory push / pull).

One GPU: "peers" are tensors on the same GPU whose addresses go in a device int64 array, as buffer_ptrs_dev does
across GPUs; push delivers every rank's rows to every peer at its offset, pull matches the reference kernel bit for
bit (the Triton interpreter cannot cast integer addresses to pointers, so these need a GPU). Two ROCm GPUs: runs
rocm_gpu_test_symm.py under torchrun and requires zero correctness failures (the report is printed). CPU: argument
validation only.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

GPU = torch.cuda.is_available()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rocm_symm_collectives as rsc  # noqa: E402
import symm_collectives as sc  # noqa: E402

needs_gpu = pytest.mark.skipif(not GPU, reason="integer-address kernels need a GPU")


def _ptrs(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=ts[0].device)


@needs_gpu
@pytest.mark.parametrize("hint", [True, False])
def test_push_payloads_delivers_rows_to_every_peer(hint):
    torch.manual_seed(0)
    world, width = 2, 896
    counts = [5, 0]
    offs = [0, 5]
    peers_q = [torch.zeros(8, width, dtype=torch.uint8, device="cuda") for _ in range(world)]
    peers_i = [torch.zeros(8, 6, dtype=torch.int32, device="cuda") for _ in range(world)]
    pq, pi = _ptrs(peers_q), _ptrs(peers_i)
    rows = [(torch.randint(0, 256, (c, width), dtype=torch.uint8, device="cuda"),
             torch.randint(0, 384, (c, 6), dtype=torch.int32, device="cuda")) for c in counts]
    for r in range(world):
        rsc.push_payloads([(rows[r][0], pq.data_ptr(), 0, width), (rows[r][1], pi.data_ptr(), 0, 24)], world,
                          counts[r], offs[r], hint=hint)
    torch.cuda.synchronize()
    for p in range(world):
        assert torch.equal(peers_q[p][:5], rows[0][0])
        assert torch.equal(peers_i[p][:5], rows[0][1])
        assert not peers_q[p][5:].any()


@needs_gpu
@pytest.mark.parametrize("n", [1, 6, 300])
def test_pull_matches_reference_kernel(n):
    torch.manual_seed(1)
    world, H, off = 2, 7168, 3
    parts = [(torch.randn(off + n + 4, H, device="cuda") * (r + 1)).to(torch.bfloat16) for r in range(world)]
    ptrs = _ptrs(parts)
    shared = torch.randn(n, H, device="cuda").to(torch.bfloat16)
    valid = torch.tensor([min(4, n)], dtype=torch.int32, device="cuda")
    ref = torch.empty(n, H, dtype=torch.bfloat16, device="cuda")
    sc.pull_reduce_rows(parts, off, n, ref, scale=2.5, shared=shared, valid=valid)
    out = torch.empty_like(ref)
    rsc.pull_reduce_v(ptrs.data_ptr(), world, off, n, out, scale=2.5, shared=shared, valid=valid)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert not out[min(4, n):].any()


def test_rejects_multicast_and_bad_shapes():
    with pytest.raises(ValueError):
        rsc.push_payloads([(torch.zeros(1, 4, dtype=torch.uint8), 0, 0, 4)], 1, 1, 0, multicast=True)
    with pytest.raises(ValueError):
        rsc.pull_reduce_v(0, 1, 0, 1, torch.zeros(1, 12, dtype=torch.bfloat16))


@pytest.mark.skipif(not (GPU and torch.cuda.device_count() >= 2), reason="needs 2 ROCm GPUs")
def test_two_gpu_torchrun():
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run(["torchrun", "--nproc-per-node", "2", os.path.join(here, "rocm_gpu_test_symm.py")],
                       capture_output=True, text=True, timeout=1200)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("@@REPORT@@")), None)
    assert line, r.stderr[-3000:]
    rep = json.loads(line[len("@@REPORT@@"):])
    print(json.dumps(rep, indent=1))
    fails = [ln for ln in r.stdout.splitlines() if " FAIL " in ln]
    assert rep["failures_all_ranks"] == 0, fails
