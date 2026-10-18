"""2-GPU ROCm test of symmetric memory and our symmetric-memory kernels (torchrun --nproc-per-node 2, mi355x box).

Modeled on gpu_test_symm_v.py (8x B200), without flashinfer. Sections:
  A. torch symmetric memory on this ROCm build: empty + rendezvous, handle attributes, peer stores through
     get_buffer(peer), barrier, put/wait_signal. If rendezvous fails the error is recorded and B-D run on
     rocm_symm_mem.IpcSymmetricBuffer (legacy HIP IPC, same handle subset: buffer_ptrs_dev, barrier).
  B. symm_collectives kernels as they are: push_rows_ptrs / pull_reduce_rows_ptrs (plain Triton) and push_payloads /
     pull_reduce_v unicast (inline PTX: expected not to compile for AMDGPU; the exact error is recorded).
  C. (SYMM_TEST_TENSOR_ARGS=1, run on its own) symm_collectives' tensor-argument push_rows on peer tensors opened
     through torch's storage IPC (tensors on the owner's device, kernel on the local one).
  D. rocm_symm_collectives (portable) push_payloads / pull_reduce_v: correctness in the production order (push, device
     barrier, read; no host sync between), uneven per-rank counts with an idle rank, a producer kernel right before
     the push; then graph-replay timing (decode 96 rows per rank, prefill 4096) against the reference kernels and
     an RCCL all_reduce of the same bytes.
Prints one JSON report on rank 0 (@@REPORT@@); exit code 1 on any correctness failure.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rocm_symm_collectives as rsc  # noqa: E402
import symm_collectives as sc  # noqa: E402
from rocm_symm_mem import IpcSymmetricBuffer  # noqa: E402

H, TOPK = 7168, 6


def step(msg: str) -> None:
    if os.environ.get("SYMM_TEST_TRACE"):
        torch.cuda.synchronize()
        print(f"[rank {os.environ.get('RANK')}] STEP {msg}", flush=True)


def err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:1200]


def timed(fn, graph: bool, iters: int = 50) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    run = fn
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


_PROBE = r"""
import sys, torch
sys.path.insert(0, %r)
import symm_collectives as sc
x = torch.zeros(64, 7168, dtype=torch.uint8, device="cuda")
ptrs = torch.tensor([x.data_ptr()], dtype=torch.int64, device="cuda")
if sys.argv[1] == "push":
    src = torch.randint(0, 256, (16, 7168), dtype=torch.uint8, device="cuda")
    sc.push_payloads([(src, ptrs.data_ptr(), 0, 7168)], 1, 16, 0)
    torch.cuda.synchronize()
    print("RESULT", "ok" if torch.equal(x[:16], src) else "wrong rows")
else:
    part = torch.randn(64, 7168, device="cuda").to(torch.bfloat16)
    pp = torch.tensor([part.data_ptr()], dtype=torch.int64, device="cuda")
    out = torch.empty(16, 7168, dtype=torch.bfloat16, device="cuda")
    sc.pull_reduce_v(pp.data_ptr(), 1, 0, 16, out)
    torch.cuda.synchronize()
    print("RESULT", "ok" if torch.equal(out, part[:16]) else "wrong result")
"""


def _ptx_probe(which: str) -> str:
    """Compile + run one inline-PTX symm_collectives kernel in a subprocess; its outcome or the compiler's words."""
    import subprocess
    code = _PROBE % os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, "-c", code, which], capture_output=True, text=True, timeout=600,
                       env={**os.environ, "HIP_VISIBLE_DEVICES": str(torch.cuda.current_device())})
    res = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")), None)
    if res:
        return res[len("RESULT "):]
    errs = [ln for ln in r.stderr.splitlines() if "error" in ln.lower() and "[aiter]" not in ln]
    return f"rc={r.returncode}: " + " | ".join(errs[:4])[:800]


class Sym:
    """A symmetric buffer: torch symmetric memory when it rendezvous, else legacy HIP IPC."""

    def __init__(self, shape, dtype, backend: str, group_name: str):
        self.backend = backend
        if backend == "torch":
            self.local = symm_mem.empty(shape, dtype=dtype, device="cuda")
            self.local.zero_()
            self.h = symm_mem.rendezvous(self.local, group_name)
            self.buffer_ptrs_dev = self.h.buffer_ptrs_dev
        else:
            self.h = IpcSymmetricBuffer(shape, dtype)
            self.local = self.h.local
            self.buffer_ptrs_dev = self.h.buffer_ptrs_dev

    def barrier(self, channel=0):
        self.h.barrier(channel=channel)


def section_a(rank, world, name) -> dict:
    a: dict = {}
    try:
        t = symm_mem.empty((1024, 256), dtype=torch.float32, device="cuda")
        a["empty"] = "ok"
        t.fill_(-1)
        hdl = symm_mem.rendezvous(t, name)
        a["rendezvous"] = "ok"
        a["multicast_ptr"] = int(getattr(hdl, "multicast_ptr", 0) or 0)
        torch.cuda.synchronize()
        hdl.barrier(channel=0)
        pb = hdl.get_buffer((rank + 1) % world, (1024, 256), torch.float32)
        pb[rank::world].fill_(float(rank))
        hdl.barrier(channel=0)
        src = (rank - 1) % world
        a["peer_store"] = bool(torch.equal(t[src::world], torch.full((1024 // world, 256), float(src), device="cuda")))
        try:
            if rank == 0:
                hdl.put_signal(dst_rank=1, channel=1)
            else:
                hdl.wait_signal(src_rank=0, channel=1)
            torch.cuda.synchronize()
            a["put_wait_signal"] = "ok"
        except Exception as e:  # noqa: BLE001
            a["put_wait_signal"] = err(e)
    except Exception as e:  # noqa: BLE001
        a.setdefault("empty", "ok")
        a["rendezvous"] = err(e)
    return a


def main() -> int:
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    name = dist.group.WORLD.group_name
    rep: dict = {"gpu": torch.cuda.get_device_name(0), "arch": torch.cuda.get_device_properties(0).gcnArchName,
                 "torch": torch.__version__, "hip": torch.version.hip, "world": world}
    fails: list[str] = []
    a = section_a(rank, world, name)
    step("rep_A_torch_symm_mem_a")
    rep["A_torch_symm_mem"] = a
    backend = "torch" if a.get("rendezvous") == "ok" else "hip-ipc"
    rep["buffer_backend"] = backend

    cap = 4096 * world
    xq = Sym((cap, H), torch.uint8, backend, name)
    xs = Sym((cap, H // 32), torch.uint8, backend, name)
    ids = Sym((cap, TOPK), torch.int32, backend, name)
    wts = Sym((cap, TOPK), torch.float32, backend, name)
    part = Sym((cap, H), torch.bfloat16, backend, name)
    torch.cuda.synchronize()
    step("xq_barrier_barrier_sanity")
    xq.barrier()

    # barrier sanity on the IPC path: every rank writes its id into every peer, one barrier, every rank reads
    counts = [((r * 37) % 211) + 5 for r in range(world)]
    if world > 1:
        counts[1] = 0                                   # an idle rank
    offs = [sum(counts[:r]) for r in range(world)]

    def rows_of(r, n=None):
        gg = torch.Generator(device="cuda").manual_seed(1000 + r)
        n = counts[r] if n is None else n
        return (torch.randint(0, 256, (n, H), dtype=torch.uint8, device="cuda", generator=gg),
                torch.randint(0, 256, (n, H // 32), dtype=torch.uint8, device="cuda", generator=gg),
                torch.randint(0, 384, (n, TOPK), dtype=torch.int32, device="cuda", generator=gg),
                torch.rand((n, TOPK), dtype=torch.float32, device="cuda", generator=gg))

    def payloads(src):
        return [(src[0], xq.buffer_ptrs_dev, 0, H), (src[1], xs.buffer_ptrs_dev, 0, H // 32),
                (src[2], ids.buffer_ptrs_dev, 0, TOPK * 4), (src[3], wts.buffer_ptrs_dev, 0, TOPK * 4)]

    def check_push(label, xform=None):
        for r in range(world):
            ref = rows_of(r)
            if xform:
                ref = xform(ref)
            o, n = offs[r], counts[r]
            for got, want, nm in ((xq.local, ref[0], "xq"), (xs.local, ref[1], "xs"), (ids.local, ref[2], "ids"),
                                  (wts.local, ref[3], "wts")):
                if not torch.equal(got[o:o + n], want):
                    return f"{label}: rank {rank} holds wrong {nm} rows of rank {r}"
        return None

    def reset():
        for x in (xq, xs, ids, wts):
            x.local.zero_()
        torch.cuda.synchronize()
        xq.barrier()

    def partial_of(r):
        g = torch.Generator(device="cuda").manual_seed(1234 + r)
        return (torch.randn(cap, H, device="cuda", generator=g) * (r + 1)).to(torch.bfloat16)

    part.local.copy_(partial_of(rank))
    torch.cuda.synchronize()
    part.barrier()
    n_mine, off_mine = 300, 300 * rank
    shared = torch.randn(n_mine, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)).to(
        torch.bfloat16)
    valid = torch.tensor([250], dtype=torch.int32, device="cuda")
    acc = torch.zeros(n_mine, H, device="cuda")
    for p in range(world):                               # fp32 in peer order 0..W-1, then scale, then shared
        acc += partial_of(p)[off_mine:off_mine + n_mine].float()
    want = acc * 2.5 + shared.float()
    want[250:] = 0
    want = want.to(torch.bfloat16)

    step("_B_")
    # ---------------------------------------------------------------- B. symm_collectives as-is
    b: dict = {}
    reset()
    try:
        mine = rows_of(rank)
        for s_, h_, w_ in ((mine[0], xq, H), (mine[1], xs, H // 32), (mine[2], ids, TOPK * 4), (mine[3], wts, TOPK * 4)):
            sc.push_rows_ptrs(s_, h_.buffer_ptrs_dev, world, w_, offs[rank])
        xq.barrier()
        f = check_push("B push_rows_ptrs")
        b["push_rows_ptrs"] = "ok" if f is None else f
        if f:
            fails.append(f)
    except Exception as e:  # noqa: BLE001
        b["push_rows_ptrs"] = err(e)
        fails.append(f"B push_rows_ptrs: {err(e)}")
    # the inline-PTX kernels abort the process when LLVM rejects them (not a Python exception), so they are
    # compiled in a throwaway single-GPU subprocess: world 1, the local buffer as its own peer
    if rank == 0:
        b["push_payloads_unicast"] = _ptx_probe("push")
        b["pull_reduce_v_unicast"] = _ptx_probe("pull")
    step("ref_torch_empty_n_mine")
    ref = torch.empty(n_mine, H, dtype=torch.bfloat16, device="cuda")
    try:
        sc.pull_reduce_rows_ptrs(part.buffer_ptrs_dev, world, off_mine, n_mine, ref, scale=2.5, shared=shared,
                                 valid=valid)
        torch.cuda.synchronize()
        ok = torch.equal(ref, want)
        b["pull_reduce_rows_ptrs"] = "ok (bit-equal to torch fp32 reference)" if ok else \
            f"max abs diff {(ref.float() - want.float()).abs().max().item()}"
        if not ok:
            fails.append("B pull_reduce_rows_ptrs != torch reference")
    except Exception as e:  # noqa: BLE001
        b["pull_reduce_rows_ptrs"] = err(e)
        fails.append(f"B pull_reduce_rows_ptrs: {err(e)}")
    rep["B_symm_collectives"] = b

    step("_D_")
    # ---------------------------------------------------------------- D. portable kernels
    d: dict = {}
    for hint in (True, False):
        tag = "hint" if hint else "nohint"
        reset()
        try:
            rsc.push_payloads(payloads(rows_of(rank)), world, counts[rank], offs[rank], hint=hint)
            xq.barrier()                   # production order: device barrier only
            f = check_push(f"D push_payloads[{tag}]")
            d[f"push_payloads_{tag}"] = "ok" if f is None else f
            if f:
                fails.append(f)
        except Exception as e:  # noqa: BLE001
            d[f"push_payloads_{tag}"] = err(e)
            fails.append(f"D push_payloads[{tag}]: {err(e)}")
        try:
            out = torch.empty_like(ref)
            rsc.pull_reduce_v(part.buffer_ptrs_dev, world, off_mine, n_mine, out, scale=2.5, shared=shared,
                              valid=valid, hint=hint)
            torch.cuda.synchronize()
            ok = torch.equal(out, want)
            d[f"pull_reduce_v_{tag}"] = "ok (bit-equal to reference)" if ok else \
                f"max abs diff {(out.float() - want.float()).abs().max().item()}"
            if not ok:
                fails.append(f"D pull_reduce_v[{tag}] wrong")
        except Exception as e:  # noqa: BLE001
            d[f"pull_reduce_v_{tag}"] = err(e)
            fails.append(f"D pull_reduce_v[{tag}]: {err(e)}")
    step("reset_src_rows_of_rank_src_src_x_A")
    # local rewrite then remote push: reset() zeroes the receive buffers and host-syncs, then the peer pushes. Known
    # coarse-grained-memory hazard (rocm_symm_mem docstring): recorded as a rate, not a failure
    stale_rounds = 0
    for it in range(10):
        reset()
        src = rows_of(rank)
        src = (src[0] ^ ((0x5A + it) & 0xFF), src[1], src[2], src[3])  # produced on-stream right before the push
        rsc.push_payloads(payloads(src), world, counts[rank], offs[rank])
        xq.barrier()
        vv = (0x5A + it) & 0xFF
        if check_push("D reset then push", xform=lambda r, vv=vv: (r[0] ^ vv, r[1], r[2], r[3])):
            stale_rounds += 1
        xq.barrier()
    d["reset_hostsync_then_push_stale_rounds_of_10"] = stale_rounds
    step("_repeated_push_barrier")
    # repeated push/barrier rounds with changing data: stale-cache detection across rounds
    stale = 0
    for it in range(20):
        mine = tuple(t ^ it if t.dtype == torch.uint8 else t for t in rows_of(rank))
        rsc.push_payloads(payloads(mine), world, counts[rank], offs[rank])
        xq.barrier()
        for r in range(world):
            if not torch.equal(xq.local[offs[r]:offs[r] + counts[r]], rows_of(r)[0] ^ it):
                stale += 1
        xq.barrier()
    d["push_20_rounds_stale"] = stale
    if stale:
        fails.append(f"D push rounds: {stale} stale reads")
    # visibility race: producer kernel -> push -> device barrier -> consumer kernel, back to back, no host sync,
    # 200 rounds; stale = rounds in which a rank read another rank's rows from an earlier round
    for fence in (False, True):
        bad_rounds = torch.zeros(1, dtype=torch.int32, device="cuda")
        base = rows_of(rank)[0]
        for it in range(200):
            v = (it * 7 + 1) & 0xFF
            rsc.push_payloads([(base ^ v, xq.buffer_ptrs_dev, 0, H)], world, counts[rank], offs[rank], fence=fence)
            xq.barrier()
            for r in range(world):
                if counts[r]:
                    want0 = rows_of(r)[0] ^ v
                    bad_rounds += (xq.local[offs[r]:offs[r] + counts[r]] != want0).any().int()
            xq.barrier()
        torch.cuda.synchronize()
        d[f"race_200_rounds_stale_fence_{fence}"] = int(bad_rounds.item())
        if fence and int(bad_rounds.item()):
            fails.append(f"D race with fence: {int(bad_rounds.item())} stale rounds")
    rep["D_portable"] = d

    # ---------------------------------------------------------------- E. local writes then remote writes
    # rank 1 zeroes its receive buffer locally, then rank 0 pushes into it: does a dirty line from the local zeroing
    # get written back over the remote rows later? 40 rounds per variant; wrong = rounds with wrong rows.
    step("_E_")
    e: dict = {}
    import rocm_symm_mem as rsm
    base0 = rows_of(0)[0]
    for variant in ("zero_then_barrier", "zero_sysfence_then_barrier", "no_local_zero"):
        wrong = torch.zeros(1, dtype=torch.int32, device="cuda")
        for it in range(40):
            v = (it * 13 + 3) & 0xFF
            if variant != "no_local_zero":
                xq.local.zero_()
                if variant == "zero_sysfence_then_barrier":
                    rsm.system_release()
            xq.barrier()
            rsc.push_payloads([((rows_of(rank)[0]) ^ v, xq.buffer_ptrs_dev, 0, H)], world, counts[rank], offs[rank])
            xq.barrier()
            wrong += (xq.local[offs[0]:offs[0] + counts[0]] != (base0 ^ v)).any().int()
            xq.barrier()
        torch.cuda.synchronize()
        e[variant] = int(wrong.item())
    rep["E_local_then_remote"] = e

    step("_timing")
    # ---------------------------------------------------------------- timing
    tim = {}
    for label, n, graph in (("decode_96", 96, True), ("prefill_4096", 4096, False)):
        src = rows_of(rank, n)
        off = n * rank
        out = torch.empty(n, H, dtype=torch.bfloat16, device="cuda")
        sh = torch.randn(n, H, device="cuda").to(torch.bfloat16)
        ar = torch.randn(n, H, device="cuda").to(torch.bfloat16)

        def bar():
            xq.barrier()

        def ref_push():
            for s_, h_, w_ in ((src[0], xq, H), (src[1], xs, H // 32), (src[2], ids, TOPK * 4),
                               (src[3], wts, TOPK * 4)):
                sc.push_rows_ptrs(s_, h_.buffer_ptrs_dev, world, w_, off)
        r: dict = {}
        for k, fn, gr in (("barrier", bar, graph),
                          ("push_rows_ptrs_x4", lambda: (ref_push(), bar()), graph),
                          ("push_portable_hint", lambda: (rsc.push_payloads(payloads(src), world, n, off), bar()), graph),
                          ("push_portable_nohint", lambda: (rsc.push_payloads(payloads(src), world, n, off, hint=False),
                                                            bar()), graph),
                          ("push_portable_nofence", lambda: (rsc.push_payloads(payloads(src), world, n, off,
                                                                               fence=False), bar()), graph),
                          ("push_portable_fence_ctas256", lambda: (rsc.push_payloads(payloads(src), world, n, off,
                                                                                     max_ctas=256), bar()), graph),
                          ("pull_reduce_rows_ptrs", lambda: (sc.pull_reduce_rows_ptrs(part.buffer_ptrs_dev, world, off,
                                                                                      n, out, 2.5, sh), bar()), graph),
                          ("pull_portable_hint", lambda: (rsc.pull_reduce_v(part.buffer_ptrs_dev, world, off, n, out,
                                                                            2.5, sh), bar()), graph),
                          ("pull_portable_nohint", lambda: (rsc.pull_reduce_v(part.buffer_ptrs_dev, world, off, n, out,
                                                                              2.5, sh, hint=False), bar()), graph),
                          ("rccl_all_reduce_same_rows", lambda: dist.all_reduce(ar), False)):
            try:
                r[k] = round(timed(fn, gr), 1)
            except Exception as e:  # noqa: BLE001
                r[k] = err(e)[:300]
        push_in = (world - 1) * n * (H + H // 32 + 2 * TOPK * 4)
        pull_in = (world - 1) * n * H * 2
        for k in ("push_rows_ptrs_x4", "push_portable_hint", "push_portable_nohint", "push_portable_nofence",
                  "push_portable_fence_ctas256"):
            if isinstance(r.get(k), float):
                r[k + "_GBps_in"] = round(push_in / r[k] / 1e3, 1)
        for k in ("pull_reduce_rows_ptrs", "pull_portable_hint", "pull_portable_nohint"):
            if isinstance(r.get(k), float):
                r[k + "_GBps_in"] = round(pull_in / r[k] / 1e3, 1)
        tim[label] = r
    rep["timing_us_rank0"] = tim
    if os.environ.get("SYMM_TEST_TENSOR_ARGS"):  # run alone: a failure here is a GPU fault that kills the job
        step("_C_")
        # ---------------------------------------------------------------- C. tensor arguments on peer tensors
        c: dict = {}
        try:
            buf = torch.zeros(cap, H // 32, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            hs: list = [None] * world
            dist.all_gather_object(hs, buf.untyped_storage()._share_cuda_())
            peers = []
            for p in range(world):
                if p == rank:
                    peers.append(buf)
                else:
                    st = torch.UntypedStorage._new_shared_cuda(*hs[p])
                    peers.append(torch.empty(0, dtype=torch.uint8, device=st.device).set_(st, 0, (cap, H // 32)))
            c["peer_tensor_devices"] = [str(t.device) for t in peers]
            dist.barrier()
            sc.push_rows(rows_of(rank)[1], peers, offs[rank])
            torch.cuda.synchronize()
            dist.barrier()
            okc = all(torch.equal(buf[offs[r]:offs[r] + counts[r]], rows_of(r)[1]) for r in range(world))
            c["push_rows_peer_tensors"] = "ok" if okc else "wrong rows"
        except Exception as e:  # noqa: BLE001
            c["push_rows_peer_tensors"] = err(e)
        rep["C_tensor_args"] = c

    return _finish(rep, fails, rank)


def _finish(rep, fails, rank) -> int:
    bad = torch.tensor([len(fails)], device="cuda")
    dist.all_reduce(bad)
    rep["failures_all_ranks"] = int(bad.item())
    rep["failures_rank0"] = fails[:8] if rank == 0 else None
    if rank == 0:
        print("@@REPORT@@" + json.dumps(rep), flush=True)
    for f in fails[:8]:
        print(f"[rank {rank}] FAIL {f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 1 if int(bad.item()) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
