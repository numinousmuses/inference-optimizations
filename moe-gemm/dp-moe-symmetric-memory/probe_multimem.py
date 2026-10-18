"""Dev-box probe (torchrun --nproc-per-node 8): 128-bit peer stores from integer-argument addresses, NVLink multicast
stores (multimem.st) and switch reductions (multimem.ld_reduce, bf16x2 with f32 accumulation) from Triton."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl


@triton.jit
def _store_intaddr(src, dst_addr, W: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(src + cols, mask=cols < W)
    base = dst_addr.to(tl.pointer_type(tl.int32))
    tl.store(base + cols, v, mask=cols < W)


@triton.jit
def _mc_store(src, mc_addr, n_chunks, BLOCK: tl.constexpr):
    """Each lane moves one 16-byte chunk: four int32 from src into the multicast mapping."""
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = c < n_chunks
    a = tl.load(src + c * 4 + 0, mask=m)
    b = tl.load(src + c * 4 + 1, mask=m)
    d = tl.load(src + c * 4 + 2, mask=m)
    e = tl.load(src + c * 4 + 3, mask=m)
    addr = mc_addr + c.to(tl.int64) * 16
    addr = tl.where(m, addr, mc_addr)       # masked lanes store chunk 0's own value again (harmless re-store)
    a = tl.where(m, a, tl.load(src + 0))
    b = tl.where(m, b, tl.load(src + 1))
    d = tl.where(m, d, tl.load(src + 2))
    e = tl.where(m, e, tl.load(src + 3))
    tl.inline_asm_elementwise(
        "multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0;",
        "=r,l,r,r,r,r", [addr, a, b, d, e], dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _mc_ld_reduce(mc_addr, out, n_chunks, BLOCK: tl.constexpr, ACC32: tl.constexpr):
    """Each lane reads one 16-byte chunk (8 bf16) summed over every GPU by the switch; writes 8 fp32."""
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = c < n_chunks
    addr = mc_addr + tl.where(m, c, 0).to(tl.int64) * 16
    if ACC32:
        r0, r1, r2, r3 = tl.inline_asm_elementwise(
            "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l", [addr], dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32), is_pure=False, pack=1)
    else:
        r0, r1, r2, r3 = tl.inline_asm_elementwise(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l", [addr], dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32), is_pure=False, pack=1)
    _store_pair(out, c * 8 + 0, r0, m)
    _store_pair(out, c * 8 + 2, r1, m)
    _store_pair(out, c * 8 + 4, r2, m)
    _store_pair(out, c * 8 + 6, r3, m)


@triton.jit
def _store_pair(out, off, r, m):
    tl.store(out + off, (r << 16).to(tl.float32, bitcast=True), mask=m)
    tl.store(out + off + 1, (r & 0xFFFF0000).to(tl.float32, bitcast=True), mask=m)


def main() -> None:
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    name = dist.group.WORLD.group_name
    n_int = 1 << 20
    t = symm_mem.empty((n_int,), dtype=torch.int32, device="cuda")
    t.zero_()
    h = symm_mem.rendezvous(t, name)
    if rank == 0:
        src = torch.arange(4096, dtype=torch.int32, device="cuda")
        c = _store_intaddr[(1,)](src, h.buffer_ptrs[1], W=4096, BLOCK=4096)
        ptx = c.asm["ptx"]
        print(f"int-arg address: st.global.v4 {ptx.count('st.global.v4')}, st.global.b32 {ptx.count('st.global.b32')}",
              flush=True)
    dist.barrier()
    # multicast store: rank r writes its slice [r*S, (r+1)*S) through the multicast mapping; every GPU must see all
    S = n_int // world
    mine = torch.full((S,), 1000 + rank, dtype=torch.int32, device="cuda")
    _mc_store[(triton.cdiv(S // 4, 1024),)](mine, h.multicast_ptr + rank * S * 4, S // 4, BLOCK=1024)
    torch.cuda.synchronize()
    h.barrier(channel=0)
    expect = torch.cat([torch.full((S,), 1000 + r, dtype=torch.int32, device="cuda") for r in range(world)])
    ok_store = bool(torch.equal(t, expect))
    # switch reduction: every GPU fills a bf16 buffer with (rank+1) * pattern; ld_reduce must return the sum
    nb = 1 << 16
    tb = symm_mem.empty((nb,), dtype=torch.bfloat16, device="cuda")
    hb = symm_mem.rendezvous(tb, name)
    pat = torch.linspace(-2, 2, nb, device="cuda")
    tb.copy_((pat * (rank + 1)).to(torch.bfloat16))
    torch.cuda.synchronize()
    hb.barrier(channel=0)
    ref = sum((pat * (r + 1)).to(torch.bfloat16).float() for r in range(world))
    res = {}
    for acc32 in (True, False):
        out = torch.empty(nb, dtype=torch.float32, device="cuda")
        try:
            _mc_ld_reduce[(triton.cdiv(nb // 8, 1024),)](hb.multicast_ptr, out, nb // 8, BLOCK=1024, ACC32=acc32)
            torch.cuda.synchronize()
            res[acc32] = float((out - ref).abs().max() / ref.abs().max())
        except Exception as e:  # noqa: BLE001
            res[acc32] = f"{type(e).__name__}: {str(e)[:160]}"
    ok = torch.tensor([1 if ok_store else 0], device="cuda")
    dist.all_reduce(ok)
    if rank == 0:
        print(f"multicast store visible on all GPUs: {int(ok.item())}/{world}", flush=True)
        print(f"ld_reduce rel err: acc::f32 {res[True]}; bf16 acc {res[False]}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
