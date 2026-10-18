"""Portable (no inline assembly) symmetric-memory push / pull kernels for ROCm (gfx950), API-compatible with the
unicast paths of symm_collectives.push_payloads and symm_collectives.pull_reduce_v.

Why: symm_collectives' vectorized kernels move 16-byte chunks with inline PTX (ld/st.global.v4.b32, multimem.*),
which cannot compile for AMDGPU. The pointer-array kernels there (push_rows_ptrs, pull_reduce_rows_ptrs) are plain
Triton and do run, but load and store one 32-bit word per lane through an integer-to-pointer cast. These kernels
are flat grid-stride copies/reductions over a payload's contiguous rows with divisibility hints on every offset and
on the peer base pointer (symmetric-memory mappings are page aligned), so the AMD backend can emit 128-bit
global accesses; semantics are identical: push = every rank's rows land at row_off.. of every peer's buffer;
pull = scale * sum over peers (fixed peer order, fp32) + shared, rows past `valid` zeroed, stored bf16.
Multicast has no equivalent on MI355X (no NVLS; torch symmetric memory reports multicast_ptr 0 there).

Measured on the mi355x box (2 GPUs, rocm_gpu_test_symm.py): see agents/status/mi355x.md.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _as_words(t: torch.Tensor) -> torch.Tensor:
    """Contiguous rows reinterpreted as int32 words [rows, row_bytes / 4]."""
    t = t.contiguous()
    rows = t.shape[0]
    b = t.view(torch.uint8).reshape(rows, -1)
    if b.shape[1] % 4:
        raise ValueError(f"row of {b.shape[1]} bytes is not a multiple of 4")
    return b.view(torch.int32)


@triton.jit
def _push_one_flat(src, dst_ptrs_addr, n_rows, row_off, pid, nprog, W: tl.constexpr, WORLD: tl.constexpr,
                   BLOCK: tl.constexpr, HINT: tl.constexpr):
    """One payload: rows of W int32 words from src into rows row_off.. of every peer's buffer."""
    total = n_rows * W
    base_off = row_off * W
    if HINT and W % 4 == 0:
        total = tl.multiple_of(total, 4)
        base_off = tl.multiple_of(base_off, 4)
    ptrs = dst_ptrs_addr.to(tl.pointer_type(tl.uint64))
    for start in tl.range(pid * BLOCK, total, nprog * BLOCK):
        w = start + tl.arange(0, BLOCK)
        if HINT:
            w = tl.max_contiguous(tl.multiple_of(w, BLOCK), BLOCK)
        m = w < total
        v = tl.load(src + w, mask=m)
        off = w + base_off
        for p in tl.static_range(WORLD):
            base = tl.load(ptrs + p).to(tl.pointer_type(tl.int32))
            if HINT:
                base = tl.multiple_of(base, 16)
            tl.store(base + off, v, mask=m)


@triton.jit
def _push4_flat_kernel(s0, d0, s1, d1, s2, d2, s3, d3, n_rows, row_off, fence_ptr,
                       W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr, WORLD: tl.constexpr,
                       BLOCK: tl.constexpr, HINT: tl.constexpr, FENCE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    pay = tl.program_id(1)
    if pay == 0:
        _push_one_flat(s0, d0, n_rows, row_off, pid, nprog, W0, WORLD, BLOCK, HINT)
    elif pay == 1:
        _push_one_flat(s1, d1, n_rows, row_off, pid, nprog, W1, WORLD, BLOCK, HINT)
    elif pay == 2:
        _push_one_flat(s2, d2, n_rows, row_off, pid, nprog, W2, WORLD, BLOCK, HINT)
    else:
        _push_one_flat(s3, d3, n_rows, row_off, pid, nprog, W3, WORLD, BLOCK, HINT)
    if FENCE:
        # system-scope release after this program's peer stores, so the remote (xGMI) stores are complete before a
        # peer acquires through the next kernel's barrier
        tl.atomic_add(fence_ptr, 0, sem="release", scope="sys")


_FENCE: dict = {}


def _fence_word(device: torch.device) -> torch.Tensor:
    t = _FENCE.get(device)
    if t is None:
        t = _FENCE[device] = torch.zeros(1, dtype=torch.int32, device=device)
    return t


def push_payloads(payloads: list, world: int, n_rows: int, row_off: int, multicast: bool = False,
                  block: int = 1024, max_ctas: int = 256, hint: bool = True, fence: bool = True) -> None:
    """Drop-in for symm_collectives.push_payloads (unicast): payloads = [(src [n_rows, ...] contiguous, peer
    base-address array address (buffer_ptrs_dev), multicast address (ignored), width_bytes)], at most four.
    fence: each program ends with a system-scope release so a peer that acquires through the following barrier sees
    the rows (measured cost with the grid capped at 256 programs: +9 us at 96 rows, +1% at 4096 rows). Stale reads
    seen in an earlier test version are described in rocm_symm_mem.IpcSymmetricBuffer."""
    if multicast:
        raise ValueError("no multicast on ROCm")
    if n_rows == 0 or not payloads:
        return
    if len(payloads) > 4:
        raise ValueError("at most four payloads per launch")
    srcs, dsts, ws = [], [], []
    for src, ptrs_addr, _mc, width_bytes in payloads:
        s = _as_words(src)
        if s.shape[0] < n_rows or s.shape[1] * 4 != width_bytes:
            raise ValueError(f"payload rows of {s.shape[1] * 4} bytes into {width_bytes}-byte buffer rows")
        srcs.append(s)
        dsts.append(int(ptrs_addr))
        ws.append(s.shape[1])
    n_pay = len(srcs)
    while len(srcs) < 4:
        srcs.append(srcs[0])
        dsts.append(dsts[0])
        ws.append(ws[0])
    units = max(n_rows * w for w in ws[:n_pay])
    nblk = triton.cdiv(units, block)
    grid = (min(nblk, max_ctas) if max_ctas > 0 else nblk, n_pay)
    _push4_flat_kernel[grid](srcs[0], dsts[0], srcs[1], dsts[1], srcs[2], dsts[2], srcs[3], dsts[3], n_rows,
                             row_off, _fence_word(srcs[0].device), W0=ws[0], W1=ws[1], W2=ws[2], W3=ws[3],
                             WORLD=world, BLOCK=block, HINT=hint, FENCE=fence)


@triton.jit
def _pull_flat_kernel(src_ptrs_addr, shared, out, valid, n_rows, row_off, scale, H: tl.constexpr,
                      WORLD: tl.constexpr, HAS_SHARED: tl.constexpr, HAS_VALID: tl.constexpr, BLOCK: tl.constexpr,
                      HINT: tl.constexpr):
    total = n_rows * H
    base_off = row_off * H
    if HINT:
        total = tl.multiple_of(total, 8)
        base_off = tl.multiple_of(base_off, 8)
    ptrs = src_ptrs_addr.to(tl.pointer_type(tl.uint64))
    n_valid = n_rows
    if HAS_VALID:
        n_valid = tl.load(valid)
    for start in tl.range(tl.program_id(0) * BLOCK, total, tl.num_programs(0) * BLOCK):
        e = start + tl.arange(0, BLOCK)
        if HINT:
            e = tl.max_contiguous(tl.multiple_of(e, BLOCK), BLOCK)
        m = e < total
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for p in tl.static_range(WORLD):                   # fixed peer order: identical sums on every rank
            base = tl.load(ptrs + p).to(tl.pointer_type(tl.bfloat16))
            if HINT:
                base = tl.multiple_of(base, 16)
            acc += tl.load(base + base_off + e, mask=m, other=0.0).to(tl.float32)
        acc = acc * scale
        if HAS_SHARED:
            acc += tl.load(shared + e, mask=m, other=0.0).to(tl.float32)
        if HAS_VALID:
            acc = tl.where(e // H < n_valid, acc, 0.0)
        tl.store(out + e, acc.to(tl.bfloat16), mask=m)


def pull_reduce_v(src: int, world: int, row_off: int, n_rows: int, out: torch.Tensor, scale: float = 1.0,
                  shared: torch.Tensor | None = None, valid: torch.Tensor | None = None, multicast: bool = False,
                  block: int = 1024, max_ctas: int = 0, hint: bool = True) -> torch.Tensor:
    """Drop-in for symm_collectives.pull_reduce_v (unicast): src = the bf16 partial buffer's peer base-address
    array (buffer_ptrs_dev). Same arithmetic order as pull_reduce_rows_ptrs (bit-identical results)."""
    if multicast:
        raise ValueError("no multicast on ROCm")
    if n_rows == 0:
        return out
    h = out.shape[1]
    if h % 8 or not out.is_contiguous() or out.dtype != torch.bfloat16:
        raise ValueError("out must be contiguous bf16 with H a multiple of 8")
    if shared is not None and (not shared.is_contiguous() or shared.dtype != torch.bfloat16 or shared.shape[1] != h):
        raise ValueError("shared must be contiguous bf16 [>= n_rows, H]")
    nblk = triton.cdiv(n_rows * h, block)
    grid = (min(nblk, max_ctas) if max_ctas > 0 else nblk,)
    _pull_flat_kernel[grid](int(src), shared if shared is not None else out, out,
                            valid if valid is not None else out, n_rows, row_off, float(scale), H=h, WORLD=world,
                            HAS_SHARED=shared is not None, HAS_VALID=valid is not None, BLOCK=block, HINT=hint)
    return out
