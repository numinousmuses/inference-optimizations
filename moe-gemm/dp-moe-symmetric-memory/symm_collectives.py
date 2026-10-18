"""Variable-length NVLink collectives over symmetric memory, for the DP-attention / TP-expert MoE path.

SGLang's DP-MoE path moves every gathered token twice at full width: prefill gathers with an all-reduce over a
zero-filled buffer (SUM_LEN: NCCL all-gather needs equal sizes, prefill steps are skewed) and combines with a second
all-reduce of the full output, although each GPU keeps only its own rows. These two kernels move only what each GPU
needs:

  push_rows          each GPU stores its own rows (any byte payload: the MXFP8 expert input, its scales, the
                     top-k) straight into every peer's gathered buffer at its row offset. Variable lengths cost
                     nothing: no padding, no zero-fill, no reduction.
  pull_reduce_rows   each GPU reads its own rows from every peer's partial expert output, sums them in fp32 in a
                     fixed peer order (identical on every rank), applies the routed scale and adds its local shared
                     expert output: the scale multiply, the all-reduce, the scatter copy and the shared add in one
                     pass over 1/8 of the data.

Peers are passed as separate pointer arguments (a symmetric buffer's per-rank views), so the kernels also run under
the Triton CPU interpreter with plain tensors standing in for peers (tests). Ordering between the phases is the
caller's (a symmetric-memory barrier after the push, before the pull); the kernels themselves are pure loads and
stores and are CUDA-graph safe.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_WORLD = 8


@triton.jit
def _push_rows_kernel(src, d0, d1, d2, d3, d4, d5, d6, d7, n_rows, row_off,
                      W: tl.constexpr, WORLD: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_w = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    m = (rows[:, None] < n_rows) & (cols[None, :] < W)
    v = tl.load(src + rows[:, None].to(tl.int64) * W + cols[None, :], mask=m)
    off = (row_off + rows[:, None]).to(tl.int64) * W + cols[None, :]
    tl.store(d0 + off, v, mask=m)
    if WORLD > 1:
        tl.store(d1 + off, v, mask=m)
    if WORLD > 2:
        tl.store(d2 + off, v, mask=m)
    if WORLD > 3:
        tl.store(d3 + off, v, mask=m)
    if WORLD > 4:
        tl.store(d4 + off, v, mask=m)
    if WORLD > 5:
        tl.store(d5 + off, v, mask=m)
    if WORLD > 6:
        tl.store(d6 + off, v, mask=m)
    if WORLD > 7:
        tl.store(d7 + off, v, mask=m)


@triton.jit
def _pull_reduce_rows_kernel(p0, p1, p2, p3, p4, p5, p6, p7, shared, out, valid, n_rows, row_off, scale,
                             H: tl.constexpr, WORLD: tl.constexpr, HAS_SHARED: tl.constexpr, HAS_VALID: tl.constexpr,
                             BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    m = (rows[:, None] < n_rows) & (cols[None, :] < H)
    off = (row_off + rows[:, None]).to(tl.int64) * H + cols[None, :]
    acc = tl.load(p0 + off, mask=m, other=0.0).to(tl.float32)
    if WORLD > 1:
        acc += tl.load(p1 + off, mask=m, other=0.0).to(tl.float32)
    if WORLD > 2:
        acc += tl.load(p2 + off, mask=m, other=0.0).to(tl.float32)
    if WORLD > 3:
        acc += tl.load(p3 + off, mask=m, other=0.0).to(tl.float32)
    if WORLD > 4:
        acc += tl.load(p4 + off, mask=m, other=0.0).to(tl.float32)
    if WORLD > 5:
        acc += tl.load(p5 + off, mask=m, other=0.0).to(tl.float32)
    if WORLD > 6:
        acc += tl.load(p6 + off, mask=m, other=0.0).to(tl.float32)
    if WORLD > 7:
        acc += tl.load(p7 + off, mask=m, other=0.0).to(tl.float32)
    acc = acc * scale
    loc = rows[:, None].to(tl.int64) * H + cols[None, :]
    if HAS_SHARED:
        acc += tl.load(shared + loc, mask=m, other=0.0).to(tl.float32)
    if HAS_VALID:
        # CUDA-graph replays: the block is fixed at capture, the real row count is read from the device
        n_valid = tl.load(valid)
        acc = tl.where(rows[:, None] < n_valid, acc, 0.0)
    tl.store(out + loc, acc.to(out.dtype.element_ty), mask=m)


def _pad_world(ts: list[torch.Tensor]) -> list[torch.Tensor]:
    if not 1 <= len(ts) <= MAX_WORLD:
        raise ValueError(f"world size {len(ts)} outside 1..{MAX_WORLD}")
    return list(ts) + [ts[0]] * (MAX_WORLD - len(ts))


def _as_words(t: torch.Tensor) -> torch.Tensor:
    """View a contiguous row-major payload as 32-bit words per row (4-byte lanes; rows must be 4-byte multiples)."""
    rows = t.shape[0]
    nbytes = t.shape[1] * t.element_size() if t.dim() == 2 else t[0].numel() * t.element_size()
    if nbytes % 4:
        raise ValueError(f"row of {nbytes} bytes is not a multiple of 4")
    return t.contiguous().view(torch.uint8).reshape(rows, nbytes).view(torch.int32)


def push_rows(src: torch.Tensor, dsts: list[torch.Tensor], row_off: int, block_r: int = 16,
              block_w: int = 256) -> None:
    """Store src's rows into rows [row_off, row_off + len(src)) of every buffer in dsts (the per-peer views of one
    symmetric buffer, in rank order). Any dtype; each row must be a multiple of 4 bytes."""
    n = src.shape[0]
    if n == 0:
        return
    s = _as_words(src)
    ds = [_as_words(d) for d in dsts]
    w = s.shape[1]
    if any(d.shape[1] != w for d in ds):
        raise ValueError("destination row width differs from the source")
    if any(row_off + n > d.shape[0] for d in ds):
        raise ValueError("rows past the end of a destination buffer")
    grid = (triton.cdiv(n, block_r), triton.cdiv(w, block_w))
    _push_rows_kernel[grid](s, *_pad_world(ds), n, row_off, W=w, WORLD=len(ds), BLOCK_R=block_r, BLOCK_W=block_w)


def pull_reduce_rows(srcs: list[torch.Tensor], row_off: int, n_rows: int, out: torch.Tensor, scale: float = 1.0,
                     shared: torch.Tensor | None = None, valid: torch.Tensor | None = None, block_r: int = 4,
                     block_h: int = 1024) -> torch.Tensor:
    """out[i] = scale * sum_p srcs[p][row_off + i] + shared[i] for i < n_rows: this GPU's rows of the TP-summed expert
    output, peers in rank order (fp32 accumulation, the same order on every rank). valid (a 1-element int32 device
    tensor) zeroes rows at or past its value, so a CUDA graph captured for a padded block stays correct on replay."""
    if n_rows == 0:
        return out
    h = srcs[0].shape[1]
    if out.shape[1] != h or out.shape[0] < n_rows or not out.is_contiguous():
        raise ValueError("output must be contiguous [>= n_rows, H]")
    if any(p.shape[1] != h or row_off + n_rows > p.shape[0] for p in srcs):
        raise ValueError("peer buffer too small for the requested rows")
    if shared is not None and (shared.shape[1] != h or shared.shape[0] < n_rows or not shared.is_contiguous()):
        raise ValueError("shared must be contiguous [>= n_rows, H]")
    if valid is not None and (valid.numel() != 1 or valid.dtype not in (torch.int32, torch.int64)):
        raise ValueError("valid must be a 1-element int32/int64 tensor")
    grid = (triton.cdiv(n_rows, block_r), triton.cdiv(h, block_h))
    _pull_reduce_rows_kernel[grid](*_pad_world(list(srcs)), shared if shared is not None else out, out,
                                   valid if valid is not None else out, n_rows, row_off, float(scale), H=h,
                                   WORLD=len(srcs), HAS_SHARED=shared is not None, HAS_VALID=valid is not None,
                                   BLOCK_R=block_r, BLOCK_H=block_h)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Pointer-array variants for symmetric memory on the GPU. Triton validates tensor pointer arguments and rejects a
# peer's mapping, so the peers' base addresses travel as a device int64 array whose own address is a plain integer
# (torch symmetric memory's buffer_ptrs_dev); each peer pointer is loaded and cast inside the kernel.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _push_rows_ptrs_kernel(src, dst_ptrs_addr, n_rows, row_off, W: tl.constexpr, WORLD: tl.constexpr,
                           BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_w = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    m = (rows[:, None] < n_rows) & (cols[None, :] < W)
    v = tl.load(src + rows[:, None].to(tl.int64) * W + cols[None, :], mask=m)
    off = (row_off + rows[:, None]).to(tl.int64) * W + cols[None, :]
    ptrs = dst_ptrs_addr.to(tl.pointer_type(tl.uint64))
    for p in tl.static_range(WORLD):
        base = tl.load(ptrs + p).to(tl.pointer_type(tl.int32))
        tl.store(base + off, v, mask=m)


@triton.jit
def _pull_reduce_rows_ptrs_kernel(src_ptrs_addr, shared, out, valid, n_rows, row_off, scale, H: tl.constexpr,
                                  WORLD: tl.constexpr, HAS_SHARED: tl.constexpr, HAS_VALID: tl.constexpr,
                                  BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    m = (rows[:, None] < n_rows) & (cols[None, :] < H)
    off = (row_off + rows[:, None]).to(tl.int64) * H + cols[None, :]
    ptrs = src_ptrs_addr.to(tl.pointer_type(tl.uint64))
    acc = tl.zeros((BLOCK_R, BLOCK_H), dtype=tl.float32)
    for p in tl.static_range(WORLD):                       # fixed peer order: identical sums on every rank
        base = tl.load(ptrs + p).to(tl.pointer_type(tl.bfloat16))
        acc += tl.load(base + off, mask=m, other=0.0).to(tl.float32)
    acc = acc * scale
    loc = rows[:, None].to(tl.int64) * H + cols[None, :]
    if HAS_SHARED:
        acc += tl.load(shared + loc, mask=m, other=0.0).to(tl.float32)
    if HAS_VALID:
        n_valid = tl.load(valid)
        acc = tl.where(rows[:, None] < n_valid, acc, 0.0)
    tl.store(out + loc, acc.to(out.dtype.element_ty), mask=m)


def push_rows_ptrs(src: torch.Tensor, dst_ptrs_addr: int, world: int, width_bytes: int, row_off: int,
                   block_r: int = 16, block_w: int = 256) -> None:
    """push_rows over a symmetric buffer given by its peers' base-address array (width_bytes per row)."""
    n = src.shape[0]
    if n == 0:
        return
    s = _as_words(src)
    if s.shape[1] * 4 != width_bytes:
        raise ValueError(f"source rows of {s.shape[1] * 4} bytes into {width_bytes}-byte buffer rows")
    w = s.shape[1]
    grid = (triton.cdiv(n, block_r), triton.cdiv(w, block_w))
    _push_rows_ptrs_kernel[grid](s, dst_ptrs_addr, n, row_off, W=w, WORLD=world, BLOCK_R=block_r, BLOCK_W=block_w)


def pull_reduce_rows_ptrs(src_ptrs_addr: int, world: int, row_off: int, n_rows: int, out: torch.Tensor,
                          scale: float = 1.0, shared: torch.Tensor | None = None, valid: torch.Tensor | None = None,
                          block_r: int = 4, block_h: int = 1024) -> torch.Tensor:
    """pull_reduce_rows over a symmetric bf16 buffer given by its peers' base-address array."""
    if n_rows == 0:
        return out
    h = out.shape[1]
    grid = (triton.cdiv(n_rows, block_r), triton.cdiv(h, block_h))
    _pull_reduce_rows_ptrs_kernel[grid](src_ptrs_addr, shared if shared is not None else out, out,
                                        valid if valid is not None else out, n_rows, row_off, float(scale), H=h,
                                        WORLD=world, HAS_SHARED=shared is not None, HAS_VALID=valid is not None,
                                        BLOCK_R=block_r, BLOCK_H=block_h)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Vectorized kernels (GPU). Peer mappings reach a kernel as raw addresses, and Triton emits scalar 32-bit accesses
# through an integer-to-pointer cast (measured on B200 and B300: 32x st.global.b32, 0x v4, alignment hints and
# integer-argument addresses alike). These move 16-byte chunks with predicated PTX: st.global.v4.b32 per peer, or
# one multimem.st through the NVLink multicast mapping (the switch fans the store out to every GPU), and for the
# combine ld.global.v4.b32 per peer, or one multimem.ld_reduce (the switch returns the sum over every GPU, bf16x2
# with f32 accumulation). All payloads of a push go in one launch.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _st_v4(addr, a, b, c, d):
    tl.inline_asm_elementwise("st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0;", "=r,l,r,r,r,r",
                              [addr, a, b, c, d], dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _mc_st_v4(addr, a, b, c, d):
    tl.inline_asm_elementwise("multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0;",
                              "=r,l,r,r,r,r", [addr, a, b, c, d], dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _fence_alias_release():
    """After stores through the multicast alias: order them before this thread's later generic-proxy operations.
    System-scope visibility comes from the barrier that follows (release / acquire at sys scope)."""
    tl.inline_asm_elementwise("fence.proxy.alias; mov.u32 $0, 0;", "=r", [], dtype=tl.uint32, is_pure=False,
                              pack=1)


@triton.jit
def _fence_alias_acquire():
    """Before reads through the multicast alias of data other GPUs wrote through their unicast mappings."""
    tl.inline_asm_elementwise("fence.proxy.alias; mov.u32 $0, 0;", "=r", [], dtype=tl.uint32, is_pure=False,
                              pack=1)


@triton.jit
def _ld_v4(addr):
    return tl.inline_asm_elementwise("ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", [addr],
                                     dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32), is_pure=False, pack=1)


@triton.jit
def _mc_ld_reduce_v4(addr):
    return tl.inline_asm_elementwise(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l",
        [addr], dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32), is_pure=False, pack=1)


# No predicated inline asm anywhere: a payload's rows are contiguous in the source and in every destination, so a
# push is a flat copy of n_rows * W words, and lanes past the end clamp to the last chunk (re-storing the same
# bytes to the same address). Predicated multimem.st lost rows once FlashInfer's import pointed Triton at the
# CUDA toolkit's ptxas (TRITON_PTXAS_BLACKWELL_PATH): most lanes of a 14-chunk scale row were predicated off.
@triton.jit
def _push_one(src, dst, n_rows, row_off, pid, nprog, W: tl.constexpr, WORLD: tl.constexpr, MC: tl.constexpr,
              BLOCK: tl.constexpr):
    """One payload: rows of W int32 words from src into rows row_off.. of the destination. dst is the address of the
    peers' base-address array (unicast) or the multicast base address (MC). Vector rows: src is an address.
    Grid-stride over the payload: a capped grid (max_ctas) leaves SMs to kernels running alongside."""
    if W % 4 == 0:
        total = n_rows * (W // 4)
        for start in tl.range(pid * BLOCK, total, nprog * BLOCK):
            c = tl.minimum(start + tl.arange(0, BLOCK), total - 1).to(tl.int64)
            a, b, cc, d = _ld_v4(src + c * 16)
            boff = (c + row_off * (W // 4)) * 16
            if MC:
                _mc_st_v4(dst + boff, a, b, cc, d)
            else:
                ptrs = dst.to(tl.pointer_type(tl.uint64))
                for p in tl.static_range(WORLD):
                    base = tl.load(ptrs + p).to(tl.int64)
                    _st_v4(base + boff, a, b, cc, d)
        if MC:
            _fence_alias_release()
    else:  # rows that are not 16-byte multiples (top-k ids / weights: 24-28 bytes): 32-bit unicast stores
        total = n_rows * W
        for start in tl.range(pid * BLOCK, total, nprog * BLOCK):
            w = start + tl.arange(0, BLOCK)
            m = w < total
            v = tl.load(src + w, mask=m)
            off = w + row_off * W
            ptrs = dst.to(tl.pointer_type(tl.uint64))
            for p in tl.static_range(WORLD):
                base = tl.load(ptrs + p).to(tl.pointer_type(tl.int32))
                tl.store(base + off, v, mask=m)


@triton.jit
def _push4_kernel(s0, d0, s1, d1, s2, d2, s3, d3, n_rows, row_off,
                  W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr,
                  WORLD: tl.constexpr, MC0: tl.constexpr, MC1: tl.constexpr, MC2: tl.constexpr, MC3: tl.constexpr,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    pay = tl.program_id(1)
    if pay == 0:
        _push_one(s0, d0, n_rows, row_off, pid, nprog, W0, WORLD, MC0, BLOCK)
    elif pay == 1:
        _push_one(s1, d1, n_rows, row_off, pid, nprog, W1, WORLD, MC1, BLOCK)
    elif pay == 2:
        _push_one(s2, d2, n_rows, row_off, pid, nprog, W2, WORLD, MC2, BLOCK)
    else:
        _push_one(s3, d3, n_rows, row_off, pid, nprog, W3, WORLD, MC3, BLOCK)


def push_payloads(payloads: list, world: int, n_rows: int, row_off: int, multicast: bool = False,
                  block: int = 512, max_ctas: int = 0) -> None:
    """One launch for up to four payloads: [(src [n_rows, ...] contiguous, peer base-address array address
    (buffer_ptrs_dev), multicast address (multicast_ptr, 0 if none), width_bytes)]. Rows of 16-byte multiples move
    as 16-byte chunks, through the multicast mapping when multicast is set and the buffer has one; other rows
    (the top-k) as 32-bit unicast words."""
    if n_rows == 0 or not payloads:
        return
    if len(payloads) > 4:
        raise ValueError("at most four payloads per launch")
    srcs, dsts, ws, mcs = [], [], [], []
    words = []  # the exact tensors whose addresses the kernel gets: referenced until the launch returns
    for src, ptrs_addr, mc_addr, width_bytes in payloads:
        s = _as_words(src)
        words.append(s)
        if s.shape[0] < n_rows or s.shape[1] * 4 != width_bytes:
            raise ValueError(f"payload rows of {s.shape[1] * 4} bytes into {width_bytes}-byte buffer rows")
        use_mc = bool(multicast and mc_addr and width_bytes % 16 == 0)
        # 16-byte rows: the kernel takes the source as an address (vector loads); others as a pointer
        srcs.append(s.data_ptr() if s.shape[1] % 4 == 0 else s)
        dsts.append(int(mc_addr) if use_mc else int(ptrs_addr))
        ws.append(s.shape[1])
        mcs.append(use_mc)
    while len(srcs) < 4:  # unused slots: never scheduled (grid dim 1 covers the real payloads only)
        srcs.append(srcs[0])
        dsts.append(dsts[0])
        ws.append(ws[0])
        mcs.append(mcs[0])
    units = max(n_rows * (w // 4 if w % 4 == 0 else w) for w in ws[:len(payloads)])
    nblk = triton.cdiv(units, block)
    grid = (min(nblk, max_ctas) if max_ctas > 0 else nblk, len(payloads))
    _push4_kernel[grid](srcs[0], dsts[0], srcs[1], dsts[1], srcs[2], dsts[2], srcs[3], dsts[3], n_rows, row_off,
                        W0=ws[0], W1=ws[1], W2=ws[2], W3=ws[3], WORLD=world, MC0=mcs[0], MC1=mcs[1], MC2=mcs[2],
                        MC3=mcs[3], BLOCK=block)


@triton.jit
def _bf16_pair(r):
    """(low bf16, high bf16) of a 32-bit word, as fp32."""
    r = r.to(tl.uint32, bitcast=True)
    return (r << 16).to(tl.float32, bitcast=True), ((r >> 16) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _pack_bf16_pair(lo, hi):
    lo_b = lo.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    hi_b = hi.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    return (lo_b | (hi_b << 16)).to(tl.int32, bitcast=True)


@triton.jit
def _finish_word(lo, hi, scale, shared_w, rows, n_valid, HAS_SHARED: tl.constexpr, HAS_VALID: tl.constexpr):
    lo = lo * scale
    hi = hi * scale
    if HAS_SHARED:
        s_lo, s_hi = _bf16_pair(shared_w)
        lo += s_lo
        hi += s_hi
    if HAS_VALID:
        keep = rows < n_valid
        lo = tl.where(keep, lo, 0.0)
        hi = tl.where(keep, hi, 0.0)
    return _pack_bf16_pair(lo, hi)


@triton.jit
def _pull_v_kernel(src, shared, out, valid, n_rows, row_off, scale, H: tl.constexpr, WORLD: tl.constexpr,
                   MC: tl.constexpr, HAS_SHARED: tl.constexpr, HAS_VALID: tl.constexpr, BLOCK: tl.constexpr):
    """out = scale * sum over GPUs of rows row_off.. of the bf16 partial buffer (+ shared), 8 bf16 per lane, flat
    over this GPU's contiguous rows; tail lanes clamp their loads to the last chunk and their stores are masked."""
    total = n_rows * (H // 8)
    for start in tl.range(tl.program_id(0) * BLOCK, total, tl.num_programs(0) * BLOCK):
        _pull_v_block(src, shared, out, valid, n_rows, row_off, scale, start, total, H, WORLD, MC, HAS_SHARED,
                      HAS_VALID, BLOCK)


@triton.jit
def _pull_v_block(src, shared, out, valid, n_rows, row_off, scale, start, total, H: tl.constexpr,
                  WORLD: tl.constexpr, MC: tl.constexpr, HAS_SHARED: tl.constexpr, HAS_VALID: tl.constexpr,
                  BLOCK: tl.constexpr):
    c0 = start + tl.arange(0, BLOCK)
    m = c0 < total
    c = tl.minimum(c0, total - 1).to(tl.int64)
    rows = c // (H // 8)
    boff = (c + row_off * (H // 8)) * 16
    if MC:
        _fence_alias_acquire()
        w0, w1, w2, w3 = _mc_ld_reduce_v4(src + boff)
        l0, h0 = _bf16_pair(w0)
        l1, h1 = _bf16_pair(w1)
        l2, h2 = _bf16_pair(w2)
        l3, h3 = _bf16_pair(w3)
    else:
        l0 = tl.zeros((BLOCK,), dtype=tl.float32)
        h0 = tl.zeros((BLOCK,), dtype=tl.float32)
        l1 = tl.zeros((BLOCK,), dtype=tl.float32)
        h1 = tl.zeros((BLOCK,), dtype=tl.float32)
        l2 = tl.zeros((BLOCK,), dtype=tl.float32)
        h2 = tl.zeros((BLOCK,), dtype=tl.float32)
        l3 = tl.zeros((BLOCK,), dtype=tl.float32)
        h3 = tl.zeros((BLOCK,), dtype=tl.float32)
        ptrs = src.to(tl.pointer_type(tl.uint64))
        for p in tl.static_range(WORLD):                   # fixed peer order: identical sums on every rank
            base = tl.load(ptrs + p).to(tl.int64)
            w0, w1, w2, w3 = _ld_v4(base + boff)
            a, b = _bf16_pair(w0)
            l0 += a
            h0 += b
            a, b = _bf16_pair(w1)
            l1 += a
            h1 += b
            a, b = _bf16_pair(w2)
            l2 += a
            h2 += b
            a, b = _bf16_pair(w3)
            l3 += a
            h3 += b
    n_valid = n_rows
    if HAS_VALID:
        n_valid = tl.load(valid)
    wo = c * 4                                                   # word offsets in out / shared (contiguous rows)
    s0 = tl.zeros((BLOCK,), dtype=tl.int32)
    s1 = s0
    s2 = s0
    s3 = s0
    if HAS_SHARED:
        s0 = tl.load(shared + wo, mask=m, other=0)
        s1 = tl.load(shared + wo + 1, mask=m, other=0)
        s2 = tl.load(shared + wo + 2, mask=m, other=0)
        s3 = tl.load(shared + wo + 3, mask=m, other=0)
    o0 = _finish_word(l0, h0, scale, s0, rows, n_valid, HAS_SHARED, HAS_VALID)
    o1 = _finish_word(l1, h1, scale, s1, rows, n_valid, HAS_SHARED, HAS_VALID)
    o2 = _finish_word(l2, h2, scale, s2, rows, n_valid, HAS_SHARED, HAS_VALID)
    o3 = _finish_word(l3, h3, scale, s3, rows, n_valid, HAS_SHARED, HAS_VALID)
    tl.store(out + wo, o0, mask=m)
    tl.store(out + wo + 1, o1, mask=m)
    tl.store(out + wo + 2, o2, mask=m)
    tl.store(out + wo + 3, o3, mask=m)


def pull_reduce_v(src: int, world: int, row_off: int, n_rows: int, out: torch.Tensor, scale: float = 1.0,
                  shared: torch.Tensor | None = None, valid: torch.Tensor | None = None, multicast: bool = False,
                  block: int = 256, max_ctas: int = 0) -> torch.Tensor:
    """pull_reduce_rows with 16-byte peer loads. src: the bf16 partial buffer's peer base-address array (unicast)
    or its multicast_ptr (the switch sums; one read per chunk). out / shared: contiguous bf16 [>= n_rows, H]."""
    if n_rows == 0:
        return out
    h = out.shape[1]
    if h % 8 or not out.is_contiguous() or out.dtype != torch.bfloat16:
        raise ValueError("out must be contiguous bf16 with H a multiple of 8")
    if shared is not None and (not shared.is_contiguous() or shared.dtype != torch.bfloat16 or shared.shape[1] != h):
        raise ValueError("shared must be contiguous bf16 [>= n_rows, H]")
    ow = out.view(torch.int32)
    sw = shared.view(torch.int32) if shared is not None else ow
    nblk = triton.cdiv(n_rows * (h // 8), block)
    grid = (min(nblk, max_ctas) if max_ctas > 0 else nblk,)
    _pull_v_kernel[grid](int(src), sw, ow, valid if valid is not None else ow, n_rows, row_off, float(scale), H=h,
                         WORLD=world, MC=multicast, HAS_SHARED=shared is not None, HAS_VALID=valid is not None,
                         BLOCK=block)
    return out
