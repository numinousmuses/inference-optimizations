"""Quantized (int8 / fp8) two-shot all-reduce for PCIe hosts without NVLink (RTX PRO 6000 x 4 / x 8), agent rtxv41.

Why (research/rtx-flash/v41.md section 5.7, modeled): DeepSeek-V4.1-Flash on 8 x RTX spends ~29 of ~58 ms per decode
step in 86 bf16 all-reduces of rows x 5120 (FlashInfer PCIe-IPC kernels, bf16/fp16 only). Carrying e4m3 + one fp32
scale per 128 values moves 1.03 B per element instead of 2 B, -14.6 ms/step at 64 users x 6 DSpark rows.

Algorithm (two-shot over IPC / symmetric memory, W ranks, input x[N] bf16, shard S = N / W):
  P1 reduce-scatter push   rank r quantizes shard d of x (d != r) and writes it into peer d's receive slot r;
                           stage s = 1 .. W-1 targets peer (r + s) % W, so each rank has one outbound and one inbound
                           peer at a time (the staging FlashInfer's PCIe kernels use against fabric collapse).
  barrier A
  P2 reduce + push         rank r: acc = x[shard r] (unquantized) + sum_{j != r} dequant(recv slot j), fp32, fixed
                           order j = 0 .. W-1; quantize acc ONCE and write it into every rank's gather slot r,
                           its own included (stage s = 0 .. W-1 -> (r + s) % W).
  barrier B
  P3 gather                out[shard j] = dequant(gather slot j) for all j.
Every rank dequantizes the same bytes for every shard, so all ranks hold BITWISE-IDENTICAL outputs (required: each
rank recomputes MoE routing from the reduced hidden state; a one-ulp split between ranks would mis-route experts).

Payload modes (SGLANG_RTX_QUANT_ALLREDUCE):
  q8      int8 both shots, one fp32 scale per 128 (1.03 B/elem). DEFAULT: symmetric int8 over a 128-group has a
          uniform step amax/127, ~10x lower error than e4m3's 3-bit mantissa on activation-like groups.
  q8_rs   int8 reduce-scatter, bf16 all-gather ((1.03 + 2) / 2 B/elem, one quantization).
  fp8     e4m3 both shots; fp8_rs e4m3 reduce-scatter + bf16 gather (kept for comparison).
  Measured on CPU (Triton interpreter, W=4, Gaussian bf16 inputs, relative L2 error of the sum): bf16 AR 0.17%,
  q8 0.90%, q8_rs 0.63%, fp8 3.55%, fp8_rs 2.26% (test_fp8_pcie_allreduce.py).

Quantization (identical in Triton and in the torch reference `reference_all_reduce`, bitwise):
  amax over 128 values, a = amax if amax > 0 else 1, inv = QMAX / a (div_rn; QMAX 127 or 448),
  q = round_half_even(clamp(x * inv)) (int8) or e4m3_rn(clamp(x * inv, +-448)), scale = a / QMAX (div_rn),
  dequant = fp32(q) * scale; bf16 outputs round half-even.

Buffers per rank (one symmetric allocation, byte offsets fixed by max_numel):
  RQ recv payload [W, S_max] e4m3 | RS recv scales [W, S_max / 128] fp32 | GQ gather payload [W, S_max] e4m3 or bf16 |
  GS gather scales [W, S_max / 128] fp32.
Two barriers per call make buffer reuse across calls safe (a rank's next P1 writes peers' RQ only after every rank
passed barrier B, i.e. finished reading RQ in P2; a peer's next P2 writes our GQ only after we passed its barrier A,
i.e. finished P3).
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

GROUP = 128
FP8_MAX = 448.0
BLOCK = 1024  # elements per program (8 scale groups)


@triton.jit
def _e4m3_encode_manual(y):
    """fp32 (|y| <= 448) -> e4m3fn code (uint8), round-to-nearest-even; bit-exact with torch's cast. Integer path:
    the Triton interpreter's float8 conversion is wrong at exponent boundaries (e.g. 124.33 -> 64, found here)."""
    bits = y.to(tl.uint32, bitcast=True)
    sign = (bits >> 31) & 1
    a = tl.abs(y)
    abits = a.to(tl.uint32, bitcast=True)
    E = ((abits >> 23) & 0xFF).to(tl.int32) - 127
    m = abits & 0x7FFFFF
    keep = m >> 20
    rem = m & 0xFFFFF
    up = (rem > 0x80000) | ((rem == 0x80000) & ((keep & 1) == 1))
    normal = ((E + 7) << 3).to(tl.uint32) + keep + up.to(tl.uint32)
    t = a * 512.0
    fl = tl.floor(t)
    r = t - fl
    fli = fl.to(tl.uint32)
    sub_up = (r > 0.5) | ((r == 0.5) & ((fli & 1) == 1))
    sub = fli + sub_up.to(tl.uint32)
    code = tl.where(E < -6, sub, normal)
    code = tl.minimum(code, 126)
    return (code | (sign << 7)).to(tl.uint8)


@triton.jit
def _e4m3_decode_manual(code):
    c = code.to(tl.uint32)
    e = (c >> 3) & 15
    m = (c & 7).to(tl.float32)
    mag = tl.where(e == 0, m * 0.001953125, (1.0 + m * 0.125) * tl.exp2(e.to(tl.float32) - 7.0))
    return tl.where((c >> 7) == 1, -mag, mag)


@triton.jit
def _to_bf16(x, NATIVE: tl.constexpr):
    """fp32 -> bf16, round-to-nearest-even. The interpreter's cast truncates (found here), so under it the rounding is
    done on the fp32 bits first and the cast only drops zero bits."""
    if NATIVE:
        return x.to(tl.bfloat16)
    b = x.to(tl.uint32, bitcast=True)
    r = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return r.to(tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def _rne(y):
    fl = tl.floor(y)
    r = y - fl
    odd = (fl - 2.0 * tl.floor(fl * 0.5)) == 1.0
    return tl.where((r > 0.5) | ((r == 0.5) & odd), fl + 1.0, fl)


@triton.jit
def _quant(x, GROUPS: tl.constexpr, GS: tl.constexpr, NATIVE: tl.constexpr, INT8: tl.constexpr,
           OUTLIER: tl.constexpr = False):
    """[GROUPS * GS] fp32 -> (uint8 codes, fp32 scale, fp32 outlier value, fp32 outlier index) per group.
    INT8: symmetric int8, scale = amax / 127; else e4m3, scale = amax / 448. Both round-to-nearest-even, bit-exact
    with _quant_ref. OUTLIER: the group's largest-|x| element (first on ties) is carried exactly in fp32 and zeroed
    before the scale is taken, so one outlier channel no longer flattens the other 127 values to zero."""
    x2 = tl.reshape(x, (GROUPS, GS))
    cols = tl.arange(0, GS)[None, :]
    if OUTLIER:
        idx = tl.argmax(tl.abs(x2), axis=1, tie_break_left=True)
        hit = cols == idx[:, None]
        top = tl.sum(tl.where(hit, x2, 0.0), axis=1)
        x2 = tl.where(hit, 0.0, x2)
        idxf = idx.to(tl.float32)
    else:
        top = tl.zeros((GROUPS,), tl.float32)
        idxf = tl.zeros((GROUPS,), tl.float32)
    amax = tl.max(tl.abs(x2), axis=1)
    a = tl.where(amax > 0, amax, 1.0)
    if INT8:
        inv = tl.div_rn(tl.full((GROUPS,), 127.0, tl.float32), a)
        y = tl.minimum(tl.maximum(_rne(x2 * inv[:, None]), -127.0), 127.0)
        q = y.to(tl.int32).to(tl.int8).to(tl.uint8, bitcast=True)
        scale = tl.div_rn(a, tl.full((GROUPS,), 127.0, tl.float32))
    else:
        inv = tl.div_rn(tl.full((GROUPS,), 448.0, tl.float32), a)
        y = tl.minimum(tl.maximum(x2 * inv[:, None], -448.0), 448.0)
        if NATIVE:
            q = y.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        else:
            q = _e4m3_encode_manual(y)
        scale = tl.div_rn(a, tl.full((GROUPS,), 448.0, tl.float32))
    return tl.reshape(q, (GROUPS * GS,)), scale, top, idxf


@triton.jit
def _dequant(q, scale, top, idxf, GROUPS: tl.constexpr, GS: tl.constexpr, NATIVE: tl.constexpr, INT8: tl.constexpr,
             OUTLIER: tl.constexpr = False):
    if INT8:
        f = q.to(tl.int8, bitcast=True).to(tl.float32)
    elif NATIVE:
        f = q.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        f = _e4m3_decode_manual(q)
    v = tl.reshape(f, (GROUPS, GS)) * scale[:, None]
    if OUTLIER:
        cols = tl.arange(0, GS)[None, :]
        v = tl.where(cols == idxf.to(tl.int32)[:, None], top[:, None], v)
    return tl.reshape(v, (GROUPS * GS,))


@triton.jit
def _store_meta(ptr, goffs, gmask, sc, top, idxf, OUTLIER: tl.constexpr):
    """Per-group metadata: [scale] or, with OUTLIER, [scale, outlier value, outlier index] (fp32 words)."""
    if OUTLIER:
        tl.store(ptr + goffs * 3, sc, mask=gmask)
        tl.store(ptr + goffs * 3 + 1, top, mask=gmask)
        tl.store(ptr + goffs * 3 + 2, idxf, mask=gmask)
    else:
        tl.store(ptr + goffs, sc, mask=gmask)


@triton.jit
def _load_meta(ptr, goffs, gmask, OUTLIER: tl.constexpr):
    if OUTLIER:
        sc = tl.load(ptr + goffs * 3, mask=gmask, other=0.0)
        top = tl.load(ptr + goffs * 3 + 1, mask=gmask, other=0.0)
        idxf = tl.load(ptr + goffs * 3 + 2, mask=gmask, other=0.0)
    else:
        sc = tl.load(ptr + goffs, mask=gmask, other=0.0)
        top = sc * 0.0
        idxf = sc * 0.0
    return sc, top, idxf


@triton.jit
def _rs_push_kernel(x_ptr, peer_base, rank, S, S_max, OFF_RQ, OFF_RS,
                    W: tl.constexpr, BLOCK: tl.constexpr, GS: tl.constexpr, NATIVE: tl.constexpr, INT8: tl.constexpr,
                    OUTLIER: tl.constexpr = False):
    SC: tl.constexpr = 3 if OUTLIER else 1
    GROUPS: tl.constexpr = BLOCK // GS
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    goffs = pid * GROUPS + tl.arange(0, GROUPS)
    gmask = goffs < S // GS
    for s in tl.static_range(1, W):
        d = (rank + s) % W
        x = tl.load(x_ptr + d * S + offs, mask=mask, other=0.0).to(tl.float32)
        q, sc, top, idxf = _quant(x, GROUPS, GS, NATIVE, INT8, OUTLIER)
        base = tl.load(peer_base + d)
        tl.store((base + OFF_RQ).to(tl.pointer_type(tl.uint8)) + rank * S_max + offs, q, mask=mask)
        _store_meta((base + OFF_RS).to(tl.pointer_type(tl.float32)) + rank * (S_max // GS) * SC, goffs, gmask,
                    sc, top, idxf, OUTLIER)


@triton.jit
def _reduce_push_kernel(x_ptr, peer_base, rank, S, S_max, OFF_RQ, OFF_RS, OFF_GQ, OFF_GS,
                        W: tl.constexpr, BLOCK: tl.constexpr, GS: tl.constexpr, GATHER_FP8: tl.constexpr, NATIVE: tl.constexpr, INT8: tl.constexpr,
                        OUTLIER: tl.constexpr = False):
    SC: tl.constexpr = 3 if OUTLIER else 1
    GROUPS: tl.constexpr = BLOCK // GS
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    goffs = pid * GROUPS + tl.arange(0, GROUPS)
    gmask = goffs < S // GS
    acc = tl.load(x_ptr + rank * S + offs, mask=mask, other=0.0).to(tl.float32)
    my = tl.load(peer_base + rank)
    rq = (my + OFF_RQ).to(tl.pointer_type(tl.uint8))
    rs = (my + OFF_RS).to(tl.pointer_type(tl.float32))
    for j in tl.static_range(W):
        m = mask & (j != rank)
        qj = tl.load(rq + j * S_max + offs, mask=m, other=0)
        sj, tj, ij = _load_meta(rs + j * (S_max // GS) * SC, goffs, gmask & (j != rank), OUTLIER)
        acc += _dequant(qj, sj, tj, ij, GROUPS, GS, NATIVE, INT8, OUTLIER)
    if GATHER_FP8:
        q, sc, top, idxf = _quant(acc, GROUPS, GS, NATIVE, INT8, OUTLIER)
        for s in tl.static_range(W):
            d = (rank + s) % W
            base = tl.load(peer_base + d)
            tl.store((base + OFF_GQ).to(tl.pointer_type(tl.uint8)) + rank * S_max + offs, q, mask=mask)
            _store_meta((base + OFF_GS).to(tl.pointer_type(tl.float32)) + rank * (S_max // GS) * SC, goffs, gmask,
                        sc, top, idxf, OUTLIER)
    else:
        y = _to_bf16(acc, NATIVE)
        for s in tl.static_range(W):
            d = (rank + s) % W
            base = tl.load(peer_base + d)
            tl.store((base + OFF_GQ).to(tl.pointer_type(tl.bfloat16)) + rank * S_max + offs, y, mask=mask)


@triton.jit
def _gather_kernel(out_ptr, peer_base, rank, S, S_max, OFF_GQ, OFF_GS,
                   BLOCK: tl.constexpr, GS: tl.constexpr, GATHER_FP8: tl.constexpr, NATIVE: tl.constexpr,
                   INT8: tl.constexpr, OUTLIER: tl.constexpr = False):
    SC: tl.constexpr = 3 if OUTLIER else 1
    GROUPS: tl.constexpr = BLOCK // GS
    j = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    goffs = pid * GROUPS + tl.arange(0, GROUPS)
    my = tl.load(peer_base + rank)
    if GATHER_FP8:
        q = tl.load((my + OFF_GQ).to(tl.pointer_type(tl.uint8)) + j * S_max + offs, mask=mask, other=0)
        sc, top, idxf = _load_meta((my + OFF_GS).to(tl.pointer_type(tl.float32)) + j * (S_max // GS) * SC, goffs,
                                   goffs < S // GS, OUTLIER)
        y = _to_bf16(_dequant(q, sc, top, idxf, GROUPS, GS, NATIVE, INT8, OUTLIER), NATIVE)
    else:
        y = tl.load((my + OFF_GQ).to(tl.pointer_type(tl.bfloat16)) + j * S_max + offs, mask=mask, other=0.0)
    tl.store(out_ptr + j * S + offs, y, mask=mask)


@triton.jit
def _quant_local_kernel(x_ptr, peer_base, rank, S, S_max, OFF_RQ, OFF_RS,
                        W: tl.constexpr, BLOCK: tl.constexpr, GS: tl.constexpr, NATIVE: tl.constexpr,
                        INT8: tl.constexpr, OUTLIER: tl.constexpr):
    """PULL P1: quantize every peer's shard of x into OUR OWN receive region, slot d = shard d (local writes)."""
    SC: tl.constexpr = 3 if OUTLIER else 1
    GROUPS: tl.constexpr = BLOCK // GS
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    goffs = pid * GROUPS + tl.arange(0, GROUPS)
    gmask = goffs < S // GS
    my = tl.load(peer_base + rank)
    for s in tl.static_range(1, W):
        d = (rank + s) % W
        x = tl.load(x_ptr + d * S + offs, mask=mask, other=0.0).to(tl.float32)
        q, sc, top, idxf = _quant(x, GROUPS, GS, NATIVE, INT8, OUTLIER)
        tl.store((my + OFF_RQ).to(tl.pointer_type(tl.uint8)) + d * S_max + offs, q, mask=mask)
        _store_meta((my + OFF_RS).to(tl.pointer_type(tl.float32)) + d * (S_max // GS) * SC, goffs, gmask,
                    sc, top, idxf, OUTLIER)


@triton.jit
def _reduce_pull_kernel(x_ptr, peer_base, rank, S, S_max, OFF_RQ, OFF_RS, OFF_GQ, OFF_GS,
                        W: tl.constexpr, BLOCK: tl.constexpr, GS: tl.constexpr, GATHER_FP8: tl.constexpr,
                        NATIVE: tl.constexpr, INT8: tl.constexpr, OUTLIER: tl.constexpr):
    """PULL P2: acc = x[shard r] + sum_j dequant(peer j's slot r) in the fixed order j = 0..W-1 (remote reads, the
    same bytes and order as the push variant -> bitwise-identical results); the reduced shard is quantized ONCE into
    our own gather region slot r (local write)."""
    SC: tl.constexpr = 3 if OUTLIER else 1
    GROUPS: tl.constexpr = BLOCK // GS
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    goffs = pid * GROUPS + tl.arange(0, GROUPS)
    gmask = goffs < S // GS
    acc = tl.load(x_ptr + rank * S + offs, mask=mask, other=0.0).to(tl.float32)
    for j in tl.static_range(W):
        base = tl.load(peer_base + j)
        m = mask & (j != rank)
        qj = tl.load((base + OFF_RQ).to(tl.pointer_type(tl.uint8)) + rank * S_max + offs, mask=m, other=0)
        sj, tj, ij = _load_meta((base + OFF_RS).to(tl.pointer_type(tl.float32)) + rank * (S_max // GS) * SC, goffs,
                                gmask & (j != rank), OUTLIER)
        acc += _dequant(qj, sj, tj, ij, GROUPS, GS, NATIVE, INT8, OUTLIER)
    my = tl.load(peer_base + rank)
    if GATHER_FP8:
        q, sc, top, idxf = _quant(acc, GROUPS, GS, NATIVE, INT8, OUTLIER)
        tl.store((my + OFF_GQ).to(tl.pointer_type(tl.uint8)) + rank * S_max + offs, q, mask=mask)
        _store_meta((my + OFF_GS).to(tl.pointer_type(tl.float32)) + rank * (S_max // GS) * SC, goffs, gmask,
                    sc, top, idxf, OUTLIER)
    else:
        tl.store((my + OFF_GQ).to(tl.pointer_type(tl.bfloat16)) + rank * S_max + offs, _to_bf16(acc, NATIVE),
                 mask=mask)


@triton.jit
def _gather_pull_kernel(out_ptr, peer_base, rank, S, S_max, OFF_GQ, OFF_GS,
                        BLOCK: tl.constexpr, GS: tl.constexpr, GATHER_FP8: tl.constexpr, NATIVE: tl.constexpr,
                        INT8: tl.constexpr, OUTLIER: tl.constexpr):
    """PULL P3: out[shard j] = dequant(peer j's gather slot j) (remote reads; our own shard from local memory)."""
    SC: tl.constexpr = 3 if OUTLIER else 1
    GROUPS: tl.constexpr = BLOCK // GS
    j = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    goffs = pid * GROUPS + tl.arange(0, GROUPS)
    base = tl.load(peer_base + j)
    if GATHER_FP8:
        q = tl.load((base + OFF_GQ).to(tl.pointer_type(tl.uint8)) + j * S_max + offs, mask=mask, other=0)
        sc, top, idxf = _load_meta((base + OFF_GS).to(tl.pointer_type(tl.float32)) + j * (S_max // GS) * SC, goffs,
                                   goffs < S // GS, OUTLIER)
        y = _to_bf16(_dequant(q, sc, top, idxf, GROUPS, GS, NATIVE, INT8, OUTLIER), NATIVE)
    else:
        y = tl.load((base + OFF_GQ).to(tl.pointer_type(tl.bfloat16)) + j * S_max + offs, mask=mask, other=0.0)
    tl.store(out_ptr + j * S + offs, y, mask=mask)


MODES = ("q8", "q8_rs", "fp8", "fp8_rs", "q8o", "q8o_rs")


def pull_enabled() -> bool:
    """SGLANG_RTX_QUANT_AR_PULL=1: P2P reads instead of writes (same bytes, same results bitwise). The 8 x RTX NCCL
    probe measured reads 1.70x faster than the default at prefill sizes (P2P_READ_ENABLE, 2026-10-01)."""
    return os.environ.get("SGLANG_RTX_QUANT_AR_PULL", "0") == "1"


def _gather_quantized(mode: str) -> bool:
    return mode in ("q8", "fp8", "q8o")


def _outlier(mode: str) -> bool:
    """q8o / q8o_rs: int8 with each group's largest-|x| element carried exactly (fp32 value + index, 12 B of
    metadata per 128 values instead of 4: 1.09 B/elem). The 8 x RTX accuracy gate failed plain q8 (teacher-forced
    KL 0.051 vs 0.01 max, 2026-10-01): DeepSeek hidden states have a few huge channels, and a per-128 scale set by one
    of them zeroes the rest of its group (synthetic: 19% -> 0.8% error on the non-outlier channels)."""
    return mode.startswith("q8o")


def _int8(mode: str) -> bool:
    return mode.startswith("q8")


def buffer_layout(world: int, max_numel: int, mode: str = "q8") -> dict:
    """Byte offsets of the four regions in one rank's buffer (each region 256-byte aligned)."""
    assert max_numel % (world * GROUP) == 0, "max_numel must be a multiple of world x 128"
    s_max = max_numel // world

    def al(n):
        return -(-n // 256) * 256

    sc = 3 if _outlier(mode) else 1
    rq = 0
    rs = rq + al(world * s_max)
    gq = rs + al(world * (s_max // GROUP) * 4 * sc)
    gs = gq + al(world * s_max * (1 if _gather_quantized(mode) else 2))
    total = gs + al(world * (s_max // GROUP) * 4 * sc)
    return {"S_max": s_max, "RQ": rq, "RS": rs, "GQ": gq, "GS": gs, "bytes": total}


class Fp8AllReduceRank:
    """One rank's view: the phase launches for a given peer-buffer pointer table (symmetric memory or, in tests,
    W buffers of one process)."""

    def __init__(self, rank: int, world: int, peer_base: torch.Tensor, max_numel: int, mode: str = "q8",
                 native_fp8: Optional[bool] = None, pull: Optional[bool] = None):
        assert mode in MODES, mode
        self.pull = pull_enabled() if pull is None else pull
        # native cvt on the GPU; the integer encoder under the interpreter (bit-identical by construction and test)
        self.native = (os.environ.get("TRITON_INTERPRET", "0") != "1") if native_fp8 is None else native_fp8
        self.rank, self.world, self.mode = rank, world, mode
        self.peer_base = peer_base  # [W] int64 byte addresses, on the device the kernels run on
        self.lay = buffer_layout(world, max_numel, mode)
        self.max_numel = max_numel

    def supports(self, inp: torch.Tensor) -> bool:
        return (
            inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and inp.numel() <= self.max_numel
            and inp.numel() % (self.world * GROUP) == 0
        )

    def _grid(self, S):
        return (triton.cdiv(S, BLOCK),)

    def phase1(self, x: torch.Tensor):
        S = x.numel() // self.world
        L = self.lay
        if self.pull:
            _quant_local_kernel[self._grid(S)](x, self.peer_base, self.rank, S, L["S_max"], L["RQ"], L["RS"],
                                               W=self.world, BLOCK=BLOCK, GS=GROUP, NATIVE=self.native,
                                               INT8=_int8(self.mode), OUTLIER=_outlier(self.mode))
            return
        _rs_push_kernel[self._grid(S)](x, self.peer_base, self.rank, S, L["S_max"], L["RQ"], L["RS"],
                                       W=self.world, BLOCK=BLOCK, GS=GROUP, NATIVE=self.native,
                                       INT8=_int8(self.mode), OUTLIER=_outlier(self.mode))

    def phase2(self, x: torch.Tensor):
        S = x.numel() // self.world
        L = self.lay
        if self.pull:
            _reduce_pull_kernel[self._grid(S)](x, self.peer_base, self.rank, S, L["S_max"], L["RQ"], L["RS"],
                                               L["GQ"], L["GS"], W=self.world, BLOCK=BLOCK, GS=GROUP,
                                               GATHER_FP8=_gather_quantized(self.mode), NATIVE=self.native,
                                               INT8=_int8(self.mode), OUTLIER=_outlier(self.mode))
            return
        _reduce_push_kernel[self._grid(S)](x, self.peer_base, self.rank, S, L["S_max"], L["RQ"], L["RS"], L["GQ"],
                                           L["GS"], W=self.world, BLOCK=BLOCK, GS=GROUP,
                                           GATHER_FP8=_gather_quantized(self.mode), NATIVE=self.native,
                                           INT8=_int8(self.mode), OUTLIER=_outlier(self.mode))

    def phase3(self, out: torch.Tensor):
        S = out.numel() // self.world
        L = self.lay
        if self.pull:
            _gather_pull_kernel[(self.world, triton.cdiv(S, BLOCK))](out, self.peer_base, self.rank, S, L["S_max"],
                                                                     L["GQ"], L["GS"], BLOCK=BLOCK, GS=GROUP,
                                                                     GATHER_FP8=_gather_quantized(self.mode),
                                                                     NATIVE=self.native, INT8=_int8(self.mode),
                                                                     OUTLIER=_outlier(self.mode))
            return
        _gather_kernel[(self.world, triton.cdiv(S, BLOCK))](out, self.peer_base, self.rank, S, L["S_max"], L["GQ"],
                                                            L["GS"], BLOCK=BLOCK, GS=GROUP,
                                                            GATHER_FP8=_gather_quantized(self.mode),
                                                            NATIVE=self.native, INT8=_int8(self.mode),
                                                            OUTLIER=_outlier(self.mode))


class Fp8PcieAllReduce:
    """Multi-process all-reduce on torch symmetric memory (CUDA IPC mappings; P2P over PCIe).

    Collective: every rank constructs it with the same arguments in the same order. all_reduce is out-of-place,
    CUDA-graph capturable (persistent buffers, static pointer table, symm_mem device barriers)."""

    def __init__(self, group, device: torch.device, max_numel: int, mode: str = "q8", pull: Optional[bool] = None):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.mode = mode
        lay = buffer_layout(self.world, max_numel, mode)
        self.buf = symm_mem.empty(lay["bytes"], dtype=torch.uint8, device=device)
        self.handle = symm_mem.rendezvous(self.buf, group)
        peer_base = torch.tensor(list(self.handle.buffer_ptrs), dtype=torch.int64, device=device)
        self.r = Fp8AllReduceRank(self.rank, self.world, peer_base, max_numel, mode, pull=pull)

    def supports(self, inp: torch.Tensor) -> bool:
        return self.r.supports(inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(inp)
        x = inp.reshape(-1)
        self.r.phase1(x)
        self.handle.barrier(channel=0)
        self.r.phase2(x)
        self.handle.barrier(channel=1)
        self.r.phase3(out.reshape(-1))
        return out


# ------------------------------------------------------------------------------------------ single-process simulation


class SimGroup:
    """W simulated ranks in one process sharing persistent buffers (like the symmetric allocation): the three phases
    run rank by rank, which is exactly the ordering the two device barriers enforce. For the CPU (Triton interpreter)
    tests and the single-RTX numerics check."""

    def __init__(self, world: int, max_numel: int, mode: str = "q8", device="cpu", native_fp8: Optional[bool] = None,
                 pull: Optional[bool] = None):
        lay = buffer_layout(world, max_numel, mode)
        self.bufs = [torch.zeros(lay["bytes"], dtype=torch.uint8, device=device) for _ in range(world)]
        peer_base = torch.tensor([b.data_ptr() for b in self.bufs], dtype=torch.int64, device=device)
        self.ranks = [Fp8AllReduceRank(r, world, peer_base, max_numel, mode, native_fp8, pull) for r in range(world)]

    def all_reduce(self, xs: Sequence[torch.Tensor]):
        flat = [x.reshape(-1).contiguous() for x in xs]
        assert all(r.supports(f) for r, f in zip(self.ranks, flat))
        for r in self.ranks:
            r.phase1(flat[r.rank])
        for r in self.ranks:
            r.phase2(flat[r.rank])
        outs = [torch.empty_like(f) for f in flat]
        for r in self.ranks:
            r.phase3(outs[r.rank])
        return [o.view_as(x) for o, x in zip(outs, xs)]


def simulate_all_reduce(xs: Sequence[torch.Tensor], mode: str = "q8", max_numel: Optional[int] = None,
                        native_fp8: Optional[bool] = None, pull: Optional[bool] = None):
    g = SimGroup(len(xs), max_numel or xs[0].numel(), mode, xs[0].device, native_fp8, pull)
    return g.all_reduce(xs), g.bufs


def _quant_ref(x: torch.Tensor, int8: bool = True, outlier: bool = False):
    g = x.float().reshape(-1, GROUP).clone()
    top = idx = None
    if outlier:
        idx = g.abs().argmax(-1)  # first maximal index, as tl.argmax(tie_break_left=True)
        top = g.gather(-1, idx[:, None])[:, 0].clone()
        g.scatter_(-1, idx[:, None], 0.0)
    amax = g.abs().amax(-1)
    a = torch.where(amax > 0, amax, torch.ones_like(amax))
    qmax = 127.0 if int8 else FP8_MAX
    inv = torch.full_like(a, qmax) / a
    y = g * inv[:, None]
    if int8:
        q = torch.round(y).clamp(-127, 127).to(torch.int8)  # torch.round is half-to-even
    else:
        q = y.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if outlier:
        return q.reshape(-1), (a / qmax, top, idx)
    return q.reshape(-1), a / qmax


def _dequant_ref(q: torch.Tensor, scale):
    if isinstance(scale, tuple):
        sc, top, idx = scale
        v = q.float().reshape(-1, GROUP) * sc[:, None]
        v.scatter_(-1, idx[:, None], top[:, None])
        return v.reshape(-1)
    return (q.float().reshape(-1, GROUP) * scale[:, None]).reshape(-1)


def reference_all_reduce(xs: Sequence[torch.Tensor], mode: str = "q8") -> torch.Tensor:
    """Torch model of the kernel's exact arithmetic (bitwise target)."""
    W = len(xs)
    int8 = _int8(mode)
    ol = _outlier(mode)
    flat = [x.reshape(-1).float() for x in xs]
    S = flat[0].numel() // W
    out = torch.empty(flat[0].numel(), dtype=torch.bfloat16)
    for r in range(W):
        acc = flat[r][r * S:(r + 1) * S].clone()
        for j in range(W):
            if j == r:
                continue
            q, s = _quant_ref(flat[j][r * S:(r + 1) * S], int8, ol)
            acc = acc + _dequant_ref(q, s)
        if _gather_quantized(mode):
            q, s = _quant_ref(acc, int8, ol)
            out[r * S:(r + 1) * S] = _dequant_ref(q, s).to(torch.bfloat16)
        else:
            out[r * S:(r + 1) * S] = acc.to(torch.bfloat16)
    return out.view_as(xs[0])


def enabled_mode() -> Optional[str]:
    """SGLANG_RTX_QUANT_ALLREDUCE = q8 | q8_rs | fp8 | fp8_rs | q8o | q8o_rs (unset: off)."""
    m = os.environ.get("SGLANG_RTX_QUANT_ALLREDUCE", "").strip().lower()
    return m if m in MODES else None


class StandaloneComm:
    """GroupCoordinator adapter for the pcie_ipc_comm slot when FlashInfer has no PCIe-IPC communicator (FlashInfer
    < 0.7.0, e.g. the cae69be5 nightly's 0.6.18): the quantized all-reduce then serves the same decode-sized TP
    reductions on its own. Built lazily on the first eligible tensor (all ranks issue the same reductions in the same
    order, so the symmetric-memory rendezvous is collective-safe), sized for the widest captured decode batch."""

    def __init__(self, group, device, mode: str):
        self.group, self.device, self.mode = group, device, mode
        self.disabled = False
        self.max_numel = 0
        self._ar: Optional[Fp8PcieAllReduce] = None
        self._failed = False

    def _ensure(self, inp: torch.Tensor) -> bool:
        return self._build(inp.shape[-1])

    def _build(self, hidden: int, max_rows: Optional[int] = None) -> bool:
        if self._ar is not None:
            return True
        if self._failed:
            return False
        import logging

        import torch.distributed as dist

        try:
            from sglang.srt.distributed.device_communicators.pcie_ipc_ar import _decode_width

            width = _decode_width() or 256
        except Exception:  # noqa: BLE001
            width = 256
        width = max(width, int(max_rows or 0))
        # Measured on 8 x RTX (profile 2026-10-01): an 8K-token prefill chunk spends ~0.62 s of ~0.94 s in 83 NCCL
        # all-reduces of [8192, 5120] bf16 (RING_LL, ~11 GB/s). Size the workspace for the prefill chunk as well
        # (SGLANG_RTX_QUANT_AR_PREFILL=0 keeps it decode-only): ~2 x 1.03 B x chunk x hidden per rank (86 MB at 8K).
        if os.environ.get("SGLANG_RTX_QUANT_AR_PREFILL", "1") != "0":
            try:
                from sglang.srt.runtime_context import get_schedule

                width = max(width, int(get_schedule().chunked_prefill_size or 0))
            except Exception:  # noqa: BLE001
                pass
        world = dist.get_world_size(self.group)
        cap = width * hidden
        cap -= cap % (world * GROUP)
        try:
            self._ar = Fp8PcieAllReduce(self.group, self.device, cap, self.mode)
            self.max_numel = cap
            logging.getLogger(__name__).info(
                "numinous quantized all-reduce (standalone) enabled: mode=%s world=%d max_numel=%d", self.mode, world, cap)
            return True
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).warning("numinous quantized all-reduce disabled: %s", e)
            self._failed = True
            self.disabled = True
            return False

    def should_pcie_ipc_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled or inp.dtype != torch.bfloat16 or not inp.is_contiguous() or inp.dim() < 2:
            return False
        if not self._ensure(inp):
            return False
        return self._ar.supports(inp)

    def pcie_ipc_all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return self._ar.all_reduce(inp)

    # --- the rest of PcieIpcCommunicator's interface that SGLang calls (cae69be5):
    # BaseRunner._pre_initialize_pcie_ipc_workspace -> prepare(hidden=, max_rows=) before graph capture, and
    # GroupCoordinator.destroy -> destroy(). Building in prepare() keeps the symmetric-memory rendezvous out of
    # graph capture and autotune contexts (collective: every rank runs prepare at the same point).
    def prepare(self, *, hidden: int, max_rows: Optional[int] = None) -> None:
        if self.disabled:
            return
        self._build(hidden, max_rows)

    def destroy(self) -> None:
        self._ar = None
        self.disabled = True
