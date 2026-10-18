"""4-bit (NVFP4-style) KV cache for DeepSeek V4 Pro: cell format, store, dequant, sparse decode.

Installed into the engine as ``sglang/srt/layers/attention/numinous_kv4pro.py`` by
operations/patches/kv4pro_hooks.py (which also routes the DSV4 pool, compressor, SWA store, attention
backend and pool pricing through it). Everything here is inert unless NUMINOUS_KV4PRO selects a mode.

Cell format (one token of one layer; 512 latent values = 448 nope + 64 rope):
    bytes [  0, 224)  nope values, E2M1 (FP4), two per byte: value 2i in the low nibble, 2i+1 in the high
    bytes [224, 252)  28 E4M3 block scales, one per 16 nope values (NVFP4 blocks)
    bytes [252, 256)  zero (the decode kernel reads 32 scale bytes; the zero pad makes columns 448..511 of the
                      dequantized nope tile exactly zero, so one 512-wide tile serves QK and PV)
    bytes [256, 320)  rope, E4M3 with unit scale (the TRT-LLM uniform cell stores rope the same way)   ROPE=fp8
    bytes [256, 384)  rope, bf16 (bit-exact with the FlashMLA 584-byte cell's rope)                    ROPE=bf16
Cells are contiguous (token t of the pool at byte t * CELL); pages carry no padding.
Dequant: x = e2m1(nibble) * e4m3(scale). Quant: per 16-block, scale = RNE_e4m3(amax16 / 6 * f) for the f in
SCALE_SEARCH with the smallest squared error (values past 6 clip), q = RNE_e2m1(x / scale). The search lowers the
relative error of Gaussian and checkpoint-derived Pro KV rows from 0.095 to 0.088 (measured, kv4pro_realkv.py; a
64-wide Hadamard rotation measured no gain: the rows carry no in-block outliers).

Modes (env NUMINOUS_KV4PRO; unset or "0" = off):
    c4     the c4 compressed-KV pool in 4 bit; SWA window and c128 pools keep the FP8 584-byte FlashMLA cell
    all    c4 pool and the SWA window pool in 4 bit
The c128 pool always stays FP8: it is 1.1% of the per-token bytes (derived, bytes_per_full_token) and its index
width ceil64(context / 128) would make a per-step workspace large.
NUMINOUS_KV4PRO_ROPE = fp8 (default, 320-byte cell) | bf16 (384-byte cell).
NUMINOUS_KV4PRO_DECODE = triton (default: the native FP4 sparse decode below) | torch (reference, tests only).

Decode path: ``kv4_sparse_decode`` has flash_mla_with_kvcache's sparse semantics (SWA indices + extra
c4/c128 indices, per-row topk lengths, -1 = skip, attention sink, lonely rows -> 0) and dequantizes FP4 in
the kernel: split-K over 64-key chunks of the concatenated SWA+extra key list, BLOCK_H heads per program,
tl.dot on bf16 tiles, then a combine kernel that merges splits and applies the sink. Shapes depend only on
tensor shapes, so it is CUDA-graph capturable. Prefill path: ``kv4_dequant_gather`` writes bf16 rows into
the sparse-prefill workspace that flash_mla_sparse_fwd consumes (same contract as upstream
dequantize_k_cache_paged).
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch

try:  # CPU-only hosts can still use the torch references
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

D_NOPE = 448
D_ROPE = 64
D = D_NOPE + D_ROPE
BLOCK = 16
N_SCALES = D_NOPE // BLOCK  # 28
NOPE_BYTES = D_NOPE // 2  # 224
SCALE_OFF = NOPE_BYTES  # 224
ROPE_OFF = 256
E2M1_MAX = 6.0
E4M3_MAX = 448.0
FP8_CELL = 584  # FlashMLA packed cell (448 fp8 + 64 bf16 + 7 ue8m0 + 1 pad)
UNIFORM_FP8_CELL = 512  # TRT-LLM DSV4 uniform cell


def cell_bytes(rope: str = "fp8") -> int:
    if rope not in ("fp8", "bf16"):
        raise ValueError(f"NUMINOUS_KV4PRO_ROPE must be fp8 or bf16, got {rope!r}")
    return ROPE_OFF + (D_ROPE if rope == "fp8" else 2 * D_ROPE)


def mode() -> Tuple[bool, bool, str]:
    """(swa_fp4, c4_fp4, rope) from the environment."""
    m = os.environ.get("NUMINOUS_KV4PRO", "0").strip().lower()
    rope = os.environ.get("NUMINOUS_KV4PRO_ROPE", "fp8").strip().lower()
    cell_bytes(rope)
    if m in ("", "0", "off", "none"):
        return False, False, rope
    if m == "c4":
        return False, True, rope
    if m in ("all", "1"):
        return True, True, rope
    raise ValueError(f"NUMINOUS_KV4PRO must be 0, c4 or all, got {m!r}")


# ------------------------------------------------------------------------------------------ pool pricing
def priced_cell_bytes(default_kv_bytes: float) -> Tuple[float, float, float]:
    """(swa, c4, c128) cell bytes the DSV4 pool configurator should price."""
    swa4, c44, rope = mode()
    c = cell_bytes(rope)
    return (c if swa4 else default_kv_bytes), (c if c44 else default_kv_bytes), default_kv_bytes


WS_WIDTHS = ((128, 128), (192, 128), (256, 128), (1024, 64))  # (index width, FlashMLA page): SWA decode / DSpark
# verify / wide draft tables, c4 top-1024


def ws_tokens() -> int:
    """Query rows per decode workspace chunk (NUMINOUS_KV4PRO_WS_TOKENS, default 256)."""
    return max(16, int(os.environ.get("NUMINOUS_KV4PRO_WS_TOKENS", "256")))


def workspace_bytes() -> int:
    """Upper bound of the FP8 decode workspaces the engine glue allocates in the active mode (one per width), which
    the pool configurator reserves so the non-KV headroom stays what stock v0.5.20 leaves."""
    swa4, c44, _ = mode()
    total = 0
    for width, page in WS_WIDTHS:
        if (page == 128 and swa4) or (page == 64 and c44):
            total += -(-ws_tokens() * width // page) * fp8_cell_page_bytes(page)
    return total


def bytes_per_full_token(*, swa_cell: float, comp_cell: float, c128_cell: Optional[float] = None,
                         n_layers: int = 61, n_c4: int = 30,
                         n_c128: int = 31, swa_ratio: float = 0.1, speculative: bool = True,
                         state_bytes: int = 4, indexer_bytes: int = 132, window: int = 128) -> float:
    """DSV4PoolConfigurator._get_bytes_per_full_token (non-unified) with split SWA / compressed cells.

    derived: mirrors v0.5.20 pool_configurator.py; defaults are DeepSeek V4 Pro 0813 at the served flags.
    """
    c4_ring = 16 if speculative else 8
    c4_state = 2 * 2 * D * state_bytes
    c4_idx_state = 2 * 2 * 128 * state_bytes
    c128_cell = comp_cell if c128_cell is None else c128_cell
    b = (swa_ratio * swa_cell * n_layers + 0.25 * comp_cell * n_c4 + c128_cell / 128 * n_c128
         + 0.25 * indexer_bytes * n_c4 + swa_ratio * (c4_ring / window) * (c4_state + c4_idx_state) * n_c4)
    return b * ((n_layers + 1) / n_layers if speculative else 1.0)


# ------------------------------------------------------------------------------------------ torch reference
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SCALE_SEARCH = (1.0, 0.93, 0.86, 0.8)


def _e2m1_code(a: torch.Tensor) -> torch.Tensor:
    """Round |x| (already divided by the block scale) to the nearest E2M1 code, ties to even."""
    a = a.clamp(max=E2M1_MAX)
    return ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int()
            + (a > 2.5).int() + (a >= 3.5).int() + (a > 5.0).int())


def quantize_ref(x: torch.Tensor, rope: str = "fp8") -> torch.Tensor:
    """[N, 512] float -> [N, CELL] uint8 cells (bit-exact target for the Triton store kernel)."""
    assert x.ndim == 2 and x.shape[1] == D
    x = x.float()
    n = x.shape[0]
    nope = x[:, :D_NOPE].reshape(n, N_SCALES, BLOCK)
    amax = nope.abs().amax(-1)
    table = torch.tensor(_E2M1, device=x.device)
    best_code = best_s8 = best_err = None
    for f in SCALE_SEARCH:
        s8 = (amax / E2M1_MAX * f).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
        sf = s8.float()
        inv = torch.where(sf > 0, 1.0 / torch.where(sf > 0, sf, torch.ones_like(sf)), torch.zeros_like(sf))
        q = nope * inv[..., None]
        mag = _e2m1_code(q.abs())
        code = mag | ((q < 0) & (mag > 0)).int() << 3
        err = (torch.where(q < 0, -table[mag], table[mag]) * sf[..., None] - nope).pow(2).sum(-1)
        if best_err is None:
            best_code, best_s8, best_err = code, s8, err
        else:
            m = err < best_err
            best_code = torch.where(m[..., None], code, best_code)
            best_s8 = torch.where(m, s8.view(torch.uint8), best_s8.view(torch.uint8)).view(torch.float8_e4m3fn)
            best_err = torch.where(m, err, best_err)
    code, s8 = best_code, best_s8
    code = code.reshape(n, NOPE_BYTES, 2)
    packed = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    cells = torch.zeros(n, cell_bytes(rope), dtype=torch.uint8, device=x.device)
    cells[:, :NOPE_BYTES] = packed
    cells[:, SCALE_OFF:SCALE_OFF + N_SCALES] = s8.view(torch.uint8)
    r = x[:, D_NOPE:]
    if rope == "fp8":
        cells[:, ROPE_OFF:] = r.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        cells[:, ROPE_OFF:] = r.to(torch.bfloat16).contiguous().view(torch.uint8)
    return cells


def dequantize_ref(cells: torch.Tensor, rope: str = "fp8") -> torch.Tensor:
    """[N, CELL] uint8 -> [N, 512] float32."""
    n = cells.shape[0]
    packed = cells[:, :NOPE_BYTES].int()
    code = torch.stack([packed & 0xF, packed >> 4], -1).reshape(n, D_NOPE)
    table = torch.tensor(_E2M1, device=cells.device)
    mag = table[(code & 7).long()]
    val = torch.where((code & 8) != 0, -mag, mag)
    sf = cells[:, SCALE_OFF:SCALE_OFF + N_SCALES].contiguous().view(torch.float8_e4m3fn).float()
    nope = (val.reshape(n, N_SCALES, BLOCK) * sf[..., None]).reshape(n, D_NOPE)
    if rope == "fp8":
        r = cells[:, ROPE_OFF:ROPE_OFF + D_ROPE].contiguous().view(torch.float8_e4m3fn).float()
    else:
        r = cells[:, ROPE_OFF:ROPE_OFF + 2 * D_ROPE].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope, r], 1)


def fp8_cell_page_bytes(page_size: int) -> int:
    """Row stride of the FlashMLA 584-byte paged pool (DeepSeekV4SingleKVPool.create_buffer)."""
    return -(-page_size * FP8_CELL // 576) * 576


def quantize_fp8_paged_ref(x: torch.Tensor, page_size: int) -> torch.Tensor:
    """[N, 512] -> FlashMLA 584-byte paged buffer [ceil(N / P), page_bytes] uint8 (token t at index t).

    UE8M0 per-64 scale = 2^ceil(log2(amax / 448)); used to build SWA pools for tests and benchmarks.
    """
    n = x.shape[0]
    pages = -(-n // page_size)
    pb = fp8_cell_page_bytes(page_size)
    buf = torch.zeros(pages, pb, dtype=torch.uint8, device=x.device)
    x = x.float()
    nope = x[:, :D_NOPE].reshape(n, 7, 64)
    amax = nope.abs().amax(-1).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / E4M3_MAX)).clamp(-127, 127)
    q8 = (nope / torch.exp2(e)[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    t = torch.arange(n, device=x.device)
    page, off = t // page_size, t % page_size
    flat = buf.view(-1)
    data = (page * pb + off * 576)[:, None]
    flat[(data + torch.arange(D_NOPE, device=x.device)).reshape(-1)] = q8.reshape(n, D_NOPE).view(torch.uint8).reshape(-1)
    rope = x[:, D_NOPE:].to(torch.bfloat16).contiguous().view(torch.uint8)
    flat[(data + D_NOPE + torch.arange(2 * D_ROPE, device=x.device)).reshape(-1)] = rope.reshape(-1)
    sc = (page * pb + page_size * 576 + off * 8)[:, None] + torch.arange(7, device=x.device)
    flat[sc.reshape(-1)] = (e + 127).to(torch.uint8).reshape(-1)
    return buf


def dequantize_fp8_paged_ref(buf: torch.Tensor, page_size: int, n: int) -> torch.Tensor:
    """Inverse of quantize_fp8_paged_ref for tokens 0..n-1 -> [n, 512] float32."""
    pb = buf.shape[1]
    flat = buf.reshape(-1)
    t = torch.arange(n, device=buf.device)
    page, off = t // page_size, t % page_size
    data = (page * pb + off * 576)[:, None]
    q8 = flat[data + torch.arange(D_NOPE, device=buf.device)].contiguous().view(torch.float8_e4m3fn).float()
    sc = flat[(page * pb + page_size * 576 + off * 8)[:, None] + torch.arange(7, device=buf.device)].float()
    nope = (q8.reshape(n, 7, 64) * torch.exp2(sc - 127)[..., None]).reshape(n, D_NOPE)
    rope = flat[data + D_NOPE + torch.arange(2 * D_ROPE, device=buf.device)].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope, rope], 1)


def sparse_decode_ref(q, swa_cells, swa_idx, swa_len, ext_cells=None, ext_idx=None, ext_len=None,
                      attn_sink=None, softmax_scale=None, rope="fp8", kv_rows=None):
    """fp32 reference with flash_mla_with_kvcache sparse semantics (flash_mla_sm120._sm120_sparse_decode_fwd).

    q [T, H, 512]; *_cells flat [tokens, CELL] uint8 (or ``kv_rows`` = (swa_rows, ext_rows) float [tokens, 512]
    to test against unquantized KV); *_idx [T, W] int (-1 skip); *_len [T] (None = all W).
    Returns out [T, H, 512] float32 and lse [T, H] (natural log, without the sink).
    """
    T, H, _ = q.shape
    scale = softmax_scale if softmax_scale is not None else D ** -0.5
    parts = [(swa_cells, swa_idx, swa_len, 0)]
    if ext_idx is not None:
        parts.append((ext_cells, ext_idx, ext_len, 1))
    kvs, masks = [], []
    for cells, idx, ln, which in parts:
        idx = idx.reshape(T, -1).long()
        W = idx.shape[1]
        bad = idx < 0
        if ln is not None:
            bad = bad | (torch.arange(W, device=idx.device)[None, :] >= ln.reshape(T, 1).long())
        safe = idx.clamp(min=0)
        rows = kv_rows[which] if kv_rows is not None else dequantize_ref(cells, rope)
        kv = rows.float()[safe.reshape(-1)].reshape(T, W, D)
        kvs.append(torch.where(bad[..., None], torch.zeros_like(kv), kv))
        masks.append(bad)
    kv = torch.cat(kvs, 1)
    bad = torch.cat(masks, 1)
    s = torch.einsum("thd,twd->thw", q.float(), kv) * scale
    s = s.masked_fill(bad[:, None, :], float("-inf"))
    lse = torch.logsumexp(s, -1)
    lse_out = lse.clone()
    if attn_sink is not None:
        lse_out = torch.logaddexp(lse, attn_sink.float().view(1, H).expand_as(lse))
    lonely = torch.isneginf(lse)
    lse_out = torch.where(lonely, torch.full_like(lse_out, float("inf")), lse_out)
    p = torch.exp(s - lse_out[..., None])
    out = torch.einsum("thw,twd->thd", p, kv)
    out = torch.where(lonely[..., None], torch.zeros_like(out), out)
    return out, lse


# ------------------------------------------------------------------------------------------ Triton kernels
if triton is not None:

    @triton.jit
    def _nvfp4_try(xb, amax, f: tl.constexpr, best_code, best_s, best_err):
        """One SCALE_SEARCH candidate for 32 blocks of 16: keep it where its squared error is lower."""
        s8 = tl.minimum(amax / 6.0 * f, 448.0).to(tl.float8e4nv)
        sf = s8.to(tl.float32)
        inv = tl.where(sf > 0, 1.0 / tl.where(sf > 0, sf, 1.0), 0.0)
        qv = xb * inv[:, None]
        a = tl.minimum(tl.abs(qv), 6.0)
        mag = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
               + (a >= 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
               + (a > 5.0).to(tl.int32))
        lvl = tl.where(mag < 4, mag.to(tl.float32) * 0.5,
                       tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))))
        d = tl.where(qv < 0, -lvl, lvl) * sf[:, None] - xb
        err = tl.sum(d * d, axis=1)
        code = mag | tl.where((qv < 0) & (mag > 0), 8, 0)
        better = err < best_err
        best_code = tl.where(better[:, None], code, best_code)
        best_s = tl.where(better, s8.to(tl.uint8, bitcast=True).to(tl.int32), best_s)
        best_err = tl.where(better, err, best_err)
        return best_code, best_s, best_err

    @triton.jit
    def _kv4_store_kernel(X, stride_x, LOC, CACHE, n_rows, CELL: tl.constexpr, ROPE_FP8: tl.constexpr):
        """One program per row: quantize a normed+roped [512] bf16 row into a 4-bit cell at LOC[row]."""
        row = tl.program_id(0)
        loc = tl.load(LOC + row).to(tl.int64)
        if loc >= 0:
            base = CACHE + loc * CELL
            offs = tl.arange(0, 512)
            x = tl.load(X + row.to(tl.int64) * stride_x + offs).to(tl.float32)
            xb = tl.reshape(x, (32, 16))
            amax = tl.max(tl.abs(xb), axis=1)
            best_code = tl.zeros((32, 16), tl.int32)
            best_s = tl.zeros((32,), tl.int32)
            best_err = tl.full((32,), float("inf"), tl.float32)
            best_code, best_s, best_err = _nvfp4_try(xb, amax, 1.0, best_code, best_s, best_err)
            best_code, best_s, best_err = _nvfp4_try(xb, amax, 0.93, best_code, best_s, best_err)
            best_code, best_s, best_err = _nvfp4_try(xb, amax, 0.86, best_code, best_s, best_err)
            best_code, best_s, best_err = _nvfp4_try(xb, amax, 0.8, best_code, best_s, best_err)
            code = best_code
            s8 = best_s.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
            pairs = tl.reshape(code, (256, 2))
            lo, hi = tl.split(pairs)
            packed = (lo | (hi << 4)).to(tl.uint8)
            b = tl.arange(0, 256)
            tl.store(base + b, packed, mask=b < 224)
            sidx = tl.arange(0, 32)
            s_u8 = tl.where(sidx < 28, s8.to(tl.uint8, bitcast=True), 0).to(tl.uint8)
            tl.store(base + 224 + sidx, s_u8)
            r_offs = tl.arange(0, 64)
            r = tl.load(X + row.to(tl.int64) * stride_x + 448 + r_offs).to(tl.float32)
            if ROPE_FP8:
                r8 = tl.minimum(tl.maximum(r, -448.0), 448.0).to(tl.float8e4nv)
                tl.store(base + 256 + r_offs, r8.to(tl.uint8, bitcast=True))
            else:
                rb = r.to(tl.bfloat16).to(tl.uint16, bitcast=True)
                tl.store((base + 256).to(tl.pointer_type(tl.uint16)) + r_offs, rb)

    @triton.jit
    def _kv4_half(CACHE, base, valid, HALF: tl.constexpr, BLOCK_N: tl.constexpr):
        """Dequantize nope columns [256*HALF, 256*HALF+256) of BLOCK_N 4-bit cells -> [BLOCK_N, 256] bf16.
        HALF=1 covers 256..511; columns 448..511 read the zero scale pad, so they come out exactly 0
        (callers that want the full latent overwrite them with the rope, see _kv4_tile)."""
        b = tl.arange(0, 128)
        packed = tl.load(CACHE + base[:, None] + HALF * 128 + b[None, :], mask=valid[:, None], other=0)
        lo = packed & 0xF
        hi = packed >> 4
        # E2M1 nibble -> E4M3 byte holding value * 2^-6: sign to bit 7, exponent+mantissa to bits 4..2
        lo8 = (((lo & 8) << 4) | ((lo & 7) << 2)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        hi8 = (((hi & 8) << 4) | ((hi & 7) << 2)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        vals = tl.reshape(tl.join(lo8.to(tl.float32), hi8.to(tl.float32)), (BLOCK_N, 256))
        sb = tl.arange(0, 16)
        s8 = tl.load(CACHE + base[:, None] + 224 + HALF * 16 + sb[None, :], mask=valid[:, None], other=0)
        sf = s8.to(tl.float8e4nv, bitcast=True).to(tl.float32) * 64.0
        sf = tl.reshape(tl.broadcast_to(sf[:, :, None], (BLOCK_N, 16, 16)), (BLOCK_N, 256))
        return (vals * sf).to(tl.bfloat16)

    @triton.jit
    def _kv4_tile(CACHE, idx, valid, CELL: tl.constexpr, ROPE_FP8: tl.constexpr, BLOCK_N: tl.constexpr):
        """Gather + dequantize BLOCK_N 4-bit cells into the 512-wide latent as two halves (cols 0..255, 256..511)."""
        base = tl.where(valid, idx, 0).to(tl.int64) * CELL
        base = tl.multiple_of(base, 64)
        k0 = _kv4_half(CACHE, base, valid, 0, BLOCK_N)
        k1 = _kv4_half(CACHE, base, valid, 1, BLOCK_N)
        # full-latent view: columns 448..511 of k1 hold the rope (the zero pad made the fp4 part 0 there)
        c = tl.arange(0, 256)
        rm = valid[:, None] & (c[None, :] >= 192)
        if ROPE_FP8:
            r8 = tl.load(CACHE + base[:, None] + 256 - 192 + c[None, :], mask=rm, other=0)
            r = r8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        else:
            rp = (CACHE + base + 256).to(tl.pointer_type(tl.bfloat16))
            r = tl.load(rp[:, None] - 192 + c[None, :], mask=rm, other=0.0)
        k1 = tl.where(c[None, :] >= 192, r, k1)
        return k0, k1

    @triton.jit
    def _fp8_half(CACHE, data, sbase, valid, HALF: tl.constexpr, BLOCK_N: tl.constexpr):
        c = tl.arange(0, 256)
        nm = valid[:, None] & (c[None, :] + HALF * 256 < 448)
        x8 = tl.load(CACHE + data[:, None] + HALF * 256 + c[None, :], mask=nm, other=0)
        xs = x8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sb = tl.arange(0, 4)
        sc = tl.load(CACHE + sbase[:, None] + HALF * 4 + sb[None, :],
                     mask=valid[:, None] & (sb[None, :] + HALF * 4 < 7), other=127)
        sf = tl.exp2(sc.to(tl.float32) - 127.0)
        sf = tl.reshape(tl.broadcast_to(sf[:, :, None], (BLOCK_N, 4, 64)), (BLOCK_N, 256))
        return tl.where(nm, xs * sf, 0.0).to(tl.bfloat16)

    @triton.jit
    def _fp8_tile(CACHE, idx, valid, PAGE_SIZE, PAGE_BYTES, BLOCK_N: tl.constexpr):
        """Gather + dequantize BLOCK_N FlashMLA 584-byte cells (paged: per page P x [448 fp8 | 64 bf16] then
        P x [7 ue8m0 + pad], page stride PAGE_BYTES) into the 512-wide latent as two halves."""
        safe = tl.where(valid, idx, 0).to(tl.int64)
        page = safe // PAGE_SIZE
        off = safe % PAGE_SIZE
        data = page * PAGE_BYTES + off * 576
        sbase = page * PAGE_BYTES + PAGE_SIZE * 576 + off * 8
        k0 = _fp8_half(CACHE, data, sbase, valid, 0, BLOCK_N)
        k1 = _fp8_half(CACHE, data, sbase, valid, 1, BLOCK_N)
        c = tl.arange(0, 256)
        rp = (CACHE + data + 448).to(tl.pointer_type(tl.bfloat16))
        r = tl.load(rp[:, None] - 192 + c[None, :], mask=valid[:, None] & (c[None, :] >= 192), other=0.0)
        return k0, tl.where(c[None, :] >= 192, r, k1)

    @triton.jit
    def _kv4_dequant_kernel(CACHE, IDS, OUT, stride_o, n, CELL: tl.constexpr, ROPE_FP8: tl.constexpr,
                            BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        inb = rows < n
        idx = tl.load(IDS + rows, mask=inb, other=-1)
        valid = inb & (idx >= 0)
        k0, k1 = _kv4_tile(CACHE, idx, valid, CELL, ROPE_FP8, BLOCK_N)
        c = tl.arange(0, 256)
        optr = OUT + rows.to(tl.int64)[:, None] * stride_o
        tl.store(optr + c[None, :], k0, mask=inb[:, None])
        tl.store(optr + 256 + c[None, :], k1, mask=inb[:, None])

    @triton.jit
    def _kv4_decode_kernel(Q, stride_qt, stride_qh,
                           SWA, SWA_IDX, stride_si, SWA_LEN, W_SWA,
                           EXT, EXT_IDX, stride_ei, EXT_LEN, W_EXT,
                           PO, PL, H, sm_scale_log2, SWA_PAGE_SIZE, SWA_PAGE_BYTES,
                           SWA_FP8: tl.constexpr, NSPLIT: tl.constexpr, HAS_EXT: tl.constexpr,
                           HAS_SWA_LEN: tl.constexpr, HAS_EXT_LEN: tl.constexpr, CELL_SWA: tl.constexpr,
                           CELL_EXT: tl.constexpr, ROPE_FP8: tl.constexpr, BLOCK_H: tl.constexpr,
                           BLOCK_N: tl.constexpr, DV_SPLIT: tl.constexpr):
        """Split-K sparse decode. Program (t, head block [x value half], split). DV_SPLIT=2: each program
        computes the full 512-wide QK but accumulates only one 256-wide value half (fits Blackwell TMEM with
        BLOCK_H=128); DV_SPLIT=1: one program accumulates both halves (register accumulators, small BLOCK_H)."""
        t = tl.program_id(0)
        hb = tl.program_id(1)
        sp = tl.program_id(2)
        if DV_SPLIT == 2:
            dv = hb % 2
            hb = hb // 2
        else:
            dv = 0
        heads = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = heads < H
        c256 = tl.arange(0, 256)
        qrow = Q + t.to(tl.int64) * stride_qt + heads[:, None] * stride_qh
        q0 = tl.load(qrow + c256[None, :], mask=hmask[:, None], other=0.0)
        q1 = tl.load(qrow + 256 + c256[None, :], mask=hmask[:, None], other=0.0)

        swa_len = W_SWA
        if HAS_SWA_LEN:
            swa_len = tl.minimum(tl.load(SWA_LEN + t), W_SWA)
        swa_len = tl.maximum(swa_len, 0)
        ext_len = 0
        if HAS_EXT:
            ext_len = W_EXT
            if HAS_EXT_LEN:
                ext_len = tl.minimum(tl.load(EXT_LEN + t), W_EXT)
            ext_len = tl.maximum(ext_len, 0)
        n_swa = tl.cdiv(swa_len, BLOCK_N)
        n_tot = n_swa + tl.cdiv(ext_len, BLOCK_N)

        m_i = tl.full((BLOCK_H,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        acc0 = tl.zeros((BLOCK_H, 256), tl.float32)
        if DV_SPLIT == 1:
            acc1 = tl.zeros((BLOCK_H, 256), tl.float32)
        jn = tl.arange(0, BLOCK_N)
        for c in range(sp, n_tot, NSPLIT):
            if c < n_swa:
                j = c * BLOCK_N + jn
                idx = tl.load(SWA_IDX + t.to(tl.int64) * stride_si + j)
                valid = (j < swa_len) & (idx >= 0)
                if SWA_FP8:
                    k0, k1 = _fp8_tile(SWA, idx, valid, SWA_PAGE_SIZE, SWA_PAGE_BYTES, BLOCK_N)
                else:
                    k0, k1 = _kv4_tile(SWA, idx, valid, CELL_SWA, ROPE_FP8, BLOCK_N)
            else:
                j = (c - n_swa) * BLOCK_N + jn
                idx = tl.load(EXT_IDX + t.to(tl.int64) * stride_ei + j)
                valid = (j < ext_len) & (idx >= 0)
                k0, k1 = _kv4_tile(EXT, idx, valid, CELL_EXT, ROPE_FP8, BLOCK_N)
            s = tl.dot(q0, tl.trans(k0))
            s = tl.dot(q1, tl.trans(k1), s)
            s = tl.where(valid[None, :], s * sm_scale_log2, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            pb = p.to(tl.bfloat16)
            if DV_SPLIT == 2:
                if dv == 0:
                    kv = k0
                else:
                    kv = k1
                acc0 = tl.dot(pb, kv, acc0 * alpha[:, None])
            else:
                acc0 = tl.dot(pb, k0, acc0 * alpha[:, None])
                acc1 = tl.dot(pb, k1, acc1 * alpha[:, None])
            m_i = m_new
        has = l_i > 0
        inv_l = tl.where(has, 1.0 / tl.where(has, l_i, 1.0), 0.0)
        lse = tl.where(has, m_i + tl.log2(tl.where(has, l_i, 1.0)), float("-inf"))
        prow = ((t * NSPLIT + sp) * H + heads).to(tl.int64)
        if dv == 0:
            tl.store(PL + prow, lse, mask=hmask)
        optr = PO + prow[:, None] * 512
        tl.store(optr + dv * 256 + c256[None, :], acc0 * inv_l[:, None], mask=hmask[:, None])
        if DV_SPLIT == 1:
            tl.store(optr + 256 + c256[None, :], acc1 * inv_l[:, None], mask=hmask[:, None])

    @triton.jit
    def _kv4_combine_kernel(PO, PL, SINK, OUT, stride_ot, stride_oh, LSE_OUT, H,
                            NSPLIT: tl.constexpr, NSPLIT_P2: tl.constexpr, HAS_SINK: tl.constexpr):
        t = tl.program_id(0)
        h = tl.program_id(1)
        s = tl.arange(0, NSPLIT_P2)
        sm = s < NSPLIT
        rows = ((t * NSPLIT + s) * H + h).to(tl.int64)
        lse = tl.load(PL + rows, mask=sm, other=float("-inf"))
        mx = tl.max(lse, 0)
        LOG2E: tl.constexpr = 1.4426950408889634
        if HAS_SINK:
            sink = tl.load(SINK + h).to(tl.float32) * LOG2E
            mref = tl.maximum(mx, sink)
        else:
            mref = mx
        mref = tl.where(mref == float("-inf"), 0.0, mref)
        w = tl.exp2(lse - mref)
        denom = tl.sum(w, 0)
        if HAS_SINK:
            denom = denom + tl.exp2(sink - mref)
        c = tl.arange(0, 512)
        po = tl.load(PO + rows[:, None] * 512 + c[None, :], mask=sm[:, None], other=0.0)
        o = tl.sum(po * w[:, None], 0) / tl.where(denom > 0, denom, 1.0)
        tl.store(OUT + t.to(tl.int64) * stride_ot + h * stride_oh + c, o.to(OUT.dtype.element_ty))
        # lse over the keys only (FlashMLA's softmax_lse excludes the sink), natural log
        lse_keys = tl.where(mx == float("-inf"), float("-inf"),
                            (mx + tl.log2(tl.where(mx == float("-inf"), 1.0, tl.sum(tl.exp2(lse - mx), 0)))) / LOG2E)
        tl.store(LSE_OUT + t * H + h, lse_keys)


if triton is not None:

    @triton.jit
    def _kv4_ws_half(CACHE, base, WS, data, s8, E8, ok, HALF: tl.constexpr, BLOCK_N: tl.constexpr):
        """Requantize nope half HALF (FP4 bytes [128*HALF, +128)) into FP8 at workspace bytes data + 256*HALF.
        fp8 = RNE(e2m1 * s16 * 2^-E64): one multiply and one convert per value (E64 comes from the scales)."""
        bb = tl.arange(0, 128)
        packed = tl.load(CACHE + base[:, None] + HALF * 128 + bb[None, :], mask=ok[:, None], other=0)
        lo = packed & 0xF
        hi = packed >> 4
        lo8 = (((lo & 8) << 4) | ((lo & 7) << 2)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        hi8 = (((hi & 8) << 4) | ((hi & 7) << 2)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        vals = tl.reshape(tl.join(lo8.to(tl.float32), hi8.to(tl.float32)), (BLOCK_N, 256))
        # factor per 16-block: s16 * 64 (undo the 2^-6 of the nibble trick) * 2^-E of its 64-tile
        sf = s8.to(tl.float8e4nv, bitcast=True).to(tl.float32)  # [BLOCK_N, 16] scales of this half
        e = tl.reshape(tl.broadcast_to(E8[:, :, None], (BLOCK_N, 4, 4)), (BLOCK_N, 16))
        f = sf * tl.exp2(6.0 - e)
        f = tl.reshape(tl.broadcast_to(f[:, :, None], (BLOCK_N, 16, 16)), (BLOCK_N, 256))
        q = (vals * f).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        c = tl.arange(0, 256)
        tl.store(WS + data[:, None] + HALF * 256 + c[None, :], q, mask=ok[:, None] & (c[None, :] + HALF * 256 < 448))

    @triton.jit
    def _kv4_to_fp8_ws_kernel(CACHE, IDX, stride_i, LEN, WS, OUT_IDX, W, WS_PAGE_BYTES,
                              HAS_LEN: tl.constexpr, CELL: tl.constexpr, ROPE_FP8: tl.constexpr,
                              WS_PAGE: tl.constexpr, BLOCK_N: tl.constexpr):
        """Program (t, row block): requantize the 4-bit cells IDX[t, j] into the FlashMLA 584-byte paged layout
        at workspace token t * W + j; OUT_IDX[t, j] = that token (or -1 when skipped). The UE8M0 scale of each
        64-tile is 2^ceil(log2(6 * max(s16 of its 4 blocks) / 448)), which bounds every |value| by 448."""
        t = tl.program_id(0)
        jb = tl.program_id(1)
        j = jb * BLOCK_N + tl.arange(0, BLOCK_N)
        idx = tl.load(IDX + t.to(tl.int64) * stride_i + j)
        ok = idx >= 0
        if HAS_LEN:
            ok = ok & (j < tl.load(LEN + t))
        p = t.to(tl.int64) * W + j
        tl.store(OUT_IDX + p, tl.where(ok, p, -1).to(tl.int32))
        base = tl.multiple_of(tl.where(ok, idx, 0).to(tl.int64) * CELL, 64)
        page = p // WS_PAGE
        off = p % WS_PAGE
        data = page * WS_PAGE_BYTES + off * 576
        sbase = page * WS_PAGE_BYTES + WS_PAGE * 576 + off * 8
        sb = tl.arange(0, 32)
        s8 = tl.load(CACHE + base[:, None] + 224 + sb[None, :], mask=ok[:, None], other=0)  # 28 + zero pad
        smax = tl.max(tl.reshape(s8.to(tl.float8e4nv, bitcast=True).to(tl.float32), (BLOCK_N, 8, 4)), 2)
        E = tl.where(smax > 0, tl.ceil(tl.log2(tl.maximum(smax, 1e-30) * (6.0 / 448.0))), 0.0)
        E = tl.minimum(tl.maximum(E, -127.0), 127.0)  # [BLOCK_N, 8]; tile 7 = rope, unused
        E0, E1 = tl.split(tl.reshape(E, (BLOCK_N, 2, 4)).permute(0, 2, 1))
        s0, s1 = tl.split(tl.reshape(s8, (BLOCK_N, 2, 16)).permute(0, 2, 1))
        _kv4_ws_half(CACHE, base, WS, data, s0, E0, ok, 0, BLOCK_N)
        _kv4_ws_half(CACHE, base, WS, data, s1, E1, ok, 1, BLOCK_N)
        r = tl.arange(0, 64)
        if ROPE_FP8:
            rope = tl.load(CACHE + base[:, None] + 256 + r[None, :], mask=ok[:, None], other=0)
            rope = rope.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        else:
            rope = tl.load((CACHE + base + 256).to(tl.pointer_type(tl.bfloat16))[:, None] + r[None, :],
                           mask=ok[:, None], other=0.0)
        tl.store((WS + data + 448).to(tl.pointer_type(tl.bfloat16))[:, None] + r[None, :], rope, mask=ok[:, None])
        e8 = tl.arange(0, 8)
        tl.store(WS + sbase[:, None] + e8[None, :], (E + 127.0).to(tl.uint8), mask=ok[:, None] & (e8[None, :] < 7))


def kv4_to_fp8_workspace(cache: torch.Tensor, idx: torch.Tensor, length: Optional[torch.Tensor],
                         ws: torch.Tensor, ws_page: int, rope: str = "fp8") -> torch.Tensor:
    """Gather rows idx [T, W] (or [T, 1, W]) of a 4-bit pool into FlashMLA 584-byte cells in ``ws``
    ([pages, fp8_cell_page_bytes(ws_page)] uint8, >= T * W tokens). Returns the remapped indices (same shape
    as idx, int32): workspace token t * W + j, or -1 for skipped slots (idx < 0 or j >= length[t])."""
    shape = idx.shape
    T = shape[0]
    idx2 = idx.reshape(T, -1)
    W = idx2.shape[1]
    BN = 16
    assert W % BN == 0, W
    assert ws.shape[0] * ws_page >= T * W, (ws.shape, T, W)
    assert ws.shape[1] == fp8_cell_page_bytes(ws_page), ws.shape
    if idx2.stride(1) != 1:
        idx2 = idx2.contiguous()
    out = torch.empty(T, W, dtype=torch.int32, device=idx.device)
    if T:
        _kv4_to_fp8_ws_kernel[(T, W // BN)](_flat_u8(cache), idx2, idx2.stride(0),
                                            length if length is not None else idx2, ws, out, W, ws.shape[1],
                                            HAS_LEN=length is not None, CELL=cell_bytes(rope),
                                            ROPE_FP8=(rope == "fp8"), WS_PAGE=ws_page, BLOCK_N=BN, num_warps=4)
    return out.view(shape)


def _flat_u8(buf: torch.Tensor) -> torch.Tensor:
    return buf.view(torch.uint8).reshape(-1)


def kv4_store(cache: torch.Tensor, loc: torch.Tensor, x: torch.Tensor, rope: str = "fp8") -> None:
    """Quantize rows x [N, 512] (bf16/fp32) into cells of ``cache`` at token indices ``loc`` (< 0: skipped)."""
    assert x.ndim == 2 and x.shape[1] == D, x.shape
    n = x.shape[0]
    if n == 0:
        return
    if x.stride(1) != 1:
        x = x.contiguous()
    loc = loc.reshape(-1)
    assert loc.shape[0] == n, (loc.shape, n)
    _kv4_store_kernel[(n,)](x, x.stride(0), loc, _flat_u8(cache), n, CELL=cell_bytes(rope),
                            ROPE_FP8=(rope == "fp8"), num_warps=4)


def kv4_dequant_gather(cache: torch.Tensor, token_ids: torch.Tensor, rope: str = "fp8",
                       out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """bf16 rows for ``token_ids`` (-1 -> zeros) as [N, 1, 512]: the sparse-prefill workspace contract."""
    n = token_ids.shape[0]
    if out is None:
        out = torch.empty(n, 1, D, dtype=torch.bfloat16, device=cache.device)
    assert out.shape == (n, 1, D) and out.dtype == torch.bfloat16 and out.stride(2) == 1
    if n:
        BN = 32
        _kv4_dequant_kernel[(triton.cdiv(n, BN),)](_flat_u8(cache), token_ids, out, out.stride(0), n,
                                                   CELL=cell_bytes(rope), ROPE_FP8=(rope == "fp8"),
                                                   BLOCK_N=BN, num_warps=4)
    return out


def _num_splits(T: int, n_hblk: int, max_chunks: int, num_sms: int) -> int:
    want = max(1, math.ceil(2 * num_sms / max(1, T * n_hblk)))
    return max(1, min(want, max_chunks, 64))


_SM_COUNT: dict = {}


def kv4_sparse_decode(q: torch.Tensor, swa_cache: torch.Tensor, swa_idx: torch.Tensor,
                      swa_len: Optional[torch.Tensor], attn_sink: Optional[torch.Tensor],
                      softmax_scale: float, ext_cache: Optional[torch.Tensor] = None,
                      ext_idx: Optional[torch.Tensor] = None, ext_len: Optional[torch.Tensor] = None,
                      rope: str = "fp8", swa_fp8_page_size: Optional[int] = None,
                      block_h: int = 128, num_warps: int = 8, num_stages: int = 1, dv_split: Optional[int] = None,
                      nsplit: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode over 4-bit cells. q [T, H, 512] (or [T, 1, H, 512]); indices [T, W] / [T, 1, W].

    ``swa_fp8_page_size`` set: the SWA cache is the FlashMLA 584-byte paged layout (mode ``c4``), given as
    the pool's 2-D [num_pages, page_bytes] buffer; otherwise both caches hold 4-bit cells.
    Returns (out [T, H, 512] in q.dtype, lse [T, H] float32 natural log without the sink).
    """
    if q.ndim == 4:
        q = q.squeeze(1)
    T, H, dq = q.shape
    assert dq == D, dq
    if q.stride(2) != 1:
        q = q.contiguous()
    swa_idx = swa_idx.reshape(T, -1)
    W_SWA = swa_idx.shape[1]
    has_ext = ext_idx is not None
    if has_ext:
        ext_idx = ext_idx.reshape(T, -1)
        W_EXT = ext_idx.shape[1]
    else:
        W_EXT = 0
    BN = 64
    assert W_SWA % BN == 0 and W_EXT % BN == 0, (W_SWA, W_EXT)
    if swa_idx.stride(1) != 1:
        swa_idx = swa_idx.contiguous()
    if has_ext and ext_idx.stride(1) != 1:
        ext_idx = ext_idx.contiguous()
    n_hblk = triton.cdiv(H, block_h)
    if dv_split is None:
        dv_split = 2 if block_h >= 64 else 1
    dev = q.device
    if dev not in _SM_COUNT:
        _SM_COUNT[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    max_chunks = (W_SWA + W_EXT) // BN
    ns = nsplit or _num_splits(T, n_hblk * dv_split, max_chunks, _SM_COUNT[dev])
    po = torch.empty(T * ns * H, D, dtype=torch.float32, device=dev)
    pl = torch.empty(T * ns * H, dtype=torch.float32, device=dev)
    out = torch.empty(T, H, D, dtype=q.dtype, device=dev)
    lse = torch.empty(T, H, dtype=torch.float32, device=dev)
    if T == 0:
        return out, lse
    swa_fp8 = swa_fp8_page_size is not None
    swa_page_bytes = swa_cache.view(torch.uint8).reshape(swa_cache.shape[0], -1).stride(0) if swa_fp8 else 0
    dummy_i = swa_idx
    _kv4_decode_kernel[(T, n_hblk * dv_split, ns)](
        q, q.stride(0), q.stride(1),
        _flat_u8(swa_cache), swa_idx, swa_idx.stride(0), swa_len if swa_len is not None else dummy_i, W_SWA,
        _flat_u8(ext_cache) if has_ext else _flat_u8(swa_cache), ext_idx if has_ext else dummy_i,
        ext_idx.stride(0) if has_ext else 0, ext_len if ext_len is not None else dummy_i, W_EXT,
        po, pl, H, float(softmax_scale) * 1.4426950408889634, swa_fp8_page_size or 1, swa_page_bytes,
        SWA_FP8=swa_fp8, NSPLIT=ns, HAS_EXT=has_ext, HAS_SWA_LEN=swa_len is not None, HAS_EXT_LEN=ext_len is not None,
        CELL_SWA=cell_bytes(rope), CELL_EXT=cell_bytes(rope), ROPE_FP8=(rope == "fp8"),
        BLOCK_H=block_h, BLOCK_N=BN, DV_SPLIT=dv_split, num_warps=num_warps, num_stages=num_stages)
    _kv4_combine_kernel[(T, H)](po, pl, attn_sink if attn_sink is not None else pl, out, out.stride(0),
                                out.stride(1), lse, H, NSPLIT=ns, NSPLIT_P2=triton.next_power_of_2(ns),
                                HAS_SINK=attn_sink is not None, num_warps=4)
    return out, lse
