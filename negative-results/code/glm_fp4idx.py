"""GLM-5.3 DSA lightning-indexer keys in 4 bits (agent glmidx).

Layout (the DeepGEMM / SGLang DeepSeek-V4 FP4 indexer layout, so DeepGEMM's SM100 `fp8_fp4_paged_mqa_logits` reads
it unchanged): per page of 64 tokens, 64 x 64 bytes of packed E2M1 values (two per byte, element 2i in the low
nibble) followed by 64 x 4 bytes of UE8M0 block scales (one byte per 32 elements, block j in byte j of the token's
int32). 68 B per token per indexer layer instead of the stock 132 B (128 FP8 E4M3 + one fp32 scale).

Scale rule (matches deep_gemm.utils.per_token_cast_to_fp4(use_ue8m0=True, gran_k=32) and SGLang's
quantize_fp4_indexer_tensor): s = 2^ceil(log2(max(amax / 6, 1e-4))), value code = round-to-nearest E2M1 of x / s
with SGLang's thresholds (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0).

Optional rotation: GLM's indexer rotates q and k with a 128-point Hadamard transform only when the fused indexer
path is off (SGLang's fusion drops it because q.k is invariant under an orthonormal rotation). FP4 block scaling
benefits from the rotation (it spreads channel outliers across the 32-element blocks), so the kernels can apply the
normalized Sylvester Hadamard H/sqrt(128) to both q and k (H is symmetric and H H = 128 I, so logits are unchanged in
exact arithmetic). Callers must rotate both sides or neither.

Kernels (Triton, SM100; CPU torch references for tests):
  quant_store_k(key, cache, loc, rotate)       bf16/fp32 keys -> FP4 rows in the paged cache (one launch)
  quant_q(q, rotate)                           bf16 or FP8 q [N, H, 128] -> (q_fp4 int8 [N, H, 64], q_sf int32 [N, H])
  gather_k(cache, page_table, seq_lens, ...)   paged FP4 rows -> contiguous (k_fp4 [T, 64], k_sf [T]) for prefill
  dequant rows / reference logits              torch, for tests
"""
from __future__ import annotations

import math

import torch

try:  # the engine and the dev box have Triton; the local CPU tests do not need it
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

HEAD_DIM = 128
GROUP = 32
PAGE = 64
PACKED = HEAD_DIM // 2            # 64 bytes of E2M1 values per token
SCALE_BYTES = HEAD_DIM // GROUP   # 4 UE8M0 bytes per token
ROW_BYTES = PACKED + SCALE_BYTES  # 68
FP8_ROW_BYTES = HEAD_DIM + 4      # 132, the stock layout

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def bytes_per_token(index_head_dim: int = HEAD_DIM) -> int:
    assert index_head_dim == HEAD_DIM, index_head_dim
    return ROW_BYTES


def hadamard_matrix(n: int = HEAD_DIM, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Sylvester Hadamard with entries +-1 (not normalized)."""
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


# ----------------------------------------------------------------------------------------------------------------
# torch references (CPU or GPU)
# ----------------------------------------------------------------------------------------------------------------

def _rotate_ref(x: torch.Tensor) -> torch.Tensor:
    h = hadamard_matrix(HEAD_DIM, device=x.device)
    return (x.float() @ h) * (HEAD_DIM ** -0.5)


def _e2m1_code_ref(v: torch.Tensor) -> torch.Tensor:
    a = v.abs().clamp(max=6.0)
    idx = torch.zeros_like(a, dtype=torch.int32)
    for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx += (a > t).to(torch.int32)
    sign = ((v < 0) & (idx != 0)).to(torch.int32)
    return idx | (sign << 3)


def quant_ref(x: torch.Tensor, rotate: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., 128] -> (packed uint8 [..., 64], scale exponents uint8 [..., 4])."""
    lead = x.shape[:-1]
    x = x.float().reshape(-1, HEAD_DIM)
    if rotate:
        x = _rotate_ref(x)
    g = x.view(-1, SCALE_BYTES, GROUP)
    sf = (g.abs().amax(-1) / 6.0).clamp(min=1e-4)
    bits = sf.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exp = exp + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exp = exp.clamp(1, 254)
    scale = (exp << 23).view(torch.float32)
    codes = _e2m1_code_ref(g / scale.unsqueeze(-1)).view(-1, HEAD_DIM)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed.view(*lead, PACKED), exp.to(torch.uint8).view(*lead, SCALE_BYTES)


def dequant_ref(packed: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    lead = packed.shape[:-1]
    p = packed.reshape(-1, PACKED).to(torch.int64)
    lut = E2M1_VALUES.to(packed.device)
    vals = torch.stack([lut[p & 0xF], lut[p >> 4]], -1).view(-1, SCALE_BYTES, GROUP)
    scale = (exps.reshape(-1, SCALE_BYTES).to(torch.int32) << 23).view(torch.float32)
    return (vals * scale.unsqueeze(-1)).view(*lead, HEAD_DIM)


def pack_sf_int32(exps: torch.Tensor) -> torch.Tensor:
    """uint8 [..., 4] -> int32 [...] (byte j = block j), the DeepGEMM packed-UE8M0 form."""
    e = exps.to(torch.int64)
    v = e[..., 0] | (e[..., 1] << 8) | (e[..., 2] << 16) | (e[..., 3] << 24)
    v = torch.where(v >= 2 ** 31, v - 2 ** 32, v)
    return v.to(torch.int32)


def store_ref(cache: torch.Tensor, loc: torch.Tensor, key: torch.Tensor, rotate: bool = False) -> None:
    """Reference write into a [pages, 64 * 68] uint8 cache."""
    packed, exps = quant_ref(key, rotate)
    loc = loc.long()
    page, off = loc // PAGE, loc % PAGE
    for j in range(loc.numel()):
        p, o = int(page[j]), int(off[j])
        cache[p, o * PACKED:(o + 1) * PACKED] = packed[j]
        cache[p, PAGE * PACKED + o * SCALE_BYTES: PAGE * PACKED + (o + 1) * SCALE_BYTES] = exps[j]


def read_rows_ref(cache: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """Dequantized fp32 keys [N, 128] of token slots `loc` from a [pages, 64 * 68] cache."""
    loc = loc.long()
    page, off = loc // PAGE, loc % PAGE
    c = cache.view(cache.shape[0], -1)
    idx_p = (off[:, None] * PACKED + torch.arange(PACKED, device=c.device)[None]).long()
    idx_s = (PAGE * PACKED + off[:, None] * SCALE_BYTES + torch.arange(SCALE_BYTES, device=c.device)[None]).long()
    packed = torch.gather(c[page], 1, idx_p)
    exps = torch.gather(c[page], 1, idx_s)
    return dequant_ref(packed, exps)


def logits_ref(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Indexer score: sum_h w[m, h] * relu(q[m, h] . k[n]).  q [M, H, D], k [N, D], w [M, H] -> [M, N] fp32."""
    s = torch.einsum("mhd,nd->mhn", q.float(), k.float()).clamp_min_(0)
    return torch.einsum("mhn,mh->mn", s, w.float())


# ----------------------------------------------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _e2m1_code(x):
        ax = tl.minimum(tl.abs(x), 6.0)
        idx = (ax > 0.25).to(tl.int32)
        idx += (ax > 0.75).to(tl.int32)
        idx += (ax > 1.25).to(tl.int32)
        idx += (ax > 1.75).to(tl.int32)
        idx += (ax > 2.5).to(tl.int32)
        idx += (ax > 3.5).to(tl.int32)
        idx += (ax > 5.0).to(tl.int32)
        sign = ((x < 0) & (idx != 0)).to(tl.int32)
        return idx | (sign << 3)

    @triton.jit
    def _quant_tile(x, ROWS: tl.constexpr):
        """x [ROWS, 128] fp32 -> (packed int32 codes [ROWS, 64] in 0..255, exps int32 [ROWS, 4])."""
        g = tl.reshape(x, (ROWS, 4, 32))
        amax = tl.max(tl.abs(g), axis=2)                                   # [ROWS, 4]
        sf = tl.maximum(amax / 6.0, 1.0e-4)
        bits = sf.to(tl.int32, bitcast=True)
        exp = (bits >> 23) & 0xFF
        exp += ((bits & 0x7FFFFF) != 0).to(tl.int32)
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        inv = ((254 - exp) << 23).to(tl.float32, bitcast=True)             # 2^-(exp-127), exact
        codes = _e2m1_code(g * inv[:, :, None])                            # [ROWS, 4, 32]
        codes = tl.reshape(codes, (ROWS, 64, 2))
        lo, hi = tl.split(codes)
        return lo | (hi << 4), exp

    @triton.jit
    def _load_rotated(x_ptr, rows, row_mask, stride_x, h_ptr, ROWS: tl.constexpr, ROTATE: tl.constexpr):
        cols = tl.arange(0, 128)
        x = tl.load(x_ptr + rows[:, None] * stride_x + cols[None, :], mask=row_mask[:, None], other=0.0)
        if ROTATE:
            h = tl.load(h_ptr + tl.arange(0, 128)[:, None] * 128 + cols[None, :])  # +-1 bf16, exact
            x = tl.dot(x.to(tl.bfloat16), h, out_dtype=tl.float32) * 0.08838834764831845
        else:
            x = x.to(tl.float32)
        return x

    @triton.jit
    def _quant_store_k_kernel(x_ptr, stride_x, h_ptr, cache_ptr, stride_page, loc_ptr, n_rows,
                              ROWS: tl.constexpr, ROTATE: tl.constexpr, PAGE_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        row_mask = rows < n_rows
        x = _load_rotated(x_ptr, rows, row_mask, stride_x, h_ptr, ROWS, ROTATE)
        packed, exp = _quant_tile(x, ROWS)
        loc = tl.load(loc_ptr + rows, mask=row_mask, other=0).to(tl.int64)
        page = loc // PAGE_SIZE
        off = loc - page * PAGE_SIZE
        base = page * stride_page
        pc = tl.arange(0, 64)
        tl.store(cache_ptr + base[:, None] + off[:, None] * 64 + pc[None, :], packed.to(tl.uint8),
                 mask=row_mask[:, None])
        sc = tl.arange(0, 4)
        tl.store(cache_ptr + base[:, None] + PAGE_SIZE * 64 + off[:, None] * 4 + sc[None, :], exp.to(tl.uint8),
                 mask=row_mask[:, None])

    @triton.jit
    def _quant_q_kernel(x_ptr, stride_x, h_ptr, out_ptr, sf_ptr, n_rows,
                        ROWS: tl.constexpr, ROTATE: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        row_mask = rows < n_rows
        x = _load_rotated(x_ptr, rows, row_mask, stride_x, h_ptr, ROWS, ROTATE)
        packed, exp = _quant_tile(x, ROWS)
        pc = tl.arange(0, 64)
        tl.store(out_ptr + rows[:, None].to(tl.int64) * 64 + pc[None, :], packed.to(tl.uint8), mask=row_mask[:, None])
        # [ROWS, 4] -> [ROWS, 2, 2]; splitting the last axis gives even = (e0, e2) and odd = (e1, e3)
        even, odd = tl.split(tl.reshape(exp, (ROWS, 2, 2)))
        e0, e2 = tl.split(even)
        e1, e3 = tl.split(odd)
        sf = e0 | (e1 << 8) | (e2 << 16) | (e3 << 24)
        tl.store(sf_ptr + rows, sf, mask=row_mask)

    @triton.jit
    def _gather_k_kernel(cache_ptr, stride_page, table_ptr, stride_table, cu_ptr, out_ptr, sf_ptr,
                         PAGE_SIZE: tl.constexpr):
        """One program per (request, page): copies the page's valid rows to out[cu[b] + p * 64 + i]."""
        b = tl.program_id(0)
        p = tl.program_id(1)
        start = tl.load(cu_ptr + b)
        end = tl.load(cu_ptr + b + 1)
        n = end - start
        if p * PAGE_SIZE < n:
            page = tl.load(table_ptr + b.to(tl.int64) * stride_table + p).to(tl.int64)
            i = tl.arange(0, PAGE_SIZE)
            valid = p * PAGE_SIZE + i < n
            dst = (start + p * PAGE_SIZE + i).to(tl.int64)
            pc = tl.arange(0, 64)
            v = tl.load(cache_ptr + page * stride_page + i[:, None] * 64 + pc[None, :], mask=valid[:, None])
            tl.store(out_ptr + dst[:, None] * 64 + pc[None, :], v, mask=valid[:, None])
            s = tl.load((cache_ptr + page * stride_page + PAGE_SIZE * 64).to(tl.pointer_type(tl.int32)) + i,
                        mask=valid)
            tl.store(sf_ptr + dst, s, mask=valid)


_H_CACHE: dict = {}


def _hadamard_bf16(device) -> torch.Tensor:
    key = str(device)
    if key not in _H_CACHE:
        _H_CACHE[key] = hadamard_matrix(HEAD_DIM, device=device).to(torch.bfloat16).contiguous()
    return _H_CACHE[key]


def quant_store_k(key: torch.Tensor, cache: torch.Tensor, loc: torch.Tensor, *, rotate: bool = False,
                  page_size: int = PAGE) -> None:
    """Quantize keys [N, 128] (bf16/fp16/fp32) and write them to token slots `loc` of a [pages, page*68] uint8
    cache. Graph-safe (no host sync, no allocation besides the cached Hadamard)."""
    assert key.shape[-1] == HEAD_DIM and cache.dtype == torch.uint8
    assert cache.dim() == 2 and cache.shape[1] == page_size * ROW_BYTES, cache.shape
    key = key.reshape(-1, HEAD_DIM)
    if key.stride(-1) != 1:
        key = key.contiguous()
    n = key.shape[0]
    if n == 0:
        return
    loc = loc.reshape(-1)
    assert loc.numel() >= n
    rows = 16
    _quant_store_k_kernel[(triton.cdiv(n, rows),)](
        key, key.stride(0), _hadamard_bf16(key.device), cache, cache.stride(0), loc, n,
        ROWS=rows, ROTATE=rotate, PAGE_SIZE=page_size, num_warps=4)


def quant_q(q: torch.Tensor, *, rotate: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """q [N, H, 128] in bf16 / fp16 / fp32 / float8_e4m3fn -> (q_fp4 int8 [N, H, 64], q_sf int32 [N, H])."""
    assert q.shape[-1] == HEAD_DIM
    lead = q.shape[:-1]
    x = q.reshape(-1, HEAD_DIM)
    if x.stride(-1) != 1:
        x = x.contiguous()
    n = x.shape[0]
    out = torch.empty((n, PACKED), dtype=torch.int8, device=q.device)
    sf = torch.empty((n,), dtype=torch.int32, device=q.device)
    if n:
        rows = 16
        _quant_q_kernel[(triton.cdiv(n, rows),)](
            x, x.stride(0), _hadamard_bf16(q.device), out, sf, n, ROWS=rows, ROTATE=rotate, num_warps=4)
    return out.view(*lead, PACKED), sf.view(*lead)


def gather_k(cache: torch.Tensor, page_table: torch.Tensor, seq_lens: torch.Tensor, total: int,
             max_seq_len: int, *, page_size: int = PAGE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contiguous FP4 keys of every request's first seq_lens[b] tokens, requests back to back.
    Returns (k_fp4 uint8 [total, 64], k_sf int32 [total], cu int32 [B + 1])."""
    b = seq_lens.shape[0]
    cu = torch.zeros(b + 1, dtype=torch.int32, device=cache.device)
    torch.cumsum(seq_lens.to(torch.int32), 0, out=cu[1:])
    k = torch.empty((total, PACKED), dtype=torch.uint8, device=cache.device)
    s = torch.empty((total,), dtype=torch.int32, device=cache.device)
    if total and b:
        pages = triton.cdiv(max_seq_len, page_size)
        _gather_k_kernel[(b, pages)](cache, cache.stride(0), page_table, page_table.stride(0), cu, k, s,
                                     PAGE_SIZE=page_size, num_warps=4)
    return k, s, cu
