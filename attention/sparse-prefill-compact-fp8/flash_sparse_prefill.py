"""GLM-5.3-Flash sparse-MLA PREFILL on the FP8 no-RoPE pool (amdflash, Hopper kernel program 2026-10-05).

What the prefill role runs today, under rtxsys's hopper_fp8kv hooks:
- FlashMLA's sparse DECODE kernel, once per query row, in 2,048-row slices, each with its own get_mla_metadata.
- The stock prefill path (dequantize_k_cache_paged + flash_mla_sparse_fwd) asserts the 656-byte RoPE row
  (kernels/ops/attention/dsa/dequant_k_cache.py:184). Flash's pool row is the 528-byte V32_NO_ROPE format: 512 e4m3 +
  four fp32 scales.

This module runs the sparse PREFILL kernel instead, on a compact bf16 copy of exactly the rows the batch's requests own:
  1. locs = the live token locations of the batch's requests:
     - page_table_1 (else real_page_table expanded) row r, positions [0, len_r);
     - flattened on the device from metadata.cu_seqlens_k and the host int metadata.seq_lens_sum, so no host sync.
     It returns None (stock path) for expanded per-token tables (target verify / draft extend v2) and above
     NUMINOUS_FLASH_SPARSE_PREFILL_MAX_ROWS live rows.
  2. kv_c = dequant_528_rows(pool, locs)                          [N, 1, 512] bf16 (one Triton pass)
  3. inv[pool_loc] = compact row (pool-sized int32, built once per forward, shared by the 12 MLA layers);
     idx_c = inv[topk pool locations] with -1 kept; width padded with -1 to a multiple of 128
     (FlashMLA sm90 prefill phase1.cuh asserts topk % (2*B_TOPK) == 0)
  4. o = flash_mla_sparse_fwd(q, kv_c, idx_c, sm_scale, d_v)
Index semantics are untouched: the same pool locations the decode kernel would read, gathered into a compact buffer.

Measured on 1x H200 at the served shapes (research/gpu-evidence/amdflash/flash_sparse_mla_bench_h200.json):
- hooked decode path 3.18 ms per MLA layer at 4,096 query tokens -> prefill kernel 2.08 ms (0.65x);
- at 8,192: 6.40 -> 4.04 ms (0.63x);
- -13.2 / -28.3 ms per step (x12 layers);
- rel err vs a bf16 dense-gather reference 2.2e-3 (decode path 3.3e-3).

    o = maybe_forward(backend, q_all, kv_cache, v_head_dim, sm_scale, layer, metadata, page_table_1)   # None = stock
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None

ENV = "NUMINOUS_FLASH_SPARSE_PREFILL"
MIN_ROWS = int(os.environ.get("NUMINOUS_FLASH_SPARSE_PREFILL_MIN_ROWS", "256"))
MAX_ROWS = int(os.environ.get("NUMINOUS_FLASH_SPARSE_PREFILL_MAX_ROWS", str(1 << 20)))     # 1 GiB of bf16 rows
ROW_BYTES, D_NOPE, GROUP = 528, 512, 128


def dequant_528_ref(pool_u8: torch.Tensor, locs: torch.Tensor) -> torch.Tensor:
    """Reference: rows of the 528-byte FP8 layout -> [N, 1, 512] bf16."""
    raw = pool_u8.reshape(-1, ROW_BYTES)[locs.long()]
    vals = raw[:, :D_NOPE].contiguous().view(torch.float8_e4m3fn).float().view(-1, 4, GROUP)
    scale = raw[:, D_NOPE:].contiguous().view(torch.float32).view(-1, 4, 1)
    return (vals * scale).view(-1, 1, D_NOPE).to(torch.bfloat16)


def remap_indices(indices: torch.Tensor, locs: torch.Tensor, pool_rows: int, inv: torch.Tensor | None = None,
                  width_multiple: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """(idx_c, inv): pool-location top-k rows -> compact rows; -1 stays -1; width padded with -1 to the multiple."""
    if inv is None:
        inv = torch.full((pool_rows,), -1, dtype=torch.int32, device=indices.device)
        inv[locs.long()] = torch.arange(locs.numel(), dtype=torch.int32, device=indices.device)
    live = indices >= 0
    idx_c = torch.where(live, inv[indices.clamp(min=0).long()], torch.full_like(indices, -1))
    w = idx_c.shape[-1]
    w2 = (w + width_multiple - 1) // width_multiple * width_multiple
    if w2 != w:
        idx_c = torch.nn.functional.pad(idx_c, (0, w2 - w), value=-1)
    return idx_c.to(torch.int32).contiguous(), inv


if triton is not None:
    @triton.jit
    def _dequant_528_rows_kernel(pool_ptr, locs_ptr, out_ptr, N, BLOCK_R: tl.constexpr):
        pid = tl.program_id(0)
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        ok = r < N
        loc = tl.load(locs_ptr + r, mask=ok, other=0).to(tl.int64)
        for gi in tl.static_range(4):
            c = gi * 128 + tl.arange(0, 128)
            v8 = tl.load(pool_ptr + loc[:, None] * 528 + c[None, :], mask=ok[:, None], other=0)
            v = v8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            sp = (pool_ptr + loc * 528 + 512 + gi * 4).to(tl.pointer_type(tl.float32))
            s = tl.load(sp, mask=ok, other=0.0)
            tl.store(out_ptr + r.to(tl.int64)[:, None] * 512 + c[None, :], (v * s[:, None]).to(tl.bfloat16),
                     mask=ok[:, None])


def dequant_528_rows(pool_u8: torch.Tensor, locs: torch.Tensor, block_r: int = 16) -> torch.Tensor:
    pool = pool_u8.reshape(-1, ROW_BYTES)
    if pool.dtype != torch.uint8:
        pool = pool.view(torch.uint8)
    locs = locs.contiguous()
    n = locs.numel()
    out = torch.empty(n, D_NOPE, dtype=torch.bfloat16, device=pool.device)
    if n:
        _dequant_528_rows_kernel[(triton.cdiv(n, block_r),)](pool, locs, out, n, BLOCK_R=block_r, num_warps=4)
    return out.view(n, 1, D_NOPE)


def batch_locations(metadata, page_size: int, pool_rows: int, q_rows: int):
    """Live pool locations of the batch's requests, request-major, or None when this batch is not a plain extend.

    The output size is the host int metadata.seq_lens_sum, so nothing syncs. Expanded per-token tables (target verify,
    draft extend v2: one row per query token) and batches above MAX_ROWS live rows return None.
    """
    pt = getattr(metadata, "page_table_1", None)
    if pt is None:
        rpt = metadata.real_page_table                                   # [R, pages] page ids
        pt = (rpt.long()[:, :, None] * page_size + torch.arange(page_size, device=rpt.device)).reshape(rpt.shape[0], -1)
    total = getattr(metadata, "seq_lens_sum", None)
    cu = getattr(metadata, "cu_seqlens_k", None)
    n_req = pt.shape[0]
    if total is None or cu is None or cu.numel() != n_req + 1 or n_req >= q_rows or not 0 < total <= MAX_ROWS:
        return None
    j = torch.arange(total, device=pt.device, dtype=torch.int64)
    cu64 = cu.to(torch.int64)
    r = torch.searchsorted(cu64[1:].contiguous(), j, right=True).clamp_(max=n_req - 1)
    pos = (j - cu64[r]).clamp_(0, pt.shape[1] - 1)
    return pt[r, pos].long().clamp_(0, pool_rows - 1)


def usable(backend, q_all: torch.Tensor, kv_cache: torch.Tensor) -> bool:
    if os.environ.get(ENV) != "1" or triton is None:
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    rows = q_all.reshape(-1, q_all.shape[-2], q_all.shape[-1]).shape[0]
    return (rows >= MIN_ROWS and getattr(backend, "kv_cache_dim", None) == ROW_BYTES
            and q_all.shape[-1] == D_NOPE and getattr(backend, "dsa_kv_cache_store_fp8", True))


def maybe_forward(backend, q_all, kv_cache, v_head_dim, sm_scale, layer, metadata, page_table_1):
    """The prefill-kernel path for _forward_flashmla_kv, or None (stock path)."""
    if not usable(backend, q_all, kv_cache):
        return None
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    heads = layer.tp_q_head_num
    q = q_all.reshape(-1, heads, q_all.shape[-1]).contiguous()
    pool = kv_cache.reshape(-1, ROW_BYTES)
    pool_rows = pool.shape[0]
    cache = getattr(metadata, "_numinous_nsp", None)
    if cache is None or cache[0] != pool_rows:
        locs = batch_locations(metadata, getattr(backend, "real_page_size", 64), pool_rows, q.shape[0])
        if locs is None:
            return None
        inv = torch.full((pool_rows,), -1, dtype=torch.int32, device=q.device)
        inv[locs] = torch.arange(locs.numel(), dtype=torch.int32, device=q.device)
        cache = (pool_rows, locs, inv)
        try:
            metadata._numinous_nsp = cache                               # shared by this forward's MLA layers
        except Exception:  # noqa: BLE001
            pass
    _, locs, inv = cache
    kv_c = dequant_528_rows(pool, locs)
    idx_c, _ = remap_indices(page_table_1.unsqueeze(1), locs, pool_rows, inv=inv)
    out = flash_mla_sparse_fwd(q, kv_c, idx_c, sm_scale, v_head_dim)
    o = out[0] if isinstance(out, (tuple, list)) else out                # [rows, heads, d_v]
    return o.unsqueeze(1)
