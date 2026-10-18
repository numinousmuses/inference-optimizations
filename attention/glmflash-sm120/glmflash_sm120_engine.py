"""GLM-5.3-Flash DSA attention on SM120 inside SGLang v0.5.20 (installed by operations/patches/rtxflash_attn_hooks.py as
sglang/srt/layers/attention/dsa/glmflash_sm120_engine.py next to glmflash_sm120_attn.py, glmflash_sm120_fi.py,
glm_kv4.py and glmflash_kv4.py). Inert unless NUMINOUS_RTXFLASH_ATTN=1.

Launch surface (the DSA backend name is the stock `flashinfer_sparse_mla`, so SGLang's SM120 metadata / CUDA-graph
paths that GLM-5.3 already exercises on this GPU are the ones Flash takes):

    NUMINOUS_RTXFLASH_ATTN=1 ... --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla
        --kv-cache-dtype fp8_e4m3     -> 656-byte GLM_NSA rows (zero rope), FlashInfer SM120 kernels, top-k 2112 JIT
        --kv-cache-dtype fp8_e4m3 + NUMINOUS_RTXFLASH_KV=fp4  -> 288-byte E2M1 rows, our Triton kernel
        --kv-cache-dtype bfloat16     -> stock 1024-byte bf16 rows, our Triton kernel

On SM90 (H100/H200; agents/findings/hopper_fp8kv.md) the same surface serves Flash with FP8 KV, which stock SGLang
cannot: FA3's MLA (q_v) mode is fp16/bf16-only (sgl-attn hopper/flash_api.cpp) and FlashMLA cannot take the 2051-wide
kpool table (dsa_backend_kpool._check_kpool_tail_backend):
        --kv-cache-dtype fp8_e4m3     -> the STOCK rope-free 528-byte FP8 row (stock writer), our Triton kernel
        --kv-cache-dtype fp8_e4m3 + NUMINOUS_RTXFLASH_KV=fp4  -> 288-byte E2M1 rows, our Triton kernel
NUMINOUS_RTXFLASH_FP8_KERNEL must be unset or triton there.

Env: NUMINOUS_RTXFLASH_KV (auto | fp4), NUMINOUS_RTXFLASH_FP8_KERNEL (fi_jit | fi_stock | triton; default fi_jit,
falls back to fi_stock with a loud log if the JIT build fails), NUMINOUS_RTXFLASH_BLOCK_N / _WARPS / _STAGES (Triton
tiles, all three to override; default = TILE_TABLE, measured per format and T on the RTX PRO 6000).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

try:
    from sglang.srt.layers.attention.dsa import glmflash_sm120_attn as A
    from sglang.srt.layers.attention.dsa import glmflash_sm120_fi as FI
except ImportError:  # repo checkout (tests)
    import glmflash_sm120_attn as A
    import glmflash_sm120_fi as FI

logger = logging.getLogger("sglang.numinous")

ENV = "NUMINOUS_RTXFLASH_ATTN"
KV_ENV = "NUMINOUS_RTXFLASH_KV"
FP8_KERNEL_ENV = "NUMINOUS_RTXFLASH_FP8_KERNEL"
ARCH_PREFIX = "Glm5Next"
IMPL = "flashinfer_sparse_mla"
_logged: set = set()


def log_once(key: str, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.warning(msg, *args)


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def is_flash_arch(arch) -> bool:
    return str(arch).startswith(ARCH_PREFIX)


def _archs(hf_config):
    return getattr(hf_config, "architectures", None) or []


SM_ENV = "NUMINOUS_RTXFLASH_SM"   # tests / CPU tools: pretend this SM major (the engine reads the device otherwise)


def sm_major() -> int:
    """SM major of the serving device: 12 (RTX PRO 6000) or 9 (H100/H200)."""
    env = os.environ.get(SM_ENV)
    if env:
        return int(env)
    if torch.cuda.is_available():
        return int(torch.cuda.get_device_capability()[0])
    return 12


def fmt_for(kv_cache_dtype, major: Optional[int] = None) -> int:
    """Row format for a KV dtype flag (torch dtype or string). FP8 rows: SM120 = 656-B GLM_NSA rows (FlashInfer's
    SM120 kernels need them); SM90 = the STOCK rope-free 528-B row (no vendor SM90 kernel takes the 2051-wide kpool
    table, our Triton kernel reads 528 directly, and the stock writer stays)."""
    s = str(kv_cache_dtype)
    if "bfloat16" in s or s in ("bf16", "auto"):
        return A.FMT_BF16
    if "float8_e4m3" in s or "fp8_e4m3" in s:
        kv = os.environ.get(KV_ENV, "auto")
        if kv not in ("auto", "fp4"):
            raise ValueError(f"{KV_ENV}={kv!r}: expected auto or fp4")
        if kv == "fp4":
            return A.FMT_FP4_288
        return A.FMT_FP8_528 if (sm_major() if major is None else major) == 9 else A.FMT_FP8_656
    raise ValueError(f"numinous rtxflash: unsupported kv cache dtype {kv_cache_dtype}")


FP8_SCALE_ENV = "NUMINOUS_RTXFLASH_FP8_SCALE"


def fp8_pow2() -> bool:
    """656-B row scales: pow2 (default since 2026-10-01: FlashInfer DSV3_2 kernels, block-scaled QK + single-pass P) or
    arbitrary (GLM_NSA kernels, the GLM-5.3 writer convention). Read once by the pool writer AND the reader; P and D
    servers of a PD pair must agree (rows are transferred as bytes)."""
    m = os.environ.get(FP8_SCALE_ENV, "pow2")
    if m not in ("pow2", "arbitrary"):
        raise ValueError(f"{FP8_SCALE_ENV}={m!r}: expected pow2 or arbitrary")
    return m == "pow2"


def fp8_kernel() -> str:
    m = os.environ.get(FP8_KERNEL_ENV, "fi_jit")
    if m not in ("fi_jit", "fi_stock", "triton"):
        raise ValueError(f"{FP8_KERNEL_ENV}={m!r}: expected fi_jit, fi_stock or triton")
    return m


# Measured on 1 x RTX PRO 6000 (2026-09-30, research/gpu-evidence/rtxattn/rtxattn_tiles.json, H=16, 2.4M-row pool,
# CUDA-graph replay): fastest (BLOCK_N, warps, stages) per format by query rows T. bn64_w8_s1 wins from T >= 16 for the
# quantized rows although ptxas reports a small spill for it; small T prefers short tiles (more CTAs per row).
TILE_TABLE = {  # fmt -> [(max_T, (block_n, num_warps, num_stages)), ...]
    0: [(16, (16, 4, 2)), (512, (64, 8, 1)), (1 << 30, (32, 8, 3))],   # bf16
    1: [(16, (16, 4, 2)), (1 << 30, (64, 8, 1))],                      # fp8-656
    2: [(1, (16, 4, 2)), (8, (32, 8, 2)), (1 << 30, (64, 8, 1))],      # fp4-288
    3: [(16, (16, 4, 2)), (1 << 30, (64, 8, 1))],                      # fp8-528 (SM90): the fp8-656 table, untuned
}


def tiles(fmt: int = 1, t: int = 64) -> dict:
    """Tile config: env override (NUMINOUS_RTXFLASH_BLOCK_N/_WARPS/_STAGES, all three) else the measured table;
    NUMINOUS_RTXFLASH_BLOCK_H (optional, default 16) sets the heads per CTA. On SM90 (227 KB smem) BLOCK_H 32 reads each
    row once for 32 heads/rank instead of twice (CPU sm_90a compile: (32, 8, 2) spills 120 B, the SM120 default
    (64, 8, 1) at BLOCK_H 16 spills 256 B; research/gpu-evidence/rtxattn_triton_sm90_compile.json)."""
    bh = os.environ.get("NUMINOUS_RTXFLASH_BLOCK_H")
    extra = {"block_h": int(bh)} if bh else {}
    env = [os.environ.get(k) for k in ("NUMINOUS_RTXFLASH_BLOCK_N", "NUMINOUS_RTXFLASH_WARPS", "NUMINOUS_RTXFLASH_STAGES")]
    if all(env):
        return {"block_n": int(env[0]), "num_warps": int(env[1]), "num_stages": int(env[2]), **extra}
    for max_t, (bn, nw, ns) in TILE_TABLE[fmt]:
        if t <= max_t:
            return {"block_n": bn, "num_warps": nw, "num_stages": ns, **extra}
    raise AssertionError


# ------------------------------------------------------------------------------------------------ server-arg hooks


def validate(model_arch, device_sm_major, kv_cache_dtype, prefill_impl, decode_impl) -> bool:
    """flash_mla_sm120._validate_flashinfer_sparse_mla_backend hook: True = accept (Flash on SM12x or SM90 through
    our engine). On SM90 every non-bf16 row is read by our Triton kernel (FlashInfer's GLM_NSA kernels are SM120)."""
    if not enabled() or not is_flash_arch(model_arch):
        return False
    if device_sm_major not in (9, 12):
        raise ValueError(f"{ENV}=1 serves Glm5Next DSA on SM90 / SM120 / SM121 only; got sm_major {device_sm_major}")
    if device_sm_major == 9 and os.environ.get(FP8_KERNEL_ENV, "triton") != "triton":
        raise ValueError(f"{ENV}=1 on SM90: {FP8_KERNEL_ENV} must be triton (FlashInfer GLM_NSA kernels are SM120)")
    if {prefill_impl, decode_impl} != {IMPL}:
        raise ValueError(f"{ENV}=1 needs --dsa-prefill-backend {IMPL} --dsa-decode-backend {IMPL}; got "
                         f"{prefill_impl!r} / {decode_impl!r}")
    fmt_for(kv_cache_dtype, device_sm_major)  # raises on anything but bf16 / fp8
    return True


def kv_cache_dim(model_config, kv_cache_dtype) -> Optional[int]:
    """calculate_mla_kv_cache_dim hook: the row width for Flash's DSA pool (None = stock)."""
    if not enabled() or not any(is_flash_arch(a) for a in _archs(model_config.hf_config)):
        return None
    assert model_config.kv_lora_rank == 512 and model_config.qk_rope_head_dim == 0, (
        model_config.kv_lora_rank, model_config.qk_rope_head_dim)
    f = fmt_for(kv_cache_dtype)
    if f in (A.FMT_BF16, A.FMT_FP8_528):
        return None  # stock rows: bf16 kv_lora_rank + 0, or the stock rope-free FP8 row (528 = 512 + 16 + 0)
    return A.ROW_BYTES[f]


def pool_class(default_cls, kv_dim):
    """DSA pool class swap (HybridLinearKVPool and _build_dsa_kv_pool hooks): only when the dim hook priced our rows."""
    if not enabled():
        return default_cls
    if kv_dim == A.NSA_ROW:
        return _pool_classes()[0]
    if kv_dim == A.FP4_ROW:
        return _pool_classes()[1]
    return default_cls


_POOLS = None


def _pool_classes():
    global _POOLS
    if _POOLS is not None:
        return _POOLS
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    class RtxFlashNSA656Pool(DSATokenToKVPool):
        """656-byte GLM_NSA rows with a zero rope; writes never touch slot 0 and never emit NaN codes."""
        numinous_rtx_fmt = A.FMT_FP8_656

        def __init__(self, *args, **kwargs):
            assert kwargs.get("kv_cache_dim") == A.NSA_ROW, kwargs.get("kv_cache_dim")
            super().__init__(*args, **kwargs)
            assert self.kv_lora_rank == 512 and self.qk_rope_head_dim == 0
            assert self.dsa_kv_cache_store_fp8 and self.kv_cache_dim == A.NSA_ROW
            assert (self.size + self.page_size) % 64 == 0, (self.size, self.page_size)
            self.rows_pow2 = fp8_pow2()
            logger.warning("numinous rtxflash pool: %d DSA layers x 656 B/token (%s row, zero rope, %s scales) = %d "
                           "B/token; %d tokens", self.layer_num, "DSV3_2" if self.rows_pow2 else "GLM_NSA",
                           "pow2" if self.rows_pow2 else "arbitrary", 656 * self.layer_num, self.size)

        def _write_mla_kv_buffer(self, dst_buffer, loc, cache_k_nope, cache_k_rope) -> None:
            from sglang.srt.runtime_context import get_parallel

            assert not get_parallel().dcp_enabled, "rtxflash 656 pool has no DCP write path"
            assert cache_k_rope is None or cache_k_rope.numel() == 0, "Flash DSA rows have no rope part"
            A.nsa656_quant_write(dst_buffer, loc, cache_k_nope.reshape(cache_k_nope.shape[0], 512), pow2=self.rows_pow2)

    class RtxFlashFP4Pool(DSATokenToKVPool):
        """288-byte E2M1 rows (glmflash_kv4 layout)."""
        numinous_rtx_fmt = A.FMT_FP4_288

        def __init__(self, *args, **kwargs):
            assert kwargs.get("kv_cache_dim") == A.FP4_ROW, kwargs.get("kv_cache_dim")
            super().__init__(*args, **kwargs)
            self.dsa_kv_cache_store_fp8 = False  # not the 656 scaled layout
            assert self.store_dtype == torch.uint8 and self.kv_cache_dim == A.FP4_ROW
            assert self.kv_lora_rank == 512 and self.qk_rope_head_dim == 0
            logger.warning("numinous rtxflash pool: %d DSA layers x 288 B/token (E2M1 + E4M3/16) = %d B/token; %d tokens",
                           self.layer_num, 288 * self.layer_num, self.size)

        def _write_mla_kv_buffer(self, dst_buffer, loc, cache_k_nope, cache_k_rope) -> None:
            from sglang.srt.runtime_context import get_parallel

            try:
                from sglang.srt.layers.attention.dsa import glmflash_kv4 as kv4
            except ImportError:
                import glmflash_kv4 as kv4
            assert not get_parallel().dcp_enabled, "rtxflash fp4 pool has no DCP write path"
            assert cache_k_rope is None or cache_k_rope.numel() == 0, "Flash DSA rows have no rope part"
            kv4.quant_write(dst_buffer, loc, cache_k_nope.reshape(cache_k_nope.shape[0], 512))

        def set_kv_buffer(self, *args, **kwargs):
            raise RuntimeError("rtxflash fp4 pool: use set_mla_kv_buffer")

        def get_value_buffer(self, layer_id: int):
            raise RuntimeError("rtxflash fp4 pool: latent is FP4; read through glmflash_sm120_engine")

    _POOLS = (RtxFlashNSA656Pool, RtxFlashFP4Pool)
    return _POOLS


# ------------------------------------------------------------------------------------------------ PD (disaggregation)
# PD prefill worker + EAGLE IndexShare (Flash: index_share_for_mtp_iteration): SGLang builds the draft-extend DSA backend
# with use_fused_topk=False (dsa/utils.should_use_dsa_fused_topk) because the draft's top-k is sent to the decode worker
# as request-relative positions (speculative/eagle_disaggregation.py remaps them to decode-local slots). The non-kpool
# top-k backend honours that through DSAIndexerMetadata.force_unfused_topk; the kpool indexer (dsa_indexer_kpool.py)
# checks only SGLANG_DSA_FUSE_TOPK, so it still emits page-mapped slots, and the backend then runs the stock
# transform_index on them: assert width == 2048 (the 2051-wide kpool table; rtxsys G3, 2026-10-01) and, past the
# assert, a double page-table mapping. Hooks rtxattn_kpool_unfused_* make the kpool indexer honour the flag (raw
# positions, as the seed contract needs); rtxattn_transform_* let the width-generic transform kernels take 2051.


def kpool_unfused(metadata) -> bool:
    """dsa_indexer_kpool hook: True = emit raw request-relative positions (the backend maps them, PD seeds stay raw)."""
    return enabled() and bool(getattr(metadata, "force_unfused_topk", False))


def wide_topk_ok(width: int) -> bool:
    """transform_index prefill hook: the Triton prefill transform is width-generic (TOPK constexpr + masks); stock
    asserts 2048. Accept the kpool width (2048 + kpool - 1 = 2051) only under the engine."""
    return enabled() and 0 < int(width) <= 4096


def transform_decode_any(page_table: torch.Tensor, topk_indices: torch.Tensor,
                         result: Optional[torch.Tensor] = None) -> torch.Tensor:
    """transform_index decode hook for widths != 2048 (the stock decode kernel hard-codes TOPK 2048): result[i, j] =
    page_table[i, topk[i, j]] or -1; gather + where only (CUDA-graph safe, no boolean-mask indexing)."""
    if not enabled():
        raise AssertionError(f"topk width {topk_indices.shape[1]} != 2048 (stock transform_index_page_table_decode)")
    assert page_table.shape[0] == topk_indices.shape[0]
    idx = topk_indices.clamp(min=0).to(torch.int64)
    vals = torch.gather(page_table, 1, idx).to(torch.int32)
    out = torch.where(topk_indices >= 0, vals, torch.full_like(vals, -1))
    if result is None:
        return out
    result.copy_(out)
    return result


# ------------------------------------------------------------------------------------------------ SM90 prefill
# Measured 2026-10-04 (flash-h200-fp8kv, 4xH200, OR shape; engine logs): prefill chunks of >= 8K new tokens ran at a
# median 7,835 tok/s through the Triton sparse-MLA kernel vs 12,862 tok/s through stock FA3 bf16 (-39%), the cause of
# FP8's -17% per user at 16 sessions. Decode stays on Triton (memory-bound: FP8 halves the bytes); PREFILL rows go
# through FA3 bf16 on a gather-dequantized workspace of exactly the rows the chunk's tables reference -- the same
# numerics as the FP8 decode path (dequantized FP8 values, fp32 softmax), the speed of the stock bf16 prefill.
PREFILL_FA3_ENV = "NUMINOUS_RTXFLASH_PREFILL_FA3"


def prefill_min_rows() -> int:
    return int(os.environ.get("NUMINOUS_RTXFLASH_PREFILL_MIN_ROWS", "512"))


def prefill_fa3_enabled() -> bool:
    return os.environ.get(PREFILL_FA3_ENV, "0") == "1"


def gather_dequant_rows(kv_cache: torch.Tensor, fmt: int, slots: torch.Tensor) -> torch.Tensor:
    """bf16 [n, 512] latent of the pool rows `slots` (int64 [n]) for any row format."""
    rows = A.pool_rows(kv_cache, fmt)
    r = rows[slots]
    if fmt == A.FMT_BF16:
        return r
    n = r.shape[0]
    if fmt in A.FP8_FMTS:
        codes = r[:, :512].contiguous().view(torch.float8_e4m3fn).to(torch.float32).view(n, 4, 128)
        sc = r[:, 512:528].contiguous().view(torch.float32).view(n, 4, 1)
        return (codes * sc).view(n, 512).to(torch.bfloat16)
    if fmt == A.FMT_FP4_288:
        return A.fp4_288_dequant_ref(r).to(torch.bfloat16)
    raise ValueError(fmt)


def compact_slots(idx: torch.Tensor, n_rows: int):
    """Union of the pool slots `idx` [T, K] references (-1 = none) and the table remapped onto that union:
    returns (slots int64 [n], table int32 [T, K] in [0, n) or -1)."""
    valid = (idx >= 0) & (idx < n_rows)
    mask = torch.zeros(n_rows, dtype=torch.bool, device=idx.device)
    mask[idx[valid].long()] = True
    slots = mask.nonzero(as_tuple=True)[0]
    remap = torch.full((n_rows,), -1, dtype=torch.int32, device=idx.device)
    remap[slots] = torch.arange(slots.numel(), dtype=torch.int32, device=idx.device)
    table = torch.where(valid, remap[idx.clamp(min=0, max=n_rows - 1).long()], torch.full_like(idx, -1))
    return slots, table.to(torch.int32)


def _fa3_attn(q, ws, table, lens, sm_scale):
    from sglang.kernels.ops.attention.flash_attention_v3 import flash_attn_with_kvcache

    t = q.shape[0]
    cu_q = torch.arange(t + 1, dtype=torch.int32, device=q.device)
    cu_k = torch.zeros(t + 1, dtype=torch.int32, device=q.device)
    cu_k[1:] = torch.cumsum(lens, 0)
    return flash_attn_with_kvcache(
        q=None, k_cache=None, v_cache=ws.view(ws.shape[0], 1, 1, 512), qv=q, only_qv=True,
        page_table=table.clamp(min=0), cache_seqlens=lens, cu_seqlens_q=cu_q, cu_seqlens_k_new=cu_k,
        max_seqlen_q=1, softmax_scale=sm_scale, causal=True, return_softmax_lse=False)


def prefill_fa3(q: torch.Tensor, kv_cache: torch.Tensor, idx: torch.Tensor, sm_scale: float, fmt: int,
                attn=None) -> torch.Tensor:
    """q [T, H, 512] bf16, idx [T, K] compact kpool tables (valid entries first, -1 padded) -> out [T, H, 512] bf16."""
    n_rows = A.pool_rows(kv_cache, fmt).shape[0]
    slots, table = compact_slots(idx, n_rows)
    lens = (table >= 0).sum(1).to(torch.int32)
    if slots.numel() == 0:
        return torch.zeros(q.shape[0], q.shape[1], 512, dtype=torch.bfloat16, device=q.device)
    ws = gather_dequant_rows(kv_cache, fmt, slots)
    out = (attn or _fa3_attn)(q.contiguous(), ws, table, lens, sm_scale)
    out = out.reshape(q.shape[0], q.shape[1], 512)
    if bool((lens == 0).any()):
        out = torch.where((lens == 0)[:, None, None], torch.zeros_like(out), out)
    return out


# ------------------------------------------------------------------------------------------------ backend engine


class Engine:
    """Attached to DeepseekSparseAttnBackend as `_numinous_rtx`; `forward` replaces _forward_flashinfer_sparse_mla."""

    def __init__(self, fmt: int, num_heads: int, fp8_mode: str, fi=None):
        self.fmt = fmt
        self.num_heads = num_heads
        self.fp8_mode = fp8_mode
        self.fi = fi
        self.tiles = None  # None = the measured per-T table; tests may pin a dict
        self.ws = A.Workspace()
        # SM90 only: prefill rows through gather-dequant + FA3 bf16 (decode/verify stay on the Triton kernel)
        self.prefill_fa3 = prefill_fa3_enabled() and sm_major() == 9

    def forward(self, q_all: torch.Tensor, kv_cache: torch.Tensor, page_table_1: torch.Tensor, sm_scale: float,
                workspace_buffer: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = q_all.reshape(q_all.shape[0], -1, q_all.shape[-1])[:, :, :512]
        idx = page_table_1.reshape(q.shape[0], -1)
        if idx.dtype != torch.int32:
            idx = idx.to(torch.int32)
        if idx.stride(-1) != 1:
            idx = idx.contiguous()
        if self.fmt == A.FMT_FP8_656 and self.fp8_mode != "triton":
            self.fi.workspace = workspace_buffer
            out, _ = self.fi.forward(q, kv_cache, idx, sm_scale)
            return out
        if self.prefill_fa3 and q.shape[0] >= prefill_min_rows():
            return prefill_fa3(q, kv_cache, idx, sm_scale, self.fmt)
        if workspace_buffer is not None:
            self.ws.buf = workspace_buffer
        cfg = self.tiles or tiles(self.fmt, q.shape[0])
        out, _ = A.sparse_mla(q, kv_cache, idx, sm_scale, self.fmt, workspace=self.ws, **cfg)
        return out


def numinous_preflight(ctx) -> list:
    """CPU preflight (operations/hopper/hook_preflight.py, k3amd): validate()'s and attach()'s boot-time checks on the
    resolved server args (DSA roles, device, KV dtype, the FP8 kernel env, FlashInfer head counts on the 656-B row)."""
    if not enabled():
        return []
    archs = list(getattr(ctx.hf_config, "architectures", None) or [])
    if not archs or not is_flash_arch(archs[0]):
        return []
    sm = ctx.device.get("sm_major")
    pre, dec = ctx.args.get("dsa_prefill_backend"), ctx.args.get("dsa_decode_backend")
    try:
        validate(archs[0], sm, ctx.kv_cache_torch_dtype, pre, dec)
        fmt = fmt_for(ctx.kv_cache_torch_dtype, sm)
        fp8_pow2()
        mode = fp8_kernel() if fmt == A.FMT_FP8_656 else "triton"
    except ValueError as e:
        return [str(e)]
    out = []
    if pre != IMPL or dec != IMPL:
        out.append(f"numinous rtxflash: DSA roles must both be {IMPL}; got {pre} / {dec}")
    heads = int(getattr(ctx.hf_config, "num_attention_heads", 0) or 0) // max(1, int(ctx.args.get("tp_size") or 1)
                                                                             // max(1, int(ctx.args.get("dp_size") or 1)
                                                                                    if ctx.args.get("enable_dp_attention")
                                                                                    else 1))
    if fmt == A.FMT_FP8_656 and mode != "triton" and heads and heads not in FI.HEADS:
        out.append(f"numinous rtxflash: FlashInfer GLM_NSA has kernels for heads {FI.HEADS}, got {heads}")
    return out


def attach(backend, model_runner) -> Optional[Engine]:
    """DSA backend __init__ hook. Returns the engine (or None when inert / not Flash)."""
    if not enabled():
        return None
    arch = model_runner.model_config.hf_config.architectures[0]
    if not is_flash_arch(arch):
        return None
    if backend.dsa_prefill_impl != IMPL or backend.dsa_decode_impl != IMPL:
        raise RuntimeError(f"numinous rtxflash: DSA roles must both be {IMPL}; got {backend.dsa_prefill_impl} / "
                           f"{backend.dsa_decode_impl}")
    fmt = fmt_for(model_runner.kv_cache_dtype)
    pool = getattr(backend.token_to_kv_pool, "full_kv_pool", backend.token_to_kv_pool)
    if fmt == A.FMT_BF16:
        assert pool.kv_cache_dim == 512 and not pool.dsa_kv_cache_store_fp8, (pool.kv_cache_dim,)
    elif fmt == A.FMT_FP8_528:
        assert pool.dsa_kv_cache_store_fp8 and pool.kv_cache_dim == A.STOCK_FP8_ROW, (
            f"numinous rtxflash: SM90 FP8 expects the stock 528-B DSA pool, got {type(pool).__name__} "
            f"dim {pool.kv_cache_dim} fp8 {pool.dsa_kv_cache_store_fp8}")
    else:
        assert getattr(pool, "numinous_rtx_fmt", None) == fmt, (
            f"numinous rtxflash: pool {type(pool).__name__} dim {pool.kv_cache_dim} is not the {A.FMT_NAMES[fmt]} pool "
            "(dim / pool-class hooks missing?)")
    h = backend.num_q_heads
    mode = fp8_kernel() if fmt == A.FMT_FP8_656 else "triton"   # FMT_FP8_528 (SM90) is Triton-only
    fi = None
    if fmt == A.FMT_FP8_656 and mode != "triton":
        if h not in FI.HEADS:
            raise RuntimeError(f"numinous rtxflash: FlashInfer GLM_NSA has kernels for heads {FI.HEADS}, got {h}")
        fi_mode = "jit" if mode == "fi_jit" else "stock"
        if fi_mode == "jit":
            try:
                FI.build_module()
            except Exception as e:  # loud fallback: the stock module + Triton tail is exact, only slower
                logger.error("numinous rtxflash: widened FlashInfer build failed (%s); falling back to fi_stock", e)
                fi_mode, mode = "stock", "fi_stock"
        pow2 = bool(getattr(pool, "rows_pow2", False))
        fi = FI.FlashInferGlm5Next(fi_mode, pow2=pow2)
    logger.warning("numinous rtxflash attention: arch %s heads/rank %d rows %s%s kernel %s tiles %s", arch, h,
                   A.FMT_NAMES[fmt], ("/pow2" if getattr(pool, "rows_pow2", False) else "/arbitrary")
                   if fmt == A.FMT_FP8_656 else "", mode, "measured per-T table" if not os.environ.get("NUMINOUS_RTXFLASH_BLOCK_N") else tiles(fmt))
    eng = Engine(fmt, h, mode, fi)
    if eng.prefill_fa3:
        logger.warning("numinous rtxflash attention: prefill rows >= %d via gather-dequant + FA3 bf16 (SM90); decode and "
                       "verify on the Triton kernel", prefill_min_rows())
    return eng
