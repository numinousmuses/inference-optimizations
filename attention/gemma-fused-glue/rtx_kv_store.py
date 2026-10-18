"""Fused FP8 KV-cache store (rtxsys, Gemma 31B kernel program D1c, 2026-10-06).
Installed as sglang/srt/layers/numinous_kv_store.py by operations/patches/gemma31_kv_store_hooks.py.
Env NUMINOUS_KV_STORE_FUSED=1.

What it attacks. With --kv-cache-dtype fp8_e4m3, every attention layer's KV write (v0.5.20) is a chain of kernels:
- triton_backend.forward_extend (verify and prefill), triton_backend.py:1599-1607: k.clone(), v.clone();
- MHATokenToKVPool.set_kv_buffer, memory_pool.py:2563-2569: cache_k.div_(k_scale), cache_v.div_(v_scale),
  cache_k.to(fp8), cache_v.to(fp8);
- _store_kv_layer -> store_cache (the scatter).
Measured (rtxv41 prof_decode32_composed, 1xH100, 32 sessions, ms per verify step):
  direct_copy 120 calls 0.315, BUnaryFunctor (div) 120 calls 0.352, ConvertToFloat8E4M3fn 120 calls 0.322,
  store_kvcache 50 calls 0.114.
That is 1.10 ms and ~410 launches for a few MB. The 8K prefill chunk moves ~1 GB per layer this way (derived).

One Triton kernel per layer writes K and V. It reads each bf16 row once, divides by the scale and rounds like the
stock div_ (bf16(float(x) / float(scale)), IEEE division in fp32), converts with round-to-nearest-even to e4m3 like
.to(fp8), and scatters the bytes into pool row loc[n].
- It never writes to k / v, so the backend's protective clones are skipped (only where the pool is one this module
  serves).
- Overflow: stock .to(float8_e4m3fn) turns |x| > 464 into NaN; this saturates to +-448 (cvt.rn.satfinite). In
  range the bytes are identical.
- Slot 0 is skipped, like the stock store_cache (reserved_skip_index, default 0: the CUDA-graph padding slot).
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

ON = os.environ.get("NUMINOUS_KV_STORE_FUSED", "0") == "1"
EMU = os.environ.get("TRITON_INTERPRET", "0") == "1"   # CPU interpreter: emulate the GPU's RNE conversions
BLOCK = 1024


if triton is not None:

    @triton.jit
    def _rne_bf16(x):
        b = x.to(tl.uint32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return b.to(tl.float32, bitcast=True)

    @triton.jit
    def _e4m3_bits_emu(v):
        a = tl.minimum(tl.abs(v), 448.0)
        e = tl.maximum((a.to(tl.uint32, bitcast=True) >> 23) & 0xFF, 121)
        c = ((e + 20) << 23).to(tl.float32, bitcast=True)
        r = tl.minimum((a + c) - c, 448.0)
        b = r.to(tl.uint32, bitcast=True)
        er = (b >> 23) & 0xFF
        mag = tl.where(er >= 121, ((er - 120) << 3) | ((b >> 20) & 7), (r * 512.0).to(tl.uint32))
        sign = (v.to(tl.uint32, bitcast=True) >> 31) << 7
        return (sign | mag).to(tl.uint8)

    @triton.jit
    def _quant_bits(x, s, EMU: tl.constexpr):
        if EMU:
            y = _rne_bf16(x / s)                                # interpreter: IEEE division, emulated RNE
            return _e4m3_bits_emu(y)
        else:
            y = tl.math.div_rn(x, s).to(tl.bfloat16).to(tl.float32)   # stock div_: IEEE div, bf16 result
            y = tl.minimum(tl.maximum(y, -448.0), 448.0)
            return y.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    @triton.jit
    def _kv_store_kernel(K, V, KC, VC, LOC, KSP, VSP, ks, vs,
                         skn, skh, svn, svh, ROW_K, ROW_V, DK, DV, SKIP,
                         SCALE_PTR: tl.constexpr, BLOCK: tl.constexpr, EMU: tl.constexpr):
        n = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        dst = tl.load(LOC + n).to(tl.int64)
        if dst == SKIP:      # store_cache's reserved_skip_index: the CUDA-graph padding slot is never written
            return
        if SCALE_PTR:
            ksv = tl.load(KSP).to(tl.float32)
            vsv = tl.load(VSP).to(tl.float32)
        else:
            ksv = ks
            vsv = vs
        mk = c < ROW_K
        hk = c // DK
        k = tl.load(K + n * skn + hk * skh + (c - hk * DK), mask=mk, other=0.0).to(tl.float32)
        tl.store(KC + dst * ROW_K + c, _quant_bits(k, ksv, EMU), mask=mk)
        mv = c < ROW_V
        hv = c // DV
        v = tl.load(V + n * svn + hv * svh + (c - hv * DV), mask=mv, other=0.0).to(tl.float32)
        tl.store(VC + dst * ROW_V + c, _quant_bits(v, vsv, EMU), mask=mv)


def _scale_arg(s):
    """(is_ptr, tensor_or_None, float). The stock div_ by a CPU scalar computes in opmath (fp32) with the ORIGINAL
    scalar (opmath_symmetric_gpu_kernel_with_scalars reads iter.original_scalar_value), and a CUDA 0-dim tensor is
    loaded as fp32: neither is rounded to bf16 first. The served k_scale is 1.0, where all readings agree."""
    if s is None:
        return False, None, 1.0
    if isinstance(s, torch.Tensor):
        if s.is_cuda:
            return True, s.reshape(-1)[:1].to(torch.float32), 1.0
        s = float(s.reshape(-1)[0])
    return False, None, float(torch.tensor(float(s), dtype=torch.float32))


def store_fp8(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, loc: torch.Tensor,
              k_scale=None, v_scale=None, reserved_skip_index: int = 0) -> None:
    """k / v [N, H, D(v)] bf16|fp16 (last dim contiguous); k_cache / v_cache [S, H, D] fp8 or uint8 row-contiguous.
    Writes to loc == reserved_skip_index are skipped, like the stock store_cache (kvcache.py:56-90: default 0, the
    reserved CUDA-graph padding slot; -1 disables). Found from the bench_rtx_7aa9ede3 T=288 byte mismatch."""
    n = k.shape[0]
    if n == 0:
        return
    hk, dk = k.shape[1], k.shape[2]
    hv, dv = v.shape[1], v.shape[2]
    kc = k_cache.view(torch.uint8) if k_cache.dtype != torch.uint8 else k_cache
    vc = v_cache.view(torch.uint8) if v_cache.dtype != torch.uint8 else v_cache
    kp, kt, kf = _scale_arg(k_scale)
    vp, vt, vf = _scale_arg(v_scale)
    sp = kp or vp
    if sp:   # mixed: give both as tensors
        kt = kt if kt is not None else torch.full((1,), kf, dtype=torch.float32, device=k.device)
        vt = vt if vt is not None else torch.full((1,), vf, dtype=torch.float32, device=k.device)
    grid = (n, triton.cdiv(max(hk * dk, hv * dv), BLOCK))
    _kv_store_kernel[grid](k, v, kc, vc, loc, kt if sp else kc, vt if sp else kc, kf, vf,
                           k.stride(0), k.stride(1), v.stride(0), v.stride(1), hk * dk, hv * dv, dk, dv,
                           int(reserved_skip_index), SCALE_PTR=sp, BLOCK=BLOCK, EMU=EMU, num_warps=4)


# -------------------------------------------------------------------------------------------------- pool plumbing
_POOL_OK: dict = {}


def _pool_served(pool) -> bool:
    """A plain MHATokenToKVPool (exact type) in NHD layout with FP8 e4m3 storage."""
    key = id(pool)
    ok = _POOL_OK.get(key)
    if ok is None:
        try:
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        except Exception:
            MHATokenToKVPool = None
        ok = bool(MHATokenToKVPool is not None and type(pool) is MHATokenToKVPool
                  and getattr(pool, "dtype", None) == torch.float8_e4m3fn
                  and not getattr(pool, "use_hnd", False)
                  and getattr(pool, "kv_cache_layout", "nhd") in (None, "nhd")
                  and not getattr(pool, "is_quantized_kv_cache", False)
                  and triton is not None)
        _POOL_OK[key] = ok
    return ok


def try_store(pool, layer_id: int, loc, cache_k, cache_v, k_scale, v_scale, dcp_kv_mask) -> bool:
    """Hook body in MHATokenToKVPool.set_kv_buffer. True = stored (the caller returns), False = stock path."""
    if not ON or dcp_kv_mask is not None or not _pool_served(pool):
        return False
    if (not cache_k.is_cuda or cache_k.dim() != 3 or cache_v.dim() != 3 or cache_k.stride(-1) != 1
            or cache_v.stride(-1) != 1 or cache_k.dtype not in (torch.bfloat16, torch.float16)
            or cache_v.dtype != cache_k.dtype or loc.dim() != 1 or loc.shape[0] != cache_k.shape[0]):
        return False
    idx = layer_id - pool.start_layer
    kc, vc = pool.k_buffer[idx], pool.v_buffer[idx]
    if not (kc.is_contiguous() and vc.is_contiguous()):
        return False
    store_fp8(cache_k, cache_v, kc, vc, loc, k_scale, v_scale)
    return True


def skip_clone(pool) -> bool:
    """triton_backend.forward_extend: the protective k / v clones are only needed when set_kv_buffer may divide in
    place. Under ON every pool this module serves either stores fused or falls back out of place."""
    if not ON:
        return False
    subs = [getattr(pool, "full_kv_pool", None), getattr(pool, "swa_kv_pool", None)]
    subs = [p for p in subs if p is not None] or [pool]
    return all(_pool_served(p) for p in subs)
