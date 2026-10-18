"""Fused per-head q/k/v RMSNorm + neox RoPE for Gemma 4 attention (rtxsys, Gemma 31B kernel program D1d, 2026-10-06).
Installed as sglang/srt/layers/numinous_qkv_norm_rope.py by operations/patches/gemma31_qkv_norm_rope_hooks.py.
Env NUMINOUS_QKV_NORM_ROPE=1.

What it attacks:
- Gemma4Attention.forward (gemma4_causal.py:431-512, v0.5.20) runs gemma_qkv_rmsnorm (in place over the qkv buffer
  slices), then rotary_emb -> the JIT fused_rope_kernel (in place on q and k).
- Measured, rtxv41 prof_decode32_composed, per 32-session verify step:
  - _gemma_qkv_rmsnorm_kernel 80 calls 0.543 ms;
  - fused_rope_kernel 65 + 15 calls 0.292 ms.
- 8K prefill chunk: qkv-rmsnorm 26.4 ms + rope 9.2 ms.
- Kv-shared (draft) layers additionally build a zeros dummy K just to feed the rope (FillFunctor, 20 calls).

One Triton kernel per layer does both: one program per (token, head) reads the row once and writes it once.
- q / k: x -> bf16(x * rsqrt(mean(x^2) + eps) * w) [the stock norm store]
  -> neox rotation with the layer's fp32 cos_sin_cache row [the stock rope] -> bf16.
- v: x -> bf16(x * rsqrt(mean(x^2) + eps)) (v_norm has no weight); no rotation.
- Kv-shared layers: q only, no dummy K.
Math as stock:
- gemma4_fused_ops._gemma_qkv_rmsnorm_store: sum(x^2) / D, rsqrt, x * rrms * w in fp32, then a bf16 store.
- rope.cuh rope_rotate_head (neox): out_x = x * cos - y * sin, out_y = x * sin + y * cos in fp32, then bf16.
  x = row[:D/2], y = row[D/2:], cos / sin = cos_sin_cache[pos, :D/2] / [pos, D/2:].
- Gemma4RotaryEmbedding (full layers, proportional, partial 0.25) is the same neox formula over the full head with
  zero inverse frequencies on the non-rotated dims (cos 1, sin 0), so no special case.
Expected (derived): ~-0.25 ms per verify step, ~-8 ms per 8K prefill chunk.
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

ON = os.environ.get("NUMINOUS_QKV_NORM_ROPE", "0") == "1"
EMU = os.environ.get("TRITON_INTERPRET", "0") == "1"


if triton is not None:

    @triton.jit
    def _rne_bf16(x):
        b = x.to(tl.uint32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return b.to(tl.float32, bitcast=True)

    @triton.jit
    def _bf16_round(x, EMU: tl.constexpr):
        if EMU:
            return _rne_bf16(x)
        else:
            return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _norm_rope_head(X, W, CS, pos, row_off, eps, D: tl.constexpr, HALF: tl.constexpr, HAS_W: tl.constexpr,
                        ROPE: tl.constexpr, EMU: tl.constexpr):
        c = tl.arange(0, HALF)
        x1 = tl.load(X + row_off + c).to(tl.float32)
        x2 = tl.load(X + row_off + HALF + c).to(tl.float32)
        rr = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + eps)
        y1 = x1 * rr
        y2 = x2 * rr
        if HAS_W:
            y1 = y1 * tl.load(W + c).to(tl.float32)
            y2 = y2 * tl.load(W + HALF + c).to(tl.float32)
        y1 = _bf16_round(y1, EMU)            # the stock norm stores bf16 before the rope reads it
        y2 = _bf16_round(y2, EMU)
        if ROPE:
            cs = CS + pos * D
            cos = tl.load(cs + c)
            sin = tl.load(cs + HALF + c)
            o1 = y1 * cos - y2 * sin
            o2 = y1 * sin + y2 * cos
        else:
            o1 = y1
            o2 = y2
        if EMU:
            o1 = _rne_bf16(o1)
            o2 = _rne_bf16(o2)
        tl.store(X + row_off + c, o1.to(X.dtype.element_ty))
        tl.store(X + row_off + HALF + c, o2.to(X.dtype.element_ty))

    @triton.jit
    def _qkv_norm_rope_kernel(Q, K, V, QW, KW, CS, POS, sq, sk, sv, eps,
                              HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, HAS_KV: tl.constexpr,
                              EMU: tl.constexpr):
        m = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        pos = tl.load(POS + m).to(tl.int64)
        if h < HQ:
            _norm_rope_head(Q, QW, CS, pos, m * sq + h * D, eps, D, D // 2, True, True, EMU)
        elif HAS_KV:
            if h < HQ + HKV:
                _norm_rope_head(K, KW, CS, pos, m * sk + (h - HQ) * D, eps, D, D // 2, True, True, EMU)
            else:
                _norm_rope_head(V, QW, CS, pos, m * sv + (h - HQ - HKV) * D, eps, D, D // 2, False, False, EMU)


def norm_rope(q, k, v, q_w, k_w, cos_sin_cache, positions, hq, hkv, d, eps):
    """In place on q [M, hq*d] and, when given, k / v [M, hkv*d] (strided slices of one qkv buffer are fine)."""
    m = q.shape[0]
    if m == 0:
        return
    has_kv = k is not None
    grid = (m, hq + (2 * hkv if has_kv else 0))
    _qkv_norm_rope_kernel[grid](q, k if has_kv else q, v if has_kv else q, q_w, k_w if has_kv else q_w,
                                cos_sin_cache, positions, q.stride(0), k.stride(0) if has_kv else 0,
                                v.stride(0) if has_kv else 0, float(eps), HQ=hq, HKV=hkv if has_kv else 0, D=d,
                                HAS_KV=has_kv, EMU=EMU, num_warps=4 if d <= 256 else 8)


def ref(q, k, v, q_w, k_w, cos_sin_cache, positions, hq, hkv, d, eps):
    """Stock order: gemma_qkv_rmsnorm (bf16 store) then neox rope (fp32 math, bf16 store). Returns new tensors."""
    def nrm(x, w, h):
        xf = x.float().view(-1, h, d)
        y = xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) / d + eps)
        if w is not None:
            y = y * w.float()
        return y.to(x.dtype)

    def rope(x, h):
        xf = x.float().view(-1, h, d)
        cs = cos_sin_cache[positions.long()]
        cos, sin = cs[:, None, : d // 2], cs[:, None, d // 2:]
        x1, x2 = xf[..., : d // 2], xf[..., d // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype).view(x.shape[0], -1)

    qo = rope(nrm(q, q_w, hq).view(q.shape[0], -1), hq)
    if k is None:
        return qo, None, None
    ko = rope(nrm(k, k_w, hkv).view(k.shape[0], -1), hkv)
    vo = nrm(v, None, hkv).view(v.shape[0], -1)
    return qo, ko, vo


# ------------------------------------------------------------------------------------------------------ hook body
_logged = False


def try_qkv_norm_rope(attn, positions, q, k, v):
    """Gemma4Attention.forward hook body: (q 2-D, k 3-D | None, v 3-D | None) normed + roped in place, or None."""
    global _logged
    if not ON or triton is None:
        return None
    rot = getattr(attn, "rotary_emb", None)
    cs = getattr(rot, "cos_sin_cache", None)
    d = attn.head_dim
    if (not q.is_cuda or q.dtype != torch.bfloat16 or q.stride(-1) != 1 or q.dim() != 2
            or attn.q_norm.scale_shift != 0.0 or attn.k_norm.scale_shift != 0.0 or attn.v_norm.with_scale
            or cs is None or cs.dtype != torch.float32 or cs.dim() != 2 or cs.shape[-1] != d
            or not getattr(rot, "is_neox_style", False) or getattr(rot, "rotary_dim", d) != d
            or getattr(rot, "head_size", d) != d or d % 2 or positions.dim() != 1 or positions.shape[0] != q.shape[0]):
        return None
    kv_shared = attn.is_kv_shared_layer and attn.kv_shared_layer_index is not None
    if not kv_shared and (k is None or v is None or k.stride(-1) != 1 or v.stride(-1) != 1):
        return None
    norm_rope(q, None if kv_shared else k, None if kv_shared else v, attn.q_norm.weight.data,
              attn.k_norm.weight.data, cs, positions, attn.num_heads, attn.num_kv_heads, d, attn.q_norm.eps)
    if not _logged:
        _logged = True
        import logging

        logging.getLogger(__name__).info("numinous qkv norm+rope (D1d): ENGAGED (head_dim %d)", d)
    if kv_shared:
        return q, None, None
    return q, k.reshape(-1, attn.num_kv_heads, d), v.reshape(-1, attn.num_kv_heads, d)
