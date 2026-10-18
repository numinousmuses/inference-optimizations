"""k3state: quantized storage for the Kimi K3 KDA recurrent state (FP8 and NVFP4-style 4-bit), Triton kernels.

The KDA state of one head is a [V=128, K=128] fp32 matrix stored K-contiguous (SGLang's [HV, V, K] layout). Here a
"row" is one v index: 128 K-contiguous values. A quantized row is a self-contained byte record, so every byte copy the
engine does on the pool (radix copy-on-write, track snapshots, clear-to-zero, host offload) moves values and scales
together and needs no knowledge of the format:

  fp8   132 B/row : [0,128)  e4m3 codes            value = e4m3 * s
                    [128,132) fp32 s = amax_row / 448
  bf16  256 B/row : 128 bf16 values (round-to-nearest-even), no scale
  fp4    76 B/row : [0,64)   e2m1 codes, byte j = code(k=2j) | code(k=2j+1) << 4
                    [64,72)  e4m3 block scales, one per 16 K values (NVFP4 block size)
                    [72,76)  fp32 row scale g = amax_row / (6 * 448)
                    value = e2m1 * e4m3_block * g

A zero row (clear_slots writes zeros) decodes to 0 in both formats.

Bytes per 128x128 head: fp32 65,536; bf16 32,768 (2x); fp8 16,896 (3.88x smaller than fp32); fp4 9,728 (6.74x).

Kernels (all take index vectors; a negative index skips the row, matching SGLang's padded-row convention -1):
  quant_rows(src_fp32, src_idx, dst_q, dst_idx)   fp32 [N, HV, V, K] rows -> quantized pool rows
  dequant_rows(src_q, src_idx, dst_fp32, dst_idx) quantized pool rows -> fp32
  commit_fold_all_layers(...)                      the ReplaySSM commit for the two-tier layout: replays each request's
      accepted window from its fp32 working state W (bit-identical op order to SGLang's
      kda_replayssm_exact_fold_kernel, fp32 throughout), writes W back, and writes the quantized mirror of the
      result to the request's pool slot (and to its extra_buffer track slot on the crossing step), all in one
      launch for every KDA layer.

Everything accumulates in fp32; quantization only happens at the store.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

K_DIM = 128
FP8_MAX = 448.0
E2M1_MAX = 6.0
FP4_BLOCK = 16
FORMATS = ("fp8", "fp4", "bf16")


def row_bytes(fmt: str, k: int = K_DIM) -> int:
    if fmt == "fp8":
        return k + 4
    if fmt == "fp4":
        return k // 2 + k // FP4_BLOCK + 4
    if fmt == "bf16":
        return 2 * k
    raise ValueError(f"unknown k3state format {fmt!r}")


def fmt_code(fmt: str) -> int:
    return FORMATS.index(fmt)


# ------------------------------------------------------------------------------------------------ torch reference
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_THRESH = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def ref_quant(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Reference quantizer (CPU or GPU, any leading shape): [..., K] float -> [..., row_bytes] uint8."""
    x = x.float()
    k = x.shape[-1]
    amax = x.abs().amax(-1, keepdim=True)
    if fmt == "fp8":
        s = amax / FP8_MAX
        inv = torch.where(amax > 0, FP8_MAX / amax.clamp_min(1e-30), torch.zeros_like(amax))
        q = (x * inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
        return torch.cat([q, s.contiguous().view(torch.uint8)], -1)
    if fmt == "bf16":
        return x.to(torch.bfloat16).contiguous().view(torch.uint8)
    if fmt != "fp4":
        raise ValueError(fmt)
    xb = x.unflatten(-1, (k // FP4_BLOCK, FP4_BLOCK))
    bmax = xb.abs().amax(-1)                                            # [..., nb]
    g = amax / (E2M1_MAX * FP8_MAX)                                     # [..., 1]
    ginv = torch.where(amax > 0, (E2M1_MAX * FP8_MAX) / amax.clamp_min(1e-30), torch.zeros_like(amax))
    sb = (bmax * ginv / E2M1_MAX).clamp(0, FP8_MAX).to(torch.float8_e4m3fn)
    eff = sb.float() * g                                                # [..., nb]
    einv = torch.where(eff > 0, 1.0 / eff.clamp_min(1e-30), torch.zeros_like(eff))
    y = (xb * einv.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX).flatten(-2)
    a = y.abs()
    code = (a.unsqueeze(-1) > _E2M1_THRESH.to(a.device)).sum(-1).to(torch.uint8)
    code = code | ((y < 0).to(torch.uint8) << 3)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return torch.cat([packed, sb.view(torch.uint8), g.contiguous().view(torch.uint8)], -1)


def ref_dequant(q: torch.Tensor, fmt: str, k: int = K_DIM) -> torch.Tensor:
    """Reference dequantizer: [..., row_bytes] uint8 -> [..., K] fp32."""
    if fmt == "fp8":
        vals = q[..., :k].contiguous().view(torch.float8_e4m3fn).float()
        s = q[..., k:k + 4].contiguous().view(torch.float32)
        return vals * s
    if fmt == "bf16":
        return q[..., :2 * k].contiguous().view(torch.bfloat16).float()
    nb = k // FP4_BLOCK
    packed = q[..., :k // 2].to(torch.int64)
    code = torch.stack([packed & 15, packed >> 4], -1).flatten(-2)       # [..., K]
    mag = _E2M1_VALUES.to(q.device)[code & 7]
    vals = torch.where((code & 8) != 0, -mag, mag)
    sb = q[..., k // 2:k // 2 + nb].contiguous().view(torch.float8_e4m3fn).float()
    g = q[..., k // 2 + nb:k // 2 + nb + 4].contiguous().view(torch.float32)
    return (vals.unflatten(-1, (nb, FP4_BLOCK)) * (sb * g).unsqueeze(-1)).flatten(-2)


# ------------------------------------------------------------------------------------------------ device helpers
@triton.jit
def _store_q_tile(b_h, p_row, o_k, mask_v, FMT: tl.constexpr, K: tl.constexpr, BV: tl.constexpr):
    """Quantize a [K, BV] fp32 tile (K rows, BV v-columns) and store each column as one quantized row.

    p_row: [BV] uint8 pointers to the start of each v-row record."""
    amax = tl.max(tl.abs(b_h), axis=0)                                   # [BV]
    if FMT == 0:
        inv = tl.where(amax > 0, tl.math.div_rn(tl.full(amax.shape, 448.0, tl.float32), tl.maximum(amax, 1e-30)), 0.0)
        y = tl.minimum(tl.maximum(b_h * inv[None, :], -448.0), 448.0)
        codes = y.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        tl.store(p_row[None, :] + o_k[:, None], codes, mask=mask_v[None, :])
        s = tl.math.div_rn(amax, tl.full(amax.shape, 448.0, tl.float32))
        tl.store((p_row + K).to(tl.pointer_type(tl.float32)), s, mask=mask_v)
    elif FMT == 2:
        tl.store(p_row.to(tl.pointer_type(tl.bfloat16))[None, :] + o_k[:, None], b_h.to(tl.bfloat16),
                 mask=mask_v[None, :])
    else:
        NB: tl.constexpr = K // 16
        xb = tl.reshape(b_h, (NB, 16, BV))
        bmax = tl.max(tl.abs(xb), axis=1)                                # [NB, BV]
        g = tl.math.div_rn(amax, tl.full(amax.shape, 2688.0, tl.float32))
        ginv = tl.where(amax > 0, tl.math.div_rn(tl.full(amax.shape, 2688.0, tl.float32), tl.maximum(amax, 1e-30)), 0.0)
        sb = tl.minimum(tl.maximum(tl.math.div_rn(bmax * ginv[None, :], tl.full(bmax.shape, 6.0, tl.float32)), 0.0), 448.0).to(tl.float8e4nv)
        eff = sb.to(tl.float32) * g[None, :]
        einv = tl.where(eff > 0, tl.math.div_rn(tl.full(eff.shape, 1.0, tl.float32), tl.maximum(eff, 1e-30)), 0.0)
        y = xb * einv[:, None, :]
        y = tl.minimum(tl.maximum(y, -6.0), 6.0)
        a = tl.abs(y)
        code = ((a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
                + (a > 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32)
                + (a > 5.0).to(tl.int32))
        code = code | ((y < 0).to(tl.int32) << 3)                        # [NB, 16, BV]
        code = tl.reshape(code, (K // 2, 2, BV))
        code = tl.permute(code, (0, 2, 1))                               # [K/2, BV, 2]
        lo, hi = tl.split(code)
        byte = (lo | (hi << 4)).to(tl.uint8)                             # [K/2, BV]
        o_j = tl.arange(0, K // 2)
        tl.store(p_row[None, :] + o_j[:, None], byte, mask=mask_v[None, :])
        o_b = tl.arange(0, NB)
        tl.store(p_row[None, :] + (K // 2) + o_b[:, None], sb.to(tl.uint8, bitcast=True), mask=mask_v[None, :])
        tl.store((p_row + (K // 2 + NB)).to(tl.pointer_type(tl.float32)), g, mask=mask_v)


@triton.jit
def _load_q_tile(p_row, o_k, mask_v, FMT: tl.constexpr, K: tl.constexpr, BV: tl.constexpr):
    """Inverse of _store_q_tile: [BV] row pointers -> [K, BV] fp32 tile."""
    if FMT == 0:
        codes = tl.load(p_row[None, :] + o_k[:, None], mask=mask_v[None, :], other=0)
        s = tl.load((p_row + K).to(tl.pointer_type(tl.float32)), mask=mask_v, other=0.0)
        return codes.to(tl.float8e4nv, bitcast=True).to(tl.float32) * s[None, :]
    elif FMT == 2:
        return tl.load(p_row.to(tl.pointer_type(tl.bfloat16))[None, :] + o_k[:, None], mask=mask_v[None, :],
                       other=0.0).to(tl.float32)
    else:
        NB: tl.constexpr = K // 16
        byte = tl.load(p_row[None, :] + (o_k // 2)[:, None], mask=mask_v[None, :], other=0).to(tl.int32)
        code = tl.where((o_k % 2 == 0)[:, None], byte & 15, byte >> 4)   # [K, BV]
        mag = code & 7
        e = mag >> 1
        m = (mag & 1).to(tl.float32)
        val = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * (1 << tl.maximum(e - 1, 0)).to(tl.float32))
        val = tl.where((code & 8) != 0, -val, val)
        sb = tl.load(p_row[None, :] + (K // 2) + (o_k // 16)[:, None], mask=mask_v[None, :], other=0)
        sbf = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        g = tl.load((p_row + (K // 2 + NB)).to(tl.pointer_type(tl.float32)), mask=mask_v, other=0.0)
        return val * (sbf * g[None, :])   # same association as ref_dequant


# ------------------------------------------------------------------------------------------------ row copies
@triton.jit
def _quant_rows_kernel(src, src_idx, dst, dst_idx, s_src_n, s_src_h, s_dst_n, s_dst_h,
                       HAS_SRC_IDX: tl.constexpr, FMT: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                       ROWB: tl.constexpr, BV: tl.constexpr):
    i_v, i_n, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.load(dst_idx + i_n).to(tl.int64)
    if HAS_SRC_IDX:
        s = tl.load(src_idx + i_n).to(tl.int64)
    else:
        s = i_n.to(tl.int64)
    if (d < 0) | (s < 0):
        return
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    b_h = tl.load(src + s * s_src_n + i_h * s_src_h + o_v[None, :] * K + o_k[:, None], mask=mask_v[None, :],
                  other=0.0).to(tl.float32)
    p_row = dst + d * s_dst_n + i_h * s_dst_h + o_v * ROWB
    _store_q_tile(b_h, p_row, o_k, mask_v, FMT, K, BV)


@triton.jit
def _dequant_rows_kernel(src, src_idx, dst, dst_idx, s_src_n, s_src_h, s_dst_n, s_dst_h,
                         FMT: tl.constexpr, K: tl.constexpr, V: tl.constexpr, ROWB: tl.constexpr, BV: tl.constexpr):
    i_v, i_n, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    s = tl.load(src_idx + i_n).to(tl.int64)
    d = tl.load(dst_idx + i_n).to(tl.int64)
    if (d < 0) | (s < 0):
        return
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    b_h = _load_q_tile(src + s * s_src_n + i_h * s_src_h + o_v * ROWB, o_k, mask_v, FMT, K, BV)
    tl.store(dst + d * s_dst_n + i_h * s_dst_h + o_v[None, :] * K + o_k[:, None], b_h, mask=mask_v[None, :])


def _check_pool(q: torch.Tensor, fmt: str, k: int):
    assert q.dtype == torch.uint8 and q.shape[-1] == row_bytes(fmt, k), (q.shape, q.dtype, fmt)
    assert q.stride(-1) == 1 and q.stride(-2) == q.shape[-1], "quantized rows must be packed"


def quant_rows(src: torch.Tensor, src_idx, dst: torch.Tensor, dst_idx: torch.Tensor, fmt: str) -> None:
    """dst[dst_idx[i]] = quant(src[src_idx[i]]) (src_idx None: src row i). src [N, HV, V, K] float, dst
    [slots, HV, V, row_bytes] uint8; K-contiguous src rows. Rows with a negative index are skipped."""
    n = dst_idx.shape[0]
    if n == 0:
        return
    hv, v, k = src.shape[1:]
    _check_pool(dst, fmt, k)
    assert src.stride(-1) == 1 and src.stride(-2) == k and src.stride(1) == v * k
    bv = 32
    _quant_rows_kernel[(triton.cdiv(v, bv), n, hv)](
        src, src_idx if src_idx is not None else dst_idx, dst, dst_idx, src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1), HAS_SRC_IDX=src_idx is not None, FMT=fmt_code(fmt), K=k, V=v,
        ROWB=row_bytes(fmt, k), BV=bv, num_warps=4)


def dequant_rows(src: torch.Tensor, src_idx: torch.Tensor, dst: torch.Tensor, dst_idx: torch.Tensor, fmt: str) -> None:
    """dst[dst_idx[i]] = dequant(src[src_idx[i]]) into fp32 [rows, HV, V, K]."""
    n = src_idx.shape[0]
    if n == 0:
        return
    hv, v, k = dst.shape[1:]
    _check_pool(src, fmt, k)
    assert dst.dtype == torch.float32 and dst.stride(-1) == 1 and dst.stride(-2) == k and dst.stride(1) == v * k
    bv = 32
    _dequant_rows_kernel[(triton.cdiv(v, bv), n, hv)](
        src, src_idx, dst, dst_idx, src.stride(0), src.stride(1), dst.stride(0), dst.stride(1),
        FMT=fmt_code(fmt), K=k, V=v, ROWB=row_bytes(fmt, k), BV=bv, num_warps=4)


# ------------------------------------------------------------------------------------------------ commit fold
@triton.jit
def _fold_kernel(
    w_state,            # [L, W_rows, HV, V, K] fp32 working state (folded in place)
    q_pool,             # [L, slots, HV, V, ROWB] uint8 quantized pool (mirror written)
    rawv_cache, rawk_cache, gk_cache, beta_cache,   # rings, indexed by the working row
    w_indices,          # [B] working row per request (-1 = padded)
    slot_indices,       # [B] pool slot per request (-1 = padded)
    accept_lens,        # [B]
    track_indices,      # [B] extra_buffer track slot (or unused)
    steps_to_track,     # [B] crossing step (-1 = none)
    s_w_layer, s_w_row, s_q_layer, s_q_slot,
    s_rv_layer, s_rv_row, s_rk_layer, s_rk_row, s_g_layer, s_g_row, s_b_layer, s_b_row,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    ROWB: tl.constexpr, FMT: tl.constexpr, MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr, HAS_TRACK: tl.constexpr, WRITE_MIRROR: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hvl = tl.program_id(2)
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)

    w_row = tl.load(w_indices + i_n).to(tl.int64)
    slot = tl.load(slot_indices + i_n).to(tl.int64)
    if (w_row < 0) | (slot < 0):
        return
    n_commit = tl.load(accept_lens + i_n).to(tl.int32)
    if n_commit <= 0:
        return
    if HAS_TRACK:
        track_idx = tl.load(track_indices + i_n).to(tl.int64)
        track_step = tl.load(steps_to_track + i_n).to(tl.int32)
    else:
        track_idx = -1
        track_step = -1

    w_state = w_state + i_layer * s_w_layer
    q_pool = q_pool + i_layer * s_q_layer
    rawv_cache = rawv_cache + i_layer * s_rv_layer + w_row * s_rv_row
    rawk_cache = rawk_cache + i_layer * s_rk_layer + w_row * s_rk_row
    gk_cache = gk_cache + i_layer * s_g_layer + w_row * s_g_row
    beta_cache = beta_cache + i_layer * s_b_layer + w_row * s_b_row

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_w = w_state + w_row * s_w_row + i_hv * V * K + o_v[None, :] * K + o_k[:, None]
    b_h = tl.load(p_w, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(0, n_commit):
        phys = t.to(tl.int64)
        b_k = tl.load(rawk_cache + (i_h * MAX_CACHE_LEN + phys) * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(rawv_cache + (i_hv * MAX_CACHE_LEN + phys) * V + o_v, mask=mask_v, other=0.0).to(tl.float32)
        b_gk = tl.load(gk_cache + (i_hv * MAX_CACHE_LEN + phys) * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_beta = tl.load(beta_cache + i_hv * MAX_CACHE_LEN + phys).to(tl.float32)
        # verbatim op order of sglang kda_replayssm_exact_fold_kernel (IS_KDA recurrent update)
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_gk[:, None])
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        if HAS_TRACK:
            if (t == track_step) and (track_idx >= 0):
                _store_q_tile(b_h, q_pool + track_idx * s_q_slot + i_hv * V * ROWB + o_v * ROWB, o_k, mask_v,
                              FMT, K, BV)

    tl.store(p_w, b_h, mask=mask_h)
    if WRITE_MIRROR:
        _store_q_tile(b_h, q_pool + slot * s_q_slot + i_hv * V * ROWB + o_v * ROWB, o_k, mask_v, FMT, K, BV)


def commit_fold_all_layers(*, w_state, q_pool, rawv, rawk, gk, beta, w_indices, slot_indices, accept_lens,
                           fmt: str, num_k_heads: int, track_indices=None, steps_to_track=None,
                           use_qk_l2norm_in_kernel: bool = True, write_mirror: bool = True) -> None:
    """ReplaySSM commit for the two-tier layout, all KDA layers in one launch (see module docstring).

    w_state [L, W_rows, HV, V, K] fp32; q_pool [L, slots, HV, V, ROWB] uint8; rings [L, W_rows, ...] indexed by the
    working row; w_indices / slot_indices / accept_lens [B]."""
    num_layers, _, hv, v, k = w_state.shape
    b = w_indices.shape[0]
    if b == 0:
        return
    _check_pool(q_pool[0], fmt, k)
    assert w_state.dtype == torch.float32 and w_state.stride(-1) == 1 and w_state.stride(-2) == k
    assert w_state.stride(2) == v * k and q_pool.stride(2) == v * row_bytes(fmt, k)
    # launch shape per format (measured on B200, TP16 shapes, bs 64, accept 4, CUDA-graph replay): stock fold
    # (BV 32, 1 warp) 1,038 us; fp8 BV16/1 warp 1,201 us (BV32: 1,530); fp4 BV16/4 warps 2,141 us (BV32/1 warp: 5,115);
    # bf16 BV32/1 warp 1,082 us. The extra warps parallelise the per-element fp4 encode.
    bv, warps = {"fp8": (16, 1), "fp4": (16, 4), "bf16": (32, 1)}[fmt]
    bv = min(triton.next_power_of_2(v), bv)
    has_track = track_indices is not None and steps_to_track is not None
    grid = (triton.cdiv(v, bv), b, hv * num_layers)
    _fold_kernel[grid](
        w_state, q_pool, rawv, rawk, gk, beta, w_indices, slot_indices, accept_lens,
        track_indices if has_track else slot_indices, steps_to_track if has_track else accept_lens,
        w_state.stride(0), w_state.stride(1), q_pool.stride(0), q_pool.stride(1),
        rawv.stride(0), rawv.stride(1), rawk.stride(0), rawk.stride(1), gk.stride(0), gk.stride(1),
        beta.stride(0), beta.stride(1),
        H=num_k_heads, HV=hv, K=k, V=v, BV=bv, ROWB=row_bytes(fmt, k), FMT=fmt_code(fmt),
        MAX_CACHE_LEN=rawv.shape[-2], USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, HAS_TRACK=has_track,
        WRITE_MIRROR=write_mirror, num_warps=warps, num_stages=3)


# ------------------------------------------------------------------------------------------------ reference recurrence
def kda_recurrent_ref(h: torch.Tensor, q, k, v, gk, beta, scale: float, use_l2norm: bool = True):
    """One KDA step for a batch of heads, fp32, SGLang's IS_KDA op order. h [N, V, K]; q/k/gk [N, K]; v [N, V];
    beta [N]. Returns (o [N, V], h_new)."""
    if use_l2norm:
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    h = h * torch.exp(gk)[:, None, :]
    v = (v - torch.einsum("nvk,nk->nv", h, k)) * beta[:, None]
    h = h + v[:, :, None] * k[:, None, :]
    o = torch.einsum("nvk,nk->nv", h, q * scale)
    return o, h


# ------------------------------------------------------------------------------------------------ generic row copy
@triton.jit
def _copy_rows_kernel(src, src_idx, dst, dst_idx, s_src_l, s_src_n, s_dst_l, s_dst_n, NUMEL: tl.constexpr,
                      BLOCK: tl.constexpr):
    i_n, i_l = tl.program_id(0), tl.program_id(1).to(tl.int64)
    s = tl.load(src_idx + i_n).to(tl.int64)
    d = tl.load(dst_idx + i_n).to(tl.int64)
    if (s < 0) | (d < 0):
        return
    for off in range(0, NUMEL, BLOCK):
        o = off + tl.arange(0, BLOCK)
        m = o < NUMEL
        x = tl.load(src + i_l * s_src_l + s * s_src_n + o, mask=m)
        tl.store(dst + i_l * s_dst_l + d * s_dst_n + o, x, mask=m)


def copy_rows_all_layers(src: torch.Tensor, src_idx: torch.Tensor, dst: torch.Tensor, dst_idx: torch.Tensor) -> None:
    """dst[:, dst_idx[i]] = src[:, src_idx[i]] for every layer in one launch; src/dst [L, rows, ...] with contiguous
    rows of equal shape and dtype; negative indices skip. Used for the per-request conv working copy."""
    n = src_idx.shape[0]
    if n == 0:
        return
    assert src.dtype == dst.dtype and src.shape[0] == dst.shape[0] and src.shape[2:] == dst.shape[2:]
    numel = src[0, 0].numel()
    assert src[0, 0].is_contiguous() and dst[0, 0].is_contiguous()
    _copy_rows_kernel[(n, src.shape[0])](src, src_idx, dst, dst_idx, src.stride(0), src.stride(1), dst.stride(0),
                                         dst.stride(1), NUMEL=numel, BLOCK=1024, num_warps=4)
