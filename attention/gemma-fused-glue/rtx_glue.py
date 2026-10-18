"""Gemma 4 decode/verify glue fusion (rtxsys, Gemma 31B kernel program D1, 2026-10-06).
Installed as sglang/srt/layers/numinous_gemma_glue.py by operations/patches/gemma31_glue_hooks.py.
Env NUMINOUS_GEMMA_GLUE=1 turns the chains on. Unset: every entry point is a no-op and the stock layer runs.

What it attacks. Stock dense Gemma4DecoderLayer (gemma4_causal.py:657-756, v0.5.20) runs these glue kernels per layer
around its four FP8 GEMMs:
  K1 input_layernorm                         rmsnorm                       (:679)
  K2 qkv_proj input quant                    per_token_group_quant (JIT)   (fp8_utils.py:1112-1119)
  K3 post_attention_layernorm                rmsnorm                       (:685)
  K4 pre_feedforward_layernorm(h, residual)  fused_add_rmsnorm             (:735-738)
  K5 gate_up_proj input quant                per_token_group_quant (JIT)
  K6 post_ff_norm + residual, x layer_scalar gemma_rmsnorm_residual_scalar (:741-756)
plus the attention-side kernels and k3amd's G1a act + quant. Each glue kernel re-reads and re-writes [T, 5376] bf16.

Two chains, each ONE Triton kernel:
- post-attn chain (K3 + K4 + K5): h1 = bf16(rms(h) * w1); s = h1 + residual; residual <- bf16(s);
  out = bf16(rms(s) * w2); then FP8 group-128 (q, s) of out for gate_up.
- cross-layer chain (K6 + next layer's K1 + K2): s = bf16((rms(h) * w1 + residual) * layer_scalar) is the layer output;
  out = bf16(rms(s) * w_next); then FP8 (q, s) of out for the next layer's qkv_proj. The next layer finds `out`
  through take_prenormed().
Five glue kernels become two per layer (60 layers: 300 fewer launches per forward). Three [T, 5376] bf16 round trips
and two FP8 quant passes disappear.

Math, op for op the stock kernels' (so the fused result matches up to fp32 reduction order and rsqrt rounding):
- flashinfer RMSNorm: out = T(float(x) * rsqrt(sum(x^2)/d + eps) * float(w)).
- flashinfer FusedAddRMSNorm: x = float(in) + float(res); res <- T(x); out = T(x * rsqrt(sum(x^2)/d + eps) * w),
  i.e. the norm of the fp32 sum, not of the rounded residual.
- gemma_rmsnorm_residual_scalar (gemma4_fused_ops.py:14-51): out = T((x * rrms * w + r) * scalar), fp32 math.
- per_token_group_quant fp32-scale path (kernels/jit/csrc/gemm/per_token_group_quant.cuh:290-328):
  amax = max(max|x|, 1e-10); scale = amax * (1/448); q = e4m3_satfinite(min(x * (448/amax), 448)).
  Scales are column-major and TMA-aligned, laid out as create_per_token_group_quant_fp8_output_scale builds them
  (fp8_kernel.py:496-507): element (m, g) at g * align4(M) + m.
- The (q, s) pair feeds deepgemm_w8a8_block_fp8_linear_with_fallback's pre-quantized branch (fp8_utils.py:1062-1088)
  via take_prequant() at the function entry, so the consumer GEMM call is the stock one.

Safety:
- A consumer takes the stash only when it is the same buffer (data_ptr, shape, bf16, contiguous), on the same stream,
  for a block-[128, 128] linear, and only once.
- The layer hook clears the prequant stash right after the consuming sub-module returns (self_attn / mlp).
- A prenormed stash is matched by tensor identity plus the version counter, so any in-place change in between
  drops it.
- The last local layer, PLE / MoE layers, empty batches and non-bf16 inputs stay stock.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU hosts without triton: the module still imports; the chains are off
    triton = None
    tl = None

ON = os.environ.get("NUMINOUS_GEMMA_GLUE", "0") == "1"
QUANT = os.environ.get("NUMINOUS_GEMMA_GLUE_QUANT", "1") == "1"
GROUP = 128
FP8_MAX = 448.0


EMU = os.environ.get("TRITON_INTERPRET", "0") == "1"   # CPU interpreter: emulate the GPU's RNE conversions


if triton is not None:

    @triton.jit
    def _rne_bf16(x):
        """fp32 -> nearest-even bf16 value (as fp32). The interpreter's .to(bfloat16) truncates; GPUs round RNE."""
        b = x.to(tl.uint32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return b.to(tl.float32, bitcast=True)

    @triton.jit
    def _rne_e4m3(v):
        """fp32 (|v| <= 448 after the clamp) -> nearest-even e4m3 value (as fp32), like cvt.rn.satfinite.e4m3x2."""
        a = tl.abs(v)
        e = (a.to(tl.uint32, bitcast=True) >> 23) & 0xFF
        e = tl.maximum(e, 121)                       # 2^-6: e4m3 subnormal spacing 2^-9 below it
        c = ((e + 20) << 23).to(tl.float32, bitcast=True)   # ulp(c) = 2^(e-127-3): 3 mantissa bits
        r = tl.minimum((a + c) - c, 448.0)
        return tl.where(v < 0, -r, r)

    @triton.jit
    def _to_bf16(x, EMU: tl.constexpr):
        if EMU:
            return _rne_bf16(x).to(tl.bfloat16)
        else:
            return x.to(tl.bfloat16)

    @triton.jit
    def _e4m3_bits(v):
        """e4m3fn bit pattern (uint8) of the RNE-rounded v, sign of zero kept (the interpreter's fp8 cast is lossy)."""
        r = tl.abs(_rne_e4m3(v))
        b = r.to(tl.uint32, bitcast=True)
        sign = (v.to(tl.uint32, bitcast=True) >> 31) << 7
        e = (b >> 23) & 0xFF
        normal = ((e - 120) << 3) | ((b >> 20) & 7)
        sub = (r * 512.0).to(tl.uint32)               # exact: subnormal values are k * 2^-9, k = 1..7
        mag = tl.where(e >= 121, normal, sub)
        return (sign | mag).to(tl.uint8)

    @triton.jit
    def _to_e4m3(x, EMU: tl.constexpr):
        """EMU: the uint8 e4m3 bits (store through a uint8 view); else the native cvt.rn.satfinite cast."""
        if EMU:
            return _e4m3_bits(x)
        else:
            return x.to(tl.float8e4nv)

    @triton.jit
    def _glue_chain_kernel(
        H, R, W1, W2, SC, OUT, ROUT, Q, QS,
        stride_h, stride_r, stride_o, stride_ro, stride_q, qs_m, qs_g,
        N, NG, eps1, eps2,
        CROSS: tl.constexpr, QUANT: tl.constexpr, BLOCK: tl.constexpr, NGB: tl.constexpr, GSZ: tl.constexpr,
        EMU: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(H + row * stride_h + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
        r1 = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps1)
        if CROSS:
            sc = tl.load(SC).to(tl.float32)
            sb = _to_bf16((x * r1 * w1 + r) * sc, EMU)   # the layer output, as gemma_rmsnorm_residual_scalar
            t = sb.to(tl.float32)                            # the next input_layernorm reads the stored bf16
        else:
            h1 = _to_bf16(x * r1 * w1, EMU).to(tl.float32)   # post_attention_layernorm output is stored bf16
            t = h1 + r                                       # fused_add_rmsnorm: fp32 sum
            sb = _to_bf16(t, EMU)
        tl.store(ROUT + row * stride_ro + cols, sb, mask=mask)
        w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
        r2 = tl.rsqrt(tl.sum(t * t, axis=0) / N + eps2)
        o = _to_bf16(t * r2 * w2, EMU)
        tl.store(OUT + row * stride_o + cols, o, mask=mask)
        if QUANT:
            of = tl.reshape(o.to(tl.float32), (NGB, GSZ))
            amax = tl.maximum(tl.max(tl.abs(of), axis=1), 1e-10)
            scale = amax * (1.0 / 448.0)
            qv = _to_e4m3(tl.minimum(of * (448.0 / amax)[:, None], 448.0), EMU)
            qcols = tl.reshape(cols, (NGB, GSZ))
            tl.store(Q + row * stride_q + qcols, qv, mask=qcols < N)
            g = tl.arange(0, NGB)
            tl.store(QS + row * qs_m + g * qs_g, scale, mask=g < NG)


def _num_warps(block: int) -> int:
    return 4 if block <= 1024 else (8 if block <= 4096 else 16)


def alloc_prequant(m: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """q [m, k] e4m3 and s [m, k/128] fp32 column-major TMA-aligned (stride (1, align4(m)))."""
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=device)
    s = torch.empty((k // GROUP, (m + 3) // 4 * 4), dtype=torch.float32, device=device).transpose(0, 1)[:m, :]
    return q, s


def _launch(cross, h, r, w1, eps1, w2, eps2, scalar, out, rout, quant):
    m, n = h.shape
    block = triton.next_power_of_2(n)
    ngb = max(block // GROUP, 1)
    q = s = None
    if quant:
        q, s = alloc_prequant(m, n, h.device)
    _glue_chain_kernel[(m,)](
        h, r, w1, w2, scalar if scalar is not None else w1, out, rout,
        (q.view(torch.uint8) if EMU else q) if quant else out, s if quant else out,
        h.stride(0), r.stride(0), out.stride(0), rout.stride(0),
        q.stride(0) if quant else 0, s.stride(0) if quant else 0, s.stride(1) if quant else 0,
        n, n // GROUP, eps1, eps2,
        CROSS=cross, QUANT=quant, BLOCK=block, NGB=ngb, GSZ=GROUP if block >= GROUP else block, EMU=EMU,
        num_warps=_num_warps(block),
    )
    return q, s


def post_attn_chain(h, residual, w1, eps1, w2, eps2, quant: bool):
    """K3 + K4 (+ K5). Returns (out, residual, q, s); residual is updated in place like fused_add_rmsnorm."""
    out = torch.empty_like(h)
    q, s = _launch(False, h, residual, w1, eps1, w2, eps2, None, out, residual, quant)
    return out, residual, q, s


def cross_layer_chain(h, residual, w1, eps1, scalar, w2, eps2, quant: bool):
    """K6 + next K1 (+ next K2). Returns (layer_out, next_normed, q, s)."""
    rout = torch.empty_like(h)
    out = torch.empty_like(h)
    q, s = _launch(True, h, residual, w1, eps1, w2, eps2, scalar, out, rout, quant)
    return rout, out, q, s


# ----------------------------------------------------------------------------------------------- reference (stock math)
def ref_rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) / x.shape[-1] + eps) * w.float()).to(x.dtype)


def ref_fused_add_rmsnorm(x, res, w, eps):
    s = x.float() + res.float()
    out = (s * torch.rsqrt((s * s).sum(-1, keepdim=True) / x.shape[-1] + eps) * w.float()).to(x.dtype)
    return out, s.to(res.dtype)


def ref_rmsnorm_residual_scalar(x, w, res, scalar, eps):
    xf = x.float()
    rr = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / x.shape[-1] + eps)
    return ((xf * rr * w.float() + res.float()) * scalar.float()).to(x.dtype)


def ref_group_quant(x):
    """Stock per_token_group_quant fp32-scale semantics on bf16 x [m, k]: (q e4m3 [m, k], s fp32 [m, k/128])."""
    m, k = x.shape
    g = x.float().view(m, k // GROUP, GROUP)
    amax = g.abs().amax(-1).clamp_min(1e-10)
    scale = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    qs = torch.tensor(448.0, dtype=torch.float32) / amax
    q = torch.minimum(g * qs[..., None], torch.tensor(448.0)).to(torch.float8_e4m3fn).view(m, k)
    return q, scale


# ------------------------------------------------------------------------------------------------------- stash logic
_pq: dict | None = None    # prequant (q, s) waiting for the next block-FP8 linear on `ptr`
_pre: dict | None = None   # prenormed next-layer input keyed by the layer output tensor


def _stream_id() -> int:
    return torch.cuda.current_stream().cuda_stream if torch.cuda.is_available() else 0


def set_prequant(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> None:
    global _pq
    _pq = {"ptr": x.data_ptr(), "shape": tuple(x.shape), "q": q, "s": s, "stream": _stream_id()}


def clear_prequant() -> None:
    global _pq
    _pq = None


def take_prequant(x: torch.Tensor, block_size):
    """At deepgemm_w8a8_block_fp8_linear_with_fallback's entry (input_scale is None): the (q, s) of x or None."""
    global _pq
    st = _pq
    if st is None or block_size is None or list(block_size) != [GROUP, GROUP]:
        return None
    if (x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() != st["ptr"]
            or tuple(x.shape) != st["shape"] or _stream_id() != st["stream"]):
        return None
    _pq = None
    return st["q"], st["s"]


def set_prenormed(layer_out: torch.Tensor, normed: torch.Tensor, q, s) -> None:
    global _pre
    _pre = {"res": layer_out, "ver": layer_out._version, "out": normed, "q": q, "s": s}


def take_prenormed(h: torch.Tensor):
    """At the next layer's input_layernorm: the precomputed normed input (and its prequant), or None."""
    global _pre
    st = _pre
    if st is None:
        return None
    _pre = None
    if st["res"] is not h or h._version != st["ver"]:
        return None
    if st["q"] is not None:
        set_prequant(st["out"], st["q"], st["s"])
    return st["out"]


# ---------------------------------------------------------------------------------------------------- layer plumbing
def link_layers(layers, start: int, end: int) -> None:
    """Give each local layer its successor's input_layernorm (object.__setattr__: no submodule registration)."""
    for i in range(start, end):
        nxt = layers[i + 1].input_layernorm if i + 1 < end else None
        object.__setattr__(layers[i], "_ngg_next_norm", nxt)


def _linear_prequant_ok(lin) -> bool:
    try:
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.quantization.fp8_utils import deepgemm_w8a8_block_fp8_linear_with_fallback
    except Exception:
        return False
    if getattr(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", True):
        return False
    scheme = getattr(lin, "scheme", None)
    if scheme is not None:   # compressed-tensors (RedHatAI FP8-block): CompressedTensorsW8A8Fp8
        fn = getattr(scheme, "w8a8_block_fp8_linear", None)
        bs = getattr(scheme, "weight_block_size", None)
        if getattr(scheme, "is_static_input_scheme", False):
            return False
    else:                    # sglang fp8 block checkpoints: Fp8LinearMethod
        qm = getattr(lin, "quant_method", None)
        if not getattr(qm, "block_quant", False) or getattr(qm, "use_mxfp8", False):
            return False
        fn = getattr(qm, "w8a8_block_fp8_linear", None)
        bs = getattr(getattr(qm, "quant_config", None), "weight_block_size", None)
    w = getattr(lin, "weight", None)
    # k3amd's G2 size dispatcher wraps the DeepGEMM linear (numinous_fp8_size_dispatch.make_dispatcher, G2 v2 4db1bad1):
    # its DeepGEMM branch reaches the entry hook that consumes the stash, and its swap-AB Triton branch calls
    # take_prequant itself, so the stash is consumed on both branches.
    if getattr(fn, "_numinous_stock", None) is deepgemm_w8a8_block_fp8_linear_with_fallback:
        fn = deepgemm_w8a8_block_fp8_linear_with_fallback
    return (fn is deepgemm_w8a8_block_fp8_linear_with_fallback and bs is not None and list(bs) == [GROUP, GROUP]
            and w is not None and w.dtype == torch.float8_e4m3fn and w.dim() == 2
            and w.shape[0] % 64 == 0 and w.shape[1] % GROUP == 0)


def _layer_cfg(layer) -> dict:
    cfg = getattr(layer, "_ngg_cfg", None)
    if cfg is None:
        dense = (not getattr(layer, "enable_moe_block", False) and getattr(layer, "moe", None) is None
                 and not getattr(layer, "has_ple", False))
        cfg = {"dense": dense and triton is not None,
               "q_mlp": QUANT and _linear_prequant_ok(getattr(layer.mlp, "gate_up_proj", None))}
        nxt = getattr(layer, "_ngg_next_norm", None)
        cfg["cross"] = cfg["dense"] and nxt is not None
        object.__setattr__(layer, "_ngg_cfg", cfg)
        _log_once(layer, cfg)
    return cfg


_logged = False


def _log_once(layer, cfg) -> None:
    global _logged
    if _logged:
        return
    _logged = True
    import logging

    logging.getLogger(__name__).info(
        "numinous gemma glue (D1): dense chains %s, cross-layer %s, gate_up prequant %s",
        "ENGAGED" if cfg["dense"] else "off", "ENGAGED" if cfg["cross"] else "off",
        "ENGAGED" if cfg["q_mlp"] else "off")


def _act_ok(h, residual) -> bool:
    return (h.is_cuda and h.dim() == 2 and h.dtype == torch.bfloat16 and h.is_contiguous()
            and residual.is_contiguous() and residual.shape == h.shape and residual.dtype == h.dtype
            and h.shape[0] > 0 and h.shape[1] % GROUP == 0)


def layer_post_attn(layer, h, residual):
    """Hook body for gemma4_causal.py:685 + :735-738. Returns (out, residual) or None (stock path)."""
    if not ON:
        return None
    cfg = _layer_cfg(layer)
    if not cfg["dense"] or not _act_ok(h, residual):
        return None
    n1, n2 = layer.post_attention_layernorm, layer.pre_feedforward_layernorm
    out, residual, q, s = post_attn_chain(h, residual, n1.weight.data, n1.variance_epsilon,
                                          n2.weight.data, n2.variance_epsilon, cfg["q_mlp"])
    if q is not None:
        set_prequant(out, q, s)
    return out, residual


def layer_output(layer, h, residual):
    """Hook body for gemma4_causal.py:741-756. Returns the layer output (with the next layer's normed input
    stashed) or None (stock path)."""
    if not ON:
        return None
    cfg = _layer_cfg(layer)
    if not cfg["cross"] or not _act_ok(h, residual):
        return None
    nxt = layer._ngg_next_norm
    qn = QUANT and _next_qkv_ok(layer)
    n = layer.post_feedforward_layernorm
    rout, out, q, s = cross_layer_chain(h, residual, n.weight.data, n.variance_epsilon, layer.layer_scalar,
                                        nxt.weight.data, nxt.variance_epsilon, qn)
    set_prenormed(rout, out, q, s)
    return rout


def _next_qkv_ok(layer) -> bool:
    v = getattr(layer, "_ngg_next_qkv_ok", None)
    if v is None:
        v = bool(getattr(layer, "_ngg_next_qkv", None) is not None and _linear_prequant_ok(layer._ngg_next_qkv))
        object.__setattr__(layer, "_ngg_next_qkv_ok", v)
    return v


def link_layers_full(layers, start: int, end: int) -> None:
    """link_layers plus the successor's qkv_proj, for the cross-layer prequant eligibility."""
    link_layers(layers, start, end)
    for i in range(start, end):
        nxt = layers[i + 1].self_attn.qkv_proj if i + 1 < end else None
        object.__setattr__(layers[i], "_ngg_next_qkv", nxt)
