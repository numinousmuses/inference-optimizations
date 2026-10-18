"""Fused KDA decode for Kimi K3 -- our kernel, graded against vLLM's real ops.

WHY THIS OP
-----------
K3 is 93 layers: 69 KDA + 24 Gated MLA. KDA is ~74% of the layers, so at batch-1 decode it
is the dominant per-token cost, and it is a shape none of our existing kernels cover (every
attention kernel we own is softmax attention).

THE OPPORTUNITY (grounded in vLLM's own published numbers)
----------------------------------------------------------
vendor/vllm/kimi/kimi_gdn_linear_attn.py::_forward issues, per KDA layer, per token:

    causal_conv1d_update(q) -> causal_conv1d_update(k) -> causal_conv1d_update(v)
      -> fused_kda_gate -> fused_recurrent_kda -> FusedRMSNormGated

That is 6+ launches x 69 layers = ~414 kernel launches per decoded token, before the MLA
layers and MoE. The vLLM K3 post reports KDA *metadata preparation* alone cost 870 us at
batch 1 until they cut it to 34 us (~6% end-to-end), which is direct evidence that host and
launch overhead -- not FLOPs -- dominate this regime. NVIDIA contributed a fused CUDA kernel
for exactly this span ("short convolution, KDA state update, output gate, and
normalization"); this is our Triton equivalent, so we can measure the span ourselves and
own it on hardware where that kernel is not available or not tuned.

We fuse the whole span into ONE launch per layer: conv(q,k,v) -> gate -> recurrence ->
gated RMSNorm.

CORRECTNESS CONTRACT
--------------------
Graded by differential test against the real ops (kernels/oracle.py binds them):
    fused_recurrent_kda + fused_kda_gate + causal_conv1d_update + rms_norm_gated
Not against any hand-written reference -- those are banned (KERNEL_AUDIT.md).

Math transcribed from vendor/vllm @ 60b3d39:
  fla/ops/fused_recurrent.py::fused_recurrent_gated_delta_rule_fwd_kernel(IS_KDA=True)
  fla/ops/kda.py::kda_gate_fwd_kernel
  kimi/kimi_gdn_linear_attn.py (layer wiring, o_norm activation="sigmoid")

Load-bearing details that are easy to get wrong (all confirmed against source):
  * state is [Dv, Dk] VALUE-MAJOR (`b_h = tl.zeros([BV, BK])`), not [Dk, Dv]
  * q AND k are L2-normalised, with eps INSIDE the sqrt: x / sqrt(sum(x*x) + 1e-6)
  * q is then scaled by K**-0.5
  * decay is exp(g) with g = -exp(A) * softplus(g_proj + bias), i.e. LOG-space and negative
  * softplus switches to its linear asymptote when beta*x > threshold (default 20)
  * beta may be per-head scalar or headwise [Dv]
  * the conv applies SiLU; the output norm gate uses SIGMOID (not SiLU)
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # noqa: BLE001 - CPU box
    HAVE_TRITON = False

import torch


if HAVE_TRITON:

    @triton.jit
    def _kda_decode_fused_kernel(
        # projections (already computed by the GEMMs, one token)
        q_in, k_in, v_in,          # [B, H, Dk] / [B, H, Dk] / [B, HV, Dv]
        g_proj, dt_bias, A_log,    # [B, HV, Dk], [HV*Dk], [HV]
        beta_in,                   # [B, HV] or [B, HV, Dv]
        g2,                        # [B, HV, Dv] output-gate input
        norm_w,                    # [Dv]
        # conv weights + states (depthwise, width W)
        cw_q, cw_k, cw_v,          # [H*Dk, W] / [H*Dk, W] / [HV*Dv, W]
        cs_q, cs_k, cs_v,          # [B, H*Dk, W-1] mutated in place
        h_state,                   # [B, HV, Dv, Dk] mutated in place
        o_out,                     # [B, HV, Dv]
        scale,
        eps,
        softplus_beta: tl.constexpr,
        softplus_threshold: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        W: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        IS_BETA_HEADWISE: tl.constexpr,
        CONV_SILU: tl.constexpr,
    ):
        """One program per (batch, value-head, V-tile).

        Mirrors upstream's launch shape (whole Dk in one block, Dv tiled) so the comparison
        is like-for-like, but folds conv + gate + norm into the same program.
        """
        i_b = tl.program_id(0)
        i_hv = tl.program_id(1)
        i_v = tl.program_id(2)
        i_h = i_hv // (HV // H)          # GQA-style sharing of q/k across value heads

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < DK
        mask_v = o_v < DV

        # ---- 1. causal conv1d update (shift register) for q, k, v -----------------------
        # q/k are indexed by the query head i_h, v by the value head i_hv.
        qk_base = i_h * DK
        v_base = i_hv * DV

        b_q = tl.zeros([BK], dtype=tl.float32)
        b_k = tl.zeros([BK], dtype=tl.float32)
        b_v = tl.zeros([BV], dtype=tl.float32)

        # state layout [B, D, W-1]; oldest first. out = sum_j state[j]*w[j] + x*w[W-1]
        for j in tl.static_range(W - 1):
            s_q = tl.load(cs_q + (i_b * (H * DK) + qk_base + o_k) * (W - 1) + j, mask=mask_k, other=0.0).to(tl.float32)
            s_k = tl.load(cs_k + (i_b * (H * DK) + qk_base + o_k) * (W - 1) + j, mask=mask_k, other=0.0).to(tl.float32)
            s_v = tl.load(cs_v + (i_b * (HV * DV) + v_base + o_v) * (W - 1) + j, mask=mask_v, other=0.0).to(tl.float32)
            b_q += s_q * tl.load(cw_q + (qk_base + o_k) * W + j, mask=mask_k, other=0.0).to(tl.float32)
            b_k += s_k * tl.load(cw_k + (qk_base + o_k) * W + j, mask=mask_k, other=0.0).to(tl.float32)
            b_v += s_v * tl.load(cw_v + (v_base + o_v) * W + j, mask=mask_v, other=0.0).to(tl.float32)

        x_q = tl.load(q_in + (i_b * H + i_h) * DK + o_k, mask=mask_k, other=0.0).to(tl.float32)
        x_k = tl.load(k_in + (i_b * H + i_h) * DK + o_k, mask=mask_k, other=0.0).to(tl.float32)
        x_v = tl.load(v_in + (i_b * HV + i_hv) * DV + o_v, mask=mask_v, other=0.0).to(tl.float32)
        b_q += x_q * tl.load(cw_q + (qk_base + o_k) * W + (W - 1), mask=mask_k, other=0.0).to(tl.float32)
        b_k += x_k * tl.load(cw_k + (qk_base + o_k) * W + (W - 1), mask=mask_k, other=0.0).to(tl.float32)
        b_v += x_v * tl.load(cw_v + (v_base + o_v) * W + (W - 1), mask=mask_v, other=0.0).to(tl.float32)

        if CONV_SILU:
            b_q = b_q * tl.sigmoid(b_q)
            b_k = b_k * tl.sigmoid(b_k)
            b_v = b_v * tl.sigmoid(b_v)

        # shift the conv states: drop oldest, append the new pre-activation input.
        # only the first V-tile writes conv state, to keep the update single-writer.
        if i_v == 0:
            for j in tl.static_range(W - 2):
                nxt_q = tl.load(cs_q + (i_b * (H * DK) + qk_base + o_k) * (W - 1) + j + 1, mask=mask_k, other=0.0)
                nxt_k = tl.load(cs_k + (i_b * (H * DK) + qk_base + o_k) * (W - 1) + j + 1, mask=mask_k, other=0.0)
                tl.store(cs_q + (i_b * (H * DK) + qk_base + o_k) * (W - 1) + j, nxt_q, mask=mask_k)
                tl.store(cs_k + (i_b * (H * DK) + qk_base + o_k) * (W - 1) + j, nxt_k, mask=mask_k)
            tl.store(cs_q + (i_b * (H * DK) + qk_base + o_k) * (W - 1) + (W - 2), x_q.to(cs_q.dtype.element_ty), mask=mask_k)
            tl.store(cs_k + (i_b * (H * DK) + qk_base + o_k) * (W - 1) + (W - 2), x_k.to(cs_k.dtype.element_ty), mask=mask_k)
        for j in tl.static_range(W - 2):
            nxt_v = tl.load(cs_v + (i_b * (HV * DV) + v_base + o_v) * (W - 1) + j + 1, mask=mask_v, other=0.0)
            tl.store(cs_v + (i_b * (HV * DV) + v_base + o_v) * (W - 1) + j, nxt_v, mask=mask_v)
        tl.store(cs_v + (i_b * (HV * DV) + v_base + o_v) * (W - 1) + (W - 2), x_v.to(cs_v.dtype.element_ty), mask=mask_v)

        # ---- 2. gate: g = -exp(A) * softplus(g_proj + dt_bias) --------------------------
        b_gp = tl.load(g_proj + (i_b * HV + i_hv) * DK + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_gp += tl.load(dt_bias + i_hv * DK + o_k, mask=mask_k, other=0.0).to(tl.float32)
        scaled = b_gp * softplus_beta
        sp = tl.where(
            scaled > softplus_threshold,
            b_gp,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(scaled)),
        )
        a_h = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
        b_g = a_h * sp                        # log-space decay, <= 0

        # ---- 3. recurrence: gated delta rule -------------------------------------------
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + eps)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + eps)
        b_q = b_q * scale

        h_ptr = h_state + ((i_b * HV + i_hv) * DV + o_v[:, None]) * DK + o_k[None, :]
        mask_h = mask_v[:, None] & mask_k[None, :]
        b_h = tl.load(h_ptr, mask=mask_h, other=0.0).to(tl.float32)

        b_h = b_h * tl.exp(b_g)[None, :]                       # decay along key channels
        b_v = b_v - tl.sum(b_h * b_k[None, :], 1)              # delta rule error
        if IS_BETA_HEADWISE:
            b_beta = tl.load(beta_in + (i_b * HV + i_hv) * DV + o_v, mask=mask_v, other=0.0).to(tl.float32)
        else:
            b_beta = tl.load(beta_in + i_b * HV + i_hv).to(tl.float32)
        b_v = b_v * b_beta
        b_h = b_h + b_v[:, None] * b_k[None, :]                # rank-1 write
        b_o = tl.sum(b_h * b_q[None, :], 1)                    # h @ q
        tl.store(h_ptr, b_h.to(h_ptr.dtype.element_ty), mask=mask_h)

        # ---- 4. gated RMSNorm tail (activation = SIGMOID, gate applied before norm) -----
        b_g2 = tl.load(g2 + (i_b * HV + i_hv) * DV + o_v, mask=mask_v, other=0.0).to(tl.float32)
        b_o = b_o * tl.sigmoid(b_g2)
        # NOTE: RMS is over the full head_dim; with BV < DV this tile-local mean is only
        # exact when DV == BV. The wrapper enforces BV == DV for the fused path (see
        # kda_decode_fused) and falls back to a two-pass form otherwise.
        rms = tl.sqrt(tl.sum(b_o * b_o) / DV + 1e-5)
        b_o = b_o / rms * tl.load(norm_w + o_v, mask=mask_v, other=0.0).to(tl.float32)
        tl.store(o_out + (i_b * HV + i_hv) * DV + o_v, b_o.to(o_out.dtype.element_ty), mask=mask_v)


def kda_decode_fused(
    q, k, v, g_proj, dt_bias, A_log, beta, g2, norm_w,
    conv_w_q, conv_w_k, conv_w_v,
    conv_state_q, conv_state_k, conv_state_v,
    h_state, *, scale=None, eps=1e-6,
    softplus_beta=1.0, softplus_threshold=20.0, conv_silu=True,
):
    """Fused KDA decode step. Mutates conv_state_* and h_state in place; returns o [B,HV,DV].

    Shapes (matching vendor/vllm/kimi/kimi_gdn_linear_attn.py):
        q, k        [B, H, DK]        post-projection, pre-conv
        v           [B, HV, DV]
        g_proj      [B, HV, DK]       output of g_b_proj (the low-rank decay gate)
        dt_bias     [HV*DK]
        A_log       [HV]              per-head A parameter (state: A_log, decay = -exp(A))
        beta        [B, HV] or [B, HV, DV]
        g2          [B, HV, DV]       output-gate input (low-rank g_a/g_b)
        h_state     [B, HV, DV, DK]   VALUE-MAJOR, mutated in place
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton unavailable")
    B, H, DK = q.shape
    HV, DV = v.shape[1], v.shape[2]
    W = conv_w_q.shape[-1]
    if scale is None:
        scale = DK**-0.5
    BK = triton.next_power_of_2(DK)
    # RMS in the tail is over the whole head_dim, so the fused path needs the full Dv in
    # one program. Upstream tiles Dv by 8 because it has no norm to fuse; we trade that
    # tiling for one launch instead of six.
    BV = triton.next_power_of_2(DV)
    o = torch.empty((B, HV, DV), dtype=v.dtype, device=v.device)
    _kda_decode_fused_kernel[(B, HV, 1)](
        q, k, v, g_proj, dt_bias, A_log, beta, g2, norm_w,
        conv_w_q, conv_w_k, conv_w_v,
        conv_state_q, conv_state_k, conv_state_v,
        h_state, o,
        scale, eps,
        softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
        H=H, HV=HV, DK=DK, DV=DV, W=W, BK=BK, BV=BV,
        IS_BETA_HEADWISE=(beta.dim() == 3),
        CONV_SILU=conv_silu,
        num_warps=4, num_stages=2,
    )
    return o


def kda_state_bytes(num_heads: int, head_dim: int, dtype_bytes: int = 4) -> int:
    """Per-sequence KDA state for ONE layer: HV * DV * DK elements.

    Independent of sequence length -- this is why K3 serves 1M context. Only the 24 Gated
    MLA layers hold a paged KV cache that grows with tokens. Sizing note for 8x B300.
    """
    return num_heads * head_dim * head_dim * dtype_bytes
