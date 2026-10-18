"""Low-batch (M=1) decode MLP megakernel.

Thesis (measured in bench/vs_vllm_fp4_modal.py): at M=1 each projection GEMV runs ~40x below
the HBM-bandwidth floor -- the time is per-kernel launch/setup + intermediate-activation HBM
round-trips, not useful weight reads. A transformer decode step issues ~200 such GEMVs, so
that overhead IS the decode bottleneck.

A megakernel fuses the per-layer GEMV chain into ONE launch of persistent CTAs, keeping the
intermediate activation resident (global scratch, read hot) and paying the launch cost once.
The dependency (down projection needs the whole SwiGLU vector h) forces a grid-wide barrier
between the two GEMV stages; we do it with an atomic counter across a persistent grid sized to
the SM count so every CTA is co-resident (no deadlock).

  h[i]   = silu(x . Wg[i]) * (x . Wu[i])          # stage 1: I rows, reduce over H
  out[j] = sum_i h[i] * Wd[j,i]                    # stage 2: H rows, reduce over I

Three entry points for A/B:
  mlp_unfused        - 3 separate torch/cuBLAS GEMVs + elementwise (baseline)
  mlp_fused_2launch  - fused gate+up+swiglu (1 kernel) then down (1 kernel): 2 launches
  mlp_megakernel     - single persistent launch with an atomic grid barrier
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _gate_up_swiglu_kernel(x_ptr, wg_ptr, wu_ptr, h_ptr, H: tl.constexpr, I: tl.constexpr,
                               BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr):
        """One program computes BLOCK_I rows of h = silu(x@Wg^T) * (x@Wu^T). M=1."""
        row = tl.program_id(0)
        i_idx = row * BLOCK_I + tl.arange(0, BLOCK_I)          # [BLOCK_I]
        i_mask = i_idx < I
        acc_g = tl.zeros((BLOCK_I,), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_I,), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            h_idx = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_idx < H
            x = tl.load(x_ptr + h_idx, mask=h_mask, other=0.0).to(tl.float32)   # [BLOCK_H]
            wg = tl.load(wg_ptr + i_idx[:, None] * H + h_idx[None, :],
                         mask=i_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            wu = tl.load(wu_ptr + i_idx[:, None] * H + h_idx[None, :],
                         mask=i_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            acc_g += tl.sum(wg * x[None, :], axis=1)
            acc_u += tl.sum(wu * x[None, :], axis=1)
        h = (acc_g * tl.sigmoid(acc_g)) * acc_u                # SwiGLU
        tl.store(h_ptr + i_idx, h, mask=i_mask)

    @triton.jit
    def _down_kernel(h_ptr, wd_ptr, out_ptr, I: tl.constexpr, H: tl.constexpr,
                     BLOCK_H: tl.constexpr, BLOCK_I: tl.constexpr):
        """One program computes BLOCK_H rows of out = h @ Wd^T. Wd is [H, I]."""
        row = tl.program_id(0)
        j_idx = row * BLOCK_H + tl.arange(0, BLOCK_H)
        j_mask = j_idx < H
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for i0 in range(0, I, BLOCK_I):
            i_idx = i0 + tl.arange(0, BLOCK_I)
            i_mask = i_idx < I
            h = tl.load(h_ptr + i_idx, mask=i_mask, other=0.0).to(tl.float32)
            wd = tl.load(wd_ptr + j_idx[:, None] * I + i_idx[None, :],
                         mask=j_mask[:, None] & i_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(wd * h[None, :], axis=1)
        tl.store(out_ptr + j_idx, acc, mask=j_mask)

    @triton.jit
    def _megakernel(x_ptr, wg_ptr, wu_ptr, wd_ptr, h_ptr, out_ptr, ctr_ptr,
                    H: tl.constexpr, I: tl.constexpr, NUM_CTA: tl.constexpr,
                    BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr):
        """Single persistent launch. Stage 1: each CTA does a strided set of h rows. Atomic
        grid barrier. Stage 2: each CTA does a strided set of out rows. Grid == SM count so all
        CTAs are co-resident and the spin-barrier cannot deadlock."""
        pid = tl.program_id(0)

        # ---- stage 1: h rows pid, pid+NUM_CTA, ... ----
        n_i_blocks = tl.cdiv(I, BLOCK_I)
        blk = pid
        while blk < n_i_blocks:
            i_idx = blk * BLOCK_I + tl.arange(0, BLOCK_I)
            i_mask = i_idx < I
            acc_g = tl.zeros((BLOCK_I,), dtype=tl.float32)
            acc_u = tl.zeros((BLOCK_I,), dtype=tl.float32)
            for h0 in range(0, H, BLOCK_H):
                h_idx = h0 + tl.arange(0, BLOCK_H)
                h_mask = h_idx < H
                x = tl.load(x_ptr + h_idx, mask=h_mask, other=0.0).to(tl.float32)
                wg = tl.load(wg_ptr + i_idx[:, None] * H + h_idx[None, :],
                             mask=i_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
                wu = tl.load(wu_ptr + i_idx[:, None] * H + h_idx[None, :],
                             mask=i_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
                acc_g += tl.sum(wg * x[None, :], axis=1)
                acc_u += tl.sum(wu * x[None, :], axis=1)
            h = (acc_g * tl.sigmoid(acc_g)) * acc_u
            tl.store(h_ptr + i_idx, h, mask=i_mask)
            blk += NUM_CTA

        # ---- grid barrier: each CTA arrives once, then spins until all arrived ----
        tl.debug_barrier()
        tl.atomic_add(ctr_ptr, 1, sem="release")
        while tl.atomic_add(ctr_ptr, 0, sem="acquire") < NUM_CTA:
            pass

        # ---- stage 2: out rows pid, pid+NUM_CTA, ... (reads whole h) ----
        n_h_blocks = tl.cdiv(H, BLOCK_H)
        blk = pid
        while blk < n_h_blocks:
            j_idx = blk * BLOCK_H + tl.arange(0, BLOCK_H)
            j_mask = j_idx < H
            acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for i0 in range(0, I, BLOCK_I):
                i_idx = i0 + tl.arange(0, BLOCK_I)
                i_mask = i_idx < I
                hv = tl.load(h_ptr + i_idx, mask=i_mask, other=0.0).to(tl.float32)
                wd = tl.load(wd_ptr + j_idx[:, None] * I + i_idx[None, :],
                             mask=j_mask[:, None] & i_mask[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(wd * hv[None, :], axis=1)
            tl.store(out_ptr + j_idx, acc, mask=j_mask)
            blk += NUM_CTA


if HAVE_TRITON:

    @triton.jit
    def _decode_e2m1(nib):
        """4-bit nvfp4 code -> float value. mag3 {0..7} -> {0,.5,1,1.5,2,3,4,6} via the e2m1
        exp/mantissa formula (cheap arithmetic, no nested selects). sign from bit 3."""
        mag = nib & 7
        exp = mag >> 1                       # {0,0,1,1,2,2,3,3}
        man = (mag & 1).to(tl.float32)       # {0,1}
        pow_ = (1 << tl.maximum(exp - 1, 0)).to(tl.float32)   # 2^(exp-1) for exp>=1, else 1
        normal = (1.0 + 0.5 * man) * pow_    # exp>=1 grid
        v = tl.where(exp == 0, 0.5 * man, normal)             # exp==0 subnormals: {0,.5}
        sign = 1.0 - 2.0 * (nib >> 3).to(tl.float32)
        return v * sign

    @triton.jit
    def _fp4_gemv_row(pk32_ptr, sc_ptr, vec_ptr, row_idx, row_mask, K: tl.constexpr,
                      NW: tl.constexpr, SC: tl.constexpr, BLOCK_ROW: tl.constexpr,
                      BLOCK_W: tl.constexpr):
        """acc[BLOCK_ROW] = sum_k decode(pk[row,k]) * scale * vec[k], for one row-tile.
        Loads packed weight as int32 WORDS (8 nibbles/word, coalesced), decodes all 8 nibbles
        into a CONTIGUOUS [ROW, BLOCK_W*8] tile in K-order, and does ONE reduction against a
        contiguous vec load (like the bf16 kernel). 2 words == 1 scale block."""
        n8 = tl.arange(0, 8)                                # nibble index within a word
        acc = tl.zeros((BLOCK_ROW,), dtype=tl.float32)
        for w0 in range(0, NW, BLOCK_W):
            wi = w0 + tl.arange(0, BLOCK_W)                 # [BLOCK_W] word indices
            wmask = wi < NW
            m2 = row_mask[:, None] & wmask[None, :]
            words = tl.load(pk32_ptr + row_idx[:, None] * NW + wi[None, :], mask=m2, other=0)
            sc = tl.load(sc_ptr + row_idx[:, None] * SC + (wi // 2)[None, :], mask=m2, other=0.0).to(tl.float32)
            # decode 8 nibbles -> [ROW, BW, 8]; K = 8*wi + n so (wi,n) flattens to K-order
            nib = (words[:, :, None] >> (4 * n8)[None, None, :]) & 0xF
            wv = _decode_e2m1(nib) * sc[:, :, None]         # [ROW, BW, 8]
            wv = tl.reshape(wv, (BLOCK_ROW, BLOCK_W * 8))   # contiguous in K
            k_idx = w0 * 8 + tl.arange(0, BLOCK_W * 8)
            xk = tl.load(vec_ptr + k_idx, mask=k_idx < K, other=0.0).to(tl.float32)
            acc += tl.sum(wv * xk[None, :], axis=1)
        return acc

    @triton.jit
    def _megakernel_fp4(x_ptr, gpk_ptr, gsc_ptr, upk_ptr, usc_ptr, dpk_ptr, dsc_ptr,
                        h_ptr, out_ptr, ctr_ptr, H: tl.constexpr, I: tl.constexpr,
                        NUM_CTA: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_W: tl.constexpr):
        """FP4 fused MLP megakernel with coalesced int32-word weight loads. Single persistent
        launch + atomic grid barrier between the two GEMV stages."""
        pid = tl.program_id(0)
        NWH = H // 8; SCH = H // 16      # gate/up: K=H
        NWI = I // 8; SCI = I // 16      # down:    K=I

        # ---- stage 1: h[i] = silu(gate) * up ----
        n_i_blocks = tl.cdiv(I, BLOCK_I)
        blk = pid
        while blk < n_i_blocks:
            i_idx = blk * BLOCK_I + tl.arange(0, BLOCK_I)
            i_mask = i_idx < I
            g = _fp4_gemv_row(gpk_ptr, gsc_ptr, x_ptr, i_idx, i_mask, H, NWH, SCH, BLOCK_I, BLOCK_W)
            u = _fp4_gemv_row(upk_ptr, usc_ptr, x_ptr, i_idx, i_mask, H, NWH, SCH, BLOCK_I, BLOCK_W)
            h = (g * tl.sigmoid(g)) * u
            tl.store(h_ptr + i_idx, h, mask=i_mask)
            blk += NUM_CTA

        tl.debug_barrier()
        tl.atomic_add(ctr_ptr, 1, sem="release")
        while tl.atomic_add(ctr_ptr, 0, sem="acquire") < NUM_CTA:
            pass

        # ---- stage 2: out[j] = sum_i h[i] * Wd[j,i] ----
        n_h_blocks = tl.cdiv(H, BLOCK_I)
        blk = pid
        while blk < n_h_blocks:
            j_idx = blk * BLOCK_I + tl.arange(0, BLOCK_I)
            j_mask = j_idx < H
            o = _fp4_gemv_row(dpk_ptr, dsc_ptr, h_ptr, j_idx, j_mask, I, NWI, SCI, BLOCK_I, BLOCK_W)
            tl.store(out_ptr + j_idx, o, mask=j_mask)
            blk += NUM_CTA


def pack_fp4(W):
    """bf16 [N,K] -> (packed uint8 [N,K//2], per-16-block scale bf16 [N,K//16]). Matches the
    nvfp4 grid; two 4-bit codes per byte (even k low nibble, odd k high nibble)."""
    N, K = W.shape
    grid_mid = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=W.device)
    Wb = W.float().reshape(N, K // 16, 16)
    scale = (Wb.abs().amax(-1, keepdim=True) / 6.0).clamp(min=1e-8)
    q = Wb / scale
    idx = torch.bucketize(q.abs().contiguous(), grid_mid)          # 0..7
    code = (idx + (q < 0).long() * 8).reshape(N, K).to(torch.int32)  # 0..15
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale.squeeze(-1).reshape(N, K // 16).to(torch.bfloat16)


def mlp_megakernel_fp4(x, gpk, gsc, upk, usc, dpk, dsc, H, I, NUM_CTA=132, BLOCK_I=64, BLOCK_W=64):
    # int32 view of the uint8 packed weights: 4 bytes = 8 nibbles per word (coalesced loads)
    g32 = gpk.view(torch.int32); u32 = upk.view(torch.int32); d32 = dpk.view(torch.int32)
    h = torch.empty(I, device=x.device, dtype=torch.float32)
    out = torch.empty(H, device=x.device, dtype=torch.float32)
    ctr = torch.zeros(1, device=x.device, dtype=torch.int32)
    _megakernel_fp4[(NUM_CTA,)](x.reshape(-1), g32, gsc, u32, usc, d32, dsc, h, out, ctr,
                                H, I, NUM_CTA, BLOCK_I=BLOCK_I, BLOCK_W=BLOCK_W)
    return out.reshape(1, H)


def mlp_unfused(x, Wg, Wu, Wd):
    """Baseline: 3 cuBLAS GEMVs + elementwise (separate kernel launches)."""
    g = x @ Wg.t()
    u = x @ Wu.t()
    h = (torch.nn.functional.silu(g) * u).to(x.dtype)
    return h @ Wd.t()


def mlp_fused_2launch(x, Wg, Wu, Wd, BLOCK_I=64, BLOCK_H=128):
    H = Wg.shape[1]; I = Wg.shape[0]
    h = torch.empty(I, device=x.device, dtype=torch.float32)
    _gate_up_swiglu_kernel[(triton.cdiv(I, BLOCK_I),)](
        x.reshape(-1), Wg, Wu, h, H, I, BLOCK_I=BLOCK_I, BLOCK_H=BLOCK_H)
    out = torch.empty(H, device=x.device, dtype=torch.float32)
    _down_kernel[(triton.cdiv(H, BLOCK_H),)](
        h, Wd, out, I, H, BLOCK_H=BLOCK_H, BLOCK_I=BLOCK_I)
    return out.reshape(1, H)


def mlp_megakernel(x, Wg, Wu, Wd, NUM_CTA=132, BLOCK_I=64, BLOCK_H=128):
    H = Wg.shape[1]; I = Wg.shape[0]
    h = torch.empty(I, device=x.device, dtype=torch.float32)
    out = torch.empty(H, device=x.device, dtype=torch.float32)
    ctr = torch.zeros(1, device=x.device, dtype=torch.int32)
    _megakernel[(NUM_CTA,)](
        x.reshape(-1), Wg, Wu, Wd, h, out, ctr, H, I, NUM_CTA,
        BLOCK_I=BLOCK_I, BLOCK_H=BLOCK_H)
    return out.reshape(1, H)
