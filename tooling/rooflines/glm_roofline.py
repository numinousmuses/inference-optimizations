"""GLM-5.3 (RadixArk/GLM-5.3-NVFP4) floor per GPU per step for the served Standard layout: 8 GPUs, DP8 attention
(bf16 attention weights replicated on every GPU), TP8 NVFP4 experts (shared expert fused as an always-selected
expert), EAGLE 3/1/4 (4 verify tokens per request, 3 draft steps through the MTP layer), FP8 MLA KV (656 B/token),
DSA indexer top-2048. Each term is the larger of its HBM bytes and its FLOPs at the format's dense peak; NVLink for the
DP gather + reduce-scatter. A floor, not a forecast: nothing runs at 100% of peak.

    python3 operations/kernels/glm_roofline.py [--phase decode|prefill] [--tokens N] [--context N] [--gpu B200|B300|MI355X]
        [--gpus 4] [--steps 5] [--touched] [--accept 3.2]

--gpus sets the replica size (default 8; a 4-GPU replica holds twice the expert bytes per GPU); --steps the EAGLE draft
depth (verify tokens = steps + 1); --touched reads only the experts the gathered tokens route to (expected distinct
experts E * (1 - (1 - K/E)^tokens), uniform routing; the default reads all, as before); --accept prints the per-user
decode tok/s bound = accept / step floor (amdglm, 2026-10-03).

--gpu MI355X models amd/GLM-5.3-Quark-MXFP4-AttnFP8 on 8x MI355X (amdopt, 2026-09-29): MXFP4 experts, shared expert
and dense MLPs (4-bit + one E8M0 per 32 = 0.53125 B/param), FP8 block attention weights (1 B/param at the FP8 peak),
raw FP8 MLA KV for the HIP TileLang path (576 B/token), xGMI 7 links x 76.8 GB/s per direction. Peaks are AMD's
published dense numbers (assumed): HBM3E 8 TB/s, MXFP4 10 PF, FP8 5 PF, BF16 2.5 PF.
"""
from __future__ import annotations

import argparse

# Hopper (amdglm 2026-10-04): H200 / H100 SXM, the derived mixed checkpoint (NVFP4 experts run as Marlin W4A16, so the
# expert "FP4" peak is the BF16 MMA peak; FP8 block attention and dense layers) or W4AFP8 (int4 g128 experts on FP8 MMA:
# --gpu H200-w4a8). NVLink 4: 450 GB/s per direction. Dense peaks are NVIDIA's published SXM numbers (assumed).
HBM = {"B200": 8.0e12, "B300": 8.0e12, "MI355X": 8.0e12, "H200": 4.8e12, "H200-w4a8": 4.8e12, "H100": 3.35e12,
       "H100-w4a8": 3.35e12}
FP4 = {"B200": 9.0e15, "B300": 13.5e15, "MI355X": 10.0e15, "H200": 0.989e15, "H200-w4a8": 1.979e15, "H100": 0.989e15,
       "H100-w4a8": 1.979e15}
FP8 = {"B200": 4.5e15, "B300": 4.5e15, "MI355X": 5.0e15, "H200": 1.979e15, "H200-w4a8": 1.979e15, "H100": 1.979e15,
       "H100-w4a8": 1.979e15}
BF16 = {"B200": 2.25e15, "B300": 2.25e15, "MI355X": 2.5e15, "H200": 0.989e15, "H200-w4a8": 0.989e15, "H100": 0.989e15,
        "H100-w4a8": 0.989e15}
NVLINK = 900e9                               # bytes/s per direction per GPU
LINK = {"B200": NVLINK, "B300": NVLINK, "MI355X": 7 * 76.8e9, "H200": 450e9, "H200-w4a8": 450e9, "H100": 450e9,
        "H100-w4a8": 450e9}
FP4_B = {"B200": 0.5625, "B300": 0.5625, "MI355X": 0.53125, "H200": 0.5625, "H200-w4a8": 0.516, "H100": 0.5625,
         "H100-w4a8": 0.516}   # NVFP4 (e4m3 per 16) / MXFP4 (e8m0 per 32) / int4 g128 (+bf16 scale per 128)
ATTN_B = {"B200": 2, "B300": 2, "MI355X": 1, "H200": 1, "H200-w4a8": 1, "H100": 1, "H100-w4a8": 1}
KV_BYTES = {"B200": 656, "B300": 656, "MI355X": 576, "H200": 656, "H200-w4a8": 656, "H100": 656, "H100-w4a8": 656}

H, LAYERS, DENSE, MOE = 6144, 78, 3, 75
E, K, INTER, SHARED = 256, 8, 2048, 1
HEADS, KV_LORA, Q_LORA, ROPE, NOPE, VDIM = 64, 512, 2048, 64, 192, 256
IDX_HEADS, IDX_DIM, TOPK, VOCAB = 32, 128, 2048, 154880
DENSE_INTER = 12288
NVFP4_B = 0.5625                             # 4-bit value + one e4m3 scale per 16
KV_B = 656                                   # FP8 MLA KV per token (512 latent fp8 + 64 rope bf16 + scales)
IDX_K_B = 132                                # FP8 indexer key + scale per token
IDX_LAYERS_DECODE, IDX_LAYERS_PREFILL = 21, 11   # indexer calls per step in the B200 phase profiles
DRAFT_STEPS, VERIFY = 3, 4
GPUS = 8

ATTN_PARAMS = (H * Q_LORA + H * (KV_LORA + ROPE) + Q_LORA * HEADS * (NOPE + ROPE) + KV_LORA * HEADS * (NOPE + VDIM)
               + HEADS * VDIM * H + Q_LORA * IDX_HEADS * IDX_DIM + H * IDX_DIM + H * IDX_HEADS)
EXPERT_PARAMS = 3 * H * INTER                # gate + up + down
ROUTER_PARAMS = H * E


def floors(phase: str, tokens: int, context: int, gpu: str, gpus: int = 8, steps: int = 3,
           touched: bool = False) -> list[tuple[str, float]]:
    global GPUS, DRAFT_STEPS, VERIFY
    GPUS, DRAFT_STEPS, VERIFY = gpus, steps, steps + 1
    hbm, fp4, fp8, bf16 = HBM[gpu], FP4[gpu], FP8[gpu], BF16[gpu]
    fp4_b, attn_b, kv_b, link = FP4_B[gpu], ATTN_B[gpu], KV_BYTES[gpu], LINK[gpu]
    attn_peak = fp8 if attn_b == 1 else bf16
    dense_b, dense_peak = (fp4_b, fp4) if gpu == "MI355X" else (1, fp8) if gpu.startswith("H") else (2, bf16)
    g = tokens * GPUS                                            # gathered MoE tokens
    t = lambda nbytes, flops, peak: max(nbytes / hbm, flops / peak) * 1e3   # noqa: E731  (ms)
    rows = []
    frac = (1 - (1 - K / E) ** g) if touched and phase == "decode" else 1.0
    moe_bytes = MOE * (E * frac + SHARED) * EXPERT_PARAMS * fp4_b / GPUS if touched else \
        MOE * E * EXPERT_PARAMS * fp4_b / GPUS
    moe_flops = MOE * 2 * g * (K + SHARED) * EXPERT_PARAMS / GPUS
    rows.append(("routed + shared experts (FP4)", t(moe_bytes, moe_flops, fp4)))
    if phase == "prefill":
        # TP8 layout: every GPU's FC2 writes a hidden-wide bf16 partial row for every (token, expert) pair (not
        # split across GPUs), finalize reads them back and writes one row per token
        out = MOE * (2 * g * (K + SHARED) * H * 2 + g * H * 2)
        rows.append(("MoE output rows + finalize (TP8 layout)", out / hbm * 1e3))
    rows.append((f"attention projections ({'fp8' if attn_b == 1 else 'bf16'})",
                 t(LAYERS * ATTN_PARAMS * attn_b, LAYERS * 2 * tokens * ATTN_PARAMS, attn_peak)))
    rows.append(("dense MLP layers (first 3)", t(DENSE * 3 * H * DENSE_INTER * dense_b / GPUS,
                                                 DENSE * 2 * g * 3 * H * DENSE_INTER / GPUS, dense_peak)))
    rows.append(("router", t(MOE * ROUTER_PARAMS * 2, MOE * 2 * g * ROUTER_PARAMS, bf16)))
    reqs = max(1, tokens // VERIFY)
    if phase == "decode":
        kv = LAYERS * tokens * TOPK * kv_b
        rows.append(("sparse attention KV reads", t(kv, LAYERS * tokens * HEADS * TOPK * (576 + 512) * 2, fp8)))
        idx = IDX_LAYERS_DECODE * reqs * context * IDX_K_B
        rows.append(("indexer over the context", t(idx, IDX_LAYERS_DECODE * tokens * IDX_HEADS * IDX_DIM * context * 2,
                                                   fp8)))
        rows.append((f"lm head (target + {DRAFT_STEPS} draft steps)", (1 + DRAFT_STEPS) * VOCAB * H * 2 / hbm * 1e3))
        draft = DRAFT_STEPS * (ATTN_PARAMS * attn_b + E * EXPERT_PARAMS * fp4_b / GPUS)
        rows.append((f"EAGLE draft layer ({DRAFT_STEPS} steps)", draft / hbm * 1e3))
    else:
        # each query token gathers its own top-2048 KV rows (no reuse across tokens is assumed: a floor for a
        # per-token kernel; tokens of one request share most picks, which a token-group kernel can exploit)
        rows.append(("sparse attention (FP8 QK/PV, per-token KV gather)",
                     t(LAYERS * tokens * TOPK * kv_b, LAYERS * tokens * HEADS * TOPK * (576 + 512) * 2, fp8)))
        rows.append(("indexer over the context", IDX_LAYERS_PREFILL * tokens * IDX_HEADS * IDX_DIM * context * 2
                     / fp8 * 1e3))
    comm = MOE * 2 * (GPUS - 1) / GPUS * g * H * 2               # bf16 gather + reduce-scatter
    rows.append(("DP gather + reduce-scatter (bf16, NVLink / xGMI)", comm / link * 1e3))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill"])
    ap.add_argument("--tokens", type=int, default=None, help="tokens per GPU (decode: 12 requests x 4 = 48)")
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--gpu", default="B200", choices=sorted(HBM))
    ap.add_argument("--gpus", type=int, default=8, help="GPUs per replica (experts sharded over them)")
    ap.add_argument("--steps", type=int, default=3, help="EAGLE draft steps (verify tokens = steps + 1)")
    ap.add_argument("--touched", action="store_true", help="read only the experts the step's tokens route to")
    ap.add_argument("--accept", type=float, default=None, help="accept length: print the per-user tok/s bound")
    a = ap.parse_args()
    tokens = a.tokens or (48 if a.phase == "decode" else 2048)
    context = a.context or (90000 if a.phase == "decode" else 23000)
    rows = floors(a.phase, tokens, context, a.gpu, a.gpus, a.steps, a.touched)
    print(f"{a.phase} step floor, {tokens} tokens per GPU, {context} context, {a.gpu} x{a.gpus}, EAGLE {a.steps}")
    for name, ms in sorted(rows, key=lambda r: -r[1]):
        print(f"  {ms:8.2f} ms  {name}")
    total = sum(ms for _, ms in rows)
    print(f"  {total:8.2f} ms  total")
    if a.accept:
        print(f"  per-user decode bound at accept {a.accept}: {a.accept / total * 1e3:.0f} tok/s")


if __name__ == "__main__":
    main()
