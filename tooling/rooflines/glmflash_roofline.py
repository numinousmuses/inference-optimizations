"""GLM-5.3-Flash (RadixArk/GLM-5.3-Flash-NVFP4) decode floor per GPU per step, and the measured gap per kernel family.

Anatomy (source_code, config.json f46cf340): 45 layers, hidden 4096, 34 KDA layers (64 heads x 128, conv 4) + 11 DSA
layers (MLA kv_lora 512, q_lora 1536, 64 heads, qk 256 / v 256, indexer 32 x 128, top-2048), 3 dense MLP layers
(inter 12288), 42 MoE layers with 288 routed experts top-8 + 1 shared (inter 2048), mHC 4 streams, vocab 154,880.
NVFP4 W4A4 on every Linear except attention / router / embeddings / lm_head / MTP (bf16 attention; the derived
FP8-attention checkpoint halves those bytes).

Each term is max(HBM bytes / HBM BW, FLOPs / peak). A floor, not a forecast (k3amd, 2026-10-03).

    python3 operations/kernels/glmflash_roofline.py [--tp 1] [--batch 28] [--verify 6] [--context 45000]
        [--state-bytes 2] [--accept 3.5] [--attn-bytes 1] [--measured-ms 50]
"""
from __future__ import annotations

import argparse

HBM, FP4, FP8, BF16 = 8.0e12, 13.5e12 * 1e3, 4.5e15, 2.25e15      # B300 (assumed vendor peaks)
# per-SKU peaks (assumed vendor dense peaks, no sparsity): HBM B/s, FP8 FLOP/s, BF16 FLOP/s, and the rate the routed
# experts compute at: B300 runs NVFP4 W4A4 (FP4 tensor cores); SM90 has no FP4 MMA, so Marlin W4A16 dequantizes to
# BF16 and computes at the BF16 rate while still reading 4-bit weights (modelopt_quant.py ModelOptNvFp4FusedMoEMethod
# -> fused_marlin_moe on SM80-SM90, source_code)
SKU = {
    "b300": {"hbm": 8.0e12, "fp8": 4.5e15, "bf16": 2.25e15, "moe": 13.5e15},
    "h200": {"hbm": 4.8e12, "fp8": 1.979e15, "bf16": 0.989e15, "moe": 0.989e15},
    "h100": {"hbm": 3.35e12, "fp8": 1.979e15, "bf16": 0.989e15, "moe": 0.989e15},
}
H, LAYERS, KDA_L, DSA_L, DENSE_L, MOE_L = 4096, 45, 34, 11, 3, 42
E, K, SHARED, INTER, DENSE_INTER = 288, 8, 1, 2048, 12288
KH, KD = 64, 128                                                    # KDA heads x head dim
HEADS, KV_LORA, Q_LORA, QK, V = 64, 512, 1536, 256, 256
IDX_H, IDX_D, TOPK, VOCAB = 32, 128, 2048, 154880
NVFP4_B = 0.5625                                                    # 4-bit + one e4m3 scale per 16


def expert_params() -> float:
    return 3 * H * INTER


def touched_experts(tokens: int) -> float:
    """Expected distinct routed experts hit by `tokens` gathered tokens, uniform routing (the shared one is always)."""
    return E * (1 - (1 - K / E) ** tokens)


def decode_floor(tp: int = 1, batch: int = 28, verify: int = 6, context: int = 45000, state_bytes: int = 2,
                 attn_bytes: int = 1, sku: str = "b300") -> dict:
    """Per GPU per decode step (verify pass of `verify` tokens per request; draft passes are excluded: the FP4 MTP
    head is one layer). Returns ms per term and the total."""
    p = SKU[sku]
    HBM, FP4, BF16 = p["hbm"], p["moe"], p["bf16"]   # noqa: N806 (shadow the B300 module constants per SKU)
    toks = batch * verify
    terms: dict[str, float] = {}
    # MoE weights: each touched expert read once per step; TP splits every expert's bytes
    moe_b = MOE_L * (touched_experts(toks) + SHARED) * expert_params() * NVFP4_B / tp
    moe_f = 2 * toks * MOE_L * (K + SHARED) * expert_params() / tp
    terms["moe"] = max(moe_b / HBM, moe_f / FP4)
    dense_b = DENSE_L * 3 * H * DENSE_INTER * NVFP4_B / tp
    terms["dense_mlp"] = max(dense_b / HBM, 2 * toks * DENSE_L * 3 * H * DENSE_INTER / tp / FP4)
    # KDA: projections (q, k, v, g, beta, o ~ 6 x H x heads*dim, NVFP4 in the base ckpt; FP8 in the kda ckpt) and the
    # recurrent state read + written once per request per step (the verify chain stays in registers / the replay ring)
    kda_proj = KDA_L * 6 * H * KH * KD * attn_bytes / tp
    kda_state = KDA_L * KH * KD * KD * state_bytes / tp * batch * 2
    terms["kda_proj"] = kda_proj / HBM
    terms["kda_state"] = kda_state / HBM
    # DSA: attention weights + top-2048 latent KV rows (FP8 512 B) + indexer keys over the whole context (FP8 128 B)
    dsa_w = DSA_L * (H * Q_LORA + Q_LORA * HEADS * QK + H * (KV_LORA) + HEADS * V * H + KV_LORA * HEADS * (QK + V)) \
        * attn_bytes / tp
    dsa_kv = DSA_L * batch * (min(TOPK, context) * 512 + context * IDX_H * 0 + context * 128 / 4)
    terms["dsa"] = (dsa_w + dsa_kv / tp) / HBM
    terms["lm_head"] = max(VOCAB * H * 2 / tp / HBM, 2 * toks * VOCAB * H / tp / BF16)
    terms["total"] = sum(terms.values())
    return {k: v * 1e3 for k, v in terms.items()}


def prefill_floor(tokens: int = 16384, tp: int = 4, context: int = 30000, sku: str = "h200",
                  attn_bytes: int = 1, kda_chunk: int = 64) -> dict:
    """Per GPU for one prefill chunk of `tokens` new tokens on an engine of `tp` GPUs whose requests already hold
    `context` cached tokens (derived floor: each term max(bytes / HBM, FLOPs / peak), ms).
    - moe: every routed expert is touched by a 16K chunk (touched_experts), FLOPs 2 x tokens x (top-8 + shared)
      x 3HI per MoE layer at the experts' rate (Marlin W4A16 = BF16 on SM90);
    - dense_mlp: the 3 dense layers (12,288 inter), same rate;
    - attn_proj: DSA + KDA projections at FP8 (DeepGEMM sm90_fp8_gemm_1d2d / the derived FP8-attention checkpoint);
    - dsa_attn: sparse MLA over min(top-2048, ctx) keys per query, absorbed (576 qk + 512 v per head) at BF16;
    - indexer: 32 heads x 128 over the whole visible context per query, FP8 logits;
    - kda: chunked KDA (chunk 64): intra-chunk ~ 2 x c x (dk + dv) and state update ~ 4 x dk x dv per token-head;
    - lm_head: last token only (negligible)."""
    p = SKU[sku]
    t = tokens
    terms: dict[str, float] = {}
    moe_b = MOE_L * (touched_experts(t) + SHARED) * expert_params() * NVFP4_B / tp
    moe_f = 2 * t * MOE_L * (K + SHARED) * expert_params() / tp
    terms["moe"] = max(moe_b / p["hbm"], moe_f / p["moe"])
    dense_f = 2 * t * DENSE_L * 3 * H * DENSE_INTER / tp
    terms["dense_mlp"] = max(DENSE_L * 3 * H * DENSE_INTER * NVFP4_B / tp / p["hbm"], dense_f / p["moe"])
    proj = (DSA_L * (H * Q_LORA + Q_LORA * HEADS * QK + H * KV_LORA + HEADS * V * H)
            + KDA_L * 6 * H * KH * KD)
    terms["attn_proj"] = max(proj * attn_bytes / tp / p["hbm"], 2 * t * proj / tp / p["fp8"])
    keys = min(TOPK, context + t)
    terms["dsa_attn"] = 2 * t * DSA_L * HEADS * keys * (KV_LORA + 64 + KV_LORA) / tp / p["bf16"]
    vis = context + t / 2
    terms["indexer"] = 2 * t * DSA_L * IDX_H * IDX_D * vis / tp / p["fp8"]
    terms["kda"] = t * KDA_L * KH * (2 * kda_chunk * 2 * KD + 4 * KD * KD) / tp / p["bf16"]
    terms["total"] = sum(terms.values())
    out = {k: v * 1e3 for k, v in terms.items()}
    out["tok_s_per_gpu_bound"] = t / tp / terms["total"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="b300", choices=list(SKU))
    ap.add_argument("--prefill-tokens", type=int, default=0, help="also print the prefill floor for one chunk")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=28)
    ap.add_argument("--verify", type=int, default=6)
    ap.add_argument("--context", type=int, default=45000)
    ap.add_argument("--state-bytes", type=int, default=2)
    ap.add_argument("--attn-bytes", type=int, default=1)
    ap.add_argument("--accept", type=float, default=3.5)
    ap.add_argument("--measured-ms", type=float, default=None)
    a = ap.parse_args()
    f = decode_floor(a.tp, a.batch, a.verify, a.context, a.state_bytes, a.attn_bytes, a.sku)
    for k, v in f.items():
        print(f"{k:10s} {v:7.2f} ms")
    print(f"per-user bound {a.accept / f['total'] * 1e3:.0f} tok/s at accept {a.accept}")
    if a.measured_ms:
        print(f"measured {a.measured_ms} ms = {a.measured_ms / f['total']:.2f}x the floor")
    if a.prefill_tokens:
        pf = prefill_floor(a.prefill_tokens, a.tp, a.context, a.sku, a.attn_bytes)
        print(f"prefill chunk {a.prefill_tokens} tokens, TP{a.tp}, {a.sku}:")
        for k, v in pf.items():
            print(f"  {k:20s} {v:9.2f}" + ("" if k == "tok_s_per_gpu_bound" else " ms"))


if __name__ == "__main__":
    main()
