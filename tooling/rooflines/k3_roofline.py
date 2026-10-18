"""Kimi K3 (moonshotai/Kimi-K3, MXFP4 routed experts, bf16 everything else) floor per node for the served layout:
8 GPUs, TP8 + DCP8, DSPARK block 7 (8 verify tokens per request per step) + ReplaySSM, fp8 MLA KV. Shared by k3amd
(8 x MI355X) and k3b300 (8 x B300). A floor, not a forecast: each term is its HBM bytes at peak bandwidth or its FLOPs
at the format's dense peak, whichever is larger; nothing runs at 100%.

Decode/verify step (per GPU, B running requests at context C, accept A tokens per request per step):
  routed experts   MXFP4 weights of every expert any of the B x 8 x 16 routes touches (TP8 shards each expert's
                   intermediate, so every GPU reads its shard of every touched expert); FLOPs at the FP8 peak
                   (AITER A8W4 on MI355X, trtllm-gen W4A8 on B300)
  dense            attention (69 KDA + 24 MLA layers), 2 shared experts, dense layer 0, router, lm_head: TP8-sharded;
                   the latent down/up projections (4.73 G params) are replicated: every GPU reads and runs all of
                   them. bf16, or block FP8 for attention + latent + shared with --fp8-dense
  MLA KV           B x C / 8 tokens per GPU (DCP8) x 24 layers x 576 B, read once per step (the 8 verify queries share)
  draft KV         the DSPARK draft's KV is replicated: B x C x ~1.25 KB per GPU per draft pass (derived, B300 boot)
  KDA state        verify reads the state, the commit fold reads and writes it (3 x 51.75 MB per request per GPU
                   fp32 at 12 heads/GPU); the ring writes are small
Prefill (the OR mix re-sends 89% cached prompts; each request prefills its ~7.8K new tokens): 2 x active params
FLOPs per token (bf16 dense ~61B, routed 16 of 896 experts ~48.6B at FP8) + MLA attention over the context.

Throughput floor at the OR mix, per node: an output token costs step_time / (B x A) of decode plus
(new tokens per output token) x prefill time per token; the node serves 1 / (sum).

    python3 operations/kernels/k3_roofline.py [--gpu MI355X|B300] [--seats 96] [--context 71733] [--accept 2.3]
        [--eff 0.2,0.35]      # efficiency band applied to the floor for the derived served range

Peaks (assumed, vendor dense numbers): MI355X HBM3E 8 TB/s, FP8 5 PF (MXFP4 10 PF), BF16 2.5 PF; B300 HBM3E 8 TB/s,
FP8 5 PF (FP4 15 PF), BF16 2.5 PF. Calibration (measured, 8 x B300 k3_deep 60K warm c=32, accept 2.31): 920 tok/s
served vs this floor -> the efficiency the B300 production recipe reached (printed by --calibrate).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

HBM = {"MI355X": 8.0e12, "B300": 8.0e12}
FP8 = {"MI355X": 5.0e15, "B300": 5.0e15}
BF16 = {"MI355X": 2.5e15, "B300": 2.5e15}
GPUS = 8

H, LAYERS, KDA, MLA, MOE = 7168, 93, 69, 24, 92
E, TOPK, INTER, LATENT, SHARED = 896, 16, 3072, 3584, 2
HEADS, HD = 96, 128
Q_LORA, KV_LORA, ROPE, NOPE, VD = 1536, 512, 64, 128, 128
DENSE_INTER, VOCAB = 33792, 163840
MXFP4_B = 0.53125                      # 4-bit value + one e8m0 per 32
KV_B = MLA * (KV_LORA + ROPE)          # 13,824 B per token (fp8)
DRAFT_KV_B = 1_250                     # per token, replicated (derived from the B300 boot, k3_seats.py)
STATE_B_GPU = KDA * (HEADS // GPUS) * HD * HD * 4   # 51.75 MiB fp32 per request per GPU
VERIFY = 8                             # DSPARK block 7 + 1

EXPERT_PARAMS = 3 * LATENT * INTER     # gate + up + down in the latent space
KDA_PARAMS = 4 * H * HEADS * HD + H * HEADS * HD + H * HD + HD * HEADS * HD + H * HEADS   # qkvg, o, f_a, f_b, b
MLA_PARAMS = (H * Q_LORA + Q_LORA * HEADS * (NOPE + ROPE) + H * (KV_LORA + ROPE) + KV_LORA * HEADS * (NOPE + VD)
              + HEADS * VD * H + H * HEADS * VD)    # + output gate
MOE_DENSE = SHARED * 3 * H * INTER + 2 * H * LATENT + H * E   # shared experts, latent down/up, router
DENSE_PARAMS = KDA * KDA_PARAMS + MLA * MLA_PARAMS + MOE * MOE_DENSE + 3 * H * DENSE_INTER + 2 * VOCAB * H
# the latent down/up projections are ReplicatedLinear: every GPU holds and runs all of them (k3b300 measured 4.73 G
# params from the checkpoint headers); everything else dense is TP8-sharded
LATENT_PARAMS = MOE * 2 * H * LATENT
SHARDED_DENSE = DENSE_PARAMS - LATENT_PARAMS
# block-FP8 candidates (k3b300's NUMINOUS_K3_FP8ATTN / FP8LATENT / FP8SHARED): attention, latent, shared experts
FP8ABLE_SHARDED = KDA * KDA_PARAMS + MLA * MLA_PARAMS + MOE * SHARED * 3 * H * INTER
EMBED = VOCAB * H                      # input embedding: a lookup, never a GEMM


def dense_per_gpu(fp8_dense: bool) -> tuple[float, float, float]:
    """(bytes read per step, bf16 GEMM params per row, fp8 GEMM params per row) per GPU."""
    fp8_sh = FP8ABLE_SHARDED / GPUS if fp8_dense else 0.0
    fp8_lat = LATENT_PARAMS if fp8_dense else 0.0
    bf16_sh = (SHARDED_DENSE - EMBED) / GPUS - fp8_sh
    bf16_lat = LATENT_PARAMS - fp8_lat
    nbytes = 2 * (bf16_sh + bf16_lat) + (fp8_sh + fp8_lat) + 2 * EMBED / GPUS * 0   # embedding rows: negligible
    return nbytes, bf16_sh + bf16_lat, fp8_sh + fp8_lat


def weights_gb() -> dict:
    return {"routed_experts": MOE * E * EXPERT_PARAMS * MXFP4_B / 1e9, "dense_bf16": DENSE_PARAMS * 2 / 1e9}


def decode_step_ms(gpu: str, seats: int, context: int, fp8_dense: bool = False) -> dict:
    hbm = HBM[gpu]
    routes = seats * VERIFY * TOPK
    touched = 1 - (1 - 1 / E) ** routes
    expert_b = MOE * E * EXPERT_PARAMS * MXFP4_B / GPUS * touched
    expert_f = 2 * seats * VERIFY * TOPK * MOE * EXPERT_PARAMS / GPUS
    dense_b, bf16_p, fp8_p = dense_per_gpu(fp8_dense)
    rows_n = seats * VERIFY
    kv_b = seats * context / GPUS * KV_B
    draft_b = seats * context * DRAFT_KV_B
    state_b = seats * 3 * STATE_B_GPU
    rows = {
        "routed experts": max(expert_b / hbm, expert_f / FP8[gpu]),
        "dense (bf16 / fp8)": max(dense_b / hbm, 2 * rows_n * (bf16_p / BF16[gpu] + fp8_p / FP8[gpu])),
        "MLA KV (DCP8)": kv_b / hbm,
        "draft KV (replicated)": draft_b / hbm,
        "KDA state (verify + fold)": state_b / hbm,
    }
    return {k: v * 1e3 for k, v in rows.items()} | {"_touched": touched}


def prefill_ms_per_token(gpu: str, context: int, fp8_dense: bool = False) -> float:
    """Per GPU: TP-sharded GEMMs split 8 ways; the replicated latent projections run in full on every GPU."""
    _, bf16_p, fp8_p = dense_per_gpu(fp8_dense)
    routed = MOE * TOPK * EXPERT_PARAMS / GPUS
    attn = 2 * MLA * HEADS * (context / 2) * (NOPE + ROPE + VD) / GPUS  # causal average over the prompt
    t = 2 * bf16_p / BF16[gpu] + 2 * fp8_p / FP8[gpu] + 2 * routed / FP8[gpu] + 2 * attn / BF16[gpu]
    return t * 1e3


def or_floor(gpu: str, seats: int, context: int, accept: float, new_per_out: float, fp8_dense: bool = False) -> dict:
    step = decode_step_ms(gpu, seats, context, fp8_dense)
    step_ms = sum(v for k, v in step.items() if not k.startswith("_"))
    decode_per_tok = step_ms / (seats * accept)
    prefill_per_tok = new_per_out * prefill_ms_per_token(gpu, context, fp8_dense)
    per_out = decode_per_tok + prefill_per_tok
    return {"gpu": gpu, "seats": seats, "context": context, "accept": accept, "step_ms": round(step_ms, 2),
            "step_terms_ms": {k: round(v, 2) for k, v in step.items() if not k.startswith("_")},
            "experts_touched": round(step["_touched"], 3),
            "decode_ms_per_out_token": round(decode_per_tok, 4), "prefill_ms_per_out_token": round(prefill_per_tok, 4),
            "floor_tok_s": round(1e3 / per_out), "floor_per_user_tok_s": round(1e3 / per_out / seats, 1)}


OR_NEW_PER_OUT = 70_700 * 0.11 / 1_027        # 7.57 prefilled tokens per output token at the OR mix


def calibrate() -> dict:
    """B300 k3_deep, 60K warm, c=32 (33 running), accept 2.31, 920 tok/s (measured). In-run hit 92%: ~4.8K new
    tokens per 930-token answer."""
    f = or_floor("B300", 33, 60_000, 2.31, 60_000 * (1 - 0.9216) / 930)
    return {"floor_tok_s": f["floor_tok_s"], "measured_tok_s": 920, "efficiency": round(920 / f["floor_tok_s"], 3)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="MI355X")
    ap.add_argument("--seats", default="32,48,64,96,128,160")
    ap.add_argument("--context", type=int, default=71_733)
    ap.add_argument("--accept", type=float, default=2.3)
    ap.add_argument("--eff", default="0.20,0.35")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--fp8-dense", action="store_true", help="attention + latent + shared experts in block FP8")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    cal = calibrate()
    if a.calibrate:
        print(json.dumps(cal, indent=1))
        return 0
    lo, hi = (float(x) for x in a.eff.split(","))
    w = weights_gb()
    print(f"weights: routed experts {w['routed_experts']:.0f} GB, dense bf16 {w['dense_bf16']:.0f} GB "
          f"(checkpoint 1,561 GB); B300 calibration efficiency {cal['efficiency']}")
    rows = []
    print("| gpu | seats | step floor ms | experts / dense / MLA KV / draft KV / KDA ms | floor tok/s | served range "
          f"({lo:.0%}-{hi:.0%}) | per user |")
    print("|---|---|---|---|---|---|---|")
    for gpu in a.gpu.split(","):
        for s in (int(x) for x in a.seats.split(",")):
            f = or_floor(gpu, s, a.context, a.accept, OR_NEW_PER_OUT, a.fp8_dense)
            t = f["step_terms_ms"]
            f["served_range_tok_s"] = [round(f["floor_tok_s"] * lo), round(f["floor_tok_s"] * hi)]
            rows.append(f)
            print(f"| {gpu} | {s} | {f['step_ms']} | " + " / ".join(f"{v:.1f}" for v in t.values())
                  + f" | {f['floor_tok_s']:,} | {f['served_range_tok_s'][0]:,}-{f['served_range_tok_s'][1]:,} | "
                  f"{f['served_range_tok_s'][0] / s:.0f}-{f['served_range_tok_s'][1] / s:.0f} |")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"calibration": cal, "rows": rows}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
