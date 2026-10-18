"""Per-operation roofline for DeepSeek V4 Pro (0813) on B200 / B300, at the shapes the served recipe runs.

Every kernel in a profiled step belongs to one of these operations. Its floor is max(bytes / HBM bandwidth,
FLOPs / tensor-core rate) at the operation's dtype; measured time over the floor says how much a better kernel can
buy. Shapes come from the checkpoint config (deepseek-ai/DeepSeek-V4-Pro-0813):

  hidden 7168, 61 layers, 128 query heads of 512 (64 rope), one KV head (MQA over a compressed cache),
  q_lora 1536, o_lora 1024 in 16 groups, 384 routed experts (top 6, intermediate 3072, MXFP4) + 1 shared (FP8),
  layer compress ratios: 30 x C4 (with a 64-head x 128 indexer, top 1024), 31 x C128; sliding window 128.

Served layout (registry): DP8 attention (each GPU owns its requests and holds full attention weights), TP8 experts
(each GPU holds 1/8 of every expert's intermediate dimension), DSpark speculation (verify width 6 tokens per request).

    python3 operations/kernels/pro_roofline.py --tokens 96 --context 46000 --gpu B200
"""
from __future__ import annotations

import argparse
import dataclasses

HIDDEN, LAYERS, HEADS, HEAD_DIM, ROPE = 7168, 61, 128, 512, 64
Q_LORA, O_LORA, O_GROUPS = 1536, 1024, 16
EXPERTS, TOPK, MOE_INTER = 384, 6, 3072
C4_LAYERS, C128_LAYERS = 30, 31
INDEX_HEADS, INDEX_DIM, INDEX_TOPK, WINDOW = 64, 128, 1024, 128
VOCAB = 129280
DP = TP = 8

# dense rates (no sparsity); HBM in bytes/s
GPUS = {
    "B200": {"hbm": 8.0e12, "fp8": 4.5e15, "fp4": 9.0e15, "bf16": 2.25e15, "nvlink": 0.9e12},
    "B300": {"hbm": 8.0e12, "fp8": 4.5e15, "fp4": 13.5e15, "bf16": 2.25e15, "nvlink": 0.9e12},
}


@dataclasses.dataclass
class Op:
    name: str
    per_layer: bool          # False: once per step (lm head)
    layers: int              # how many layers run it
    bytes: float             # per layer (or per step) per GPU
    flops: float
    rate: str                # tensor-core rate key for the flops

    def floor_us(self, gpu: str) -> float:
        g = GPUS[gpu]
        t = max(self.bytes / g["hbm"], self.flops / g[self.rate] if self.flops else 0.0)
        return t * 1e6

    def step_floor_us(self, gpu: str) -> float:
        return self.floor_us(gpu) * (self.layers if self.per_layer else 1)


def ops(tokens: int, context: int, phase: str = "decode") -> list[Op]:
    """Operations of one step on one GPU. tokens = this GPU's tokens in the step (decode: requests x verify width;
    prefill: the per-rank chunk). context = tokens each request attends over (for the cache reads)."""
    t = tokens
    g = t * DP                                   # tokens every GPU sees in the TP-sharded MoE (DP gather)
    act = 2                                      # bf16 activation bytes
    out = []
    # attention projections: fp8 weights (1 byte) replicated per GPU under DP attention
    out.append(Op("wqkv_a 7168->2048 (fp8)", True, LAYERS, HIDDEN * 2048 + t * (HIDDEN + 2048) * act,
                  2 * t * HIDDEN * 2048, "fp8"))
    out.append(Op("wq_b 1536->65536 (fp8)", True, LAYERS, Q_LORA * HEADS * HEAD_DIM + t * (Q_LORA + HEADS * HEAD_DIM) * act,
                  2 * t * Q_LORA * HEADS * HEAD_DIM, "fp8"))
    out.append(Op("wo_a 16x(4096->1024) (fp8 einsum)", True, LAYERS,
                  HEADS * HEAD_DIM * O_LORA + t * (HEADS * HEAD_DIM + O_GROUPS * O_LORA) * act,
                  2 * t * HEADS * HEAD_DIM * O_LORA, "fp8"))
    out.append(Op("wo_b 16384->7168 (fp8)", True, LAYERS, O_GROUPS * O_LORA * HIDDEN + t * (O_GROUPS * O_LORA + HIDDEN) * act,
                  2 * t * O_GROUPS * O_LORA * HIDDEN, "fp8"))
    # attention core: each query token reads its selected compressed entries (C4: top 1024; C128: all of
    # context/128) plus the 128-token window; the cache stores 448 fp8 nope + 64 bf16 rope per entry
    entry = 448 + 64 * 2
    sel_c4 = min(INDEX_TOPK, context // 4) + WINDOW
    sel_c128 = context // 128 + WINDOW
    reqs = max(1, t // 6) if phase == "decode" else 1
    for name, n_layers, sel in (("sparse attention C4 layers", C4_LAYERS, sel_c4),
                                ("sparse attention C128 layers", C128_LAYERS, sel_c128)):
        kv_bytes = (reqs if phase == "decode" else 1) * sel * entry          # verify tokens of a request share its KV
        out.append(Op(name, True, n_layers, kv_bytes + t * HEADS * HEAD_DIM * act * 2,
                      2 * 2 * t * HEADS * sel * HEAD_DIM, "fp8"))
    # indexer (C4 layers): q_idx (64 x 128) against every compressed entry of the context, fp8
    comp = context // 4
    out.append(Op("indexer logits (C4)", True, C4_LAYERS, reqs * comp * INDEX_DIM + t * INDEX_HEADS * INDEX_DIM,
                  2 * t * INDEX_HEADS * comp * INDEX_DIM, "fp8"))
    # MoE: router (bf16 7168 x 384), shared expert (fp8 7168->6144, 3072->7168) on the gathered tokens,
    # routed experts TP-sharded: every GPU holds 1/8 of each expert (MXFP4, 0.5 byte + 1/32 scale)
    out.append(Op("router 7168->384 (bf16)", True, LAYERS, HIDDEN * EXPERTS * 2 + g * HIDDEN * act,
                  2 * g * HIDDEN * EXPERTS, "bf16"))
    out.append(Op("shared expert (fp8, gathered tokens)", True, LAYERS,
                  HIDDEN * 2 * MOE_INTER + MOE_INTER * HIDDEN + g * (HIDDEN + 2 * MOE_INTER) * act,
                  2 * g * 3 * HIDDEN * MOE_INTER, "fp8"))
    pairs = g * TOPK
    active = min(EXPERTS, pairs)                                          # experts touched this step
    w_bytes = active * (HIDDEN * 2 * MOE_INTER + MOE_INTER * HIDDEN) / TP * (0.5 + 1 / 32)
    out.append(Op("routed experts FC1+FC2 (MXFP4 w, MXFP8 act)", True, LAYERS, w_bytes + pairs * HIDDEN * 2,
                  2 * pairs * 3 * HIDDEN * MOE_INTER / TP, "fp8"))
    # DP gather (all tokens to every GPU) and the TP combine (all-reduce of the full output): NVLink
    comm = g * HIDDEN * act
    out.append(Op("DP all-gather + TP all-reduce (NVLink)", True, LAYERS, 0.0, 0.0, "bf16"))
    out[-1].bytes = comm * (7 / 8 + 2 * 7 / 8) * GPUS["B200"]["hbm"] / GPUS["B200"]["nvlink"]   # time-equivalent bytes
    # lm head once per step (each GPU its own tokens under dp-lm-head)
    out.append(Op("lm head 7168->129280 (bf16)", False, 1, HIDDEN * VOCAB * 2 + t * VOCAB * 4,
                  2 * t * HIDDEN * VOCAB, "bf16"))
    return out


def table(tokens: int, context: int, gpu: str, phase: str = "decode") -> tuple[list[tuple[str, float]], float]:
    rows = [(o.name, o.step_floor_us(gpu)) for o in ops(tokens, context, phase)]
    return rows, sum(r[1] for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--context", type=int, default=46000)
    ap.add_argument("--gpu", default="B200")
    ap.add_argument("--phase", default="decode")
    a = ap.parse_args()
    rows, total = table(a.tokens, a.context, a.gpu, a.phase)
    print(f"{a.phase} step floor, {a.tokens} tokens per GPU, {a.context} context, {a.gpu}")
    for name, us in sorted(rows, key=lambda r: -r[1]):
        print(f"  {us / 1000:8.2f} ms  {name}")
    print(f"  {total / 1000:8.2f} ms  total")


if __name__ == "__main__":
    main()
