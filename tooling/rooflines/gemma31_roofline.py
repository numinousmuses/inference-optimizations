"""Gemma 4 31B (dense) on one H100 SXM: config-derived floor, v0.5.20 pool math, and the OpenRouter-shape throughput
model behind agents/findings/rtxv41_gemma31.md (agent rtxv41, 2026-10-05).

Every architecture number comes from config.json (RedHatAI/gemma-4-31B-it-FP8-block, text_config) and every KV
number from SGLang v0.5.20's own pool formulas:
  - model_executor/pool_configurator.py:603-616: full = kv_heads x (head_dim + v_head_dim) x dtype bytes; SWA ditto
    with the swa_* fields.
  - :657-667: hybrid cell = F x n_full + ratio x S x n_swa.
  - utils/hf_transformers/config.py:212-239: Gemma 4 maps its base (sliding) fields to swa_* and global_* to the base.
The throughput model is calibrated on the only measured H100 rows (research/gpu-evidence/gemma31_fp8_h100.json, August
image, FP8-block, FP8 KV):
  - std arm: 8K prompts, 10 sessions, 45.8 tok/s per user;
  - prefill: ~7.1K tok/s on 8K prompts;
  - mtp arm: 9 sessions, accept 5.44, 68.8 tok/s per user.
Labels: architecture terms are source_code; floors are derived; calibrated terms are modeled; the OR mix is measured
(research/openrouter/or_google_gemma-4-31b-it.json, 2026-10-05).

    python3 operations/kernels/gemma31_roofline.py                 # pool math + OR-shape table for the composed cell
    python3 operations/kernels/gemma31_roofline.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field

# ------------------------------------------------------------------------------------------------ hardware (assumed)
HBM = 3.35e12          # H100 SXM HBM3, bytes/s (NVIDIA published)
FP8 = 1.979e15         # dense FP8 MMA peak
BF16 = 0.989e15        # dense BF16 MMA peak
GPU_BYTES = 85.52e9    # 81,559 MiB as nvidia-smi reports an H100 80GB


# ------------------------------------------------------------------------------------- model (source_code: config)
@dataclass(frozen=True)
class Gemma31:
    layers: int = 60
    full_layers: int = 10            # layer_types: every 6th is full_attention
    hidden: int = 5376
    inter: int = 21504
    q_heads: int = 32
    swa_kv_heads: int = 16           # base num_key_value_heads (sliding) -> swa_num_key_value_heads
    swa_head_dim: int = 256
    full_kv_heads: int = 4           # num_global_key_value_heads -> num_key_value_heads
    full_head_dim: int = 512         # global_head_dim -> head_dim
    window: int = 1024               # sliding_window
    vocab: int = 262144              # tied embedding / lm_head (BF16: the FP8-block checkpoint ignores lm_head)

    @property
    def swa_layers(self) -> int:
        return self.layers - self.full_layers

    def linear_params(self) -> int:
        """All quantized Linear weights. A full layer's qkv_proj carries the v slot (SGLang builds QKVParallelLinear
        with the KV heads, and attention_k_eq_v loads k_proj into it: gemma4_causal.py:1187-1290)."""
        h = self.hidden
        mlp = 3 * h * self.inter
        swa_attn = h * (self.q_heads * self.swa_head_dim + 2 * self.swa_kv_heads * self.swa_head_dim) \
            + self.q_heads * self.swa_head_dim * h
        full_attn = h * (self.q_heads * self.full_head_dim + 2 * self.full_kv_heads * self.full_head_dim) \
            + self.q_heads * self.full_head_dim * h
        return self.swa_layers * (mlp + swa_attn) + self.full_layers * (mlp + full_attn)

    def embed_params(self) -> int:
        return self.vocab * self.hidden

    def kv_full_per_token_layer(self, kv_bytes: float = 1.0) -> float:
        return self.full_kv_heads * (self.full_head_dim * 2) * kv_bytes      # K and V rows (V = v_norm(k_proj x))

    def kv_swa_per_token_layer(self, kv_bytes: float = 1.0) -> float:
        return self.swa_kv_heads * (self.swa_head_dim * 2) * kv_bytes

    def kv_full_length_per_token(self, kv_bytes: float = 1.0) -> float:
        """Every layer holds the whole context (no hybrid pool)."""
        return self.full_layers * self.kv_full_per_token_layer(kv_bytes) + \
            self.swa_layers * self.kv_swa_per_token_layer(kv_bytes)

    def hybrid_cell(self, ratio: float, kv_bytes: float = 1.0) -> float:
        """pool_configurator.py:664-691: bytes per full-pool token with swa_tokens = ratio x full_tokens."""
        return self.full_layers * self.kv_full_per_token_layer(kv_bytes) + \
            ratio * self.swa_layers * self.kv_swa_per_token_layer(kv_bytes)


# Gemma 4 31B assistant (google/gemma-4-31B-it-assistant, config.json): 4 layers (3 sliding + 1 full) at hidden 1024
# that read the TARGET's KV (num_kv_shared_layers 4: FROZEN_KV_MTP, speculative_hook.py:70-75) and a centroid head
# (2048 centroids, top 32 -> 32 x 128 candidate rows of the 262,144 x 1024 BF16 embedding).
DRAFT_WEIGHTS_BYTES = 469_518_596 * 2               # BF16 safetensors total (measured file metadata)
DRAFT_STEP_READ_BYTES = 2 * (4 * (3 * 1024 * 8192 + 2 * 1024 * 8192) + 2 * 5376 * 1024 + 2048 * 1024
                             + 32 * 128 * 1024)    # 4 MLP+attn blocks, pre/post projections, centroids, selected rows


# RedHatAI/gemma-4-31B-it-FP8-block @ d7242548: model.safetensors.index.json metadata.total_size (measured)
CHECKPOINT_BYTES = 33_263_025_112

# ------------------------------------------------------------------------------------------- OR mix (measured)
OR_PROMPT = 2315        # 32.12B prompt / 13.87M requests (2026-10-05 analytics row)
OR_OUTPUT = 277         # 3.84B completion / 13.87M requests
OR_CACHE_HIT = 0.277    # weighted cache_hit_rate across providers


@dataclass
class Calib:
    """modeled: efficiencies fitted to the August H100 rows (calibrate()); nominal = v0.5.20 expectations."""
    bw_eff: float = 0.85             # achieved fraction of HBM on weight/KV streams
    step_fixed_s: float = 7.8e-3     # per-target-step fixed cost (fitted to the std arm at bw_eff 0.85)
    gemm_eff: float = 0.265          # FP8 GEMM fraction of peak in prefill (August image, fitted)
    attn_tflops: float = 150e12      # Triton extend attention, BF16 math (RTX measured 72-76 of 380: ~20%)
    draft_step_s: float = 11.6e-3    # August mtp arm: (79 ms cycle - verify) / 5 draft steps
    verify_flops_eff: float = 0.6    # FP8 MMA fraction once a verify batch is compute-bound
    notes: list = field(default_factory=list)


# weight formats for the linears: (bytes per param, MMA peak). int4 g128 + a BF16 scale per 128 = 0.516 B/param.
# W4A16 (Marlin, compressed_tensors_wNa16.py) runs BF16 MMA; W4A8 would keep the FP8 MMA (no dense W4A8 linear exists in
# v0.5.20: compressed_tensors_w4a8_fp8_moe.py is MoE-only).
WEIGHTS = {"fp8": (1.0, FP8), "w4a16": (0.516, BF16), "w4a8": (0.516, FP8)}
DEQUANT_COST = {"fp8": 1.0, "w4a16": 1.0, "w4a8": 1.15}   # int4 -> fp8 conversion in the mainloop (assumed 15%)


def decode_bytes(m: Gemma31, n_req: int, ctx: int, kv_bytes: float = 1.0, weights: str = "fp8") -> float:
    w = m.linear_params() * WEIGHTS[weights][0] + m.embed_params() * 2.0    # linears + BF16 lm_head
    kv = n_req * (m.swa_layers * min(ctx, m.window) * m.kv_swa_per_token_layer(kv_bytes)
                  + m.full_layers * ctx * m.kv_full_per_token_layer(kv_bytes))
    return w + kv


def verify_time(m: Gemma31, c: Calib, n_req: int, ctx: int, k: int, weights: str = "fp8",
                kv_bytes: float = 1.0) -> float:
    """One target forward over n_req x k tokens (k = 1: plain decode)."""
    t_bw = decode_bytes(m, n_req, ctx, kv_bytes, weights) / (HBM * c.bw_eff)
    flops = 2.0 * m.linear_params() * n_req * k
    t_fl = flops * DEQUANT_COST[weights] / (WEIGHTS[weights][1] * c.verify_flops_eff) \
        + 2.0 * m.embed_params() * n_req * k / (BF16 * c.verify_flops_eff)
    return max(t_bw, t_fl) + c.step_fixed_s


def cycle_time(m: Gemma31, c: Calib, n_req: int, ctx: int, steps: int, weights: str = "fp8",
               kv_bytes: float = 1.0) -> float:
    if steps == 0:
        return verify_time(m, c, n_req, ctx, 1, weights, kv_bytes)
    return verify_time(m, c, n_req, ctx, steps + 1, weights, kv_bytes) + steps * c.draft_step_s


def prefill_token_time(m: Gemma31, c: Calib, ctx: int, weights: str = "fp8") -> float:
    """Seconds per uncached prompt token for a prompt of length ctx, prefilled in one chunk. gemm_eff is the fraction
    of the format's own MMA peak (FP8 for fp8 / w4a8, BF16 for w4a16)."""
    gemm = 2.0 * m.linear_params() * DEQUANT_COST[weights] / (WEIGHTS[weights][1] * c.gemm_eff)
    avg_keys = ctx / 2.0
    attn = 4.0 * m.q_heads * (m.swa_layers * m.swa_head_dim * min(avg_keys, m.window)
                              + m.full_layers * m.full_head_dim * avg_keys)
    return gemm + attn / c.attn_tflops


def calibrate() -> Calib:
    """Fit step_fixed_s, gemm_eff and draft_step_s to the August rows (modeled)."""
    m, c = Gemma31(), Calib()
    ctx = 8192 + 400                                  # 8K prompt, mid-decode of 800 output tokens
    std_step = 1.0 / 45.84                            # std arm steady 45.84 tok/s per user at 10 sessions
    c.step_fixed_s = std_step - decode_bytes(m, 10, ctx) / (HBM * c.bw_eff)
    pf = 1.0 / 7100.0                                 # ~7.1K tok/s prefill on 8K prompts
    attn = prefill_token_time(m, Calib(gemm_eff=1e9, attn_tflops=c.attn_tflops), 8192)
    c.gemm_eff = 2.0 * m.linear_params() / (FP8 * (pf - attn))
    mtp_cycle = 5.44 / 68.77                          # accept 5.44 at 68.77 tok/s per user, 9 sessions, steps 5
    c.draft_step_s = (mtp_cycle - verify_time(m, c, 9, ctx, 6)) / 5
    c.notes.append(f"fit: step_fixed {c.step_fixed_s*1e3:.1f} ms, gemm_eff {c.gemm_eff:.3f}, "
                   f"draft step {c.draft_step_s*1e3:.1f} ms")
    return c


def nominal_v0520(c: Calib) -> Calib:
    """modeled expectations for v0.5.20 (each an assumption the ONE test measures):
    - DeepGEMM FP8 block GEMMs (fp8_utils.py:801-805 picks DeepGEMM first): 0.55 of the FP8 peak at prefill shapes;
    - overlap scheduling hides the host: step_fixed halves;
    - the frozen-KV MTP draft is CUDA-graphed (frozen_kv_mtp_cuda_graph_runner.py): a draft step is its bytes plus
      a 1.5 ms fixed cost instead of the August 11.6 ms."""
    return Calib(bw_eff=c.bw_eff, step_fixed_s=c.step_fixed_s / 2, gemm_eff=0.55, attn_tflops=c.attn_tflops,
                 draft_step_s=DRAFT_STEP_READ_BYTES / (HBM * c.bw_eff) + 1.5e-3, verify_flops_eff=0.6,
                 notes=["nominal v0.5.20 (assumed until the ONE test)"])


# ----------------------------------------------------------------------------------------- pool math (derived)
def kv_budget(weights_bytes: float, mem_fraction: float = 0.88, draft_bytes: float = DRAFT_WEIGHTS_BYTES,
              free_at_start: float = GPU_BYTES - 0.6e9) -> float:
    """v0.5.20 profile_max_num_token: KV = free after load - (1 - mem_fraction) x total."""
    return free_at_start - weights_bytes - draft_bytes - (1.0 - mem_fraction) * GPU_BYTES


def pool(m: Gemma31, kv_bytes_total: float, ratio: float, kv_dtype_bytes: float = 1.0) -> dict:
    full = int(kv_bytes_total // m.hybrid_cell(ratio, kv_dtype_bytes))
    return {"ratio": ratio, "cell_bytes": m.hybrid_cell(ratio, kv_dtype_bytes), "full_tokens": full,
            "swa_tokens": int(full * ratio)}


def concurrency(m: Gemma31, p: dict, ctx_full: int, swa_per_req: int, chunk: int) -> int:
    """Requests the pools hold: full tokens / context, and (SWA tokens - one prefill chunk) / per-request window."""
    return max(0, min(p["full_tokens"] // ctx_full, (p["swa_tokens"] - chunk) // swa_per_req))


def or_shape(m: Gemma31, c: Calib, n_req: int, accept: float, steps: int, cache_hit: float = OR_CACHE_HIT,
             prompt: int = OR_PROMPT, out: int = OR_OUTPUT, weights: str = "fp8", kv_bytes: float = 1.0) -> dict:
    """Fluid steady state at the OpenRouter mix: per output token the GPU spends (uncached prompt / output) prefill
    tokens plus 1/(n_req x accept) of a decode cycle. warm = output tok/s; per_user = the decode rate a running
    request sees with prefill interleaved."""
    ctx = prompt + out // 2
    uncached = prompt * (1.0 - cache_hit)
    t_pf = (uncached / out) * prefill_token_time(m, c, prompt, weights)
    tcyc = cycle_time(m, c, n_req, ctx, steps, weights, kv_bytes)
    tok_per_cycle = n_req * (accept if steps else 1.0)
    t_dec = tcyc / tok_per_cycle
    warm = 1.0 / (t_pf + t_dec)
    decode_share = t_dec / (t_pf + t_dec)
    return {"n_req": n_req, "warm_tok_s": round(warm, 1), "per_user": round(warm / n_req, 1),
            "decode_share": round(decode_share, 3), "cycle_ms": round(tcyc * 1e3, 1),
            "prefill_tok_s": round(1.0 / prefill_token_time(m, c, prompt, weights))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--accept", type=float, default=3.5, help="real-traffic accept length incl. bonus (assumed)")
    ap.add_argument("--ratio", type=float, default=0.35)
    a = ap.parse_args(argv)
    m = Gemma31()
    cal = calibrate()
    nom = nominal_v0520(cal)
    w_fp8 = CHECKPOINT_BYTES                    # the FP8-block safetensors total (vision tower and scales included)
    budget = kv_budget(w_fp8)
    out = {"linear_params": m.linear_params(), "embed_params": m.embed_params(),
           "kv_full_length_per_token": m.kv_full_length_per_token(), "weights_fp8_bytes": w_fp8,
           "kv_budget_bytes": budget, "calibration": cal.notes,
           "pools": [pool(m, budget, r) for r in (0.8, 0.5, 0.35, 0.25)],
           "full_length_tokens": int(budget // m.kv_full_length_per_token())}
    p = pool(m, budget, a.ratio)
    out["concurrency_or"] = concurrency(m, p, OR_PROMPT + OR_OUTPUT, m.window + 8, 8192)
    rows = []
    for label, c in (("august-calibrated", cal), ("v0.5.20-nominal", nom)):
        for steps, acc in ((0, 1.0), (5, a.accept), (3, min(a.accept, 3.2))):
            for n in (8, 16, 24, 32, 48, 64):
                r = or_shape(m, c, n, acc, steps)
                r.update(model=label, steps=steps, accept=acc)
                rows.append(r)
    out["or_rows"] = rows
    print(json.dumps({k: v for k, v in out.items() if k != "or_rows"}, indent=1))
    print(f"{'model':18s} {'steps':>5s} {'acc':>4s} {'N':>3s} {'warm':>7s} {'per_user':>8s} {'dec%':>5s} {'cycle ms':>8s}")
    for r in rows:
        print(f"{r['model']:18s} {r['steps']:5d} {r['accept']:4.1f} {r['n_req']:3d} {r['warm_tok_s']:7.0f} "
              f"{r['per_user']:8.1f} {100*r['decode_share']:5.0f} {r['cycle_ms']:8.1f}")
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
