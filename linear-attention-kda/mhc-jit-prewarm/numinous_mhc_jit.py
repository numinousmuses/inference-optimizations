"""mHC TileLang JIT stall fix for GLM-5.3-Flash (Glm5Next) on SGLang v0.5.20 (agent rtxv41, 2026-10-01).

Measured (rtxsys run 8, 4 x RTX DP4+EP4, research/gpu-evidence/rtxv41_glmstall/): every long prefill stall was one
rank's host compiling `mhc_pre_big_fuse_with_norm_tilelang` for a new `n_splits` (11.5-11.9 s each, no CUDA API
call in between) while the other ranks spun in the next AllReduce: 6 stalls = ~70 s of a 92.6 s window with
4.7-6.7 s of GPU compute per rank.

Why (source_code v0.5.20 kernels/ops/layernorm/mhc.py):
- the big-fuse kernels are @tilelang.jit with `n_splits` as a compile-time int;
- with SGLANG_OPT_DEEPGEMM_HC_PRENORM on, n_splits = min(n_sms // ceil(T/64), K/256): ~24 distinct values on
  188 SMs for T <= 4096, so each new prefill tail length can trigger a compile on whichever rank meets it first;
- upstream prewarms one token count per n_splits bucket at load only for DeepseekV4 (deepseek_v4.py
  _prewarm_mhc_kernels); Glm5Next has no prewarm;
- TileLang's disk cache is not redirected to SGLANG_CACHE_DIR, so every fresh container recompiles.

Fixes (installed into the engine as srt/layers/numinous_mhc_jit.py by operations/patches/rtx_mhc_jit_hooks.py):
- prewarm_glm5next(model): at the end of Glm5NextForConditionalGeneration.post_load_weights, once per process,
  run each mHC decoder layer kind's real hc_attn_pre path for one token count per n_splits bucket (with and
  without the fused output norm), before any forward and before CUDA-graph capture
  (NUMINOUS_MHC_PREWARM=1).
- nsplit_pow2(n): round the split count down to a power of two, so there are 6-7 buckets instead of ~24 (fewer
  compiles at prewarm and on a cold cache; split-K changes only the fp32 summation order of the prenorm GEMM)
  (NUMINOUS_MHC_NSPLIT_POW2=1).
- Env for the arm (no code): TILELANG_CACHE_DIR on the persistent volume (DeepGEMM already follows
  SGLANG_CACHE_DIR), so the second boot loads every compiled bucket from disk.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

_PREWARMED = False


def nsplit_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n.bit_length() - 1)


def maybe_pow2(n: int) -> int:
    return nsplit_pow2(n) if os.environ.get("NUMINOUS_MHC_NSPLIT_POW2", "0") == "1" else n


def bucket_token_counts(max_num_tokens: int, hc_hidden_size: int, split_fn) -> tuple:
    """One representative token count per distinct split_fn value over [1, max_num_tokens] (64-token grid, as
    upstream's get_mhc_pre_token_count_representatives)."""
    reps = {}
    for grid in range(1, (max(1, max_num_tokens) + 63) // 64 + 1):
        t = min(grid * 64, max_num_tokens)
        reps[split_fn(t, hc_hidden_size)] = t
    return tuple(sorted(reps.values()))


def _mhc_layers(model):
    """One representative mHC decoder layer per kind (linear-attention vs DSA layers share the kernels, but the
    cost of one extra prewarm pass per kind is small and guards against kind-specific arguments)."""
    seen, out = set(), []
    for m in model.modules():
        if hasattr(m, "hc_attn_fn") and hasattr(m, "hc_attn_pre") and getattr(getattr(m, "config", None), "mhc", False):
            kind = (bool(getattr(m, "is_linear_attn", False)), tuple(m.hc_attn_fn.shape))
            if kind not in seen:
                seen.add(kind)
                out.append(m)
    return out


def prewarm_glm5next(model, max_num_tokens: int | None = None) -> dict:
    """Compile the mHC pre kernels for every n_splits bucket through the layers' own code path. Idempotent per
    process. Returns a small report (also logged)."""
    global _PREWARMED
    if _PREWARMED or os.environ.get("NUMINOUS_MHC_PREWARM", "0") != "1":
        return {"skipped": True}
    _PREWARMED = True
    import torch

    from sglang.kernels.ops.layernorm import mhc as _mhc

    if max_num_tokens is None:
        try:
            from sglang.srt.runtime_context import get_schedule

            max_num_tokens = int(get_schedule().chunked_prefill_size or 0) or 4096
        except Exception:  # noqa: BLE001
            max_num_tokens = 4096
    layers = _mhc_layers(model)
    if not layers:
        return {"skipped": True, "reason": "no mHC layers"}
    t0 = time.perf_counter()
    calls = 0
    with torch.inference_mode():
        for layer in layers:
            hc_mult = layer.config.hc_mult
            hidden = layer.config.hidden_size
            buckets = bucket_token_counts(max_num_tokens, hc_mult * hidden, _mhc._compute_num_split_for_mhc_pre)
            norms = [(None, None)]
            ln = getattr(layer, "input_layernorm", None)
            if ln is not None:
                norms.append((ln.weight.data, getattr(ln, "variance_epsilon", layer.config.rms_norm_eps)))
            dev = layer.hc_attn_fn.device
            for t in buckets:
                x = torch.zeros((t, hc_mult * hidden), dtype=torch.bfloat16, device=dev)
                for w, eps in norms:
                    layer.hc_attn_pre(x, w, eps)
                    calls += 1
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
    rep = {"layers": len(layers), "calls": calls, "max_num_tokens": max_num_tokens,
           "seconds": round(time.perf_counter() - t0, 1)}
    logger.info("numinous mHC prewarm (Glm5Next): %s", rep)
    return rep
