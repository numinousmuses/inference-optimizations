# Upstream-ready diffs (not pushed, not committed upstream)

These are catalogue items from `research/upstream-pr-candidates.md`, ported onto upstream main. Bases:
SGLang origin/main 069523f999 (2026-09-28); FlashInfer main 795a96b229 (2026-09-29). Each diff was checked with
`patch -p1 --dry-run` against a `git archive` of that base (all 9 apply). The Python files pass `py_compile`. None
has run on a GPU against main yet. The GPU evidence for each fix comes from the equivalent v0.5.20 hook (see the
catalogue).

| diff | catalogue | what | checked on CPU | still needed before the PR |
|---|---|---|---|---|
| A1-sglang-nvfp4-workspace-kv-scale.diff | A1 | `k_scale/v_scale=None` when FlashInfer reads the FP4 dequant workspace (2 prefill sites + workspace decode) | applies, compiles | port `operations/patches/test_q38kv4_fp4_prefill_scale.py` GPU chain as an upstream unit test; one RTX run |
| A2-sglang-unquantized-kv-ignore-ckpt-scales.diff | A2 | `reset_kv_scales_for_unquantized_cache()` in `quantization/kv_cache.py`, called from `ModelRunner.configure_kv_cache_dtype` (after `load_model` on main: L654 vs L674) | applies, compiles; helper tested on a stub model (resets 0.03 -> 1.0, leaves 1.0 layers, idempotent) | unit test in the upstream suite; one bf16-KV boot of unsloth/Qwen3.8-27B-NVFP4 |
| A4-sglang-cutedsl-gdn-exclude-sm120.diff | A4 | `_is_blackwell()` -> `major in (10, 11)` | applies, compiles | none beyond CI (SM120 already falls back via `supports_prefill`) |
| A5-sglang-triton-decode-swa-window.diff | A5 | `update_sliding_window_buffer` reads `window_left + 1` keys; CUDA-graph window index buffer grown to `max_num_tokens * (W + 1)` | applies, compiles | extend/verify callers stay exact: the extend kernel masks `q <= kv + W`, so the extra prefix key is masked (source read). Needs a decode-vs-extend GPU test on a Gemma SWA layer |
| A10-flashinfer-nvfp4-kv-64bit-offsets.diff | A10 | 64-bit row offsets in `nvfp4_kv_quantize` and the linear `nvfp4_kv_dequant` kernel (same int*int overflow; the paged dequant kernel already uses int64) | applies | GPU test: one call with M*K > 2^31 against two half calls |
| A3-sglang-gemma4-trtllm-moe-routing.diff | A3 | Gemma4MoE builds its TopK with `output_format=STANDARD` under flashinfer_trtllm, so its own routing runs (model-level; a generic TopK change would also force STANDARD on cohere2_moe, which pairs its custom routing with a matching in-kernel `routing_method_type` on purpose) | applies, compiles | the check reads the global runner backend at construction, the same global `TopK.forward_cuda` reads; confirm it is final by then on main. B200 layer test from `operations/modal/gemma_trtmoe_probe.py`. afmoe/grok/interns1pro/llama4/phimoe need the same review (issue) |
| A7-sglang-dsa-index-key-cache-move.diff | A7 | `IndexKeyCache.move` moves each token's key and scale bytes inside its page (standard layout), or token rows directly for the AITER preshuffle layout (token-interleaved) | applies, compiles; CPU layout test: moved tokens match a per-token reference, other tokens untouched; stock indexing raises IndexError on the same data | GPU: tree-EAGLE (topk > 1) on a DSA model, or a pool-level unit test |
| A13-sglang-eagle3-norm-before-residual.diff | A13 | EAGLE3 input layer honours `norm_before_residual` (unset: identical code path) | applies, compiles | unit test on a tiny config |
| A15-docs-dsv4-drop-deprecated-env.diff | A15 | drop `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` from 3 DeepSeek-V4 cookbook commands (environ.py: "the unified radix tree is the default tree cache now; unset this env") | applies | none |

Not ported: A6 (SM120 NVFP4 + speculative routing), A8 (Quark loader, needs the current ROCm image), A9 (DSV4 trtllm + DSpark),
A11 (validation), A12 (Gemma DSpark: the vocab-size fallback must be TP-safe; low value, no viable Gemma DSpark draft),
A14 (needs its isolation boot).
