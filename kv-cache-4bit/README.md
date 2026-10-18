# 4-bit KV cache

Long agent sessions run out of KV-cache memory well before they run out of compute. Past that point the scheduler starts dropping sessions and per-user speed falls apart. A 4-bit cache moves that limit out by 1.4 to 1.8x.

Stock SGLang had no 4-bit KV path for any of these models. On the Triton and FlashInfer backends, asking for `--kv-cache-dtype nvfp4` crashed or silently misread the cache. Each folder below adds the cache format, the kernels that read and write it, and the hooks that wire it into the engine.

The common format is NVFP4-style: E2M1 values with one E4M3 scale per 16 values, quantized from the bf16 latent before the engine's own FP8 path runs.

| Folder | Model / GPU | What it adds | Measured |
|---|---|---|---|
| [glm-kv4/](glm-kv4/) | GLM-5.3 on 8xB200, plus H200 and MI355X ports | 352-byte DSA latent row in place of 576-byte FP8. Triton quant-write, dequant gather, dedup compaction, sequence dequant for prefill, and a fused sparse MLA path. Reads through the served TRT-LLM op on an FP8 scratch | Pool 1.576x. At 64 sessions x 204k: FP8 2.4 tok/s per user, FP4 26.9. Costs 0 to 7% below the memory limit. KL 0.194 vs a 0.176 noise floor. Bit-exact on gfx950 |
| [kv4mla/](kv4mla/) | GLM-5.3, B200 (sm100a) | CUDA sparse MLA decode/verify that reads the 4-bit rows directly and dequantizes in shared memory. Derived from FlashMLA's SM100 head64 sparse kernel | 1.03 to 1.34x the FP8 op's time, 2.1 to 3.3x faster than the scratch path (`bench_kv4mla_b200.json`). Served: GSM8K 0.970 vs 0.975, 27.7 tok/s per user at 48 x 204k where FP8 collapses |
| [kv4tri-triton-backend/](kv4tri-triton-backend/) | Qwen3.8, Qwen3.6, Gemma 4 on RTX PRO 6000 (sm120) | Native NVFP4 decode and extend kernels for SGLang's Triton backend. E2M1/E4M3 are bit-placed into fp16 in registers, with no dequant workspace | Decode 1.4 to 1.8x faster than FP8 Triton. Extend matches the FP8 GQA overlay (16.8 vs stock 33.9 ms). Pool 1.5x. Stock crashed |
| [kv4fi-flashinfer-backend/](kv4fi-flashinfer-backend/) | Qwen on RTX PRO 6000 | NVFP4 decode for the FlashInfer backend: page-1 indices, packed verify mask, split-KV | 29.3 us per layer at batch 1 x 8k. Pool 1.5x. 17 to 22 tok/s per user at 12 sessions; serves 20 to 24 sessions where FP8 collapses at 16 |
| [kv4trt-xqa/](kv4trt-xqa/) | Qwen on RTX PRO 6000 | Routes NVFP4 KV decode through TRT-LLM XQA on sm120 and fixes the speculative-decode capture crash | XQA NVFP4 1,669 vs FlashInfer FP8 2,317 us at batch 16 x 104k. Found a stock FlashInfer FP4 prefill double-scale bug (KL 2.39 to 0.23 after the fix) |
| [kv4pro-deepseek/](kv4pro-deepseek/) | DeepSeek V4 Pro, B200/B300 | NVFP4 for both the SWA and compressed (c4) pools, with an MSE scale search. Requantizes into FlashMLA's workspace for decode | Pool 1.385x. KL 0.023. GSM8K 0.985 to 0.975 |
| [glmflash-kv4/](glmflash-kv4/) | GLM-5.3-Flash, B300 / sm120 | 288-byte rope-free 4-bit row | 1.78x tokens per byte. GSM8K equal to FP8 on B300 |
| [qwen38-kv4/](qwen38-kv4/) | Qwen3.8-27B, RTX PRO 6000 | 4-bit KV for a GDN hybrid model, plus the prefill scale fix | Pool 1.5 to 1.78x across RTX models |
| [kimi-k27-mla-kv4/](kimi-k27-mla-kv4/) | Kimi K2.7, B200 | 4-bit dense MLA KV, using kv4mla's dense mode | Built; the prefill CUDA graph needs a fix |
| [kvmix-mixed-precision/](kvmix-mixed-precision/) | GLM-5.3, B200 | Two pools, FP8 and FP4, in one slot space. Each request's precision depends on its length, and a redirect gather feeds FP4 picks into the FP8 op | Kernel max error 0.0099 vs fp32. Gate runs exist, no summarized served delta |
| [k3-quantized-state/](k3-quantized-state/) | Kimi K3, B300 / MI355X | Two-tier KDA recurrent state: FP8 (132 B) or NVFP4 (76 B) rows with an fp32 working copy | Pool 3.88x / 6.74x smaller. 36 kernel tests pass on B200. Drift 1.0 to 1.6% |

## Tests

- `test_glm_kv4.py`: CPU bit-exact torch reference, plus GPU graph-replay tests.
- `test_glm_kv4_sm90_hooks.py`: 737 lines, covering the production `_forward_trtllm` sequence in FP8 vs FP4.
- `kv4mla/test_kv4mla.py`: 56 GPU tests in both modes, including a determinism test.
- `test_kv4tri_*`: 15 CPU tests (interpreter) plus 10 GPU tests, checked against SGLang's own dequantizer.
