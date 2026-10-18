# Attention kernels

These kernels fill gaps for head shapes, GPUs and cache formats that the stock backends either handle slowly or refuse.

| Folder | Model / GPU | Mechanism | Measured |
|---|---|---|---|
| [extend-gqa/](extend-gqa/) | Gemma 4 (head-dim 512) on RTX PRO 6000 and H100 | Extend-over-paged-cache attention. One program covers G query heads x a block of tokens, so each KV tile is read once. Head-dim chunking and a value split fit hd512 into small shared memory. FP8 tensor-core dots on an FP8 cache | Full hd512 layer at 8k: 36.5 to 13.4 ms (2.73x). Sliding layers 8.7x on sm120. In the engine: +33 to 50% per user, +46 to 78% aggregate at 4 to 16 users, retrieval probes 6/6 |
| [verify-splitk/](verify-splitk/) | Gemma 4 31B, H100 | GQA-packed split-K target-verify attention. The prefix is read once per KV head, then an LSE combine | 77.0 to 14.1 ms per verify step at 32 sessions. +36 to 46% warm in the engine. GSM8K 0.958 |
| [fa4-hd512-hopper/](fa4-hd512-hopper/) | Gemma 4 31B, H100 | Lifts FlashAttention 4's sm90 head-dim gate. Runs hd512 as two 256-wide V passes with a 128x32 tile, plus FP8-prefix gather glue | 8k prefill: stock 36.5 ms, Triton GQA 12.8, FA4 6.36 (5.7x). SWA 3.34 to 0.84 ms |
| [sparse-prefill-compact-fp8/](sparse-prefill-compact-fp8/) | GLM-5.3-Flash, H200 | Compacts the live FP8 rows into bf16 with a Triton dequant and inverse slot map, no host sync, then runs FlashMLA's sparse prefill kernel instead of its per-row decode kernel | 3.18 to 2.08 ms per MLA layer at 4k rows. -13.2 / -28.3 ms per step |
| [dsa-splitk-decode/](dsa-splitk-decode/) | GLM-5.3, H200 | Splits each token's top-k into 64-row chunks so small decode batches fill 132 SMs. Base-2 LSE merge | 6 tokens: 57.0 to 22.1 us. 36 tokens: 63.1 to 53.3 us |
| [swa-window-skip/](swa-window-skip/) | Gemma 4, H100 | Skips KV blocks outside the sliding window in prefill | 1.79 to 1.9x, bit-identical |
| [glmflash-sm120/](glmflash-sm120/) | GLM-5.3-Flash, RTX PRO 6000 | DSA sparse MLA on sm120, which stock didn't support. A Triton split-K kernel for bf16 / 656-byte FP8 / 288-byte FP4 rows, plus FlashInfer's own sm120 sources rebuilt with the GLM top-k widened from 2048 to 2112 | FP4 decode at batch 64: 101 us vs FlashInfer's 104. Pool bf16 15.9M, FP8 23.3M, FP4 43.7M tokens (2.75x). 46 GPU tests |
| [hopper-fp8-kv/](hopper-fp8-kv/) | GLM-5.3-Flash, H100/H200 | FP8 KV through FlashMLA's 528-byte V32 rows, which stock SGLang refuses on sm90 | At 48 sessions, bf16 collapses (19.1 tok/s per user, TTFT p90 51 s) while FP8 keeps serving |
| [gemma-fused-glue/](gemma-fused-glue/) | Gemma 4 31B, H100 | Two fused norm/add/quant chains per layer (300 fewer launches), a fused FP8 KV store, q/k/v norm + NeoX RoPE, and a fused draft top-1 | Glue per step 4.9 to 3.1 ms in the engine. The other pieces are CPU-verified only |

All kernels carry an fp32 or torch reference and CPU tests that run under `TRITON_INTERPRET=1`.
