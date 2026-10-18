# What didn't work

These kernels were built, tested for correctness and measured, and they lost to the thing they meant to replace. I kept them because the numbers are useful to anyone about to try the same idea. Code is in [code/](code/).

| Attempt | Idea | Result |
|---|---|---|
| [idx_epilogue/](code/idx_epilogue/) | Patch DeepGEMM's sm100 MQA-logits epilogue to do an exact fused top-k (threshold + bitmap) for GLM's DSA indexer, plus a TMEM hoist | Fused top-k 10,532 us vs stock 5,845 at 524k tokens (1.8x slower). Hoist -2 to -11% on the bf16 path only |
| [glm_fp4idx.py](code/glm_fp4idx.py) | FP4 indexer keys (132 to 68 bytes per token) scored by DeepGEMM's FP4 MQA logits | Speed about equal to FP8. Top-2048 overlap with real keys only 0.71 at 524k. Not worth the accuracy loss |
| [moe_decode_w4/](code/moe_decode_w4/) | W4A16 decode-regime grouped MoE for sm90, with register prefetch and a TMA bulk-ring variant | Slower than Marlin at every batch size: 524 to 632 vs 380 to 614 us |
| [rocm_fp8_splitk_gemm.py](code/rocm_fp8_splitk_gemm.py) | Triton split-K FP8 block-scale GEMM reading AMD CK's preshuffled layout | Slower than CK at every shape. The useful by-product: retuning found 106 AITER config rows beaten by more than 10% |
| [pro_mhc_fused.py](code/pro_mhc_fused.py) | mHC post + next pre + Sinkhorn + RMSNorm fused into one Triton kernel | 2.2 to 3.5x slower than stock TileLang |
| [skinny_fp8_gemm.py](code/skinny_fp8_gemm.py) | Split-K swap-AB FP8 block GEMM for M around 96 | 1.3x at 8 rows, 0.2 to 0.7x from 48 rows up |
| [megakernel.py](code/megakernel.py) | Persistent single-launch decode MLP with an atomic grid barrier | 0.19x cuBLAS on B200. 1.47x only on L4 |

## Other measured negatives

- Two-batch overlap on MI355X: -5%, because RCCL holds 112 of 256 CUs.
- DP extend cap on DeepSeek V4 Pro: -12% at 64 sessions, -7% at 128.
- Fused DP-MoE EP8 on DeepSeek V4 Pro: -11 to -14% against MegaMoE.
- FlashMLA sparse prefill on B200 for GLM: 0.63x the speed of the TRT-LLM op.
- Canvas attention for DiffusionGemma: 2.6 to 2.9x faster as a kernel, but slower in the engine (161 vs 175 tok/s).
- FP8 LM head on GLM-5.3 (H200): 1.67 to 1.9x faster, but top-1 agreement 0.89 to 0.92 against a 0.995 gate.
