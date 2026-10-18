# Linear attention and hybrid models (KDA)

GLM-5.3-Flash and Kimi K3 mix Kimi Delta Attention (KDA) linear-attention layers with sparse MLA. In stock SGLang these layers were the slowest part of prefill and the source of long stalls.

| Folder | Mechanism | Measured (GLM-5.3-Flash, H200 unless noted) |
|---|---|---|
| [flashkda-split/](flashkda-split/) | Runs MoonshotAI's FlashKDA past SGLang's 64 to 2048 token cap, and for tracked batches, by splitting at the 64-aligned checkpoint. Pass A runs the heads; pass B runs the tails from the head states | 0.51 to 0.81x stock Triton `chunk_kda`. -5.7 to -28.6 ms per step over 34 layers. Relative error 5.9e-3 |
| [conv-l2norm-fused/](conv-l2norm-fused/) | One Triton pass: causal conv1d + SiLU + per-head q/k l2norm + token-major q/k/v store. Replaces conv1d, 3 copies and 2 norms | 0.462 to 0.177 ms per layer at 4k tokens. -9.7 ms per step at 4k, -18.9 at 8k. Conv state exact |
| [small-projection-fuse/](small-projection-fuse/) | Fuses KDA's small bf16 projections into one GEMM | 96.7 to 70.1 us, exact |
| [mhc-jit-prewarm/](mhc-jit-prewarm/) | Prewarms the mHC TileLang kernels for every `n_splits` bucket at load. Upstream prewarms them only for DeepSeek V4 | Before: 6 compiles x 11.5 s = 70 s of stalls in a 92.6 s window, freezing every request on the node. After: none |
| [kda-fused-decode/](kda-fused-decode/) | One-launch fused KDA decode: conv + gate + delta rule + gated RMSNorm | Built. Has a GPU numerics test against `fused_recurrent_kda` |
