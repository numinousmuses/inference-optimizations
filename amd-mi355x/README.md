# AMD MI355X

This folder ports DeepSeek V4 Pro, GLM-5.3 (AMD Quark MXFP4 checkpoints), GLM-5.3-Flash and Kimi K3 to 4 and 8x MI355X (gfx950, ROCm 7.2). It includes the kernels the stock ROCm image lacked and the fixes that made it produce correct output at all.

Result: DeepSeek V4 Pro on 8x MI355X went from the stock ROCm recipe (1,377 warm tok/s at 64 sessions, 24.7 per user, accuracy failing) to 2,849 warm tok/s at 80 sessions and 55.6 per user, with TTFT p90 of 5.6 s.

| Folder | Mechanism | Measured |
|---|---|---|
| [sparse-mla-prefill-triton/](sparse-mla-prefill-triton/) | Prefill-shaped FP8 sparse MLA for gfx950. The Q tile stays resident, FP8 MFMA does the math, and a P*448 trick keeps the probabilities in FP8 range. Also a split-K decode variant | Prefill 1.77 vs 4.43 ms for stock TileLang at 4096 tokens x 64 heads (2.5x). Split-K decode 2.1x. Shipped in the GLM recipe |
| [sparse-mla-decode-hip/](sparse-mla-decode-hip/) | Hand-written HIP. A 256-thread workgroup per (row, split); the K tile in LDS is reused as the PV B-operand through `ds_read_b64_tr_b8` | 0.55 to 0.91x the Triton kernel's time. Gate: 0 faults, GSM8K 0.965 |
| [flash-dsa-decode/](flash-dsa-decode/) | Split-K rope-free decode/verify for GLM-5.3-Flash | 2.1x over stock TileLang |
| [aiter-moe-dispatch-fix/](aiter-moe-dispatch-fix/) | Fixes AITER's tuned-config lookup for `inter_dim % 256` shapes, which was falling back to a slow path | 657 to about 350 ms per 65k-token forward |
| [fp8-ptpc/](fp8-ptpc/) | Per-token-per-channel FP8 for block-FP8 checkpoints | 1.46 to 1.92x on GEMMs at 16k rows, about 8% of prefill |
| [quark-checkpoint-loading/](quark-checkpoint-loading/) | Maps Quark `.weight_scale` to SGLang's `.weight_scale_inv`. Fixes hash-routed expert scaling and adds a deterministic router GEMM | Before the fix the fused q_a/kv_a scale was the float32-min sentinel and attention output was exactly 0 in every layer. After it, greedy decode matches HF. [Write-up](quark-checkpoint-loading/sglang-rocm-dsv4-hash-routed-scaling.md) |
| [dp-idle-rank-fixes/](dp-idle-rank-fixes/) | Fixes for idle DP ranks with zero tokens on hybrid models, the eager DSA page-table width, and greedy-vs-sampled draft graph buffer mismatches | Cured the DP4 + EAGLE garbage output (traced to AITER custom all-reduce capture) and the memory access faults |
| [symm-mem-cu-streams-onehot/](symm-mem-cu-streams-onehot/) | A legacy HIP IPC buffer, since PyTorch's ROCm symmetric memory passes a pointer where HIP wants an fd. CU-masked HIP streams in place of CUDA green contexts. Exact one-hot chain rejection sampling for greedy EAGLE, which avoids a full-vocab softmax | Infrastructure. The one-hot sampler targets 19 ms of a 153 ms step |
| [kda-prep/](kda-prep/) | One-pass KDA q/k/v split + l2norm | 1.52x on the op |
