# 60+ inference optimizations & kernels

Kernels, engine patches and tooling I built to serve frontier open models on six kinds of GPU. The work sits on top of [SGLang](https://github.com/sgl-project/sglang) and covers what SGLang and its kernel libraries don't: 4-bit KV caches for models that had none, sparse attention on chips the vendors never targeted, a new context-parallel mode, and 25+ upstream bugs found along the way.

Most numbers here are an A/B against stock SGLang v0.5.20 on the same node: real 8-turn coding-agent sessions, two repeats with alternating order, and an accuracy gate calibrated against stock-vs-stock noise. Kernel numbers are graph-timed microbenchmarks against the op they replace. The raw run records are in [`evidence/gate/`](evidence/gate/).

## Hardware and models

| GPU | Architecture | Models served |
|---|---|---|
| B200 | Blackwell sm100 | DeepSeek V4 Pro, GLM-5.2, GLM-5.3, Kimi K2.7-Code, Qwen3.8 |
| B300 | Blackwell sm100 | DeepSeek V4 Pro, GLM-5.3, GLM-5.3-Flash, Kimi K3 |
| RTX PRO 6000 | Blackwell sm120 | Gemma 4 26B-A4B, Qwen3.6-35B, Qwen3.8-27B, GLM-5.3-Flash, DeepSeek V4.1, DiffusionGemma |
| H100 / H200 | Hopper sm90 | Gemma 4 31B, GLM-5.3, GLM-5.3-Flash, DeepSeek V4 Pro, Qwen3.8-27B |
| MI355X | AMD CDNA4 gfx950 | DeepSeek V4 Pro, GLM-5.3 (Quark MXFP4), GLM-5.3-Flash, Kimi K3 |

## Headline results

| What | Result |
|---|---|
| DeepSeek V4 Pro, 8xB200, composed recipe vs stock SGLang | **+61 to 65% throughput**, accuracy at the noise floor |
| GLM-5.3 4-bit KV cache | **1.58x** more tokens in the KV pool; at 64 sessions x 204k context, **26.9 tok/s per user where FP8 drops to 2.4** |
| Gemma 4 GQA extend attention, RTX PRO 6000 | **2.7x** kernel, **+33 to 76%** per user in the engine |
| GLM-5.3 sparse MLA prefill, MI355X | **2.5x** over stock TileLang |
| Decode burst scheduler, Kimi K2.7 short turns | **15 to 67 tok/s** per user |

## Everything built

### 4-bit KV cache ([kv-cache-4bit/](kv-cache-4bit/))

None of these models had a 4-bit KV path in stock SGLang. On two backends, asking for one crashed.

- **GLM-5.3 4-bit KV.** A 352-byte NVFP4 row (E2M1 values, one E4M3 scale per 16) replaces the 576-byte FP8 row for DSA sparse attention. Triton kernels handle the quantized write, the dequantizing gather and dedup compaction. The pool holds 1.58x more tokens, and accuracy stays at the noise floor. [`glm_kv4.py`](kv-cache-4bit/glm-kv4/glm_kv4.py), [`glm_kv4_engine.py`](kv-cache-4bit/glm-kv4/glm_kv4_engine.py), [`glm_kv4_hooks.py`](kv-cache-4bit/glm-kv4/glm_kv4_hooks.py), [design doc](kv-cache-4bit/glm-kv4/glm_kv4.md)
- **The same 4-bit KV on Hopper and AMD.** H200 runs it through FlashMLA, MI355X through TileLang. Bit-exact on gfx950. [`glm_kv4_sm90.py`](kv-cache-4bit/glm-kv4/glm_kv4_sm90.py), [`glm_kv4_rocm.py`](kv-cache-4bit/glm-kv4/glm_kv4_rocm.py)
- **kv4mla, a native CUDA sparse MLA decode kernel for sm100.** It dequantizes the 4-bit rows in shared memory instead of building an FP8 scratch copy. 2.1 to 3.3x faster than the scratch path and near FP8 speed. GSM8K 0.970 vs 0.975. [`csrc/glm_fp4_decode_fp8.cu`](kv-cache-4bit/kv4mla/csrc/glm_fp4_decode_fp8.cu), [`kv4mla.py`](kv-cache-4bit/kv4mla/kv4mla.py), [README](kv-cache-4bit/kv4mla/README.md)
- **NVFP4 KV on SGLang's Triton backend.** Stock crashes with `KeyError: float4`. FP4 is decoded in registers with no dequant workspace. Decode is 1.4 to 1.8x faster than FP8, and the pool is 1.5x larger. [`kv4tri_nvfp4_attention.py`](kv-cache-4bit/kv4tri-triton-backend/kv4tri_nvfp4_attention.py), [`kv4tri_backend.py`](kv-cache-4bit/kv4tri-triton-backend/kv4tri_backend.py)
- **NVFP4 KV decode for the FlashInfer backend on RTX PRO 6000.** It serves 20 to 24 sessions where FP8 stalls at 16. [`kv4fi_nvfp4_decode.py`](kv-cache-4bit/kv4fi-flashinfer-backend/kv4fi_nvfp4_decode.py)
- **TRT-LLM XQA NVFP4 decode on sm120.** It beats FlashInfer FP8 (1.67 vs 2.32 ms at batch 16 x 104k). The work also turned up a double-scaling bug in stock FlashInfer FP4 prefill. [`kv4trt_hooks.py`](kv-cache-4bit/kv4trt-xqa/kv4trt_hooks.py)
- **4-bit KV for DeepSeek V4 Pro.** Covers both the SWA and the compressed pools, with an MSE-searched scale. Pool 1.385x, KL 0.023. [`kv4pro_fp4.py`](kv-cache-4bit/kv4pro-deepseek/kv4pro_fp4.py)
- **4-bit KV for GLM-5.3-Flash.** A 288-byte rope-free row gives 1.78x tokens per byte, and GSM8K matches FP8 on B300. [`glmflash_kv4.py`](kv-cache-4bit/glmflash-kv4/glmflash_kv4.py)
- **4-bit KV for Qwen3.8 on RTX PRO 6000.** [`q38kv4_hooks.py`](kv-cache-4bit/qwen38-kv4/q38kv4_hooks.py)
- **4-bit KV for dense MLA (Kimi K2.7).** [`k27_kv4_engine.py`](kv-cache-4bit/kimi-k27-mla-kv4/k27_kv4_engine.py)
- **kvmix, FP8 and FP4 KV in one slot space.** Each request's precision is picked by sequence length. [`kvmix_core.py`](kv-cache-4bit/kvmix-mixed-precision/kvmix_core.py), [design doc](kv-cache-4bit/kvmix-mixed-precision/kvmix_glm.md)
- **Two-tier quantized recurrent state for Kimi K3's KDA layers.** The FP8 or NVFP4 pool shrinks 3.9x or 6.7x, with an fp32 working copy. [`k3state_quant.py`](kv-cache-4bit/k3-quantized-state/k3state_quant.py)

### Attention kernels ([attention/](attention/))

- **GQA-packed extend attention for head-dim 512,** on H100 and RTX PRO 6000. One program reads each KV tile once for all its query heads. 2.2 to 2.9x faster, 8.7x on sliding-window layers, +33 to 76% per user in the engine. [`extend_attention_gqa.py`](attention/extend-gqa/extend_attention_gqa.py), [`numinous_extend_gqa.py`](attention/extend-gqa/numinous_extend_gqa.py)
- **Split-K speculative verify attention for Gemma 4 31B.** 77 to 14 ms per verify step at 32 sessions, +36 to 46% in the engine. [`gemma31_verify.py`](attention/verify-splitk/gemma31_verify.py)
- **FlashAttention 4 at head-dim 512 on Hopper.** I lifted the head-dim gate and run it as two 256-wide V passes. 5.7x over stock prefill. [`g31_fa4_hd512_probe.py`](attention/fa4-hd512-hopper/g31_fa4_hd512_probe.py)
- **Sparse prefill on a compacted FP8 pool.** FlashMLA's prefill kernel runs on a compact bf16 copy of only the live rows, with no host sync. 0.65x layer time, -13 to -28 ms per step on H200. [`flash_sparse_prefill.py`](attention/sparse-prefill-compact-fp8/flash_sparse_prefill.py)
- **Split-K sparse decode.** Fills all 132 SMs at small token counts. 57 to 22 us per call. [`dsa_sparse_splitk.py`](attention/dsa-splitk-decode/dsa_sparse_splitk.py)
- **Sliding-window skip in Gemma prefill.** 1.8x, bit-identical. [`gemma31_swa_window_skip_hooks.py`](attention/swa-window-skip/gemma31_swa_window_skip_hooks.py)
- **GLM-5.3-Flash sparse attention on RTX PRO 6000 (sm120),** which stock didn't support. A Triton split-K kernel plus FlashInfer's sm120 sources rebuilt with a wider top-k table. The pool grows 2.75x in FP4. [`glmflash_sm120_attn.py`](attention/glmflash-sm120/glmflash_sm120_attn.py), [`glmflash_sm120_fi.py`](attention/glmflash-sm120/glmflash_sm120_fi.py)
- **FP8 KV for GLM-5.3-Flash on Hopper,** which stock SGLang refuses. It serves 48 sessions where bf16 collapses to 19 tok/s per user. [`hopper_fp8kv_hooks.py`](attention/hopper-fp8-kv/hopper_fp8kv_hooks.py)
- **Fused norm, add, quant, RoPE and KV-store chains for Gemma.** Per-step glue drops from 4.9 to 3.1 ms. [`rtx_glue.py`](attention/gemma-fused-glue/rtx_glue.py)

### AMD MI355X ([amd-mi355x/](amd-mi355x/))

- **Sparse MLA prefill in Triton for gfx950,** using FP8 MFMA. 2.5x over stock TileLang (1.77 vs 4.43 ms), shipped in the GLM recipe. [`rocm_sparse_mla_prefill.py`](amd-mi355x/sparse-mla-prefill-triton/rocm_sparse_mla_prefill.py)
- **Hand-written HIP sparse MLA decode.** It reuses the K tile in LDS as the PV operand through `ds_read_b64_tr_b8`. 1.1 to 1.8x over the Triton kernel, zero faults in the gate. [`rocm_sparse_mla_decode_hip.py`](amd-mi355x/sparse-mla-decode-hip/rocm_sparse_mla_decode_hip.py)
- **Split-K DSA decode for GLM-5.3-Flash.** 2.1x over stock. [`rocm_flash_dsa_decode.py`](amd-mi355x/flash-dsa-decode/rocm_flash_dsa_decode.py)
- **AITER fused-MoE dispatch fix.** 657 to about 350 ms per 65k-token forward. [`numinous_aiter_moe_dispatch.py`](amd-mi355x/aiter-moe-dispatch-fix/numinous_aiter_moe_dispatch.py)
- **Per-token-per-channel FP8 for block-FP8 checkpoints.** 1.46 to 1.92x on GEMMs. [`numinous_fp8_ptpc.py`](amd-mi355x/fp8-ptpc/numinous_fp8_ptpc.py)
- **Quark checkpoint loading fix.** Attention output was exactly zero before it. Also a hash-routed scaling fix and a deterministic router GEMM. [`rocm_quark_fused_hooks.py`](amd-mi355x/quark-checkpoint-loading/rocm_quark_fused_hooks.py), [write-up](amd-mi355x/quark-checkpoint-loading/sglang-rocm-dsv4-hash-routed-scaling.md)
- **Idle-rank and capture fixes under DP + EAGLE.** These cured garbage output and memory faults. [`rocm_dp_idle_hooks.py`](amd-mi355x/dp-idle-rank-fixes/rocm_dp_idle_hooks.py), [`rocm_dsa_pad_hooks.py`](amd-mi355x/dp-idle-rank-fixes/rocm_dsa_pad_hooks.py)
- **Three ROCm stand-ins for missing pieces.** A symmetric-memory workaround for a PyTorch ROCm bug, CU-masked HIP streams in place of CUDA green contexts, and exact one-hot rejection sampling for EAGLE on HIP. [`rocm_symm_mem.py`](amd-mi355x/symm-mem-cu-streams-onehot/rocm_symm_mem.py), [`rocm_cu_streams.py`](amd-mi355x/symm-mem-cu-streams-onehot/rocm_cu_streams.py), [`rocm_onehot_verify.py`](amd-mi355x/symm-mem-cu-streams-onehot/rocm_onehot_verify.py)
- **One-pass KDA q/k/v split + l2norm.** 1.52x on the op. [`numinous_kda_prep.py`](amd-mi355x/kda-prep/numinous_kda_prep.py)

### MoE and GEMM ([moe-gemm/](moe-gemm/))

- **MegaMoE W4A4 with per-rank tile balancing on DeepSeek V4 Pro.** +61 to 65% throughput over stock on 8xB200, +30 to 45% on 8xB300. [`deepgemm_megamoe_hooks.py`](moe-gemm/megamoe-tile-balance/deepgemm_megamoe_hooks.py)
- **Measured MoE tactic tables.** Each token count gets the fastest FlashInfer TRT-LLM MoE tactic, where the autotuner only buckets to powers of two. 497 to 374 us per MoE call. [`numinous_moe_tactics.py`](moe-gemm/moe-tactic-tables/numinous_moe_tactics.py), [tables](moe-gemm/moe-tactic-tables/moe_tactics/)
- **Fused DP-MoE communication.** NVFP4 rows and top-k ids are pushed into peers' symmetric memory with multimem PTX. Decode 356 to 260 us, prefill 4.8 to 3.1 ms per layer, bit-exact against NCCL. [`numinous_dp_moe_nvfp4.py`](moe-gemm/dp-moe-symmetric-memory/numinous_dp_moe_nvfp4.py), [`symm_collectives.py`](moe-gemm/dp-moe-symmetric-memory/symm_collectives.py)
- **Fused RMSNorm + FP8 quant into DeepGEMM's UE8M0 scale layout.** 1.6 to 2.0x, bit-exact. [`pro_norm_quant.py`](moe-gemm/norm-quant-fused/pro_norm_quant.py)
- **FP8 LM head through DeepGEMM.** 1.9 to 2.3x on a 1.85 GB read per step. [`pro_fp8_lm_head.py`](moe-gemm/fp8-lm-head/pro_fp8_lm_head.py)
- **FP8 passthrough.** The DP gather's FP8 tensors are reused as the MoE input quant, with a proof that the requant is exact. 99 to 5.8 us per layer. [`dp_fp8_passthrough.py`](moe-gemm/fp8-passthrough/dp_fp8_passthrough.py)
- **Router piggyback.** Top-k ids ride inside the FP8 gather's scale tensor. -5.5 ms per prefill pass with no extra collective. [`dp_router_piggyback.py`](moe-gemm/router-piggyback/dp_router_piggyback.py)
- **SM90 W4A8 grouped GEMM** with TMA, WGMMA and an int4 to e4m3 lookup through PRMT. 1.5x over the stock CUTLASS kernel. [`w4a8dg_kernel.cuh`](moe-gemm/w4a8-sm90-grouped-gemm/w4a8dg/w4a8dg_kernel.cuh), [`w4a8_tiles/`](moe-gemm/w4a8-sm90-grouped-gemm/w4a8_tiles/)
- **Marlin to CUTLASS W4A8 layout inverse,** so MoE can pick its kernel by batch size. -31 to -35% per decode call. [`glm_marlin_inverse.py`](moe-gemm/marlin-layout-inverse/glm_marlin_inverse.py)
- **Shared expert folded into the MoE post-reorder.** -6.8 ms per prefill forward. [`moe_post_reorder_shared.py`](moe-gemm/shared-expert-post-reorder/moe_post_reorder_shared.py)
- **CuTe-DSL grouped NVFP4 GEMM driven from PyTorch.** 9.4x over Triton. Per-call layout conversion dropped from 68 ms to 0.009 ms. [`grouped_gemm.py`](moe-gemm/cutedsl-grouped-nvfp4/grouped_gemm.py)
- **INT4 to NVFP4/MXFP4 expert converter** with an MSE scale search, exact when the codes fit. [`k27_fp4_convert.py`](moe-gemm/int4-to-nvfp4-converter/k27_fp4_convert.py)
- **Grouped W4A4 MoE decode GEMM,** measured against the byte roof. 3.9x over `torch._grouped_mm`. [`moe_decode_gemm.py`](moe-gemm/w4a4-moe-decode-gemm/moe_decode_gemm.py), [write-up](moe-gemm/w4a4-moe-decode-gemm/qwen38-moe-kernel.md)

### Linear attention and hybrid models ([linear-attention-kda/](linear-attention-kda/))

- **FlashKDA past SGLang's 64 to 2048 token cap.** Long prefills are split at checkpoint boundaries. 0.51 to 0.81x time, up to -28.6 ms per step. [`flash_kda_prefill.py`](linear-attention-kda/flashkda-split/flash_kda_prefill.py)
- **One-pass causal conv + SiLU + l2norm.** 0.46 to 0.18 ms per layer, conv state exact. [`flash_kda_conv_l2norm.py`](linear-attention-kda/conv-l2norm-fused/flash_kda_conv_l2norm.py)
- **Fused KDA small projections.** 96.7 to 70.1 us, exact. [`kda_smallproj_fuse_hooks.py`](linear-attention-kda/small-projection-fuse/kda_smallproj_fuse_hooks.py)
- **mHC TileLang kernels prewarmed at load.** This removed 11.6 s global stalls during serving. [`numinous_mhc_jit.py`](linear-attention-kda/mhc-jit-prewarm/numinous_mhc_jit.py)
- **One-launch fused KDA decode** (conv, gate, delta rule, gated RMSNorm). [`kda.py`](linear-attention-kda/kda-fused-decode/kda.py)

### Parallelism and disaggregation ([parallelism/](parallelism/))

- **Decode context parallel for DSA sparse attention,** a new mode in SGLang. KV is sharded by an owner rule. Each rank attends over its own selected slots and returns (o, lse) to a cross-rank merge, across decode, extend and EAGLE verify. [17 patches](parallelism/dsa-decode-context-parallel/), [`dcp_topk.py`](parallelism/dsa-decode-context-parallel/dcp_topk.py)
- **Sparse FP8 prefill Triton kernel** for the DCP case. 6 ms per 4k-token layer at top-k 2048. [`sparse_prefill_fp8.py`](parallelism/dsa-decode-context-parallel/sparse_prefill_fp8.py)
- **In-node prefill/decode disaggregation without RDMA NICs.** SM-partitioned lanes, CUDA IPC and MORI XGMI transport, and a decode-side radix cache for hybrid models. [`numinous_pdmux_dp.py`](parallelism/in-node-pd-disaggregation/numinous_pdmux_dp.py), [`pd_nvlink_copy.py`](parallelism/in-node-pd-disaggregation/pd_nvlink_copy.py)
- **INT8/FP8 two-shot all-reduce for PCIe boxes without NVLink.** [`fp8_pcie_allreduce.py`](parallelism/fp8-pcie-allreduce/fp8_pcie_allreduce.py)

### Scheduler ([scheduler/](scheduler/))

- **Decode burst.** Upstream defines `--num-continuous-decode-steps` but never reads it; this makes it work, for TP and DP attention, with a queue cap. Kimi K2.7 short turns went from 15 to 67 tok/s per user, and GLM decode-heavy traffic gained 55%. [5 patches](scheduler/decode-burst/), [`decode_burst.py`](scheduler/decode-burst/decode_burst.py)
- **DP prefill balancing, per-rank skew budgets and prefill pacing.** Measured; some kept, some dropped. [`numinous_dp_pace.py`](scheduler/dp-balance-and-pacing/numinous_dp_pace.py), [`pro_skew_budget_hooks.py`](scheduler/dp-balance-and-pacing/pro_skew_budget_hooks.py)

### Speculative decoding ([speculative/](speculative/))

- **MLA draft models for Kimi K3,** exact to 1e-9 against the reference. [`k3_mla_draft_hooks.py`](speculative/k3-mla-draft/k3_mla_draft_hooks.py)
- **Load-adaptive DSpark verify rows,** which upstream's policy rejects under DP. [`numinous_spec_policy.py`](speculative/adaptive-spec-policy/numinous_spec_policy.py)
- **DSpark and EAGLE3 working on Gemma,** after fixing upstream draft-model bugs. [`gemma_dspark_hooks.py`](speculative/gemma-dspark-eagle3/gemma_dspark_hooks.py), [`gemma_eagle3_hooks.py`](speculative/gemma-dspark-eagle3/gemma_eagle3_hooks.py)
- **Frozen-KV MTP draft with no device-to-host sync,** plus a fused top-1. [`frozen_mtp_nosync_hooks.py`](speculative/frozen-mtp-draft/frozen_mtp_nosync_hooks.py)

### Upstream bugs found ([upstream-fixes/](upstream-fixes/))

25+ bugs found in SGLang, FlashInfer, AITER, DeepGEMM and PyTorch ROCm. Nine are ported as diffs against upstream main. The full list, with the symptom each one caused, is in [upstream-fixes/README.md](upstream-fixes/README.md).

### Tooling ([tooling/](tooling/))

- **Hot-patch layer.** Declarative edits anchored to exact lines of the installed engine, with an idempotency marker, compile-before-write, and `--check` as a boot gate. About 140 patches use it. [`sglang_numinous_hooks.py`](tooling/hotpatch/sglang_numinous_hooks.py), [`git_series_hooks.py`](tooling/hotpatch/git_series_hooks.py)
- **A/B serving gate.** Replays coding-agent sessions, pins DP ranks, alternates arm order and takes repeats. [`cell_gate.py`](tooling/serving-gate/cell_gate.py), [`dc_bench.py`](tooling/serving-gate/dc_bench.py)
- **Noise-calibrated accuracy gate.** Teacher-forced KL, top-1 agreement and GSM8K with McNemar's test, judged against a measured stock-vs-stock floor. [`accuracy_gate.py`](tooling/accuracy-gate/accuracy_gate.py)
- **Per-model rooflines** for DeepSeek V4 Pro, GLM-5.3, GLM-5.3-Flash, Kimi K3 and Gemma 4. [`tooling/rooflines/`](tooling/rooflines/)
- **Diagnostics.** A scheduler stall tracer, a sync-free illegal-memory-access locator, and a byte model. [`tooling/diagnostics/`](tooling/diagnostics/)

### Serving recipes ([recipes/](recipes/))

Measured launch configs for 7 model and GPU combinations, plus the rules behind them. [recipes/README.md](recipes/README.md)

### What didn't work ([negative-results/](negative-results/))

Seven attempts that came out slower than what they meant to replace, kept with their numbers. [negative-results/README.md](negative-results/README.md)

## Reading the code

- This code is extracted from a larger private repo, grouped by topic here. Paths inside docstrings (`operations/kernels/...`, `operations/patches/...`) refer to that layout. Most docstrings also keep dated lab notes from the original runs; I left them in because they explain why each choice was made.
- Each kernel module carries a torch or fp32 reference. Most have CPU tests that run under `TRITON_INTERPRET=1` and GPU tests that skip without CUDA.
- `*_hooks.py` files wire a kernel into SGLang v0.5.20 (`lmsysorg/sglang@sha256:06e4f2ed...`) through the [hot-patch layer](tooling/README.md). ROCm hooks target the `lmsysorg/sglang-rocm` 20260923 and 20261002 images. Features are off unless their `NUMINOUS_*` env var is set.
- Fork patches (`*.patch`) apply to SGLang v0.5.18.
- [docs/methodology.md](docs/methodology.md) explains how results were measured.

## License

Apache-2.0, see [LICENSE](LICENSE). Some files derive from third-party code; [NOTICE](NOTICE) lists them, and their licenses are in [LICENSES/](LICENSES/).
