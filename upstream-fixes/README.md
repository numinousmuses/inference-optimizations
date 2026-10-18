# Upstream bugs found

These bugs turned up while bringing the models up and measuring them. Each was fixed locally as a hook and validated on GPU in the served gate. Nine are ported as clean diffs against upstream main in [diffs/](diffs/) (see [diffs/README.md](diffs/README.md) for each one's base and test status). The hooks for the rest are in [hooks/](hooks/) or next to the feature they belong to.

## Accuracy bugs (silent wrong output)

| Project | Bug | Symptom | Fix |
|---|---|---|---|
| SGLang | NVFP4 KV prefill re-applies the checkpoint FP8 KV scale over the dequant workspace | GSM8K 0.88 to 0.785 | [A1 diff](diffs/A1-sglang-nvfp4-workspace-kv-scale.diff) |
| SGLang | bf16 KV cache still multiplied by the checkpoint's FP8 KV scales | GSM8K 0.005 | [A2 diff](diffs/A2-sglang-unquantized-kv-ignore-ckpt-scales.diff) |
| FlashInfer | FP4 prefill scales K/V twice | KL 2.39, 0.23 after the fix | [kv4trt hook](../kv-cache-4bit/kv4trt-xqa/kv4trt_hooks.py) |
| SGLang | Quark `.weight_scale` not mapped to `.weight_scale_inv` | FP8 attention runs unscaled; attention output exactly 0 | [hook](../amd-mi355x/quark-checkpoint-loading/rocm_quark_fused_hooks.py) |
| SGLang | `custom_routing_function` bypassed under `flashinfer_trtllm` MoE (Gemma 4 and 6 other models) | Wrong expert routing | [A3 diff](diffs/A3-sglang-gemma4-trtllm-moe-routing.diff), [hook](hooks/gemma_trtmoe_hooks.py) |
| SGLang | Shared-expert TP1 path on v0.5.20 | Garbage from the first token | [hook](hooks/glm_tp1_shared_hooks.py) (fixed upstream in #41195) |
| SGLang | Speculative MoE backend defaults to the main backend, so the bf16 NextN draft rounds its fp32 correction bias to bf16 (174 values collapse to 3) | EAGLE acceptance collapses to about 1.0 | flag `--speculative-moe-runner-backend auto` |
| SGLang | Triton decode reads one sliding-window key fewer than extend and FlashInfer | Decode/extend mismatch on SWA layers | [A5 diff](diffs/A5-sglang-triton-decode-swa-window.diff) |
| SGLang | EAGLE3 draft ignores `norm_before_residual` | Low acceptance | [A13 diff](diffs/A13-sglang-eagle3-norm-before-residual.diff) |
| SGLang (ROCm) | Hash-routed expert scaling on DeepSeek V4 | Wrong expert scale | [write-up](../amd-mi355x/quark-checkpoint-loading/sglang-rocm-dsv4-hash-routed-scaling.md) |
| SGLang (ROCm) | Greedy vs sampled draft graph variants register different buffers, so custom all-reduce reads the wrong peer | Garbage under DP + EAGLE | [hook](../amd-mi355x/dp-idle-rank-fixes/rocm_spec_variant_hooks.py) |

## Crashes, OOMs, stalls

| Project | Bug | Symptom | Fix |
|---|---|---|---|
| SGLang | DSA kpool indexer materializes dense fp32 logits for the whole prefill chunk | 3.37 GiB OOM on 4xH200 | [hook](hooks/kpool_logits_chunk_hooks.py) |
| SGLang | DeepSeek V4 pool has no `get_cpu_copy`; decode retraction raises inside the scheduler | Kills the node instead of one request | [hook](../parallelism/in-node-pd-disaggregation/pd_retract_abort_hooks.py) |
| SGLang | SM120 NVFP4 KV + speculative decoding (issue #36010) | Crash in target-verify graph capture | [hook](../kv-cache-4bit/kv4trt-xqa/kv4trt_hooks.py) |
| SGLang | nvfp4 accepted for the Triton backend and hybrid SWA pools that can't read it | `KeyError` at runtime | [kv4tri hook](../kv-cache-4bit/kv4tri-triton-backend/kv4tri_hooks.py) |
| FlashInfer | `nvfp4_kv_quantize` uses 32-bit row offsets | Illegal memory access above 2^31 elements | [A10 diff](diffs/A10-flashinfer-nvfp4-kv-64bit-offsets.diff) |
| SGLang | `IndexKeyCache.move` indexes a page-major buffer by token slot | IndexError / corruption under tree EAGLE | [A7 diff](diffs/A7-sglang-dsa-index-key-cache-move.diff) |
| SGLang | CuTe DSL GDN prefill claims sm120 support | Fails to compile | [A4 diff](diffs/A4-sglang-cutedsl-gdn-exclude-sm120.diff) |
| SGLang | DeepSeek V4 TRT-LLM sparse attention + DSpark trips a 128-wide SWA table assert | Boot failure | [hook](hooks/trtllm_attn_hooks.py) |
| SGLang | mHC TileLang JIT compiles per new `n_splits` bucket mid-serve | 11.6 s global stalls | [hook](../linear-attention-kda/mhc-jit-prewarm/rtx_mhc_jit_hooks.py) |
| SGLang | 3 sm120 blockers for DeepSeek V4.1; `SGLANG_OPT_SM120_DIRECT_SWA_KV` passes the logical page size to byte-addressed writers | Won't run on RTX PRO 6000 | [hook](hooks/rtx_v41_sm120_hooks.py) |
| SGLang (ROCm) | Hybrid-SSM idle DP rank with 0 tokens rewritten to a 0-row extend | Every M=0 op crashes | [hook](../amd-mi355x/dp-idle-rank-fixes/rocm_dp_empty_idle_hooks.py) |
| SGLang (ROCm) | Eager DSA page-table width taken from host `seq_lens` | Memory access fault under DP + EAGLE | [hook](../amd-mi355x/dp-idle-rank-fixes/rocm_dsa_pad_hooks.py) |
| SGLang | SM90 Humming W4A8 weight prep | OOM at boot on H200 | [hook](hooks/humming_chunk_hooks.py) |

## Performance bugs

| Project | Bug | Fix |
|---|---|---|
| SGLang | `fused_marlin_moe` picks its M-block from the local expert count under EP, so it always gets the smallest | [hook](hooks/marlin_ep_block_hooks.py) |
| AITER | Tuned-config lookup misses `inter_dim % 256` shapes; 657 vs about 350 ms per 65k forward | [fix](../amd-mi355x/aiter-moe-dispatch-fix/numinous_aiter_moe_dispatch.py) |
| SGLang | `concat_mla_absorb_q` takes the slow path with an empty rope; skipping it saves 48 us per layer at 2k tokens, bit-identical | [hook](hooks/mla_q_cat_skip_hooks.py) |
| SGLang | `--num-continuous-decode-steps` defined but never read | [decode burst](../scheduler/decode-burst/) |
| DeepGEMM | Host asserts refuse SiTU with MXFP4 activations, which the device template supports | noted |
| PyTorch | ROCm symmetric-memory rendezvous passes a pointer where HIP wants an fd | [workaround](../amd-mi355x/symm-mem-cu-streams-onehot/rocm_symm_mem.py) |
| AITER | `gemm_afp4wfp4` non-EVEN_K path returns NaN on gfx950 | noted |
