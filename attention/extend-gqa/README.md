# sglang-v0.5.18-extend-gqa

Module overlay for SGLang v0.5.18 (lmsysorg/sglang:v0.5.18-cu129): large-head extend (prefill-over-cache) attention
on SM120 through a GQA-packed kernel with fp8 tensor-core dots on an fp8 cache.

- `numinous_extend_gqa.py` -> `sglang/kernels/ops/attention/numinous_extend_gqa.py` (the kernel)
- `extend_attention.py` -> `sglang/kernels/ops/attention/extend_attention.py` (v0.5.18's file plus one dispatch hook:
  SM120, head_dim >= 256, plain causal path; anything else stays on the stock kernel; a Python-level failure falls
  back to stock; `SGLANG_NUMINOUS_EXTEND_GQA=0` disables)

Measured on an RTX PRO 6000 (numinous-inference research/gpu-evidence/dc_arch/rtxattn_run_*): Gemma 4 global layers
2.72-2.79x the stock kernel, sliding layers 8.7x. In the engine on real agent traffic (Gemma 4 26B-A4B shipped
recipe, 107-113k-token sessions, 3 repeats, research/gpu-evidence/gate/gemma-4-26b-a4b_standard_*): per-user
+33-50%, aggregate +46-78% at 4-16 users, retrieval probes 6/6.
