# SGLang ROCm: DeepSeek-V4 hash-routed MoE layers run without routed_scaling_factor on the AITER path

Found by amdpro, 2026-10-04 (claims amdpro.pro-mi355x.hash-layer-scaling-bug, amdpro.pro-mi355x.hash-fix-verified).
Draft for an upstream issue / PR.

## Affected

lmsysorg/sglang-rocm v0.5.21-rocm724-mi35x-20261002 (SGLang 1330e437e1), python/sglang/srt/models/deepseek_v2.py,
DeepseekV2MoE.__init__, any DeepSeek-V4 checkpoint with num_hash_layers > 0 (V4 Pro 0813: 3) served with
SGLANG_USE_AITER=1 and no fused shared expert in the hash top-k (--disable-shared-experts-fusion, or MegaMoE).

## Mechanism

- On the AITER path DeepseekV2MoE.forward never multiplies the routed output by routed_scaling_factor
  ("aiter moe call will handle routed_scaling_factor"; the `final_hidden_states *= self.routed_scaling_factor`
  branches are all guarded by `not _use_aiter`).
- For the score-routed layers, the DeepSeek-V4 TopK override passes
  `apply_routed_scaling_factor_on_output=(True if _use_aiter else self.experts.should_fuse_routed_scaling_factor_in_topk)`,
  so the top-k weights carry the factor.
- The HashTopK for the first num_hash_layers is built with
  `apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk` (False on AITER), so
  in those layers nothing applies the factor: the routed experts run at 1 / routed_scaling_factor (1/2.5) of their
  weight.

## Evidence (8x MI355X vs 8x B200 SGLang v0.5.20, same checkpoint, TP8, per-module tensor dumps)

- Layers 0-2: identical expert ids (90/90 for a 15-token prompt); MI355X top-k weights sum to 1.0, B200's to 2.5;
  routed-expert output rel diff 0.600 = 1 - 1/2.5. Layers 3+: weights equal.
- Layer 1 input differs from B200 by 22.7% (MI355X run-to-run noise 0.6%).
- Teacher-forced KL vs a B200 reference (200 prompts x 256 positions): 0.031 on every MI355X configuration tested
  (TP8, DP8, EP2, MegaMoE, AITER MoE), against 0.021 for B300 cells and 0.013 for B200 self-noise; top-1 96.5%.
- The T=0 prompt "List the first five prime numbers, separated by commas." answers "1, 2, 3, 5, 7" on MI355X
  (P("1") ~0.5) and "The first five prime numbers are: 2, 3, 5, 7, 11" on B200 / B300 (P("The") 0.64).

## Fix

Give HashTopK the same override (operations/amdpro/hash_scale_hooks.py does exactly this as a runtime patch):

    apply_routed_scaling_factor_on_output=(
        True
        if _use_aiter and self.num_fused_shared_experts == 0
        else self.experts.should_fuse_routed_scaling_factor_in_topk
    ),

(HashTopK rejects on-output scaling with a fused shared slot; that combination needs the factor applied to the routed
columns only.) With the fix on MI355X: KL vs B200 0.0178, KL p99 0.277, top-1 97.4%, primes prompt P("The") 0.647.
