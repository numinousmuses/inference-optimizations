"""Anchored edit of DeepGEMM 0.2.0's JIT header deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh (agent glmidx).

Variant "hoist": the math warps load all BLOCK_Q rows' head accumulators of a KV split from TMEM in one batch
(<= 128 columns), wait once, release the TMEM stage to the MMA warp immediately, then reduce the rows. Stock issues
two 16-column loads + two waits per row and holds the stage until the last row, which serializes TMEM latency with
the ReLU-weight-sum on only 8 math warps per SM (the measured bound: FP4 keys, half the MMA time, run no faster).

    python3 patch_mqa_logits.py <path to sm100_mqa_logits.cuh> [--check]
DeepGEMM JIT-compiles this header at first use (cache /root/.dj keyed on the source), so the edit takes effect in a
fresh process without rebuilding _C.so.
"""
import sys

MARK = "// numinous-hook:mqa_logits_hoist"
A1 = ("                    #pragma unroll\n"
      "                    for (uint32_t i = 0; i < kNumValidTokens; ++ i) {\n"
      "                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;\n"
      "                        if constexpr (kNumHeads == 8) {\n")
R1 = (f"                    {MARK}\n"
      "                    constexpr bool kHoistTmem = (kNumValidTokens * kNumHeads <= 128) and not kPagedWideLoad;\n"
      "                    float accum_all[kHoistTmem ? kNumValidTokens * kNumHeads : 1];\n"
      "                    if constexpr (kHoistTmem) {\n"
      "                        load_tmem_heads_decomposed<kNumValidTokens * kNumHeads>(tmem_stage_idx * UMMA_N, accum_all);\n"
      "                        cutlass::arch::fence_view_async_tmem_load();\n"
      "                        ptx::tcgen05_before_thread_sync();\n"
      "                        smem.empty_tmem_barriers[tmem_stage_idx].arrive();\n"
      "                    }\n"
      "                    #pragma unroll\n"
      "                    for (uint32_t i = 0; i < kNumValidTokens; ++ i) {\n"
      "                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;\n"
      "                        if constexpr (kHoistTmem) {\n"
      "                            #pragma unroll\n"
      "                            for (uint32_t j = 0; j < kNumHeads; ++ j)\n"
      "                                accum[j] = accum_all[i * kNumHeads + j];\n"
      "                        } else if constexpr (kNumHeads == 8) {\n")
A2 = ("                        if (i == kNumValidTokens - 1) {\n"
      "                            ptx::tcgen05_before_thread_sync();\n"
      "                            smem.empty_tmem_barriers[tmem_stage_idx].arrive();\n"
      "                        }\n")
R2 = ("                        if (not kHoistTmem and i == kNumValidTokens - 1) {\n"
      "                            ptx::tcgen05_before_thread_sync();\n"
      "                            smem.empty_tmem_barriers[tmem_stage_idx].arrive();\n"
      "                        }\n")


def patch(src: str) -> str:
    if MARK in src:
        return src
    for a in (A1, A2):
        n = src.count(a)
        if n != 1:
            raise SystemExit(f"anchor found {n} times: {a[:80]!r}")
    return src.replace(A1, R1, 1).replace(A2, R2, 1)


if __name__ == "__main__":
    path = sys.argv[1]
    src = open(path).read()
    new = patch(src)
    if "--check" in sys.argv:
        print("ok" if new != src or MARK in src else "no change")
    else:
        open(path, "w").write(new)
        print("patched" if new != src else "already patched", path)
