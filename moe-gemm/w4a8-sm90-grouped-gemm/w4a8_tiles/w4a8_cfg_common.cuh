// One SM90 W4A8 grouped-GEMM tile configuration per translation unit (k3amd). The kernel template and its caller are
// sglang-kernel 0.4.7's own (csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh, shipped in the production image at
// /sgl-workspace/sglang/python/sglang/kernels/aot/csrc); only the TileShape / ClusterShape / schedule vary.
#pragma once
#include <cudaTypedefs.h>
#include <type_traits>

#include "w4a8_cfg_decl.h"
#include "cutlass/cutlass.h"
#include "w4a8_grouped_mm_c3x.cuh"

using namespace cute;

enum class Sched { PP, CO };

template <int M, int N, int K, int A, int B, int C, Sched S>
struct SM90W4A8Config {
  using KernelSchedule = std::conditional_t<S == Sched::PP, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
                                            cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
  using EpilogueSchedule = std::conditional_t<S == Sched::PP, cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
                                              cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename Config>
inline void invoke_cfg(W4A8_ARGS) {
  cutlass_w4a8_group_gemm_caller<typename Config::Gemm>(W4A8_PASS);
}
