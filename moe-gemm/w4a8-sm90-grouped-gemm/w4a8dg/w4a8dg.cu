// Host side of the k3amd W4A8 grouped GEMM (w4a8dg_kernel.cuh): TMA descriptors, tile prefix, launch, torch op.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include "w4a8dg_kernel.cuh"
#include "w4a8rs_kernel.cuh"

namespace k3amd_w4a8dg {

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeFn encode_fn() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p,
                "k3amd_w4a8dg: cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeFn>(p);
  }
  return fn;
}

static CUtensorMap make_2d(void* base, uint64_t inner, uint64_t outer, uint64_t row_bytes, uint32_t box_inner,
                           uint32_t box_outer, CUtensorMapSwizzle swz) {
  CUtensorMap m;
  cuuint64_t dims[2] = {inner, outer};
  cuuint64_t strides[1] = {row_bytes};
  cuuint32_t box[2] = {box_inner, box_outer};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "k3amd_w4a8dg: cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

__global__ void tile_prefix(const int32_t* offs, int32_t* tile_start, int E, int n_blocks, int block_m) {
  // one block, E <= 1024 threads: exclusive scan of n_blocks * ceil(rows_e / block_m)
  extern __shared__ int32_t s[];
  int e = threadIdx.x;
  int v = 0;
  if (e < E) {
    int rows = offs[e + 1] - offs[e];
    v = n_blocks * ((rows + block_m - 1) / block_m);
  }
  s[e] = v;
  __syncthreads();
  for (int off = 1; off < blockDim.x; off <<= 1) {
    int t = e >= off ? s[e - off] : 0;
    __syncthreads();
    s[e] += t;
    __syncthreads();
  }
  if (e < E) tile_start[e + 1] = s[e];
  if (e == 0) tile_start[0] = 0;
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kStages, bool kDoubleAcc = false, uint32_t kProbe = 0,
          bool kOffset = false, bool kSwap = false>
void run(torch::Tensor& d, const torch::Tensor& a, const torch::Tensor& w, const torch::Tensor& scales,
         const torch::Tensor& offs, const torch::Tensor& a_scale) {
  const int64_t M = a.size(0), K = a.size(1), E = w.size(0), N = w.size(1);
  TORCH_CHECK(w.size(2) * 2 == K && K % 128 == 0 && N % BLOCK_N == 0, "k3amd_w4a8dg: shapes");
  TORCH_CHECK(scales.size(0) == E && scales.size(1) == K / 128 && scales.size(2) == N, "k3amd_w4a8dg: scales [E, K/128, N]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(scales.data_ptr()) % 16 == 0, "k3amd_w4a8dg: TMA needs 16-byte aligned bases");
  TORCH_CHECK(E <= 1024, "k3amd_w4a8dg: at most 1024 experts");
  at::cuda::CUDAGuard guard(a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto tile_start = torch::empty({E + 1}, offs.options());
  int threads = 1;
  while (threads < E) threads <<= 1;
  tile_prefix<<<1, threads, threads * sizeof(int32_t), stream>>>(offs.data_ptr<int32_t>(), tile_start.data_ptr<int32_t>(),
                                                                 (int)E, (int)(N / BLOCK_N), (int)BLOCK_M);
  CUtensorMap ta = make_2d(a.data_ptr(), K, M, K, 128, BLOCK_M, CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap tw = make_2d(w.data_ptr(), K / 2, E * N, K / 2, 64, BLOCK_N, CU_TENSOR_MAP_SWIZZLE_NONE);
  // scales [E, K/128, N] bf16 as bytes: rows E*K/128, row = 2N bytes, one K block's BLOCK_N scales per box
  CUtensorMap ts = make_2d(scales.data_ptr(), 2 * N, E * (K / 128), 2 * N, 2 * BLOCK_N, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  using SM = Smem<BLOCK_M, BLOCK_N, kStages>;
  auto kernel = w4a8dg_kernel<BLOCK_M, BLOCK_N, kStages, kDoubleAcc, kProbe, kOffset, kSwap>;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SM::TOTAL) == cudaSuccess,
                "k3amd_w4a8dg: smem attribute");
    attr = true;
  }
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  kernel<<<sms, kThreads, SM::TOTAL, stream>>>(ta, tw, ts, offs.data_ptr<int32_t>(), tile_start.data_ptr<int32_t>(),
                                         a_scale.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
                                         (uint32_t)E, (uint32_t)N, (uint32_t)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// v4: register-sourced weights (w4a8rs_kernel.cuh); the W4 tile is TMA'd with the 64B swizzle
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kStages, uint32_t kProbe = 0, bool kSilu = false>
void run_rs(torch::Tensor& d, const torch::Tensor& a, const torch::Tensor& w, const torch::Tensor& scales,
            const torch::Tensor& offs, const torch::Tensor& a_scale, const torch::Tensor* a2_scale = nullptr) {
  const int64_t M = a.size(0), K = a.size(1), E = w.size(0), N = w.size(1);
  TORCH_CHECK(w.size(2) * 2 == K && K % 128 == 0 && N % BLOCK_N == 0, "k3amd_w4a8dg: shapes");
  TORCH_CHECK(scales.size(0) == E && scales.size(1) == K / 128 && scales.size(2) == N, "k3amd_w4a8dg: scales [E, K/128, N]");
  TORCH_CHECK(E <= 1024, "k3amd_w4a8dg: at most 1024 experts");
  at::cuda::CUDAGuard guard(a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto tile_start = torch::empty({E + 1}, offs.options());
  int threads = 1;
  while (threads < E) threads <<= 1;
  tile_prefix<<<1, threads, threads * sizeof(int32_t), stream>>>(offs.data_ptr<int32_t>(), tile_start.data_ptr<int32_t>(),
                                                                 (int)E, (int)(N / BLOCK_N), (int)BLOCK_M);
  CUtensorMap ta = make_2d(a.data_ptr(), K, M, K, 128, BLOCK_M, CU_TENSOR_MAP_SWIZZLE_128B);
  // kSilu loads 64 gate + 64 up rows per stage (two half boxes), and their two half scale rows
  constexpr uint32_t kBoxRows = kSilu ? BLOCK_N / 2 : BLOCK_N;
  CUtensorMap tw = make_2d(w.data_ptr(), K / 2, E * N, K / 2, 64, kBoxRows, CU_TENSOR_MAP_SWIZZLE_64B);
  CUtensorMap ts = make_2d(scales.data_ptr(), 2 * N, E * (K / 128), 2 * N, 2 * kBoxRows, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  using SM = SmemRS<BLOCK_M, BLOCK_N, kStages>;
  auto kernel = w4a8rs_kernel<BLOCK_M, BLOCK_N, kStages, kProbe, kSilu>;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SM::TOTAL) == cudaSuccess,
                "k3amd_w4a8dg: smem attribute");
    attr = true;
  }
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  kernel<<<sms, 384, SM::TOTAL, stream>>>(ta, tw, ts, offs.data_ptr<int32_t>(), tile_start.data_ptr<int32_t>(),
                                         a_scale.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
                                         (uint32_t)E, (uint32_t)N, (uint32_t)K,
                                         a2_scale ? a2_scale->data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace k3amd_w4a8dg

// GEMM1 of the W4A8 MoE with the silu*up -> FP8 epilogue: d8 [M_total, N/2] float8_e4m3fn (rows in expert order,
// only rows < offs[E] are written), w = w13 [E, N, K/2] (gate rows 0..N/2-1, up rows N/2..N-1), a2_scale = the static
// per-tensor scale of the down projection's input (cutlass_w4a8_moe's a2_scale)
void k3amd_w4a8dg_mm_silu(torch::Tensor d8, torch::Tensor a, torch::Tensor w, torch::Tensor scales, torch::Tensor offs,
                          torch::Tensor a_scale, torch::Tensor a2_scale, int64_t cfg) {
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && w.scalar_type() == torch::kInt8, "fp8 a, int8 w");
  TORCH_CHECK(d8.scalar_type() == torch::kFloat8_e4m3fn && scales.scalar_type() == torch::kBFloat16, "fp8 out, bf16 scales");
  TORCH_CHECK(offs.scalar_type() == torch::kInt32 && a_scale.scalar_type() == torch::kFloat32 &&
              a2_scale.scalar_type() == torch::kFloat32, "int32 offs, fp32 a_scale / a2_scale");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && scales.is_contiguous() && d8.is_contiguous(), "contiguous");
  TORCH_CHECK(d8.size(1) * 2 == w.size(1) && w.size(1) % 256 == 0, "d8 [M, N/2], N % 256 == 0");
  using namespace k3amd_w4a8dg;
  switch (cfg) {
    case 50: run_rs<128, 128, 6, 0, true>(d8, a, w, scales, offs, a_scale, &a2_scale); return;
    case 51: run_rs<128, 128, 5, 0, true>(d8, a, w, scales, offs, a_scale, &a2_scale); return;
  }
  TORCH_CHECK(false, "k3amd_w4a8dg: unknown silu cfg ", cfg);
}

void k3amd_w4a8dg_mm(torch::Tensor d, torch::Tensor a, torch::Tensor w, torch::Tensor scales, torch::Tensor offs,
                     torch::Tensor a_scale, int64_t cfg) {
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && w.scalar_type() == torch::kInt8, "fp8 a, int8 w");
  TORCH_CHECK(scales.scalar_type() == torch::kBFloat16 && d.scalar_type() == torch::kBFloat16, "bf16 scales / out");
  TORCH_CHECK(offs.scalar_type() == torch::kInt32 && a_scale.scalar_type() == torch::kFloat32, "int32 offs, fp32 a_scale");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && scales.is_contiguous() && d.is_contiguous(), "contiguous");
  using namespace k3amd_w4a8dg;
  switch (cfg) {
    // 0: 4 stages; 3: warpgroup phase offset; 9-14: timing ablations (garbage output), see P_* in the kernel header
    case 0: k3amd_w4a8dg::run<128, 128, 4>(d, a, w, scales, offs, a_scale); return;
    case 3: k3amd_w4a8dg::run<128, 128, 4, false, 0, true>(d, a, w, scales, offs, a_scale); return;
    case 9: k3amd_w4a8dg::run<128, 128, 4, false, P_NOCVT>(d, a, w, scales, offs, a_scale); return;
    case 10: k3amd_w4a8dg::run<128, 128, 4, false, P_NOCVT | P_NOEPI>(d, a, w, scales, offs, a_scale); return;
    case 11: k3amd_w4a8dg::run<128, 128, 4, false, P_NOCVT | P_DIRECT>(d, a, w, scales, offs, a_scale); return;
    case 12: k3amd_w4a8dg::run<128, 128, 4, false, P_NOCVT | P_DIRECT | P_NOTMA>(d, a, w, scales, offs, a_scale); return;
    case 13: k3amd_w4a8dg::run<128, 128, 4, false, P_NOCVT | P_NOPROMO>(d, a, w, scales, offs, a_scale); return;
    case 14: k3amd_w4a8dg::run<128, 128, 4, false, P_NOCVT | P_DIRECT | P_NOTMA | P_NOPROMO | P_NOEPI>(
        d, a, w, scales, offs, a_scale); return;
    case 15: k3amd_w4a8dg::run<128, 128, 4, false, P_NOPROMO>(d, a, w, scales, offs, a_scale); return;
    // kSwap: weights = WGMMA A (accumulator rows), tokens = B
    case 20: k3amd_w4a8dg::run<128, 128, 4, false, 0, false, true>(d, a, w, scales, offs, a_scale); return;
    case 21: k3amd_w4a8dg::run<128, 128, 4, false, P_NOCVT, false, true>(d, a, w, scales, offs, a_scale); return;
    case 22: k3amd_w4a8dg::run<128, 128, 4, false, P_NOCVT, true, true>(d, a, w, scales, offs, a_scale); return;
    case 23: k3amd_w4a8dg::run<128, 128, 4, false, 0, true, true>(d, a, w, scales, offs, a_scale); return;
    case 24: k3amd_w4a8dg::run<128, 128, 3, false, 0, false, true>(d, a, w, scales, offs, a_scale); return;
    case 25: k3amd_w4a8dg::run<128, 128, 4, false, P_NOCVT | P_NOPROMO, false, true>(d, a, w, scales, offs, a_scale);
      return;
    // v4 register-sourced weights: 40/41/42 = 6/5/4 stages; 43 no epilogue, 44 no promotion (timing ablations)
    case 40: k3amd_w4a8dg::run_rs<128, 128, 6>(d, a, w, scales, offs, a_scale); return;
    case 41: k3amd_w4a8dg::run_rs<128, 128, 5>(d, a, w, scales, offs, a_scale); return;
    case 42: k3amd_w4a8dg::run_rs<128, 128, 4>(d, a, w, scales, offs, a_scale); return;
    case 43: k3amd_w4a8dg::run_rs<128, 128, 6, P_NOEPI>(d, a, w, scales, offs, a_scale); return;
    case 44: k3amd_w4a8dg::run_rs<128, 128, 6, P_NOPROMO>(d, a, w, scales, offs, a_scale); return;
  }
  TORCH_CHECK(false, "k3amd_w4a8dg: unknown cfg ", cfg);
}

TORCH_LIBRARY(k3amd_w4a8dg, m) {
  m.def("mm(Tensor(a!) d, Tensor a, Tensor w, Tensor scales, Tensor offs, Tensor a_scale, int cfg) -> ()");
  m.impl("mm", torch::kCUDA, &k3amd_w4a8dg_mm);
  m.def("mm_silu(Tensor(a!) d8, Tensor a, Tensor w, Tensor scales, Tensor offs, Tensor a_scale, Tensor a2_scale, "
        "int cfg) -> ()");
  m.impl("mm_silu", torch::kCUDA, &k3amd_w4a8dg_mm_silu);
}
