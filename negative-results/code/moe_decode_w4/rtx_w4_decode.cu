// rtx_w4_decode torch extension: host bindings. Device code and design notes: rtx_w4_decode_kernels.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#include "rtx_w4_decode_kernels.cuh"


// ------------------------------------------------------------------------------------------------ host bindings
static int g_sms = 0;

void prep(torch::Tensor topk_ids, int64_t E, torch::Tensor pair_ids, torch::Tensor runs, torch::Tensor num_runs) {
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 && topk_ids.is_contiguous());
  const int P = static_cast<int>(topk_ids.numel());
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(E <= rtxw4::PREP_THREADS && P <= 65535, "prep: E <= 1024 and M * topk <= 65535");
  rtxw4::prep_kernel<<<1, rtxw4::PREP_THREADS, 0, stream>>>(
      topk_ids.data_ptr<int>(), P, static_cast<int>(E), pair_ids.data_ptr<int>(),
      reinterpret_cast<int4*>(runs.data_ptr<int>()), num_runs.data_ptr<int>());
}

void gemm(int64_t fmt, bool silu, torch::Tensor W, torch::Tensor S, c10::optional<torch::Tensor> factor,
          torch::Tensor X, torch::Tensor pair_ids, torch::Tensor runs, torch::Tensor num_runs, torch::Tensor out,
          c10::optional<torch::Tensor> topk_w, int64_t K, int64_t R, int64_t N, int64_t top_k, double clamp,
          int64_t max_runs, int64_t ctas_per_sm, int64_t tpw) {
  using namespace rtxw4;
  TORCH_CHECK(X.scalar_type() == torch::kBFloat16 && X.is_contiguous() && X.size(1) == K);
  TORCH_CHECK(K % KC == 0, "K must be a multiple of ", KC);
  TORCH_CHECK(tpw == 2 || tpw == 4, "tpw must be 2 or 4");
  const int rpw = silu ? 16 * (int)(tpw / 2) : 16 * (int)tpw;
  TORCH_CHECK(silu ? (N % (WARPS * rpw) == 0 && R == 2 * N) : (R % (WARPS * rpw) == 0), "row tiling");
  Params p;
  p.W = reinterpret_cast<const uint4*>(W.data_ptr());
  p.S = S.data_ptr();
  p.factor = factor.has_value() ? factor->data_ptr<float>() : nullptr;
  p.X = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
  p.pair_ids = pair_ids.data_ptr<int>();
  p.runs = reinterpret_cast<const int4*>(runs.data_ptr<int>());
  p.num_runs = num_runs.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.topk_w = topk_w.has_value() ? topk_w->data_ptr<float>() : nullptr;
  p.K = static_cast<int>(K);
  p.R = static_cast<int>(R);
  p.N = static_cast<int>(N);
  p.top_k = static_cast<int>(top_k);
  p.groups = silu ? static_cast<int>(N / (WARPS * rpw)) : static_cast<int>(R / (WARPS * rpw));
  p.clamp = static_cast<float>(clamp);
  if (g_sms == 0) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, X.get_device());
  const size_t smem = (size_t)(2 * TMAX * TROW + WARPS * TMAX * (rpw + 8)) * sizeof(__nv_bfloat16);
  const int grid = static_cast<int>(std::min<int64_t>(max_runs * p.groups, (int64_t)g_sms * ctas_per_sm));
  auto stream = at::cuda::getCurrentCUDAStream();
#define RTXW4_LAUNCH(F, SIL, T, MB)                                                                   \
  do {                                                                                                \
    auto kern = w4_kernel<F, SIL, T, MB>;                                                             \
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));  \
    kern<<<grid, THREADS, smem, stream>>>(p);                                                         \
  } while (0)
#define RTXW4_FMT(F)                                  \
  do {                                                \
    if (tpw == 2) {                                   \
      if (silu) RTXW4_LAUNCH(F, true, 2, 4);          \
      else RTXW4_LAUNCH(F, false, 2, 4);              \
    } else {                                          \
      if (silu) RTXW4_LAUNCH(F, true, 4, 2);          \
      else RTXW4_LAUNCH(F, false, 4, 2);              \
    }                                                 \
  } while (0)
  if (fmt == FMT_NVFP4) RTXW4_FMT(FMT_NVFP4);
  else RTXW4_FMT(FMT_U4B8);
#undef RTXW4_FMT
#undef RTXW4_LAUNCH
}

void gemm_bulk(int64_t fmt, bool silu, torch::Tensor W, torch::Tensor S, c10::optional<torch::Tensor> factor,
               torch::Tensor X, torch::Tensor pair_ids, torch::Tensor runs, torch::Tensor num_runs, torch::Tensor out,
               c10::optional<torch::Tensor> topk_w, int64_t K, int64_t R, int64_t N, int64_t top_k, double clamp,
               int64_t max_runs, int64_t ctas_per_sm, int64_t tpw) {
  using namespace rtxw4;
  TORCH_CHECK(X.scalar_type() == torch::kBFloat16 && X.is_contiguous() && X.size(1) == K);
  TORCH_CHECK(K % KC == 0 && (reinterpret_cast<uintptr_t>(W.data_ptr()) % 16) == 0 &&
              (reinterpret_cast<uintptr_t>(S.data_ptr()) % 16) == 0, "bulk copies need 16-B alignment");
  TORCH_CHECK(tpw == 2 || tpw == 4, "tpw must be 2 or 4");
  const int rpw = silu ? 16 * (int)(tpw / 2) : 16 * (int)tpw;
  TORCH_CHECK(silu ? (N % (WARPS * rpw) == 0 && R == 2 * N) : (R % (WARPS * rpw) == 0), "row tiling");
  Params p;
  p.W = reinterpret_cast<const uint4*>(W.data_ptr());
  p.S = S.data_ptr();
  p.factor = factor.has_value() ? factor->data_ptr<float>() : nullptr;
  p.X = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
  p.pair_ids = pair_ids.data_ptr<int>();
  p.runs = reinterpret_cast<const int4*>(runs.data_ptr<int>());
  p.num_runs = num_runs.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.topk_w = topk_w.has_value() ? topk_w->data_ptr<float>() : nullptr;
  p.K = static_cast<int>(K);
  p.R = static_cast<int>(R);
  p.N = static_cast<int>(N);
  p.top_k = static_cast<int>(top_k);
  p.groups = silu ? static_cast<int>(N / (WARPS * rpw)) : static_cast<int>(R / (WARPS * rpw));
  p.clamp = static_cast<float>(clamp);
  if (g_sms == 0) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, X.get_device());
  constexpr int DEPTH = 4;
  const int srec = fmt == FMT_NVFP4 ? 64 : 32;
  const size_t smem = (size_t)(2 * TMAX * TROW + WARPS * TMAX * (rpw + 8)) * sizeof(__nv_bfloat16) +
                      (size_t)WARPS * DEPTH * tpw * (512 + srec) + (size_t)WARPS * DEPTH * 8;
  const int grid = static_cast<int>(std::min<int64_t>(max_runs * p.groups, (int64_t)g_sms * ctas_per_sm));
  auto stream = at::cuda::getCurrentCUDAStream();
#define RTXB_LAUNCH(F, SIL, T, MB)                                                                    \
  do {                                                                                                \
    auto kern = w4b_kernel<F, SIL, T, MB, DEPTH>;                                                     \
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));  \
    kern<<<grid, THREADS, smem, stream>>>(p);                                                         \
  } while (0)
#define RTXB_FMT(F)                                   \
  do {                                                \
    if (tpw == 2) {                                   \
      if (silu) RTXB_LAUNCH(F, true, 2, 4);           \
      else RTXB_LAUNCH(F, false, 2, 4);               \
    } else {                                          \
      if (silu) RTXB_LAUNCH(F, true, 4, 2);           \
      else RTXB_LAUNCH(F, false, 4, 2);               \
    }                                                 \
  } while (0)
  if (fmt == FMT_NVFP4) RTXB_FMT(FMT_NVFP4);
  else RTXB_FMT(FMT_U4B8);
#undef RTXB_FMT
#undef RTXB_LAUNCH
}

void gemm_w4a8(bool silu, torch::Tensor W, torch::Tensor S, torch::Tensor X, torch::Tensor pair_ids,
               torch::Tensor runs, torch::Tensor num_runs, torch::Tensor out, c10::optional<torch::Tensor> topk_w,
               torch::Tensor a_in, c10::optional<torch::Tensor> a_out, int64_t K, int64_t R, int64_t N,
               int64_t top_k, double clamp, int64_t max_runs, int64_t ctas_per_sm, int64_t tpw, int64_t s_align) {
  using namespace rtxw4;
  TORCH_CHECK(X.scalar_type() == torch::kFloat8_e4m3fn && X.is_contiguous() && X.size(1) == K);
  TORCH_CHECK(W.is_contiguous() && S.is_contiguous() && S.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(K % KC8 == 0, "K must be a multiple of ", KC8);
  TORCH_CHECK(tpw == 2 || tpw == 4, "tpw must be 2 or 4");
  const int rpw = silu ? 16 * (int)(tpw / 2) : 16 * (int)tpw;
  TORCH_CHECK(silu ? (N % (WARPS * rpw) == 0 && R == 2 * N) : (R % (WARPS * rpw) == 0), "row tiling");
  TORCH_CHECK(!silu || a_out.has_value(), "SILU output needs a_out (the FP8 scale of the intermediate)");
  ParamsF8 p;
  p.W = reinterpret_cast<const uint8_t*>(W.data_ptr());
  p.S = reinterpret_cast<const __nv_bfloat16*>(S.data_ptr());
  p.X = reinterpret_cast<const uint8_t*>(X.data_ptr());
  p.pair_ids = pair_ids.data_ptr<int>();
  p.runs = reinterpret_cast<const int4*>(runs.data_ptr<int>());
  p.num_runs = num_runs.data_ptr<int>();
  p.out = out.data_ptr();
  p.topk_w = topk_w.has_value() ? topk_w->data_ptr<float>() : nullptr;
  p.a_in = a_in.data_ptr<float>();
  p.a_out = a_out.has_value() ? a_out->data_ptr<float>() : a_in.data_ptr<float>();
  p.K = static_cast<int>(K);
  p.R = static_cast<int>(R);
  p.N = static_cast<int>(N);
  p.top_k = static_cast<int>(top_k);
  p.groups = silu ? static_cast<int>(N / (WARPS * rpw)) : static_cast<int>(R / (WARPS * rpw));
  p.clamp = static_cast<float>(clamp);
  TORCH_CHECK(s_align == 1 || s_align == 4, "s_align must be 1 or 4");
  TORCH_CHECK(S.dim() == 3 && S.size(1) * s_align == K / 128 && S.size(2) == R * s_align, "scale layout");
  p.s_align = static_cast<int>(s_align);
  if (g_sms == 0) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, X.get_device());
  const int eb = silu ? 1 : 2;
  const size_t smem = (size_t)2 * TMAX * TROW8 + (size_t)WARPS * TMAX * (rpw * eb + 16);
  const int grid = static_cast<int>(std::min<int64_t>(max_runs * p.groups, (int64_t)g_sms * ctas_per_sm));
  auto stream = at::cuda::getCurrentCUDAStream();
#define RTXF8_LAUNCH(SIL, T, MB)                                                                      \
  do {                                                                                                \
    auto kern = w4f8_kernel<SIL, T, MB>;                                                              \
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));  \
    kern<<<grid, THREADS, smem, stream>>>(p);                                                         \
  } while (0)
  // ptxas (compile_check.py): 2 tiles/warp spills under MINB 4 (128 regs) and MINB 3 (168 regs, 20-48 B), and is
  // clean only at MINB 2. 4 tiles/warp spills even at 255 regs, so it is not offered.
  TORCH_CHECK(tpw == 2, "gemm_w4a8: tpw 2 only (4 spills at 255 registers)");
  if (silu) RTXF8_LAUNCH(true, 2, 2);
  else RTXF8_LAUNCH(false, 2, 2);
#undef RTXF8_LAUNCH
}

void quant_e4m3(torch::Tensor x, torch::Tensor q, torch::Tensor scale) {
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.numel() % 8 == 0);
  const int64_t n8 = x.numel() / 8;
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((n8 + threads - 1) / threads, 1024));
  auto stream = at::cuda::getCurrentCUDAStream();
  rtxw4::quant_e4m3_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                                           reinterpret_cast<uint8_t*>(q.data_ptr()),
                                                           scale.data_ptr<float>(), n8);
}

void topk_sum(torch::Tensor c3, torch::Tensor topk_ids, torch::Tensor out, double scale) {
  const int M = static_cast<int>(out.size(0)), H = static_cast<int>(out.size(1));
  const int topk = static_cast<int>(topk_ids.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();
  rtxw4::sum_kernel<<<M, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(c3.data_ptr()),
                                           topk_ids.data_ptr<int>(),
                                           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), topk, H,
                                           static_cast<float>(scale));
}

std::vector<int64_t> kernel_attrs() {
  using namespace rtxw4;
  std::vector<int64_t> out;
  cudaFuncAttributes a;
  for (auto f : {(const void*)w4_kernel<FMT_NVFP4, true, 2, 4>, (const void*)w4_kernel<FMT_NVFP4, false, 2, 4>,
                 (const void*)w4_kernel<FMT_NVFP4, true, 4, 2>, (const void*)w4_kernel<FMT_NVFP4, false, 4, 2>,
                 (const void*)w4_kernel<FMT_U4B8, true, 2, 4>, (const void*)w4_kernel<FMT_U4B8, false, 2, 4>,
                 (const void*)w4_kernel<FMT_U4B8, true, 4, 2>, (const void*)w4_kernel<FMT_U4B8, false, 4, 2>,
                 (const void*)w4f8_kernel<true, 2, 2>, (const void*)w4f8_kernel<false, 2, 2>,
                 (const void*)w4b_kernel<FMT_NVFP4, true, 2, 4, 4>, (const void*)w4b_kernel<FMT_NVFP4, false, 2, 4, 4>,
                 (const void*)w4b_kernel<FMT_NVFP4, true, 4, 2, 4>, (const void*)w4b_kernel<FMT_NVFP4, false, 4, 2, 4>}) {
    cudaFuncGetAttributes(&a, f);
    out.push_back(a.numRegs);
    out.push_back(static_cast<int64_t>(a.localSizeBytes));
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel_attrs", &kernel_attrs);
  m.def("prep", &prep);
  m.def("gemm", &gemm);
  m.def("topk_sum", &topk_sum);
  m.def("gemm_w4a8", &gemm_w4a8);
  m.def("gemm_bulk", &gemm_bulk);
  m.def("quant_e4m3", &quant_e4m3);
}
