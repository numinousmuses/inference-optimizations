// k3amd G1b (Gemma 4 31B on H100, kernel program 2026-10-05): RMSNorm (optionally fused with the residual add) +
// per-token-group-128 FP8 quant in ONE pass, writing the (q, scale) pair the next FP8 linear (qkv_proj after the input
// norm, gate_up_proj after the pre-FFN norm) consumes on its pre-quantized path. The normalized bf16 activation is never
// written. Accuracy contract (not bit-exact: the stock flashinfer norm's reduction order and build flags are outside
// this tree): the normalized value is computed as bf16(x * rsqrt(mean(x^2) + eps) * w) in fp32 from the fp32 row
// (for the fused add, the unrounded fp32 x + residual, the residual stored rounded to bf16 as the stock
// fused_add_rmsnorm does), then quantized with the stock per_token_group_quant formulas (gemm/per_token_group_quant.cuh
// fp32-scale branch): amax over the bf16 values, raw scale amax * (1/448), quant scale 448 / amax, clamp, SATFINITE.
//
// Schedule: one block per token row, 16 lanes per 128-wide group (8 bf16 per lane, as the stock quant), so a 5,376
// hidden row = 42 groups = 672 threads. Row sum of squares: per-thread fp32 over its 8 values, warp shuffle, one
// shared-memory pass across warps.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace sglang {
namespace numinous_gemma {

constexpr uint32_t kGroup = 128;
constexpr uint32_t kLanesPerGroup = 16;     // 8 bf16 (16 bytes) per lane
constexpr uint32_t kMaxThreads = 1024;      // hidden <= 8,192

struct NormQuantParams {
  const __nv_bfloat16* __restrict__ x;      // [T, H]
  __nv_bfloat16* __restrict__ residual;     // [T, H] (fused add: read, then overwritten with x + residual)
  const __nv_bfloat16* __restrict__ weight; // [H]
  uint8_t* __restrict__ q;                  // [T, H] e4m3
  float* __restrict__ scale;                // column-major: scale[token + group * scale_stride]
  uint32_t hidden;
  uint32_t x_stride, residual_stride, q_stride, scale_stride;
  float eps;
};

template <bool kAddResidual, bool kUsePDL>
__global__ void rmsnorm_quant_kernel(const __grid_constant__ NormQuantParams p) {
  extern __shared__ float warp_sums[];
  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % 32, warp = tid / 32, nwarps = blockDim.x / 32;
  const uint32_t col = tid * 8;                     // this lane's 8 contiguous values
  device::PDLWaitPrimary<kUsePDL>();
  float v[8];
  {
    const uint4 raw = *reinterpret_cast<const uint4*>(p.x + size_t(token) * p.x_stride + col);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(h[i]);
      v[2 * i] = f.x;
      v[2 * i + 1] = f.y;
    }
  }
  if constexpr (kAddResidual) {
    __nv_bfloat16* rrow = p.residual + size_t(token) * p.residual_stride + col;
    const uint4 raw = *reinterpret_cast<const uint4*>(rrow);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    uint4 outr;
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(&outr);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(h[i]);
      v[2 * i] += f.x;
      v[2 * i + 1] += f.y;
      o[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    }
    *reinterpret_cast<uint4*>(rrow) = outr;          // residual = x + residual (bf16), as fused_add_rmsnorm
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (lane == 0) warp_sums[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float t = lane < nwarps ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
    if (lane == 0) warp_sums[0] = t;
  }
  __syncthreads();
  const float rms_rcp = rsqrtf(warp_sums[0] / float(p.hidden) + p.eps);

  // normalized bf16 values (the stock norm's output), then the group quant
  __nv_bfloat162 y[4];
  {
    const uint4 raw = *reinterpret_cast<const uint4*>(p.weight + col);
    const __nv_bfloat162* w = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 wf = __bfloat1622float2(w[i]);
      y[i] = __floats2bfloat162_rn(v[2 * i] * rms_rcp * wf.x, v[2 * i + 1] * rms_rcp * wf.y);
    }
  }
  __nv_bfloat162 m2 = __habs2(y[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) m2 = __hmax2(m2, __habs2(y[i]));
  float amax_l = fmaxf(__bfloat162float(m2.x), __bfloat162float(m2.y));
#pragma unroll
  for (int off = kLanesPerGroup / 2; off > 0; off >>= 1)    // the 16 lanes of this group (aligned half-warp)
    amax_l = fmaxf(amax_l, __shfl_xor_sync(0xffffffffu, amax_l, off));
  constexpr float kMax = 448.0f;
  constexpr float kMaxInv = 1.f / kMax;
  const float amax = fmaxf(amax_l, 1e-10f);
  const float raw_scale = amax * kMaxInv;
  const float qs = kMax / amax;
  uint32_t packed[2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(y[i]);
    const __nv_fp8x2_storage_t two = __nv_cvt_float2_to_fp8x2(
        make_float2(fminf(f.x * qs, kMax), fminf(f.y * qs, kMax)), __NV_SATFINITE, __NV_E4M3);
    reinterpret_cast<uint16_t*>(packed)[i] = two;
  }
  *reinterpret_cast<uint2*>(p.q + size_t(token) * p.q_stride + col) = make_uint2(packed[0], packed[1]);
  const uint32_t group = col / kGroup;
  if ((tid % kLanesPerGroup) == 0) p.scale[token + size_t(group) * p.scale_stride] = raw_scale;
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kAddResidual, bool kUsePDL>
struct RmsNormQuantKernel {
  // x [T, H] bf16, residual [T, H] bf16 (fused add only; pass x again otherwise), weight [H] bf16,
  // q [T, H] e4m3, scale [T, H/128] fp32 column-major (token stride 1)
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView residual, tvm::ffi::TensorView weight,
                  tvm::ffi::TensorView q, tvm::ffi::TensorView scale, double eps) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto N = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto G = SymbolicSize{"num_groups"};
    TensorMatcher({N, H}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, H}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({H}).with_strides({1}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({N, H}).with_strides({-1, 1}).with_dtype<fp8_e4m3_t>().with_device(device).verify(q);
    TensorMatcher({N, G}).with_strides({1, -1}).with_dtype<fp32_t>().with_device(device).verify(scale);
    const uint32_t tokens = N.unwrap(), hidden = H.unwrap();
    CHECK_HOST(hidden % kGroup == 0) << "hidden must be a multiple of 128";
    CHECK_HOST(G.unwrap() == hidden / kGroup);
    const uint32_t threads = hidden / 8;
    CHECK_HOST(threads % 32 == 0 && threads <= kMaxThreads) << "hidden must be a multiple of 256 and <= 8192";
    CHECK_HOST(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.stride(0) * 2) % 16 == 0);
    CHECK_HOST(reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16 == 0 && (residual.stride(0) * 2) % 16 == 0);
    if (tokens == 0) return;
    const NormQuantParams p{
        static_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(residual.data_ptr()),
        static_cast<const __nv_bfloat16*>(weight.data_ptr()), static_cast<uint8_t*>(q.data_ptr()),
        static_cast<float*>(scale.data_ptr()), hidden, static_cast<uint32_t>(x.stride(0)),
        static_cast<uint32_t>(residual.stride(0)), static_cast<uint32_t>(q.stride(0)),
        static_cast<uint32_t>(scale.stride(1)), static_cast<float>(eps)};
    LaunchKernel(tokens, threads, device.unwrap(), (threads / 32) * sizeof(float))
        .config({.use_pdl = kUsePDL})(rmsnorm_quant_kernel<kAddResidual, kUsePDL>, p);
  }
};

}  // namespace numinous_gemma
}  // namespace sglang
