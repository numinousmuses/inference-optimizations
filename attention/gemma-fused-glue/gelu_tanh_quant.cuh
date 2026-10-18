// k3amd G1a (Gemma 4 31B on H100, kernel program 2026-10-05): gelu_tanh(gate) * up + per-token-group-128 FP8 quant in
// ONE pass over the gate_up GEMM output, writing the (q, scale) pair the FP8 down projection consumes directly
// (fp8_utils.py deepgemm_w8a8_block_fp8_linear_with_fallback, pre-quantized path).
//
// Bit-exact by construction with the stock two-kernel sequence on SM90, both JIT-compiled with --use_fast_math:
//   1. elementwise/activation.cuh act_and_mul_kernel<bf16, kGELUTanh>: out = bf16(gelu_tanh_f32(gate) * up_f32)
//      (one rounding; apply_activation_f32<kGELUTanh> copied verbatim below);
//   2. gemm/per_token_group_quant.cuh QuantTrait::run_packed16 on those bf16 values (amax in the bf16 domain,
//      raw_scale = amax * (1/448), quant_scale = 448 / amax, fp32 multiply, clamp, SATFINITE cvt). This file reuses
//      the stock trait, kernels and host checks; only the fused-activation slot of run_packed16 differs (the stock
//      slot is silu with two roundings, act rounded to bf16 then __hmul2).
#pragma once

#include <gemm/per_token_group_quant.cuh>

namespace sglang {
namespace numinous_gemma {

// elementwise/activation.cuh apply_activation_f32<ActivationKind::kGELUTanh>, verbatim
SGL_DEVICE float gelu_tanh_f32(float x_f32) {
  constexpr auto kGeluTanhAlpha = 0.044715f;
  constexpr auto kGeluTanhBeta = 0.7978845608028654f;
  const float cdf = 0.5f * (1.0f + tanhf(kGeluTanhBeta * (x_f32 + kGeluTanhAlpha * x_f32 * x_f32 * x_f32)));
  return x_f32 * cdf;
}

// The stock QuantTrait (bf16 in, e4m3 out, group 128, fp32 scales, column-major, fused slot on so the host checks
// expect [T, 2H] input) with run() replaced: the fused slot computes gelu_tanh(gate) * up.
struct GeluTanhQuantTrait : QuantTrait<bf16_t, fp8_e4m3_t, 128, false, false, true, true> {
  SGL_DEVICE static void
  run(const QuantKernelParams& params,
      const uint32_t expert_idx,
      const uint32_t token_idx,
      const uint32_t group_idx,
      const uint32_t lane_id) {
    using namespace device;
    using T = InputType;
    using T2 = packed_t<T>;
    using Q = QuantType;
    using WTrait = detail::WeightTrait<Q>;
    using Q2 = typename WTrait::packed2_t;
    using in_vec_t = detail::Vec32B<T2>;
    using out_vec_t = AlignedVector<Q2, kVecSize / 2>;
    constexpr float kMaxValue = WTrait::kMaxValue;
    constexpr float kMaxValueInv = 1.f / kMaxValue;

    const T* token_in = params.input.get<const T>(expert_idx, token_idx);
    const uint32_t group_offset = group_idx * kGroupSize;

    in_vec_t in;
    in.load(token_in + group_offset, lane_id);
    {
      in_vec_t up;
      up.load(token_in + group_offset + params.hidden_size, lane_id);
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        const auto gate = cast<float2>(in[i]);
        const auto upf = cast<float2>(up[i]);
        // act_and_mul_kernel: out[i] = cast<T>(apply_activation_f32(gate_f32) * up_f32), one rounding
        in[i] = cast<T2>(float2{gelu_tanh_f32(gate.x) * upf.x, gelu_tanh_f32(gate.y) * upf.y});
      }
    }

    // from here: QuantTrait::run_packed16, fp32-scale branch, verbatim
    T2 local_amax2 = math::abs(in[0]);
#pragma unroll
    for (uint32_t i = 1; i < kVecSize / 2; ++i) {
      local_amax2 = math::max(local_amax2, math::abs(in[i]));
    }
    const auto amax2 = cast<float2>(warp::reduce_max<kNumLanes>(local_amax2));
    const auto amax = math::max(math::max(amax2.x, amax2.y), 1e-10f);
    const float raw_scale = amax * kMaxValueInv;

    out_vec_t out;
    const float scale_inv = raw_scale;
    const float quant_scale = kMaxValue / amax;
    const float2 quant_scale2 = {quant_scale, quant_scale};
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      out[i] = WTrait::quant(detail::mul2(cast<float2>(in[i]), quant_scale2));
    }

    out.store(params.output.get<Q>(expert_idx, token_idx) + group_offset, lane_id);
    params.scale.store<kUe8m0, kRowMajor, kAligned>(expert_idx, token_idx, group_idx, scale_inv);
  }
};

template <bool kUsePDL>
struct GeluTanhQuantFlatKernel {
  using Trait = GeluTanhQuantTrait;

  // input [T, 2H] bf16 (gate | up), output_q [T, H] e4m3, output_s [T, H/128] fp32 column-major (token stride 1)
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
    using namespace host;
    const auto ctx = build_quant_context<Trait, /*kMasked=*/false>(input, output_q, output_s);
    const auto& p = ctx.params;
    const int64_t total_threads = int64_t{p.num_tokens} * p.scale.num_groups * Trait::kNumLanes;
    if (total_threads == 0) return;
    const uint32_t num_blocks = div_ceil(total_threads, int64_t{Trait::kBlockSize});
    LaunchKernel(num_blocks, Trait::kBlockSize, ctx.device)
        .config({.use_pdl = kUsePDL})(per_token_group_quant_flat_kernel<Trait, kUsePDL>, p);
  }
};

}  // namespace numinous_gemma
}  // namespace sglang
