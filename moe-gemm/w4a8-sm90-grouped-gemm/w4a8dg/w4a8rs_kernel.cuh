// k3amd W4A8 grouped GEMM v4 for SM90: register-sourced weights (RS WGMMA), conversion inside the math warps.
//
// Same contract as w4a8dg_kernel.cuh (D[t, n] = a_scale * sum_g S[e, g, n] * sum_{k in g} A[t, k] * W[e, n, k]; A fp8
// [M_total, K] grouped by expert; W int8 [E, N, K/2] = signed int4 pairs, low nibble = even k (the resident cutlass_w4a8
// bytes, claim glm-moe-one-resident-layout); S bf16 [E, K/128, N]; D bf16 [M_total, N]).
//
// Why v4 (measured 18:16 / 18:26 on 1xH200, EP8 w13 16K): with converter warps writing an FP8 weight tile to shared
// memory, conversion cost +533 us on top of an 810 us structure (v3 swap). Here the weights never exist as FP8 in shared
// memory: each math thread loads exactly the int4 it needs for its WGMMA A fragment and converts in registers.
//
//   WGMMA m64n128k32 RS: A = 64 weight rows (registers), B = 128 tokens (shared memory, TMA 128B swizzle).
//   A fragment (cute ALayout_64x32): thread (warp w, lane = 4 g + c) holds, per k32 step j,
//     reg0 = row 16w+g,   k 32j+4c .. +3      reg1 = row 16w+g+8, k 32j+4c .. +3
//     reg2 = row 16w+g,   k 32j+16+4c .. +3   reg3 = row 16w+g+8, k 32j+16+4c .. +3
//   four int4 of one row at k 4c.. are ONE u16 of the row-major int4 row (byte 16j + 2c, or 16j + 8 + 2c), so each
//   register = one LDS.U16 + one PRMT-LUT cvt4. The W4 tile is TMA'd with the 64B swizzle, which makes every one of
//   those LDS.U16 warp instructions a single bank-conflict-free wavefront (test_w4a8rs_layout.py emulates it).
//   A weight row is an accumulator ROW, so promotion uses 2 register scales per thread per K block.
//   The next K block's fragments are loaded and converted while the current K block's WGMMAs run (two A buffers).
//   The epilogue runs on the producer warpgroup's three idle warps from a double-buffered staging tile, so the math
//   warpgroups go straight to the next tile.
//
// kSilu (GEMM1 of the MoE only): the tile's 128 weight rows are 64 gate rows (warpgroup 0) and the matching 64 up rows
// (warpgroup 1, row + N/2) of w13, so the epilogue emits silu(gate) * up / a2_scale as FP8 directly (the served
// path writes c1 in bf16 and re-reads it in silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe,
// ep_moe_kernels.py:766-795). Same arithmetic as that kernel: gate and up rounded to bf16 first (they were bf16 in
// c1), out = gate / (1 + exp(-gate)) * up * (1 / a2_scale), exp as Triton's ex2.approx(x * log2 e), cvt.rn.satfinite.
#pragma once

#include "w4a8dg_kernel.cuh"

namespace k3amd_w4a8dg {

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kStages>
struct SmemRS {
  static constexpr uint32_t A = BLOCK_M * 128;          // fp8 tokens, 128B swizzled (TMA): the WGMMA B operand
  static constexpr uint32_t W4 = BLOCK_N * 64;          // raw int4 pairs, 64B swizzled (TMA)
  static constexpr uint32_t SF = 1024;                  // bf16 scales of the K block (TMA), padded
  static constexpr uint32_t STAGE = A + W4 + SF;
  static constexpr uint32_t D_PITCH = BLOCK_N + 8;      // bf16 per staged output row ([token][weight])
  static constexpr uint32_t D = BLOCK_M * D_PITCH * 2;
  static constexpr uint32_t D_ALLOC = (D + 1023) / 1024 * 1024;
  static constexpr uint32_t TOTAL = kStages * STAGE + 2 * D_ALLOC + (2 * kStages + 4) * 8 + 1024;
  static_assert(A % 1024 == 0 && W4 % 1024 == 0 && STAGE % 1024 == 0 && BLOCK_N * 2 <= SF, "alignment");
  static_assert(TOTAL <= 232448, "shared memory");
};

struct RS128 {
  using Op = cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_RS_TN<>;
  static constexpr int kAcc = 64;
  template <size_t... I>
  __device__ static void call(const uint32_t* a, uint64_t b, float* d, bool acc, cute::index_sequence<I...>) {
    Op::fma(a[0], a[1], a[2], a[3], b, d[I]..., acc ? cute::SM90::GMMA::ScaleOut::One : cute::SM90::GMMA::ScaleOut::Zero);
  }
  __device__ static void fma(const uint32_t* a, uint64_t b, float* d, bool acc) {
    call(a, b, d, acc, cute::make_index_sequence<kAcc>{});
  }
};

__device__ __forceinline__ uint32_t lds_u16(const void* p) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
  return v;
}

// byte offset of logical (row, byte) in a 64-byte-row tile written by TMA with CU_TENSOR_MAP_SWIZZLE_64B
// (cute Swizzle<2,4,3>: 16-byte chunk index ^= address bits 7..8 = (row >> 1) & 3 for 64-byte rows)
__device__ __forceinline__ uint32_t sw64(uint32_t row, uint32_t byte) {
  return row * 64 + ((((byte >> 4) ^ (row >> 1)) & 3) << 4) + (byte & 15);
}

// the 16 A-fragment registers of one K block (4 k32 steps x 4 regs) for weight rows r and r + 8 of the stage tile
__device__ __forceinline__ void load_frags(const uint8_t* w4, uint32_t r, uint32_t c, uint32_t (&f)[16]) {
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    const uint32_t b0 = 16 * j + 2 * c, b1 = b0 + 8;
    const uint32_t u0 = lds_u16(w4 + sw64(r, b0)), u1 = lds_u16(w4 + sw64(r + 8, b0));
    const uint32_t u2 = lds_u16(w4 + sw64(r, b1)), u3 = lds_u16(w4 + sw64(r + 8, b1));
    f[4 * j + 0] = cvt4(u0 & 0x7777u, (u0 & 0x8888u) >> 1);
    f[4 * j + 1] = cvt4(u1 & 0x7777u, (u1 & 0x8888u) >> 1);
    f[4 * j + 2] = cvt4(u2 & 0x7777u, (u2 & 0x8888u) >> 1);
    f[4 * j + 3] = cvt4(u3 & 0x7777u, (u3 & 0x8888u) >> 1);
  }
}

__device__ __forceinline__ float ex2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe's element op on bf16 inputs
__device__ __forceinline__ float silu_mul(float gate_bf16, float up_bf16, float inv_a2) {
  return gate_bf16 / (1.0f + ex2_approx(-gate_bf16 * 1.4426950408889634f)) * up_bf16 * inv_a2;
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kStages, uint32_t kProbe = 0, bool kSilu = false>
__global__ void __launch_bounds__(384, 1)
w4a8rs_kernel(const __grid_constant__ CUtensorMap tmap_a, const __grid_constant__ CUtensorMap tmap_w,
              const __grid_constant__ CUtensorMap tmap_s, const int32_t* __restrict__ offs,
              const int32_t* __restrict__ tile_start, const float* __restrict__ a_scale_ptr,
              __nv_bfloat16* __restrict__ d, uint32_t E, uint32_t N, uint32_t K,
              const float* __restrict__ a2_scale_ptr = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using SM = SmemRS<BLOCK_M, BLOCK_N, kStages>;
  static_assert(BLOCK_M == 128 && BLOCK_N == 128, "2 math warpgroups x 64 weight rows, WGMMA N = 128 tokens");
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  auto st_a = [&](uint32_t s) { return smem + s * SM::STAGE; };
  auto st_w4 = [&](uint32_t s) { return smem + s * SM::STAGE + SM::A; };
  auto st_sf = [&](uint32_t s) { return reinterpret_cast<const __nv_bfloat16*>(smem + s * SM::STAGE + SM::A + SM::W4); };
  auto st_d = [&](uint32_t b) { return reinterpret_cast<__nv_bfloat16*>(smem + kStages * SM::STAGE + b * SM::D_ALLOC); };
  Barrier* bars = reinterpret_cast<Barrier*>(smem + kStages * SM::STAGE + 2 * SM::D_ALLOC);
  auto full = [&](uint32_t s) { return bars + s; };
  auto empty = [&](uint32_t s) { return bars + kStages + s; };
  auto d_full = [&](uint32_t b) { return bars + 2 * kStages + b; };
  auto d_empty = [&](uint32_t b) { return bars + 2 * kStages + 2 + b; };

  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  // kSilu: a tile = 64 gate rows + their 64 up rows, so N/2 / 64 = N / BLOCK_N tiles along N as well
  const uint32_t k_blocks = K / 128, n_blocks = N / BLOCK_N;
  constexpr uint32_t kHalf = BLOCK_N / 2;
  constexpr uint32_t X_PITCH = kHalf + 8;              // kSilu exchange tile [token][gate col] bf16
  constexpr uint32_t Q_OFF = BLOCK_M * X_PITCH * 2;     // kSilu FP8 staging [token][col] after the exchange tile
  constexpr uint32_t Q_PITCH = kHalf + 16;
  static_assert(!kSilu || Q_OFF + BLOCK_M * Q_PITCH <= SM::D_ALLOC, "silu staging fits the output staging");
  if (threadIdx.x == 256) {
    cute::prefetch_tma_descriptor(&tmap_a);
    cute::prefetch_tma_descriptor(&tmap_w);
    cute::prefetch_tma_descriptor(&tmap_s);
#pragma unroll
    for (uint32_t s = 0; s < kStages; ++s) {
      full(s)->init(1);
      empty(s)->init(8);              // the eight math warps
    }
#pragma unroll
    for (uint32_t b = 0; b < 2; ++b) {
      d_full(b)->init(8);             // math warps staged their part of the tile
      d_empty(b)->init(3);            // the three store warps drained it
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  if (warp >= 8) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    uint32_t e, nb, mb, rows;
    if (warp == 8) {
      if (lane == 0) {
        uint32_t stage = 0, phase = 0;
        for (uint32_t tile = blockIdx.x; tile_coords(tile, tile_start, E, n_blocks, offs, BLOCK_M, e, nb, mb, rows);
             tile += gridDim.x) {
          const int32_t row0 = offs[e] + (int32_t)(mb * BLOCK_M);
          const int32_t wrow0 = (int32_t)(e * N + nb * BLOCK_N);
          for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            empty(stage)->wait(phase ^ 1);
            Barrier& fb = *full(stage);
            cute::SM90_TMA_LOAD_2D::copy(&tmap_a, reinterpret_cast<uint64_t*>(&fb), 0ull, st_a(stage),
                                         (int32_t)(kb * 128), row0);
            if constexpr (kSilu) {
              // gate rows nb*64.. -> tile rows 0..63, up rows N/2 + nb*64.. -> tile rows 64..127 (box = 64 rows)
              const int32_t g0 = (int32_t)(e * N + nb * kHalf), u0 = g0 + (int32_t)(N / 2);
              cute::SM90_TMA_LOAD_2D::copy(&tmap_w, reinterpret_cast<uint64_t*>(&fb), 0ull, st_w4(stage),
                                           (int32_t)(kb * 64), g0);
              cute::SM90_TMA_LOAD_2D::copy(&tmap_w, reinterpret_cast<uint64_t*>(&fb), 0ull, st_w4(stage) + kHalf * 64,
                                           (int32_t)(kb * 64), u0);
              __nv_bfloat16* sf = const_cast<__nv_bfloat16*>(st_sf(stage));
              cute::SM90_TMA_LOAD_2D::copy(&tmap_s, reinterpret_cast<uint64_t*>(&fb), 0ull, sf,
                                           (int32_t)(nb * kHalf * 2), (int32_t)(e * k_blocks + kb));
              cute::SM90_TMA_LOAD_2D::copy(&tmap_s, reinterpret_cast<uint64_t*>(&fb), 0ull, sf + kHalf,
                                           (int32_t)((N / 2 + nb * kHalf) * 2), (int32_t)(e * k_blocks + kb));
            } else {
              cute::SM90_TMA_LOAD_2D::copy(&tmap_w, reinterpret_cast<uint64_t*>(&fb), 0ull, st_w4(stage),
                                           (int32_t)(kb * 64), wrow0);
              cute::SM90_TMA_LOAD_2D::copy(&tmap_s, reinterpret_cast<uint64_t*>(&fb), 0ull,
                                           const_cast<__nv_bfloat16*>(st_sf(stage)), (int32_t)(nb * BLOCK_N * 2),
                                           (int32_t)(e * k_blocks + kb));
            }
            fb.arrive_and_expect_tx(SM::A + SM::W4 + BLOCK_N * 2);
            stage = (stage + 1 == kStages) ? 0 : stage + 1;
            phase ^= (stage == 0);
          }
        }
      }
    } else {
      // store warps 9-11: drain the staged tile [token][weight] to global, rows masked to the expert's tokens
      const uint32_t st = threadIdx.x - 288;   // 0..95
      uint32_t buf = 0, dphase = 0;
      for (uint32_t tile = blockIdx.x; tile_coords(tile, tile_start, E, n_blocks, offs, BLOCK_M, e, nb, mb, rows);
           tile += gridDim.x) {
        d_full(buf)->wait(dphase);
        if constexpr (kSilu && (kProbe & P_NOEPI) == 0) {
          const uint8_t* sq = reinterpret_cast<const uint8_t*>(st_d(buf)) + Q_OFF;
          const uint32_t m_off = mb * BLOCK_M;
          const uint32_t valid = (rows - m_off) < BLOCK_M ? (rows - m_off) : BLOCK_M;
          uint8_t* dst = reinterpret_cast<uint8_t*>(d) + ((size_t)offs[e] + m_off) * (N / 2) + nb * kHalf;
          constexpr uint32_t kChunks = kHalf / 16;
          for (uint32_t idx = st; idx < valid * kChunks; idx += 96) {
            const uint32_t r = idx / kChunks, cch = idx % kChunks;
            *reinterpret_cast<uint4*>(dst + (size_t)r * (N / 2) + cch * 16) =
                *reinterpret_cast<const uint4*>(sq + r * Q_PITCH + cch * 16);
          }
        } else if constexpr ((kProbe & P_NOEPI) == 0) {
          const __nv_bfloat16* sd = st_d(buf);
          const uint32_t m_off = mb * BLOCK_M;
          const uint32_t valid = (rows - m_off) < BLOCK_M ? (rows - m_off) : BLOCK_M;
          __nv_bfloat16* dst = d + ((size_t)offs[e] + m_off) * N + nb * BLOCK_N;
          constexpr uint32_t kChunks = BLOCK_N / 8;
          for (uint32_t idx = st; idx < valid * kChunks; idx += 96) {
            const uint32_t r = idx / kChunks, cch = idx % kChunks;
            *reinterpret_cast<uint4*>(dst + (size_t)r * N + cch * 8) =
                *reinterpret_cast<const uint4*>(sd + r * SM::D_PITCH + cch * 8);
          }
        }
        __syncwarp();
        if (lane == 0) d_empty(buf)->arrive();
        buf ^= 1;
        dphase ^= (buf == 0);
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    const uint32_t wg = threadIdx.x / 128, wl = warp % 4, g = lane / 4, c = lane % 4;
    const uint32_t wr = wg * 64 + wl * 16 + g;     // weight row of this thread's accumulator row 0 (row 1 = wr + 8)
    const float a_scale = *a_scale_ptr;
    float acc[RS128::kAcc], fin[RS128::kAcc];
    uint32_t fa[16], fb[16];
    uint32_t stage = 0, phase = 0, buf = 0, dphase = 0;
    auto advance = [&]() { stage = (stage + 1 == kStages) ? 0 : stage + 1; phase ^= (stage == 0); };
    // stage after `stage` (the next K block's), without moving
    auto next_of = [&](uint32_t s, uint32_t p, uint32_t& ns, uint32_t& np) {
      ns = (s + 1 == kStages) ? 0 : s + 1;
      np = p ^ (ns == 0);
    };
    auto issue = [&](uint32_t (&f)[16], uint32_t s) {
      const uint64_t b0 = smem_desc(st_a(s));
#pragma unroll
      for (int i = 0; i < RS128::kAcc; ++i) fence_operand(acc[i]);
#pragma unroll
      for (int i = 0; i < 16; ++i) asm volatile("" : "+r"(f[i]) :: "memory");
      wg_arrive();
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) RS128::fma(&f[4 * j], b0 + (j * 32) / 16, acc, j > 0);
      wg_commit();
    };
    auto promote = [&](uint32_t s) {
#pragma unroll
      for (int i = 0; i < RS128::kAcc; ++i) fence_operand(acc[i]);
      const __nv_bfloat16* sf = st_sf(s);
      if constexpr ((kProbe & P_NOPROMO) != 0) {
#pragma unroll
        for (int i = 0; i < RS128::kAcc; ++i) fin[i] += acc[i];
      } else {
        const float s0 = __bfloat162float(sf[wr]), s1 = __bfloat162float(sf[wr + 8]);
#pragma unroll
        for (int i = 0; i < RS128::kAcc / 4; ++i) {
          fin[i * 4 + 0] += acc[i * 4 + 0] * s0;
          fin[i * 4 + 1] += acc[i * 4 + 1] * s0;
          fin[i * 4 + 2] += acc[i * 4 + 2] * s1;
          fin[i * 4 + 3] += acc[i * 4 + 3] * s1;
        }
      }
      __syncwarp();
      if (lane == 0) empty(s)->arrive();
    };
    const uint32_t rl = wl * 16 + g + wg * 64;      // row inside the stage's 128-row W4 tile
    uint32_t e, nb, mb, rows;
    for (uint32_t tile = blockIdx.x; tile_coords(tile, tile_start, E, n_blocks, offs, BLOCK_M, e, nb, mb, rows);
         tile += gridDim.x) {
#pragma unroll
      for (int i = 0; i < RS128::kAcc; ++i) fin[i] = 0.f;
      // K block 0's fragments
      full(stage)->wait(phase);
      load_frags(st_w4(stage), rl, c, fa);
      for (uint32_t kb = 0; kb < k_blocks; kb += 2) {
        // fa holds K block kb (stage `stage`)
        issue(fa, stage);
        if (kb + 1 < k_blocks) {
          uint32_t ns, np;
          next_of(stage, phase, ns, np);
          full(ns)->wait(np);
          load_frags(st_w4(ns), rl, c, fb);     // overlaps K block kb's WGMMAs
        }
        wg_wait<0>();
        promote(stage);
        advance();
        if (kb + 1 >= k_blocks) break;
        // fb holds K block kb + 1
        issue(fb, stage);
        if (kb + 2 < k_blocks) {
          uint32_t ns, np;
          next_of(stage, phase, ns, np);
          full(ns)->wait(np);
          load_frags(st_w4(ns), rl, c, fa);
        }
        wg_wait<0>();
        promote(stage);
        advance();
      }
      // epilogue: stage [token][weight] for the store warps (double buffer)
      d_empty(buf)->wait(dphase ^ 1);
      __nv_bfloat16* sd = st_d(buf);
      if constexpr (kSilu) {
        // warpgroup 0 holds gate(t, j), warpgroup 1 holds up(t, j) at the SAME (thread, register) positions:
        // j = wl*16 + g (+8), t = i*8 + 2c (+1). Gate goes through the exchange tile, the up side finishes.
        const uint32_t j0 = wl * 16 + g;
        if (wg == 0) {
#pragma unroll
          for (int i = 0; i < RS128::kAcc / 4; ++i) {
            const uint32_t t = i * 8 + c * 2;
            sd[t * X_PITCH + j0] = __float2bfloat16_rn(fin[i * 4 + 0] * a_scale);
            sd[(t + 1) * X_PITCH + j0] = __float2bfloat16_rn(fin[i * 4 + 1] * a_scale);
            sd[t * X_PITCH + j0 + 8] = __float2bfloat16_rn(fin[i * 4 + 2] * a_scale);
            sd[(t + 1) * X_PITCH + j0 + 8] = __float2bfloat16_rn(fin[i * 4 + 3] * a_scale);
          }
        }
        cutlass::arch::NamedBarrier::sync(256, 3);
        if (wg == 1) {
          const float inv_a2 = 1.0f / *a2_scale_ptr;
          uint8_t* sq = reinterpret_cast<uint8_t*>(sd) + Q_OFF;
#pragma unroll
          for (int i = 0; i < RS128::kAcc / 4; ++i) {
            const uint32_t t = i * 8 + c * 2;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const uint32_t tt = t + (q & 1), jj = j0 + (q >> 1) * 8;
              const float gate = __bfloat162float(sd[tt * X_PITCH + jj]);
              const float up = __bfloat162float(__float2bfloat16_rn(fin[i * 4 + q] * a_scale));
              const __nv_fp8_e4m3 o(silu_mul(gate, up, inv_a2));   // cvt.rn.satfinite
              sq[tt * Q_PITCH + jj] = *reinterpret_cast<const uint8_t*>(&o);
            }
          }
        }
        __syncwarp();
        if (lane == 0) d_full(buf)->arrive();
        buf ^= 1;
        dphase ^= (buf == 0);
        continue;
      }
#pragma unroll
      for (int i = 0; i < RS128::kAcc / 4; ++i) {
        const uint32_t t = i * 8 + c * 2;          // token column of acc[i*4 + 0 / 2]; +1 for acc[i*4 + 1 / 3]
        sd[t * SM::D_PITCH + wr] = __float2bfloat16_rn(fin[i * 4 + 0] * a_scale);
        sd[(t + 1) * SM::D_PITCH + wr] = __float2bfloat16_rn(fin[i * 4 + 1] * a_scale);
        sd[t * SM::D_PITCH + wr + 8] = __float2bfloat16_rn(fin[i * 4 + 2] * a_scale);
        sd[(t + 1) * SM::D_PITCH + wr + 8] = __float2bfloat16_rn(fin[i * 4 + 3] * a_scale);
      }
      __syncwarp();
      if (lane == 0) d_full(buf)->arrive();
      buf ^= 1;
      dphase ^= (buf == 0);
    }
  }
#endif
}

}  // namespace k3amd_w4a8dg
