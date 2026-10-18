// k3amd W4A8 grouped GEMM for SM90 (GLM-5.3 W4AFP8 prefill MoE; k3amd, operator kernel program 2026-10-05).
//
// D[t, n] = a_scale * sum_g S[e, g, n] * sum_{k in g} A[t, k] * W[e, n, k]          (g = 128-wide K groups)
//   A  fp8 e4m3 [M_total, K], rows grouped by expert (offs[e] .. offs[e+1]), one per-tensor scale (cutlass_w4a8_moe)
//   W  int8 [E, N, K/2]: two signed int4 per byte along K, low nibble = even k (W4AFp8MoEMethod), exact in e4m3
//   S  bf16 [E, K/128, N]: the group-128 weight scales (alignment-1 interleave = checkpoint scales transposed)
//   D  bf16 [M_total, N]
//
// Structure (DeepGEMM sm90_fp8_gemm_1d2d.cuh's warp-specialized persistent design, adapted):
//   - 512 threads: warpgroups 0-1 = math (each owns 64 of the BLOCK_M token rows, WGMMA SS m64nBLOCK_Nk32), warpgroups
//     2-3 = producers: warp 8 lane 0 issues TMA (A tile 128B-swizzled, raw int4 W tile, the K block's bf16 scales),
//     warps 9-15 CONVERT the int4 tile into an FP8 tile in shared memory with the 128B swizzle the WGMMA descriptor
//     reads (v1 measured: 3 converter warps + scales loaded by the converters from global = the bottleneck).
//   - per 128-K block: 4 WGMMAs into a fresh accumulator, then promotion final += accum * S[col] (per weight row =
//     per accumulator column; the per-tensor activation scale is applied once in the epilogue).
//   - tiles: (expert, n-block, m-block) with the m-block innermost, so concurrent CTAs share an expert's W tile in L2;
//     tile ids map to experts through a per-expert prefix sum (device, no host sync, CUDA-graph safe); rows past the
//     expert's last token are computed but never stored.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/cluster_sm90.hpp>

namespace k3amd_w4a8dg {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

__device__ __forceinline__ void wg_arrive() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r)::"memory"); }

__device__ __forceinline__ uint64_t smem_desc(const void* p) {
  // K-major, 128B swizzle (layout type 1), LBO 0, SBO 1024 B (8 rows x 128 B): one 128-wide FP8 K block per row
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = static_cast<uint32_t>(__cvta_generic_to_shared(p)) >> 4;
  d.bitfield.layout_type_ = 1;
  d.bitfield.leading_byte_offset_ = 0;
  d.bitfield.stride_byte_offset_ = 1024 >> 4;
  d.bitfield.base_offset_ = 0;
  return d.desc_;
}

template <int N> struct MMA;
template <> struct MMA<128> {
  using Op = cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>;
  static constexpr int kAcc = 64;
  template <size_t... I>
  __device__ static void call(uint64_t a, uint64_t b, float* d, bool acc, cute::index_sequence<I...>) {
    Op::fma(a, b, d[I]..., acc ? cute::SM90::GMMA::ScaleOut::One : cute::SM90::GMMA::ScaleOut::Zero);
  }
  __device__ static void fma(uint64_t a, uint64_t b, float* d, bool acc) { call(a, b, d, acc, cute::make_index_sequence<kAcc>{}); }
};
template <> struct MMA<256> {
  using Op = cute::SM90::GMMA::MMA_64x256x32_F32E4M3E4M3_SS_TN<>;
  static constexpr int kAcc = 128;
  template <size_t... I>
  __device__ static void call(uint64_t a, uint64_t b, float* d, bool acc, cute::index_sequence<I...>) {
    Op::fma(a, b, d[I]..., acc ? cute::SM90::GMMA::ScaleOut::One : cute::SM90::GMMA::ScaleOut::Zero);
  }
  __device__ static void fma(uint64_t a, uint64_t b, float* d, bool acc) { call(a, b, d, acc, cute::make_index_sequence<kAcc>{}); }
};

// 4 packed signed int4 (16 bits, low nibble first = k order) -> 4 e4m3 bytes: CUTLASS's LUT converter
// (numeric_conversion.h NumericArrayConverter<float_e4m3_t, int4b_t>): one PRMT into the positive table, one into the
// negative table, one PRMT picking per byte by the sign bit. Exact: -8..7 are e4m3 values.
__device__ __forceinline__ uint32_t cvt4(uint32_t idx, uint32_t sign) {
  uint32_t r;
  asm volatile(
      "{\n"
      "  .reg .b32 pos_f8s, neg_f8s;\n"
      "  prmt.b32 pos_f8s, %1, %2, %5;\n"
      "  prmt.b32 neg_f8s, %3, %4, %5;\n"
      "  prmt.b32 %0, pos_f8s, neg_f8s, %6;\n"
      "}\n"
      : "=r"(r)
      : "n"(0x44403800), "n"(0x4E4C4A48), "n"(0xCACCCED0), "n"(0xB8C0C4C8), "r"(idx), "r"(0x3210u | sign));
  return r;
}

// one 32-bit word = 8 int4 (k order) -> 8 e4m3 bytes (two words, k order)
__device__ __forceinline__ void cvt8(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t sign = (w & 0x88888888u) >> 1;
  const uint32_t idx = w & 0x77777777u;
  lo = cvt4(idx, sign & 0xFFFFu);
  hi = cvt4(idx >> 16, sign >> 16);
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kStages>
struct Smem {
  static constexpr uint32_t A = BLOCK_M * 128;          // fp8, 128B swizzled (TMA)
  static constexpr uint32_t W4 = BLOCK_N * 64;          // raw int4 pairs (TMA, no swizzle)
  static constexpr uint32_t W8 = BLOCK_N * 128;         // fp8, 128B swizzled (converters)
  static constexpr uint32_t SF = 1024;                  // bf16 scales of the stage's K group (TMA; padded to 1024)
  static constexpr uint32_t STAGE = A + W4 + W8 + SF;   // multiples of 1024 keep A / W8 1024-aligned
  static constexpr uint32_t D_PITCH = BLOCK_N + 8;      // bf16 elements per staged output row (bank spread)
  static constexpr uint32_t D = BLOCK_M * D_PITCH * 2;
  static constexpr uint32_t TOTAL = kStages * STAGE + D + 3 * kStages * 8 + 1024;
  static_assert(A % 1024 == 0 && W4 % 1024 == 0 && W8 % 1024 == 0 && STAGE % 1024 == 0 && BLOCK_N * 2 <= SF,
                "alignment");
  static_assert(TOTAL <= 232448, "shared memory");
};

// tile -> (expert, n block, m block), m block innermost; tile_start[e] = sum over experts < e of n_blocks * m_blocks(e)
__device__ __forceinline__ bool tile_coords(uint32_t tile, const int32_t* tile_start, uint32_t E, uint32_t n_blocks,
                                            const int32_t* offs, uint32_t block_m, uint32_t& e, uint32_t& nb,
                                            uint32_t& mb, uint32_t& m_rows) {
  if (tile >= (uint32_t)tile_start[E]) return false;
  uint32_t lo = 0, hi = E;
  while (hi - lo > 1) {
    uint32_t mid = (lo + hi) >> 1;
    if ((uint32_t)tile_start[mid] <= tile) lo = mid; else hi = mid;
  }
  e = lo;
  const uint32_t rows = offs[e + 1] - offs[e];
  const uint32_t m_blocks = (rows + block_m - 1) / block_m;
  const uint32_t local = tile - tile_start[e];
  nb = local / m_blocks;
  mb = local % m_blocks;
  m_rows = rows;
  (void)n_blocks;
  return true;
}

// kDoubleAcc: two accumulator sets, so the promotion of K block i overlaps the WGMMAs of K block i + 1 (ptxas C7514
//   serializes the WGMMAs of this variant in v1/v2: kept for attribution only).
// kOffset: math warpgroup 1 starts each tile after warpgroup 0 has issued its first K block, so one warpgroup's
//   promotion runs while the other's WGMMAs occupy the tensor cores (pingpong phase without a second accumulator).
// kProbe != 0 is a timing ablation only (garbage output): attribution of the kernel's time to its parts.
constexpr uint32_t P_NOCVT = 1;    // converters skip the int4 -> FP8 pass
constexpr uint32_t P_NOEPI = 2;    // no epilogue (no output staging or stores)
constexpr uint32_t P_DIRECT = 4;   // math waits on the TMA barrier; converters idle (needs P_NOCVT)
constexpr uint32_t P_NOTMA = 8;    // producer arrives without loading (stale shared memory; pure on-chip rate)
constexpr uint32_t P_NOPROMO = 16; // WGMMAs accumulate straight into the output registers (no per-group scaling)
// kSwap: the converted weight tile is the WGMMA A operand (each math warpgroup owns 64 weight rows) and the token tile
//   is B (N = BLOCK_M tokens). A weight row is then an accumulator ROW, so the promotion needs 2 scales per thread
//   per K block (registers) instead of 32 per-column smem loads; the epilogue stages the tile transposed.
//   (Ablation 18:16: promotion = 425 of the 1,030 us no-conversion time at EP8 w13 16K.)
constexpr uint32_t kThreads = 512, kConvThreads = 224;   // converters: warps 9-15

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kStages, bool kDoubleAcc = false, uint32_t kProbe = 0,
          bool kOffset = false, bool kSwap = false>
__global__ void __launch_bounds__(kThreads, 1)
w4a8dg_kernel(const __grid_constant__ CUtensorMap tmap_a, const __grid_constant__ CUtensorMap tmap_w,
              const __grid_constant__ CUtensorMap tmap_s, const int32_t* __restrict__ offs,
              const int32_t* __restrict__ tile_start, const float* __restrict__ a_scale_ptr,
              __nv_bfloat16* __restrict__ d, uint32_t E, uint32_t N, uint32_t K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using SM = Smem<BLOCK_M, BLOCK_N, kStages>;
  using WGMMA = MMA<kSwap ? BLOCK_M : BLOCK_N>;
  static_assert(BLOCK_M == 128 && BLOCK_N == 128, "two math warpgroups x 64 rows (tokens, or weights with kSwap)");
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  auto st_a = [&](uint32_t s) { return smem + s * SM::STAGE; };
  auto st_w4 = [&](uint32_t s) { return smem + s * SM::STAGE + SM::A; };
  auto st_w8 = [&](uint32_t s) { return smem + s * SM::STAGE + SM::A + SM::W4; };
  auto st_sf = [&](uint32_t s) { return reinterpret_cast<__nv_bfloat16*>(smem + s * SM::STAGE + SM::A + SM::W4 + SM::W8); };
  __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem + kStages * SM::STAGE);
  Barrier* bars = reinterpret_cast<Barrier*>(smem + kStages * SM::STAGE + SM::D);
  auto full_tma = [&](uint32_t s) { return bars + s; };
  auto full_w8 = [&](uint32_t s) { return bars + kStages + s; };
  auto empty = [&](uint32_t s) { return bars + 2 * kStages + s; };

  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t k_blocks = K / 128, n_blocks = N / BLOCK_N;
  if (threadIdx.x == 256) {
    cute::prefetch_tma_descriptor(&tmap_a);
    cute::prefetch_tma_descriptor(&tmap_w);
    cute::prefetch_tma_descriptor(&tmap_s);
#pragma unroll
    for (uint32_t s = 0; s < kStages; ++s) {
      full_tma(s)->init(1);
      full_w8(s)->init(kConvThreads / 32);   // the converter warps
      empty(s)->init(8);               // the eight math warps
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  uint32_t stage = 0, phase = 0;
  auto advance = [&]() { stage = (stage + 1 == kStages) ? 0 : stage + 1; phase ^= (stage == 0); };

  if (warp >= 8) {
    cutlass::arch::warpgroup_reg_dealloc<48>();
    if (warp == 8) {
      if (lane == 0) {
        uint32_t e, nb, mb, rows;
        for (uint32_t tile = blockIdx.x; tile_coords(tile, tile_start, E, n_blocks, offs, BLOCK_M, e, nb, mb, rows);
             tile += gridDim.x) {
          const int32_t row0 = offs[e] + (int32_t)(mb * BLOCK_M);
          const int32_t wrow0 = (int32_t)(e * N + nb * BLOCK_N);
          for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            empty(stage)->wait(phase ^ 1);
            Barrier& fb = *full_tma(stage);
            if constexpr ((kProbe & P_NOTMA) != 0) {
              fb.arrive();
              advance();
              continue;
            }
            cute::SM90_TMA_LOAD_2D::copy(&tmap_a, reinterpret_cast<uint64_t*>(&fb), 0ull, st_a(stage),
                                         (int32_t)(kb * 128), row0);
            cute::SM90_TMA_LOAD_2D::copy(&tmap_w, reinterpret_cast<uint64_t*>(&fb), 0ull, st_w4(stage),
                                         (int32_t)(kb * 64), wrow0);
            cute::SM90_TMA_LOAD_2D::copy(&tmap_s, reinterpret_cast<uint64_t*>(&fb), 0ull, st_sf(stage),
                                         (int32_t)(nb * BLOCK_N * 2), (int32_t)(e * k_blocks + kb));
            fb.arrive_and_expect_tx(SM::A + SM::W4 + BLOCK_N * 2);
            advance();
          }
        }
      }
    } else {
      // converters: int4 tile -> 128B-swizzled fp8 tile. Items = BLOCK_N rows x 4 chunks of 16 input bytes (32 k);
      // each thread loads all of its items first, then converts and stores (ILP over items).
      const uint32_t ct = threadIdx.x - 288;   // 0 .. kConvThreads - 1
      constexpr uint32_t kItems = BLOCK_N * 4, kPer = (kItems + kConvThreads - 1) / kConvThreads;
      uint32_t e, nb, mb, rows;
      for (uint32_t tile = blockIdx.x;
           (kProbe & P_DIRECT) == 0 && tile_coords(tile, tile_start, E, n_blocks, offs, BLOCK_M, e, nb, mb, rows);
           tile += gridDim.x) {
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
          full_tma(stage)->wait(phase);
          if constexpr ((kProbe & P_NOCVT) == 0) {
            const uint8_t* w4 = st_w4(stage);
            uint8_t* w8 = st_w8(stage);
            uint4 in[kPer];
#pragma unroll
            for (uint32_t j = 0; j < kPer; ++j) {
              const uint32_t item = ct + j * kConvThreads;
              if (item < kItems) in[j] = *reinterpret_cast<const uint4*>(w4 + item * 16);   // row item/4, chunk item%4
            }
#pragma unroll
            for (uint32_t j = 0; j < kPer; ++j) {
              const uint32_t item = ct + j * kConvThreads;
              if (item < kItems) {
                const uint32_t r = item >> 2, q = item & 3;
                uint32_t o[8];
                cvt8(in[j].x, o[0], o[1]);
                cvt8(in[j].y, o[2], o[3]);
                cvt8(in[j].z, o[4], o[5]);
                cvt8(in[j].w, o[6], o[7]);
                // output k [32 q, 32 q + 32) = logical 16-byte chunks 2q, 2q + 1 of row r, 128B-swizzled
                const uint32_t c0 = (2 * q) ^ (r & 7), c1 = (2 * q + 1) ^ (r & 7);
                *reinterpret_cast<uint4*>(w8 + r * 128 + c0 * 16) = make_uint4(o[0], o[1], o[2], o[3]);
                *reinterpret_cast<uint4*>(w8 + r * 128 + c1 * 16) = make_uint4(o[4], o[5], o[6], o[7]);
              }
            }
          }
          cutlass::arch::fence_view_async_shared();  // generic-proxy writes -> visible to the WGMMA (async proxy)
          __syncwarp();
          if (lane == 0) full_w8(stage)->arrive();
          advance();
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<208>();
    const uint32_t wg = threadIdx.x / 128;
    const float a_scale = *a_scale_ptr;
    float acc0[WGMMA::kAcc], acc1[kDoubleAcc ? WGMMA::kAcc : 1], fin[WGMMA::kAcc];
    // WGMMAs of one K block (the stage's FP8 token rows of this warpgroup x the stage's FP8 weight tile)
    auto issue = [&](float* acc, uint32_t s) {
      const uint64_t a0 = kSwap ? smem_desc(st_w8(s) + wg * 64 * 128) : smem_desc(st_a(s) + wg * 64 * 128);
      const uint64_t b0 = kSwap ? smem_desc(st_a(s)) : smem_desc(st_w8(s));
#pragma unroll
      for (int i = 0; i < WGMMA::kAcc; ++i) fence_operand(acc[i]);
      wg_arrive();
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k) WGMMA::fma(a0 + (k * 32) / 16, b0 + (k * 32) / 16, acc, k > 0);
      wg_commit();
    };
    // promotion: accumulator column c is weight row nb * BLOCK_N + c; its group scale sits in the stage's smem
    auto promote = [&](float* acc, uint32_t s) {
#pragma unroll
      for (int i = 0; i < WGMMA::kAcc; ++i) fence_operand(acc[i]);
      const __nv_bfloat16* sf = st_sf(s);
      if constexpr (kSwap) {
        // accumulator rows = weight rows wg*64 + (warp%4)*16 + lane/4 and +8
        const uint32_t wr = wg * 64 + (warp % 4) * 16 + lane / 4;
        const float s0 = __bfloat162float(sf[wr]), s1 = __bfloat162float(sf[wr + 8]);
#pragma unroll
        for (int i = 0; i < WGMMA::kAcc / 4; ++i) {
          fin[i * 4 + 0] += acc[i * 4 + 0] * s0;
          fin[i * 4 + 1] += acc[i * 4 + 1] * s0;
          fin[i * 4 + 2] += acc[i * 4 + 2] * s1;
          fin[i * 4 + 3] += acc[i * 4 + 3] * s1;
        }
      } else {
#pragma unroll
        for (int i = 0; i < WGMMA::kAcc / 4; ++i) {
          const uint32_t c = i * 8 + (lane & 3) * 2;
          const float2 s2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sf + c));
          fin[i * 4 + 0] += acc[i * 4 + 0] * s2.x;
          fin[i * 4 + 1] += acc[i * 4 + 1] * s2.y;
          fin[i * 4 + 2] += acc[i * 4 + 2] * s2.x;
          fin[i * 4 + 3] += acc[i * 4 + 3] * s2.y;
        }
      }
      __syncwarp();
      if (lane == 0) empty(s)->arrive();
    };
    uint32_t e, nb, mb, rows;
    for (uint32_t tile = blockIdx.x; tile_coords(tile, tile_start, E, n_blocks, offs, BLOCK_M, e, nb, mb, rows);
         tile += gridDim.x) {
#pragma unroll
      for (int i = 0; i < WGMMA::kAcc; ++i) fin[i] = 0.f;
      if constexpr (kDoubleAcc) {
        full_w8(stage)->wait(phase);
        issue(acc0, stage);
        uint32_t held = stage;          // stage of the K block in flight
        advance();
        for (uint32_t kb = 0; kb < k_blocks; kb += 2) {
          // acc0 holds K block kb
          if (kb + 1 < k_blocks) {
            full_w8(stage)->wait(phase);
            issue(acc1, stage);
            wg_wait<1>();
          } else {
            wg_wait<0>();
          }
          promote(acc0, held);
          if (kb + 1 >= k_blocks) break;
          held = stage;
          advance();
          // acc1 holds K block kb + 1
          if (kb + 2 < k_blocks) {
            full_w8(stage)->wait(phase);
            issue(acc0, stage);
            wg_wait<1>();
          } else {
            wg_wait<0>();
          }
          promote(acc1, held);
          if (kb + 2 >= k_blocks) break;
          held = stage;
          advance();
        }
      } else if constexpr ((kProbe & P_NOPROMO) != 0) {
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
          if constexpr ((kProbe & P_DIRECT) != 0) full_tma(stage)->wait(phase); else full_w8(stage)->wait(phase);
          const uint64_t a0 = kSwap ? smem_desc(st_w8(stage) + wg * 64 * 128) : smem_desc(st_a(stage) + wg * 64 * 128);
          const uint64_t b0 = kSwap ? smem_desc(st_a(stage)) : smem_desc(st_w8(stage));
#pragma unroll
          for (int i = 0; i < WGMMA::kAcc; ++i) fence_operand(fin[i]);
          wg_arrive();
#pragma unroll
          for (uint32_t k = 0; k < 4; ++k) WGMMA::fma(a0 + (k * 32) / 16, b0 + (k * 32) / 16, fin, k > 0 || kb > 0);
          wg_commit();
          wg_wait<0>();
#pragma unroll
          for (int i = 0; i < WGMMA::kAcc; ++i) fence_operand(fin[i]);
          __syncwarp();
          if (lane == 0) empty(stage)->arrive();
          advance();
        }
      } else {
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
          if constexpr ((kProbe & P_DIRECT) != 0) full_tma(stage)->wait(phase); else full_w8(stage)->wait(phase);
          if constexpr (kOffset) {
            if (kb == 0 && wg == 1) cutlass::arch::NamedBarrier::sync(256, 2);
          }
          issue(acc0, stage);
          if constexpr (kOffset) {
            if (kb == 0 && wg == 0) cutlass::arch::NamedBarrier::arrive(256, 2);
          }
          wg_wait<0>();
          promote(acc0, stage);
          advance();
        }
      }
      if constexpr ((kProbe & P_NOEPI) != 0) {
        float x = 0.f;
#pragma unroll
        for (int i = 0; i < WGMMA::kAcc; ++i) x += fin[i];
        if (x == 1234.5f) d[threadIdx.x] = __float2bfloat16(x);   // keeps the mainloop live
        continue;
      }
      // epilogue: stage the tile in shared memory ([token][weight row]), then row-masked 16-byte stores
      const uint32_t r0 = wg * 64 + (warp % 4) * 16 + lane / 4;
      if constexpr (kSwap) {
        // accumulator (row = weight r0 / r0 + 8, column = token c / c + 1) -> smem_d[token][weight]
#pragma unroll
        for (int i = 0; i < WGMMA::kAcc / 4; ++i) {
          const uint32_t c = i * 8 + (lane & 3) * 2;
          smem_d[c * SM::D_PITCH + r0] = __float2bfloat16_rn(fin[i * 4 + 0] * a_scale);
          smem_d[(c + 1) * SM::D_PITCH + r0] = __float2bfloat16_rn(fin[i * 4 + 1] * a_scale);
          smem_d[c * SM::D_PITCH + r0 + 8] = __float2bfloat16_rn(fin[i * 4 + 2] * a_scale);
          smem_d[(c + 1) * SM::D_PITCH + r0 + 8] = __float2bfloat16_rn(fin[i * 4 + 3] * a_scale);
        }
      } else {
#pragma unroll
        for (int i = 0; i < WGMMA::kAcc / 4; ++i) {
          const uint32_t c = i * 8 + (lane & 3) * 2;
          *reinterpret_cast<__nv_bfloat162*>(smem_d + r0 * SM::D_PITCH + c) =
              __floats2bfloat162_rn(fin[i * 4 + 0] * a_scale, fin[i * 4 + 1] * a_scale);
          *reinterpret_cast<__nv_bfloat162*>(smem_d + (r0 + 8) * SM::D_PITCH + c) =
              __floats2bfloat162_rn(fin[i * 4 + 2] * a_scale, fin[i * 4 + 3] * a_scale);
        }
      }
      cutlass::arch::NamedBarrier::sync(256, 1);
      const uint32_t m_off = mb * BLOCK_M;
      const uint32_t valid = (rows - m_off) < BLOCK_M ? (rows - m_off) : BLOCK_M;
      __nv_bfloat16* dst = d + ((size_t)offs[e] + m_off) * N + nb * BLOCK_N;
      constexpr uint32_t kChunks = BLOCK_N / 8;      // 16-byte chunks per row
      for (uint32_t idx = threadIdx.x; idx < valid * kChunks; idx += 256) {
        const uint32_t r = idx / kChunks, cch = idx % kChunks;
        *reinterpret_cast<uint4*>(dst + (size_t)r * N + cch * 8) =
            *reinterpret_cast<const uint4*>(smem_d + r * SM::D_PITCH + cch * 8);
      }
      cutlass::arch::NamedBarrier::sync(256, 1);
    }
  }
#endif
}

}  // namespace k3amd_w4a8dg
