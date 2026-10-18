// rtx_w4_decode: decode-regime W4A16 grouped MoE GEMMs for Hopper (rtxsys, decode-MoE owner, 2026-10-05).
//
// Why a new kernel (measured, research/gpu-evidence/rtxsys/moe_decode_bench_h200*.json):
// - Served Marlin MoE at GLM-5.3-Flash TP4 m480 streams its weights at 56% (gemm1) / 46% (gemm2) of HBM.
// - Its issue slots are 66% busy (~4.2 thread-instructions per FP4 weight).
// - It re-streams an expert once per extra 16-token block.
// - Its tile-config / CTAs-per-SM knobs buy <= 5%.
//
// Design:
// - swap AB. The expert weights are mma.sync's A operand (m16 x k16, row-major fragments built in registers); the
//   expert's tokens are the B operand (k16 x n8, staged in shared memory, ldmatrix).
// - Fragment-major weight layout [E][R/16][K/64][lane][16 B]. One coalesced LDG.128 per lane per 64 k per row tile
//   gives the lane exactly its 4 k16 A fragments. Weights go straight to registers (no shared-memory round trip),
//   prefetched one step ahead.
// - One weight pass per expert for up to 32 tokens (4 n8 tiles; more tokens re-stream, rare at decode).
// - gemm1 epilogue: silu(gate) * up, with the optional swiglu clamp (fused_marlin_moe.swiglu_limit_func), written
//   straight to the [pairs, N] activation. Each warp owns 32 gate rows and the matching 32 up rows.
// - gemm2 epilogue: topk weight * expert factor, written to [pairs, H] rows; a valid-pair sum kernel reduces top-k.
// - Routing prep in one CTA: per-expert counts, runs, pair ids sorted by expert. No host sync, CUDA-graph safe.
//
// Formats:
// - NVFP4: e2m1 codes. Group-16 e4m3 scales per expert, normalised by 2^k_e (no subnormals). Per-expert fp32 factor
//   = global * 2^(7 - k_e).
//   code -> bf16 bit trick: value * 2^-126 (subnormal-aligned). Scale byte -> bf16 = s' * 2^119. Product = value *
//   s' * 2^-7.
// - U4B8: uint4 with bias 8, group-128 bf16 scales. (bf16(128 + q) - 136) * s: exact subtract, one rounding.
#pragma once
// Device code only (no torch): compiled by rtx_w4_decode.cu (torch extension) and, for the CPU-side resource check,
// by NVRTC + ptxas (compile_check.py).
#ifdef __CUDACC_RTC__
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned long long uintptr_t;
#else
#include <stdint.h>
#endif
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace rtxw4 {

constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int NTMAX = 4;             // n8 token tiles per pass
constexpr int TMAX = NTMAX * 8;      // tokens per pass
constexpr int KC = 256;              // k per token-staging chunk
constexpr int KSTEP = 64;            // k per weight step
constexpr int SPC = KC / KSTEP;      // steps per chunk (4)
constexpr int TROW = KC + 8;         // padded shared-memory token row (bf16): 528 B -> conflict-free ldmatrix

enum { FMT_NVFP4 = 0, FMT_U4B8 = 1 };

struct Params {
  const uint4* W;          // [E][R/16][K/64][32] 16-B words
  const void* S;           // NVFP4: [E][R/16][K/64][8] uint2 ; U4B8: [E][R/16][K/128][8] uint32 (bf16x2); per row group
  const float* factor;     // NVFP4 [E]
  const __nv_bfloat16* X;  // [rows_in][K]
  const int* pair_ids;     // [P] sorted by expert
  const int4* runs;        // [E] (expert, start, count, 0)
  const int* num_runs;     // device scalar
  __nv_bfloat16* out;      // SILU: [P_total][N] ; else [P_total][R]
  const float* topk_w;     // [P_total] (non-SILU)
  int K, R, N, top_k, groups;
  float clamp;             // swiglu limit; <= 0: none
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t hmul2_u(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t hfma2_u(uint32_t a, uint32_t b, uint32_t c) {
  __nv_bfloat162 r = __hfma2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b),
                             *reinterpret_cast<__nv_bfloat162*>(&c));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t lo2(uint32_t x) { return (x & 0xFFFFu) | (x << 16); }
__device__ __forceinline__ uint32_t hi2(uint32_t x) { return (x & 0xFFFF0000u) | (x >> 16); }

// One 32-bit word = the lane's 8 codes of one k16 sub-step -> its 4 A registers (bf16x2), scales applied.
// NVFP4 word: a_p.lo at bits [12-4p, 16-4p), a_p.hi at [28-4p, 32-4p). sc = bf16x2 (s_r0, s_r1).
// U4B8 word: a_p.lo at bits [4p, 4p+4), a_p.hi at [16+4p, 20+4p). sc = (s_r0, s_r1).
template <int FMT>
__device__ __forceinline__ void dequant(uint32_t q, uint32_t sc, uint32_t (&a)[4]) {
  const uint32_t s0 = lo2(sc), s1 = hi2(sc);
  if constexpr (FMT == FMT_NVFP4) {
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const uint32_t qq = q << (4 * p);
      const uint32_t v = (qq & 0x80008000u) | ((qq & 0x70007000u) >> 6);
      a[p] = hmul2_u(v, (p & 1) ? s1 : s0);
    }
  } else {
    // bf16(128 + q) - 136 is exact (a small integer); one rounding at the scale multiply, as in Marlin's u4b8 path.
    // (A single fma with a bf16 -136*s addend would round that addend: up to 0.27 s of error.)
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const uint32_t v = ((q >> (4 * p)) & 0x000F000Fu) | 0x43004300u;
      const __nv_bfloat162 d = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v),
                                       __halves2bfloat162(__float2bfloat16(136.f), __float2bfloat16(136.f)));
      a[p] = hmul2_u(*reinterpret_cast<const uint32_t*>(&d), (p & 1) ? s1 : s0);
    }
  }
}

// NVFP4 scale word for sub-step s from the lane's uint2 {x: [s0r0, s1r0, s0r1, s1r1], y: [s2r0, s3r0, s2r1, s3r1]}.
__device__ __forceinline__ uint32_t nvfp4_scale(uint2 sw, int s) {
  uint32_t w = (s < 2) ? sw.x : sw.y;
  if (s & 1) w >>= 8;
  return ((w & 0x007F007Fu) << 4) + 0x77807780u;
}

// Per-warp tile set: SILU -> TPW/2 gate tiles + the matching TPW/2 up tiles (16*TPW/2 features per warp);
// otherwise TPW consecutive 16-row tiles. Weights are double-buffered two steps (128 k) at a time in two statically
// named register sets, so the prefetch of stage s+2 is in flight during stage s+1 and nothing is copied.
template <int FMT, bool SILU, int TPW, int MINB>
__global__ void __launch_bounds__(THREADS, MINB) w4_kernel(Params p) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  constexpr int RPW = SILU ? 16 * (TPW / 2) : 16 * TPW;   // output features (SILU) / rows per warp
  constexpr int EP_ROW = RPW + 8;                          // padded epilogue row (bf16)
  __nv_bfloat16* tok = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [2][TMAX][TROW]
  __nv_bfloat16* epi = tok + 2 * TMAX * TROW;                        // [WARPS][TMAX][EP_ROW]
  __shared__ int s_rows[TMAX];
  __shared__ int s_pairs[TMAX];
  __shared__ float s_tw[TMAX];

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int total = (*p.num_runs) * p.groups;
  const int ksteps = p.K / KSTEP;
  const int kchunks = p.K / KC;
  const int RT = p.R / 16;
  __nv_bfloat16* my_epi = epi + warp * TMAX * EP_ROW;

  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int run = item / p.groups, grp = item % p.groups;
    const int4 ri = p.runs[run];
    const int e = ri.x, start = ri.y, count = ri.z;
    const int f0 = (grp * WARPS + warp) * RPW;
    int rt[TPW];
    if constexpr (SILU) {
#pragma unroll
      for (int i = 0; i < TPW / 2; i++) {
        rt[i] = f0 / 16 + i;
        rt[TPW / 2 + i] = (p.N + f0) / 16 + i;
      }
    } else {
#pragma unroll
      for (int i = 0; i < TPW; i++) rt[i] = f0 / 16 + i;
    }
    const uint4* wb[TPW];
    const uint2* sb2[TPW];
    const uint32_t* sb4[TPW];
#pragma unroll
    for (int i = 0; i < TPW; i++) {
      const size_t tile = (size_t)e * RT + rt[i];
      wb[i] = p.W + tile * ksteps * 32 + lane;
      if constexpr (FMT == FMT_NVFP4) sb2[i] = reinterpret_cast<const uint2*>(p.S) + tile * ksteps * 8 + g;
      else sb4[i] = reinterpret_cast<const uint32_t*>(p.S) + tile * (p.K / 128) * 8 + g;
    }
    const float fac = (FMT == FMT_NVFP4) ? p.factor[e] : 1.0f;

    for (int t0 = 0; t0 < count; t0 += TMAX) {
      const int tc = min(TMAX, count - t0);
      const int ntiles = (tc + 7) >> 3;
      __syncthreads();
      if (threadIdx.x < TMAX) {
        const int j = threadIdx.x;
        const int pr = j < tc ? p.pair_ids[start + t0 + j] : -1;
        s_pairs[j] = pr;
        s_rows[j] = pr < 0 ? -1 : (SILU ? pr / p.top_k : pr);
        if constexpr (!SILU) s_tw[j] = pr < 0 ? 0.f : p.topk_w[pr];
      }
      __syncthreads();

      auto stage_tokens = [&](int buf, int chunk) {
        __nv_bfloat16* dst = tok + buf * TMAX * TROW;
#pragma unroll
        for (int it = 0; it < (TMAX * KC / 8) / THREADS; it++) {
          const int idx = threadIdx.x + it * THREADS;
          const int row = idx / (KC / 8), seg = idx % (KC / 8);
          const int xr = s_rows[row];
          const __nv_bfloat16* src = p.X + (size_t)(xr < 0 ? 0 : xr) * p.K + chunk * KC + seg * 8;
          cp_async16(dst + row * TROW + seg * 8, src, xr >= 0);
        }
      };

      float acc[TPW][NTMAX][4];
#pragma unroll
      for (int i = 0; i < TPW; i++)
#pragma unroll
        for (int n = 0; n < NTMAX; n++)
#pragma unroll
          for (int j = 0; j < 4; j++) acc[i][n][j] = 0.f;

      // two register stages (A: even 128-k stages, B: odd), each holding 2 steps x TPW tiles
      uint4 wA[2][TPW], wB[2][TPW];
      uint2 sA2[2][TPW], sB2[2][TPW];
      uint32_t sA4[2][TPW], sB4[2][TPW];
      auto load_stage = [&](uint4 (&w)[2][TPW], uint2 (&s2)[2][TPW], uint32_t (&s4)[2][TPW], int st) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int ks = 2 * st + h;
          if (ks < ksteps) {
#pragma unroll
            for (int i = 0; i < TPW; i++) {
              w[h][i] = __ldg(wb[i] + (size_t)ks * 32);
              if constexpr (FMT == FMT_NVFP4) s2[h][i] = __ldg(sb2[i] + (size_t)ks * 8);
              else s4[h][i] = __ldg(sb4[i] + (size_t)(ks >> 1) * 8);
            }
          }
        }
      };
      auto compute_step = [&](const uint4 (&w)[TPW], const uint2 (&s2)[TPW], const uint32_t (&s4)[TPW],
                              const __nv_bfloat16* tb, int ksl) {
        // scales for the 4 k16 sub-steps of every tile, decoded once per step
        uint32_t sc[TPW][4];
#pragma unroll
        for (int i = 0; i < TPW; i++)
#pragma unroll
          for (int s = 0; s < 4; s++) sc[i][s] = (FMT == FMT_NVFP4) ? nvfp4_scale(s2[i], s) : s4[i];
#pragma unroll
        for (int s = 0; s < 4; s++) {
          const int koff = ksl * KSTEP + s * 16;
          uint32_t b[NTMAX][2];
#pragma unroll
          for (int np = 0; np < NTMAX / 2; np++) {
            if (np * 2 < ntiles) {
              const int m = lane >> 3;
              const int row = np * 16 + (m >> 1) * 8 + (lane & 7);
              const int col = koff + (m & 1) * 8;
              uint32_t r[4];
              ldsm_x4(r, tb + row * TROW + col);
              b[2 * np][0] = r[0];
              b[2 * np][1] = r[1];
              b[2 * np + 1][0] = r[2];
              b[2 * np + 1][1] = r[3];
            } else {
              b[2 * np][0] = b[2 * np][1] = b[2 * np + 1][0] = b[2 * np + 1][1] = 0u;
            }
          }
#pragma unroll
          for (int i = 0; i < TPW; i++) {
            const uint32_t q = (s == 0) ? w[i].x : (s == 1) ? w[i].y : (s == 2) ? w[i].z : w[i].w;
            uint32_t a[4];
            dequant<FMT>(q, sc[i][s], a);
#pragma unroll
            for (int n = 0; n < NTMAX; n++)
              if (n < ntiles) mma_bf16(acc[i][n], a, b[n][0], b[n][1]);
          }
        }
      };

      stage_tokens(0, 0);
      cp_async_commit();
      load_stage(wA, sA2, sA4, 0);
      load_stage(wB, sB2, sB4, 1);

      for (int c = 0; c < kchunks; c++) {   // KC = 256 = 2 register stages
        if (c + 1 < kchunks) stage_tokens((c + 1) & 1, c + 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const __nv_bfloat16* tb = tok + (c & 1) * TMAX * TROW;
        compute_step(wA[0], sA2[0], sA4[0], tb, 0);
        compute_step(wA[1], sA2[1], sA4[1], tb, 1);
        load_stage(wA, sA2, sA4, 2 * c + 2);
        compute_step(wB[0], sB2[0], sB4[0], tb, 2);
        compute_step(wB[1], sB2[1], sB4[1], tb, 3);
        load_stage(wB, sB2, sB4, 2 * c + 3);
        __syncthreads();
      }

      // ---- epilogue: fragments -> padded shared tile [token][feature] -> coalesced 16-B row stores
      if constexpr (SILU) {
#pragma unroll
        for (int i = 0; i < TPW / 2; i++)
#pragma unroll
          for (int n = 0; n < NTMAX; n++) {
            if (n < ntiles) {
#pragma unroll
              for (int j = 0; j < 4; j++) {
                const int fr = 16 * i + g + 8 * (j >> 1);
                const int tk = n * 8 + 2 * t + (j & 1);
                float gv = acc[i][n][j] * fac, uv = acc[TPW / 2 + i][n][j] * fac;
                if (p.clamp > 0.f) {
                  gv = fminf(gv, p.clamp);
                  uv = fmaxf(fminf(uv, p.clamp), -p.clamp);
                }
                my_epi[tk * EP_ROW + fr] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
              }
            }
          }
      } else {
#pragma unroll
        for (int i = 0; i < TPW; i++)
#pragma unroll
          for (int n = 0; n < NTMAX; n++) {
            if (n < ntiles) {
#pragma unroll
              for (int j = 0; j < 4; j++) {
                const int fr = 16 * i + g + 8 * (j >> 1);
                const int tk = n * 8 + 2 * t + (j & 1);
                my_epi[tk * EP_ROW + fr] = __float2bfloat16(acc[i][n][j] * fac * s_tw[tk]);
              }
            }
          }
      }
      __syncwarp();
      constexpr int SEG = RPW / 8;                     // 16-B segments per token row
      const int ostride = SILU ? p.N : p.R;
#pragma unroll
      for (int it = 0; it < (TMAX * SEG + 31) / 32; it++) {
        const int idx = lane + it * 32;
        const int tk = idx / SEG, part = idx % SEG;
        if (tk < tc) {
          const int pr = s_pairs[tk];
          *reinterpret_cast<uint4*>(p.out + (size_t)pr * ostride + f0 + part * 8) =
              *reinterpret_cast<const uint4*>(my_epi + tk * EP_ROW + part * 8);
        }
      }
    }
  }
}

// ================================================================================================ bulk-copy ring
// w4b_kernel: w4_kernel with the weight / scale stream moved from registers to a per-warp shared-memory ring filled
// by cp.async.bulk (TMA bulk copies, one mbarrier per slot). Lane 0 runs DEPTH steps ahead of the warp, across item
// and pass boundaries, so the stream never drains and no prefetch registers are held (v2: 128-reg cap spilled
// 96 B; v1.1 ncu: issue 50%, no-eligible 47%).
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes));
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done = 0;
  long long spins = 0;
  while (true) {
    asm volatile(
        "{\n .reg .pred P;\n mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n selp.u32 %0, 1, 0, P;\n}\n"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    if (done) break;
    if (++spins > (1ll << 28)) __trap();   // a lost transaction is a kernel error, never a hang
  }
}

template <int FMT, bool SILU, int TPW, int MINB, int DEPTH>
__global__ void __launch_bounds__(THREADS, MINB) w4b_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  constexpr int RPW = SILU ? 16 * (TPW / 2) : 16 * TPW;
  constexpr int EP_ROW = RPW + 8;
  constexpr int SREC = (FMT == FMT_NVFP4) ? 64 : 32;        // scale bytes per tile per step (one record per g)
  constexpr int TSTRIDE = 512 + SREC;                       // per tile in a slot: 32 lanes x 16 B + scales
  constexpr int SLOT = TPW * TSTRIDE;
  __nv_bfloat16* tok = reinterpret_cast<__nv_bfloat16*>(smem_raw);           // [2][TMAX][TROW]
  __nv_bfloat16* epi = tok + 2 * TMAX * TROW;                                 // [WARPS][TMAX][EP_ROW]
  uint8_t* ring = reinterpret_cast<uint8_t*>(epi + WARPS * TMAX * EP_ROW);    // [WARPS][DEPTH][SLOT] (16-B aligned)
  uint64_t* bars = reinterpret_cast<uint64_t*>(ring + WARPS * DEPTH * SLOT);  // [WARPS][DEPTH]
  __shared__ int s_rows[TMAX];
  __shared__ int s_pairs[TMAX];
  __shared__ float s_tw[TMAX];

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int total = (*p.num_runs) * p.groups;
  const int ksteps = p.K / KSTEP;
  const int kchunks = p.K / KC;
  const int RT = p.R / 16;
  __nv_bfloat16* my_epi = epi + warp * TMAX * EP_ROW;
  uint8_t* my_ring = ring + warp * DEPTH * SLOT;
  const uint32_t my_bar0 = smem_u32(bars + warp * DEPTH);

  if (lane == 0) {
#pragma unroll
    for (int d = 0; d < DEPTH; d++) mbar_init(my_bar0 + 8 * d, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  auto warp_tiles = [&](int item, int& e, int& count, int (&rt)[TPW], int& f0) {
    const int run = item / p.groups, grp = item % p.groups;
    const int4 ri = p.runs[run];
    e = ri.x;
    count = ri.z;
    f0 = (grp * WARPS + warp) * RPW;
    if constexpr (SILU) {
#pragma unroll
      for (int i = 0; i < TPW / 2; i++) {
        rt[i] = f0 / 16 + i;
        rt[TPW / 2 + i] = (p.N + f0) / 16 + i;
      }
    } else {
#pragma unroll
      for (int i = 0; i < TPW; i++) rt[i] = f0 / 16 + i;
    }
  };

  // ---- producer (lane 0 only): walks (item, pass, step) DEPTH steps ahead of the consumer
  int pr_item = blockIdx.x, pr_pass = 0, pr_ks = 0, pr_np = 0;
  bool pr_live = false;
  const uint8_t* pr_w[TPW];
  const uint8_t* pr_s[TPW];
  auto pr_setup = [&]() {
    pr_live = pr_item < total;
    if (!pr_live) return;
    int e, count, rt[TPW], f0;
    warp_tiles(pr_item, e, count, rt, f0);
    pr_np = (count + TMAX - 1) / TMAX;
#pragma unroll
    for (int i = 0; i < TPW; i++) {
      const size_t tile = (size_t)e * RT + rt[i];
      pr_w[i] = reinterpret_cast<const uint8_t*>(p.W) + tile * ksteps * 512;
      pr_s[i] = reinterpret_cast<const uint8_t*>(p.S) +
                (FMT == FMT_NVFP4 ? tile * ksteps * 64 : tile * (p.K / 128) * 32);
    }
  };
  uint32_t pr_count = 0;   // steps issued
  auto pr_issue = [&]() {
    if (!pr_live) return;
    const int slot = pr_count % DEPTH;
    const uint32_t bar = my_bar0 + 8 * slot;
    const uint32_t dst = smem_u32(my_ring + slot * SLOT);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    mbar_expect_tx(bar, SLOT);
#pragma unroll
    for (int i = 0; i < TPW; i++) {
      bulk_g2s(dst + i * TSTRIDE, pr_w[i] + (size_t)pr_ks * 512, 512, bar);
      const uint8_t* ssrc = (FMT == FMT_NVFP4) ? pr_s[i] + (size_t)pr_ks * 64 : pr_s[i] + (size_t)(pr_ks >> 1) * 32;
      bulk_g2s(dst + i * TSTRIDE + 512, ssrc, SREC, bar);
    }
    pr_count++;
    if (++pr_ks == ksteps) {
      pr_ks = 0;
      if (++pr_pass == pr_np) {
        pr_pass = 0;
        pr_item += gridDim.x;
        pr_setup();
      }
    }
  };
  if (lane == 0) {
    pr_setup();
    for (int d = 0; d < DEPTH; d++) pr_issue();
  }
  uint32_t cs_count = 0;   // steps consumed

  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    int e, count, rt[TPW], f0;
    warp_tiles(item, e, count, rt, f0);
    const int start = p.runs[item / p.groups].y;
    const float fac = (FMT == FMT_NVFP4) ? p.factor[e] : 1.0f;

    for (int t0 = 0; t0 < count; t0 += TMAX) {
      const int tc = min(TMAX, count - t0);
      const int ntiles = (tc + 7) >> 3;
      __syncthreads();
      if (threadIdx.x < TMAX) {
        const int j = threadIdx.x;
        const int pr = j < tc ? p.pair_ids[start + t0 + j] : -1;
        s_pairs[j] = pr;
        s_rows[j] = pr < 0 ? -1 : (SILU ? pr / p.top_k : pr);
        if constexpr (!SILU) s_tw[j] = pr < 0 ? 0.f : p.topk_w[pr];
      }
      __syncthreads();

      auto stage_tokens = [&](int buf, int chunk) {
        __nv_bfloat16* dst = tok + buf * TMAX * TROW;
#pragma unroll
        for (int it = 0; it < (TMAX * KC / 8) / THREADS; it++) {
          const int idx = threadIdx.x + it * THREADS;
          const int row = idx / (KC / 8), seg = idx % (KC / 8);
          const int xr = s_rows[row];
          const __nv_bfloat16* src = p.X + (size_t)(xr < 0 ? 0 : xr) * p.K + chunk * KC + seg * 8;
          cp_async16(dst + row * TROW + seg * 8, src, xr >= 0);
        }
      };

      float acc[TPW][NTMAX][4];
#pragma unroll
      for (int i = 0; i < TPW; i++)
#pragma unroll
        for (int n = 0; n < NTMAX; n++)
#pragma unroll
          for (int j = 0; j < 4; j++) acc[i][n][j] = 0.f;

      stage_tokens(0, 0);
      cp_async_commit();
      for (int c = 0; c < kchunks; c++) {
        if (c + 1 < kchunks) stage_tokens((c + 1) & 1, c + 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const __nv_bfloat16* tb = tok + (c & 1) * TMAX * TROW;
#pragma unroll
        for (int ksl = 0; ksl < SPC; ksl++) {
          // wait for this step's slot, pull it into registers, hand the slot back to the producer
          const int slot = cs_count % DEPTH;
          mbar_wait(my_bar0 + 8 * slot, (cs_count / DEPTH) & 1);
          const uint8_t* sl = my_ring + slot * SLOT;
          uint4 w[TPW];
          uint32_t sc[TPW][4];
#pragma unroll
          for (int i = 0; i < TPW; i++) {
            w[i] = *reinterpret_cast<const uint4*>(sl + i * TSTRIDE + lane * 16);
            if constexpr (FMT == FMT_NVFP4) {
              const uint2 s2 = *reinterpret_cast<const uint2*>(sl + i * TSTRIDE + 512 + g * 8);
#pragma unroll
              for (int s = 0; s < 4; s++) sc[i][s] = nvfp4_scale(s2, s);
            } else {
              const uint32_t s4 = *reinterpret_cast<const uint32_t*>(sl + i * TSTRIDE + 512 + g * 4);
#pragma unroll
              for (int s = 0; s < 4; s++) sc[i][s] = s4;
            }
          }
          cs_count++;
          __syncwarp();
          if (lane == 0) pr_issue();
#pragma unroll
          for (int s = 0; s < 4; s++) {
            const int koff = ksl * KSTEP + s * 16;
            uint32_t b[NTMAX][2];
#pragma unroll
            for (int np = 0; np < NTMAX / 2; np++) {
              if (np * 2 < ntiles) {
                const int m = lane >> 3;
                const int row = np * 16 + (m >> 1) * 8 + (lane & 7);
                const int col = koff + (m & 1) * 8;
                uint32_t r[4];
                ldsm_x4(r, tb + row * TROW + col);
                b[2 * np][0] = r[0];
                b[2 * np][1] = r[1];
                b[2 * np + 1][0] = r[2];
                b[2 * np + 1][1] = r[3];
              } else {
                b[2 * np][0] = b[2 * np][1] = b[2 * np + 1][0] = b[2 * np + 1][1] = 0u;
              }
            }
#pragma unroll
            for (int i = 0; i < TPW; i++) {
              const uint32_t q = (s == 0) ? w[i].x : (s == 1) ? w[i].y : (s == 2) ? w[i].z : w[i].w;
              uint32_t a[4];
              dequant<FMT>(q, sc[i][s], a);
#pragma unroll
              for (int n = 0; n < NTMAX; n++)
                if (n < ntiles) mma_bf16(acc[i][n], a, b[n][0], b[n][1]);
            }
          }
        }
        __syncthreads();
      }

      if constexpr (SILU) {
#pragma unroll
        for (int i = 0; i < TPW / 2; i++)
#pragma unroll
          for (int n = 0; n < NTMAX; n++) {
            if (n < ntiles) {
#pragma unroll
              for (int j = 0; j < 4; j++) {
                const int fr = 16 * i + g + 8 * (j >> 1);
                const int tk = n * 8 + 2 * t + (j & 1);
                float gv = acc[i][n][j] * fac, uv = acc[TPW / 2 + i][n][j] * fac;
                if (p.clamp > 0.f) {
                  gv = fminf(gv, p.clamp);
                  uv = fmaxf(fminf(uv, p.clamp), -p.clamp);
                }
                my_epi[tk * EP_ROW + fr] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
              }
            }
          }
      } else {
#pragma unroll
        for (int i = 0; i < TPW; i++)
#pragma unroll
          for (int n = 0; n < NTMAX; n++) {
            if (n < ntiles) {
#pragma unroll
              for (int j = 0; j < 4; j++) {
                const int fr = 16 * i + g + 8 * (j >> 1);
                const int tk = n * 8 + 2 * t + (j & 1);
                my_epi[tk * EP_ROW + fr] = __float2bfloat16(acc[i][n][j] * fac * s_tw[tk]);
              }
            }
          }
      }
      __syncwarp();
      constexpr int SEG = RPW / 8;
      const int ostride = SILU ? p.N : p.R;
#pragma unroll
      for (int it = 0; it < (TMAX * SEG + 31) / 32; it++) {
        const int idx = lane + it * 32;
        const int tk = idx / SEG, part = idx % SEG;
        if (tk < tc) {
          const int pr = s_pairs[tk];
          *reinterpret_cast<uint4*>(p.out + (size_t)pr * ostride + f0 + part * 8) =
              *reinterpret_cast<const uint4*>(my_epi + tk * EP_ROW + part * 8);
        }
      }
    }
  }
}

// ================================================================================================ W4A8 (GLM-5.3)
// Reads the resident cutlass_w4a8 bytes directly (one-layout decision with k3amd / amdglm, 2026-10-05):
//   W [E][R][K/2] int8 = two SIGNED int4 per byte along K, low nibble = even k; S bf16 group scales, interleaved
//   by s_align (1: [E][K/128][R], the ep8-tiles layout; 4: stock interleave_scales [E][K/512][4R]);
//   activations FP8 e4m3 with a static per-layer scale (cutlass_w4a8_moe: a1_scale / a2_scale).
// mma.sync m16n8k32 e4m3 -> f32 with the weights as A. A dot product is invariant to a k permutation applied to both
// operands, so lane (g, t) takes 32 consecutive k of rows g and g+8 per 128-k group (two coalesced LDG.128 = 64 B row
// segments per row group) and uses physical k [32t + 8s, +4) / [32t + 8s + 4, +4) as its logical A fragment k
// {4t..4t+3} / {4t+16..4t+19} of step s; the matching B fragment is one LDS.64 at token byte 32t + 8s.
// Group scales are applied by promotion: a per-tile temporary accumulator over the group's 4 steps, then
// acc += temp * scale(row).
constexpr int KC8 = 512;          // k (= bytes) per FP8 token chunk
// Padded row 528 B: a multiple of 16 B, as cp.async's 16-byte destination requires. The 1xH200 bundle faulted with
// "misaligned address" on 520-B rows. 528 B = 33 chunks of 16 B: the B fragments are read as one LDS.128 per two
// steps; the 8 lanes of a 128-bit phase (g in {0,1}, t in 0..3) hit chunk (33 g + 2 t + k) mod 8 = 8 distinct bank
// groups, so the reads are conflict-free.
constexpr int TROW8 = KC8 + 16;

struct ParamsF8 {
  const uint8_t* W;          // [E][R][K/2]
  const __nv_bfloat16* S;    // [E][K/128][R]
  const uint8_t* X;          // e4m3 [rows][K]
  const int* pair_ids;
  const int4* runs;
  const int* num_runs;
  void* out;                 // SILU: e4m3 [P][N] ; else bf16 [P][R]
  const float* topk_w;       // [P] (down)
  const float* a_in;         // device scalar: scale of X
  const float* a_out;        // device scalar: scale used to quantize the SILU output (SILU only)
  int K, R, N, top_k, groups;
  int s_align;               // scale interleave: 1 = [E][K/128][R] (ep8 tiles), 4 = stock interleave_scales [E][K/512][4R]
  float clamp;
};

// 4 signed int4 (one 16-bit chunk, nibble j = k j) -> 4 e4m3 bytes (byte j = k j). Exact LUT: two 8-entry byte
// permutes (0..7 and -8..-1, indexed by the low 3 bits) and a third that picks per byte by the nibble's sign bit.
__device__ __forceinline__ uint32_t s4x4_to_e4m3x4(uint32_t c) {
  const uint32_t sel = c & 0x7777u;
  const uint32_t pos = __byte_perm(0x44403800u, 0x4E4C4A48u, sel);   // 0, 1, 2, 3 | 4, 5, 6, 7
  const uint32_t neg = __byte_perm(0xCACCCED0u, 0xB8C0C4C8u, sel);   // -8, -7, -6, -5 | -4, -3, -2, -1
  return __byte_perm(pos, neg, 0x3210u | ((c >> 1) & 0x4444u));
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint2 lds64(const void* p) {
  uint2 r;
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "r"(a));
  return r;
}

template <bool SILU, int TPW, int MINB>
__global__ void __launch_bounds__(THREADS, MINB) w4f8_kernel(ParamsF8 p) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  constexpr int RPW = SILU ? 16 * (TPW / 2) : 16 * TPW;
  constexpr int EB = SILU ? 1 : 2;                         // epilogue element bytes (e4m3 / bf16)
  constexpr int EP_ROWB = RPW * EB + 16;                   // padded epilogue row (bytes)
  uint8_t* tok = smem_raw;                                 // [2][TMAX][TROW8]
  uint8_t* epi = tok + 2 * TMAX * TROW8;                   // [WARPS][TMAX][EP_ROWB]
  __shared__ int s_rows[TMAX];
  __shared__ int s_pairs[TMAX];
  __shared__ float s_tw[TMAX];

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int total = (*p.num_runs) * p.groups;
  const int kgroups = p.K / 128;
  const int kchunks = p.K / KC8;
  const float a_in = *p.a_in;
  const float inv_out = SILU ? 1.f / *p.a_out : 1.f;
  uint8_t* my_epi = epi + warp * TMAX * EP_ROWB;

  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int run = item / p.groups, grp = item % p.groups;
    const int4 ri = p.runs[run];
    const int e = ri.x, start = ri.y, count = ri.z;
    const int f0 = (grp * WARPS + warp) * RPW;
    // 32-bit offsets from one expert base pointer (expert bytes = R * K/2 <= 12.6 MB on GLM): 64-bit pointers per
    // row cost 16 registers per tile and spilled under the launch bound (ptxas, compile_check.py).
    const uint8_t* wbase = p.W + (size_t)e * p.R * (p.K / 2) + 16 * t;
    const __nv_bfloat16* sbase = p.S + (size_t)e * kgroups * p.R;
    uint32_t wrow[TPW][2];
#pragma unroll
    for (int i = 0; i < TPW; i++) {
      int r0;
      if constexpr (SILU) r0 = (i < TPW / 2) ? (f0 + 16 * i) : (p.N + f0 + 16 * (i - TPW / 2));
      else r0 = f0 + 16 * i;
#pragma unroll
      for (int h = 0; h < 2; h++) wrow[i][h] = static_cast<uint32_t>(r0 + g + 8 * h);
    }

    for (int t0 = 0; t0 < count; t0 += TMAX) {
      const int tc = min(TMAX, count - t0);
      const int ntiles = (tc + 7) >> 3;
      __syncthreads();
      if (threadIdx.x < TMAX) {
        const int j = threadIdx.x;
        const int pr = j < tc ? p.pair_ids[start + t0 + j] : -1;
        s_pairs[j] = pr;
        s_rows[j] = pr < 0 ? -1 : (SILU ? pr / p.top_k : pr);
        if constexpr (!SILU) s_tw[j] = pr < 0 ? 0.f : p.topk_w[pr];
      }
      __syncthreads();

      auto stage_tokens = [&](int buf, int chunk) {
        uint8_t* dst = tok + buf * TMAX * TROW8;
#pragma unroll
        for (int it = 0; it < (TMAX * KC8 / 16) / THREADS; it++) {
          const int idx = threadIdx.x + it * THREADS;
          const int row = idx / (KC8 / 16), seg = idx % (KC8 / 16);
          const int xr = s_rows[row];
          const uint8_t* src = p.X + (size_t)(xr < 0 ? 0 : xr) * p.K + chunk * KC8 + seg * 16;
          cp_async16(dst + row * TROW8 + seg * 16, src, xr >= 0);
        }
      };

      float acc[TPW][NTMAX][4];
#pragma unroll
      for (int i = 0; i < TPW; i++)
#pragma unroll
        for (int n = 0; n < NTMAX; n++)
#pragma unroll
          for (int j = 0; j < 4; j++) acc[i][n][j] = 0.f;

      // group double buffer (A: even 128-k groups, B: odd), loaded after the buffer's group is consumed
      uint4 wA[TPW][2], wB[TPW][2];
      float sA[TPW][2], sB[TPW][2];
      auto load_group = [&](uint4 (&w)[TPW][2], float (&sc)[TPW][2], int kg) {
        if (kg < kgroups) {
#pragma unroll
          for (int i = 0; i < TPW; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
              w[i][h] = __ldg(reinterpret_cast<const uint4*>(wbase + wrow[i][h] * (uint32_t)(p.K / 2) + kg * 64));
              // interleave_scales (w4afp8.py:111-125): [e][kg / al][row * al + kg % al]
              const uint32_t al = static_cast<uint32_t>(p.s_align);
              sc[i][h] = __bfloat162float(
                  sbase[((uint32_t)kg / al) * ((uint32_t)p.R * al) + wrow[i][h] * al + ((uint32_t)kg % al)]);
            }
        }
      };
      auto compute_group = [&](const uint4 (&w)[TPW][2], const float (&sc)[TPW][2], const uint8_t* tb, int kgl) {
#pragma unroll
        for (int i = 0; i < TPW; i++) {
          float tmp[NTMAX][4];
#pragma unroll
          for (int n = 0; n < NTMAX; n++)
#pragma unroll
            for (int j = 0; j < 4; j++) tmp[n][j] = 0.f;
#pragma unroll
          for (int sp = 0; sp < 2; sp++) {         // steps 2sp, 2sp+1: one 16-B B load per n8 tile covers both
            const uint32_t q0a = sp ? w[i][0].z : w[i][0].x, q0b = sp ? w[i][0].w : w[i][0].y;
            const uint32_t q1a = sp ? w[i][1].z : w[i][1].x, q1b = sp ? w[i][1].w : w[i][1].y;
            uint32_t a0[4], a1[4];                 // A fragments of the two steps, kept across the n8 tiles
            a0[0] = s4x4_to_e4m3x4(q0a & 0xFFFFu);  // row g,   logical k 4t..4t+3
            a0[1] = s4x4_to_e4m3x4(q1a & 0xFFFFu);  // row g+8
            a0[2] = s4x4_to_e4m3x4(q0a >> 16);      // row g,   logical k 4t+16..4t+19
            a0[3] = s4x4_to_e4m3x4(q1a >> 16);      // row g+8
            a1[0] = s4x4_to_e4m3x4(q0b & 0xFFFFu);
            a1[1] = s4x4_to_e4m3x4(q1b & 0xFFFFu);
            a1[2] = s4x4_to_e4m3x4(q0b >> 16);
            a1[3] = s4x4_to_e4m3x4(q1b >> 16);
#pragma unroll
            for (int n = 0; n < NTMAX; n++) {
              if (n < ntiles) {
                const uint4 b4 = *reinterpret_cast<const uint4*>(tb + (n * 8 + g) * TROW8 + kgl * 128 + 32 * t + 16 * sp);
                mma_e4m3(tmp[n], a0, b4.x, b4.y);
                mma_e4m3(tmp[n], a1, b4.z, b4.w);
              }
            }
          }
#pragma unroll
          for (int n = 0; n < NTMAX; n++) {
            acc[i][n][0] += tmp[n][0] * sc[i][0];
            acc[i][n][1] += tmp[n][1] * sc[i][0];
            acc[i][n][2] += tmp[n][2] * sc[i][1];
            acc[i][n][3] += tmp[n][3] * sc[i][1];
          }
        }
      };

      stage_tokens(0, 0);
      cp_async_commit();
      load_group(wA, sA, 0);
      load_group(wB, sB, 1);

      for (int c = 0; c < kchunks; c++) {   // KC8 = 512 = 4 groups: A, B, A, B
        if (c + 1 < kchunks) stage_tokens((c + 1) & 1, c + 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const uint8_t* tb = tok + (c & 1) * TMAX * TROW8;
        const int kg0 = 4 * c;
        compute_group(wA, sA, tb, 0);
        load_group(wA, sA, kg0 + 2);
        compute_group(wB, sB, tb, 1);
        load_group(wB, sB, kg0 + 3);
        compute_group(wA, sA, tb, 2);
        load_group(wA, sA, kg0 + 4);
        compute_group(wB, sB, tb, 3);
        load_group(wB, sB, kg0 + 5);
        __syncthreads();
      }

      // ---- epilogue
      if constexpr (SILU) {
#pragma unroll
        for (int i = 0; i < TPW / 2; i++)
#pragma unroll
          for (int n = 0; n < NTMAX; n++) {
            if (n < ntiles) {
#pragma unroll
              for (int j = 0; j < 4; j++) {
                const int fr = 16 * i + g + 8 * (j >> 1);
                const int tk = n * 8 + 2 * t + (j & 1);
                float gv = acc[i][n][j] * a_in, uv = acc[TPW / 2 + i][n][j] * a_in;
                if (p.clamp > 0.f) {
                  gv = fminf(gv, p.clamp);
                  uv = fmaxf(fminf(uv, p.clamp), -p.clamp);
                }
                const float h = gv / (1.f + __expf(-gv)) * uv * inv_out;
                my_epi[tk * EP_ROWB + fr] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(h, __NV_SATFINITE, __NV_E4M3));
              }
            }
          }
      } else {
#pragma unroll
        for (int i = 0; i < TPW; i++)
#pragma unroll
          for (int n = 0; n < NTMAX; n++) {
            if (n < ntiles) {
#pragma unroll
              for (int j = 0; j < 4; j++) {
                const int fr = 16 * i + g + 8 * (j >> 1);
                const int tk = n * 8 + 2 * t + (j & 1);
                reinterpret_cast<__nv_bfloat16*>(my_epi + tk * EP_ROWB)[fr] =
                    __float2bfloat16(acc[i][n][j] * a_in * s_tw[tk]);
              }
            }
          }
      }
      __syncwarp();
      constexpr int SEG = RPW * EB / 16;
      const size_t ostride_b = (size_t)(SILU ? p.N : p.R) * EB;
#pragma unroll
      for (int it = 0; it < (TMAX * SEG + 31) / 32; it++) {
        const int idx = lane + it * 32;
        const int tk = idx / SEG, part = idx % SEG;
        if (tk < tc) {
          const int pr = s_pairs[tk];
          *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(p.out) + (size_t)pr * ostride_b + (size_t)f0 * EB +
                                    part * 16) = *reinterpret_cast<const uint4*>(my_epi + tk * EP_ROWB + part * 16);
        }
      }
    }
  }
}

// x [M][K] bf16 -> e4m3 with a static scale (cutlass_w4a8: gateup_input = fp8(x / a1_scale))
__global__ void quant_e4m3_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ q, const float* scale,
                                  int64_t n8) {
  const float inv = 1.f / *scale;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n8; i += (int64_t)gridDim.x * blockDim.x) {
    const uint4 v = reinterpret_cast<const uint4*>(x)[i];
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint32_t o[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const float2 f0 = __bfloat1622float2(h[2 * j]), f1 = __bfloat1622float2(h[2 * j + 1]);
      const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(f0.x * inv, f0.y * inv), __NV_SATFINITE, __NV_E4M3);
      const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(f1.x * inv, f1.y * inv), __NV_SATFINITE, __NV_E4M3);
      o[j] = (lo & 0xFFFFu) | (hi << 16);
    }
    reinterpret_cast<uint2*>(q)[i] = make_uint2(o[0], o[1]);
  }
}

// ------------------------------------------------------------------------------------------------ routing prep
constexpr int PREP_THREADS = 1024;

__global__ void __launch_bounds__(PREP_THREADS)
    prep_kernel(const int* __restrict__ topk_ids, int P, int E, int* __restrict__ pair_ids, int4* __restrict__ runs,
                int* __restrict__ num_runs) {
  // E <= PREP_THREADS: one expert per thread for the scan. P = M * topk <= 65535 (decode), so a count and a run
  // flag pack into one 32-bit word and a single block scan yields both offsets.
  __shared__ int cnt[PREP_THREADS];
  __shared__ int cur[PREP_THREADS];
  __shared__ int wsum[PREP_THREADS / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < P; i += PREP_THREADS) {
    const int e = topk_ids[i];
    if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int c = tid < E ? cnt[tid] : 0;
  const int own = (c << 16) | (c > 0 ? 1 : 0);
  int v = own;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += n;
  }
  if (lane == 31) wsum[warp] = v;
  __syncthreads();
  if (warp == 0) {
    int w = wsum[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += n;
    }
    wsum[lane] = w;
  }
  __syncthreads();
  const int excl = v - own + (warp > 0 ? wsum[warp - 1] : 0);
  if (tid < E) {
    cur[tid] = excl >> 16;
    if (c > 0) runs[excl & 0xFFFF] = make_int4(tid, excl >> 16, c, 0);
  }
  if (tid == PREP_THREADS - 1) *num_runs = (excl + own) & 0xFFFF;
  __syncthreads();
  for (int i = tid; i < P; i += PREP_THREADS) {
    const int e = topk_ids[i];
    if (e >= 0 && e < E) pair_ids[atomicAdd(&cur[e], 1)] = i;
  }
}

// ------------------------------------------------------------------------------------------------ top-k sum
__global__ void sum_kernel(const __nv_bfloat16* __restrict__ c3, const int* __restrict__ topk_ids,
                           __nv_bfloat16* __restrict__ out, int topk, int H, float scale) {
  const int tkn = blockIdx.x;
  for (int v = threadIdx.x; v < H / 8; v += blockDim.x) {
    float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < topk; k++) {
      const int pr = tkn * topk + k;
      if (topk_ids[pr] < 0) continue;
      const uint4 w = *reinterpret_cast<const uint4*>(c3 + (size_t)pr * H + v * 8);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
      for (int j = 0; j < 8; j++) a[j] += __bfloat162float(h[j]);
    }
    uint4 o;
    __nv_bfloat16* ho = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int j = 0; j < 8; j++) ho[j] = __float2bfloat16(a[j] * scale);
    *reinterpret_cast<uint4*>(out + (size_t)tkn * H + v * 8) = o;
  }
}

}  // namespace rtxw4
