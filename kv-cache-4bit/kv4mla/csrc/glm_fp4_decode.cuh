/*
kv4mla: sparse MLA decode over GLM-5.3's 4-bit (NVFP4-style) DSA KV rows, SM100 (B200/B300), h_q == 64.

Derived from FlashMLA's SM100 head64 sparse decode kernel (deepseek-ai/FlashMLA, MIT license,
csrc/kernels/sm100/decode/sparse/head64/kernel.cuh at ba89a34). The warp-specialized structure, the TMEM plan, the
dual-GEMM QK trick, the softmax warpgroup and the split-KV scheduling are FlashMLA's. What is different:

  KV row      kv4glm's 352-byte row (operations/kernels/glm_kv4.py), token-major, flat slots:
                [0, 256)   512 E2M1 latent values, element 2i in the low nibble of byte i
                [256, 288) 32 E4M3 scales, one per 16 latent values
                [288, 352) 64 E4M3 rope values (unscaled)
              One TMA gather4 row per token brings data, scales and rope in together, so the index warp does no
              per-token scale loads (FlashMLA's V3.2 path issues one random 16-byte LDG per token for its fp32 scales).
  q           FP8 E4M3 [b, s_q, 64, 576] as the served TRT-LLM path produces it. The dequant warpgroup loads it once per
              request segment and converts it (exactly) into the fp16 SW128 / SW64 tiles the UTCCP copies to TMEM.
  arithmetic  fp16 operands (UMMA kind::f16), fp32 accumulation. Every value involved is exact in fp16: E2M1 x E4M3
              products (<= 5 significant bits, magnitudes 2^-10 .. 2688), E4M3 rope and q (|x| <= 448), and the softmax
              probabilities (<= 1) carry 10 mantissa bits instead of bf16's 7. The conversions are single cvt.f16x2
              instructions on CUDA 13.0 (the bf16x2 forms need CUDA 13.2).
  dequant     one warpgroup turns each 64-token block into the fp16 K tile: 8 steps of fp4 (latent, SW128) and one step
              of fp8 (rope, SW64), 8 threads per token, 16 tokens per pass.

Grid: (s_q, num_sm_parts) with split-KV (FlashMLA's scheduler + combine kernel), or (s_q, b) without.
Threads: 384 = softmax warpgroup | MMA warp, raw-row TMA warp, (idle), index warp | dequant warpgroup.
*/
#pragma once

#include <math_constants.h>
#include <cuda_fp16.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <kerutils/kerutils.cuh>

#include "kernels/params.h"
#include "kernels/utils.h"
#include "kernels/defines.h"
#include "kernels/sm100/helpers.h"
#include "kernels/sm100/common_subroutine.h"

namespace kv4mla {

using namespace cute;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using f16 = cutlass::half_t;

// Row layout (source of truth: operations/kernels/glm_kv4.py)
static constexpr int ROW_BYTES = 352;
static constexpr int SCALE_OFF = 256;
static constexpr int ROPE_OFF = 288;

enum NamedBarriers : uint32_t {
    main_loop_sync = 0,
    wg0_sync = 1,
    wg0_warp02_sync = 2,
    wg0_warp13_sync = 3,
    everyone_sync = 4
};

// 2 E4M3 -> 2 fp16 (exact)
CUTE_DEVICE uint32_t e4m3x2_to_f16x2(uint16_t v) {
    uint32_t out;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(v));
    return out;
}

// 8 E2M1 (4 bytes, element 2i in the low nibble of byte i) -> 4 x fp16x2 (exact); one asm block so ptxas picks the
// source byte with the F2FP byte selector
CUTE_DEVICE void e2m1x8_to_f16x2x4(uint32_t packed, uint32_t (&out)[4]) {
    asm(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(packed));
}

CUTE_DEVICE uint32_t hmul2_u32(uint32_t a, uint32_t b) {
    __half2 r = __hmul2(*(__half2*)&a, *(__half2*)&b);
    return *(uint32_t*)&r;
}

CUTE_DEVICE void sts128(uint32_t addr, const uint32_t (&v)[4]) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" :: "r"(addr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]));
}

template<bool ENABLE_SPLITKV_>
struct GlmFp4DecodeKernel {

static constexpr bool ENABLE_SPLITKV = ENABLE_SPLITKV_;
static constexpr int B_H = 64;
static constexpr int D_Q = 576;
static constexpr int D_K = D_Q;
static constexpr int D_V = 512;
static constexpr int D_NOPE = 512;
static constexpr int D_ROPE = 64;
static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 8;   // the index warp runs ahead and L2-prefetches these blocks' rows
static constexpr int NUM_THREADS = 128 * 3;
static constexpr float MAX_INIT_VAL = -1e30f;
static constexpr int D_Q_SW128 = 512;
static constexpr int D_Q_SW64 = 64;
static constexpr int K_ROPE_SW = 64;    // bytes; the rope tile is two SW64 column tiles of 32 fp16

// Tensor memory columns: 0..256 O, 256.. Q (dual-GEMM halves), 400..464 P
struct tmem_cols {
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int Q_Tail = 256 + B_H * D_NOPE / 2 / 128;
    static constexpr int P = 400;
};

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<f16>{}, Shape<Int<B_H>, Int<NUM_TILES * 64>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
using SmemLayoutQ_SW128 = SmemLayoutQTiles<D_Q_SW128 / 64>;

using SmemLayoutOBuf = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H>, Int<D_V>>{}));
using SmemLayoutOBuf_TMA = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H>, Int<64>>{}));

using SmemLayoutS = decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<f16>{}, Shape<Int<B_H>, Int<B_TOPK>>{}, Step<_1, _2>{}));

template<int NUM_TILES>
using SmemLayoutKTiles_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<f16>{}, Shape<Int<B_TOPK>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<f16>{}, Shape<Int<B_TOPK * 2>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
template<int NUM_TILES>
using SmemLayoutKTilesTransposed_SW128 = decltype(composition(
    SmemLayoutKTiles_SW128<NUM_TILES>{}, Layout<Shape<Int<64 * NUM_TILES>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}));
template<int NUM_TILES>
using SmemLayoutKTiles_SW64 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<f16>{}, Shape<Int<B_TOPK>, Int<32 * NUM_TILES>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW64 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<f16>{}, Shape<Int<B_TOPK * 2>, Int<32 * NUM_TILES>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
static_assert(B_H == B_TOPK);   // q's SW64 tail uses the rope tile layout

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<f16, cosize_v<SmemLayoutQ_SW128>> q;
            f16 q_sw64[B_H * D_Q_SW64];
            array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;   // split-KV partials go to global from registers
        } qo;
        struct {
            struct {
                alignas(1024) f16 nope[B_TOPK * D_NOPE];   // SW128 K-major, 8 column tiles of 64
                alignas(1024) f16 rope[B_TOPK * D_ROPE];   // SW64 K-major, 2 column tiles of 32
            } dequant[NUM_BUFS];
            static_assert(sizeof(dequant) >= sizeof(f16) * (B_H * D_Q));   // q (+ tail) never covers raw
            array_aligned<uint8_t, B_TOPK * ROW_BYTES, 128> raw[NUM_BUFS];
        } kv;
    } u;
    // not a union: softmax(i+1) exchanges its P halves while SV(i) may still read S(i)
    float p_exchange_buf[4][32 * (B_TOPK / (128 / B_H))];
    array_aligned<f16, cosize_v<SmemLayoutS>> s;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK / 8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_ready, bar_q_utccp;
    transac_bar_t bar_p_empty;   // the softmax warpgroup has P of the current block in registers: QK(i+1) may overwrite it
    transac_bar_t bar_dequant_ready[NUM_BUFS];
    transac_bar_t bar_raw_ready[NUM_BUFS], bar_raw_free[NUM_BUFS];
    transac_bar_t bar_index_ready[NUM_INDEX_BUFS], bar_index_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<f16, f16, float, B_H, B_TOPK * 2, UMMA::Major::K, UMMA::Major::K>{}));
static constexpr int PV_GEMM_N = 256;
using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<f16, f16, float, B_H, PV_GEMM_N, UMMA::Major::K, UMMA::Major::MN>{}));

template<typename Shape_O, typename TMA_O>
struct TmaParams {
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv;   // [num_rows, 88 x uint32], box one row, for gather4
};

// ---------------------------------------------------------------------------------------------------- dequant

// One warpgroup, one 64-token block: raw rows (ROW_BYTES apart, TMA-gathered) -> fp16 nope tile (SW128) + rope tile
// (SW64). Thread = (group g = tid / 8 -> token rows g, g+16, g+32, g+48; lane-in-group i = tid % 8 -> 8 elements per
// step). Each step writes one 16-byte chunk; the 8 chunks of an 8-thread group form one 128-byte swizzle-atom row
// (nope) or two 64-byte rows (rope), so the STS.128 of a wavefront is conflict-free. Raw rows: 352 B = 88 words,
// 88 % 32 = 24, so the 4 groups of a warp read 4 different bank quarters (LDS.32 conflict-free).
struct Dequantizer {
    static constexpr int GROUP = 8, NUM_GROUPS = 16, PASSES = B_TOPK / NUM_GROUPS;
    int g, i;
    uint32_t nope_dst;    // byte offset of this thread's chunk (row g, step 0) inside the nope tile
    uint32_t rope_dst;    // byte offset of this thread's chunk (row g) inside the rope tile
    uint32_t scale_sel;   // PRMT selector: byte (i / 2) of a scale word, replicated into bytes 0 and 1

    CUTE_DEVICE explicit Dequantizer(int tid): g(tid / GROUP), i(tid % GROUP) {
        const int r8 = g % 8;
        nope_dst = g / 8 * 1024 + r8 * 128 + ((i ^ r8) * 16);
        // rope: thread i covers rope elements 8i..8i+7 -> column tile i / 4, 16-byte chunk i % 4 in a 64-byte row;
        // SW64 XORs the chunk index with (row / 2) % 4 inside each 512-byte (8-row) atom
        rope_dst = (i / 4) * (B_TOPK * 64) + (g / 8) * 512 + r8 * 64 + (((i % 4) ^ ((r8 >> 1) & 3)) * 16);
        scale_sel = (uint32_t)(i / 2) * 0x11;
    }

    // This thread's share of one block's raw rows, staged in registers so the raw buffer is released before the
    // (longer) conversion and before waiting for the destination tile
    struct Staged {
        uint32_t nib[PASSES][8];
        uint32_t sc[PASSES][8];
        uint2 rope[PASSES];
    };

    CUTE_DEVICE void load(const uint8_t *raw, Staged &st) const {
        CUTE_UNROLL
        for (int p = 0; p < PASSES; ++p) {
            const uint8_t *r = raw + (p * NUM_GROUPS + g) * ROW_BYTES;
            *(uint4*)&st.sc[p][0] = *(const uint4*)(r + SCALE_OFF);   // 16-byte aligned: 352 and 256 are multiples of 16
            *(uint4*)&st.sc[p][4] = *(const uint4*)(r + SCALE_OFF + 16);
            CUTE_UNROLL
            for (int s = 0; s < 8; ++s) st.nib[p][s] = *(const uint32_t*)(r + s * 32 + i * 4);
            st.rope[p] = *(const uint2*)(r + ROPE_OFF + i * 8);
        }
    }

    CUTE_DEVICE void store(const Staged &st, uint32_t nope_base, uint32_t rope_base) const {
        CUTE_UNROLL
        for (int p = 0; p < PASSES; ++p) {
            CUTE_UNROLL
            for (int s = 0; s < 8; ++s) {
                // elements s*64 + 8i .. +8 -> scale block s*4 + i/2 -> byte (i/2) of scale word s
                const uint32_t sw = __byte_perm(st.sc[p][s], 0, scale_sel);
                const uint32_t scale2 = e4m3x2_to_f16x2((uint16_t)(sw & 0xFFFF));
                uint32_t v[4];
                e2m1x8_to_f16x2x4(st.nib[p][s], v);
                CUTE_UNROLL
                for (int k = 0; k < 4; ++k) v[k] = hmul2_u32(v[k], scale2);   // exact
                sts128(nope_base + nope_dst + p * (NUM_GROUPS * 128) + s * (B_TOPK * 128), v);
            }
            uint32_t v[4];
            v[0] = e4m3x2_to_f16x2((uint16_t)(st.rope[p].x & 0xFFFF));
            v[1] = e4m3x2_to_f16x2((uint16_t)(st.rope[p].x >> 16));
            v[2] = e4m3x2_to_f16x2((uint16_t)(st.rope[p].y & 0xFFFF));
            v[3] = e4m3x2_to_f16x2((uint16_t)(st.rope[p].y >> 16));
            sts128(rope_base + rope_dst + p * (NUM_GROUPS * 64), v);
        }
    }
};

template<typename TmaParam>
static __device__ void devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params);
static void run(const SparseAttnDecodeParams &params);
};

}  // namespace kv4mla
