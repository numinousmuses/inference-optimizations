/*
kv4mla FP8-MMA kernel: GLM-5.3 sparse MLA decode over the 352-byte FP4 rows with E4M3 tensor-core math (UMMA
kind::f8f6f4), as the served TRT-LLM op computes on its FP8 pool.

Same row format, interface, schedule and softmax structure as glm_fp4_decode.cu (FlashMLA-derived); what differs:
  - the dequant warpgroup writes E4M3 tiles: E2M1 x E4M3-scale products (exact in fp16) rounded once to E4M3 (the same
    rounding kv4glm's FP8 scratch path applies), rope bytes copied verbatim. Tiles are half the size of fp16 tiles.
  - q is FP8 already: TMA loads it straight into the SW128 (latent) / SW32 (rope) tiles the UTCCP copies to TMEM.
  - P is written as E4M3 scaled by 128 with the lazy-rescale threshold lowered to 1 (P <= 2: no saturation), divided
    out with 1/l.
  - pipelining: 3 KV tile buffers, 3 raw buffers, 2 P tiles in TMEM and 2 S tiles in SMEM, so QK(i+1) runs on the tensor
    core while the softmax warpgroup works on block i, and the softmax never waits for SV except to rescale O.
Measured reason (clock64 timeline of the fp16 kernel): per 64-token block QK ~600-1000 + softmax ~1450 + SV cycles ran
strictly in series; fp16 UMMA at 64 heads is 2x the tensor time of FP8, and fp16 tiles left room for 2 buffers only.
*/
#include <climits>
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

namespace cute {
// tcgen05.mma.ws kind::f8f6f4 with A and B from shared memory (FlashMLA's kerutils has the kind::f16 atom only)
template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major, UMMA::Major b_major>
struct KV4MLA_MMA_F8_WS_SS {
  static_assert(M == 32 || M == 64 || M == 128);
  static_assert(N == 64 || N == 128 || N == 256);
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];
  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a, uint64_t const& desc_b, uint32_t const& tmem_c,
                                   uint32_t const& scaleC, uint64_t const& idescE) {
    asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p, 0; \n\t}\n"
      :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)), "r"(scaleC));
  }
};

template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major, UMMA::Major b_major>
struct MMA_Traits<KV4MLA_MMA_F8_WS_SS<a_type, b_type, c_type, M, N, a_major, b_major>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == 8 && cute::sizeof_bits_v<b_type> == 8);
  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_elastic<c_type, uint32_t, 128 / M>;
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;
  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;
  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major>();
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const& traits, Tensor<TD, DLayout>& D,
                                                    Tensor<TA, ALayout> const& A, Tensor<TB, BLayout> const& B,
                                                    Tensor<TC, CLayout> const& C) {
    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
    KV4MLA_MMA_F8_WS_SS<a_type, b_type, c_type, M, N, a_major, b_major>::fma(desc_a, desc_b, tmem_c,
                                                                            uint32_t(traits.accumulate_), idesc);
  }
};
}  // namespace cute

namespace kv4mla::f8 {

#ifdef KV4MLA_TRACE_ON
// Debug timeline of CTA (0, 0), slot [ev][j % 64] = clock64 (see glm_fp4_decode.cu for the event list)
static __device__ unsigned long long trace_buf[10 * 64];
#define F8_TRACE(ev, blk) do { if (blockIdx.x == 0 && blockIdx.y == 0) trace_buf[(ev) * 64 + ((blk) & 63)] = clock64(); } while (0)
#else
#define F8_TRACE(ev, blk) do {} while (0)
#endif

using namespace cute;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using fp8 = cutlass::float_e4m3_t;

static constexpr int ROW_BYTES = 352, SCALE_OFF = 256, ROPE_OFF = 288;
// Softmax probabilities are stored as E4M3 x P_SCALE. FlashMLA's lazy rescale (O is rescaled only when the running max
// grows by more than RESCALE_THRESHOLD in log2 units) lets exp2(p - max) reach 2^threshold, so threshold and scale are
// chosen together: P <= 2 -> stored <= 256 < 448 (no saturation); probabilities down to 2^-13 stay representable.
#ifndef KV4MLA_F8_RESCALE_LOG2
#define KV4MLA_F8_RESCALE_LOG2 1
#endif
static constexpr float RESCALE_THRESHOLD = KV4MLA_F8_RESCALE_LOG2;
static constexpr float P_SCALE = 256.0f / (1 << KV4MLA_F8_RESCALE_LOG2);   // P * P_SCALE <= 256 < 448

enum NamedBarriers : uint32_t { wg0_sync = 1, wg0_warp02_sync = 2, wg0_warp13_sync = 3, everyone_sync = 4 };

CUTE_DEVICE uint32_t e4m3x2_to_f16x2(uint16_t v) {
    uint32_t out;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(v));
    return out;
}
CUTE_DEVICE void e2m1x8_to_f16x2x4(uint32_t packed, uint32_t (&out)[4]) {
#ifdef KV4MLA_DQ_NOCVT_IN
    out[0] = packed & 0x0F0F0F0F; out[1] = (packed >> 4) & 0x0F0F0F0F; out[2] = packed ^ 0x3C003C00; out[3] = packed | 0x38003800;
    return;   // timing experiment only
#endif
    asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\ncvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]) : "r"(packed));
}
CUTE_DEVICE uint32_t hmul2_u32(uint32_t a, uint32_t b) {
    __half2 r = __hmul2(*(__half2*)&a, *(__half2*)&b);
    return *(uint32_t*)&r;
}
// two fp16x2 -> four E4M3 in one word (round to nearest even, saturating), element order kept
CUTE_DEVICE uint32_t f16x2x2_to_e4m3x4(uint32_t lo, uint32_t hi) {
#ifdef KV4MLA_DQ_NOCVT_OUT
    return lo ^ (hi << 3);   // timing experiment only
#endif
    uint16_t a, b;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(a) : "r"(lo));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(b) : "r"(hi));
    return (uint32_t)a | ((uint32_t)b << 16);
}
// (x, y) f32 -> E4M3 pair, x in the low byte
CUTE_DEVICE uint16_t f32x2_to_e4m3x2(float x, float y) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(y), "f"(x));
    return r;
}
CUTE_DEVICE void sts128(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d));
}
CUTE_DEVICE void sts64(uint32_t addr, uint2 v) {
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" :: "r"(addr), "r"(v.x), "r"(v.y));
}
// wait on an mbarrier for the completion with parity `phase`
CUTE_DEVICE void bar_wait(transac_bar_t &b, int phase) { b.wait(phase & 1); }

template<bool ENABLE_SPLITKV_>
struct Kernel {
static constexpr bool ENABLE_SPLITKV = ENABLE_SPLITKV_;
static constexpr int B_H = 64, D_Q = 576, D_V = 512, D_NOPE = 512, D_ROPE = 64, B_TOPK = 64;
#ifndef KV4MLA_F8_TILES
#define KV4MLA_F8_TILES 4
#endif
#ifndef KV4MLA_F8_RAW
#define KV4MLA_F8_RAW 3
#endif
static constexpr int NUM_TILES = KV4MLA_F8_TILES;   // dequantized E4M3 KV tiles: dequant(j) only waits for SV(j - NUM_TILES)
static constexpr int NUM_RAW = KV4MLA_F8_RAW;       // raw row buffers (TMA targets, released right after the register staging)
static constexpr int NUM_P = 2;            // P tiles in TMEM, S tiles in SMEM
static constexpr int NUM_INDEX_BUFS = 8;
// Warpgroups: 0 softmax | 1 MMA / raw TMA / index warps | 2 dequant | 3 correction (rescales O in TMEM when a head's running
// max grew, off the softmax critical path: measured, the in-line rescale + its wait for SV(j-1) cost ~15% at bs128)
static constexpr int NUM_DQ_WG = 1;
static constexpr int NUM_THREADS = 512;
static constexpr int REG_SOFTMAX = 200, REG_PRODUCER = 56, REG_DQ = 128, REG_CORR = 128;   // x128 threads <= 65536
static_assert(REG_SOFTMAX + REG_PRODUCER + REG_DQ + REG_CORR <= 512);
static constexpr float MAX_INIT_VAL = -1e30f;

// TMEM: O 0..256 | Q 256..400 (576 E4M3 per head, duplicated into lanes h and h + 64) | P0 400..432 | P1 432..464.
// QK is a weight-stationary M=64 N=64 MMA: output lanes 0-63 hold tokens 0-31 and lanes 64-127 tokens 32-63 of the same
// 64 heads (A is read from the matching lane half), so every softmax thread owns 32 finished scores of one head: no
// dual-GEMM partial sums to exchange (the fp16 kernel's p_exchange round trip).
struct tmem_cols {
    static constexpr int O = 0, Q = 256, P = 400;
};

using SmemLayoutNope = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8>{}, Shape<Int<B_TOPK>, Int<D_NOPE>>{}, Step<_1, _2>{}));
using SmemLayoutRope = decltype(tile_to_shape(UMMA::Layout_K_SW32_Atom<fp8>{}, Shape<Int<B_TOPK>, Int<D_ROPE>>{}, Step<_1, _2>{}));
using SmemLayoutV = decltype(composition(SmemLayoutNope{}, Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}));
using SmemLayoutS = decltype(tile_to_shape(UMMA::Layout_K_SW64_Atom<fp8>{}, Shape<Int<B_H>, Int<B_TOPK>>{}, Step<_1, _2>{}));
using SmemLayoutOBuf = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H>, Int<D_V>>{}));
using SmemLayoutOBuf_TMA = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H>, Int<64>>{}));
static_assert(B_H == B_TOPK);   // q tiles use the KV tile layouts

static constexpr int PV_GEMM_N = 256;
using TiledMMA_O = decltype(make_tiled_mma(KV4MLA_MMA_F8_WS_SS<fp8, fp8, float, B_H, PV_GEMM_N, UMMA::Major::K, UMMA::Major::MN>{}));

struct Tile {
    alignas(1024) fp8 nope[B_TOPK * D_NOPE];   // 32 KB
    alignas(1024) fp8 rope[B_TOPK * D_ROPE];   // 4 KB
};
struct SharedMemoryPlan {
    union {
        Tile tiles[NUM_TILES];
        struct {
            Tile q;   // q uses a tile's layout; copied to TMEM before any tile is written in a segment
            array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
        } qo;
    } u;
    array_aligned<uint8_t, B_TOPK * ROW_BYTES, 128> raw[NUM_RAW];
    alignas(1024) fp8 s[NUM_P][B_H * B_TOPK];
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK / 8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done, bar_q_tma, bar_q_utccp;
    transac_bar_t bar_raw_ready[NUM_RAW], bar_raw_free[NUM_RAW], bar_tile_ready[NUM_TILES], bar_sv_done[NUM_TILES];
    transac_bar_t bar_qk_done[NUM_P], bar_p_empty[NUM_P], bar_so_ready[NUM_P], bar_corr_done[NUM_P];
    float o_scale_buf[NUM_P][128];   // softmax thread t -> correction thread t (same TMEM lane): factor for block j
    int o_scale_flag[NUM_P][4];      // per warp: rescale needed
    transac_bar_t bar_index_ready[NUM_INDEX_BUFS], bar_index_free[NUM_INDEX_BUFS];
};

template<typename Shape_Q, typename TMA_Q, typename Shape_O, typename TMA_O>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_q_rope;
    CUtensorMap tensor_map_kv;
};

template<typename TmaParam>
static __device__ void devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params);
static void run(const SparseAttnDecodeParams &params);
};

template<bool S>
template<typename TmaParam>
__device__ void Kernel<S>::devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params) {
#if defined(KERUTILS_ENABLE_SM100A)
    const int s_q_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_q_rope);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
    }
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_last_store_done.init(128);
            plan.bar_q_tma.init(1);
            plan.bar_q_utccp.init(1);
            for (int i = 0; i < NUM_RAW; ++i) {
                plan.bar_raw_ready[i].init(1);
                plan.bar_raw_free[i].init(128);
            }
            for (int i = 0; i < NUM_TILES; ++i) {
                plan.bar_tile_ready[i].init(128);
                plan.bar_sv_done[i].init(1);
            }
            for (int i = 0; i < NUM_P; ++i) {
                plan.bar_qk_done[i].init(1);
                plan.bar_p_empty[i].init(128);
                plan.bar_so_ready[i].init(128);
                plan.bar_corr_done[i].init(128);
            }
            for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
                plan.bar_index_ready[i].init(32);
                plan.bar_index_free[i].init(128 + 1);
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    struct MainLoopArgs {
        int batch_idx, start_block_idx, end_block_idx;
        bool is_no_split; int n_split_idx;
        int seg;                 // segment ordinal in this CTA (parity of the once-per-segment barriers)
        int topk_length;
        bool is_last_batch;
    };
    auto run_main_loop = [&](auto f) {
        DecodingSchedMeta sched_meta;
        if constexpr (ENABLE_SPLITKV) {
            KU_LDG_256(params.tile_scheduler_metadata_ptr + partition_idx, &sched_meta, ".nc", "no_allocate", "evict_normal", "256B");
        } else {
            sched_meta.begin_req_idx = sched_meta.end_req_idx = partition_idx;
        }
        if (sched_meta.begin_req_idx >= params.b) return;
        int seg = 0;
        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx, ++seg) {
            int start_block_idx, end_block_idx; bool is_split; int n_split_idx;
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
            if constexpr (ENABLE_SPLITKV) {
                start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
                end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : topk_padded / B_TOPK;
                is_split = batch_idx == sched_meta.begin_req_idx ? sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted : false);
                n_split_idx = batch_idx == sched_meta.begin_req_idx ? (__ldg(params.num_splits_ptr + batch_idx) + sched_meta.begin_split_idx) : __ldg(params.num_splits_ptr + batch_idx);
            } else {
                start_block_idx = 0; end_block_idx = topk_padded / B_TOPK; is_split = false; n_split_idx = 0;
            }
            MainLoopArgs args = {batch_idx, start_block_idx, end_block_idx, !is_split, n_split_idx, seg, topk_length,
                                 batch_idx == sched_meta.end_req_idx};
            f(args);
            NamedBarrier(NUM_THREADS, NamedBarriers::everyone_sync).arrive_and_wait_unaligned();
        }
    };

    // Every role walks the same block sequence; j counts blocks over all segments of this CTA and selects the ring
    // slots: tile/raw j % 3, P/S j % 2, index j % 8, each with the parity of j / ring size.
    int j = 0;

    if (warpgroup_idx == 0) {
        // ---------------------------------------------------------------- softmax warpgroup
        cutlass::arch::warpgroup_reg_alloc<REG_SOFTMAX>();
        constexpr int B_EPI = 64;
        Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o_buf.data()), SmemLayoutOBuf{});
        bf16* sO_bases[B_EPI / 8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI / 8; ++i)
            sO_bases[i] = &sO(idx_in_warpgroup % B_H, (idx_in_warpgroup / B_H) * (PV_GEMM_N / 2) + i * 8);
        // this thread's P row: head (warp & 1) * 32 + lane, tokens (warp / 2) * 32 .. +32 (after the dual-GEMM reduce)
        uint32_t s_off[2];
        const int row_h = (warp_idx & 1) * 32 + lane_idx, row_t0 = (warp_idx / 2) * 32;
        {
            Tensor sS = make_tensor(make_smem_ptr(plan.s[0]), SmemLayoutS{});
            s_off[0] = (uint32_t)(&sS(row_h, row_t0) - plan.s[0]);
            s_off[1] = (uint32_t)(&sS(row_h, row_t0 + 16) - plan.s[0]);
        }
        const float2 scale = float2{params.sm_scale_div_log2, params.sm_scale_div_log2};
        constexpr int NE = B_TOPK / 2;

        run_main_loop([&](const MainLoopArgs &args) {
            cute::tma_store_wait<0>();
            plan.bar_last_store_done.arrive();
            float mi = MAX_INIT_VAL, li = 0.0f, real_mi = -CUDART_INF_F;
            const int j0 = j;
            // dense mode: this q row (token x head) sees positions < row_limit only (causal over the draft rows)
            const int row_lim = params.row_limit
                ? __ldg(params.row_limit + args.batch_idx * params.stride_row_limit_b + row_h) : INT_MAX;
            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++j) {
                const int pb = j % NUM_P, ib = j % NUM_INDEX_BUFS;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                bar_wait(plan.bar_index_ready[ib], j / NUM_INDEX_BUFS);
                bar_wait(plan.bar_qk_done[pb], j / NUM_P);
                ku::tcgen05_after_thread_sync();
                if (idx_in_warpgroup == 0) F8_TRACE(7, j);
                float p[NE];
                ku::tmem_ld_32dp32bNx<NE>(tmem_cols::P + pb * 32, p);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();
                plan.bar_p_empty[pb].arrive();   // P is in registers: the MMA warp may issue QK(j + 2) into this tile
                {
                    const uint32_t valid = *(const uint32_t*)(plan.is_token_valid[ib] + (warp_idx >= 2 ? NE / 8 : 0));
                    CUTE_UNROLL
                    for (int k = 0; k < NE; ++k)
                        if (!((valid >> k) & 1)) p[k] = -CUDART_INF_F;
                }
                if (params.row_limit) {
                    const int base = block_idx * B_TOPK + row_t0;
                    CUTE_UNROLL
                    for (int k = 0; k < NE; ++k)
                        if (base + k >= row_lim) p[k] = -CUDART_INF_F;
                }
                float cur_pi_max = sm100::get_max<NE>(p) * params.sm_scale_div_log2;
                plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                plan.bar_index_free[ib].arrive();
                cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup ^ 64]);
                real_mi = max(real_mi, cur_pi_max);
                const bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > RESCALE_THRESHOLD);
                float new_max, scale_for_old;
                if (!should_scale_o) { scale_for_old = 1.0f; new_max = mi; }
                else { new_max = max(cur_pi_max, mi); scale_for_old = exp2f(mi - new_max); }
                mi = new_max;
                // S = exp2(p * scale - max) (<= 1), stored as E4M3 x 256
                uint32_t sw[NE / 4];
                const float2 neg_new_max = float2{-new_max, -new_max};
                float2 cur_sum = float2{0.0f, 0.0f};
                CUTE_UNROLL
                for (int i = 0; i < NE / 4; ++i) {
                    float2 d0 = ku::float2_fma(float2{p[i * 4], p[i * 4 + 1]}, scale, neg_new_max);
                    float2 d1 = ku::float2_fma(float2{p[i * 4 + 2], p[i * 4 + 3]}, scale, neg_new_max);
                    d0.x = exp2f(d0.x); d0.y = exp2f(d0.y); d1.x = exp2f(d1.x); d1.y = exp2f(d1.y);
                    cur_sum = ku::float2_add(cur_sum, ku::float2_add(d0, d1));
                    sw[i] = (uint32_t)f32x2_to_e4m3x2(d0.x * P_SCALE, d0.y * P_SCALE)
                          | ((uint32_t)f32x2_to_e4m3x2(d1.x * P_SCALE, d1.y * P_SCALE) << 16);
                }
                li = fma(li, scale_for_old, cur_sum.x + cur_sum.y);
                // S tile pb was read by SV(j - 2)
                if (j - 2 >= j0) bar_wait(plan.bar_sv_done[(j - 2) % NUM_TILES], (j - 2) / NUM_TILES);
                const uint32_t sbase = cute::cast_smem_ptr_to_uint(plan.s[pb]);
                sts128(sbase + s_off[0], sw[0], sw[1], sw[2], sw[3]);
                sts128(sbase + s_off[1], sw[4], sw[5], sw[6], sw[7]);
                // the correction warpgroup rescales O (after SV(j - 1)) before the MMA warp issues SV(j)
                plan.o_scale_buf[pb][idx_in_warpgroup] = scale_for_old;
                if (lane_idx == 0) plan.o_scale_flag[pb][warp_idx] = (j != j0) && should_scale_o;
                fence_view_async_shared();
                plan.bar_so_ready[pb].arrive();
                if (idx_in_warpgroup == 0) F8_TRACE(8, j);
            }
            if (real_mi == -CUDART_INF_F) { li = 0.0f; mi = -CUDART_INF_F; }
            plan.rowwise_max_buf[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            li += plan.rowwise_max_buf[idx_in_warpgroup ^ 64];
            if (idx_in_warpgroup < params.h_q) {
                if (args.is_no_split) {
                    float cur_lse = fma(mi, CUDART_LN2_F, logf(li));
                    cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                    ((float*)params.lse)[args.batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q + idx_in_warpgroup] = cur_lse;
                } else {
                    ((float*)params.lse_accum)[args.n_split_idx * params.stride_lse_accum_split + s_q_idx * params.stride_lse_accum_s_q + idx_in_warpgroup] = log2f(li) + mi;
                }
            }
            bar_wait(plan.bar_sv_done[(j - 1) % NUM_TILES], (j - 1) / NUM_TILES);
            ku::tcgen05_after_thread_sync();
            if (args.is_last_batch) cudaTriggerProgrammaticLaunchCompletion();

            const float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li * P_SCALE);
            const float2 o_scale2 = {o_scale, o_scale};
            static constexpr int O_FOLD = 2;
            if (args.is_no_split) {
                Tensor tma_gO = flat_divide(tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx, args.batch_idx),
                                            Shape<Int<B_H>, Int<64>>{})(_, _, _0{}, _);
                auto thr_tma = tma_params.tma_O.get_slice(_0{});
                Tensor tma_sO = flat_divide(sO, Shape<Int<B_H>, Int<64>>{})(_, _, _0{}, _);
                float2 o[B_EPI / 2];
                __nv_bfloat162 o_bf16[B_EPI / 2];
                CUTE_UNROLL
                for (int i = 0; i < (D_V / O_FOLD) / B_EPI; ++i) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i * B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int jj = 0; jj < B_EPI / 2; ++jj) o_bf16[jj] = __float22bfloat162_rn(ku::float2_mul(o[jj], o_scale2));
                    int col_base = (i >= PV_GEMM_N / O_FOLD / B_EPI ? D_V / 2 : 0) + (i * B_EPI % (PV_GEMM_N / O_FOLD));
                    CUTE_UNROLL
                    for (int jj = 0; jj < B_EPI / 8; ++jj)
                        *(__int128_t*)(sO_bases[jj] + col_base * B_H) = *(__int128_t*)(&o_bf16[jj * 4]);
                    fence_view_async_shared();
                    NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                    if (warp_idx < O_FOLD && elect_one_sync()) {
                        int copy_block_idx = warp_idx * (PV_GEMM_N / 2 / B_EPI) + col_base / B_EPI;
                        cute::copy(tma_params.tma_O, thr_tma.partition_S(tma_sO(_, _, copy_block_idx)),
                                   thr_tma.partition_D(tma_gO(_, _, copy_block_idx)));
                    }
                }
                cute::tma_store_arrive();
            } else {
                float2 o[B_EPI / 2];
                const int row = idx_in_warpgroup % B_H;
                float* gOa = (float*)params.o_accum + (int64_t)args.n_split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q + row * params.stride_o_accum_h_q;
                CUTE_UNROLL
                for (int i = 0; i < (D_V / O_FOLD) / B_EPI; ++i) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i * B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int jj = 0; jj < B_EPI / 2; ++jj) o[jj] = ku::float2_mul(o[jj], o_scale2);
                    int col_base = (idx_in_warpgroup / B_H) * (PV_GEMM_N / O_FOLD) + (i >= PV_GEMM_N / O_FOLD / B_EPI ? D_V / 2 : 0) + (i * B_EPI % (PV_GEMM_N / O_FOLD));
                    if (row < params.h_q) {
                        CUTE_UNROLL
                        for (int jj = 0; jj < B_EPI / 4; ++jj) *(float4*)(gOa + col_base + jj * 4) = *(float4*)(&o[jj * 2]);
                    }
                }
            }
        });
        if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, 512);
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<REG_PRODUCER>();
        const int warp_idx = cutlass::canonical_warp_idx_sync();
        if (warp_idx == 4 && elect_one_sync()) {
            // ------------------------------------------------------------ MMA warp
            const uint32_t idesc_qk = uint32_t(UMMA::make_runtime_instr_desc<>(
                UMMA::make_instr_desc<fp8, fp8, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>()) >> 32);
            run_main_loop([&](const MainLoopArgs &args) {
                if (args.start_block_idx >= args.end_block_idx) ku::trap();
                // q (FP8) straight into the q tile: latent SW128 (TMA via CuTe) + rope SW32 (5D map)
                {
                    Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, args.batch_idx);
                    Tensor sQ = make_tensor(make_smem_ptr(plan.u.qo.q.nope), SmemLayoutNope{});
                    ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q_tma, TMA::CacheHintSm90::EVICT_FIRST);
                    cute::SM90_TMA_LOAD_5D::copy(&tma_params.tensor_map_q_rope, (uint64_t*)&plan.bar_q_tma,
                                                 (uint64_t)TMA::CacheHintSm90::EVICT_FIRST, plan.u.qo.q.rope, 0, 0, 0,
                                                 s_q_idx, args.batch_idx);
                    plan.bar_q_tma.arrive_and_expect_tx(B_H * D_Q);
                }
                bar_wait(plan.bar_q_tma, args.seg);
                ku::tcgen05_after_thread_sync();
                {
                    // q byte b of head h -> TMEM lane h and h + 64 (64x128b.warpx2::02_13 copies), column Q + b / 4
                    UMMA::SmemDescriptor d = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(make_smem_ptr(plan.u.qo.q.nope),
                        tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8>{}, Shape<Int<B_H>, _128>{})));
                    CUTE_UNROLL
                    for (int b = 0; b < D_NOPE; b += 16)
                        SM100_UTCCP_2x64dp128bitlw0213_1cta::copy(d + ((b / 128) * 8192 + b % 128) / 16, tmem_cols::Q + b / 4);
                    UMMA::SmemDescriptor dr = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(make_smem_ptr(plan.u.qo.q.rope),
                        tile_to_shape(UMMA::Layout_K_SW32_Atom<fp8>{}, Shape<Int<B_H>, _32>{})));
                    CUTE_UNROLL
                    for (int b = 0; b < D_ROPE; b += 16)
                        SM100_UTCCP_2x64dp128bitlw0213_1cta::copy(dr + ((b / 32) * 2048 + b % 32) / 16, tmem_cols::Q + (D_NOPE + b) / 4);
                }
                ku::umma_arrive_noelect(plan.bar_q_utccp);
                bar_wait(plan.bar_q_utccp, args.seg);
                ku::tcgen05_after_thread_sync();

                TiledMMA tiled_mma_O = TiledMMA_O{};
                Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
                tO.data().get() = tmem_cols::O;

                // QK of block jq into P tile jq % 2: 18 k-steps of 32 bytes (16 latent + 2 rope), N = 64 tokens
                auto issue_qk = [&](int jq) {
                    const int tb = jq % NUM_TILES, pb = jq % NUM_P;
                    bar_wait(plan.bar_tile_ready[tb], jq / NUM_TILES);
                    bar_wait(plan.bar_p_empty[pb], jq / NUM_P + 1);   // softmax(jq - 2) holds its P in registers
                    ku::tcgen05_after_thread_sync();
                    const uint32_t tmem_p = tmem_cols::P + pb * 32;
                    // descriptors of the first 128-wide latent column tile and the first 32-wide rope tile
                    UMMA::SmemDescriptor dk = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
                        make_smem_ptr(plan.u.tiles[tb].nope), tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8>{}, Shape<Int<B_TOPK>, _128>{})));
                    UMMA::SmemDescriptor dr = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
                        make_smem_ptr(plan.u.tiles[tb].rope), tile_to_shape(UMMA::Layout_K_SW32_Atom<fp8>{}, Shape<Int<B_TOPK>, _32>{})));
#ifndef KV4MLA_SKIP_MMA
                    CUTE_UNROLL
                    for (int k = 0; k < 18; ++k) {
                        const uint64_t db = k < 16 ? (uint64_t)(dk + ((k / 4) * 8192 + (k % 4) * 32) / 16)
                                                   : (uint64_t)(dr + ((k - 16) * 2048) / 16);
                        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                                     "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p, 0;\n\t}\n"
                                     :: "r"(tmem_p), "r"((uint32_t)(tmem_cols::Q + k * 8)), "l"(db), "r"(idesc_qk), "r"((uint32_t)(k > 0)));
                    }
#endif
                    ku::umma_arrive_noelect(plan.bar_qk_done[pb]);
                    F8_TRACE(4, jq);
                };

                const int j0 = j;
                issue_qk(j);
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++j) {
                    if (block_idx + 1 < args.end_block_idx) issue_qk(j + 1);
                    const int pb = j % NUM_P, tb = j % NUM_TILES;
                    bar_wait(plan.bar_so_ready[pb], j / NUM_P);
                    bar_wait(plan.bar_corr_done[pb], j / NUM_P);
                    ku::tcgen05_after_thread_sync();
                    Tensor sS = make_tensor(make_smem_ptr(plan.s[pb]), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.tiles[tb].nope), SmemLayoutV{});
#if defined(KV4MLA_SV_HALF)
                    {   // timing experiment: half of the PV work (dims 0..255 only)
                        Tensor tOh = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V / 2>>{});
                        tOh.data().get() = tmem_cols::O;
                        Tensor sVh = flat_divide(sV, Shape<Int<D_V / 2>, Int<B_TOPK>>{})(_, _, _0{}, _0{});
                        ku::utcmma_ss(tiled_mma_O, sS, sVh, tOh, j == j0);
                    }
#elif !defined(KV4MLA_SKIP_MMA)
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, j == j0);
#endif
                    ku::umma_arrive_noelect(plan.bar_sv_done[tb]);
                    F8_TRACE(5, j);
                }
            });
        } else if (warp_idx == 5 && elect_one_sync()) {
            // ------------------------------------------------------------ raw row TMA warp
            run_main_loop([&](const MainLoopArgs &args) {
                bar_wait(plan.bar_last_store_done, args.seg);   // raw rows never overlap q / o_buf, but tiles do
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++j) {
                    const int rb = j % NUM_RAW, ib = j % NUM_INDEX_BUFS;
                    bar_wait(plan.bar_index_ready[ib], j / NUM_INDEX_BUFS);
                    bar_wait(plan.bar_raw_free[rb], j / NUM_RAW + 1);
                    int4 cur = *(int4*)(plan.tma_coord[ib] + 0), nxt;
                    CUTE_UNROLL
#ifdef KV4MLA_HALF_GATHER
                    constexpr int GATHER_ROWS = B_TOPK / 2;   // timing experiment only
#else
                    constexpr int GATHER_ROWS = B_TOPK;
#endif
                    for (int row = 0; row < GATHER_ROWS; row += 4) {
                        if (row + 4 < B_TOPK) nxt = *(int4*)(plan.tma_coord[ib] + row + 4);
                        ku::tma_gather4(&tma_params.tensor_map_kv, plan.bar_raw_ready[rb], plan.raw[rb].data() + ROW_BYTES * row, 0, cur,
                                        (int64_t)TMA::CacheHintSm90::EVICT_FIRST);
                        cur = nxt;
                    }
                    plan.bar_raw_ready[rb].arrive_and_expect_tx(GATHER_ROWS * ROW_BYTES);
                    F8_TRACE(1, j);
                    plan.bar_index_free[ib].arrive();
                }
            });
        } else if (warp_idx == 7) {
            // ------------------------------------------------------------ index warp
            run_main_loop([&](const MainLoopArgs &args) {
                const int* indices = (const int*)params.indices + params.stride_indices_b * args.batch_idx + params.stride_indices_s_q * s_q_idx;
                // dense mode (page_block_size > 1): `indices` is the sequence's page table; rows come from positions
                const int ps = params.page_block_size;
                auto load_idx = [&](int b) -> int2 {
                    if (ps <= 1) return __ldg((const int2*)(indices + b * B_TOPK + lane_idx * 2));
                    const int pos = b * B_TOPK + lane_idx * 2;
                    int2 r;
                    r.x = pos < args.topk_length ? __ldg(indices + pos / ps) * ps + pos % ps : -1;
                    r.y = pos + 1 < args.topk_length ? __ldg(indices + (pos + 1) / ps) * ps + (pos + 1) % ps : -1;
                    return r;
                };
                int2 my = load_idx(args.start_block_idx);
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++j) {
                    const int ib = j % NUM_INDEX_BUFS;
                    const bool has_next = block_idx + 1 < args.end_block_idx;
                    const int2 nx = has_next ? load_idx(block_idx + 1) : int2{};
                    const int abs_pos = block_idx * B_TOPK + lane_idx * 2;
                    int coords[2]; char valid_mask = 0;
                    CUTE_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        const int c = i == 0 ? my.x : my.y;
                        const bool v = c >= 0 && c < params.num_blocks && abs_pos + i < args.topk_length;
                        valid_mask |= v << i;
                        coords[i] = v ? c : -1;
                    }
                    valid_mask <<= lane_idx % 4 * 2;
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                    bar_wait(plan.bar_index_free[ib], j / NUM_INDEX_BUFS + 1);
                    *(int2*)(plan.tma_coord[ib] + lane_idx * 2) = *(int2*)coords;
                    if (lane_idx % 4 == 0) plan.is_token_valid[ib][lane_idx / 4] = valid_mask;
                    plan.bar_index_ready[ib].arrive();
                    if (lane_idx == 0) F8_TRACE(0, j);
                    if (has_next) my = nx;
                }
            });
        } else {
            run_main_loop([&](const MainLoopArgs &args) {});
        }
    } else if (warpgroup_idx == 3) {
        // ---------------------------------------------------------------- correction warpgroup
        cutlass::arch::warpgroup_reg_alloc<REG_CORR>();
        const int cw = warp_idx % 4;   // TMEM lane quarter of this warp = the softmax warp with the same lanes
        run_main_loop([&](const MainLoopArgs &args) {
            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++j) {
                const int pb = j % NUM_P;
                bar_wait(plan.bar_so_ready[pb], j / NUM_P);
                if (plan.o_scale_flag[pb][cw]) {
                    const float f = plan.o_scale_buf[pb][idx_in_warpgroup];
                    bar_wait(plan.bar_sv_done[(j - 1) % NUM_TILES], (j - 1) / NUM_TILES);   // O holds SV(j - 1)
                    ku::tcgen05_after_thread_sync();
                    sm100::rescale_O<D_V / 2, 64, tmem_cols::O>(f);
                    ku::tcgen05_before_thread_sync();
                }
                plan.bar_corr_done[pb].arrive();
            }
        });
    } else {
        // ---------------------------------------------------------------- dequant warpgroup
        // thread = group g (tokens g, g+16, g+32, g+48) x lane-in-group i: per step s (0..3) elements s*128 + 16i .. +16
        // (8 raw bytes, one scale) -> one 16-byte E4M3 chunk of an SW128 row; rope bytes 8i..8i+7 -> SW32 rope tile
        cutlass::arch::warpgroup_reg_alloc<REG_DQ>();
        const int dq_id = 0;
        const int g = idx_in_warpgroup / 8, i = idx_in_warpgroup % 8, r8 = g % 8;
        const uint32_t nope_dst = g / 8 * 1024 + r8 * 128 + ((i ^ r8) * 16);
        uint32_t rope_dst;
        {
            Tensor sR = make_tensor(make_smem_ptr(plan.u.tiles[0].rope), SmemLayoutRope{});
            rope_dst = (uint32_t)(&sR(g, i * 8) - plan.u.tiles[0].rope);
        }
        const uint32_t sel = (uint32_t)(i % 4);
        run_main_loop([&](const MainLoopArgs &args) {
            bool q_waited = false;
            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++j) {
                if (j % NUM_DQ_WG != dq_id) continue;
                const int tb = j % NUM_TILES, rb = j % NUM_RAW;
                bar_wait(plan.bar_raw_ready[rb], j / NUM_RAW);
                if (idx_in_warpgroup == 0) F8_TRACE(2, j);
                // stage this thread's bytes in registers, then release the raw buffer to the TMA warp
                const uint8_t *raw = plan.raw[rb].data();
                uint2 nib[4][4]; uint32_t scb[4]; uint2 rope[4];
                CUTE_UNROLL
                for (int p = 0; p < 4; ++p) {
                    const uint8_t *r = raw + (p * 16 + g) * ROW_BYTES;
                    const uint4 s0 = *(const uint4*)(r + SCALE_OFF), s1 = *(const uint4*)(r + SCALE_OFF + 16);
                    const uint32_t w[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
                    scb[p] = 0;
                    CUTE_UNROLL
                    for (int s = 0; s < 4; ++s) {
                        nib[p][s] = *(const uint2*)(r + s * 64 + i * 8);
                        // the scale of elements s*128 + 16i .. +16 is byte s*8 + i = byte i%4 of word 2s + i/4
                        const uint32_t word = i < 4 ? w[2 * s] : w[2 * s + 1];
                        scb[p] |= (__byte_perm(word, 0, sel) & 0xFFu) << (8 * s);
                    }
                    rope[p] = *(const uint2*)(r + ROPE_OFF + i * 8);
                }
                plan.bar_raw_free[rb].arrive();   // release semantics: the reads above are done
                bar_wait(plan.bar_sv_done[tb], j / NUM_TILES + 1);   // SV(j - NUM_TILES) no longer reads this tile
                if (idx_in_warpgroup == 0) F8_TRACE(3, j);
                if (!q_waited) {   // the first tiles of a segment overlap q until the UTCCP copied it
                    bar_wait(plan.bar_q_utccp, args.seg);
                    q_waited = true;
                }
#ifdef KV4MLA_DQ_LDS_ONLY
                {   // timing experiment: keep the staging loads alive, skip conversions and tile stores
                    uint32_t x = 0;
                    CUTE_UNROLL
                    for (int p = 0; p < 4; ++p) {
                        x ^= scb[p] ^ rope[p].x ^ rope[p].y;
                        CUTE_UNROLL
                        for (int s = 0; s < 4; ++s) x ^= nib[p][s].x ^ nib[p][s].y;
                    }
                    sts128(cute::cast_smem_ptr_to_uint(plan.u.tiles[tb].nope) + nope_dst, x, x, x, x);
                }
#elif !defined(KV4MLA_SKIP_DEQUANT)
                uint32_t acc_x = 0;
                const uint32_t nope_base = cute::cast_smem_ptr_to_uint(plan.u.tiles[tb].nope) + nope_dst;
                const uint32_t rope_base = cute::cast_smem_ptr_to_uint(plan.u.tiles[tb].rope) + rope_dst;
                CUTE_UNROLL
                for (int p = 0; p < 4; ++p) {
                    CUTE_UNROLL
                    for (int s = 0; s < 4; ++s) {
                        const uint32_t b = (scb[p] >> (8 * s)) & 0xFFu;
                        const uint32_t scale2 = e4m3x2_to_f16x2((uint16_t)(b | (b << 8)));
                        uint32_t lo[4], hi[4];
                        e2m1x8_to_f16x2x4(nib[p][s].x, lo);
                        e2m1x8_to_f16x2x4(nib[p][s].y, hi);
                        CUTE_UNROLL
                        for (int k = 0; k < 4; ++k) { lo[k] = hmul2_u32(lo[k], scale2); hi[k] = hmul2_u32(hi[k], scale2); }
#ifdef KV4MLA_DQ_NOSTS
                        acc_x ^= f16x2x2_to_e4m3x4(lo[0], lo[1]) ^ f16x2x2_to_e4m3x4(lo[2], lo[3])
                               ^ f16x2x2_to_e4m3x4(hi[0], hi[1]) ^ f16x2x2_to_e4m3x4(hi[2], hi[3]);   // timing experiment
#else
                        sts128(nope_base + p * 2048 + s * (B_TOPK * 128), f16x2x2_to_e4m3x4(lo[0], lo[1]),
                               f16x2x2_to_e4m3x4(lo[2], lo[3]), f16x2x2_to_e4m3x4(hi[0], hi[1]), f16x2x2_to_e4m3x4(hi[2], hi[3]));
#endif
                    }
                    sts64(rope_base + p * 512, rope[p]);
                }
#ifdef KV4MLA_DQ_NOSTS
                sts128(nope_base, acc_x, acc_x, acc_x, acc_x);
#endif
#endif
                fence_view_async_shared();
                plan.bar_tile_ready[tb].arrive();
                if (idx_in_warpgroup == 0) F8_TRACE(6, j);
            }
        });
    }
#else
    if (cute::thread0()) CUTE_INVALID_CONTROL_PATH("kv4mla: this kernel only supports sm100a");
#endif
}

}  // namespace kv4mla::f8

namespace kv4mla::f8 {

template<typename K, typename TmaParams>
__global__ void __launch_bounds__(K::NUM_THREADS, 1, 1)
glm_fp4_sparse_decode_fp8_kernel(__grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TmaParams tma_params) {
    K::devfunc(params, tma_params);
}

template<bool S>
void Kernel<S>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.topk % B_TOPK == 0);
    KU_ASSERT(params.h_q == 64 && params.h_kv == 1 && params.d_qk == D_Q && params.d_v == D_V);
    KU_ASSERT(params.stride_kv_row == ROW_BYTES);
    KU_ASSERT((int64_t)params.kv % 16 == 0 && (int64_t)params.q % 16 == 0);
    KU_ASSERT(params.stride_q_h_q % 16 == 0 && params.stride_q_s_q % 16 == 0 && params.stride_q_b % 16 == 0,
              "q strides must be multiples of 16 bytes (TMA)");
    auto shape_Q = make_shape(params.h_q, D_NOPE, params.s_q, params.b);
    auto tma_Q = cute::make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((fp8*)params.q), make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b))),
        SmemLayoutNope{});
    auto shape_O = make_shape(params.h_q, D_V, params.s_q, params.b);
    auto tma_O = cute::make_tma_copy(SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr((bf16*)params.out), make_layout(shape_O, make_stride(params.stride_o_h_q, _1{}, params.stride_o_s_q, params.stride_o_b))),
        SmemLayoutOBuf_TMA{});
    CUtensorMap tensor_map_q_rope = ku::make_tensor_map(
        {32, (uint64_t)params.h_q, 2, (uint64_t)params.s_q, (uint64_t)params.b},
        {(uint64_t)params.stride_q_h_q, 32, (uint64_t)params.stride_q_s_q, (uint64_t)params.stride_q_b},
        {32, (uint32_t)B_H, 2, 1, 1},
        (uint8_t*)params.q + D_NOPE,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    CUtensorMap tensor_map_kv = ku::make_tensor_map(
        {(uint64_t)ROW_BYTES / 4, (uint64_t)params.num_blocks}, {(uint64_t)ROW_BYTES}, {ROW_BYTES / 4, 1}, params.kv,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    TmaParams<decltype(shape_Q), decltype(tma_Q), decltype(shape_O), decltype(tma_O)> tp = {
        shape_Q, tma_Q, shape_O, tma_O, tensor_map_q_rope, tensor_map_kv};
    auto kernel = &glm_fp4_sparse_decode_fp8_kernel<Kernel<S>, decltype(tp)>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size <= 227 * 1024);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dim3(params.s_q, ENABLE_SPLITKV ? params.num_sm_parts : params.b, 1), dim3(NUM_THREADS, 1, 1), smem_size, params.stream>>>(params, tp);
    KU_CHECK_KERNEL_LAUNCH();
}

template struct Kernel<true>;
template struct Kernel<false>;

}  // namespace kv4mla::f8

namespace kv4mla {
void trace_read_fp8(unsigned long long *host) {
#ifdef KV4MLA_TRACE_ON
    cudaMemcpyFromSymbol(host, f8::trace_buf, sizeof(f8::trace_buf));
#endif
}
void run_glm_fp4_sparse_decode_fp8(const SparseAttnDecodeParams &params) {
    if (params.enable_split_kv) f8::Kernel<true>::run(params);
    else f8::Kernel<false>::run(params);
}
}
