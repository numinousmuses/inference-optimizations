// kv4mla kernel body and host launcher; see glm_fp4_decode.cuh for the design notes.
#include "glm_fp4_decode.cuh"

#ifndef KV4MLA_QK_AHEAD
#define KV4MLA_QK_AHEAD 0
#endif
#ifndef KV4MLA_TMA_LOAD
#define KV4MLA_TMA_LOAD 1   // 1: one TMA gather4 warp loads the raw rows; 0: 64-thread cp.async (measured slower, spills)
#endif
#if KV4MLA_TMA_LOAD
static constexpr int RAW_LOADER_ARRIVALS = 1;
#else
static constexpr int RAW_LOADER_ARRIVALS = 64;
#endif

namespace kv4mla {

#ifdef KV4MLA_TRACE_ON
// Debug timeline of CTA (0, 0): slot [ev][blk % 64] = clock64 (plain stores, no atomics: the probe must not serialize the
// warps it measures). Events 0 index ready, 1 raw TMA issued, 2 dequant saw raw, 3 dequant got its tile, 6 dequant done,
// 4 QK issued, 5 SV issued, 7 softmax saw QK done, 8 softmax S ready, 9 q ready (blk = batch)
__device__ unsigned long long kv4mla_trace_buf[10 * 64];
#define KV4MLA_TRACE(ev, blk) do { if (blockIdx.x == 0 && blockIdx.y == 0) { \
    kv4mla_trace_buf[(ev) * 64 + ((blk) & 63)] = clock64(); } } while (0)
void trace_read(unsigned long long *host, unsigned int *n) {
    cudaMemcpyFromSymbol(host, kv4mla_trace_buf, sizeof(kv4mla_trace_buf));
    *n = 640;
    cudaMemset(0, 0, 0);
}
#else
#define KV4MLA_TRACE(ev, blk) do {} while (0)
void trace_read(unsigned long long *, unsigned int *n) { *n = 0; }
#endif

template<bool S>
template<typename TmaParam>
__device__ void GlmFp4DecodeKernel<S>::devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params) {
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
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
    }

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_last_store_done.init(128);
            plan.bar_q_ready.init(128);     // the dequant warpgroup converts q
            plan.bar_q_utccp.init(1);
            plan.bar_p_empty.init(128);
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_dequant_ready[i].init(128);
                plan.bar_raw_ready[i].init(RAW_LOADER_ARRIVALS);
                plan.bar_raw_free[i].init(128);
                plan.bar_qk_done[i].init(1);
                plan.bar_so_ready[i].init(128);
                plan.bar_sv_done[i].init(1);
            }
            for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
                plan.bar_index_ready[i].init(32);
                plan.bar_index_free[i].init(128 + RAW_LOADER_ARRIVALS);   // softmax warpgroup (valid mask) + raw loaders (coords)
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
        bool bar_phase_batch_rel;
        int topk_length;
        bool is_last_batch;
    };

    auto run_main_loop = [&](auto f) {
        DecodingSchedMeta sched_meta;
        if constexpr (ENABLE_SPLITKV) {
            KU_LDG_256(params.tile_scheduler_metadata_ptr + partition_idx, &sched_meta, ".nc", "no_allocate",
                       "evict_normal", "256B");
        } else {
            sched_meta.begin_req_idx = sched_meta.end_req_idx = partition_idx;
        }
        if (sched_meta.begin_req_idx >= params.b) {
            return;
        }
        bool bar_phase_batch_rel = 0;
        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx, bar_phase_batch_rel ^= 1) {
            int start_block_idx, end_block_idx;
            bool is_split; int n_split_idx;
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
            if constexpr (ENABLE_SPLITKV) {
                start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
                end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : topk_padded / B_TOPK;
                is_split = batch_idx == sched_meta.begin_req_idx ? sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted : false);
                n_split_idx = batch_idx == sched_meta.begin_req_idx ? (__ldg(params.num_splits_ptr + batch_idx) + sched_meta.begin_split_idx) : __ldg(params.num_splits_ptr + batch_idx);
            } else {
                start_block_idx = 0;
                end_block_idx = topk_padded / B_TOPK;
                is_split = false;
                n_split_idx = 0;
            }
            MainLoopArgs args = {batch_idx, start_block_idx, end_block_idx, !is_split, n_split_idx, bar_phase_batch_rel,
                                 topk_length, batch_idx == sched_meta.end_req_idx};
            f(args);
            NamedBarrier(NUM_THREADS, NamedBarriers::everyone_sync).arrive_and_wait_unaligned();
        }
    };

    struct RingState {
        int buf_idx = 0;
        bool bar_phase = 0;
        int index_buf_idx = 0;
        bool index_bar_phase = 0;
        CUTE_DEVICE void update() {
            bar_phase ^= (buf_idx == NUM_BUFS - 1);
            buf_idx = (buf_idx + 1) % NUM_BUFS;
            index_bar_phase ^= (index_buf_idx == NUM_INDEX_BUFS - 1);
            index_buf_idx = (index_buf_idx + 1) % NUM_INDEX_BUFS;
        }
    };
    RingState rs;

    if (warpgroup_idx == 0) {
        // ---------------------------------------------------------------- softmax (scale & exp) warpgroup
        cutlass::arch::warpgroup_reg_alloc<224>();

        constexpr int B_EPI = 64;
        Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o_buf.data()), SmemLayoutOBuf{});
        bf16* sO_bases[B_EPI / 8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI / 8; ++i)
            sO_bases[i] = &sO(idx_in_warpgroup % B_H, (idx_in_warpgroup / B_H) * (PV_GEMM_N / (128 / B_H)) + i * 8);

        const float2 scale = float2{params.sm_scale_div_log2, params.sm_scale_div_log2};
        f16* sS_base = plan.s.data() + lane_idx * 8 + (warp_idx & 1) * (B_H / 2) * 8 + (warp_idx / 2) * B_H * (B_TOPK / 2);
        constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        run_main_loop([&](const MainLoopArgs &args) {
            cute::tma_store_wait<0>();
            plan.bar_last_store_done.arrive();

            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;
            RingState rs_prev = rs;

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                plan.bar_index_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                plan.bar_qk_done[rs.buf_idx].wait(rs.bar_phase);
                ku::tcgen05_after_thread_sync();
                if (idx_in_warpgroup == 0) KV4MLA_TRACE(7, block_idx);

                float p[NUM_ELEMS_PER_THREAD];
                sm100::retrieve_mask_and_reduce_p<NUM_ELEMS_PER_THREAD, NamedBarriers::wg0_warp02_sync,
                                                  NamedBarriers::wg0_warp13_sync, false>(
                    tmem_cols::P, plan.is_token_valid[rs.index_buf_idx], warp_idx, lane_idx,
                    [&]() { plan.bar_p_empty.arrive(); },   // P is in registers: the MMA warp may issue QK(i+1)
                    plan.p_exchange_buf, p);

                float cur_pi_max = sm100::get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= params.sm_scale_div_log2;

                plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                plan.bar_index_free[rs.index_buf_idx].arrive();
                cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup ^ 64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                float new_max, scale_for_old;
                if (!should_scale_o) {
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;

                // S = exp2(p * scale - max) in fp16 (<= 1: 10 mantissa bits)
                __half2 s[NUM_ELEMS_PER_THREAD / 2];
                float2 neg_new_max = float2{-new_max, -new_max};
                float2 cur_sum = float2{0.0f, 0.0f};
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD / 2; i += 1) {
                    float2 d = ku::float2_fma(float2{p[i * 2], p[i * 2 + 1]}, scale, neg_new_max);
                    d.x = exp2f(d.x);
                    d.y = exp2f(d.y);
                    cur_sum = ku::float2_add(cur_sum, d);
                    s[i] = __float22half2_rn(d);
                }
                li = fma(li, scale_for_old, (cur_sum.x + cur_sum.y));

                // QK(i) was issued before SV(i-1): S and O are free only once SV(i-1) completed
                if (block_idx != args.start_block_idx) {
                    plan.bar_sv_done[rs_prev.buf_idx].wait(rs_prev.bar_phase);
                    ku::tcgen05_after_thread_sync();
                }
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD / 8; i += 1) {
                    *(uint128_t*)(sS_base + B_H * 8 * i) = *(uint128_t*)(s + i * 4);
                }

                if (block_idx != args.start_block_idx && should_scale_o) {
                    ku::tcgen05_after_thread_sync();
                    sm100::rescale_O<D_V / (128 / B_H), 64, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                plan.bar_so_ready[rs.buf_idx].arrive();
                if (idx_in_warpgroup == 0) KV4MLA_TRACE(8, block_idx);

                if (block_idx != args.end_block_idx - 1) {
                    rs_prev = rs;
                    rs.update();
                }
            }

            if (real_mi == -CUDART_INF_F) {
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            plan.rowwise_max_buf[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            li += plan.rowwise_max_buf[idx_in_warpgroup ^ 64];

            if (idx_in_warpgroup < params.h_q) {
                if (args.is_no_split) {
                    float cur_lse = fma(mi, CUDART_LN2_F, logf(li));
                    cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                    float* gLse = (float*)params.lse + args.batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q + idx_in_warpgroup;
                    *gLse = cur_lse;
                } else {
                    float cur_lse = log2f(li) + mi;
                    float* gLseAccum = (float*)params.lse_accum + args.n_split_idx * params.stride_lse_accum_split + s_q_idx * params.stride_lse_accum_s_q + idx_in_warpgroup;
                    *gLseAccum = cur_lse;
                }
            }

            plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase);
            rs.update();
            ku::tcgen05_after_thread_sync();

            if (args.is_last_batch) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

            static constexpr int O_FOLD = 2;
            if (args.is_no_split) {
                Tensor tma_gO = flat_divide(
                    tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx, args.batch_idx),
                    Shape<Int<B_H>, Int<64>>{})(_, _, _0{}, _);
                auto thr_tma = tma_params.tma_O.get_slice(_0{});
                Tensor tma_sO = flat_divide(sO, Shape<Int<B_H>, Int<64>>{})(_, _, _0{}, _);

                float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li);
                float2 o_scale_float2 = {o_scale, o_scale};
                float2 o[B_EPI / 2];
                __nv_bfloat162 o_bf16[B_EPI / 2];
                CUTE_UNROLL
                for (int i = 0; i < (D_V / O_FOLD) / B_EPI; ++i) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i * B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 2; ++j) {
                        o[j] = ku::float2_mul(o[j], o_scale_float2);
                        o_bf16[j] = __float22bfloat162_rn(o[j]);
                    }
                    int col_base = (i >= PV_GEMM_N / O_FOLD / B_EPI ? D_V / 2 : 0) + (i * B_EPI % (PV_GEMM_N / O_FOLD));
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 8; ++j)
                        *(__int128_t*)(sO_bases[j] + col_base * B_H) = *(__int128_t*)(&o_bf16[j * 4]);
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
                float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li);
                float2 o_scale_float2 = {o_scale, o_scale};
                constexpr int B_EPI = 64;
                float2 o[B_EPI / 2];
                // each thread owns 64 contiguous floats of one head row per step: 16 x 16-byte stores, whole sectors
                const int row = idx_in_warpgroup % B_H;
                float* gOa = (float*)params.o_accum + (int64_t)args.n_split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q + row * params.stride_o_accum_h_q;
                CUTE_UNROLL
                for (int i = 0; i < (D_V / O_FOLD) / B_EPI; ++i) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i * B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 2; ++j)
                        o[j] = ku::float2_mul(o[j], o_scale_float2);
                    int col_base = (idx_in_warpgroup / B_H) * (PV_GEMM_N / O_FOLD) + (i >= PV_GEMM_N / O_FOLD / B_EPI ? D_V / 2 : 0) + (i * B_EPI % (PV_GEMM_N / O_FOLD));
                    if (row < params.h_q) {
                        CUTE_UNROLL
                        for (int j = 0; j < B_EPI / 4; ++j)
                            *(float4*)(gOa + col_base + j * 4) = *(float4*)(&o[j * 2]);
                    }
                }
            }
        });

        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<72>();
        const int warp_idx = cutlass::canonical_warp_idx_sync();

        if (warp_idx == 4 && elect_one_sync()) {
            // ------------------------------------------------------------ MMA warp
            bool p_phase = 0;   // completions of bar_p_empty consumed (one per block)
            run_main_loop([&](const MainLoopArgs &args) {
                if (args.start_block_idx >= args.end_block_idx) {
                    ku::trap();
                }
                // q converted to fp16 by the dequant warpgroup
                plan.bar_q_ready.wait(args.bar_phase_batch_rel);
                ku::tcgen05_after_thread_sync();
                {
                    UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
                        make_smem_ptr(plan.u.qo.q.data()),
                        tile_to_shape(UMMA::Layout_K_SW128_Atom<f16>{}, Shape<Int<B_H * 2>, Int<64>>{})));
                    #pragma unroll D_Q_SW128/128
                    for (int tile_idx = 0; tile_idx < D_Q_SW128 / 128; ++tile_idx) {
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < B_H / 16; ++subtile_idx) {
                            SM100_UTCCP_128dp256bit_1cta::copy(sQ_desc + (tile_idx * (B_H * 128) + subtile_idx * 16) * 2 / 16,
                                                               tmem_cols::Q + tile_idx * 32 + subtile_idx * 8);
                        }
                    }
                }
                {
                    UMMA::SmemDescriptor sQ_SW64_desc = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
                        make_smem_ptr(plan.u.qo.q_sw64),
                        tile_to_shape(UMMA::Layout_K_SW64_Atom<f16>{}, Shape<Int<B_H * 2>, Int<32>>{})));
                    CUTE_UNROLL
                    for (int tile_idx = 0; tile_idx < D_Q_SW64 / 64; ++tile_idx) {
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < 32 / 16; ++subtile_idx) {
                            SM100_UTCCP_128dp256bit_1cta::copy(sQ_SW64_desc + (tile_idx * (B_H * 64) + subtile_idx * 16) * 2 / 16,
                                                               tmem_cols::Q + (B_H * D_Q_SW128 / 2 / 128) + tile_idx * 16 + subtile_idx * 8);
                        }
                    }
                }
                ku::umma_arrive_noelect(plan.bar_q_utccp);

                TiledMMA tiled_mma_P = TiledMMA_P{};
                TiledMMA tiled_mma_O = TiledMMA_O{};
                Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
                Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
                tP.data().get() = tmem_cols::P;
                tO.data().get() = tmem_cols::O;

                plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                ku::tcgen05_after_thread_sync();

                // QK of block j (its dequantized tile in buffer st.buf_idx) into the single P tile
                auto issue_qk = [&](const RingState &st) {
                    plan.bar_dequant_ready[st.buf_idx].wait(st.bar_phase);
                    ku::tcgen05_after_thread_sync();
                    // QK rope (a 64-wide extra block, dual-GEMM halves of 32)
                    Tensor tQ_rope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_ROPE / 2>>{}));
                    tQ_rope.data().get() = tmem_cols::Q_Tail;
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.kv.dequant[st.buf_idx].rope), SmemLayoutKTiles_DualGemm_SW64<1>{});
#ifndef KV4MLA_SKIP_MMA
                    ku::utcmma_ts(tiled_mma_P, tQ_rope, sK_rope, tP, true);
#endif
                    // QK nope
                    Tensor tQ_nope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_NOPE / 2>>{}));
                    tQ_nope.data().get() = tmem_cols::Q;
                    Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.kv.dequant[st.buf_idx].nope), SmemLayoutKTiles_DualGemm_SW128<512 / 64 / 2>{});
#ifndef KV4MLA_SKIP_MMA
                    ku::utcmma_ts(tiled_mma_P, tQ_nope, sK_nope, tP, false);
#endif
                    ku::umma_arrive_noelect(plan.bar_qk_done[st.buf_idx]);
                };

#if KV4MLA_QK_AHEAD
                issue_qk(rs);
#endif
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
#if KV4MLA_QK_AHEAD
                    // software pipelining: QK(i+1) runs on the tensor core while the softmax warpgroup works on block i
                    if (block_idx + 1 < args.end_block_idx) {
                        plan.bar_p_empty.wait(p_phase);
                        p_phase ^= 1;
                        RingState nx = rs;
                        nx.update();
                        issue_qk(nx);
                    } else {
                        plan.bar_p_empty.wait(p_phase);   // keep the P-empty phase in step with the softmax warpgroup
                        p_phase ^= 1;
                    }
#else
                    issue_qk(rs);
                    KV4MLA_TRACE(4, block_idx);
                    plan.bar_p_empty.wait(p_phase);
                    p_phase ^= 1;
#endif

                    // SV
                    plan.bar_so_ready[rs.buf_idx].wait(rs.bar_phase);
                    ku::tcgen05_after_thread_sync();
                    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].nope), SmemLayoutKTilesTransposed_SW128<D_V / 64>{});
#ifndef KV4MLA_SKIP_MMA
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, block_idx == args.start_block_idx);
#endif
                    ku::umma_arrive_noelect(plan.bar_sv_done[rs.buf_idx]);
                    KV4MLA_TRACE(5, block_idx);

                    rs.update();
                }
            });
#if !KV4MLA_TMA_LOAD
        } else if (warp_idx == 5 || warp_idx == 6) {
            // ------------------------------------------------------------ raw row loaders: 64 threads, cp.async 16 B
            // A block is 64 rows x 22 chunks of 16 B; thread lt takes chunks lt, lt + 64, ... so consecutive lanes copy
            // consecutive chunks (coalesced), 22 per thread, all in flight at once. Invalid rows (coord -1) are zero-filled.
            const int lt = threadIdx.x - 5 * 32;
            constexpr int CHUNKS_PER_ROW = ROW_BYTES / 16;
            const uint8_t *kv = (const uint8_t*)params.kv;
            run_main_loop([&](const MainLoopArgs &args) {
                plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    plan.bar_index_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                    plan.bar_raw_free[rs.buf_idx].wait(rs.bar_phase ^ 1);
                    const uint32_t dst = cute::cast_smem_ptr_to_uint(plan.u.kv.raw[rs.buf_idx].data());
                    const int *coords = plan.tma_coord[rs.index_buf_idx];
                    CUTE_UNROLL
                    for (int k = 0; k < CHUNKS_PER_ROW; ++k) {
                        const int idx = lt + 64 * k;
                        const int r = idx / CHUNKS_PER_ROW, c = idx % CHUNKS_PER_ROW;
                        const int coord = coords[r];
                        const uint8_t *src = kv + (int64_t)max(coord, 0) * ROW_BYTES + c * 16;
                        const uint32_t nbytes = coord >= 0 ? 16u : 0u;
#ifndef KV4MLA_SKIP_LOAD
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                                     :: "r"(dst + r * ROW_BYTES + c * 16), "l"(src), "r"(nbytes) : "memory");
#endif
                    }
                    plan.bar_index_free[rs.index_buf_idx].arrive();
                    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n"
                                 :: "r"(cute::cast_smem_ptr_to_uint(&plan.bar_raw_ready[rs.buf_idx])) : "memory");
                    if (lt == 0) KV4MLA_TRACE(1, block_idx);
                    rs.update();
                }
            });
#else
        } else if (warp_idx == 5 && elect_one_sync()) {
            // ------------------------------------------------------------ raw row TMA warp
            run_main_loop([&](const MainLoopArgs &args) {
                // raw buffers overlap the split-KV o_accum staging of the previous segment, not q
                plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    plan.bar_index_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                    plan.bar_raw_free[rs.buf_idx].wait(rs.bar_phase ^ 1);
                    int4 cur_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + 0);
                    int4 nxt_indices;
#ifdef KV4MLA_SKIP_LOAD
                    if (false)
#endif
                    CUTE_UNROLL
                    for (int row = 0; row < B_TOPK; row += 4) {
                        if (row + 4 < B_TOPK)
                            nxt_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + row + 4);
                        ku::tma_gather4(&tma_params.tensor_map_kv, plan.bar_raw_ready[rs.buf_idx],
                                        plan.u.kv.raw[rs.buf_idx].data() + ROW_BYTES * row, 0, cur_indices,
                                        (int64_t)TMA::CacheHintSm90::EVICT_FIRST);
                        cur_indices = nxt_indices;
                    }
#ifdef KV4MLA_SKIP_LOAD
                    plan.bar_raw_ready[rs.buf_idx].arrive();
#else
                    plan.bar_raw_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK * ROW_BYTES);
#endif
                    KV4MLA_TRACE(1, block_idx);
                    plan.bar_index_free[rs.index_buf_idx].arrive();
                    rs.update();
                }
            });
#endif
        } else if (warp_idx == 7) {
            // ------------------------------------------------------------ index warp: TMA row coords + valid mask
            static_assert(B_TOPK == 64);
            run_main_loop([&](const MainLoopArgs &args) {
                const int* indices = (const int*)params.indices + params.stride_indices_b * args.batch_idx + params.stride_indices_s_q * s_q_idx;
                auto load_block_indices = [&](int block_idx) -> int2 {
                    return __ldg((const int2*)(indices + block_idx * B_TOPK + lane_idx * 2));
                };
                int2 my_indices = load_block_indices(args.start_block_idx);
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    const bool has_next = block_idx + 1 < args.end_block_idx;
                    const int2 next_indices = has_next ? load_block_indices(block_idx + 1) : int2{};
                    const int abs_pos = block_idx * B_TOPK + lane_idx * 2;
                    int tma_coords[2];
                    char valid_mask = 0;
                    CUTE_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        const int cur_idx = i == 0 ? my_indices.x : my_indices.y;
                        // -1 padding, slots outside the pool, and positions past topk_length are masked; their TMA
                        // coordinate is -1 (out of bounds -> zero fill, never NaN)
                        const bool valid = cur_idx >= 0 && cur_idx < params.num_blocks && abs_pos + i < args.topk_length;
                        valid_mask |= valid << i;
                        tma_coords[i] = valid ? cur_idx : -1;
                    }
#ifdef KV4MLA_PREFETCH
                    {
                        // L2 prefetch of this block's rows now, blocks ahead of the raw TMA warp (4 rows per gather4:
                        // even lanes combine their 2 coordinates with the next lane's)
                        const int nb_x = __shfl_down_sync(0xFFFFFFFF, tma_coords[0], 1);
                        const int nb_y = __shfl_down_sync(0xFFFFFFFF, tma_coords[1], 1);
                        if (lane_idx % 2 == 0)
                            ku::tma_gather4_prefetch(&tma_params.tensor_map_kv, 0, int4{tma_coords[0], tma_coords[1], nb_x, nb_y},
                                                     (int64_t)TMA::CacheHintSm90::EVICT_NORMAL);
                    }
#endif
                    valid_mask <<= lane_idx % 4 * 2;
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                    plan.bar_index_free[rs.index_buf_idx].wait(rs.index_bar_phase ^ 1);
                    *(int2*)(plan.tma_coord[rs.index_buf_idx] + lane_idx * 2) = *(int2*)tma_coords;
                    if (lane_idx % 4 == 0)
                        plan.is_token_valid[rs.index_buf_idx][lane_idx / 4] = valid_mask;
                    plan.bar_index_ready[rs.index_buf_idx].arrive();
                    if (lane_idx == 0) KV4MLA_TRACE(0, block_idx);
                    rs.update();
                    if (has_next) my_indices = next_indices;
                }
            });
        } else {
            run_main_loop([&](const MainLoopArgs &args) {});
        }
    } else {
        // ---------------------------------------------------------------- dequant warpgroup (also converts q)
        cutlass::arch::warpgroup_reg_alloc<208>();
        static_assert(NUM_BUFS == 2);
        const Dequantizer dq(idx_in_warpgroup);
        const uint8_t *raw_0 = plan.u.kv.raw[0].data(), *raw_1 = plan.u.kv.raw[1].data();
        const uint32_t nope_0 = cute::cast_smem_ptr_to_uint(plan.u.kv.dequant[0].nope);
        const uint32_t nope_1 = cute::cast_smem_ptr_to_uint(plan.u.kv.dequant[1].nope);
        const uint32_t rope_0 = cute::cast_smem_ptr_to_uint(plan.u.kv.dequant[0].rope);
        const uint32_t rope_1 = cute::cast_smem_ptr_to_uint(plan.u.kv.dequant[1].rope);
        Tensor sQ = make_tensor(make_smem_ptr(plan.u.qo.q.data()), SmemLayoutQ_SW128{});
        Tensor sQt = make_tensor(make_smem_ptr(plan.u.qo.q_sw64), SmemLayoutKTiles_SW64<2>{});

        run_main_loop([&](const MainLoopArgs &args) {
            // q [64, 576] E4M3 -> fp16 SW128 (dims 0..511) + SW64 tail (dims 512..575). 4608 chunks of 8 values;
            // consecutive threads take consecutive chunks (coalesced LDG.64). Heads >= h_q are zero.
            {
                const uint8_t *gq = (const uint8_t*)params.q + (int64_t)args.batch_idx * params.stride_q_b + (int64_t)s_q_idx * params.stride_q_s_q;
                constexpr int CHUNKS = B_H * D_Q / 8, PER_ROW = D_Q / 8, INNER = 12;
                static_assert(CHUNKS % (128 * INNER) == 0);
                CUTE_NO_UNROLL
                for (int k0 = 0; k0 < CHUNKS / 128; k0 += INNER)
                CUTE_UNROLL
                for (int k = k0; k < k0 + INNER; ++k) {
                    const int c = k * 128 + idx_in_warpgroup;
                    const int h = c / PER_ROW, col = (c % PER_ROW) * 8;
                    uint2 w = h < params.h_q ? __ldg((const uint2*)(gq + (int64_t)h * params.stride_q_h_q + col)) : uint2{0u, 0u};
                    uint32_t v[4];
                    v[0] = e4m3x2_to_f16x2((uint16_t)(w.x & 0xFFFF));
                    v[1] = e4m3x2_to_f16x2((uint16_t)(w.x >> 16));
                    v[2] = e4m3x2_to_f16x2((uint16_t)(w.y & 0xFFFF));
                    v[3] = e4m3x2_to_f16x2((uint16_t)(w.y >> 16));
                    f16 *dst = col < D_Q_SW128 ? &sQ(h, col) : &sQt(h, col - D_Q_SW128);
                    sts128(cute::cast_smem_ptr_to_uint(dst), v);
                }
                fence_view_async_shared();
                plan.bar_q_ready.arrive();
                if (idx_in_warpgroup == 0) KV4MLA_TRACE(9, args.batch_idx);
            }
            // the dequant tiles hold q until the UTCCP copied it into TMEM
            plan.bar_q_utccp.wait(args.bar_phase_batch_rel);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                plan.bar_raw_ready[rs.buf_idx].wait(rs.bar_phase);
                if (idx_in_warpgroup == 0) KV4MLA_TRACE(2, block_idx);
                typename Dequantizer::Staged staged;
                dq.load(rs.buf_idx == 0 ? raw_0 : raw_1, staged);
                plan.bar_raw_free[rs.buf_idx].arrive();   // release semantics: the reads above are done
                plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase ^ 1);
                if (idx_in_warpgroup == 0) KV4MLA_TRACE(3, block_idx);
#ifndef KV4MLA_SKIP_DEQUANT
                dq.store(staged, rs.buf_idx == 0 ? nope_0 : nope_1, rs.buf_idx == 0 ? rope_0 : rope_1);
#endif
                fence_view_async_shared();
                plan.bar_dequant_ready[rs.buf_idx].arrive();
                if (idx_in_warpgroup == 0) KV4MLA_TRACE(6, block_idx);
                rs.update();
            }
        });
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("kv4mla: this kernel only supports sm100a");
    }
#endif
}

template<typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
glm_fp4_sparse_decode_kernel(__grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TmaParams tma_params) {
    Kernel::devfunc(params, tma_params);
}

template<bool S>
void GlmFp4DecodeKernel<S>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.topk % B_TOPK == 0, "topk (%d) mod B_TOPK (%d) must be 0", params.topk, B_TOPK);
    KU_ASSERT(params.h_q == 64);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_Q);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.stride_kv_row == ROW_BYTES, "kv rows must be %d bytes apart", ROW_BYTES);
    KU_ASSERT((int64_t)params.kv % 16 == 0);
    KU_ASSERT(params.stride_q_h_q % 8 == 0 && params.stride_q_s_q % 8 == 0 && params.stride_q_b % 8 == 0 && (int64_t)params.q % 8 == 0,
              "q must allow 8-byte loads");

    auto shape_O = make_shape(params.h_q, D_V, params.s_q, params.b);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr((bf16*)params.out),
                    make_layout(shape_O, make_stride(params.stride_o_h_q, _1{}, params.stride_o_s_q, params.stride_o_b))),
        SmemLayoutOBuf_TMA{});

    // num_blocks carries the number of pool rows (kv is [rows, 352] uint8, one row per slot)
    CUtensorMap tensor_map_kv = ku::make_tensor_map(
        {(uint64_t)ROW_BYTES / 4, (uint64_t)params.num_blocks},
        {(uint64_t)ROW_BYTES},
        {ROW_BYTES / 4, 1},
        params.kv,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);

    TmaParams<decltype(shape_O), decltype(tma_O)> tma_params = {shape_O, tma_O, tensor_map_kv};
    auto kernel = &glm_fp4_sparse_decode_kernel<GlmFp4DecodeKernel<S>, decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size <= 227 * 1024);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dim3(params.s_q, ENABLE_SPLITKV ? params.num_sm_parts : params.b, 1), dim3(NUM_THREADS, 1, 1), smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

template struct GlmFp4DecodeKernel<true>;
template struct GlmFp4DecodeKernel<false>;

void run_glm_fp4_sparse_decode(const SparseAttnDecodeParams &params) {
    if (params.enable_split_kv) GlmFp4DecodeKernel<true>::run(params);
    else GlmFp4DecodeKernel<false>::run(params);
}

}  // namespace kv4mla
