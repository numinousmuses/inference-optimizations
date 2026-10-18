# kv4mla: native NVFP4 sparse MLA attention for GLM-5.3 (SM100)

Reads kv4glm's 4-bit DSA KV rows (352 B/token: 512 E2M1 latent + 32 E4M3 scales per 16 + 64 E4M3 rope; layout in
operations/kernels/glm_kv4.py) straight from the pool and runs sparse MLA decode/verify on them in one CUDA kernel,
instead of dequantizing the selected rows into an FP8 scratch for the TRT-LLM op (kv4glm's scratch path, 2.1x FP8 time).
Interface = the served TRT-LLM call: q [T, 64, 576] FP8 with RoPE applied, table [T, K] pool slots (-1 padded), bmm1
scale; out [T, 64, 512] bf16.

## Files

| file | what |
|---|---|
| csrc/glm_fp4_decode_fp8.cu | default kernel (`KV4MLA_MODE=fp8`): E4M3 tensor-core math (UMMA kind::f8f6f4) |
| csrc/glm_fp4_decode.{cuh,cu} | exact kernel (`KV4MLA_MODE=exact`): fp16 math over exactly dequantized rows |
| csrc/api.cpp | torch bindings: `sparse_decode`, `sched_meta`, `num_sm_parts`, `trace` (debug) |
| csrc/flashmla/ | vendored from deepseek-ai/FlashMLA ba89a34 (MIT): kerutils, split-KV scheduler, combine kernel, helpers |
| kv4mla.py | JIT build (sm_100a, cached by source hash), `Decoder` (schedules + workspaces, graph-safe), `split_policy` |
| kv4mla_engine.py | engine glue installed by operations/patches/kv4mla_hooks.py |
| test_kv4mla.py | GPU tests, both modes (56) |
| bench_kv4mla.py, bench_kv4mla_b200.json | per-layer graph-timed comparison vs TRT-LLM FP8 and the kv4glm scratch path |
| probe_*.py/.sh | diagnostics used to find the bottlenecks (FlashMLA baseline, ablations, clock64 timeline) |

Both kernels derive from FlashMLA's SM100 head64 sparse decode kernel (warp specialization, TMEM plan, split-KV).

## Served result (measured, 8xB200, gate-kv4mla-7f3a, one node, real 8-turn sessions, glm_v20_dpmoe base)

Warm tok/s per user (repeat 0 / repeat 1), FP8 control vs kv4glm scratch vs this kernel (FP4 pool 2,447,552 tokens/rank
vs 1,552,832 FP8):

| sessions | FP8 control | FP4 scratch | FP4 native (kv4mla) |
|---|---|---|---|
| 64 | 37.43 | 33.18 / 32.94 | 37.88 / 37.56 |
| 96 | 23.53 | 21.84 | 21.75 / 22.66 |
| 128 | 15.65 | 17.52 | 18.82 / 17.73 |

Steady aggregate at 128: native 1,505 vs FP8 1,146 vs scratch 1,351 tok/s. Accuracy vs the control: native KL 0.181,
top-1 0.897, GSM8K 0.970 (control 0.975), at GLM's control-vs-control floor (KL 0.177 / top-1 0.90). At ~204k-token
sessions (native 27.66 / 24.11 / 22.81 tok/s/user at 48 / 64 / 80) the FP8 control collapses from 64 sessions; there, with
6-8 users per rank, the kernel's small-batch floor makes it no faster than scratch (use NUMINOUS_GLM_KV4MLA_MIN_T=16).

## Design of the fp8 kernel (source_code)

- One TMA gather4 row per token brings latent, scales and rope together (no per-token scale loads).
- Dequant warpgroup: E2M1 x E4M3 scale in fp16 (exact), rounded once to E4M3 (the rounding kv4glm's scratch path also
  applies), rope bytes copied; E4M3 tiles are half the size of fp16 tiles, so 4 tile buffers + 3 raw buffers fit.
- q (FP8) arrives by TMA and is copied to TMEM duplicated into both lane halves (tcgen05.cp 64x128b.warpx2::02_13), so
  QK is a weight-stationary M=64 N=64 MMA whose output lanes 0-63 / 64-127 hold tokens 0-31 / 32-63 of the same heads:
  each softmax thread owns 32 finished scores (no dual-GEMM partial-sum exchange).
- P stored E4M3 x 128; FlashMLA's lazy O rescale threshold lowered from 2^6 to 2^1 so P never saturates E4M3.
- QK(i+1) overlaps softmax(i) (2 P tiles in TMEM, 2 S tiles in SMEM); a correction warpgroup rescales O in TMEM when a
  head's running max grows, off the softmax critical path. 512 threads, setmaxnreg 200/56/128/128.
- Split-KV (FlashMLA scheduler + combine) or one CTA per token by `split_policy(T)` (measured table below).

## Measured (b200x1-kv4mla, B200, us per layer call, CUDA-graph replay, pool 600k rows, top-2048, 64 heads)

| rows T | FP8 TRT-LLM | kv4mla fp8 | kv4mla exact (fp16) | kv4glm FP4 scratch |
|---|---|---|---|---|
| decode 1 | 8.8 | 22.1 | 21.9 | 12.0 |
| decode 8 | 10.9 | 22.6 | 23.7 | 17.0 |
| decode 32 | 17.1 | 27.6 | 30.6 | 33.5 |
| decode 48 | 22.6 | 30.4 | 36.4 | 48.8 |
| decode 64 | 26.8 | 34.9 | 42.0 | 61.5 |
| decode 96 | 39.7 | 43.5 | 56.4 | 87.5 |
| decode 128 | 42.3 | 44.1 | 61.5 | 104.9 |
| verify 192 (48x4) | 69.3 | 74.5 | 94.5 | 163.3 |
| verify 288 (48x6) | 70.2 | 91.0 | 126.1 | 209.4 |
| verify 384 (96x4) | 105.3 | 127.5 | 178.7 | 278.9 |

Raw: bench_kv4mla_b200.json. Bytes (derived): FP4 reads 352 B/row vs 576; achieved 0.9-2.6 TB/s: the kernel is not
bandwidth-bound. Derived step cost for the DP8 standard cell (78 layers, 192-row verify): +0.4 ms over FP8 (scratch path:
+7.4 ms). Small batches (T <= 8) sit on a ~22 us floor (FlashMLA's own FP8 kernel: 21 us at T=1).

Accuracy (measured, Gaussian rows): fp8 mode vs fp32 over the E4M3-rounded rows 0.014 (TRT-LLM FP8 op on the same rows:
0.009); vs fp32 over the exact FP4 rows 0.052 (kv4glm scratch 0.056). Exact mode vs exact FP4 rows 0.002.

## What limits it now (measured; probe_dq.sh, probe_ablate.sh, probe_trace.py; ncu cannot read counters in the sandbox)

- Per-CTA serial latency, not bandwidth or tensor throughput: a single CTA alone on the GPU (T=1, no split) runs at the
  same ~2,200 cycles per 64-token block as T=128 (clock64 timeline), and the pool size (7 MB L2-resident vs 211 MB) does
  not change it. TMA gather of one block's 64 rows takes ~4,500 cycles from issue to data even unloaded.
- Changed nothing (+-3%): raw buffers 2..5, tile buffers 3..4, a second dequant warpgroup, halving the PV work, removing
  the tile stores, replacing the conversion instructions with plain ALU ops, loading the rows with 64-thread cp.async.
- Helped: E4M3 MMA instead of fp16 (61.5 -> 47 us at T=128), overlapping QK(i+1) with softmax(i), non-dual QK (no
  cross-warp P exchange), moving the O rescale to a correction warpgroup (47 -> 44), raising the lazy-rescale threshold
  (2^6: 44 -> 42, but P then needs a smaller E4M3 scale; kept at 2^1 for accuracy).
- Handoff latency dominates: letting one lane poll each barrier (then __syncwarp) in the dequant/correction warps made it
  1.66x slower (44 -> 73 us at T=128). A clock64 timeline at T=1 shows the softmax/MMA chain at ~1,250 cycles per block
  once the dequant warpgroup has finished, ~2,100 while it runs.
- Compiling out the whole dequant (tiles stale) gives 26.5 us at T=128, but that also removes all data-dependent work
  (constant scores: no rescales), so it is an upper bound, not an attribution.
- Small batches (T <= 16) sit on a ~22 us floor (FlashMLA's own FP8 kernel: 21 us at T=1); NUMINOUS_GLM_KV4MLA_MIN_T=16
  routes them to kv4glm's scratch path (T=8: 17.0 vs 22.6 us).
Next levers: fewer handoffs per token (TRT-LLM's 128-token tiles: needs a single P tile in TMEM and one 45 KB raw buffer to
fit), and a scheduler overhead tuned per T (KV4MLA_SCHED_OVERHEAD=10: T=8 22.4 -> 19.3 us, T=16 25.9 -> 22.3, but T=48
30.4 -> 32.7, so not enabled).

## Engine

`NUMINOUS_GLM_KV4=1 NUMINOUS_GLM_KV4MLA=1` (arms kv4mla_ctl / kv4mla_scratch / kv4mla_native in
operations/modal/kv4mla_arms.py). Decode, verify, draft extend and prefill batches the sequence scratch cannot hold go to
the native kernel; plain prefill keeps kv4glm's sequence path (level with FP8). Schedules are built before graph capture
(no FP8 graph scratch is reserved; with NUMINOUS_GLM_KV4MLA_MIN_T=N a scratch for N tokens is reserved for the small
batches it routes to kv4glm's path). The patcher compiles the extension at image build with the CUTLASS 4.5 headers
flashinfer bundles. Boot lines: `numinous kv4mla native FP4 sparse MLA active ... mode fp8`, `numinous kv4mla: graph
schedules for N batch sizes`.

## Dense mode (k27, 2026-09-29): dense MLA over a paged FP4 pool

Added for Kimi K2.7 Code (dense MLA, 64 heads, TP8 = 8 q heads per rank), behind two params fields; the sparse entry
point is unchanged (`page_block_size` stays 1 there).

- `page_block_size > 1`: `indices` is the backend's page table `[b, n_pages]` (SGLang `block_kv_indices`). The index
  warp maps position p to pool row `table[p / ps] * ps + p % ps`, valid while `p < topk_length[b]` (= the sequence
  length including its own new rows), so the KV of one sequence is gathered by the same TMA row path as the sparse
  top-k, without materialising a slot table.
- `row_limit [b, 64]` (optional): q row h attends positions `< row_limit[b, h]`, masked in the softmax warps next to the
  block validity mask. This lets one 64-row q tile hold several tokens of one sequence (row = t * heads + h) with
  causal masking over their own draft rows: EAGLE3 verify with 4 draft tokens x 8 heads = 32 rows reads the KV once per
  sequence per layer instead of once per token.
- API: `api.cpp dense_decode(q [b, 64, 576] fp8, kv rows, page_table, seq_lens, page_size, max_seq_len, row_limit,
  sm_scale, out, lse, split-KV workspaces)`; `kv4mla.DenseDecoder` (schedule from the real lengths each call, so it is
  graph-capturable with changing seq_lens; `topk` = max_seq_len rounded to 64 bounds the block loop) and
  `kv4mla.dense_reference` (fp32 over exactly dequantized rows). fp8-MMA kernel only.
- Tests: `test_kv4mla_dense.py` (single-token vs reference at page 32/64 with and without split-KV; dense == sparse on
  an identity slot table; the verify fold == four single-token calls; zero-limit rows finite and zero; graph replay with
  shorter lengths; the sparse mode unchanged).
- Engine glue: `operations/kernels/k27_kv4_engine.py`, hooks `operations/patches/k27_kv4_hooks.py`.
