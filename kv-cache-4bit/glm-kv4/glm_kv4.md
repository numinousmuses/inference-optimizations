# GLM-5.3 4-bit KV cache (kv4glm)

Engine feature for SGLang v0.5.20 on 8xB200: the DSA latent KV rows of GLM-5.3 stored in NVFP4-style 4-bit
(E2M1 values, one E4M3 scale per 16) instead of FP8, read back through the served TRT-LLM sparse MLA op. Everything
is behind `NUMINOUS_GLM_KV4=1`; the arms are `kv4glm_ctl`, `kv4glm_scratch`, `kv4glm_triton`
(operations/modal/kv4glm_arms.py, on the fused DP-MoE lead `glm_v20_dpmoe`).

## Files

| file | what |
|---|---|
| operations/kernels/glm_kv4.py | the 352-byte row, torch reference, Triton kernels (quantize-write, dequantize gather, sequence dequant, dedup, fused sparse MLA) |
| operations/kernels/glm_kv4_engine.py | `GLMKV4DSATokenToKVPool`, `attach`, `engine_forward` (scratch and triton read paths), scratch buffers |
| operations/patches/glm_kv4_hooks.py | 7 anchored hooks (kv_cache_configurator x2, dsa_backend x5) + module install |
| operations/kernels/test_glm_kv4.py | CPU reference tests, GPU bit-exact kernel tests, fused kernel vs fp32, graph replay |
| operations/patches/test_glm_kv4_hooks.py | anchors/idempotence/import, the real pool class, the production `_forward_trtllm` sequence FP8 vs FP4 (3 read paths) and decode under CUDA-graph capture |
| operations/kernels/glm_kv4_bench.py | per-layer graph-timed comparison FP8 TRT-LLM vs the FP4 paths |
| operations/modal/kv4glm_arms.py, test_kv4glm_arms.py | the gate arms |

## Row layout (source_code)

Served today (trtllm DSA roles, SM100): `[512 E4M3 latent | 64 E4M3 rope]` = 576 B/token/layer.
FP4 row: `[256 B packed E2M1 latent (element 2i in the low nibble) | 32 B E4M3 block scales | 64 B E4M3 rope]`
= 352 B/token/layer. Quantization from the bf16 latent (the hook keeps it before the engine's FP8 quantize+RoPE
kernel): scale = round_e4m3(clamp(amax/6, 2^-9, 448)), code = nearest E2M1 with ties down, saturate at 6, no
negative zero. The indexer cache (128 FP8 + 4 scale bytes per layer) is unchanged.

Cell bytes per token (derived; 78 target layers + 1 NextN draft layer, each with its indexer cache):
FP8 79 x (576 + 132) = 55,932 B; FP4 79 x (352 + 132) = 38,236 B; ratio 1.463x tokens at the same pool bytes.
The brief's 1.6x counts the latent only; the indexer cache is 27% of the FP4 cell.

## Read paths

`scratch` (default): the selected rows (top-2048 per token) are dequantized into an FP8 scratch in the raw 576-byte
layout and the unchanged TRT-LLM op runs on the scratch with the table remapped to scratch rows (identity for the
plain gather, owner rows with dedup, slot->position map for the prefill sequence path). Plain prefill (EXTEND /
MIXED) dequantizes each request's whole context once (sum(seq_lens) rows) instead of T x 2048 rows. The scratch
values carry one more E4M3 rounding (product of an E2M1 code and an E4M3 scale has up to 5 mantissa bits); measured
on Gaussian rows the L1 relative error goes 0.0897 -> 0.0916 (FP8 alone: 0.0225). An E8M0 scale would make that
rounding exact but costs 0.111 by itself, so E4M3 stays.

`triton`: the fused kernel dequantizes in-kernel (exact, no requantization), flash-decoding split over the selected
rows plus a combine kernel. Correct (rel 0.002-0.003 vs fp32) but slow (below).

## Measured (b200x1-kv4glm, B200, CUDA-graph replay medians, us per layer call, pool 600k slots)

| case | rows | fp8 trtllm | fp4 scratch gather (dequant) | fp4 scratch dedup (dequant) | fp4 sequence (dequant) | fp4 triton |
|---|---|---|---|---|---|---|
| decode bs1 | 1 | 8.6 | 11.9 (2.2) | 28.5 (19.8) | | 77 |
| decode bs8 | 8 | 11.1 | 17.0 (4.7) | 34.3 (23.3) | | 85 |
| decode bs16 | 16 | 14.3 | 23.0 (8.0) | 40.7 (26.8) | | 114 |
| decode bs32 | 32 | 17.4 | 33.9 (14.1) | 53.8 (35.7) | | 217 |
| decode bs48 | 48 | 23.0 | 48.8 (20.9) | 70.0 (43.4) | | 325 |
| verify bs16x4 | 64 | 25.8 | 58.2 (27.1) | 60.7 (34.2) | | 388 |
| verify bs48x4 | 192 | 68.8 | 163.5 (76.5) | 145.0 (71.3) | | 1177 |
| prefill 512 tok, 2 x 64k ctx | 512 | 131.7 | 367 (203) | 272 (136) | 172 (33) | 2765 |
| prefill 2048 tok, 2 x 64k ctx | 2048 | 463 | 1458 (792) | 937 (474) | 492 (33) | 11100 |

Raw: operations/kernels/glm_kv4_bench_b200.json. The engine uses: plain gather for decode, dedup for verify /
draft extend / prefill fallback, sequence for plain prefill.
Verify and prefill rows model 90% shared top-k inside a request (the dedup case). The gather is HBM-bound
(393k rows x 928 B = 364 MB in 78 us = 4.7 TB/s; tile shapes, alignment hints, packed stores and scale-gather
variants all land within 10%); the FP8 op reads 226 MB in 69 us. The scratch path therefore moves ~2.6x the bytes
of the FP8 path and costs 2.1-2.2x its time at decode and verify, and 0.9-1.3x at prefill through the sequence path.

Derived step cost for the standard cell (DP8, 48 users per rank, EAGLE 3/1/4: the target step is a 192-row verify):
78 layers x (145 - 69) us = +5.9 ms per step over a ~40 ms step, about +15% (dedup at the simulated 90% overlap;
the plain-gather bound is 78 x (163.5 - 69) = +7.4 ms, +18%). Pool: 1.46x tokens. Whether more resident sessions at
96-160 users outweigh the slower step is the ladder's question, not decided here.

## Accuracy (measured on random Gaussian rows, kernel-level)

Fused triton vs fp32 over the FP4 pool: 0.0020 (max err / max |ref|). Scratch paths vs fp32 over the requantized
rows: 0.0091 (the FP8 TRT-LLM op itself: 0.0082). FP4 output vs FP8 output gap on random data: 0.21 by the same
max metric; the in-engine accuracy gate vs kv4glm_ctl (GLM noise floor KL 0.177 / top-1 0.90) decides.

## In-engine results

See research/glm-kv4-4bit-kv.md: pool 1.576x at boot, the FP8 pool wall at 64 x 204k-token sessions (2.4 tok/s per
user) becomes 26.9 with FP4, accuracy at the GLM noise floor, 524K-context findings and the prefill-cap fix.

## Protocol for the coordinator

1. preflight (done on b200x1-kv4glm: kv_cache_dtype fp8_e4m3, dsa roles trtllm/trtllm, page 64: what `attach`
   requires; the arms change no flags),
2. boot kv4glm_scratch: expect the log line `numinous glm_kv4 pool: 78 layers x 352 B/token ... = 27456 B/token vs
   44928` on the target and `1 layers x 352` on the draft, `numinous glm_kv4 attention active: mode=scratch`, and
   a KV pool of ~1.46x the control's tokens,
3. accuracy gate vs kv4glm_ctl, 4. ladder 64/96/128/160 sessions with kv4glm_ctl on the same node,
5. quote check if it wins. kv4glm_triton is the exact-dequant reference at small batch (1-8 users per rank).

Knobs: NUMINOUS_GLM_KV4_ATTN scratch|triton, NUMINOUS_GLM_KV4_DEDUP 1|0, NUMINOUS_GLM_KV4_SEQ_ROWS (prefill
sequence scratch cap, default 524288 rows = 302 MB), NUMINOUS_GLM_KV4_SEL_ROWS (eager selected-row scratch, default
256 tokens = 302 MB). Graph scratch = max graph tokens x 2048 x 576 B (192 rows: 226 MB per backend).
