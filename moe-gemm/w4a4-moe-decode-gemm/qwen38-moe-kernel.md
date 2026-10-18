# Qwen 3.8 MoE decode GEMM: a Triton W4A4 grouped kernel measured against the byte roof

Status 2026-09-20. All numbers `measured` on one Modal B200 (SM100, 7.7 TB/s reported HBM) in the
`lmsysorg/sglang:qwen38` image, kernel-only CUDA-event timing over 20 iterations, real top-10 routing over
512 experts, correctness checked against a bf16 oracle at the quantised weights on every arm. Source:
`operations/kernels/moe_decode_gemm.py`, bench `operations/modal/moe_kernel_bench.py`, raw
`research/gpu-evidence/moe_decode_gemm_bench_{gateup,down}.json`, CPU tests
`operations/kernels/test_moe_decode_gemm.py` (9 pass).

## Why this kernel

Qwen 3.8 routes 10 of 512 experts per token. With speculative decoding (4 drafts) a decode step at c=32
carries 128 token slots and touches `1 - (1 - 10/512)^128 = 92%` of the experts. Past c=128 it touches all
of them. So a decode step reads essentially the whole 1.2 TB of NVFP4 expert weight, ~147 GB per GPU under
TP8, and the roofline is ~18 ms at 8 TB/s. The measured step at c=32 was 26 ms. Whatever kernel streams
those bytes sets the price of the model. The earlier roofline in `qwen38-architecture.md` (95B active
params per token) only holds at c=1 and is superseded (`derived`).

Per-GPU shapes under TP8, per layer: gate+up `[m_e, 8192] x [8192, 512]` (1.07 GB fp4), down
`[m_e, 256] x [256, 8192]` (0.54 GB), `m_e` = tokens routed to expert e, 1 to ~30.

## Results, 128 token slots (c=32 x 4 drafts), 92-94% of experts touched

Gate+up shape, roof 0.139 ms:

| kernel | ms | of HBM | note |
|---|---|---|---|
| `torch._grouped_mm` bf16 | 0.778 | 67% | 2 bytes/param |
| ours, Triton bf16 (m16 n128 k256) | 0.646 | 81% | 1.2x torch |
| ours, W4A16 (bf16 act, fp4 weight, m16) | 0.725 | 19% | unpack-bound, dead |
| ours, W4A4 m128 row-major | 0.367 | 38% | fp4 MMA needs M=128 |
| ours, W4A4 tile-major | 0.244 | 57% | one contiguous block per k-slab |
| ours, W4A4 tile-major + TMA | 0.242 | 57% | no change: fetch is not the limit |
| ours, W4A4 pre-transposed tiles | 1.264 | 11% | dead: MMA wants K-major |
| **ours, W4A4 operand-swapped (m16 n128 k256 w4 s3)** | **0.202** | **69%** | 3.9x torch, bit-exact vs quantised oracle |

Down shape, roof 0.069 ms:

| kernel | ms | of HBM |
|---|---|---|
| `torch._grouped_mm` bf16 | 0.620 | 42% |
| ours, W4A4 swapped, one N tile per program (n256) | 0.167 | 41% |
| **ours, W4A4 swapped, 8 N tiles per program** | **0.138** | **50%** |

Full load (768 slots, every expert, ~15 rows each): gate+up 0.225 ms (66%), down 0.168 ms (44%); torch
0.855 / 0.671.

Per step, 92 layers, on B200: ours `92 x (0.202 + 0.138) = 31 ms` at 128 slots vs roof 19 ms; torch bf16
would be 129 ms. Scaling to B300 (same HBM class) the MoE alone with our kernel is ~31 ms, which is more
than the whole measured 26 ms step, so the shipped `flashinfer_trtllm` fp4 MoE is already faster than ours.
The sweep's per-kernel profile will give its exact per-layer time; until then this kernel is a measured
floor for a Triton path, not a candidate to ship.

## What moved the number and what did not

1. Weight format decides the regime. With bf16 activations, Triton unpacks every 4-bit weight to bf16 in
   registers; at M=16 that unpack is the whole kernel (19% of HBM). Quantising activations to e2m1 too
   (W4A4, as TRT-LLM's fp4 MoE and MegaMoE do) puts the contraction on the native fp4 tensor-core path.
2. The fp4 MMA needs M of 128 (M=64 and `num_warps=2` blow up to 255 registers with spills; N=64 tiles
   likewise). Padding the token dimension to 128 wastes compute, which is free here, and went 0.72 to
   0.37 ms.
3. Tile-major weight layout (the pre-shuffle TRT-LLM does at load): 0.37 to 0.24 ms.
4. Operand swap (weights on the MMA's M side, tokens on its N side, N=16 legal): removes the 16 KB padded
   activation tile from every pipeline stage. 0.24 to 0.20 ms. Resource usage after: 40 registers, no
   spills, 39 KB smem, several CTAs per SM, so occupancy is not what caps it at 69%.
5. Did not help: TMA for the weight tile (identical), more pipeline stages (worse past 3-4), BLOCK_K 512,
   pre-transposed tiles (5x worse), grouping more than 8 N tiles per program on the down shape.
6. Python in the launch path costs more than the kernel: the first host-side work list took 0.49 ms per
   call. It is now vectorised (`repeat_interleave`) and, in a server, would be one kernel per layer
   alongside the routing sort.

## What the shipped stack's profile says (first read, 2026-09-20)

Sweep v3 `kv_fp8` arm, 8xB300, c=32 with 8k prompts and 400 outputs, 40-step window; the window mixes
prefill and decode and the first classifier left 57% unlabelled, so these are the re-bucketed shares of the
same trace (`measured`, mixed-phase):

| family | share | top kernel |
|---|---|---|
| MoE grouped GEMM (TRT-LLM `bmm_*E2m1E2m1*`) | 30% | 17.0% gate+up tile 128x192x256, 11.3% down |
| dense GEMM (cuBLAS `nvjet`) | 26% | 16.7% `128x256 2cta` (prefill-shaped tiles) |
| collective (`MoeFinalizeAllReduceRMSNorm` MNNVL fused + NCCL) | 25% | 24.2% fused finalize all-reduce |
| GDN (flashinfer chunked delta net) | 6% | |
| elementwise / quantize | 6% | fused gate sigmoid 3.1%, nvfp4 quantize 2.2% |
| attention (fmha sm103a) | 1.3% | |

The c=128 profile from the same server is invalid: a second `start_profile` re-exported the first session's
events (identical kernel totals to 0.1%) plus 6.9 s of NCCL ring all-reduce, so its "53% collective" is not
a measurement of c=128. A dedicated run (`operations/modal/qwen38_profile.py`) with decode-only windows,
timestamp filtering, and fixed buckets replaces it.

Consequence for the kernel programme: even in this mixed window the MoE GEMM is 30% of GPU time and the
fused all-reduce is 25%. A perfect MoE kernel saves at most 30%; the collective share says topology
(EP or DP-attention instead of flat TP8) is the larger axis. Confirm with the decode-only profile.

## Superseded roof (2026-09-20, later)

The decode-only profile (`research/qwen38-economics.md`) shows TRT-LLM's two fp4 MoE GEMMs at 60 + 50 us
per layer at 128 slots. Under the uniform-routing model above that is 13 TB/s, above HBM, so the model
overstates the bytes: real routing is skewed and touches fewer experts per step. The kernel here is a
measured Triton floor at a byte count the shipped kernel never has to move. Closed.

## Open

- The last 30% (gate+up) and 50% (down) are unexplained without a profiler; ncu is not available on Modal.
  Candidates: DRAM page locality of 16 KB tiles scattered across experts, the smem staging of the A
  operand per MMA, grid tail at ~3 waves.
- EP8 (64 whole experts per GPU) would make the down shape `[m_e, 2048] x [2048, 8192]`, eight k-slabs
  instead of one, and the per-GPU token count 8x higher, both of which favour this kernel over TP8.
- Compare with `flashinfer.fused_moe.trtllm_fp4_block_scale_moe` on the same shape once the sweep profile
  gives its per-layer time.
