# kvmix: FP8 + 4-bit KV in one GLM-5.3 engine (SGLang v0.5.20, 8xB200)

One node, one replica: requests stay FP8 up to NUMINOUS_KVMIX_THRESHOLD tokens (default 262,144) and use 4-bit
(NVFP4-style, glm_kv4's 352-byte row) KV beyond it, on the same GPUs. Everything is behind NUMINOUS_KVMIX=1
(operations/patches/kvmix_glm_hooks.py). Status and measurements: agents/status/kvmix.md.

## Pieces

| file | role |
| --- | --- |
| operations/kernels/kvmix_core.py | KVMixAllocator (two paged pools, one slot space), precision policy, promotion planner; model-agnostic |
| operations/kernels/kvmix_glm.py | Triton kernels: write_mixed, redirect_gather, seq_dequant + redirect_seq, promote_rows, move_rows, gather_bf16, index_move; torch references |
| operations/kernels/kvmix_glm_engine.py | KVMixDSATokenToKVPool (storage, doors, move, promotion executor), attention routing (engine_forward), graph reservation |
| operations/patches/kvmix_glm_hooks.py | 12 anchored hooks (configurator x3, common x1, allocation x3, dsa_backend x5), installs the 4 modules |
| operations/modal/kvmix_arms.py, kvmix_sessions.py | gate arms kvmix_ctl / kvmix_f50 / kvmix_f30; mixed-length session file |

## Slot space and storage

Unified slots: FP8 slots [0, B), B = N8 + 64 (page 0 is the stock dummy page); FP4 slots [B, N8 + N4 + 64), FP4 row =
slot - N8 (FP4 rows [0, 64) are never allocated). Two stock PagedTokenToKVPoolAllocators (sizes N8, N4) sit behind
KVMixAllocator; FP4 pages are offset by N8 / 64 when they leave the sub-allocator and de-offset when they come back.
Everything else in the scheduler (req_to_token, radix tree values, free_segment, free groups, PD transfer ids) keeps
seeing plain token slots.

FP8 storage is one tensor per rank: [78 x (N8 + 64) + S, 576] bytes; kv_buffer[l] are views, and the last S rows
(NUMINOUS_KVMIX_SCRATCH_ROWS, default 524,288 = 302 MB) are the FP8 scratch that every layer shares. FP4 storage is
[78 x (N4 + 64), 352]. The indexer cache is the stock one, sized for N8 + N4 tokens.

Pricing: calculate_mla_kv_cache_dim returns the blended row b = 1 / ((1 - f) / 576 + f / 352) rounded up (f =
NUMINOUS_KVMIX_FP4_FRAC, the FP4 share of the latent byte budget; f = 0.5 gives 437), so the configurator's token
count N already reflects the split; the pool then takes N4 as the smallest multiple of 64 with N8 x 576 + N4 x 352 <=
N x b (kvmix_core.split_tokens). Derived at f = 0.5 against the 1.9M-token FP8 control per rank: N = 2.5M tokens,
N8 = 0.95M, N4 = 1.55M. At f = 0.3: N8 = 1.33M, N4 = 0.93M.

## Read path (dsa_backend._forward_trtllm, after the top-k page table)

The TRT-LLM sparse MLA op takes one kv tensor and an int32 table of token slots per query. For layer l it is handed
the tail of the FP8 tensor starting at layer l's rows, so layer l's FP8 slots are valid indices as they are and the
scratch starts at row (78 - l) x (N8 + 64) of that view (box test: the op addresses a 4.5 GiB tensor at byte
offsets past 4 GiB correctly, max rel err 0.0109).

- Eager batch with no FP4 slot in metadata.page_table_1 (one scan + sync per eager batch): the stock call. Bit-identical
  to stock (box test 0.00000).
- Decode / verify / draft extend (captured in CUDA graphs) and any other mixed batch: redirect_gather rewrites the
  [T, K] table into a graph-static table: FP8 entries copy through, FP4 entries are dequantized into scratch row
  (flat entry index) and point there. Fixed shapes, no host sync; init_cuda_graph_state checks T_max x K_pad <=
  S (192 x 2048 = 393,216 at cuda-graph-max-bs-decode 48 with 4 draft tokens).
- Plain prefill with FP4-holding requests: seq_dequant dequantizes each such request's FP4 positions once into the
  scratch (slot_to_pos records the row), redirect_seq rewrites the table; chunked by contiguous request ranges when
  the sequences exceed S. FP8-only requests in the same batch cost one table rewrite.

Measured on B200 (box, random Gaussian rows, real _forward_trtllm): kernel vs fp32 reference max err / max |ref|
0.0099 (seq and gather paths, identical rows), promoted request 0.0080; decode graph replay == eager. Per-layer
timing of the FP4 read is kv4glm's scratch path (2.1-2.2x the FP8 op at decode/verify for FP4 tokens only,
glm_kv4.md); FP8 tokens in a mixed batch pay the table rewrite (~2 us per layer) and nothing else.

## Write path

write_mixed (one launch per layer): slot < B -> the 576 FP8 bytes the trtllm path produced; slot >= B -> FP4
quantization of the bf16 latent stashed before the FP8 quantize + RoPE kernel (rope from the FP8 output). The generic
set_mla_kv_buffer door (dense MHA one-shot prefill) converts bf16 in the same way; get_mla_kv_buffer gathers either
format to bf16; move_kv_cache handles the four format combinations (same-format moves copy bytes) and moves the
indexer tokens correctly (the stock IndexKeyCache.move indexes a page-major buffer by token slot).

## Policy and promotion (kvmix_core.before_alloc; node promotion needs the plain RadixCache, see below), hooked before alloc_for_extend / alloc_for_decode / alloc_for_spec_decode)

- A request is FP4 once max(prompt length, KV length) >= threshold: its new pages come from the FP4 pool. Requests
  known long at admission are FP4 from the first prefill chunk.
- The first time a request turns FP4, promote_request requantizes its existing FP8 pages in place: pages of radix
  nodes on its path with lock_ref == 1 (held by no other running request) and its own pages beyond the tree-owned
  prefix. New FP4 pages are taken, one kernel requantizes all 78 layers (grid rows x layers) and the indexer pages
  are copied, the FP8 pages are freed, node.value / req_to_token / req.prefix_indices are rewritten. Shared nodes
  (lock_ref > 1) and nodes shorter than NUMINOUS_KVMIX_MIN_PROMOTE_TOKENS (4096: system prompts) stay FP8; the read
  path accepts mixed rows per request, so nothing needs a uniform precision. Cached FP4 nodes are read as FP4 by
  later sharers (their memory is what the feature buys).
- Ordering: the GPU work waits on the model runner's forward stream (KVCacheConfigurator.forward_stream) because
  the overlap scheduler runs this while the previous step may still write the request's newest rows; the next
  forward waits on the schedule stream as in stock. Promotion cost: 256K tokens x 78 layers x (576 + 352) B ~ 19 GB
  of traffic, ~5 ms once per long request (derived).
- Capacity: available_size() is the total. evict_from_tree_cache (hooked) evicts the TOTAL shortfall in one call
  (stock semantics); a pool that is still short is served by the other pool (NUMINOUS_KVMIX_FALLBACK=1, counted in
  allocator.stats, logged), so capacity is the sum and the split is a preference. The first version looped on the
  per-pool shortfall and thrashed the cache at 64 sessions (measured, see Gate). FP8 -> FP4 fallback counts say the
  split is wrong for the traffic (tune NUMINOUS_KVMIX_FP4_FRAC).
- Served cache: the GLM cell runs UnifiedRadixCache, whose nodes are not the plain RadixCache's; promotion then
  covers only the request's own pages beyond cache_protected_len and cached prefixes keep their precision. Node
  promotion for UnifiedRadixCache (tree_core node ids + component values) is the follow-up that moves the memory.

## Tests

- CPU: operations/kernels/test_kvmix_core.py (13: allocator vs stock kernel mirror, mixed-group scatter, free routing,
  fallback, promotion against a real RadixCache), test_kvmix_glm.py (layout, references),
  operations/patches/test_kvmix_glm_hooks.py (anchors once, idempotent, compile, placement, exclusive with glm_kv4).
- Box (1xB200): test_kvmix_glm.py 8 kernels bit-exact + graph replay; test_kvmix_core.py on CUDA; hooks box tests:
  real pool + allocator, doors, move, promotion, production _forward_trtllm over FP8 / FP4 / mixed requests, stock
  path equality, promotion on the real pool, decode graph replay, > 4 GiB TRT-LLM addressing.
- Gate (2026-09-29, 8xB200, one node per arm, mixed session file; agents/status/kvmix.md has the full rows):
  ctl pool 1,552,832 tokens/rank; kvmix_f30 1,827,136 (1.177x), kvmix_f50 2,008,448 (1.293x). Rung 32: f30 38.3 and
  f50 37.8 tok/s/user vs ctl 38.5-43.8 (PASS, inside the control's spread); accuracy f30 vs ctl at the GLM noise
  floor (KL 0.177, top-1 0.899, GSM8K 0.96 vs 0.955). Rung 64: f30 5.5 tok/s/user vs ctl 28-30 (FAIL): with
  UnifiedRadixCache the long sessions' cached prefixes stayed FP8 (own-page promotion only), the FP8 pool filled at
  8 sessions/rank, and the per-pool eviction loop then evicted other sessions' caches every decode step (hit 0.976 ->
  0.931, decode 5x slower, warm TTFT unchanged). Fixed in the tree (stock single-call eviction + cross-pool fallback,
  no promotion retry storm, allocator stats to /cache/kvmix_stats) but not re-gated: the operator concluded the
  workstream. Rung 96, where the control collapses (1.2 tok/s/user) and the split should pay, was not reached.

## DeepSeek V4 Pro (source read 2026-09-29; recommendation, not built)

Source facts (v0.5.20): the c4 / c128 compressed pools mirror the full slot space (metadata_kernel.py:
c4_out_loc = raw_out_loc // 4, c128_out_loc = raw_out_loc // 128; page 256 -> 64 c4 rows and 2 c128 rows per
page), the SWA window has its own paged allocator and full_to_swa_index_mapping, and the c4 / c128 compress-state
pools and the indexer pools are sized alongside. kv4pro's 4-bit cells read through an FP8 FlashMLA workspace with
remapped indices, which is this design's redirect (kv4_to_fp8_workspace).

A mixed Pro pool would therefore be: KVMixAllocator over the full paged allocator (the c4 boundary follows as
B / 4, c128 stays FP8) and, if the SWA rows are to be 4-bit too, a second KVMixAllocator over the SWA allocator;
a c4 tensor per layer of [FP8 cells | 4-bit cells] with a workspace tail for FlashMLA's single k_cache; promotion
moving, per page, 256 SWA-mapped rows, 64 c4 rows (requant), 2 c128 rows (copy), the indexer rows and the compress
state entries. Per-token bytes (kv4pro's derived pricing, 12,914 B/full token): c4 KV is 34% and SWA KV 27%, so
the mixed pool buys 1.18x (c4) to 1.39x (c4 + SWA) tokens for the long requests only.

Recommendation: do not build it. kv4pro's all-4-bit pool (kv4pro_all, gated 2026-09-29) already passes Pro's
noise-calibrated accuracy gate (KL 0.0229 = 1.76x the control-vs-control noise, the same shift as the accepted
TRT-LLM attention and W4A4 arms) and holds 20 tok/s/user at 192 sessions where the FP8 lead collapses to 5.3. A
mixed pool would give up most of that capacity (1.18-1.39x vs 1.39-1.85x) to protect short requests from an
accuracy shift that Pro does not show. The mixed design is for GLM, whose FP4 rows carry 4x the FP8 quantization
error (L1 0.090 vs 0.0225, glm_kv4.md) and whose all-FP4 gate is still open.
