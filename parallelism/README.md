# Parallelism and disaggregation

## [dsa-decode-context-parallel/](dsa-decode-context-parallel/): a new SGLang mode

On stock SGLang v0.5.18, `--dcp-size 8` with GLM's DSA sparse attention asserts `page_size == 64`. With page 64 it still allocates the full pool on every rank and then dies. Decode context parallel just didn't shard DSA's KV. These 17 commits, against SGLang v0.5.18, implement it:

- **Owner rule.** Token slots are sharded across the DCP group using the existing write kernels. The indexer key cache is replicated over the virtual slot span so top-k selection still sees every token.
- **Decode.** Each rank runs FlashMLA FP8 decode on only the selected slots it owns and returns (o, lse) into the existing cross-rank merge. No per-layer compaction or metadata recompute. FlashMLA scheduler metadata is cached per row count.
- **Extend** joins the merge phase. Query heads are gathered and attention runs over this rank's KV shard: one eighth of the dequantization, and 2x head padding instead of 16x.
- **Speculative.** EAGLE target-verify rows go through the owned-slot path, and the draft extend becomes a merge phase too.
- **Pool sizing** counts the replicated indexer cache.
- [`dcp_topk.py`](dsa-decode-context-parallel/dcp_topk.py) plus its tests is the executable spec for the index math: owner rule, local and global positions, cross-rank top-k merge.
- [`sparse_prefill_fp8.py`](dsa-decode-context-parallel/sparse_prefill_fp8.py) is a Triton split-K sparse MLA prefill that reads FlashMLA FP8 KV rows, dequantizing in registers fused into `tl.dot`, for 16 heads or fewer. Correct (max error 5e-4), 6.0 ms per 4k tokens at K=2048. Not yet wired in.

Status: passes the 6/6 retrieval probe on GLM. One known bug before upstreaming: `pool_kwargs` is undefined in the out-of-tree pool builders (`kv_cache_configurator.py`), so `--dcp-size > 1` on an out-of-tree platform raises `NameError`. The CUDA path is unaffected.

## [in-node-pd-disaggregation/](in-node-pd-disaggregation/)

Prefill/decode split inside one node, for boxes without RDMA NICs:

- **pdmux lanes.** SM-partitioned prefill and decode lanes under DP attention.
- **Transport.** CUDA IPC on NVIDIA and MORI XGMI on AMD.
- **Decode-side radix cache** for hybrid models.
- [`pd_nvlink_copy.py`](in-node-pd-disaggregation/pd_nvlink_copy.py) is a one-kernel intra-node KV copy over CUDA IPC, replacing Mooncake's per-entry batch, which measured 1.3 GB/s p50.
- [`pd_retract_abort_hooks.py`](in-node-pd-disaggregation/pd_retract_abort_hooks.py) fixes a bug where a DeepSeek V4 decode retraction killed the whole node instead of one request.

PD with the radix-spec path was served on 8xB300. The pdmux lanes are CPU-harness verified only.

## [fp8-pcie-allreduce/](fp8-pcie-allreduce/)

An INT8/FP8 two-shot all-reduce over IPC or symmetric memory for PCIe boxes without NVLink, with bitwise-identical output on every rank. Simulated on one GPU (q8 relative error 0.87 to 0.90%, ranks identical 24/24). The 8-GPU timing run is still to do.
