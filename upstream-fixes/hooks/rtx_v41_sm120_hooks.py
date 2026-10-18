"""Anchored hooks: DeepSeek-V4.1-Flash on SM120 (RTX PRO 6000) for SGLang main (agent rtxv41, 2026-09-30).

Target trees (anchors verified by operations/patches/test_rtx_v41_sm120_hooks.py):
    lmsysorg/sglang:nightly-dev-cu13-20260930-cae69be5  (main cae69be5, the recipe image; has the PCIe-IPC AR)
    origin/main 21ed9fbf (2026-09-30) and nightly 20260929-79cafec0 (same V4.1 files)

    python3 rtx_v41_sm120_hooks.py                  # patch the installed sglang package
    python3 rtx_v41_sm120_hooks.py --root <dir>     # patch a tree (<dir>/python/sglang or a package dir)
    python3 rtx_v41_sm120_hooks.py --check          # report which anchors resolve, change nothing

Contract: every hook names its file, an exact anchor that must occur exactly once, and its replacement; a missing
or ambiguous anchor fails the whole run before any file is written; a file already carrying the hook marker is
skipped (idempotent). The overlay operations/kernels/v41_sm120_overlay.py is copied next to the V4.1 indexer as
numinous_sm120.py.

What breaks on SM120 without them (source_code, reasoning in research/rtx-flash/v41.md section 3):

  v41_candidate_sm120      make_candidate_indexer takes the SparseTable path on any SM >= 10 (is_sm100_or_newer),
                           then raises "the candidate indexer needs DeepGEMM's paged sparse MQA logits" because
                           DEEPGEMM_PAGED_SPARSE_MQA_LOGITS is SM100-only (configurer: DEEPGEMM_BLACKWELL = is_sm100);
                           the kernel itself asserts arch_major == 10. V4.1 has candidate_topk_blocks 2048, so the
                           backend cannot even be constructed on SM120 -> boot failure. Fix: SM120 uses the dense
                           block-id backend (DenseBlocksBackend), which upstream already ships for non-SM100.
  v41_sm120_dg_metadata    V4.1's ratio-1/2 indexers call DeepGEMM fp8_fp4_paged_mqa_logits on SM120, but on SM120
                           model_hook sets SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 (for V4's c4 indexer), so
                           PagedIndexerMetadata leaves deep_gemm_metadata None unless --enable-deepseek-v4-fp4-indexer
                           is passed, and the prefill-graph metadata never forces it; the SM120 kernel also needs
                           DeepGEMM's own planner (split_kv 128) rather than sglang's JIT planner (256). Fix: force
                           DeepGEMM metadata for ratio 1/2 on SM120.
  v41_sm120_extra_page64   FlashInfer's SM120 DSV4 prefill kernel serves the compressed cache only at page_block_size
                           64 or 2 (V4 c4 / c128); V4.1's ratio-1/2 pools page at 256 / 128, so any batch of > 64
                           query tokens (every prefill chunk, every decode/verify batch above 64 rows) on 38 of 40
                           layers has no kernel ("SM120 sparse-MLA ... no kernel" / dispatch false). Fix: allocate
                           those pools with 64-token physical pages (same bytes, same flat slots).
  v41_sm120_paged_scores   (performance) DenseBlocksBackend's decode materialises [rows, L] int64 slots and scores all
                           L keys with a Triton kernel on layers 20, 24, 28, 32, 36; on SM120 the scores come from the
                           SM120 DeepGEMM paged kernel through the page table and slots are gathered for the k
                           selected positions only. Off switch: SGLANG_RTX_V41_PAGED_DECODE_SCORES=0.

  v41_sm120_sparse_consumer (performance) consumer layers 24..36 score only the 2048 x 8 candidates layer 20 published
                           (SM100 does this with a tcgen05 DeepGEMM kernel; SM120 scored all L keys): Triton kernel
                           operations/kernels/v41_sm120_candidates.py, bitwise the stock logits at those positions,
                           engaged when lmax > 2 x candidate columns. At 57K compressed keys: 3.9 -> 1.1 MB per row
                           per layer. Off switch: SGLANG_RTX_V41_SPARSE_CONSUMER=0.

  v41_sm120_page64_hicache / v41_sm120_page64_pd   keep HiCache host mirroring and PD transfer of the ratio-1/2 pools
                           correct under v41_sm120_extra_page64 (both address FULL pages; a FULL page is now a run of
                           64-token rows). Note: upstream itself refuses HiCache with SGLANG_OPT_SM120_DIRECT_SWA_KV
                           (_require_single_row_dsv4_swa_pages), so a HiCache run sets DIRECT_SWA_KV=0.
  rtx_quant_pcie_ar_build / rtx_quant_pcie_ar_call  (performance, opt-in) SGLANG_RTX_QUANT_ALLREDUCE=q8 routes the
                           decode-sized reductions the FlashInfer PCIe-IPC path would take through the int8 two-shot
                           all-reduce of operations/kernels/fp8_pcie_allreduce.py (torch symmetric memory; ranks'
                           outputs bitwise identical). Needs the PCIe-IPC communicator (FlashInfer >= 0.7.0) to exist;
                           absent in trees older than cae69be5 (hooks skipped there).

Off switches for the layout/perf hooks: SGLANG_OPT_SM120_DIRECT_EXTRA_KV=0, SGLANG_RTX_V41_PAGED_DECODE_SCORES=0,
SGLANG_RTX_V41_SPARSE_CONSUMER=0.
The two correctness hooks have no off switch: without them the model cannot run on SM120.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY_SRC = os.path.join(HERE, "..", "kernels", "v41_sm120_overlay.py")
OVERLAY_DST = "srt/layers/attention/dsv4/v41_indexer/numinous_sm120.py"
CAND_SRC = os.path.join(HERE, "..", "kernels", "v41_sm120_candidates.py")
CAND_DST = "srt/layers/attention/dsv4/v41_indexer/numinous_sm120_cand.py"
QAR_SRC = os.path.join(HERE, "..", "kernels", "fp8_pcie_allreduce.py")
QAR_DST = "srt/distributed/device_communicators/numinous_quant_ar.py"


@dataclass(frozen=True)
class Hook:
    name: str
    path: str  # relative to the sglang package directory
    old: str
    new: str
    # the file (or, for the all-reduce hooks, the PCIe-IPC code they anchor on) may be absent in older trees: the hook
    # is then skipped, not failed
    optional_file: bool = False

    @property
    def marker(self) -> str:
        return f"numinous-hook:{self.name}"


HOOKS = [
    Hook(
        "v41_candidate_sm120",
        "srt/layers/attention/dsv4/v41_indexer/__init__.py",
        """    # Without candidate blocks no layer is a candidate source or consumer.
    if not is_sm100_or_newer() or candidate_topk_blocks <= 0:
        return dense_blocks, dense_blocks
""",
        """    # Without candidate blocks no layer is a candidate source or consumer.
    # numinous-hook:v41_candidate_sm120 -- the SparseTable scheme needs DeepGEMM's paged sparse MQA
    # logits, which exist for SM100 only (arch_major == 10); SM120 publishes dense block ids.
    from .numinous_sm120 import candidate_sparse_table_supported

    if (
        not is_sm100_or_newer()
        or candidate_topk_blocks <= 0
        or not candidate_sparse_table_supported(torch.cuda.get_device_capability()[0])
    ):
        return dense_blocks, dense_blocks
""",
    ),
    Hook(
        "v41_sm120_dg_metadata",
        "srt/layers/attention/dsv4/metadata.py",
        """    def __post_init__(self):
        if (
            is_hip() or is_xpu() or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        ) and not self.force_deep_gemm_metadata:
""",
        """    def __post_init__(self):
        # numinous-hook:v41_sm120_dg_metadata -- V4.1 ratio-1/2 indexers always call DeepGEMM on SM120,
        # whose paged fp4 kernel needs DeepGEMM's own schedule (split_kv 128).
        if _IS_SM120 and self.compress_ratio in (1, 2):
            self.force_deep_gemm_metadata = True
        if (
            is_hip() or is_xpu() or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        ) and not self.force_deep_gemm_metadata:
""",
    ),
    Hook(
        "v41_sm120_extra_page64",
        "srt/mem_cache/deepseek_v4_memory_pool.py",
        """            for ratio, sources in low_ratio_sources.items():
                self.kv_pools[ratio] = self._make_kv_pool(
                    size=self.full_size // ratio,
                    page_size=page_size // ratio,
""",
        """            for ratio, sources in low_ratio_sources.items():
                # numinous-hook:v41_sm120_extra_page64 -- FlashInfer's SM120 DSV4 prefill kernel reads the
                # compressed cache at page_block_size 64 (or 2) only; 64-token physical pages keep the same
                # bytes and flat slot indices (the allocator's logical page stays page_size // ratio).
                from sglang.srt.layers.attention.dsv4.v41_indexer.numinous_sm120 import (
                    low_ratio_physical_page as _numinous_low_ratio_page,
                )

                _low_ratio_page = _numinous_low_ratio_page(
                    page_size // ratio,
                    is_v4_layout=self.compressed_kv_layout(ratio) is KVLayout.V4,
                    is_sm120=get_platform().is_sm120,
                )
                if _low_ratio_page != page_size // ratio:
                    logger.info(
                        "DeepSeek-V4.1 SM120 direct extra KV: ratio=%d logical_page=%d physical_page=%d",
                        ratio,
                        page_size // ratio,
                        _low_ratio_page,
                    )
                self.kv_pools[ratio] = self._make_kv_pool(
                    size=self.full_size // ratio,
                    page_size=_low_ratio_page,
""",
    ),
    Hook(
        "v41_sm120_paged_scores",
        "srt/layers/attention/dsv4/v41_indexer/scoring.py",
        """    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    j = torch.arange(lmax, device=pos.device)
""",
        """    # numinous-hook:v41_sm120_paged_scores -- SM120: dense scores from the paged DeepGEMM kernel,
    # slots gathered only for the selected positions (no [rows, L] slot matrix).
    from . import numinous_sm120 as _nsm120

    if _nsm120.use_paged_decode_scores(metadata):
        from sglang.kernels.ops.attention.dsv4.index_logits import (
            deep_gemm_fp4_paged_mqa_logits,
        )

        _data = get_deep_gemm_decode_data(inputs, pool)
        _scores = _nsm120.paged_decode_scores(
            q_fp4=_data.q_fp4,
            q_sf=_data.q_sf,
            weights=_data.weights,
            k_cache=_data.k_cache,
            paged_metadata=metadata,
            lens=lens,
            lmax=lmax,
            logits_fn=deep_gemm_fp4_paged_mqa_logits,
        )
        return DecodeScores(
            bs=bs,
            lmax=lmax,
            lens=lens,
            slots=_nsm120.LazySlots(req_to_token, req, ratio, lmax),
            scores=_scores,
        )
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    j = torch.arange(lmax, device=pos.device)
""",
    ),
    Hook(
        "v41_sm120_sparse_consumer",
        "srt/layers/attention/dsv4/v41_indexer/dense_blocks.py",
        """    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[BlockIds],
        out: Selection,
    ) -> None:
""",
        """    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[BlockIds],
        out: Selection,
    ) -> None:
        # numinous-hook:v41_sm120_sparse_consumer -- SM120: score only the published candidate blocks
        # (bitwise the stock Triton logits at those positions), not every compressed key.
        from . import numinous_sm120 as _nsm120
        from . import numinous_sm120_cand as _ncand

        if published is not None and _ncand.use_sparse_consumer(
            inputs, published.blocks, self.block_size, _nsm120._is_sm120_device()
        ):
            from .scoring import DecodeScores as _DecodeScores

            _ncand.sparse_consume_decode(
                inputs=inputs,
                blocks=published.blocks,
                out=out,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                block_size=self.block_size,
                decode_scores_cls=_DecodeScores,
                write_decode=write_decode,
                lazy_slots_cls=_nsm120.LazySlots,
            )
            return
""",
    ),
    Hook(
        "v41_sm120_page64_hicache",
        "srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py",
        """        regions = [(names[0], kv_pool, kv_pool.kv_buffer)]
""",
        """        # numinous-hook:v41_sm120_page64_hicache -- with v41_sm120_extra_page64 a FULL page's ratio-1/2 latents
        # span several 64-token physical rows; HiCache moves whole FULL pages, so present each run of rows as one
        # row (contiguous view, same bytes), exactly as the index regions below already do.
        _kv_rows = slots_per_page // kv_pool.page_size
        if _kv_rows == 1:
            regions = [(names[0], kv_pool, kv_pool.kv_buffer)]
        else:
            regions = [
                (
                    names[0],
                    kv_pool,
                    [
                        b[: (b.shape[0] // _kv_rows) * _kv_rows].reshape(b.shape[0] // _kv_rows, -1)
                        for b in kv_pool.kv_buffer
                    ],
                )
            ]
""",
    ),
    Hook(
        "v41_sm120_page64_pd",
        "srt/mem_cache/deepseek_v4_memory_pool.py",
        """            elif kv_pool is not None:
                for buf in kv_pool.kv_buffer:
                    append_page_buffer(buf)
""",
        """            elif kv_pool is not None:
                # numinous-hook:v41_sm120_page64_pd -- PD addresses buffers by FULL page id; with 64-token physical
                # pages one FULL page of a ratio-1/2 pool is a run of rows (item = run length x row bytes).
                _kv_rows = (
                    max(1, (self.page_size // ratio) // kv_pool.page_size) if ratio in (1, 2) else 1
                )
                for buf in kv_pool.kv_buffer:
                    append_page_buffer(buf)
                    item_lens[-1] *= _kv_rows
""",
    ),
    Hook(
        "rtx_quant_pcie_ar_build",
        "srt/distributed/device_communicators/pcie_ipc_ar.py",
        """        self.max_numel = max_numel
        logger.info(
            "FlashInfer PCIe-IPC workspace built (world=%d, max_numel=%d, hidden=%d)",
""",
        """        self.max_numel = max_numel
        # numinous-hook:rtx_quant_pcie_ar_build -- SGLANG_RTX_QUANT_ALLREDUCE=q8|q8_rs|fp8|fp8_rs: an int8/fp8
        # two-shot all-reduce on symmetric memory for the same decode-sized reductions (1.03 B/elem instead of 2).
        self._numinous_qar = None
        from sglang.srt.distributed.device_communicators import numinous_quant_ar as _nqar

        _qmode = _nqar.enabled_mode()
        if _qmode is not None:
            _qcap = max_numel - max_numel % (self._world_size * _nqar.GROUP)
            try:
                self._numinous_qar = _nqar.Fp8PcieAllReduce(
                    self._group, self._device, _qcap, mode=_qmode
                )
                logger.info(
                    "numinous quantized PCIe all-reduce enabled (mode=%s, max_numel=%d)",
                    _qmode,
                    _qcap,
                )
            except Exception as e:
                logger.warning("numinous quantized PCIe all-reduce disabled: %s", e)
        logger.info(
            "FlashInfer PCIe-IPC workspace built (world=%d, max_numel=%d, hidden=%d)",
""",
        optional_file=True,
    ),
    Hook(
        "rtx_quant_pcie_ar_call",
        "srt/distributed/device_communicators/pcie_ipc_ar.py",
        """        stream = torch.cuda.current_stream()
        if self._bound_stream is None or stream != self._bound_stream:
            self._workspace.rebind_stream()
""",
        """        # numinous-hook:rtx_quant_pcie_ar_call
        _qar = getattr(self, "_numinous_qar", None)
        if _qar is not None and _qar.supports(inp):
            return _qar.all_reduce(inp)
        stream = torch.cuda.current_stream()
        if self._bound_stream is None or stream != self._bound_stream:
            self._workspace.rebind_stream()
""",
        optional_file=True,
    ),
    Hook(
        "rtx_quant_ar_standalone",
        "srt/distributed/parallel_state.py",
        """                logger.warning(f"Setup FlashInfer PCIe-IPC allreduce failed with {e}.")
        if use_custom_allreduce and self.world_size > 1:
""",
        """                logger.warning(f"Setup FlashInfer PCIe-IPC allreduce failed with {e}.")
        # numinous-hook:rtx_quant_ar_standalone -- FlashInfer without pcie_ipc_comm (0.6.18): the quantized all-reduce
        # takes the pcie_ipc slot itself when SGLANG_RTX_QUANT_ALLREDUCE is set (same eligibility: TP group, opt-in env).
        if self.pcie_ipc_comm is None or getattr(self.pcie_ipc_comm, "disabled", True):
            from sglang.srt.distributed.device_communicators import numinous_quant_ar as _nqar

            _qmode = _nqar.enabled_mode()
            if _qmode is not None and eligible_group(
                group_name=group_name,
                world_size=self.world_size,
                deterministic=self._deterministic_collectives_enabled(),
            ):
                self.pcie_ipc_comm = _nqar.StandaloneComm(self.device_group, self.device, _qmode)
        if use_custom_allreduce and self.world_size > 1:
""",
        optional_file=True,
    ),
    # PD over mooncake's intra-node CUDA-IPC transport (rtxsys g157a/b, ported 2026-10-02): new Python threads start on
    # device 0 and the IPC transport creates its events on the calling thread's device, so every prefill rank needed a
    # CUDA context on GPU 0 (cudaSetDevice(0) / cudaEventRecord OOM under load, half of all requests dropped in g157a).
    # Pin the sender pool threads and the queue worker to the rank's GPU. Inert without --disaggregation-mode.
    Hook(
        "mooncake_threads_own_gpu",
        "srt/disaggregation/mooncake/conn.py",
        """            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]
""",
        """            # numinous-hook:mooncake_threads_own_gpu (sender threads on the rank GPU, not device 0)
            import torch as _nf_torch

            _nf_dev = _nf_torch.cuda.current_device() if _nf_torch.cuda.is_available() else None
            self._nf_dev = _nf_dev

            def _nf_pin(_d=_nf_dev):
                if _d is not None:
                    _nf_torch.cuda.set_device(_d)

            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size, initializer=_nf_pin
                )
                for _ in range(transfer_queue_size)
            ]
""",
        optional_file=True,
    ),
    Hook(
        "mooncake_worker_own_gpu",
        "srt/disaggregation/mooncake/conn.py",
        """        staging_buffer=None,
        worker_index=0,
    ):
""",
        """        staging_buffer=None,
        worker_index=0,
    ):
        # numinous-hook:mooncake_worker_own_gpu (the queue worker thread also issues transfers)
        if getattr(self, "_nf_dev", None) is not None:
            import torch as _nf_torch

            _nf_torch.cuda.set_device(self._nf_dev)
""",
        optional_file=True,
    ),
    Hook(
        "v41_tp_shard_prefill_indexer",
        "srt/layers/attention/deepseek_v4_backend.py",
        """        else:
            inputs = self._make_low_ratio_prefill_indexer_inputs(
                layer, x, q_lora, req, pos, forward_batch, rows_per_request
            )
            if is_source:
""",
        """        else:
            # numinous-hook:v41_tp_shard_prefill_indexer -- each TP rank scores an interleaved 1/W of every request's
            # rows and the selections are all-gathered (rows are independent); off: SGLANG_RTX_V41_TP_SHARD_INDEXER=0.
            from sglang.srt.layers.attention.dsv4.v41_indexer import numinous_sm120 as _nsm120
            from sglang.srt.runtime_context import get_parallel as _get_parallel

            _plan = _nsm120.use_tp_shard(
                forward_batch,
                x.shape[0],
                rows_per_request,
                self.forward_metadata.late_layer_tail,
                _get_parallel().attn_tp_size,
            )
            if _plan is not None:
                _nsm120.sharded_prefill_topk(
                    self, layer, x, q_lora, req, pos, forward_batch, out, _plan, is_source, is_consumer
                )
                return
            inputs = self._make_low_ratio_prefill_indexer_inputs(
                layer, x, q_lora, req, pos, forward_batch, rows_per_request
            )
            if is_source:
""",
    ),
]


def resolve_pkg(root: str | None) -> str:
    if root is None:
        import importlib.util

        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("sglang is not importable; pass --root")
        return list(spec.submodule_search_locations)[0]
    for cand in (os.path.join(root, "python", "sglang"), root):
        if os.path.isdir(os.path.join(cand, "srt")):
            return cand
    raise SystemExit(f"no sglang package under {root}")


def plan(pkg: str):
    """[(hook, state)] with state in {'apply', 'done', 'missing', 'ambiguous', 'nofile'}."""
    out = []
    for h in HOOKS:
        p = os.path.join(pkg, h.path)
        if not os.path.exists(p):
            out.append((h, "absent" if h.optional_file else "nofile"))
            continue
        src = open(p).read()
        if h.marker in src:
            out.append((h, "done"))
            continue
        n = src.count(h.old)
        if n == 0 and h.optional_file:
            out.append((h, "absent"))
            continue
        out.append((h, "apply" if n == 1 else ("missing" if n == 0 else "ambiguous")))
    return out


def apply(pkg: str, check: bool = False) -> int:
    states = plan(pkg)
    bad = [(h.name, s) for h, s in states if s in ("missing", "ambiguous", "nofile")]
    for h, s in states:
        print(f"{s:9s} {h.name:26s} {h.path}")
    if bad:
        print(f"FAIL: unresolved anchors {bad}", file=sys.stderr)
        return 1
    if check:
        return 0
    dst = os.path.join(pkg, OVERLAY_DST)
    shutil.copyfile(OVERLAY_SRC, dst)
    print(f"overlay   {OVERLAY_DST}")
    shutil.copyfile(CAND_SRC, os.path.join(pkg, CAND_DST))
    print(f"overlay   {CAND_DST}")
    if os.path.isdir(os.path.dirname(os.path.join(pkg, QAR_DST))):
        shutil.copyfile(QAR_SRC, os.path.join(pkg, QAR_DST))
        print(f"overlay   {QAR_DST}")
    for h, s in states:
        if s != "apply":
            continue
        p = os.path.join(pkg, h.path)
        src = open(p).read()
        open(p, "w").write(src.replace(h.old, h.new, 1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    return apply(resolve_pkg(a.root), a.check)


if __name__ == "__main__":
    sys.exit(main())
