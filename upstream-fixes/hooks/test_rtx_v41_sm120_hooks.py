"""CPU tests for rtx_v41_sm120_hooks.py and its overlay (agent rtxv41).

    python3 -m pytest -q operations/patches/test_rtx_v41_sm120_hooks.py

* anchors resolve exactly once on every pinned main tree present (cae69be5 recipe image, 21ed9fbf, 79cafec0),
  the patched files compile, a second run is a no-op, and --check changes nothing;
* the patched decode_scores/metadata/pool/indexer text reads the overlay the way the tests below exercise it;
* overlay logic: page64 gate, metadata force gate, SparseTable gate, and the paged-scores glue against a torch
  reference of DeepGEMM's paged fp4 MQA logits (masking past lens, row-chunked schedule, lazy slots == the
  materialised req_to_token slot matrix, and write_decode's gather through LazySlots).
"""
from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "kernels"))
import rtx_v41_sm120_hooks as hooks  # noqa: E402
import v41_sm120_overlay as ov  # noqa: E402

TREES = [t for t in ("/tmp/sgl-cae69be5", "/tmp/sgl-main-latest", "/tmp/sgl-main-wt") if os.path.isdir(t)]


def _copy_tree(tree: str) -> str:
    tmp = tempfile.mkdtemp()
    pkg = os.path.join(tmp, "sglang")
    for h in hooks.HOOKS:
        src = os.path.join(tree, "python/sglang", h.path)
        dst = os.path.join(pkg, h.path)
        if not os.path.exists(src) and h.optional_file:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    return pkg


def _present(pkg, h):
    return os.path.exists(os.path.join(pkg, h.path))


@pytest.mark.parametrize("tree", TREES)
def test_apply_compile_idempotent(tree):
    pkg = _copy_tree(tree)
    live = [h for h, st in hooks.plan(pkg) if st == "apply"]
    before = {h.path: open(os.path.join(pkg, h.path)).read() for h in live}
    assert hooks.apply(pkg, check=True) == 0
    assert all(open(os.path.join(pkg, h.path)).read() == before[h.path] for h in live)
    assert hooks.apply(pkg) == 0
    for h in live:
        p = os.path.join(pkg, h.path)
        src = open(p).read()
        assert src.count(h.marker) == 1, h.name
        py_compile.compile(p, doraise=True)
    py_compile.compile(os.path.join(pkg, hooks.OVERLAY_DST), doraise=True)
    py_compile.compile(os.path.join(pkg, hooks.CAND_DST), doraise=True)
    if any(h.optional_file for h in live):
        py_compile.compile(os.path.join(pkg, hooks.QAR_DST), doraise=True)
        assert "numinous-hook:rtx_quant_ar_standalone" in open(
            os.path.join(pkg, "srt/distributed/parallel_state.py")).read()
    after = {h.path: open(os.path.join(pkg, h.path)).read() for h in live}
    assert hooks.apply(pkg) == 0  # idempotent
    assert all(open(os.path.join(pkg, h.path)).read() == after[h.path] for h in live)
    assert all(s in ("done", "absent") for _, s in hooks.plan(pkg))


def test_quant_ar_call_routing():
    """The patched pcie_ipc_all_reduce hands supported tensors to the quantized all-reduce, others to FlashInfer."""
    import ast
    import types

    tree = next((t for t in TREES if os.path.exists(os.path.join(t, "python/sglang", hooks.QAR_DST.replace(
        "numinous_quant_ar.py", "pcie_ipc_ar.py")))), None)
    if tree is None:
        pytest.skip("no tree with pcie_ipc_ar.py")
    pkg = _copy_tree(tree)
    assert hooks.apply(pkg) == 0
    src = open(os.path.join(pkg, "srt/distributed/device_communicators/pcie_ipc_ar.py")).read()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "PcieIpcCommunicator")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "pcie_ipc_all_reduce")
    ns = {"torch": types.SimpleNamespace(cuda=types.SimpleNamespace(current_stream=lambda: "s")),
          "Optional": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "pcie", "exec"), ns)
    f = ns["pcie_ipc_all_reduce"]
    qar = types.SimpleNamespace(supports=lambda t: t == "small", all_reduce=lambda t: "quant:" + t)
    ws = types.SimpleNamespace(rebind_stream=lambda: None, all_reduce=lambda t: "fi:" + t)
    me = types.SimpleNamespace(_numinous_qar=qar, _bound_stream="s", _workspace=ws)
    assert f(me, "small") == "quant:small"
    assert f(me, "big") == "fi:big"
    me._numinous_qar = None
    assert f(me, "small") == "fi:small"
    assert "self._numinous_qar = _nqar.Fp8PcieAllReduce(" in src and "_qcap = max_numel - max_numel %" in src


def test_missing_anchor_fails_without_writing(tmp_path):
    pkg = _copy_tree(TREES[0]) if TREES else pytest.skip("no tree")
    p = os.path.join(pkg, hooks.HOOKS[2].path)
    open(p, "w").write(open(p).read().replace("page_size=page_size // ratio,", "page_size=X,"))
    other = os.path.join(pkg, hooks.HOOKS[0].path)
    before = open(other).read()
    assert hooks.apply(pkg) == 1
    assert open(other).read() == before


def test_cli_check_runs():
    if not TREES:
        pytest.skip("no tree")
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "rtx_v41_sm120_hooks.py"), "--root", TREES[0], "--check"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("apply") + r.stdout.count("done") == len(hooks.HOOKS)


def test_hook_texts_reference_real_names():
    """Names the hook bodies use must exist in the unpatched files (catches upstream renames)."""
    if not TREES:
        pytest.skip("no tree")
    pkg = os.path.join(TREES[0], "python/sglang")
    meta = open(os.path.join(pkg, "srt/layers/attention/dsv4/metadata.py")).read()
    assert "_IS_SM120 = is_sm120_supported()" in meta and "compress_ratio: int = 4" in meta
    pool = open(os.path.join(pkg, "srt/mem_cache/deepseek_v4_memory_pool.py")).read()
    assert "    KVLayout,\n" in pool and "get_platform" in pool and "def compressed_kv_layout" in pool
    sc = open(os.path.join(pkg, "srt/layers/attention/dsv4/v41_indexer/scoring.py")).read()
    body = sc[sc.index("def decode_scores(") : sc.index("def write_decode(")]
    for name in ("metadata = inputs.paged_metadata", "pool = token_to_kv_pool", "lens = (pos + 1) // ratio", "lmax = min("):
        assert name in body, name
    assert "def get_deep_gemm_decode_data(" in sc and "d.slots.gather(1," in sc
    idx = open(os.path.join(pkg, "kernels/ops/attention/dsv4/index_logits.py")).read()
    assert "def deep_gemm_fp4_paged_mqa_logits(" in idx
    db = open(os.path.join(pkg, "srt/layers/attention/dsv4/v41_indexer/dense_blocks.py")).read()
    be = open(os.path.join(pkg, "srt/layers/attention/deepseek_v4_backend.py")).read()
    for name in ("def _make_low_ratio_prefill_indexer_inputs(", "self.prefill_candidates.publish_prefill(",
                 "self.full_topk_indexer.topk_prefill(inputs, out)", "def _publish_candidate_metadata("):
        assert name in be, name
    for name in ("    write_decode,\n", "self.block_size = candidate_block_size", "self.token_to_kv_pool = token_to_kv_pool",
                 "self.req_to_token = req_to_token", "published.blocks.shape[0] == d.bs"):
        assert name in db, name
    assert "class DecodeScores(msgspec.Struct" in sc
    asm = open(os.path.join(pkg, "srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py")).read()
    blk = asm[asm.index("slots_per_page = page_size // ratio"): asm.index("regions = [(names[0], kv_pool, kv_pool.kv_buffer)]")]
    assert "kv_pool = kvcache.kv_pools[ratio]" in asm[asm.index("kv_pool = kvcache.kv_pools[ratio]"):] and blk
    gcb = pool[pool.index("def get_contiguous_buf_infos"): pool.index("def get_unified_swa_ring_buf_infos")]
    assert "for ratio, kv_pool in self.kv_pools.items():" in gcb and "rows_per_page = self.page_size // ratio" in gcb
    init = open(os.path.join(pkg, "srt/layers/attention/dsv4/v41_indexer/__init__.py")).read()
    assert "import torch" in init


# ------------------------------------------------------------------------------------------------- overlay logic


def test_page64_gate(monkeypatch):
    monkeypatch.delenv("SGLANG_OPT_SM120_DIRECT_EXTRA_KV", raising=False)
    f = ov.low_ratio_physical_page
    assert f(256, is_v4_layout=True, is_sm120=True) == 64
    assert f(128, is_v4_layout=True, is_sm120=True) == 64
    assert f(64, is_v4_layout=True, is_sm120=True) == 64
    assert f(256, is_v4_layout=True, is_sm120=False) == 256  # SM100: untouched
    assert f(256, is_v4_layout=False, is_sm120=True) == 256  # V41 layouts: no SM120 reader anyway
    monkeypatch.setenv("SGLANG_OPT_SM120_DIRECT_EXTRA_KV", "0")
    assert f(256, is_v4_layout=True, is_sm120=True) == 256


def test_metadata_and_sparse_table_gates():
    assert ov.force_sm120_deep_gemm_metadata(1, True) and ov.force_sm120_deep_gemm_metadata(2, True)
    assert not ov.force_sm120_deep_gemm_metadata(4, True)  # V4 c4 keeps its own flag
    assert not ov.force_sm120_deep_gemm_metadata(1, False)
    assert ov.candidate_sparse_table_supported(10)
    assert not ov.candidate_sparse_table_supported(12)


class _MD:
    def __init__(self, seq_lens, table, width, chunk=None):
        self.compressed_seq_lens = seq_lens
        self.page_table = table
        self.max_compressed_seq_len = width
        self._chunk = chunk
        self.deep_gemm_metadata = ["plan"] * 2 if chunk else "plan"

    def row_chunks(self):
        n = self.compressed_seq_lens.shape[0]
        for s in range(0, n, self._chunk):
            yield slice(s, min(s + self._chunk, n)), "plan"


def _paged_case(seed=0):
    g = torch.Generator().manual_seed(seed)
    lens_req = [150, 70, 201]
    keys = [torch.randn(n, 128, generator=g) for n in lens_req]
    k_cache, table, slots = ov.build_paged_index_cache(keys, 64, seed)
    heads = 32
    # verify-like rows: 2 rows per request at consecutive positions
    rows_req = torch.tensor([0, 0, 1, 1, 2, 2])
    lens = torch.tensor([149, 150, 69, 70, 200, 201])
    q = torch.randn(6, heads, 128, generator=g)
    qp, qs = ov.pack_fp4_rows(q.reshape(-1, 128))
    qd = ov.unpack_fp4_rows(qp, qs).reshape(6, heads, 128)
    w = torch.rand(6, heads, generator=g)
    return keys, k_cache, table, slots, rows_req, lens, qd, w


def _fake_logits(qd):
    def fn(q_pair, k_cache, weights, seq_lens, page_table, plan, width):
        rows = q_pair[0]
        return ov.paged_fp4_logits_ref(qd[rows], weights, k_cache, seq_lens, page_table, width)

    return fn


@pytest.mark.parametrize("chunk", [None, 4])
def test_paged_scores_glue(chunk):
    keys, k_cache, table, slots, rows_req, lens, qd, w = _paged_case()
    width = 256
    md = _MD(lens.clone(), table[rows_req], width, chunk)
    row_ids = torch.arange(6)
    lmax = 201
    scores = ov.paged_decode_scores(
        q_fp4=row_ids,
        q_sf=row_ids,
        weights=w,
        k_cache=k_cache,
        paged_metadata=md,
        lens=lens,
        lmax=lmax,
        logits_fn=_fake_logits(qd),
    )
    assert scores.shape == (6, lmax)
    for r in range(6):
        kd = ov.unpack_fp4_rows(*ov.pack_fp4_rows(keys[int(rows_req[r])]))[: int(lens[r])]
        want = (torch.relu(qd[r] @ kd.T) * w[r][:, None]).sum(0)
        assert torch.allclose(scores[r, : int(lens[r])], want, rtol=1e-5, atol=1e-4)
        assert torch.isinf(scores[r, int(lens[r]) :]).all()
    # the dense-blocks consumers' selections see no NaN/garbage past the length
    assert not torch.isnan(scores).any()


def test_lazy_slots_equal_materialised():
    g = torch.Generator().manual_seed(3)
    ratio, lmax = 2, 90
    req_to_token = torch.randint(0, 10_000, (4, 400), generator=g)
    req = torch.tensor([2, 2, 0, 3])
    lazy = ov.LazySlots(req_to_token, req, ratio, lmax)
    j = torch.arange(lmax)
    dense = req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
    assert torch.equal(lazy.materialize(), dense)
    idx = torch.randint(0, lmax + 5, (4, 17), generator=g)  # includes the d.lmax fill of consume_decode
    assert torch.equal(ov.slots_at(lazy, idx.clamp_max(lmax - 1)), dense.gather(1, idx.clamp_max(lmax - 1)))


def test_use_paged_decode_scores_gate(monkeypatch):
    md = _MD(torch.tensor([1]), torch.zeros(1, 1, dtype=torch.int32), 64)
    monkeypatch.delenv("SGLANG_RTX_V41_PAGED_DECODE_SCORES", raising=False)
    assert ov.use_paged_decode_scores(md, is_sm120=True)
    assert not ov.use_paged_decode_scores(md, is_sm120=False)
    md.deep_gemm_metadata = None
    assert not ov.use_paged_decode_scores(md, is_sm120=True)
    md.deep_gemm_metadata = "plan"
    monkeypatch.setenv("SGLANG_RTX_V41_PAGED_DECODE_SCORES", "0")
    assert not ov.use_paged_decode_scores(md, is_sm120=True)


# --------------------------------------------------------------------------- TP row sharding of the prefill indexer


def test_tp_row_shard_plan_partition():
    plan = ov.tp_row_shard_plan([1500, 9, 700], world=8, min_rows=64)
    assert plan is not None and plan["total"] == 2209
    allidx = torch.cat(plan["rank_idx"]).sort().values
    assert torch.equal(allidx, torch.arange(2209))  # a partition
    for r in range(8):
        assert sum(plan["rank_rows_per_request"][r]) == plan["rank_idx"][r].numel()
        # rows of one request stay consecutive and ordered inside a rank's slice
        assert torch.all(plan["rank_idx"][r][1:] > plan["rank_idx"][r][:-1])
    assert ov.tp_row_shard_plan([1500, 5], world=8, min_rows=64) is None  # a request with fewer rows than ranks
    assert ov.tp_row_shard_plan([100], world=8) is None  # below min_rows
    assert ov.tp_row_shard_plan([5000], world=1) is None


def test_gather_selection_reassembles_unsharded_result():
    import types as _t

    world, rpr, k = 4, [1031, 777, 260], 16
    plan = ov.tp_row_shard_plan(rpr, world, min_rows=64)
    total = plan["total"]
    g = torch.Generator().manual_seed(0)
    full_page = torch.randint(0, 10**6, (total, k), generator=g, dtype=torch.int32)  # "unsharded indexer output"
    full_raw = torch.randint(0, 10**5, (total, k), generator=g, dtype=torch.int32)
    locals_ = [(full_page[plan["rank_idx"][r]], full_raw[plan["rank_idx"][r]]) for r in range(world)]

    def padded(t):
        src = torch.full((plan["n_max"], t.shape[1]), -1, dtype=t.dtype)
        src[: t.shape[0]] = t
        return src

    for r in range(world):
        out = _t.SimpleNamespace(page_indices=torch.zeros(total + 7, k, dtype=torch.int32),
                                 raw_indices=torch.zeros(total + 7, k, dtype=torch.int32))
        which = {"n": 0}

        def ag(dst, src, which=which):
            part = 0 if which["n"] == 0 else 1
            which["n"] += 1
            for q in range(world):
                dst[q] = padded(locals_[q][part])

        ov.gather_selection(locals_[r][0], locals_[r][1], out, plan, r, ag)
        assert torch.equal(out.page_indices[:total], full_page)
        assert torch.equal(out.raw_indices[:total], full_raw)


def test_use_tp_shard_gate(monkeypatch):
    import types as _t

    fb = _t.SimpleNamespace(extend_seq_lens_cpu=[4096, 4096])
    monkeypatch.delenv("SGLANG_RTX_V41_TP_SHARD_INDEXER", raising=False)
    assert ov.use_tp_shard(fb, 8192, None, None, 8) is None  # default off (measured: indexer is cheap)
    monkeypatch.setenv("SGLANG_RTX_V41_TP_SHARD_INDEXER", "1")
    assert ov.use_tp_shard(fb, 8192, None, None, 8) is not None
    assert ov.use_tp_shard(fb, 8192, [1, 2], None, 8) is None  # CP passes rank-local rows already
    assert ov.use_tp_shard(fb, 8192, None, object(), 8) is None  # SWA-replay tail
    assert ov.use_tp_shard(fb, 8000, None, None, 8) is None  # row count mismatch -> unsharded
    assert ov.use_tp_shard(fb, 8192, None, None, 1) is None
    monkeypatch.setenv("SGLANG_RTX_V41_TP_SHARD_INDEXER", "0")
    assert ov.use_tp_shard(fb, 8192, None, None, 8) is None
    monkeypatch.setenv("SGLANG_RTX_V41_TP_SHARD_INDEXER", "1")
