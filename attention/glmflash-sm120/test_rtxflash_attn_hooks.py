"""rtxflash_attn_hooks: anchors on SGLang v0.5.20 (tag in the local fork), idempotence, compile, hook placement, a full
apply() onto a copy of the v0.5.20 tree, and the engine's server-arg logic on CPU.

    TRITON_INTERPRET=1 python3 -m pytest -q operations/patches/test_rtxflash_attn_hooks.py
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import types

os.environ.setdefault("TRITON_INTERPRET", "1")
import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kernels"))

import rtxflash_attn_hooks as hooks  # noqa: E402

FORK = pathlib.Path("/home/ubuntu/sglang-fork")


def _v0520(path: str) -> str:
    if not (FORK / ".git").exists():
        pytest.skip("local SGLang fork not present")
    r = subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{path}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        pytest.skip("tag v0.5.20 not available")
    return r.stdout


@pytest.mark.parametrize("path", hooks.FILES)
def test_hooks_apply_once_and_compile(path):
    src = _v0520(path)
    new, lines = hooks.patch_source(src, path)
    assert lines == [f"{h.name}: applied to {path}" for h in hooks.HOOKS if h.path == path]
    compile(new, path, "exec")
    again, lines2 = hooks.patch_source(new, path)
    assert again == new and all(ln.endswith("already applied") for ln in lines2)


def _func(src: str, name: str) -> str:
    i = src.index(f"def {name}(")
    j = src.find("\ndef ", i + 10)
    k = src.find("\n    def ", i + 10)
    ends = [x for x in (j, k) if x > 0]
    return src[i: min(ends) if ends else len(src)]


def test_hook_placement():
    fm = hooks.patch_source(_v0520(hooks.FLASH_MLA_SM120), hooks.FLASH_MLA_SM120)[0]
    v = _func(fm, "_validate_flashinfer_sparse_mla_backend")
    assert v.index("rtxattn_validate") < v.index("is_glm_sm12_fp8 = (")  # before the GLM-5.3-only refusal
    cf = hooks.patch_source(_v0520(hooks.CONFIGURATOR), hooks.CONFIGURATOR)[0]
    d = _func(cf, "calculate_mla_kv_cache_dim")
    # after the trtllm / HIP raw-layout early returns, before the 512 + 16 + rope fp8 formula
    assert d.index("if uses_trtllm_kv_layout:") < d.index("rtxattn_dim") < d.index("kv_lora_rank // quant_block_size")
    b = _func(cf, "_build_dsa_kv_pool") if "def _build_dsa_kv_pool(" in cf else cf
    assert b.index("PoolCls = DSATokenToKVPool") < b.index("rtxattn_dsa_pool") < b.index("token_to_kv_pool = PoolCls(")
    mp = hooks.patch_source(_v0520(hooks.MEMORY_POOL), hooks.MEMORY_POOL)[0]
    assert "rtxattn_hybrid_pool" in mp and hooks.HYBRID_ANCHOR not in mp
    be = hooks.patch_source(_v0520(hooks.BACKEND), hooks.BACKEND)[0]
    init = be[be.index("class DeepseekSparseAttnBackend("):]
    init = init[init.index("    def __init__("):]
    init = init[: init.index("\n    def ", 10)]
    # attach after kv_cache_dtype / the pool exist and before the flashinfer validation + workspace allocation
    assert init.index("self.kv_cache_dtype = model_runner.kv_cache_dtype") < init.index("rtxattn_init")
    assert init.index("rtxattn_init") < init.index("_validate_flashinfer_sparse_mla_backend(")
    r = _func(be, "_forward_flashinfer_sparse_mla")
    assert r.index("rtxattn_route") < r.index("return flashinfer_sparse_mla_forward(")
    kp = hooks.patch_source(_v0520(hooks.KPOOL), hooks.KPOOL)[0]
    c = _func(kp, "_check_kpool_tail_backend")
    assert "flashinfer_sparse_mla" in c and c.index("rtxattn_kpool") < c.index("raise NotImplementedError")


def test_forward_paths_reach_the_route():
    """Both forward_decode and forward_extend send dsa_impl == flashinfer_sparse_mla to _forward_flashinfer_sparse_mla
    with the pool buffer and the PAGED page_table_1 (token slots), after the pool write."""
    be = _v0520(hooks.BACKEND)
    for fn in ("forward_decode", "forward_extend"):
        body = _func(be, fn)
        w = body.index("self.token_to_kv_pool.set_mla_kv_buffer(")
        r = body.index("return self._forward_flashinfer_sparse_mla(")
        assert w < r
        assert "kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)" in body
    # the flashinfer impl never takes the RAGGED transform (only flashmla_sparse* does)
    tm = _func(be, "get_topk_transform_method")
    assert 'self.dsa_prefill_impl in ("flashmla_sparse", "flashmla_sparse_q8")' in tm


def test_server_args_accept_explicit_flashinfer_backend():
    """_check_dsa_backend_constraints (overrides.py) refuses only triton (ROCm) and tilelang+fp8 on CUDA; the explicit
    flashinfer_sparse_mla roles pass for Glm5Next (is_glm_sm12_fp8 is GlmMoeDsa-only, so no default is forced)."""
    ov = _v0520("srt/arg_groups/overrides.py")
    c = _func(ov, "_check_dsa_backend_constraints")
    assert 'rocm_only = {"triton"} & chosen' in c and 'cuda_fp8_unsupported = {"tilelang"} & chosen' in c
    s = _func(ov, "_dsa_split_backend_resolution")
    assert 'model_arch == "GlmMoeDsaForCausalLM"' in s
    ex = _v0520("srt/arg_groups/fields/exec_.py")
    assert ex.count('"flashinfer_sparse_mla"') >= 2  # a CLI choice for both roles


def test_full_apply_on_a_copy(tmp_path):
    root = tmp_path / "sglang"
    for path in hooks.FILES:
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_v0520(path))
    lines = hooks.apply(str(root))
    assert sum("applied to" in ln for ln in lines) == len(hooks.HOOKS)
    for rel in hooks.MODULES.values():
        assert (root / rel).exists()
    assert all("already applied" in ln for ln in hooks.apply(str(root), check=True))


def test_refuses_on_glmflash_kv4_tree():
    src = _v0520(hooks.MEMORY_POOL).replace(hooks.HYBRID_ANCHOR, hooks.HYBRID_ANCHOR.rstrip("\n") + "  "
                                            + hooks.CONFLICT_MARKER + "\n")
    with pytest.raises(SystemExit):
        hooks.patch_source(src, hooks.MEMORY_POOL)


# ------------------------------------------------------------------------------------------------ engine logic (CPU)


@pytest.fixture
def eng(monkeypatch):
    monkeypatch.setenv("NUMINOUS_RTXFLASH_ATTN", "1")
    monkeypatch.delenv("NUMINOUS_RTXFLASH_KV", raising=False)
    import glmflash_sm120_engine as E
    return E


def test_validate(eng, monkeypatch):
    fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
    m = "flashinfer_sparse_mla"
    assert eng.validate("Glm5NextForConditionalGeneration", 12, fp8, m, m)
    assert eng.validate("Glm5NextForConditionalGenerationNextN", 12, bf16, m, m)
    assert not eng.validate("GlmMoeDsaForCausalLM", 12, fp8, m, m)  # GLM-5.3 keeps the stock path
    with pytest.raises(ValueError):
        eng.validate("Glm5NextForConditionalGeneration", 10, fp8, m, m)
    with pytest.raises(ValueError):
        eng.validate("Glm5NextForConditionalGeneration", 12, fp8, "tilelang", m)
    monkeypatch.setenv("NUMINOUS_RTXFLASH_ATTN", "0")
    assert not eng.validate("Glm5NextForConditionalGeneration", 12, fp8, m, m)


def test_kv_dims(eng, monkeypatch):
    mc = types.SimpleNamespace(hf_config=types.SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"]),
                               kv_lora_rank=512, qk_rope_head_dim=0)
    assert eng.kv_cache_dim(mc, torch.float8_e4m3fn) == 656
    assert eng.kv_cache_dim(mc, torch.bfloat16) is None
    monkeypatch.setenv("NUMINOUS_RTXFLASH_KV", "fp4")
    assert eng.kv_cache_dim(mc, torch.float8_e4m3fn) == 288
    other = types.SimpleNamespace(hf_config=types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]))
    assert eng.kv_cache_dim(other, torch.float8_e4m3fn) is None
    assert eng.pool_class(object, 512) is object


@pytest.mark.parametrize("fmt", ["bf16", "fp4", "fp8_triton", "fp8_fi"])
def test_engine_forward(eng, fmt):
    import glmflash_sm120_attn as A
    import glmflash_sm120_fi as FI
    from test_glmflash_sm120_fi import FakeModule
    import glmflash_kv4

    n, t, h = 2304, 2, 16
    lat = torch.randn(n, 512)
    if fmt == "bf16":
        buf, f, ref = lat.to(torch.bfloat16).reshape(n, 1, 512), A.FMT_BF16, lat.to(torch.bfloat16).float()
    elif fmt == "fp4":
        rows = glmflash_kv4.quantize_rows_ref(lat)
        buf, f, ref = rows.reshape(n, 1, 288), A.FMT_FP4_288, A.fp4_288_dequant_ref(rows)
    else:
        rows = A.nsa656_quantize_ref(lat)
        rows[0] = 0
        buf, f, ref = rows.reshape(n, 1, 656).view(torch.float8_e4m3fn), A.FMT_FP8_656, A.nsa656_dequant_ref(rows)
    mode = "fi_jit" if fmt == "fp8_fi" else "triton"
    fi = FI.FlashInferGlm5Next("jit", module=FakeModule()) if mode == "fi_jit" else None
    e = eng.Engine(f, h, mode, fi)
    e.tiles = {"block_n": 16, "num_warps": 4, "num_stages": 1}
    q_all = (torch.randn(t, h, 512) * 0.1).to(torch.bfloat16)
    idx = torch.full((t, 2051), -1, dtype=torch.int32)
    idx[0] = torch.randperm(n - 1)[:2051].to(torch.int32) + 1
    idx[1, :9] = torch.arange(1, 10, dtype=torch.int32)
    out = e.forward(q_all, buf, idx, 0.0625, torch.zeros(32 << 20, dtype=torch.uint8))
    ro, _ = A.sparse_mla_ref(q_all.float(), ref, idx, 0.0625)
    assert out.shape == (t, h, 512) and out.dtype == torch.bfloat16
    assert (out.float() - ro).norm() / ro.norm() < 1e-2


def test_tile_table(eng, monkeypatch):
    assert eng.tiles(1, 64) == {"block_n": 64, "num_warps": 8, "num_stages": 1}
    assert eng.tiles(1, 8) == {"block_n": 16, "num_warps": 4, "num_stages": 2}
    assert eng.tiles(2, 1)["block_n"] == 16 and eng.tiles(2, 8)["block_n"] == 32 and eng.tiles(2, 256)["block_n"] == 64
    assert eng.tiles(0, 8192) == {"block_n": 32, "num_warps": 8, "num_stages": 3}
    for k, v in (("NUMINOUS_RTXFLASH_BLOCK_N", "32"), ("NUMINOUS_RTXFLASH_WARPS", "4"), ("NUMINOUS_RTXFLASH_STAGES", "2")):
        monkeypatch.setenv(k, v)
    assert eng.tiles(1, 64) == {"block_n": 32, "num_warps": 4, "num_stages": 2}


# ------------------------------------------------------------------------------------------------ PD coverage


def _fake_engine_module(monkeypatch):
    """Map sglang.srt.layers.attention.dsa.glmflash_sm120_engine to the repo engine (the hooks import it by name)."""
    import glmflash_sm120_engine as E
    for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.attention",
                 "sglang.srt.layers.attention.dsa"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention.dsa.glmflash_sm120_engine", E)
    return E


def _load_patched_transform(tmp_path):
    src, _ = hooks.patch_source(_v0520(hooks.TRANSFORM), hooks.TRANSFORM)
    f = tmp_path / "transform_index_patched.py"
    f.write_text(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("transform_index_patched", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kpool_table(rows, max_seq, width=2051, seed=0):
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(100, max_seq, (rows,), generator=g)
    topk = torch.full((rows, width), -1, dtype=torch.int32)
    for i in range(rows):
        n = min(int(seq[i]), width)
        topk[i, :n] = torch.randperm(int(seq[i]), generator=g)[:n].to(torch.int32)
    return topk


@pytest.mark.parametrize("expanded", [False, True])
def test_pd_transform_prefill_2051(tmp_path, monkeypatch, expanded):
    monkeypatch.setenv("NUMINOUS_RTXFLASH_ATTN", "1")
    _fake_engine_module(monkeypatch)
    T = _load_patched_transform(tmp_path)
    ext = [3, 5]
    n = sum(ext)
    page_table = torch.randint(1, 10_000, (n if expanded else len(ext), 3000), dtype=torch.int32)
    topk = _kpool_table(n, 3000)
    got = T.transform_index_page_table_prefill_fast(page_table=page_table, topk_indices=topk, extend_lens_cpu=ext,
                                                    page_table_is_expanded=expanded, output_num_tokens=n + 2)
    ref = T.transform_index_page_table_prefill_ref(page_table=page_table, topk_indices=topk, extend_lens_cpu=ext,
                                                   page_table_is_expanded=expanded, output_num_tokens=n + 2)
    assert got.shape == (n + 2, 2051) and torch.equal(got, ref)


def test_pd_transform_decode_2051_and_stock_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMINOUS_RTXFLASH_ATTN", "1")
    _fake_engine_module(monkeypatch)
    T = _load_patched_transform(tmp_path)
    page_table = torch.randint(1, 10_000, (6, 3000), dtype=torch.int32)
    topk = _kpool_table(6, 3000)
    got = T.transform_index_page_table_decode_fast(page_table=page_table, topk_indices=topk)
    assert torch.equal(got, T.transform_index_page_table_decode_ref(page_table=page_table, topk_indices=topk))
    out = torch.empty_like(topk)
    assert T.transform_index_page_table_decode_fast(page_table=page_table, topk_indices=topk, result=out) is out
    monkeypatch.setenv("NUMINOUS_RTXFLASH_ATTN", "0")  # inert without the env: the stock asserts stand
    with pytest.raises(AssertionError):
        T.transform_index_page_table_decode_fast(page_table=page_table, topk_indices=topk)
    with pytest.raises(AssertionError):
        T.transform_index_page_table_prefill_fast(page_table=page_table[:1], topk_indices=topk[:2],
                                                  extend_lens_cpu=[2])


def _kpool_mapping_fn(src: str):
    """Exec dsa_indexer_kpool's _kpool_fused_topk_mapping body with stubs (the module itself needs a full sglang)."""
    import ast
    import enum
    import textwrap
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_kpool_fused_topk_mapping")
    code = textwrap.dedent(ast.get_source_segment(src, fn))

    class TopkTransformMethod(enum.Enum):
        PAGED = 1
        RAGGED = 2

    envs = types.SimpleNamespace(SGLANG_DSA_FUSE_TOPK=types.SimpleNamespace(get=lambda: True))
    ns = {"envs": envs, "TopkTransformMethod": TopkTransformMethod, "Optional": __import__("typing").Optional,
          "Tuple": __import__("typing").Tuple, "torch": torch, "BaseIndexerMetadata": object}
    exec(code, ns)
    return ns["_kpool_fused_topk_mapping"], TopkTransformMethod


def test_pd_kpool_honours_force_unfused(monkeypatch):
    monkeypatch.setenv("NUMINOUS_RTXFLASH_ATTN", "1")
    _fake_engine_module(monkeypatch)
    stock = _v0520(hooks.KPOOL_INDEXER)
    patched, _ = hooks.patch_source(stock, hooks.KPOOL_INDEXER)
    pt = torch.zeros(2, 4, dtype=torch.int32)
    for src, expect_unfused in ((stock, False), (patched, True)):
        fn, M = _kpool_mapping_fn(src)
        md = types.SimpleNamespace(topk_transform_method=M.PAGED, force_unfused_topk=True,
                                   attn_metadata=types.SimpleNamespace(page_table_1=pt, topk_indices_offset=None))
        page_table_1, _, _ = fn(md)
        # stock: the flag is ignored and page-mapped slots are produced (the PD bug); patched: raw positions
        assert (page_table_1 is None) == expect_unfused
        md.force_unfused_topk = False
        assert fn(md)[0] is pt  # the normal fused path is unchanged
    monkeypatch.setenv("NUMINOUS_RTXFLASH_ATTN", "0")
    fn, M = _kpool_mapping_fn(patched)
    md = types.SimpleNamespace(topk_transform_method=M.PAGED, force_unfused_topk=True,
                               attn_metadata=types.SimpleNamespace(page_table_1=pt, topk_indices_offset=None))
    assert fn(md)[0] is pt  # inert without the env


def test_pd_unfused_cause_in_source():
    """Why PD needs this: the PD prefill draft-extend backend is built unfused for IndexShare seeds, the backend then
    transforms raw positions, and the kpool indexer's four fused-mapping sites read only the env."""
    u = _v0520("srt/layers/attention/dsa/utils.py")
    f = _func(u, "should_use_dsa_fused_topk")
    assert 'disaggregation_mode != "null" and seed_dsa_topk_from_draft_extend' in f
    be = _v0520(hooks.BACKEND)
    assert "force_unfused = not self.use_fused_topk" in be
    kp = _v0520(hooks.KPOOL_INDEXER)
    assert kp.count("envs.SGLANG_DSA_FUSE_TOPK.get()") == 4 and "force_unfused_topk" not in kp
    ti = _v0520(hooks.TRANSFORM)
    assert ti.count("assert topk_indices.shape[1] == 2048") == 2 and "TOPK: tl.constexpr = 2048" in ti
    ed = _v0520("srt/speculative/eagle_disaggregation.py")
    assert "PD sends request-relative positions" in ed
    patched = hooks.patch_source(kp, hooks.KPOOL_INDEXER)[0]
    assert patched.count("kpool_unfused(metadata)") == 4
