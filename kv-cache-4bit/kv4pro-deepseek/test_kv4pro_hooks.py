"""Tests for operations/patches/kv4pro_hooks.py and operations/kernels/kv4pro_engine.py.

CPU (local, needs the SGLang fork at ~/sglang-fork): every anchor resolves exactly once in the v0.5.20 sources,
patching is idempotent, patched files compile, the pricing expression prices per-pool cells.
Box (needs the installed SGLang v0.5.20 + CUDA): anchors on the installed package, patch a copy of the package and
import the patched modules from it, then drive numinous_kv4pro_engine.forward_kv4 through the real FlashMLA kernels
with stub pools / metadata: decode in modes all and c4 (c4 and c128 layers), query chunking, CUDA-graph replay, and
the sparse-prefill path, each against the fp32 reference over the same stored rows.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "kernels"))
import kv4pro_hooks as H  # noqa: E402

FORK = os.path.expanduser("~/sglang-fork")


def _fork_src(rel):
    return subprocess.run(["git", "-C", FORK, "show", f"v0.5.20:python/sglang/{rel}"], capture_output=True,
                          text=True, check=True).stdout


def _have_fork():
    return os.path.exists(os.path.join(FORK, ".git"))


def _installed():
    try:
        return H.sglang_root()
    except SystemExit:
        return None


fork = pytest.mark.skipif(not _have_fork(), reason="needs ~/sglang-fork")


@fork
def test_anchors_resolve_once_and_compile_on_v0520():
    paths = sorted({h.path for h in H.HOOKS})
    for rel in paths:
        src = _fork_src(rel)
        hooks = [h for h in H.HOOKS if h.path == rel]
        new, lines, problems = H.patch_source(src, hooks)
        assert not problems, problems
        assert all("applied" in ln for ln in lines), lines
        compile(new, rel, "exec")
        again, lines2, problems2 = H.patch_source(new, hooks)
        assert again == new and not problems2 and all("already" in ln for ln in lines2)


@fork
def test_patched_semantics_on_v0520():
    cfg, _, _ = H.patch_source(_fork_src(H.CFG), [h for h in H.HOOKS if h.path == H.CFG])
    assert "self.swa_ratio * _nkv4_swa * self.num_layers_total" in cfg
    assert "c4_frac * _nkv4_c4 * self.num_layers_ca4" in cfg
    assert "1 / 128 * _nkv4_c128 * self.num_layers_ca128" in cfg
    assert cfg.index("numinous-hook:kv4pro_price_cells") < cfg.index("numinous-hook:kv4pro_price_expr")
    pool, _, _ = H.patch_source(_fork_src(H.POOL), [h for h in H.HOOKS if h.path == H.POOL])
    i_sel = pool.index("numinous-hook:kv4pro_pool_select")
    assert i_sel < pool.index("numinous-hook:kv4pro_pool_swa") < pool.index("self._init_compressed_pools(")
    be, _, _ = H.patch_source(_fork_src(H.BACKEND), [h for h in H.HOOKS if h.path == H.BACKEND])
    # the hook runs after the SWA store and before the stock views (which assume the 584-byte cell)
    i = be.index("numinous-hook:kv4pro_forward")
    assert be.index("self.store_cache(layer_id, swa_k, forward_batch)") < i
    assert i < be.index("k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim")


def test_modules_exist():
    for src in H.MODULES:
        assert os.path.isfile(os.path.normpath(src)), src


# ------------------------------------------------------------------------------------------------ box
box = pytest.mark.skipif(_installed() is None or not os.path.isfile("/sgl-workspace/sglang/python/sglang/__init__.py"),
                         reason="needs the installed SGLang box image")


def _cuda():
    import torch

    return torch.cuda.is_available()


@box
def test_box_anchors_and_patched_copy_imports(tmp_path):
    root = _installed()
    assert H.apply(root, [], check=True)
    copy = tmp_path / "pkg" / "sglang"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns("__pycache__", "*.so.*"), symlinks=True)
    out = H.apply(str(copy), [])
    assert any(ln.startswith("module:") for ln in out)
    assert all("already" in ln for ln in H.apply(str(copy), []) if not ln.startswith("module:"))
    code = ("import sglang, sys; assert sglang.__file__.startswith(sys.argv[1]), sglang.__file__\n"
            "import sglang.srt.mem_cache.deepseek_v4_memory_pool as p\n"
            "import sglang.srt.layers.attention.deepseek_v4_backend as b\n"
            "import sglang.srt.layers.attention.dsv4.compressor_v2 as c\n"
            "import sglang.srt.model_executor.pool_configurator as g\n"
            "import sglang.srt.layers.attention.numinous_kv4pro_engine as e\n"
            "cls = e.fp4_pool_cls(); assert issubclass(cls, p.DeepSeekV4SingleKVPool)\n"
            "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code, str(copy)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(copy.parent)}, timeout=600)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-3000:]


def _engine(monkeypatch, mode):
    """Import the engine glue with the repo kernel module standing in for the installed copy."""
    import kv4pro_fp4

    monkeypatch.setenv("NUMINOUS_KV4PRO", mode)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention.numinous_kv4pro", kv4pro_fp4)
    import sglang.srt.layers.attention as attn_pkg

    monkeypatch.setattr(attn_pkg, "numinous_kv4pro", kv4pro_fp4, raising=False)
    spec = importlib.util.spec_from_file_location("kv4pro_engine_t", os.path.join(HERE, "..", "kernels",
                                                                                  "kv4pro_engine.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, kv4pro_fp4


class _Pool4:
    numinous_kv4 = True


def _stub(K, mode, ratio, T, seed, swa_w=192, n_swa=4096, n_ext=16384):
    import torch
    from sgl_kernel.flash_mla import FlashMLASchedMeta

    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(seed)
    swa_x = (torch.randn(n_swa, 512, generator=g) * 1.5).to(dev)
    ext_x = (torch.randn(n_ext, 512, generator=g) * 1.5).to(dev)
    swa4 = mode == "all"
    ext4 = ratio == 4
    ext_page = 64 if ratio == 4 else 2
    swa_buf = K.quantize_ref(swa_x) if swa4 else K.quantize_fp8_paged_ref(swa_x, 128)
    swa_rows = K.dequantize_ref(swa_buf) if swa4 else K.dequantize_fp8_paged_ref(swa_buf, 128, n_swa)
    ext_buf = K.quantize_ref(ext_x) if ext4 else K.quantize_fp8_paged_ref(ext_x, ext_page)
    ext_rows = K.dequantize_ref(ext_buf) if ext4 else K.dequantize_fp8_paged_ref(ext_buf, ext_page, n_ext)
    w_ext = 1024 if ratio == 4 else 448
    core = types.SimpleNamespace(
        swa_page_indices=torch.randint(0, n_swa, (T, swa_w), generator=g).to(dev, torch.int32),
        swa_topk_lengths=torch.randint(1, swa_w + 1, (T,), generator=g).to(dev, torch.int32),
        c4_sparse_page_indices=None, c4_sparse_topk_lengths=None, c128_page_indices=None,
        c128_topk_lengths_clamp1=None)
    idx = torch.randint(0, n_ext, (T, w_ext), generator=g).to(dev, torch.int32)
    idx[:, 3::13] = -1
    ln = torch.randint(1, w_ext + 1, (T,), generator=g).to(dev, torch.int32)
    if ratio == 4:
        core.c4_sparse_page_indices, core.c4_sparse_topk_lengths = idx, ln
    else:
        core.c128_page_indices, core.c128_topk_lengths_clamp1 = idx, ln
    metas = {0: FlashMLASchedMeta(), 4: FlashMLASchedMeta(), 128: FlashMLASchedMeta()}
    core.get_flashmla_metadata = lambda r: metas[r]
    ext_pool = _Pool4() if ext4 else object()
    pool = types.SimpleNamespace(
        swa_kv_pool=_Pool4() if swa4 else object(),
        layer_mapping={7: types.SimpleNamespace(compress_kv_pool=ext_pool)},
        swa_page_size=128,
        get_swa_key_buffer_radix=lambda lid: swa_buf if swa4 else swa_buf.view(torch.float8_e4m3fn),
        get_extra_key_buffer=lambda lid: ext_buf if ext4 else ext_buf.view(torch.float8_e4m3fn),
        get_extra_key_page_size=lambda lid: ext_page)
    backend = types.SimpleNamespace(token_to_kv_pool=pool, head_dim_v=512, softmax_scale=512 ** -0.5)
    q = (torch.randn(T, 128, 512, generator=g) * 0.05).to(dev, torch.bfloat16)
    sink = torch.randn(128, generator=g).to(dev)
    fb = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_extend_without_speculative=lambda: False))
    return dict(core=core, backend=backend, q=q, sink=sink, fb=fb, rows=(swa_rows, ext_rows), idx=idx, ln=ln)


def _decode_err(E, K, c, ratio):
    import torch

    layer = types.SimpleNamespace(layer_id=7)
    out = E.forward_kv4(c["backend"], q=c["q"], layer=layer, forward_batch=c["fb"], compress_ratio=ratio,
                        core_attn_metadata=c["core"], attn_sink=c["sink"])
    core = c["core"]
    ref, _ = K.sparse_decode_ref(c["q"].float(), None, core.swa_page_indices, core.swa_topk_lengths, None, c["idx"],
                                 c["ln"], attn_sink=c["sink"], softmax_scale=512 ** -0.5, kv_rows=c["rows"])
    return out, ((out.float() - ref).norm() / ref.norm()).item()


@box
@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
@pytest.mark.parametrize("mode,ratio", [("all", 4), ("all", 128), ("c4", 4)])
@pytest.mark.parametrize("T,chunk", [(6, 512), (96, 512), (40, 16)])
def test_box_engine_decode(monkeypatch, mode, ratio, T, chunk):
    import flashinfer  # noqa: F401

    monkeypatch.setenv("NUMINOUS_KV4PRO_WS_TOKENS", str(chunk))
    E, K = _engine(monkeypatch, mode)
    c = _stub(K, mode, ratio, T, seed=T + ratio)
    _, err = _decode_err(E, K, c, ratio)
    # FlashMLA on the FP8 requant of the stored rows vs fp32 attention over the stored rows themselves
    assert err < 0.03, err


@box
@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
def test_box_engine_c128_layer_in_mode_c4_falls_through(monkeypatch):
    E, K = _engine(monkeypatch, "c4")
    c = _stub(K, "c4", 128, 4, seed=3)
    out = E.forward_kv4(c["backend"], q=c["q"], layer=types.SimpleNamespace(layer_id=7), forward_batch=c["fb"],
                        compress_ratio=128, core_attn_metadata=c["core"], attn_sink=c["sink"])
    assert out is None


@box
@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
def test_box_engine_decode_cuda_graph(monkeypatch):
    import flashinfer  # noqa: F401
    import torch

    monkeypatch.setenv("NUMINOUS_KV4PRO_WS_TOKENS", "32")
    E, K = _engine(monkeypatch, "all")
    c = _stub(K, "all", 4, 48, seed=21)  # 2 chunks inside the graph
    layer = types.SimpleNamespace(layer_id=7)

    def run():
        return E.forward_kv4(c["backend"], q=c["q"], layer=layer, forward_batch=c["fb"], compress_ratio=4,
                             core_attn_metadata=c["core"], attn_sink=c["sink"])

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = run()
    c2 = _stub(K, "all", 4, 48, seed=22)
    for f in ("swa_page_indices", "swa_topk_lengths", "c4_sparse_page_indices", "c4_sparse_topk_lengths"):
        getattr(c["core"], f).copy_(getattr(c2["core"], f))
    gr.replay()
    torch.cuda.synchronize()
    core = c["core"]
    ref, _ = K.sparse_decode_ref(c["q"].float(), None, core.swa_page_indices, core.swa_topk_lengths, None,
                                 core.c4_sparse_page_indices, core.c4_sparse_topk_lengths, attn_sink=c["sink"],
                                 softmax_scale=512 ** -0.5, kv_rows=c["rows"])
    assert ((out.float() - ref).norm() / ref.norm()).item() < 0.03


@box
@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
def test_box_engine_sparse_prefill_c0(monkeypatch):
    """compress_ratio 0 prefill over a 4-bit SWA pool == flash_mla_sparse_fwd on the exactly dequantized rows."""
    import flashinfer  # noqa: F401
    import torch
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import SparsePrefillChunkCache, SparsePrefillWorkspace

    monkeypatch.setenv("SGLANG_OPT_FLASHMLA_SPARSE_PREFILL", "1")
    E, K = _engine(monkeypatch, "all")
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(5)
    seq, ext = [300, 77], [120, 77]
    n_tok = 1024
    req_to_token = torch.zeros(4, 1024, dtype=torch.int32, device=dev)
    perm = torch.randperm(n_tok - 1, generator=g).to(dev, torch.int32) + 1
    req_to_token[0, :300] = perm[:300]
    req_to_token[1, :77] = perm[300:377]
    full_to_swa = torch.arange(n_tok, device=dev, dtype=torch.int64)
    total_swa = sum(min(s, e + 128 - 1) for s, e in zip(seq, ext))
    cache = SparsePrefillChunkCache.build(
        seq_lens=torch.tensor(seq, dtype=torch.int32, device=dev),
        extend_seq_lens=torch.tensor(ext, dtype=torch.int32, device=dev),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32, device=dev), req_to_token=req_to_token,
        full_to_swa=full_to_swa, swa_window_size=128, swa_page_size=128, num_qo_tokens=sum(ext),
        max_seq_len=max(seq), total_swa=total_swa)
    swa_x = (torch.randn(n_tok, 512, generator=g) * 1.5).to(dev)
    swa_buf = K.quantize_ref(swa_x)
    pool = types.SimpleNamespace(swa_kv_pool=_Pool4(), layer_mapping={7: types.SimpleNamespace(compress_kv_pool=None)},
                                 get_swa_key_buffer_radix=lambda lid: swa_buf)
    backend = types.SimpleNamespace(token_to_kv_pool=pool, head_dim_v=512, softmax_scale=512 ** -0.5,
                                    forward_metadata=types.SimpleNamespace(sparse_prefill_cache=cache),
                                    sparse_prefill_workspace=SparsePrefillWorkspace(torch.device(dev)))
    fb = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_extend_without_speculative=lambda: True))
    q = (torch.randn(sum(ext), 1, 128, 512, generator=g) * 0.05).to(dev, torch.bfloat16)
    sink = torch.randn(128, generator=g).to(dev)
    out = E.forward_kv4(backend, q=q, layer=types.SimpleNamespace(layer_id=7), forward_batch=fb, compress_ratio=0,
                        core_attn_metadata=types.SimpleNamespace(), attn_sink=sink)
    rows = K.dequantize_ref(swa_buf)[cache.swa_token_ids.long()].to(torch.bfloat16).unsqueeze(1)
    ref, _, _ = flash_mla_sparse_fwd(q=q.squeeze(1), kv=rows, indices=cache.c0_combined_indices.unsqueeze(1),
                                     sm_scale=512 ** -0.5, d_v=512, attn_sink=sink, topk_length=cache.c0_combined_lens)
    assert torch.equal(out, ref)


_REAL_POOL_SCRIPT = r'''
import os, sys, types, json, torch
sys.path.insert(0, sys.argv[2])  # repo operations/kernels (references)
import flashinfer  # noqa: F401
import kv4pro_fp4 as K
from sglang.srt.runtime_context import get_context
get_context().override_server_args(speculative_algorithm="DSPARK", speculative_num_draft_tokens=6,
                                   swa_full_tokens_ratio=0.1, max_running_requests=512).install()
import sglang.srt.mem_cache.deepseek_v4_memory_pool as P
import sglang.srt.model_executor.pool_configurator as C
mode = os.environ["NUMINOUS_KV4PRO"]
ratios = [128, 128, 4, 128, 4]
pool = P.DeepSeekV4TokenToKVPool(
    max_num_reqs=8, swa_size=8192, c4_size=2048, c128_size=64, c4_state_pool_size=1024, c128_state_pool_size=64,
    page_size=256, swa_page_size=128, dtype=torch.float8_e4m3fn, c4_state_dtype=torch.float32,
    c128_state_dtype=torch.float32, qk_nope_head_dim=448, qk_rope_head_dim=64, indexer_head_dim=128, layer_num=5,
    device="cuda", enable_memory_saver=False, compression_ratios=ratios, sliding_window=128)
out = {"swa": type(pool.swa_kv_pool).__name__, "c4": type(pool.kv_pools[4]).__name__,
       "c128": type(pool.kv_pools[128]).__name__, "swa_row": pool.swa_kv_pool.kv_buffer[0].shape[1],
       "c4_row": pool.kv_pools[4].kv_buffer[0].shape[1], "c128_row": pool.kv_pools[128].kv_buffer[0].shape[1]}
# SWA store through the patched norm+rope route, identity rope (freqs 1+0j), then read back
g = torch.Generator(device="cpu").manual_seed(0)
kv = (torch.randn(64, 512, generator=g) * 3).cuda().to(torch.bfloat16)
w = (torch.rand(512, generator=g) + 0.5).cuda().to(torch.bfloat16)
freqs = torch.ones(4096, 32, dtype=torch.complex64, device="cuda")
loc = torch.arange(64, device="cuda", dtype=torch.int32) + 300
expect = (kv.float() * torch.rsqrt(kv.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float())
pool.set_swa_key_buffer_radix_fused_norm_rope(layer_id=2, swa_loc=loc, kv=kv.clone(), kv_weight=w, eps=1e-6,
                                              freqs_cis=freqs, positions=torch.arange(64, device="cuda"))
if out["swa"] == "DeepSeekV4FP4KVPool":
    rows = K.dequantize_ref(pool.get_swa_key_buffer_radix(2).view(-1, 320)[300:364])
    out["swa_store_rel_err"] = ((rows - expect).norm() / expect.norm()).item()
# c4 store (compressor route lands in set_extra_key_buffer_fused)
pool.set_extra_key_buffer_fused(layer_id=2, loc=loc, cache_k=expect.to(torch.bfloat16))
if out["c4"] == "DeepSeekV4FP4KVPool":
    rows = K.dequantize_ref(pool.get_extra_key_buffer(2).view(-1, 320)[300:364])
    out["c4_store_rel_err"] = ((rows - expect).norm() / expect.norm()).item()
# pool pricing with the Pro 0813 layer ratios
mc = types.SimpleNamespace(qk_nope_head_dim=448, qk_rope_head_dim=64, index_head_dim=128, context_len=1048576,
                           compress_ratios=[128, 128] + [4, 128] * 29 + [4], window_size=128)
kvc = types.SimpleNamespace(kv_cache_dtype_str="fp8_e4m3", model_config=mc,
                            layer_info=types.SimpleNamespace(start_layer=0, end_layer=61),
                            ps=types.SimpleNamespace(pp_size=1, attn_dp_size=8), sliding_window_size=128,
                            spec_algorithm=None)
out["bytes_per_full_token"] = C.DSV4PoolConfigurator(kvc).bytes_per_full_token
print("RESULT " + json.dumps(out))
'''


@box
@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
@pytest.mark.parametrize("mode", ["0", "c4", "all"])
@pytest.mark.parametrize("state", ["float32", "bf16"])
def test_box_real_pool_store_and_pricing(tmp_path, mode, state):
    """The patched DeepSeekV4TokenToKVPool + DSV4PoolConfigurator from a patched package copy: sub-pool classes and
    row widths, the SWA norm+RoPE store route and the compressor store route, and the priced bytes per token."""
    import json

    root = _installed()
    copy = tmp_path / "pkg" / "sglang"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
    H.apply(str(copy), [])
    script = tmp_path / "real_pool.py"
    script.write_text(_REAL_POOL_SCRIPT)
    env = {**os.environ, "PYTHONPATH": str(copy.parent), "NUMINOUS_KV4PRO": mode,
           "SGLANG_DSV4_COMPRESS_STATE_DTYPE": state}
    r = subprocess.run([sys.executable, str(script), str(copy), os.path.join(HERE, "..", "kernels")],
                       capture_output=True, text=True, env=env, timeout=900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and line, r.stderr[-4000:]
    out = json.loads(line[0][7:])
    import kv4pro_fp4 as K

    swa4, c44 = mode == "all", mode in ("c4", "all")
    assert out["swa"] == ("DeepSeekV4FP4KVPool" if swa4 else "DeepSeekV4SingleKVPool")
    assert out["c4"] == ("DeepSeekV4FP4KVPool" if c44 else "DeepSeekV4SingleKVPool")
    assert out["c128"] == "DeepSeekV4SingleKVPool"
    assert out["swa_row"] == 128 * (320 if swa4 else 1) or not swa4
    assert out["c4_row"] == (64 * 320 if c44 else -(-64 * 584 // 576) * 576)
    for k in ("swa_store_rel_err", "c4_store_rel_err"):
        if k in out:
            assert out[k] < 0.1, out
    want = K.bytes_per_full_token(swa_cell=320 if swa4 else 584, comp_cell=320 if c44 else 584, c128_cell=584,
                                  state_bytes=2 if state == "bf16" else 4)
    assert abs(out["bytes_per_full_token"] - want) < 1e-6 * want, (out["bytes_per_full_token"], want)
