"""kv4fi_flashinfer_hooks: anchors apply to the installed SGLang (box) or to a v0.5.20 checkout copy (CPU), the
patched modules import, the validator accepts flashinfer + nvfp4, the registry resolves a NATIVE_FP4 decode access
for flashinfer, and the injected `_kv4fi_forward` runs the kernel over a fake wrapper's planned buffers (GPU)."""
import importlib
import importlib.util
import os
import pathlib
import subprocess
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kv4fi_flashinfer_hooks as h  # noqa: E402

FORK = pathlib.Path.home() / "sglang-fork"


def _v0520_copy(tmp_path) -> pathlib.Path:
    """The three target files from the local v0.5.20 tag (CPU host) or from the installed package (box)."""
    root = tmp_path / "sglang"
    if importlib.util.find_spec("sglang") is not None:
        src = pathlib.Path(h.sglang_root())
        for t in (h.VALIDATOR, h.REGISTRY, h.BACKEND):
            (root / t).parent.mkdir(parents=True, exist_ok=True)
            (root / t).write_text((src / t).read_text())
        return root
    if not (FORK / ".git").exists():
        pytest.skip("needs the installed SGLang or ~/sglang-fork")
    for t in (h.VALIDATOR, h.REGISTRY, h.BACKEND):
        (root / t).parent.mkdir(parents=True, exist_ok=True)
        out = subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{t}"],
                             capture_output=True, text=True, check=True).stdout
        (root / t).write_text(out)
    return root


def test_anchors_apply_and_are_idempotent(tmp_path):
    root = _v0520_copy(tmp_path)
    assert h.main(["--root", str(root)]) == 0
    for t in (h.VALIDATOR, h.REGISTRY, h.BACKEND):
        assert h.MARK in (root / t).read_text(), t
    assert (root / h.KERNEL_DST).exists()
    backend = (root / h.BACKEND).read_text()
    assert backend.count("def _kv4fi_forward") == 1
    assert "self.kv4fi_native_decode" in backend
    # the prefill scale fix: both paged calls over the dequant workspace drop the FP8 per-tensor scales
    assert backend.count("k_scale=None if self.prefill_uses_dequant_workspace else layer.k_scale_float") == 2
    assert backend.count("v_scale=None if self.prefill_uses_dequant_workspace else layer.v_scale_float") == 2
    assert h.main(["--root", str(root), "--check"]) == 0
    # a second apply is a no-op
    before = {t: (root / t).read_text() for t in (h.VALIDATOR, h.REGISTRY, h.BACKEND)}
    assert h.main(["--root", str(root)]) == 0
    assert before == {t: (root / t).read_text() for t in (h.VALIDATOR, h.REGISTRY, h.BACKEND)}


def test_missing_anchor_fails_loudly():
    with pytest.raises(SystemExit):
        h.apply("def nothing(): pass\n", h.VALIDATOR)


def test_validator_list_and_registry_text(tmp_path):
    root = _v0520_copy(tmp_path)
    h.main(["--root", str(root)])
    validator = (root / h.VALIDATOR).read_text()
    i = validator.index("KV4_ATTENTION_MHA_BACKEND_CHOICES = [")
    block = validator[i : validator.index("]", i)]
    assert '"flashinfer"' in block and '"trtllm_mha"' in block
    registry = (root / h.REGISTRY).read_text()
    assert '_NVFP4_DECODE_BACKENDS = frozenset({"trtllm_mha", "flashinfer"})' in registry


needs_box = pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="needs the installed SGLang (box)")


@needs_box
def test_registry_resolves_native_fp4_decode_for_flashinfer(tmp_path):
    root = _v0520_copy(tmp_path)
    h.main(["--root", str(root)])
    spec = importlib.util.spec_from_file_location("kv4fi_registry_patched", root / h.REGISTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    method = mod.NVFP4KVCacheMethod(num_layers=4, device="cpu")
    dec = method.resolve_attention_access("decode", "flashinfer")
    pre = method.resolve_attention_access("prefill", "flashinfer")
    assert dec is not None and dec.kind == mod.KVCacheAttentionAccessKind.NATIVE_FP4 and dec.scale_recipe == "nvfp4"
    assert pre is not None and pre.kind == mod.KVCacheAttentionAccessKind.DEQUANT_WORKSPACE
    assert method.resolve_attention_access("decode", "trtllm_mha").kind == mod.KVCacheAttentionAccessKind.NATIVE_FP4
    assert method.resolve_attention_access("decode", "triton") is None
    # the pool still allocates exactly one FP8 dequant workspace (prefill) and packed FP4 + scale buffers
    import torch
    assert method.dequant_workspace_dtype() == torch.float8_e4m3fn
    assert method.kv_storage_dtype() == torch.uint8


@needs_box
def test_validator_accepts_flashinfer_nvfp4(tmp_path):
    """Run the patched handle_kv4_compatibility through the real resolver helpers with a stub server_args."""
    root = _v0520_copy(tmp_path)
    h.main(["--root", str(root)])
    import torch
    if not torch.cuda.is_available():
        pytest.skip("validator checks the platform")
    import sglang.srt.arg_groups.kv_cache_hook as stock
    src = (root / h.VALIDATOR).read_text()
    mod = types.ModuleType("kv4fi_validator_patched")
    mod.__dict__.update({k: v for k, v in vars(stock).items() if not k.startswith("__")})
    exec(compile(src, str(root / h.VALIDATOR), "exec"), mod.__dict__)
    cfg = types.SimpleNamespace(kv_cache_dtype="nvfp4", attention_backend="flashinfer",
                                prefill_attention_backend=None, decode_attention_backend=None)
    # patch the resolver views to return our stub and MHA
    mod.resolving_view = lambda sa: cfg
    mod.resolved_view = lambda sa: cfg
    mod.attention_backends_of = lambda view: ("flashinfer", "flashinfer")
    mod.use_mla_backend = lambda sa: False
    mod.handle_kv4_compatibility(cfg)  # stock raises AssertionError here
    stock_src = (pathlib.Path(h.sglang_root()) / h.VALIDATOR).read_text()
    if h.MARK in stock_src:
        return  # the installed package is already patched on this box; the stock-raises half needs a stock copy
    with pytest.raises(AssertionError):
        stock_mod = types.ModuleType("kv4fi_validator_stock")
        stock_mod.__dict__.update(mod.__dict__)
        exec(compile(stock_src, "stock", "exec"), stock_mod.__dict__)
        stock_mod.resolving_view = lambda sa: cfg
        stock_mod.resolved_view = lambda sa: cfg
        stock_mod.attention_backends_of = lambda view: ("flashinfer", "flashinfer")
        stock_mod.use_mla_backend = lambda sa: False
        stock_mod.handle_kv4_compatibility(cfg)


@needs_box
def test_patched_backend_imports_and_kv4fi_forward_runs(tmp_path):
    """Import the patched flashinfer_backend as a module, then drive `_kv4fi_forward` on a minimal fake self."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU")
    import flashinfer  # noqa: F401
    root = _v0520_copy(tmp_path)
    h.main(["--root", str(root)])
    # the kernel module the backend imports must resolve to the installed path: install into a temp package tree
    spec = importlib.util.spec_from_file_location("sglang.srt.layers.attention.kv4fi_nvfp4_decode",
                                                  root / h.KERNEL_DST)
    kmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kmod)
    sys.modules["sglang.srt.layers.attention.kv4fi_nvfp4_decode"] = kmod
    bspec = importlib.util.spec_from_file_location("kv4fi_backend_patched", root / h.BACKEND)
    bmod = importlib.util.module_from_spec(bspec)
    sys.modules["kv4fi_backend_patched"] = bmod  # dataclasses resolve annotations through sys.modules
    bspec.loader.exec_module(bmod)
    Backend = bmod.FlashInferAttnBackend
    assert hasattr(Backend, "_kv4fi_forward")

    hq, hk, d, L = 24, 4, 256, 500
    dev = "cuda"
    k_fp4 = torch.randint(0, 256, (L + 8, hk, d // 2), dtype=torch.uint8, device=dev)
    v_fp4 = torch.randint(0, 256, (L + 8, hk, d // 2), dtype=torch.uint8, device=dev)
    k_sf = torch.randint(0x30, 0x40, (L + 8, hk, d // 16), dtype=torch.uint8, device=dev)
    v_sf = torch.randint(0x30, 0x40, (L + 8, hk, d // 16), dtype=torch.uint8, device=dev)

    class Pool:
        def get_raw_kv_buffer(self, layer_id):
            return k_fp4, v_fp4, k_sf.view(torch.float8_e4m3fn), v_sf.view(torch.float8_e4m3fn)

    class QM:
        def get_bmm_scales(self, layer_id):
            return 1.25, 0.75

    fake = types.SimpleNamespace(token_to_kv_pool=Pool(), kv_cache_quant_method=QM())
    layer = types.SimpleNamespace(is_cross_attention=False, sliding_window_size=-1, layer_id=3, tp_q_head_num=hq,
                                  head_dim=d, scaling=d**-0.5, logit_cap=0.0)
    ql = 4
    wrapper = types.SimpleNamespace(
        _paged_kv_indptr_buf=torch.tensor([0, L], dtype=torch.int32, device=dev),
        _paged_kv_indices_buf=torch.randperm(L + 8, device=dev)[:L].to(torch.int32),
        _qo_indptr_buf=torch.tensor([0, ql], dtype=torch.int32, device=dev),
    )
    q = torch.randn((ql, hq * d), device=dev, dtype=torch.bfloat16)
    o = Backend._kv4fi_forward(fake, q, layer, wrapper, ql, None, None)
    assert o.shape == (ql, hq * d)
    ref = kmod.nvfp4_attention_ref(q.view(ql, hq, d), k_fp4, k_sf, v_fp4, v_sf, wrapper._paged_kv_indptr_buf,
                                   wrapper._paged_kv_indices_buf, 1.25, 0.75, d**-0.5, ql=ql)
    err = (o.float().view(ql, hq, d) - ref.float()).norm() / ref.float().norm()
    assert err < 2.5e-2, err
    o2, lse = Backend._kv4fi_forward(fake, q, layer, wrapper, ql, None, None, causal=False, return_lse=True)
    assert o2.shape == (ql, hq, d) and lse.shape == (ql, hq)
    layer_swa = types.SimpleNamespace(**{**vars(layer), "sliding_window_size": 1024})
    with pytest.raises(NotImplementedError):
        Backend._kv4fi_forward(fake, q, layer_swa, wrapper, ql, None, None)
