"""kv4trt_hooks: anchors apply to the v0.5.20 trtllm_mha_backend.py (CPU, from the local sglang-fork tag or the
installed package), the XQA causal-mask builder produces the documented bit layout (CPU), and on a GPU box the
patched module imports under the real package and its `_numinous_fp4_decode_mode_extend` (the TARGET_VERIFY /
DRAFT_EXTEND_V2 path) writes K/V through SGLang's NVFP4 quant method and matches a causal reference over the
dequantized cache, with and without CUDA-graph capture.

    python3 -m pytest -q operations/patches/test_kv4trt_hooks.py                         # CPU parts
    python3 operations/devbox/devbox.py test rtx-qwen38 operations/patches/test_kv4trt_hooks.py -q -s   # all
"""
import importlib.util
import math
import os
import pathlib
import re
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kv4trt_hooks as h  # noqa: E402

FORK = pathlib.Path("/home/ubuntu/sglang-fork")
HAVE_SGL = importlib.util.find_spec("sglang") is not None
need_gpu = pytest.mark.skipif(not (HAVE_SGL and torch.cuda.is_available()), reason="needs SGLang + a CUDA GPU")


def _stock_src(target: str = h.TARGET) -> str:
    if HAVE_SGL:
        return open(os.path.join(h.sglang_root(), target)).read()
    if FORK.exists():
        return subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{target}"],
                              capture_output=True, text=True, check=True).stdout
    pytest.skip("no SGLang source (installed package or local sglang-fork)")


@pytest.mark.parametrize("target", [h.HYBRID, h.DRAFT, h.REGISTRY, h.FLASHINFER])
def test_other_targets_apply_compile_idempotent(target):
    src = _stock_src(target)
    new = h.apply(src, target)
    compile(new, target, "exec")
    assert h.MARK in new and h.apply(new, target) == new


def test_main_patches_every_target_in_a_copy(tmp_path):
    for t in h.TARGETS:
        (tmp_path / t).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / t).write_text(_stock_src(t))
    assert h.main(["--root", str(tmp_path), "--check"]) == 0
    assert all(h.MARK not in (tmp_path / t).read_text() for t in h.TARGETS)   # --check writes nothing
    assert h.main(["--root", str(tmp_path)]) == 0
    assert all(h.MARK in (tmp_path / t).read_text() for t in h.TARGETS)
    assert h.main(["--root", str(tmp_path)]) == 0                               # idempotent


def test_anchors_apply_compile_idempotent():
    src = _stock_src()
    new = h.apply(src)
    compile(new, h.TARGET, "exec")
    assert new.count(h.MARK) >= 6
    assert h.apply(new) == new
    # __init__ ordering: the mask prebuild reads is_xqa_impl / speculative_num_draft_tokens, both set before it
    init = new[new.index("class TRTLLMHAAttnBackend"):new.index("    def _check_decode_kv_access")]
    call = init.index("self._numinous_xqa_causal_mask(int(self.speculative_num_draft_tokens)")
    assert init.index("self.is_xqa_impl = ") < call
    assert init.index("self.speculative_num_draft_tokens = ") < call
    assert init.index("self._numinous_masks = {}") < call
    # real prefills with native FP4 still raise (the error text survives)
    assert "supports decode only" in new
    with pytest.raises(SystemExit):
        h.apply("x = 1\n")


def _mask_fn():
    src = h.apply(_stock_src())
    m = re.search(r"    def _numinous_xqa_causal_mask\(self.*?\n(?=\n    def )", src, re.S)
    ns = {"torch": torch}
    exec(compile("\n".join(ln[4:] for ln in m.group(0).splitlines()), "mask", "exec"), ns)
    return ns["_numinous_xqa_causal_mask"]


@pytest.mark.parametrize("q_len", [2, 4, 5, 31, 33])
def test_mask_layout(q_len, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)  # CPU-only torch has no CUDA
    fn = _mask_fn()
    self = types.SimpleNamespace(is_xqa_impl=True, _numinous_masks={}, _numinous_mask_rows=3, device="cpu")
    m = fn(self, q_len)
    words = ((q_len + 31) // 32) * 2
    assert m.dtype == torch.uint16 and tuple(m.shape) == (3, q_len, words) and m.is_contiguous()
    v = m[0].to(torch.int64)
    for j in range(q_len):
        bits = [(int(v[j, w]) >> b) & 1 for w in range(words) for b in range(16)]
        assert bits[: j + 1] == [1] * (j + 1) and not any(bits[j + 1:]), (q_len, j)
    assert fn(self, q_len) is m                               # cached
    assert fn(types.SimpleNamespace(is_xqa_impl=False), q_len) is None   # trtllm-gen takes no mask
    assert fn(self, 1) is None


# ---------------------------------------------------------------------------------------------------------- GPU box
def _patched_module(target: str = h.TARGET):
    """A patched engine module loaded under a separate name (its imports resolve to the installed SGLang), without
    touching the installed file."""
    import flashinfer  # noqa: F401  (engine import order: sets the toolkit ptxas path)
    importlib.import_module("sglang." + target[:-3].replace("/", "."))   # stock module + its imports first
    name = "numinous_kv4trt_patched_" + target[:-3].replace("/", "_")
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(h.sglang_root(), target)
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    sys.modules[name] = mod                                # dataclasses resolve their module through sys.modules
    exec(compile(h.apply(open(path).read(), target), path, "exec"), mod.__dict__)
    return mod


@need_gpu
@pytest.mark.parametrize("fp4", [True, False])
@pytest.mark.parametrize("spec_mode", ["prefill", "decode"])
def test_hybrid_routing(fp4, spec_mode):
    """HybridAttnBackend: with a native-FP4 decode child, verify + draft-extend-v2 go to it; prompt extend and
    the draft prompt extend (EXTEND) stay on the prefill child; without FP4 the stock routing is unchanged."""
    mod = _patched_module(h.HYBRID)
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    mod.get_spec = lambda: types.SimpleNamespace(speculative_attention_mode=spec_mode, speculative_algorithm=None)
    pf = types.SimpleNamespace(needs_cpu_seq_lens=False, decode_uses_native_fp4=False)
    dc = types.SimpleNamespace(needs_cpu_seq_lens=False, decode_uses_native_fp4=fp4)
    mr = types.SimpleNamespace(kv_cache_dtype=None, token_to_kv_pool=None, req_to_token_pool=None,
                               kv_index_translator=None,
                               model_config=types.SimpleNamespace(context_len=4096))
    hb = mod.HybridAttnBackend(mr, pf, dc)
    sel = hb._select_backend
    assert sel(ForwardMode.DECODE) is dc and sel(ForwardMode.EXTEND) is pf
    stock_verify = dc if spec_mode == "decode" else pf
    assert sel(ForwardMode.TARGET_VERIFY) is (dc if fp4 else stock_verify)
    assert sel(ForwardMode.DRAFT_EXTEND_V2) is (dc if fp4 else pf)
    assert hb.spec_attn_is_decode == (fp4 or spec_mode == "decode")



@need_gpu
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("mode,hq,hkv,q_len", [("verify", 24, 4, 4), ("draft_extend_v2", 16, 2, 5)])
def test_fp4_decode_mode_extend_matches_reference(mode, hq, hkv, q_len, graph):
    mod = _patched_module()
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import NVFP4KVCacheMethod
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    dev, d, page = "cuda", 256, 64
    seqs = [70, 1500, 9000]                       # total lengths incl. the q_len new tokens
    pages_per = [math.ceil(s / page) for s in seqs]
    n_tok = (sum(pages_per) + 1) * page
    qm = NVFP4KVCacheMethod(num_layers=1, device=dev)
    qm.k_scales_gpu.fill_(0.02)
    qm.v_scales_gpu.fill_(0.01)
    qm.k_scales_float = [0.02]
    qm.v_scales_float = [0.01]
    buf = qm.create_buffers(n_tok, hkv, d, 1, dev)

    class Pool:  # the pool calls SGLang makes, over the real NVFP4 quant method and buffers
        def set_kv_buffer(self, layer, loc, k, v, ks=None, vs=None):
            qm.quantize_and_store(buf["k_buffer"][0], buf["v_buffer"][0], buf["k_scale_buffer"][0],
                                  buf["v_scale_buffer"][0], loc.loc if hasattr(loc, "loc") else loc[0],
                                  k.view(-1, hkv, d), v.view(-1, hkv, d),
                                  qm.k_scales_gpu[:1], qm.v_scales_gpu[:1])

        def get_raw_kv_buffer(self, layer_id):
            return (buf["k_buffer"][0], buf["v_buffer"][0], buf["k_scale_buffer"][0].view(torch.float8_e4m3fn),
                    buf["v_scale_buffer"][0].view(torch.float8_e4m3fn))

    g = torch.Generator(device=dev).manual_seed(7)
    perm = torch.randperm(n_tok // page - 1, generator=torch.Generator().manual_seed(1)) + 1
    bt = torch.zeros(len(seqs), max(pages_per), dtype=torch.int32)
    off = 0
    for i, p in enumerate(pages_per):
        bt[i, :p] = perm[off: off + p]
        off += p
    bt = bt.to(dev)
    slot = lambda i, t: int(bt[i, t // page]) * page + t % page  # noqa: E731
    # prefix already in the cache (written through the same quant method), new tokens written by the hook
    k_all = (torch.randn(n_tok, hkv, d, device=dev, generator=g) * 2).bfloat16()
    v_all = torch.randn(n_tok, hkv, d, device=dev, generator=g).bfloat16()
    pre = torch.tensor([slot(i, t) for i, L in enumerate(seqs) for t in range(L - q_len)], device=dev)
    new = torch.tensor([slot(i, t) for i, L in enumerate(seqs) for t in range(L - q_len, L)], device=dev)
    pool = Pool()
    pool.set_kv_buffer(None, (pre,), k_all[pre], v_all[pre])

    be = object.__new__(mod.TRTLLMHAAttnBackend)
    be.is_xqa_impl = torch.cuda.get_device_capability()[0] in (9, 12)
    be.is_nvfp4_kvcache = True
    be.decode_uses_native_fp4 = True
    be.token_to_kv_pool = pool
    be.kv_cache_quant_method = qm
    be.q_data_type = torch.bfloat16
    be.page_size = page
    be.max_context_len = 131072
    be.device = dev
    be.decode_seq_len_splits = 1
    be._multi_ctas_kv_counter_buffer = None
    be.workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    be._numinous_masks = {}
    be._numinous_mask_rows = 64
    be._numinous_xqa_causal_mask(q_len, dev)          # prebuilt, as __init__ does
    be.forward_metadata = mod.TRTLLMMHAMetadata(
        cache_seqlens_int32=torch.tensor(seqs, dtype=torch.int32, device=dev), max_seq_len_q=q_len,
        page_table=bt)
    be._get_layer_page_table = lambda layer, fb: bt
    layer = types.SimpleNamespace(tp_q_head_num=hq, tp_k_head_num=hkv, tp_v_head_num=hkv, head_dim=d, layer_id=0,
                                  scaling=1 / math.sqrt(d), sliding_window_size=-1, attn_type=None,
                                  k_scale=None, v_scale=None)
    fm = types.SimpleNamespace(is_target_verify=lambda: mode == "verify",
                               is_draft_extend_v2=lambda: mode == "draft_extend_v2")
    fb = types.SimpleNamespace(forward_mode=fm, out_cache_loc=new)
    q = torch.randn(len(seqs) * q_len, hq * d, device=dev, generator=g).bfloat16()
    k_new, v_new = k_all[new].reshape(-1, hkv * d), v_all[new].reshape(-1, hkv * d)

    # route check: forward_extend dispatches this batch to the hook path (a real prefill would raise)
    mod.is_cp_active = lambda fb: False
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            be.forward_extend(q, k_new, v_new, layer, fb)   # warm-up (JIT) outside capture
        torch.cuda.current_stream().wait_stream(s)
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            o = be.forward_extend(q, k_new, v_new, layer, fb)
        buf["k_buffer"][0][new] = 0                         # replay must rewrite the new tokens itself
        cg.replay()
        torch.cuda.synchronize()
    else:
        o = be.forward_extend(q, k_new, v_new, layer, fb)
    fm_prefill = types.SimpleNamespace(is_target_verify=lambda: False, is_draft_extend_v2=lambda: False)
    with pytest.raises(RuntimeError, match="decode only"):
        be.forward_extend(q, k_new, v_new, layer, types.SimpleNamespace(forward_mode=fm_prefill, out_cache_loc=new))

    # reference: dequantize the whole cache the way the pool stores it, causal over each request's draft block
    kd = NVFP4KVQuantizeUtil.dequantize(buf["k_buffer"][0], buf["k_scale_buffer"][0].view(torch.float8_e4m3fn),
                                        qm.k_scales_gpu[:1]).float()
    vd = NVFP4KVQuantizeUtil.dequantize(buf["v_buffer"][0], buf["v_scale_buffer"][0].view(torch.float8_e4m3fn),
                                        qm.v_scales_gpu[:1]).float()
    qv = q.view(len(seqs), q_len, hq, d).float()
    ref = []
    for i, L in enumerate(seqs):
        idx = torch.tensor([slot(i, t) for t in range(L)], device=dev)
        k = kd[idx].repeat_interleave(hq // hkv, 1)
        v = vd[idx].repeat_interleave(hq // hkv, 1)
        for j in range(q_len):
            n = L - q_len + j + 1
            p = torch.softmax(torch.einsum("hd,lhd->hl", qv[i, j], k[:n]) / math.sqrt(d), -1)
            ref.append(torch.einsum("hl,lhd->hd", p, v[:n]))
    ref = torch.stack(ref).reshape(-1, hq * d)
    err = float((o.float() - ref).norm() / ref.norm())
    # the written new tokens really are the quantized k_new (not stale / zero)
    kq = NVFP4KVQuantizeUtil.dequantize(buf["k_buffer"][0][new], buf["k_scale_buffer"][0][new].view(
        torch.float8_e4m3fn), qm.k_scales_gpu[:1]).float()
    werr = float((kq - k_all[new].float()).norm() / k_all[new].float().norm())
    print(f"\n[kv4trt-hook] {mode} hq{hq}/hkv{hkv} q_len {q_len} graph={graph}: rel_err {err:.4f}, "
          f"new-token FP4 write err {werr:.3f}")
    assert o.shape == (len(seqs) * q_len, hq * d)
    assert err < 0.02, err
    assert werr < 0.2, werr
