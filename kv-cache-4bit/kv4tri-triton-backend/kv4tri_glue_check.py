"""kv4tri end-to-end glue check, run inside a PATCHED SGLang (operations/patches/test_kv4tri_hooks.py builds a patched
copy of the installed package and runs this with it first on PYTHONPATH). Uses the real pool classes the RTX models
get and the real TritonAttnBackend.forward_decode / forward_extend (called on a minimal backend object), so the hook
text, the glue module and the kernels are exercised together, with the production write-then-read sequence (no host
synchronization in between). Prints one JSON line per check and exits non-zero on any failure.

Checks:
  hybrid  : HybridLinearKVPool (Qwen full-attention pool) with NVFP4 -> get_v_head_dim is the real head dim; decode +
            extend through the patched backend methods match the float reference from SGLang's own dequantizer.
  swa     : SWAKVPool (Gemma 4: sliding 8 KV heads x 256, window 1023; global 2 KV heads x 512) built with the
            numinous_quant_methods kwarg -> both sub-pools NVFP4 without dequant workspace, get_v_head_dim 512,
            decode on a sliding layer (window indices) and a global layer, extend on both.
  disabled: NUMINOUS_KV4TRI=0 leaves the backend on the stock path (_numinous_kv4tri False).
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import flashinfer  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv4tri_nvfp4_attention as ref  # noqa: E402  (references only)

DEV = "cuda"
FAIL = []


def report(name, ok, **kw):
    print(json.dumps({"check": name, "ok": bool(ok), **kw}), flush=True)
    if not ok:
        FAIL.append(name)


def _dequant(sub, local):
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    kp, vp, ks, vs = sub.get_raw_kv_buffer(local)
    g = torch.ones(1, device=DEV)
    return (NVFP4KVQuantizeUtil.dequantize(kp, ks, g).float(), NVFP4KVQuantizeUtil.dequantize(vp, vs, g).float())


def _layer(lid, hq, hkv, d, window=-1):
    from sglang.srt.layers.radix_attention import AttentionType

    return types.SimpleNamespace(layer_id=lid, tp_q_head_num=hq, tp_k_head_num=hkv, qk_head_dim=d, v_head_dim=d,
                                 k_scale=None, v_scale=None, k_scale_float=None, v_scale_float=None,
                                 scaling=d ** -0.5, sliding_window_size=window, logit_capping_method="tanh",
                                 logit_cap=0.0, xai_temperature_len=-1, is_cross_attention=False,
                                 attn_type=AttentionType.DECODER)


def _backend(pool, md, swa_v=None):
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    b = types.SimpleNamespace(token_to_kv_pool=pool, forward_metadata=md, dcp_size=1, use_mla=False, page_size=1,
                              enable_deterministic=False, max_kv_splits=8, swa_v_head_dim=swa_v,
                              allow_bidirectional_attention_in_extend=False)
    b._set_kv_buffer = types.MethodType(TritonAttnBackend._set_kv_buffer, b)
    b.forward_decode = types.MethodType(TritonAttnBackend.forward_decode, b)
    b.forward_extend = types.MethodType(TritonAttnBackend.forward_extend, b)
    return b


def _md(**kw):
    base = dict(attn_logits=None, attn_lse=None, max_extend_len=0, num_kv_splits=None, kv_indptr=None,
                kv_indices=None, qo_indptr=None, custom_mask=None, mask_indptr=None, window_kv_indptr=None,
                window_kv_indices=None, window_num_kv_splits=None, window_kv_offsets=None, swa_attn_logits=None,
                swa_out_cache_loc=None, out_cache_loc_full_physical=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _fb(out_cache_loc, prefix_lens=None):
    return types.SimpleNamespace(out_cache_loc=out_cache_loc, mha_one_shot=False, positions=None,
                                 extend_prefix_lens_cpu=prefix_lens)


def run_decode(b, layer, hq, hkv, d, lens, cache_slots, window=-1, swa_translate=None, sub_local=None):
    """Prefill-free decode: context K/V already in the pool at cache_slots (per request, in order); the decode token
    of each request is written by forward_decode itself at a fresh slot."""
    B = len(lens)
    q = torch.randn(B, hq * d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, d, device=DEV, dtype=torch.bfloat16)
    new_slots = torch.arange(B, device=DEV) + int(max(max(s) for s in cache_slots)) + 1
    full_lists = [torch.cat([s, new_slots[i:i + 1]]) for i, s in enumerate(cache_slots)]
    if window > 0:
        full_lists = [s[-window:] for s in full_lists]
    tr = (lambda x: x) if swa_translate is None else swa_translate
    kv_lists = [tr(s) for s in full_lists]
    kv_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor([len(s) for s in kv_lists]), 0)),
                             dtype=torch.int32, device=DEV)
    kv_indices = torch.cat(kv_lists).to(torch.int64)
    md = b.forward_metadata
    md.attn_logits = torch.empty(B, hq, 8, d, device=DEV)
    md.attn_lse = torch.empty(B, hq, 8, device=DEV)
    md.num_kv_splits = torch.full((B,), 8, dtype=torch.int32, device=DEV)
    if window > 0:
        md.window_kv_indptr, md.window_kv_indices = kv_indptr, kv_indices
        md.swa_out_cache_loc = tr(new_slots)
    else:
        md.kv_indptr, md.kv_indices = kv_indptr, kv_indices
    o = b.forward_decode(q, k, v, layer, _fb(new_slots))
    torch.cuda.synchronize()
    kd, vd = _dequant(*sub_local)
    r = ref.decode_reference(q.view(B, hq, d), kd, vd, kv_indptr, kv_indices, d ** -0.5)
    return float((o.view(B, hq, d).float() - r).abs().max())


def run_extend(b, layer, hq, hkv, d, prefixes, news, slot0, window=-1, swa_translate=None, sub_local=None,
               cpu_lens=True):
    """Prefix K/V are written by a first extend (no prefix), then a second extend attends over them. cpu_lens=True
    gives the batch its host prefix lengths (long chunks take the gather-dequant + GQA FP8 route), False forces the
    native FP4 extend kernel."""
    tr = (lambda x: x) if swa_translate is None else swa_translate
    md = b.forward_metadata
    errs = []
    starts = []
    s = slot0
    for P, T in zip(prefixes, news):
        starts.append(s)
        s += P + T
    for phase in ("prefix", "extend"):
        lens_q = prefixes if phase == "prefix" else news
        pre = [0] * len(prefixes) if phase == "prefix" else prefixes
        T = sum(lens_q)
        q = torch.randn(T, hq * d, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(T, hkv * d, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(T, hkv * d, device=DEV, dtype=torch.bfloat16)
        loc = torch.cat([torch.arange(st + (0 if phase == "prefix" else P), st + (0 if phase == "prefix" else P) + n,
                                      device=DEV) for st, P, n in zip(starts, prefixes, lens_q)])
        pref_lists = [torch.arange(st, st + p, device=DEV) for st, p in zip(starts, pre)]
        if window > 0:
            pref_lists = [x[-window:] for x in pref_lists]
        kvl = [tr(x) for x in pref_lists]
        kv_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor([len(x) for x in kvl]), 0)),
                                 dtype=torch.int32, device=DEV)
        kv_indices = torch.cat(kvl).to(torch.int64) if sum(len(x) for x in kvl) else torch.zeros(1, dtype=torch.int64,
                                                                                                device=DEV)
        md.qo_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens_q), 0)), dtype=torch.int32, device=DEV)
        md.max_extend_len = max(lens_q)
        if window > 0:
            md.window_kv_indptr, md.window_kv_indices = kv_indptr, kv_indices
            md.swa_out_cache_loc = tr(loc)
        else:
            md.kv_indptr, md.kv_indices = kv_indptr, kv_indices
        o = b.forward_extend(q.view(T, hq, d), k.view(T, hkv, d), v.view(T, hkv, d), layer,
                             _fb(loc, list(pre) if cpu_lens else None))
        torch.cuda.synchronize()
        kd, vd = _dequant(*sub_local)
        r = ref.extend_reference(q.view(T, hq, d), k.view(T, hkv, d), v.view(T, hkv, d), kd, vd, md.qo_indptr,
                                 kv_indptr, kv_indices, d ** -0.5, sliding_window_size=window)
        errs.append(float((o.view(T, hq, d).float() - r).abs().max()))
    return errs


def check_hybrid():
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import get_kv_cache_quant_method
    from sglang.srt.layers.attention import numinous_kv4tri as nk4
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    qm = get_kv_cache_quant_method("nvfp4", num_layers=2, device=DEV)
    qm.__class__ = nk4._triton_cls()
    hq, hkv, d = 24, 4, 256
    pool = HybridLinearKVPool(size=20000, dtype=torch.float4_e2m1fn_x2, page_size=1, head_num=hkv, head_dim=d,
                              full_attention_layer_ids=[3, 7], device=DEV, mamba_pool=None, quant_method=qm)
    report("hybrid_v_head_dim", pool.get_v_head_dim() == d, got=pool.get_v_head_dim())
    report("hybrid_no_workspace", pool.full_kv_pool.dq_k_buffer is None)
    layer = _layer(7, hq, hkv, d)
    b = _backend(pool, _md())
    errs = run_extend(b, layer, hq, hkv, d, prefixes=[900, 300], news=[64, 5], slot0=1, sub_local=(pool, 7))
    # the FP8 workspace path rounds the dequantized prefix and the queries to e4m3 (as the FP8-KV recipe does)
    report("hybrid_extend_dq_fp8", max(errs) < 0.08, err=errs)
    layer_n = _layer(3, hq, hkv, d)
    errs = run_extend(b, layer_n, hq, hkv, d, prefixes=[700, 40], news=[70, 5], slot0=1, sub_local=(pool, 3),
                      cpu_lens=False)
    report("hybrid_extend_native", max(errs) < 0.03, err=errs)
    errs = run_extend(b, layer_n, hq, hkv, d, prefixes=[1200, 333], news=[4, 3], slot0=3000, sub_local=(pool, 3))
    report("hybrid_extend_verify_split", max(errs) < 0.03, err=errs)
    st = [torch.arange(1, 965, device=DEV), torch.arange(965, 1270, device=DEV)]
    err = run_decode(b, layer, hq, hkv, d, [964, 305], st, sub_local=(pool, 7))
    report("hybrid_decode", err < 0.02, err=err)
    # production: decode (store + attention) captured in a CUDA graph with layer.k_scale None
    B = 2
    qg = torch.randn(B, hq * d, device=DEV, dtype=torch.bfloat16)
    kg = torch.randn(B, hkv, d, device=DEV, dtype=torch.bfloat16)
    vg = torch.randn(B, hkv, d, device=DEV, dtype=torch.bfloat16)
    slots_g = torch.tensor([5001, 5002], device=DEV)
    fb = _fb(slots_g)
    try:
        torch.cuda.synchronize()
        s_ = torch.cuda.Stream()
        with torch.cuda.stream(s_):
            b.forward_decode(qg, kg, vg, layer, fb)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            og = b.forward_decode(qg, kg, vg, layer, fb)
        g.replay()
        torch.cuda.synchronize()
        report("hybrid_decode_graph_capture", bool(torch.isfinite(og).all()))
    except Exception as e:  # noqa: BLE001
        report("hybrid_decode_graph_capture", False, error=f"{type(e).__name__}: {str(e)[:200]}")
    report("hybrid_state", b._numinous_kv4tri is True)


def check_swa():
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import get_kv_cache_quant_method
    from sglang.srt.layers.attention import numinous_kv4tri as nk4
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    qms = []
    for n in (1, 2):
        qm = get_kv_cache_quant_method("nvfp4", num_layers=n, device=DEV)
        qm.__class__ = nk4._triton_cls()
        qms.append(qm)
    pool = SWAKVPool(size=16000, size_swa=8000, page_size=1, dtype=torch.float4_e2m1fn_x2, head_num=2, head_dim=512,
                     swa_attention_layer_ids=[0, 1], full_attention_layer_ids=[2], device=DEV,
                     enable_kv_cache_copy=False, swa_head_num=8, swa_head_dim=256, swa_v_head_dim=256,
                     v_head_dim=512, numinous_quant_methods=tuple(qms))
    report("swa_both_nvfp4", pool.full_kv_pool.quant_method is qms[0] and pool.swa_kv_pool.quant_method is qms[1])
    report("swa_no_workspace", pool.full_kv_pool.dq_k_buffer is None and pool.swa_kv_pool.dq_k_buffer is None)
    report("swa_v_head_dim", pool.get_v_head_dim() == 512, got=pool.get_v_head_dim())
    report("swa_shapes", tuple(pool.swa_kv_pool.k_buffer[0].shape[1:]) == (8, 128)
           and tuple(pool.full_kv_pool.k_buffer[0].shape[1:]) == (2, 256))
    # static SWA pool: full slot ids map to swa slot ids through a table (identity-offset here)
    mapping = torch.arange(16001 + 1, device=DEV) % 7999 + 1
    swa_tr = lambda x: mapping[x]  # noqa: E731
    # global layer (full pool)
    layer_g = _layer(2, 16, 2, 512)
    b = _backend(pool, _md(), swa_v=256)
    errs = run_extend(b, layer_g, 16, 2, 512, prefixes=[700, 0], news=[70, 33], slot0=1, sub_local=(pool.full_kv_pool, 0))
    report("swa_global_extend_dq_fp8", max(errs) < 0.08, err=errs)
    st = [torch.arange(1, 771, device=DEV), torch.arange(771, 804, device=DEV)]
    b.forward_metadata.swa_attn_logits = None
    err = run_decode(b, layer_g, 16, 2, 512, [770, 33], st, sub_local=(pool.full_kv_pool, 0))
    report("swa_global_decode", err < 0.02, err=err)
    # sliding layer: window 1023, the pool addressed through the full->swa slot table
    layer_s = _layer(1, 16, 8, 256, window=1023)
    b2 = _backend(pool, _md(), swa_v=256)
    errs = run_extend(b2, layer_s, 16, 8, 256, prefixes=[1500, 200], news=[80, 7], slot0=1, window=1023,
                      swa_translate=swa_tr, sub_local=(pool.swa_kv_pool, 1))
    report("swa_sliding_extend_dq_bf16", max(errs) < 0.03, err=errs)
    st = [torch.arange(1, 1581, device=DEV), torch.arange(1581, 1788, device=DEV)]
    b2.forward_metadata.swa_attn_logits = torch.empty(2, 16, 8, 256, device=DEV)
    err = run_decode(b2, layer_s, 16, 8, 256, [1580, 207], st, window=1023, swa_translate=swa_tr,
                     sub_local=(pool.swa_kv_pool, 1))
    report("swa_sliding_decode", err < 0.02, err=err)


def check_disabled():
    os.environ["NUMINOUS_KV4TRI"] = "0"
    from sglang.srt.layers.attention import numinous_kv4tri as nk4

    b = types.SimpleNamespace(token_to_kv_pool=types.SimpleNamespace(quant_method=types.SimpleNamespace(name="nvfp4")))
    report("disabled_stock_path", nk4.forward_decode(b, None, None, None, None, None) is None
           and b._numinous_kv4tri is False)
    os.environ.pop("NUMINOUS_KV4TRI")


if __name__ == "__main__":
    import sglang

    pkg = os.path.dirname(sglang.__file__)
    report("patched_sglang", os.path.isfile(os.path.join(pkg, "srt/layers/attention/numinous_kv4tri.py")), path=pkg)
    torch.manual_seed(0)
    for fn in (check_hybrid, check_swa, check_disabled):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback

            report(fn.__name__, False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc().splitlines()[-4:])
    sys.exit(1 if FAIL else 0)
