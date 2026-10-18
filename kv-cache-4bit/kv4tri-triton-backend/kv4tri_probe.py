"""kv4tri probe: what SGLang v0.5.20 actually does with `--kv-cache-dtype nvfp4 --attention-backend triton`.

Builds the two pool shapes the RTX models get (no model load):
  * Qwen 3.8 / 3.6 : HybridLinearKVPool -> full-attention MHATokenToKVPool WITH NVFP4KVCacheMethod
  * Gemma 4        : SWAKVPool -> MHATokenToKVPool WITHOUT a quant method (v0.5.20 passes none to the SWA branch)
then writes random K/V through the pool's own set_kv_buffer and feeds get_key_buffer/get_value_buffer to the stock
Triton decode kernel, comparing against a bf16 reference. Prints one JSON line per case.

Run on an RTX box:  cd /root/work && python3 operations/kernels/kv4tri_probe.py
"""
from __future__ import annotations

import json
import traceback
import types

import flashinfer  # noqa: F401  (engine ptxas; see CONTEXT lessons)
import torch


def _ref_decode(q, k, v, sm_scale):
    # q [B, Hq, D], k/v [B, T, Hkv, D] -> [B, Hq, D]
    g = q.shape[1] // k.shape[2]
    k = k.repeat_interleave(g, dim=2).float()
    v = v.repeat_interleave(g, dim=2).float()
    s = torch.einsum("bhd,bthd->bht", q.float(), k) * sm_scale
    p = s.softmax(-1)
    return torch.einsum("bht,bthd->bhd", p, v)


def _run_decode(pool, layer_id, q, locs_per_req, sm_scale):
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

    B, Hq, D = q.shape
    kb = pool.get_key_buffer(layer_id)
    vb = pool.get_value_buffer(layer_id)
    T = locs_per_req.shape[1]
    kv_indptr = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
    kv_indices = locs_per_req.reshape(-1).to(torch.int64)
    splits = 4
    o = torch.empty_like(q)
    att = torch.empty((B, Hq, splits, D), dtype=torch.float32, device=q.device)
    lse = torch.empty((B, Hq, splits), dtype=torch.float32, device=q.device)
    nsp = torch.full((B,), splits, dtype=torch.int32, device=q.device)
    decode_attention_fwd(q, kb, vb, o, kv_indptr, kv_indices, att, lse, nsp, splits, sm_scale, 1.0, 1.0)
    return o, kb


def case(name, build):
    torch.manual_seed(0)
    out = {"case": name}
    try:
        pool, layer_id, Hkv, D = build()
        out["pool"] = type(pool).__name__
        inner = getattr(pool, "full_kv_pool", None)
        out["quant_method"] = type(getattr(inner or pool, "quant_method", None)).__name__
        B, T, Hq = 2, 200, Hkv * 6
        dev = "cuda"
        k = torch.randn(B, T, Hkv, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, T, Hkv, D, device=dev, dtype=torch.bfloat16)
        q = torch.randn(B, Hq, D, device=dev, dtype=torch.bfloat16)
        locs = torch.arange(1, 1 + B * T, device=dev).view(B, T)
        layer = types.SimpleNamespace(layer_id=layer_id, k_scale=None, v_scale=None)
        pool.set_kv_buffer(layer, locs.reshape(-1), k.reshape(-1, Hkv, D), v.reshape(-1, Hkv, D))
        o, kb = _run_decode(pool, layer_id, q, locs, D ** -0.5)
        out["key_buffer"] = f"{tuple(kb.shape)} {kb.dtype}"
        ref = _ref_decode(q, k, v, D ** -0.5)
        err = (o.float() - ref).abs()
        out["max_abs_err"] = float(err.max())
        out["rel_err"] = float(err.norm() / ref.norm())
        out["verdict"] = "ok" if out["rel_err"] < 0.15 else "WRONG"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["tb"] = traceback.format_exc().strip().splitlines()[-3:]
        out["verdict"] = "CRASH"
    print(json.dumps(out), flush=True)


def build_hybrid_linear_nvfp4():
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import get_kv_cache_quant_method
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    qm = get_kv_cache_quant_method("nvfp4", num_layers=1, device="cuda")
    pool = MHATokenToKVPool(size=1024, page_size=1, dtype=torch.float4_e2m1fn_x2, head_num=4, head_dim=256,
                            layer_num=1, device="cuda", enable_memory_saver=False, quant_method=qm)
    return pool, 0, 4, 256


def build_mha_fp8():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = MHATokenToKVPool(size=1024, page_size=1, dtype=torch.float8_e4m3fn, head_num=4, head_dim=256,
                            layer_num=1, device="cuda", enable_memory_saver=False)
    return pool, 0, 4, 256


def build_swa_fp4_noquant():
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    pool = SWAKVPool(size=1024, size_swa=512, page_size=1, dtype=torch.float4_e2m1fn_x2, head_num=8, head_dim=256,
                     swa_attention_layer_ids=[0], full_attention_layer_ids=[1], device="cuda",
                     enable_kv_cache_copy=False, enable_memory_saver=False)
    return pool, 1, 8, 256


if __name__ == "__main__":
    case("mha_fp8_control", build_mha_fp8)
    case("qwen_hybrid_full_pool_nvfp4", build_hybrid_linear_nvfp4)
    case("gemma_swa_pool_fp4_dtype", build_swa_fp4_noquant)
