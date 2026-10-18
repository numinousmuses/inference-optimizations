"""GLM-5.3-Flash sparse-MLA prefill microbench at the served shapes (amdflash, Hopper kernel program 2026-10-05).

Profile rows attacked (F4 prefill TP0, ms per 4,096-token step; 12 MLA layers, 2 row slices each), 37.8 ms in all:
- flash_fwd_splitkv_mla_fp8_sparse_kernel 32.3 (24 calls);
- get_mla_metadata_kernel 3.39 (26 calls);
- the slice-output cat 1.59;
- flash_fwd_mla_combine 0.48.
Today the prefill role runs FlashMLA's sparse DECODE kernel per query row, because the KV cache is FP8
(--dsa-prefill-backend flashmla_kv). hopper_fp8kv_slice cuts the rows into 2,048-row calls, each with its own
get_mla_metadata.

Variants on identical inputs (q [s_q, 64, 512] bf16; FP8 KV in the 656-byte FlashMLA layout over s_kv tokens; top-k rows
2,112 wide, padded with -1):
  D   stock-hooked: per 2,048-row slice get_mla_metadata + flash_mla_with_kvcache, then torch.cat
  D2  D with the scheduler metadata computed once per slice shape (it depends only on rows / topk / heads)
  P   dequantize_k_cache_paged over the request's KV + flash_mla_sparse_fwd (the sparse PREFILL kernel, bf16)
      (= --dsa-prefill-backend flashmla_sparse with an FP8 pool, dsa_backend.py:2086-2104)
Accuracy: rel err of D2 and P vs D.

    python3 flash_sparse_mla_bench.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback

import torch

H, DQK, DV, TOPK_W, SLICE = 64, 512, 512, 2112, 2048


def timed(fn, iters=10, warmup=3):
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        out = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, out


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-12))


def quant_528(kv_bf16: torch.Tensor) -> torch.Tensor:
    """FlashMLA's no-RoPE FP8 KV row ("V3.2-no-RoPE", 528 bytes): 512 e4m3 values + 4 fp32 scales (one per 128)."""
    n = kv_bf16.shape[0]
    x = kv_bf16.float().view(n, 4, 128)
    scale = (x.abs().amax(-1).clamp(min=1e-8) / 448.0)                     # [n, 4]
    q8 = (x / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(n, 512)
    out = torch.empty(n, 528, dtype=torch.uint8, device=kv_bf16.device)
    out[:, :512] = q8.view(torch.uint8)
    out[:, 512:] = scale.contiguous().view(torch.uint8).view(n, 16)
    return out.view(n, 1, 528)


def dequant_528(kv8: torch.Tensor) -> torch.Tensor:
    """Inverse of quant_528 -> [n, 1, 512] bf16 (what a no-RoPE dequantize_k_cache_paged would return)."""
    n = kv8.shape[0]
    raw = kv8.view(torch.uint8).view(n, 528)
    vals = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float().view(n, 4, 128)
    scale = raw[:, 512:].contiguous().view(torch.float32).view(n, 4, 1)
    return (vals * scale).view(n, 1, 512).to(torch.bfloat16)


def build(s_q: int, s_kv: int, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv_bf16 = (torch.randn(s_kv, 512, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    kv8 = quant_528(kv_bf16)                                       # the served Flash pool row (528 B, no rope)
    q = (torch.randn(s_q, H, DQK, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    idx = torch.randint(0, s_kv, (s_q, 1, TOPK_W), device="cuda", generator=g, dtype=torch.int32)
    idx[..., 2051:] = -1                                           # index_topk + kpool - 1 = 2051 live, padded
    return q, kv8, idx


def run_d(q, kv_fp8, idx, cached: dict | None):
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
    s_q = q.shape[0]
    kv = kv_fp8.view(torch.float8_e4m3fn)
    n_blocks = math.ceil(kv.shape[0] / 64)
    pad = n_blocks * 64 - kv.shape[0]
    if pad:
        kv = torch.cat([kv, kv.new_zeros(pad, *kv.shape[1:])], 0)
    k_cache = kv.view(n_blocks, 64, 1, kv.shape[-1])
    outs = []
    for s0 in range(0, s_q, SLICE):
        s1 = min(s_q, s0 + SLICE)
        ones = torch.ones(s1 - s0, dtype=torch.int32, device=q.device)
        key = (s1 - s0,)
        if cached is not None and key in cached:
            meta, splits = cached[key]
        else:
            meta, splits = get_mla_metadata(cache_seqlens=ones, num_q_tokens_per_head_k=H, num_heads_k=1,
                                            num_heads_q=H, is_fp8_kvcache=True, topk=TOPK_W)
            if cached is not None:
                cached[key] = (meta, splits)
        o, _ = flash_mla_with_kvcache(q=q[s0:s1].unsqueeze(1), k_cache=k_cache, cache_seqlens=ones, head_dim_v=DV,
                                      tile_scheduler_metadata=meta, num_splits=splits, softmax_scale=DQK ** -0.5,
                                      indices=idx[s0:s1].contiguous(),
                                      block_table=torch.empty((s1 - s0, 0), dtype=torch.int32, device=q.device),
                                      is_fp8_kvcache=True)
        outs.append(o)
    return torch.cat(outs, 0).squeeze(1)


def run_p(q, kv_fp8, idx):
    """Dequantize the request's no-RoPE FP8 rows, then FlashMLA's sparse PREFILL kernel (d_qk 512, top-k padded to a
    multiple of 128: phase1.cuh asserts topk % (2*B_TOPK) == 0)."""
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    kv = dequant_528(kv_fp8)                                        # [s_kv, 1, 512] bf16
    w = idx.shape[-1]
    w2 = (w + 127) // 128 * 128
    idx2 = torch.nn.functional.pad(idx, (0, w2 - w), value=-1) if w2 != w else idx
    out = flash_mla_sparse_fwd(q, kv, idx2, DQK ** -0.5, DV)
    return out[0] if isinstance(out, (tuple, list)) else out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cases": []}
    for s_q, s_kv in ((4096, 32768), (8192, 32768)):
        r = {"s_q": s_q, "s_kv": s_kv}
        try:
            q, kv_fp8, idx = build(s_q, s_kv)
            r["kv_dtype"] = str(kv_fp8.dtype)
            r["kv_shape"] = list(kv_fp8.shape)
            ms_d, o_d = timed(lambda: run_d(q, kv_fp8, idx, None))
            cache = {}
            ms_d2, o_d2 = timed(lambda: run_d(q, kv_fp8, idx, cache))
            r.update({"D_ms_per_layer": round(ms_d, 4), "D2_ms_per_layer": round(ms_d2, 4),
                      "D2_saving_ms_per_step_x12": round((ms_d - ms_d2) * 12, 2), "D2_rel_err": rel(o_d2, o_d)})
        except Exception as e:  # noqa: BLE001
            r["D_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            r["D_tb"] = traceback.format_exc()[-1500:]
        try:
            ms_p, o_p = timed(lambda: run_p(q, kv_fp8, idx))
            r["P_ms_per_layer"] = round(ms_p, 4)
            if "D_ms_per_layer" in r:
                r["P_vs_D"] = round(ms_p / r["D_ms_per_layer"], 3)
                r["P_saving_ms_per_step_x12"] = round((r["D_ms_per_layer"] - ms_p) * 12, 2)
                r["P_rel_err"] = rel(o_p[..., :DV], o_d)
            # reference: bf16 dense-gather attention for a few rows (both D and P against the same exact math)
            rows = torch.arange(0, s_q, max(1, s_q // 16), device=q.device)
            kvb = dequant_528(kv_fp8)[:, 0].float()
            ref = []
            for i in rows.tolist():
                ii = idx[i, 0]
                ii = ii[ii >= 0].long()
                kk = kvb[ii]                                   # [topk, 512]
                sc = (q[i].float() @ kk.T) * (DQK ** -0.5)     # [H, topk]
                ref.append(torch.softmax(sc, -1) @ kk)         # V = K (absorbed MLA: v is the nope latent)
            ref = torch.stack(ref)
            r["P_rel_err_vs_ref"] = rel(o_p[rows][..., :DV], ref)
            if "D_ms_per_layer" in r:
                r["D_rel_err_vs_ref"] = rel(o_d[rows], ref)
        except Exception as e:  # noqa: BLE001
            r["P_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            r["P_tb"] = traceback.format_exc()[-1500:]
        res["cases"].append(r)
        print(json.dumps({k: v for k, v in r.items() if not k.endswith("_tb")})[:800], flush=True)
    json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
