"""Kernel-level check of FP8 KV for GLM-5.3-Flash DSA on SM90 (rtxsys, 2026-10-04; agents/findings/hopper_fp8kv.md).

Runs on one H100 / H200 with the v0.5.20 image (sgl_kernel FlashMLA + SGLang's KV quantizer). No model, no server.
It exercises exactly the path operations/patches/hopper_fp8kv_hooks.py opens:
  * KV rows written by SGLang's own quantizer (quant_k_cache.quantize_k_cache_separate with an empty rope): 528 B per
    token = 512 e4m3 + four fp32 per-128 scales (FlashMLA's V32_NO_ROPE format);
  * kpool-shaped index rows: up to 2048 selected history slots, then up to 3 tail slots right after the history
    (kpool_fp8_index.append_kpool_tail_to_topk), -1 elsewhere; padded with -1 to 2112 as the hook does;
  * FlashMLA's SM90 sparse decode (flash_mla_with_kvcache, legacy tensor-metadata API, as dsa_backend._forward_flashmla_kv
    calls it) with tile-scheduler metadata built for topk 2112 (as the hook builds it);
  * heads 64 (TP1 / DP attention) and 32 zero-padded to 64 (TP2, dsa_backend pads to flashmla_kv_num_q_heads).
References in fp32 torch: (a) attention over the dequantized FP8 rows: kernel correctness; (b) attention over the
original bf16 latents: FP8 quantization error. It also records whether the unpadded 2051-wide call is refused.

    python3 hopper_fp8kv_check.py [--rows 64] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

D_LATENT, TOPK, KPOOL, PAGE = 512, 2048, 4, 64
WIDTH = (TOPK + KPOOL - 1 + 63) // 64 * 64        # 2112
SCALE = 256 ** -0.5                                 # GLM-5.3-Flash qk_head_dim 256 (absorbed NoPE MLA keeps it)


def build_rows(n_rows: int, n_tokens: int, gen: torch.Generator, device) -> tuple[torch.Tensor, list[int]]:
    """kpool-shaped index rows (int32, -1 padded to 2112) and each row's valid count. Context lengths span short
    (history < 2048) and long rows; tails of 0..3 tokens."""
    rows = torch.full((n_rows, WIDTH), -1, dtype=torch.int32)
    valid = []
    for r in range(n_rows):
        seq = int(torch.randint(64, min(n_tokens, 60_000), (1,), generator=gen)) if r % 4 else int(
            torch.randint(64, 2048, (1,), generator=gen))
        tail = seq % KPOOL
        pool_len = seq // KPOOL
        hist = min(pool_len * KPOOL, TOPK)
        # history: hist distinct slots (a random token subset, as the pooled top-k expands to whole pools)
        perm = torch.randperm(n_tokens, generator=gen)[: hist + tail]
        rows[r, :hist] = perm[:hist].to(torch.int32)
        rows[r, hist: hist + tail] = perm[hist: hist + tail].to(torch.int32)
        valid.append(hist + tail)
    return rows.to(device), valid


def ref_attention(q: torch.Tensor, kv: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """fp32 sparse attention, MQA over one latent: out[b, h] = softmax(q k^T * SCALE) k, k = kv[rows[b]] (masked -1)."""
    b, h, d = q.shape
    out = torch.zeros((b, h, d), dtype=torch.float32, device=q.device)
    for i in range(b):
        idx = rows[i]
        idx = idx[idx >= 0].long()
        k = kv[idx].float()                                   # (n, d)
        s = (q[i].float() @ k.T) * SCALE                      # (h, n)
        p = torch.softmax(s, dim=-1)
        out[i] = p @ k
    return out


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def run(n_rows: int, heads: int, seed: int = 0) -> dict:
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate

    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(seed)
    n_pages = 1024
    n_tokens = n_pages * PAGE
    # latents with a per-channel spread like real MLA latents (heavy channels stress the per-128 scales)
    chan = torch.exp(torch.randn(D_LATENT, generator=gen) * 0.75)
    k_bf16 = (torch.randn(n_tokens, D_LATENT, generator=gen) * chan).to(torch.bfloat16).to(dev)
    nope_u8, rope_u8 = quantize_k_cache_separate(k_bf16, None)
    assert nope_u8.shape[-1] == 528 and rope_u8.shape[-1] == 0, (nope_u8.shape, rope_u8.shape)
    pool = nope_u8.view(n_pages, PAGE, 1, 528).contiguous()
    # dequantize the rows exactly as the kernel reads them: 512 e4m3 values, then four fp32 scales (one per 128 dims)
    rows_u8 = pool.view(n_tokens, 528)
    nope = rows_u8[:, :D_LATENT].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows_u8[:, D_LATENT:].contiguous().view(torch.float32)          # (n_tokens, 4)
    k_deq = (nope.view(n_tokens, 4, 128) * scales[:, :, None]).view(n_tokens, D_LATENT)

    rows, valid = build_rows(n_rows, n_tokens, gen, dev)
    q = (torch.randn(n_rows, heads, D_LATENT, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    # SGLang pads q heads to 64 for FlashMLA (dsa_backend.flashmla_kv_num_q_heads)
    hq = 64 if heads <= 64 else 128
    q_in = torch.zeros((n_rows, 1, hq, D_LATENT), dtype=torch.bfloat16, device=dev)
    q_in[:, 0, :heads] = q

    seqlens = torch.ones(n_rows, dtype=torch.int32, device=dev)
    meta, splits = get_mla_metadata(cache_seqlens=seqlens, num_q_tokens_per_head_k=hq, num_heads_k=1,
                                    num_heads_q=hq, is_fp8_kvcache=True, topk=WIDTH)
    empty_bt = torch.empty((n_rows, 0), dtype=torch.int32, device=dev)
    out, _ = flash_mla_with_kvcache(q=q_in, k_cache=pool, block_table=empty_bt, cache_seqlens=seqlens,
                                    head_dim_v=D_LATENT, tile_scheduler_metadata=meta, num_splits=splits,
                                    softmax_scale=SCALE, indices=rows.unsqueeze(1), is_fp8_kvcache=True)
    torch.cuda.synchronize()
    o = out[:, 0, :heads].float()

    ref_fp8 = ref_attention(q, k_deq, rows)
    ref_bf16 = ref_attention(q, k_bf16, rows)
    res = {"rows": n_rows, "heads": heads, "q_heads_padded": hq, "width": WIDTH,
           "valid_min": min(valid), "valid_max": max(valid),
           "rel_err_vs_fp8_dequant_ref": rel(o, ref_fp8),
           "max_abs_err_vs_fp8_dequant_ref": float((o - ref_fp8).abs().max()),
           "rel_err_vs_bf16_ref (fp8 quantization)": rel(o, ref_bf16),
           "rel_err_fp8ref_vs_bf16ref": rel(ref_fp8, ref_bf16),
           "nan": bool(torch.isnan(o).any())}

    # timing (decode kernel only, no graph): FP8 rows at 2112 wide
    for _ in range(3):
        flash_mla_with_kvcache(q=q_in, k_cache=pool, block_table=empty_bt, cache_seqlens=seqlens, head_dim_v=D_LATENT,
                               tile_scheduler_metadata=meta, num_splits=splits, softmax_scale=SCALE,
                               indices=rows.unsqueeze(1), is_fp8_kvcache=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    reps = 20
    for _ in range(reps):
        flash_mla_with_kvcache(q=q_in, k_cache=pool, block_table=empty_bt, cache_seqlens=seqlens, head_dim_v=D_LATENT,
                               tile_scheduler_metadata=meta, num_splits=splits, softmax_scale=SCALE,
                               indices=rows.unsqueeze(1), is_fp8_kvcache=True)
    torch.cuda.synchronize()
    res["us_per_call"] = (time.perf_counter() - t) / reps * 1e6
    res["kv_bytes_per_token"] = {"fp8_v32_no_rope": 528, "bf16": 2 * D_LATENT}
    return res


def probe_unpadded(n_rows: int = 8) -> str:
    """The stock path: 2051-wide rows straight into FlashMLA (what the stock assert prevents). Run in its own process:
    a host-side abort here must not lose the main results."""
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate

    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(1)
    n_pages = 64
    k = torch.randn(n_pages * PAGE, D_LATENT, generator=gen).to(torch.bfloat16).to(dev)
    pool = quantize_k_cache_separate(k, None)[0].view(n_pages, PAGE, 1, 528).contiguous()
    rows, _ = build_rows(n_rows, n_pages * PAGE, gen, dev)
    narrow = rows[:, : TOPK + KPOOL - 1].contiguous()
    q_in = torch.randn(n_rows, 1, 64, D_LATENT, generator=gen).to(torch.bfloat16).to(dev)
    seqlens = torch.ones(n_rows, dtype=torch.int32, device=dev)
    try:
        m2, s2 = get_mla_metadata(cache_seqlens=seqlens, num_q_tokens_per_head_k=64, num_heads_k=1,
                                  num_heads_q=64, is_fp8_kvcache=True, topk=TOPK + KPOOL - 1)
        flash_mla_with_kvcache(q=q_in, k_cache=pool, block_table=torch.empty((n_rows, 0), dtype=torch.int32, device=dev),
                               cache_seqlens=seqlens, head_dim_v=D_LATENT, tile_scheduler_metadata=m2, num_splits=s2,
                               softmax_scale=SCALE, indices=narrow.unsqueeze(1), is_fp8_kvcache=True)
        torch.cuda.synchronize()
        return "accepted"
    except Exception as e:  # noqa: BLE001
        return f"refused: {type(e).__name__}: {str(e)[:300]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--json", default="")
    ap.add_argument("--probe-unpadded", action="store_true")
    a = ap.parse_args()
    if a.probe_unpadded:
        print(json.dumps({"unpadded_2051": probe_unpadded()}))
        return
    out = {"device": torch.cuda.get_device_name(), "cap": list(torch.cuda.get_device_capability()), "cases": []}
    for heads in (64, 32):
        out["cases"].append(run(a.rows, heads))
    # pass/fail: the kernel matches its own dequantized reference to bf16-accumulation noise
    out["pass"] = all(c["rel_err_vs_fp8_dequant_ref"] < 2e-2 and not c["nan"] for c in out["cases"])
    s = json.dumps(out, indent=1)
    print(s)
    if a.json:
        open(a.json, "w").write(s)


if __name__ == "__main__":
    main()
