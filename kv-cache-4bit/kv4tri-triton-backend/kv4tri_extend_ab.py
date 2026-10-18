"""kv4tri extend A/B on one GPU (CUDA-graph timed): one long prefill chunk over a long cached prefix.
  fp4_native     : kv4tri_nvfp4_attention.nvfp4_extend_fwd straight from the FP4 pool
  dq_bf16_gqa    : gather+dequantize the request prefix to a bf16 workspace (kv4tri gather kernel), GQA overlay bf16
  dq_fp8_gqa     : same into an fp8 workspace, GQA overlay fp8 path
  fp8_gqa        : GQA overlay on an FP8 pool (the FP8-KV Triton recipe with the overlay)
  fp8_stock      : stock SGLang extend kernel on an FP8 pool
    cd /root/work && python3 operations/kernels/kv4tri_extend_ab.py"""
import json
import sys
from pathlib import Path

import flashinfer  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extend_attention_gqa as eg  # noqa: E402
import kv4tri_bench as kb  # noqa: E402
import kv4tri_nvfp4_attention as k4  # noqa: E402

DEV = "cuda"
SHAPES = {"qwen38": (24, 4, 256), "qwen36": (16, 2, 256), "gemma_global": (16, 2, 512), "gemma_sliding": (16, 8, 256)}


def gqa_cfg(hq, kvh, d):
    if d >= 512:
        return dict(block_t=8, block_n=128, dk=64, dv=256, num_warps=8, num_stages=3)
    if hq // kvh >= 4:
        return dict(block_t=8, block_n=128, dk=64, dv=d, num_warps=8, num_stages=3)
    return dict(block_t=32, block_n=64, dk=64, dv=d, num_warps=8, num_stages=1)


def run(name, prefix, new, window=-1):
    hq, kvh, d = SHAPES[name]
    fp4, fp8 = kb._pools(prefix + 1, kvh, d)
    q = torch.randn(new, hq, d, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(new, kvh, d, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(new, kvh, d, device=DEV, dtype=torch.bfloat16)
    o = torch.empty_like(q)
    qo = torch.tensor([0, new], dtype=torch.int32, device=DEV)
    pe = min(prefix, window) if window > 0 else prefix
    kvp = torch.tensor([0, pe], dtype=torch.int32, device=DEV)
    idx = torch.arange(1 + prefix - pe, prefix + 1, device=DEV, dtype=torch.int64)
    ident = torch.arange(pe, device=DEV, dtype=torch.int64)
    cfg = gqa_cfg(hq, kvh, d)
    ws16 = (torch.empty(pe, kvh, d, device=DEV, dtype=torch.bfloat16),
            torch.empty(pe, kvh, d, device=DEV, dtype=torch.bfloat16))
    ws8 = (torch.empty(pe, kvh, d, device=DEV, dtype=torch.float8_e4m3fn),
           torch.empty(pe, kvh, d, device=DEV, dtype=torch.float8_e4m3fn))
    out = {"case": name, "prefix": prefix, "new": new, "window": window}
    t = {}
    t["fp4_native"] = lambda: k4.nvfp4_extend_fwd(q, ke, ve, o, *fp4[:2], *fp4[2:], qo, kvp, idx, new, d ** -0.5,
                                                  sliding_window_size=window)

    def dq(ws):
        k4.nvfp4_gather_dequant(fp4[0], fp4[2], idx, ws[0])
        k4.nvfp4_gather_dequant(fp4[1], fp4[3], idx, ws[1])

    t["dq_only_bf16"] = lambda: dq(ws16)
    t["dq_bf16_gqa"] = lambda: (dq(ws16), eg.extend_attention_gqa(q, ke, ve, o, ws16[0], ws16[1], qo, kvp, ident, new,
                                                                  d ** -0.5, sliding_window_size=window, fp8=False,
                                                                  **cfg))
    t["dq_fp8_gqa"] = lambda: (dq(ws8), eg.extend_attention_gqa(q, ke, ve, o, ws8[0], ws8[1], qo, kvp, ident, new,
                                                                d ** -0.5, sliding_window_size=window, **cfg))
    t["fp8_gqa"] = lambda: eg.extend_attention_gqa(q, ke, ve, o, fp8[0], fp8[1], qo, kvp, idx, new, d ** -0.5,
                                                   sliding_window_size=window, **cfg)
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

    t["fp8_stock"] = lambda: extend_attention_fwd(q, ke, ve, o, fp8[0], fp8[1], qo, kvp, idx, None, True, None, new,
                                                  1.0, 1.0, d ** -0.5, sliding_window_size=window)
    for k, fn in t.items():
        try:
            out[k + "_us"] = round(kb.graph_time_us(fn, reps=3, iters=3), 1)
        except Exception as e:  # noqa: BLE001
            out[k + "_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    print(json.dumps(out), flush=True)
    del fp4, fp8, ws16, ws8
    torch.cuda.empty_cache()


if __name__ == "__main__" and len(sys.argv) == 1:
    for name in ("qwen38", "qwen36", "gemma_global"):
        run(name, 32768, 4096)
    run("gemma_sliding", 32768, 4096, window=1023)
    run("qwen38", 4096, 4096)


def verify(name, B, prefix, qlen):
    """EAGLE target-verify shape: B requests x qlen draft tokens over long prefixes, chain custom mask."""
    hq, kvh, d = SHAPES[name]
    fp4, fp8 = kb._pools(B * prefix + 1, kvh, d)
    T = B * qlen
    q = torch.randn(T, hq, d, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(T, kvh, d, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(T, kvh, d, device=DEV, dtype=torch.bfloat16)
    o = torch.empty_like(q)
    qo = torch.arange(0, (B + 1) * qlen, qlen, dtype=torch.int32, device=DEV)
    kvp = torch.arange(0, (B + 1) * prefix, prefix, dtype=torch.int32, device=DEV)
    idx = torch.arange(1, B * prefix + 1, device=DEV, dtype=torch.int64)
    m = torch.ones(qlen, prefix + qlen, dtype=torch.bool, device=DEV)
    m[:, prefix:] = torch.tril(torch.ones(qlen, qlen, dtype=torch.bool, device=DEV))
    cm = m.flatten().repeat(B)
    mi = torch.arange(0, B * m.numel(), m.numel(), dtype=torch.int64, device=DEV)
    out = {"case": f"verify/{name}", "B": B, "prefix": prefix, "qlen": qlen}
    out["fp4_native_us"] = round(kb.graph_time_us(lambda: k4.nvfp4_extend_fwd(
        q, ke, ve, o, *fp4[:2], *fp4[2:], qo, kvp, idx, qlen, d ** -0.5, custom_mask=cm, mask_indptr=mi)), 1)
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

    out["fp8_stock_us"] = round(kb.graph_time_us(lambda: extend_attention_fwd(
        q, ke, ve, o, fp8[0], fp8[1], qo, kvp, idx, cm, True, mi, qlen, 1.0, 1.0, d ** -0.5)), 1)
    out["kv_bytes_fp4_MB"] = round(B * prefix * kvh * 2 * (d // 2 + d // 16) / 1e6, 1)
    print(json.dumps(out), flush=True)
    del fp4, fp8
    torch.cuda.empty_cache()


if __name__ == "__main__" and sys.argv[1:2] == ["verify"]:
    verify("qwen38", 16, 104000, 4)
    verify("qwen38", 1, 104000, 4)
    verify("qwen36", 16, 104000, 5)
    verify("qwen36", 1, 104000, 5)


def verify2(name, B, prefix, qlen):
    hq, kvh, d = SHAPES[name]
    fp4, _ = kb._pools(B * prefix + 1, kvh, d)
    T = B * qlen
    q = torch.randn(T, hq, d, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(T, kvh, d, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(T, kvh, d, device=DEV, dtype=torch.bfloat16)
    o = torch.empty_like(q)
    qo = torch.arange(0, (B + 1) * qlen, qlen, dtype=torch.int32, device=DEV)
    kvp = torch.arange(0, (B + 1) * prefix, prefix, dtype=torch.int32, device=DEV)
    idx = torch.arange(1, B * prefix + 1, device=DEV, dtype=torch.int64)
    m = torch.ones(qlen, prefix + qlen, dtype=torch.bool, device=DEV)
    m[:, prefix:] = torch.tril(torch.ones(qlen, qlen, dtype=torch.bool, device=DEV))
    cm = m.flatten().repeat(B)
    mi = torch.arange(0, B * m.numel(), m.numel(), dtype=torch.int64, device=DEV)
    out = {"case": f"verify_split/{name}", "B": B, "prefix": prefix, "qlen": qlen}
    for bn, nw in ((32, 4), (64, 4), (32, 8)):
        out[f"fp4_split_bn{bn}_w{nw}_us"] = round(kb.graph_time_us(lambda: k4.nvfp4_verify_fwd(
            q, ke, ve, o, *fp4[:2], *fp4[2:], qo, kvp, idx, qlen, d ** -0.5, custom_mask=cm, mask_indptr=mi,
            block_n=bn, num_warps=nw)), 1)
    kvb = B * prefix * kvh * 2 * (d // 2 + d // 16)
    out["best_GBps"] = round(kvb / min(v for k, v in out.items() if k.endswith("_us")) / 1e3, 1)
    print(json.dumps(out), flush=True)
    del fp4
    torch.cuda.empty_cache()


if __name__ == "__main__" and "verify2" in sys.argv:
    verify2("qwen38", 16, 104000, 4)
    verify2("qwen38", 1, 104000, 4)
    verify2("qwen36", 16, 104000, 5)
    verify2("qwen36", 1, 104000, 5)
    verify2("qwen38", 8, 32768, 4)


def verify_sweep(name, B, prefix, qlen):
    import itertools
    hq, kvh, d = SHAPES[name]
    fp4, _ = kb._pools(B * prefix + 1, kvh, d)
    T = B * qlen
    q = torch.randn(T, hq, d, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(T, kvh, d, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(T, kvh, d, device=DEV, dtype=torch.bfloat16)
    o = torch.empty_like(q)
    qo = torch.arange(0, (B + 1) * qlen, qlen, dtype=torch.int32, device=DEV)
    kvp = torch.arange(0, (B + 1) * prefix, prefix, dtype=torch.int32, device=DEV)
    idx = torch.arange(1, B * prefix + 1, device=DEV, dtype=torch.int64)
    auto = k4.verify_splits(B, kvh)
    res = []
    for sp, bn, ns in itertools.product(sorted({max(1, auto // 2), auto, min(64, auto * 2), min(128, auto * 4)}),
                                        (16, 32, 64), (1, 2, 3)):
        try:
            t = kb.graph_time_us(lambda: k4.nvfp4_verify_fwd(q, ke, ve, o, *fp4[:2], *fp4[2:], qo, kvp, idx, qlen,
                                                              d ** -0.5, num_splits=sp, block_n=bn, num_stages=ns),
                                 reps=10, iters=3)
        except Exception:  # noqa: BLE001
            t = float("inf")
        res.append((round(t, 1), dict(splits=sp, block_n=bn, num_stages=ns)))
    res.sort(key=lambda x: x[0])
    print(json.dumps({"verify_sweep": name, "B": B, "prefix": prefix, "qlen": qlen, "auto_splits": auto,
                      "best": res[:4]}), flush=True)
    del fp4
    torch.cuda.empty_cache()


if __name__ == "__main__" and "vsweep" in sys.argv:
    verify_sweep("qwen38", 16, 104000, 4)
    verify_sweep("qwen38", 8, 32768, 4)
    verify_sweep("qwen38", 1, 104000, 4)
    verify_sweep("qwen36", 16, 104000, 5)
    verify_sweep("qwen36", 1, 104000, 5)


def midq(name, B, prefix, qlen):
    """Incremental-prefill shape: B requests, qlen new tokens each over long prefixes (no custom mask)."""
    hq, kvh, d = SHAPES[name]
    fp4, _ = kb._pools(B * prefix + 1, kvh, d)
    T = B * qlen
    q = torch.randn(T, hq, d, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(T, kvh, d, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(T, kvh, d, device=DEV, dtype=torch.bfloat16)
    o = torch.empty_like(q)
    qo = torch.arange(0, (B + 1) * qlen, qlen, dtype=torch.int32, device=DEV)
    kvp = torch.arange(0, (B + 1) * prefix, prefix, dtype=torch.int32, device=DEV)
    idx = torch.arange(1, B * prefix + 1, device=DEV, dtype=torch.int64)
    ws = torch.empty(B * prefix, kvh, d, device=DEV, dtype=torch.float8_e4m3fn), \
        torch.empty(B * prefix, kvh, d, device=DEV, dtype=torch.float8_e4m3fn)
    ident = torch.arange(B * prefix, device=DEV)
    out = {"case": f"midq/{name}", "B": B, "prefix": prefix, "qlen": qlen}
    out["native_us"] = round(kb.graph_time_us(lambda: k4.nvfp4_extend_fwd(q, ke, ve, o, *fp4[:2], *fp4[2:], qo, kvp,
                                                                          idx, qlen, d ** -0.5), reps=3, iters=2), 1)

    def dq():
        k4.nvfp4_gather_dequant(fp4[0], fp4[2], idx, ws[0])
        k4.nvfp4_gather_dequant(fp4[1], fp4[3], idx, ws[1])
        eg.extend_attention_gqa(q, ke, ve, o, ws[0], ws[1], qo, kvp, ident, qlen, d ** -0.5, **gqa_cfg(hq, kvh, d))
    out["dq_gqa_us"] = round(kb.graph_time_us(dq, reps=3, iters=2), 1)
    if True:  # split-KV extend (nvfp4_verify_fwd handles token blocks)
        out["split_us"] = round(kb.graph_time_us(lambda: k4.nvfp4_verify_fwd(q, ke, ve, o, *fp4[:2], *fp4[2:], qo,
                                                                              kvp, idx, qlen, d ** -0.5)), 1)
    print(json.dumps(out), flush=True)
    del fp4, ws
    torch.cuda.empty_cache()


if __name__ == "__main__" and "midq" in sys.argv:
    midq("qwen38", 1, 100000, 4)
    midq("qwen38", 1, 100000, 32)
    midq("qwen38", 1, 100000, 1024)
    midq("qwen38", 1, 100000, 2048)
    midq("qwen38", 1, 100000, 300)
    midq("qwen38", 4, 100000, 300)
    midq("gemma_global", 1, 100000, 32)
    midq("gemma_global", 1, 100000, 300)
