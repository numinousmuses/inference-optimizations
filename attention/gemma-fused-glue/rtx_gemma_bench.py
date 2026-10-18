"""Microbench of rtxsys's Gemma 31B levers, stock vs fused, at the served shapes, for the ONE bundled 1xH100 run.

    python3 operations/kernels/gemma31/rtx_gemma_bench.py --out research/gpu-evidence/rtxsys/gemma31_rtx_bench_h100.json
    TRITON_INTERPRET=1 python3 operations/kernels/gemma31/rtx_gemma_bench.py --smoke     # CPU structure check

Rows (all on the v0.5.20 image's own stock kernels):
- D1 post-attn chain vs flashinfer rmsnorm + fused_add_rmsnorm + sglang_per_token_group_quant_fp8 (col-major TMA),
  at T = 192 / 288 (verify, 32 / 48 sessions) and 8192 (prefill chunk), hidden 5376.
- D1 cross-layer chain vs gemma_rmsnorm_residual_scalar + rmsnorm + quant, same shapes.
- D1c fused FP8 KV store vs k.clone / v.clone + div_ + .to(fp8) + store_cache. SWA rows: 16 heads x 256; full
  k_eq_v rows: 4 x 512.
- D2 verify sampling probabilities: dense (div + softmax + top_k_renorm_prob + top_p_renorm_prob + zeros_like)
  vs compact (flashinfer top_k(64) + uno normalize + Gumbel draw), [rows, 262144] fp32.
- D2-b draft top-1 vs torch.softmax + torch.max over [bs, 262144] fp32, bs 32 / 48.
- D2-c fused draft head (bf16 and fp8 copies) vs the stock head GEMM + fp32 copy + softmax + max, bs 32 / 48.
- Draft decode attention (rtx_draft_decode) vs stock decode_attention_fwd over FP8 KV: SWA 32/16 x 256 window 1023
  (with a geometry sweep), full 32/4 x 512 at 2K, bs 32 / 48.
- D1d fused q/k/v norm + rope vs stock gemma_qkv_rmsnorm + apply_rope_with_cos_sin_cache_inplace, T 192/288/8192.

Accuracy per row: bf16 ulp histogram / byte equality / index equality versus the stock output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import rtx_draft_decode as DD  # noqa: E402
import rtx_draft_top1 as T1  # noqa: E402
import rtx_qkv_norm_rope as QR  # noqa: E402
import rtx_glue as G  # noqa: E402
import rtx_kv_store as KS  # noqa: E402

H = 5376
VOCAB = 262144


def _time_us(fn, iters: int, warm: int = 3, graph: bool = True) -> float:
    """graph=True: CUDA-graph replay timing (what a captured verify / draft step pays); False: eager (the stock
    verify sampling runs eagerly)."""
    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            fn()
        return (time.perf_counter() - t0) * 1e6 / max(1, iters)
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    replay = fn
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        replay = g.replay
    for _ in range(warm):
        replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def _ulp_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()
    return {"max_ulp": int(d.max()), "frac_diff": round(float((d > 0).float().mean()), 7)}


def _stock_ops(smoke: bool):
    """Stock kernels from the image (or torch references in --smoke)."""
    if smoke:
        def quant(x):
            q, s = G.ref_group_quant(x)
            qq, ss = G.alloc_prequant(x.shape[0], x.shape[1], x.device)
            qq.copy_(q)
            ss.copy_(s)
            return qq, ss
        return {"rmsnorm": G.ref_rmsnorm, "fused_add": lambda x, r, w, e: G.ref_fused_add_rmsnorm(x, r, w, e),
                "res_scalar": G.ref_rmsnorm_residual_scalar, "quant": quant}
    from sgl_kernel import fused_add_rmsnorm, rmsnorm

    from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_rmsnorm_residual_scalar
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    def fused_add(x, r, w, e):
        fused_add_rmsnorm(x, r, w, e)
        return x, r

    return {"rmsnorm": lambda x, w, e: rmsnorm(x, w, e), "fused_add": fused_add,
            "res_scalar": gemma_rmsnorm_residual_scalar,
            "quant": lambda x: sglang_per_token_group_quant_fp8(x, 128, column_major_scales=True,
                                                                scale_tma_aligned=True)}


def bench_chains(dev, shapes, iters, smoke) -> list:
    ops = _stock_ops(smoke)
    rows = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for t in shapes:
        h = (torch.randn(t, H, generator=g) * 2).to(torch.bfloat16).to(dev)
        r = (torch.randn(t, H, generator=g) * 4).to(torch.bfloat16).to(dev)
        w1 = (1 + 0.2 * torch.randn(H, generator=g)).to(torch.bfloat16).to(dev)
        w2 = (1 + 0.2 * torch.randn(H, generator=g)).to(torch.bfloat16).to(dev)
        sc = torch.tensor([0.8125], dtype=torch.bfloat16, device=dev)

        def stock_pa():
            h1 = ops["rmsnorm"](h, w1, 1e-6)
            out, res = ops["fused_add"](h1, r.clone(), w2, 1e-6)
            return out, res, ops["quant"](out)

        def fused_pa():
            return G.post_attn_chain(h, r.clone(), w1, 1e-6, w2, 1e-6, True)

        so, sr, (sq, ss) = stock_pa()
        fo, fr, fq, fs = fused_pa()
        acc = {"out": _ulp_stats(fo, so), "residual": _ulp_stats(fr, sr),
               "q_bytes_frac_diff": round(float((fq.view(torch.uint8) != sq.view(torch.uint8)).float().mean()), 7),
               "scale_max_rel": float(((fs - ss).abs() / ss.abs().clamp_min(1e-30)).max())}
        rows.append({"lever": "D1 post-attn chain", "T": t, "stock_us": round(_time_us(stock_pa, iters), 2),
                     "fused_us": round(_time_us(fused_pa, iters), 2), "accuracy": acc})

        def stock_cx():
            out_l = ops["res_scalar"](h, w1, r, sc, 1e-6)
            nrm = ops["rmsnorm"](out_l, w2, 1e-6)
            return out_l, nrm, ops["quant"](nrm)

        def fused_cx():
            return G.cross_layer_chain(h, r, w1, 1e-6, sc, w2, 1e-6, True)

        sl, sn, (sq, ss) = stock_cx()
        fl, fn, fq, fs = fused_cx()
        acc = {"layer_out": _ulp_stats(fl, sl), "next_normed": _ulp_stats(fn, sn),
               "q_bytes_frac_diff": round(float((fq.view(torch.uint8) != sq.view(torch.uint8)).float().mean()), 7)}
        rows.append({"lever": "D1 cross-layer chain", "T": t, "stock_us": round(_time_us(stock_cx, iters), 2),
                     "fused_us": round(_time_us(fused_cx, iters), 2), "accuracy": acc})
    return rows


def bench_kv_store(dev, shapes, iters, smoke) -> list:
    rows = []
    g = torch.Generator(device="cpu").manual_seed(1)
    impl = None
    if not smoke:
        try:
            from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl as impl
        except Exception:
            impl = None
    for t in shapes:
        for heads, hd, label in ((16, 256, "swa"), (4, 512, "full")):
            slots = 4 * t + 64
            qkv = (torch.randn(t, 32 * 256 + 2 * heads * hd, generator=g) * 3).to(torch.bfloat16).to(dev)
            k = qkv[:, 32 * 256:32 * 256 + heads * hd].view(t, heads, hd)
            v = qkv[:, 32 * 256 + heads * hd:].view(t, heads, hd)
            loc = torch.randperm(slots, generator=g)[:t].to(dev)
            kc = torch.zeros(slots, heads, hd, dtype=torch.uint8, device=dev)
            vc = torch.zeros_like(kc)
            kc2, vc2 = kc.clone(), vc.clone()

            def stock():
                kk, vv = k.clone(), v.clone()
                kk.div_(1.0)
                vv.div_(1.0)
                kk = kk.to(torch.float8_e4m3fn).view(torch.uint8)
                vv = vv.to(torch.float8_e4m3fn).view(torch.uint8)
                if impl is not None:     # the served scatter (store_cache JIT where the row size is supported)
                    impl(kk, vv, kc, vc, loc, row_dim=heads * hd, store_dtype=torch.uint8,
                         device_module=torch.cuda, size_limit=slots)
                else:
                    kc[loc] = kk
                    vc[loc] = vv

            def fused():
                KS.store_fp8(k, v, kc2, vc2, loc, 1.0, 1.0)

            stock()
            fused()
            diag = {}
            for name, ref_c, got_c, src in (("k", kc, kc2, k), ("v", vc, vc2, v)):
                bad = (ref_c != got_c).nonzero()
                if bad.numel():
                    r0, h0, c0 = (int(x) for x in bad[0])
                    tok = int((loc == r0).nonzero()[0]) if bool((loc == r0).any()) else -1
                    diag[name] = {"n_bytes": int(bad.shape[0]), "rows": int(bad[:, 0].unique().numel()),
                                  "rows_in_loc": int(torch.isin(bad[:, 0].unique(), loc).sum()),
                                  "first": [r0, h0, c0], "stock_byte": int(ref_c[r0, h0, c0]),
                                  "fused_byte": int(got_c[r0, h0, c0]),
                                  "input": float(src[tok, h0, c0]) if tok >= 0 else None}
            rows.append({"lever": "D1c fused FP8 KV store", "T": t, "rows": label, "mismatch": diag,
                         "stock_us": round(_time_us(stock, iters), 2), "fused_us": round(_time_us(fused, iters), 2),
                         "stock_scatter": "store_cache via _set_kv_buffer_impl" if impl else "index_put",
                         "accuracy": {"bytes_equal": bool(torch.equal(kc, kc2) and torch.equal(vc, vc2))}})
    return rows


def bench_draft_top1(dev, bss, iters, smoke) -> list:
    rows = []
    g = torch.Generator(device="cpu").manual_seed(2)
    vocab = 8192 if smoke else VOCAB
    for bs in bss:
        x = (torch.randn(bs, vocab, generator=g) * 4).to(dev)

        def stock():
            return torch.max(torch.softmax(x, dim=-1), dim=-1, keepdim=True)

        def fused():
            return T1.softmax_top1(x)

        sp, si = stock()
        fp, fi = fused()
        rows.append({"lever": "D2-b draft softmax top-1", "bs": bs, "vocab": vocab,
                     "stock_us": round(_time_us(stock, iters), 2), "fused_us": round(_time_us(fused, iters), 2),
                     "accuracy": {"index_equal": bool(torch.equal(fi, si)),
                                  "p_max_rel": float(((fp - sp).abs() / sp).max())}})
    return rows


def bench_draft_head(dev, bss, iters, smoke) -> list:
    """Per draft step: stock (bf16 head GEMM -> fp32 logits buffer copy -> softmax -> max), D2-b (stock head +
    fused top-1), D2-c (fused head, bf16 / fp8 weights). Assistant head: 262,144 x 1,024 tied bf16."""
    rows = []
    g = torch.Generator(device="cpu").manual_seed(4)
    vocab, k = (2048, 256) if smoke else (VOCAB, 1024)
    w = (torch.randn(vocab, k, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    q, sc = T1.quantize_head_fp8(w)
    for bs in bss:
        h = torch.randn(bs, k, generator=g).to(torch.bfloat16).to(dev)
        buf = torch.empty(bs, vocab, dtype=torch.float32, device=dev)

        def stock():
            buf.copy_(h @ w.t())
            return torch.max(torch.softmax(buf, dim=-1), dim=-1, keepdim=True)

        def d2b():
            buf.copy_(h @ w.t())
            return T1.softmax_top1(buf)

        def d2c():
            return T1.head_top1(h, w, vocab)

        def d2c8():
            return T1.head_top1(h, q, vocab, sc)

        sp, si = stock()
        cp, ci = d2c()
        fp, fi = d2c8()
        rows.append({"lever": "D2-b/D2-c draft head + top-1", "bs": bs, "vocab": vocab,
                     "stock_us": round(_time_us(stock, iters), 2), "d2b_us": round(_time_us(d2b, iters), 2),
                     "d2c_bf16_us": round(_time_us(d2c, iters), 2), "d2c_fp8_us": round(_time_us(d2c8, iters), 2),
                     "accuracy": {"d2c_index_equal": bool(torch.equal(ci, si)),
                                  "d2c_p_max_rel": float(((cp - sp).abs() / sp).max()),
                                  "fp8_index_agree_frac": float((fi == si).float().mean())}})
    return rows


def bench_draft_decode(dev, bss, iters, smoke) -> list:
    """Draft decode attention over the target's FP8 KV: stock decode_attention_fwd (grouped kernel, 8 kv splits)
    vs rtx_draft_decode. SWA layer: 32 q / 16 kv heads x 256, window 1024. Full layer: 32 q / 4 kv heads x 512, 2K."""
    rows = []
    g = torch.Generator(device="cpu").manual_seed(5)
    stock = None
    if not smoke:
        from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd as stock
    cases = [("swa", 32, 16, 256, 1023), ("full", 32, 4, 512, 2048)]   # run5's logged window: 1,023
    if smoke:
        cases = [("swa", 4, 2, 64, 96)]
    for bs in bss:
        for label, hq, hkv, d, ctx in cases:
            slots = bs * ctx + 64
            q = torch.randn(bs, hq, d, generator=g).to(torch.bfloat16).to(dev)
            kb = (torch.randn(slots, hkv, d, generator=g) * 2).to(torch.float8_e4m3fn).to(dev)
            vb = torch.randn(slots, hkv, d, generator=g).to(torch.float8_e4m3fn).to(dev)
            indptr = (torch.arange(bs + 1, dtype=torch.int32) * ctx).to(dev)
            # int32 like TritonAttnBackend's kv_indices; the 7aa9ede3 bench fed int64, which the stock kernel misreads
            indices = torch.randperm(slots, generator=g)[: bs * ctx].to(torch.int32).to(dev)
            o1 = torch.empty(bs, hq, d, dtype=torch.bfloat16, device=dev)
            o2 = torch.empty_like(o1)
            window = ctx if label == "swa" else 0

            def mine():
                DD.decode(q, kb, vb, o2, indptr, indices, 1.0 / d ** 0.5, 1.0, 1.0, window)

            row = {"lever": "draft decode attention", "bs": bs, "layer": label, "ctx": ctx,
                   "fused_us": round(_time_us(mine, iters), 2)}
            if not smoke and label == "swa":
                # run5 lost to stock in-server (0.189 vs 0.163 ms per call at bs 32): sweep the geometry here
                sweep = {}
                for ns in (2, 3, 4, 8):
                    for bn, nw, st in ((32, 4, 2), (32, 4, 3), (64, 4, 2), (64, 8, 3), (32, 8, 4)):
                        def var(ns=ns, bn=bn, nw=nw, st=st):
                            DD.decode(q, kb, vb, o2, indptr, indices, 1.0 / d ** 0.5, 1.0, 1.0, window,
                                      n_splits=ns, block_n=bn, num_warps=nw, num_stages=st)
                        try:
                            sweep[f"s{ns}_bn{bn}_w{nw}_st{st}"] = round(_time_us(var, iters), 2)
                        except Exception as e:  # noqa: BLE001
                            sweep[f"s{ns}_bn{bn}_w{nw}_st{st}"] = f"error: {type(e).__name__}"
                row["sweep_us"] = sweep
                ok = {k: v for k, v in sweep.items() if isinstance(v, float)}
                if ok:
                    row["best"] = min(ok, key=ok.get)
            mine()
            if stock is not None:
                ms = 8
                logits = torch.empty(bs, hq, ms, d, dtype=torch.float32, device=dev)
                lse = torch.empty(bs, hq, ms, dtype=torch.float32, device=dev)
                nsplit = torch.full((bs,), ms, dtype=torch.int32, device=dev)

                def ref():
                    stock(q, kb, vb, o1, indptr, indices, logits, lse, nsplit, ms, 1.0 / d ** 0.5, 1.0, 1.0)

                row["stock_us"] = round(_time_us(ref, iters), 2)
                ref()
                row["accuracy"] = {"max_abs_vs_stock": float((o1.float() - o2.float()).abs().max())}
            else:
                ref = DD.decode_ref(q, kb, vb, indptr, indices, 1.0 / d ** 0.5)
                row["accuracy"] = {"max_abs_vs_ref": float((ref - o2.float()).abs().max())}
            rows.append(row)
    return rows


def bench_qkv_norm_rope(dev, shapes, iters, smoke) -> list:
    """D1d: stock gemma_qkv_rmsnorm + apply_rope_with_cos_sin_cache_inplace vs the fused kernel, in place on a qkv
    buffer. SWA 32/16 x 256 (theta 1e4) and full 32/4 x 512 (proportional, 64 rotated angles)."""
    rows = []
    g = torch.Generator(device="cpu").manual_seed(6)
    stock = None
    if not smoke:
        from sglang.kernels.ops.attention.rope import apply_rope_with_cos_sin_cache_inplace
        from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm

        def stock(q, k, v, qw, kw, cs, pos, hq, hkv, d):
            gemma_qkv_rmsnorm(q, k, v, qw, kw, num_q_heads=hq, num_kv_heads=hkv, head_dim=d, eps=1e-6)
            apply_rope_with_cos_sin_cache_inplace(positions=pos, q=q.view(q.shape[0], -1, d),
                                                  k=k.view(k.shape[0], -1, d), cos_sin_cache=cs, is_neox=True)
    cases = [("swa", 32, 16, 256, 1e4, None), ("full", 32, 4, 512, 1e6, 64)]
    if smoke:
        cases = [("swa", 4, 2, 64, 1e4, None)]
    for t in shapes:
        for label, hq, hkv, d, base, ra in cases:
            r = d // 2 if ra is None else ra
            inv = 1.0 / (base ** (torch.arange(0, 2 * r, 2, dtype=torch.float) / d))
            inv = torch.cat([inv, torch.zeros(d // 2 - r)])
            f = torch.outer(torch.arange(8192, dtype=torch.float), inv)
            cs = torch.cat([f.cos(), f.sin()], -1).to(dev)
            qkv = (torch.randn(t, (hq + 2 * hkv) * d, generator=g) * 2).to(torch.bfloat16).to(dev)
            qw = (1 + 0.1 * torch.randn(d, generator=g)).to(torch.bfloat16).to(dev)
            kw = (1 + 0.1 * torch.randn(d, generator=g)).to(torch.bfloat16).to(dev)
            pos = torch.randint(0, 8192, (t,), generator=g).to(dev)
            a, b = qkv.clone(), qkv.clone()
            qa, ka, va = a.split([hq * d, hkv * d, hkv * d], dim=-1)
            qb, kb, vb = b.split([hq * d, hkv * d, hkv * d], dim=-1)

            def fused():
                QR.norm_rope(qb, kb, vb, qw, kw, cs, pos, hq, hkv, d, 1e-6)

            row = {"lever": "D1d qkv norm+rope", "T": t, "layer": label, "fused_us": round(_time_us(fused, iters), 2)}
            b.copy_(qkv)
            fused()
            if stock is not None:
                def ref():
                    stock(qa, ka, va, qw, kw, cs, pos, hq, hkv, d)
                row["stock_us"] = round(_time_us(ref, iters), 2)
                a.copy_(qkv)
                ref()
                row["accuracy"] = _ulp_stats(b, a)
            rows.append(row)
    return rows


def bench_verify_sampling(dev, rows_list, iters, smoke) -> list:
    if smoke:
        return [{"lever": "D2 sparse verify sampling", "skipped": "needs sgl_kernel / flashinfer (GPU)"}]
    import torch.nn.functional as F
    from flashinfer import top_k as fi_top_k
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

    out = []
    g = torch.Generator(device="cpu").manual_seed(3)
    for rws in rows_list:
        logits = (30 * torch.tanh(torch.randn(rws, VOCAB, generator=g) * 3 / 30)).to(dev)
        temps = torch.full((rws, 1), 0.6, device=dev)
        top_ks = torch.full((rws,), 64, dtype=torch.int32, device=dev)
        top_ps = torch.full((rws,), 0.95, device=dev)

        def dense():
            p = F.softmax(logits / temps, dim=-1)
            p = top_k_renorm_prob(p, top_ks)
            p = top_p_renorm_prob(p, top_ps)
            return p, torch.zeros_like(p)

        def compact():
            vals, ids = fi_top_k(logits, 64, sorted=True, deterministic=False)
            sc = vals.float() / temps
            pr = torch.softmax(sc, dim=-1)
            cdf = torch.cumsum(pr, dim=-1)
            pr = pr.masked_fill((cdf - pr) > top_ps[:, None], 0.0)
            pr = pr / pr.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            pick = (pr / torch.empty_like(pr).exponential_(1)).argmax(-1, keepdim=True)
            return ids.gather(1, pick)

        dp, _ = dense()
        vals, ids = fi_top_k(logits, 64, sorted=True, deterministic=False)
        sc = torch.softmax(vals.float() / temps, dim=-1)
        cdf = torch.cumsum(sc, dim=-1)
        sc = sc.masked_fill((cdf - sc) > top_ps[:, None], 0.0)
        sc = sc / sc.sum(dim=-1, keepdim=True)
        dense_at = dp.gather(1, ids)
        err = (dense_at - sc).abs()
        wr = int(err.max(dim=1).values.argmax())
        p_sorted = torch.softmax(vals.float()[wr] / temps[wr], dim=-1)
        cum = torch.cumsum(p_sorted, 0)
        cross = int((cum >= 0.95).nonzero()[0]) if bool((cum >= 0.95).any()) else -1
        worst = {"row": wr, "dense_support": int((dense_at[wr] > 0).sum()), "compact_support": int((sc[wr] > 0).sum()),
                 "dense_nonzero_total": int((dp[wr] > 0).sum()), "crossing_rank": cross,
                 "cum_at_crossing": float(cum[cross]) if cross >= 0 else None,
                 "cum_before_crossing": float(cum[cross - 1]) if cross > 0 else None,
                 "p_at_crossing": float(p_sorted[cross]) if cross >= 0 else None,
                 "logit_gap_at_64": float(vals[wr, 63] - logits[wr].topk(65).values[64])}
        out.append({"lever": "D2 sparse verify sampling (probability construction)", "rows": rws,
                    "stock_us": round(_time_us(dense, iters, graph=False), 2),
                    "fused_us": round(_time_us(compact, iters, graph=False), 2),
                    "accuracy": {"support_prob_max_abs": float(err.max()),
                                 "rows_over_1e-5": int((err.max(dim=1).values > 1e-5).sum()),
                                 "dense_mass_outside_top64": float((1 - dense_at.sum(-1)).abs().max()),
                                 "worst_row": worst}})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="CPU interpreter structure check at tiny shapes")
    a = ap.parse_args(argv)
    dev = "cpu" if a.smoke else "cuda"
    iters = 1 if a.smoke else a.iters
    res = {"gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "rows": []}
    res["rows"] += bench_chains(dev, [3] if a.smoke else [192, 288, 8192], iters, a.smoke)
    res["rows"] += bench_kv_store(dev, [3] if a.smoke else [192, 288, 8192], iters, a.smoke)
    res["rows"] += bench_draft_top1(dev, [2] if a.smoke else [32, 48], iters, a.smoke)
    res["rows"] += bench_draft_head(dev, [2] if a.smoke else [32, 48], iters, a.smoke)
    res["rows"] += bench_draft_decode(dev, [2] if a.smoke else [32, 48], iters, a.smoke)
    res["rows"] += bench_qkv_norm_rope(dev, [3] if a.smoke else [192, 288, 8192], iters, a.smoke)
    res["rows"] += bench_verify_sampling(dev, [192, 288], iters, a.smoke)
    for r in res["rows"]:
        print(json.dumps(r))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
