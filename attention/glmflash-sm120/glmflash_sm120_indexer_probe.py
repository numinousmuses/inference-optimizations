"""Flash DSA indexer kernels on SM120 (RTX PRO 6000): correctness vs torch + graph-timed cost at served shapes. NOT RUN.

What Flash's kpool indexer calls on CUDA (source_code, SGLang v0.5.20 dsa_indexer_kpool.py; sgl-deep-gemm 0.2.0 ships
sm120_fp8_paged_mqa_logits / sm120_fp8_mqa_logits, host checks `arch_major == 12 and block_kv == 64`, head_dim 128):
  decode / verify   deep_gemm.get_paged_mqa_logits_metadata + deep_gemm.fp8_paged_mqa_logits over POOLED keys
                    (q [B, 1, 32, 128] e4m3, cache [pages, 64, 1, 132] = 64 x 128 e4m3 then 64 fp32 scales per page,
                    weights [B, 32] fp32), called WITHOUT the batch <= num_sms chunking the non-kpool path applies
  prefill (ragged)  deep_gemm.fp8_mqa_logits(q [S, 32, 128] e4m3, (k [N, 128] e4m3, k_scale [N] fp32), w [S, 32], ks, ke)
  top-k             sglang.kernels.ops.moe.kpool_topk_transform.fast_kpool_topk_transform_fused (CUDA JIT, radix
                    top-512 groups -> 2048 tokens + tail), group_topk 512
Reference: logits[b, n] = sum_h w[b, h] * relu(q[b, h] . k[n]) * k_scale[n] (DeepGEMM contract; dsv4/indexer.py's
fp8_paged_mqa_logits_torch is the same math).

    python3 operations/kernels/glmflash_sm120_indexer_probe.py --out /root/rtxattn_indexer.json
Prints INDEXER_OK / INDEXER_FAIL <which>.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

DEV = "cuda"
H, D, BLOCK = 32, 128, 64


def graph_us(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3 / reps


def paged_case(b: int, ctx_pooled: int, seed: int = 0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    pages_per = (ctx_pooled + BLOCK - 1) // BLOCK
    n_pages = b * pages_per + 1
    kv = torch.empty(n_pages, BLOCK * (D + 4), dtype=torch.uint8, device=DEV)
    kval = (torch.randn(n_pages, BLOCK, D, generator=g, device=DEV)).to(torch.float8_e4m3fn)
    ksc = torch.rand(n_pages, BLOCK, generator=g, device=DEV) * 0.02 + 0.01
    kv[:, :BLOCK * D] = kval.view(torch.uint8).reshape(n_pages, -1)
    kv[:, BLOCK * D:] = ksc.contiguous().view(torch.uint8).reshape(n_pages, -1)
    q = torch.randn(b, 1, H, D, generator=g, device=DEV).to(torch.float8_e4m3fn)
    w = torch.randn(b, H, generator=g, device=DEV)
    perm = torch.randperm(n_pages - 1, generator=g, device=DEV) + 1
    table = perm[: b * pages_per].reshape(b, pages_per).to(torch.int32)
    lens = torch.randint(ctx_pooled // 2, ctx_pooled + 1, (b,), generator=g, device=DEV).to(torch.int32)
    return q, kv, kval, ksc, w, table, lens


def paged_ref(q, kval, ksc, w, table, lens, max_len):
    b = q.shape[0]
    k = kval[table.long()].float().reshape(b, -1, D)  # [b, pages*64, D]
    sc = ksc[table.long()].reshape(b, -1)
    s = torch.einsum("bhd,bnd->bhn", q[:, 0].float(), k).relu()
    logits = (s * w[:, :, None]).sum(1) * sc
    pos = torch.arange(logits.shape[1], device=DEV)[None]
    logits = logits.masked_fill(pos >= lens[:, None], float("nan"))
    return logits[:, :max_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/rtxattn_indexer.json")
    a = ap.parse_args()
    import deep_gemm
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    res, fails = [], []
    # decode / verify: B query rows, 31K-token contexts pooled by 4 -> 7.75K pooled keys (the OR Flash mix)
    for b in (1, 64, 188, 256):
        q, kv, kval, ksc, w, table, lens = paged_case(b, 7750)
        max_len = table.shape[1] * BLOCK
        kv4 = kv.view(kv.shape[0], BLOCK, 1, D + 4)
        rec = {"case": "fp8_paged_mqa_logits", "B": b, "pooled_ctx": 7750}
        try:
            meta = deep_gemm.get_paged_mqa_logits_metadata(lens.view(-1, 1).contiguous(), BLOCK, num_sms)
            fn = lambda: deep_gemm.fp8_paged_mqa_logits(q, kv4, w, lens.view(-1, 1).contiguous(), table, meta,
                                                        max_len, clean_logits=False)
            out = fn()
            ref = paged_ref(q, kval, ksc, w, table, lens, max_len)
            m = ~torch.isnan(ref)
            err = float((out[m] - ref[m]).norm() / ref[m].norm())
            rec.update(rel_err=err, us=graph_us(fn), ok=err < 1e-2)
        except Exception as e:
            rec.update(ok=False, error=str(e)[:300])
        (res.append(rec), print(json.dumps(rec), flush=True))
        if not rec["ok"]:
            fails.append(f"paged B={b}")
    # prefill: S query tokens of one chunk against N pooled keys (ragged)
    for s_q, n_k in ((2048, 7750), (8192, 7750)):
        g = torch.Generator(device=DEV).manual_seed(3)
        q = torch.randn(s_q, H, D, generator=g, device=DEV).to(torch.float8_e4m3fn)
        k = torch.randn(n_k, D, generator=g, device=DEV).to(torch.float8_e4m3fn)
        ks = torch.rand(n_k, generator=g, device=DEV) * 0.02 + 0.01
        w = torch.randn(s_q, H, generator=g, device=DEV)
        ke = torch.linspace(n_k // 2, n_k, s_q, device=DEV).to(torch.int32)
        kst = torch.zeros(s_q, dtype=torch.int32, device=DEV)
        rec = {"case": "fp8_mqa_logits", "S": s_q, "N": n_k}
        try:
            fn = lambda: deep_gemm.fp8_mqa_logits(q, (k, ks), w, kst, ke, clean_logits=False)
            out = fn()
            ref = (torch.einsum("shd,nd->shn", q.float(), k.float()).relu() * w[:, :, None]).sum(1) * ks[None]
            mask = torch.arange(n_k, device=DEV)[None] < ke[:, None]
            err = float((out[mask] - ref[mask]).norm() / ref[mask].norm())
            rec.update(rel_err=err, us=graph_us(fn, reps=3), ok=err < 1e-2)
        except Exception as e:
            rec.update(ok=False, error=str(e)[:300])
        (res.append(rec), print(json.dumps(rec), flush=True))
        if not rec["ok"]:
            fails.append(f"ragged S={s_q}")
    # kpool top-k transform (group_topk 512 of pooled logits -> 2048 tokens + tail), compare selected groups
    try:
        from sglang.kernels.ops.moe.kpool_topk_transform import fast_kpool_topk_transform_fused
        b, groups = 64, 7750
        g = torch.Generator(device=DEV).manual_seed(5)
        score = torch.randn(b, groups, generator=g, device=DEV)
        lengths = torch.full((b,), groups, dtype=torch.int32, device=DEV)
        seq = lengths * 4 + 3
        out = fast_kpool_topk_transform_fused(score, lengths, 4, 2048, seq_lens=seq)
        ref_groups = score.topk(512, dim=1).indices.sort(1).values
        got_groups = (out[:, :2048:4] // 4).sort(1).values
        tail_ok = bool(torch.all(out[:, 2048:] == (lengths[:, None] * 4 + torch.arange(3, device=DEV))))
        rec = {"case": "kpool_topk_transform", "B": b, "width": out.shape[1],
               "groups_equal": bool(torch.equal(got_groups.long(), ref_groups)), "tail_ok": tail_ok,
               "us": graph_us(lambda: fast_kpool_topk_transform_fused(score, lengths, 4, 2048, seq_lens=seq))}
        rec["ok"] = rec["groups_equal"] and tail_ok and out.shape[1] == 2051
    except Exception as e:
        rec = {"case": "kpool_topk_transform", "ok": False, "error": str(e)[:300]}
    (res.append(rec), print(json.dumps(rec), flush=True))
    if not rec["ok"]:
        fails.append("kpool_topk")
    pathlib.Path(a.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "rows": res}, indent=1))
    print("INDEXER_OK" if not fails else "INDEXER_FAIL " + ",".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
