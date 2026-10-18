"""1xH100: Gemma full-layer prefill through extend_fa4_hd512 (FA4 two-pass, glue incl. FP8-prefix gather) vs stock
extend_attention_fwd vs Triton extend_gqa_window; rel L2 vs stock. python3 operations/kernels/g31_fa4_glue_bench.py"""
import json
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gemma31_verify as GV  # noqa: E402


def timeit(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd  # noqa: E402
dev = "cuda"
hq, hkv, d = 32, 4, 512
rows = []
for prefix, n in ((0, 8192), (4096, 8192), (0, 2048)):
    g = torch.Generator(device=dev).manual_seed(3)
    q = torch.randn(n, hq, d, device=dev, generator=g).to(torch.bfloat16)
    ke = torch.randn(n, hkv, d, device=dev, generator=g).to(torch.bfloat16)
    ve = torch.randn(n, hkv, d, device=dev, generator=g).to(torch.bfloat16)
    ks = vs = 0.02
    pool = max(prefix, 1) + 16
    kb = (torch.randn(pool, hkv, d, device=dev, generator=g) / ks).clamp(-448, 448).to(torch.float8_e4m3fn)
    vb = (torch.randn(pool, hkv, d, device=dev, generator=g) / vs).clamp(-448, 448).to(torch.float8_e4m3fn)
    kvi = torch.arange(prefix, device=dev, dtype=torch.int32) if prefix else torch.zeros(1, device=dev, dtype=torch.int32)
    kvp = torch.tensor([0, prefix], device=dev, dtype=torch.int32)
    qo = torch.tensor([0, n], device=dev, dtype=torch.int32)
    o_s, o_t, o_f = (torch.empty_like(q) for _ in range(3))
    sm = d ** -0.5
    fs = lambda: extend_attention_fwd(q, ke, ve, o_s, kb, vb, qo, kvp, kvi, None, True, None, n, ks, vs, sm,  # noqa
                                      logit_cap=0.0, sliding_window_size=-1)
    ft = lambda: GV.extend_gqa_window(q, ke, ve, kb, vb, o_t, qo, kvp, kvi, n, sm, ks, vs, window=0)  # noqa
    ff = lambda: GV.extend_fa4_hd512(q, ke, ve, kb, vb, o_f, qo, kvp, kvi, sm, ks, vs)  # noqa
    row = {"prefix": prefix, "new_tokens": n}
    for tag, f in (("stock", fs), ("triton", ft), ("fa4", ff)):
        try:
            row[f"{tag}_ms"] = round(timeit(f), 3)
        except Exception as e:  # noqa: BLE001
            row[f"{tag}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    for tag, o in (("triton", o_t), ("fa4", o_f)):
        dl = o.float() - o_s.float()
        row[f"{tag}_rel_l2_vs_stock"] = round((dl.norm() / o_s.float().norm()).item(), 6)
    row["fa4_ms_10_layers"] = round(row.get("fa4_ms", 0) * 10, 1)
    print(json.dumps(row), flush=True)
    rows.append(row)
pathlib.Path("/root/g31_fa4_glue.json").write_text(json.dumps(rows, indent=1))
