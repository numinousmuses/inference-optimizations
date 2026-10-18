"""1xH100 microbench: the stock extend kernel on a Gemma 4 31B SWA prefill chunk, as shipped vs with the window
loop-start patch (operations/patches/gemma31_swa_window_skip_hooks.py) - time and bit-identity (rtxattn).

The patched kernel is the stock extend_attention.py source with the hook applied, loaded as a separate module, so
both run in one process on the same inputs. Shapes: 32 q / 16 kv heads, hd 256, window 1024, prefill chunk of N new
tokens after a prefix P (0 / 4K), bf16 K/V for the chunk and an FP8 pool for the prefix.

    python3 operations/kernels/gemma31_swa_prefill_bench.py --json /root/g31_swa_prefill.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import statistics
import sys
import tempfile

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "patches"))
import gemma31_swa_window_skip_hooks as H  # noqa: E402


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
        ts.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(ts)


def load_patched():
    import sglang
    src_root = pathlib.Path(list(sglang.__path__)[0])
    tmp = pathlib.Path(tempfile.mkdtemp()) / "sglang"
    (tmp / H.REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_root / H.REL, tmp / H.REL)
    H.apply(str(tmp))
    spec = importlib.util.spec_from_file_location("sglang.kernels.ops.attention.extend_attention_numinous",
                                                  tmp / H.REL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--full-only", action="store_true")
    a = ap.parse_args(argv)
    from sglang.kernels.ops.attention import extend_attention as stock_mod
    patched_mod = load_patched()
    dev = "cuda"
    hq, hkv, d, window = 32, 16, 256, 1024
    rows = []
    for prefix, n_new in (() if a.full_only else ((0, 8192), (4096, 8192), (0, 2048))):
        g = torch.Generator(device=dev).manual_seed(0)
        q = torch.randn(n_new, hq, d, device=dev, generator=g).to(torch.bfloat16)
        ke = torch.randn(n_new, hkv, d, device=dev, generator=g).to(torch.bfloat16)
        ve = torch.randn(n_new, hkv, d, device=dev, generator=g).to(torch.bfloat16)
        w_list = min(prefix, window)
        pool = max(w_list, 1) + 16
        kb = torch.randn(pool, hkv, d, device=dev, generator=g).to(torch.bfloat16)
        vb = torch.randn(pool, hkv, d, device=dev, generator=g).to(torch.bfloat16)
        kvi = torch.arange(w_list, device=dev, dtype=torch.int32) if w_list else torch.zeros(1, device=dev, dtype=torch.int32)
        kvp = torch.tensor([0, w_list], device=dev, dtype=torch.int32)
        qo = torch.tensor([0, n_new], device=dev, dtype=torch.int32)
        outs = {}
        row = {"prefix": prefix, "new_tokens": n_new}
        for tag, mod in (("stock", stock_mod), ("patched", patched_mod)):
            o = torch.empty_like(q)

            def f(mod=mod, o=o):
                mod.extend_attention_fwd(q, ke, ve, o, kb, vb, qo, kvp, kvi, None, True, None, n_new, 1.0, 1.0,
                                         d ** -0.5, logit_cap=0.0, sliding_window_size=window)
            try:
                row[f"{tag}_us"] = round(timeit(f), 1)
                outs[tag] = o.clone()
            except Exception as e:  # noqa: BLE001
                row[f"{tag}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        # rtxattn's GQA-packed window extend (gemma31_verify.extend_gqa_window) + a small launch sweep
        sys.path.insert(0, str(HERE))
        import gemma31_verify as GV
        best = None
        for bm in (32, 64):
            for bn in (32, 64):
                for nw in (4, 8):
                    o3 = torch.empty_like(q)

                    def f3(bm=bm, bn=bn, nw=nw, o3=o3):
                        GV.extend_gqa_window(q, ke, ve, kb, vb, o3, qo, kvp, kvi, n_new, d ** -0.5, 1.0, 1.0,
                                             window=window, block_m=bm, block_n=bn, num_warps=nw)
                    try:
                        us = timeit(f3)
                    except Exception as e:  # noqa: BLE001
                        row.setdefault("gqa_errors", []).append(f"bm{bm} bn{bn} nw{nw}: {type(e).__name__}")
                        continue
                    if best is None or us < best[0]:
                        best = (us, bm, bn, nw, o3.clone())
        if best is not None:
            row["gqa_us"] = round(best[0], 1)
            row["gqa_cfg"] = {"block_m": best[1], "block_n": best[2], "num_warps": best[3]}
            if "stock" in outs:
                dlt = best[4].float() - outs["stock"].float()
                row["gqa_rel_l2_vs_stock"] = round((dlt.norm() / outs["stock"].float().norm()).item(), 6)
                row["gqa_speedup_vs_stock"] = round(row["stock_us"] / row["gqa_us"], 2)
                row["gqa_ms_50_layers"] = round(row["gqa_us"] * 50 / 1000, 1)
        if len(outs) == 2:
            row["bit_identical"] = bool(torch.equal(outs["stock"], outs["patched"]))
            row["speedup"] = round(row["stock_us"] / row["patched_us"], 2)
            row["stock_ms_50_layers"] = round(row["stock_us"] * 50 / 1000, 1)
            row["patched_ms_50_layers"] = round(row["patched_us"] * 50 / 1000, 1)
        print(json.dumps(row), flush=True)
        rows.append(row)
    # full-attention layers (hd 512, 32 q / 4 kv heads, no window): stock extend vs extend_gqa_window(window 0)
    import gemma31_verify as GV
    for prefix, n_new in ((0, 8192), (4096, 8192), (0, 2048)):
        g = torch.Generator(device=dev).manual_seed(1)
        hq2, hkv2, d2 = 32, 4, 512
        q = torch.randn(n_new, hq2, d2, device=dev, generator=g).to(torch.bfloat16)
        ke = torch.randn(n_new, hkv2, d2, device=dev, generator=g).to(torch.bfloat16)
        ve = torch.randn(n_new, hkv2, d2, device=dev, generator=g).to(torch.bfloat16)
        pool = max(prefix, 1) + 16
        ks = vs = 0.02
        kb = (torch.randn(pool, hkv2, d2, device=dev, generator=g) / ks).clamp(-448, 448).to(torch.float8_e4m3fn)
        vb = (torch.randn(pool, hkv2, d2, device=dev, generator=g) / vs).clamp(-448, 448).to(torch.float8_e4m3fn)
        kvi = torch.arange(prefix, device=dev, dtype=torch.int32) if prefix else torch.zeros(1, device=dev, dtype=torch.int32)
        kvp = torch.tensor([0, prefix], device=dev, dtype=torch.int32)
        qo = torch.tensor([0, n_new], device=dev, dtype=torch.int32)
        row = {"layer": "full", "prefix": prefix, "new_tokens": n_new}
        o_s = torch.empty_like(q)

        def fs():
            stock_mod.extend_attention_fwd(q, ke, ve, o_s, kb, vb, qo, kvp, kvi, None, True, None, n_new, ks, vs,
                                           d2 ** -0.5, logit_cap=0.0, sliding_window_size=-1)
        try:
            row["stock_us"] = round(timeit(fs), 1)
        except Exception as e:  # noqa: BLE001
            row["stock_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        best = None
        for bm, bn, nw, ns, ds in [(4, 32, 4, 2, 1), (8, 32, 4, 2, 2), (8, 64, 8, 2, 2), (16, 32, 8, 2, 2),
                                   (16, 64, 8, 2, 2), (8, 32, 8, 2, 2), (16, 32, 8, 3, 2), (8, 64, 4, 2, 2),
                                   (16, 64, 8, 2, 4), (8, 32, 4, 2, 4), (4, 64, 4, 2, 2), (16, 32, 4, 2, 2)]:
                        o3 = torch.empty_like(q)

                        def f3(bm=bm, bn=bn, nw=nw, ns=ns, ds=ds, o3=o3):
                            GV.extend_gqa_window(q, ke, ve, kb, vb, o3, qo, kvp, kvi, n_new, d2 ** -0.5, ks, vs,
                                                 window=0, block_m=bm, block_n=bn, num_warps=nw, num_stages=ns,
                                                 d_split=ds)
                        try:
                            us = timeit(f3)
                        except Exception as e:  # noqa: BLE001
                            continue
                        if best is None or us < best[0]:
                            best = (us, {"block_m": bm, "block_n": bn, "num_warps": nw, "num_stages": ns,
                                         "d_split": ds}, o3.clone())
        if best is not None:
            row["gqa_us"] = round(best[0], 1)
            row["gqa_cfg"] = best[1]
            if "stock_us" in row:
                dlt = best[2].float() - o_s.float()
                row["gqa_rel_l2_vs_stock"] = round((dlt.norm() / o_s.float().norm()).item(), 6)
                row["speedup"] = round(row["stock_us"] / row["gqa_us"], 2)
                row["stock_ms_10_layers"] = round(row["stock_us"] * 10 / 1000, 1)
                row["gqa_ms_10_layers"] = round(row["gqa_us"] * 10 / 1000, 1)
        print(json.dumps(row), flush=True)
        rows.append(row)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"device": torch.cuda.get_device_name(), "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
