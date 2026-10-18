"""1xH100 sweep: FA4 tile / pack_gqa for the Gemma hd256 SWA prefill layer (window 1023), plus the hd512 full layer
with FA4 writing straight into the strided output halves. Reference = the shipped default (FA4's own tile, copy-out
path recomputed via flash_attn_varlen_func). python3 operations/kernels/g31_fa4_swa_sweep.py"""
import json
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gemma31_verify as GV  # noqa: E402


def timeit(fn, reps=7):
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


def case(hq, hkv, d, prefix, n, seed=4):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
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
    return q, ke, ve, kb, vb, qo, kvp, kvi, ks, vs


def ref_copy_path(q, ke, ve, kb, vb, qo, kvp, kvi, ks, vs, window):
    """the shipped (pre-out=) path: flash_attn_varlen_func + copy, FA4's default tile"""
    o = torch.empty_like(q)
    fi = GV._fa4()
    GV.extend_fa4_hd512(q, ke, ve, kb, vb, o, qo, kvp, kvi, q.shape[-1] ** -0.5, ks, vs,
                        flash=fi.flash_attn_varlen_func, window=window)
    return o


rows = []
W = 1023
for prefix, n in ((0, 8192), (1024, 8192)):
    args = case(32, 16, 256, prefix, n)
    q = args[0]
    ref = ref_copy_path(*args, W)
    t_ref = timeit(lambda: ref_copy_path(*args, W))
    rows.append({"layer": "swa", "prefix": prefix, "n": n, "cfg": "shipped(copy-out, default tile)", "ms": round(t_ref, 3)})
    print(json.dumps(rows[-1]), flush=True)
    for tile in (None, (128, 64, True, True), (128, 80, True, True), (128, 96, True, True), (128, 112, True, True),
                 (128, 128, True, True), (128, 80, False, True), (128, 64, True, False), (128, 96, True, False),
                 (128, 80, True, False)):
        for pack in (False, True):
            GV._FA4_TILE_256, GV._FA4_PACK_GQA_256 = tile, pack
            o = torch.empty_like(q)
            f = lambda: GV.extend_fa4_hd512(args[0], args[1], args[2], args[3], args[4], o, args[5], args[6],  # noqa
                                           args[7], 256 ** -0.5, args[8], args[9], window=W)
            row = {"layer": "swa", "prefix": prefix, "n": n, "cfg": f"tile={tile} pack_gqa={pack} out=direct"}
            try:
                row["ms"] = round(timeit(f), 3)
                row["rel_l2_vs_shipped"] = round(((o.float() - ref.float()).norm() / ref.float().norm()).item(), 6)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            print(json.dumps(row), flush=True)
            rows.append(row)
GV._FA4_TILE_256, GV._FA4_PACK_GQA_256 = None, False
for prefix, n in ((0, 8192), (4096, 8192)):
    args = case(32, 4, 512, prefix, n)
    q = args[0]
    ref = ref_copy_path(*args, 0)
    t_ref = timeit(lambda: ref_copy_path(*args, 0))
    o = torch.empty_like(q)
    f = lambda: GV.extend_fa4_hd512(args[0], args[1], args[2], args[3], args[4], o, args[5], args[6], args[7],  # noqa
                                   512 ** -0.5, args[8], args[9])
    row = {"layer": "full_hd512", "prefix": prefix, "n": n, "shipped_ms": round(t_ref, 3)}
    try:
        row["out_direct_ms"] = round(timeit(f), 3)
        row["rel_l2_vs_shipped"] = round(((o.float() - ref.float()).norm() / ref.float().norm()).item(), 6)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    print(json.dumps(row), flush=True)
    rows.append(row)
json.dump(rows, open("/root/g31/out/fa4_swa_sweep.json", "w"), indent=1)
