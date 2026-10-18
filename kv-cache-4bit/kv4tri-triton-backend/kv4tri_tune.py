"""kv4tri decode/extend launch-geometry sweep (CUDA-graph timed) on the RTX shapes; prints the best configs.
    cd /root/work && python3 operations/kernels/kv4tri_tune.py decode|extend"""
import itertools
import json
import sys
from pathlib import Path

import flashinfer  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv4tri_bench as kb  # noqa: E402
import kv4tri_nvfp4_attention as k4  # noqa: E402

DEV = "cuda"
SHAPES = {"qwen38": (24, 4, 256), "qwen36": (16, 2, 256), "gemma_global": (16, 2, 512), "gemma_sliding": (16, 8, 256)}


def decode(name, B, ctx, splits_list=(8,)):
    hq, kvh, d = SHAPES[name]
    slots = B * ctx + 1
    fp4, _ = kb._pools(slots, kvh, d)
    q = torch.randn(B, hq, d, device=DEV, dtype=torch.bfloat16)
    kv_indptr = torch.arange(0, (B + 1) * ctx, ctx, dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(1, B * ctx + 1, device=DEV, dtype=torch.int64)
    o = torch.empty(B, hq, d, device=DEV, dtype=torch.bfloat16)
    res = []
    for splits in splits_list:
        att = torch.empty(B, hq, splits, d, device=DEV)
        lse = torch.empty(B, hq, splits, device=DEV)
        nsp = torch.full((B,), splits, dtype=torch.int32, device=DEV)
        for bn, nw, ns in itertools.product((16, 32, 64), (4, 8), (1, 2, 3)):
            try:
                t = kb.graph_time_us(lambda: k4.nvfp4_decode_fwd(q, *fp4[:2], *fp4[2:], o, kv_indptr, kv_indices, att,
                                                                  lse, nsp, splits, d ** -0.5, block_n=bn,
                                                                  num_warps=nw, num_stages=ns), reps=10, iters=3)
            except Exception as e:  # noqa: BLE001
                t = float("inf")
            res.append((t, dict(splits=splits, block_n=bn, num_warps=nw, num_stages=ns)))
    res.sort(key=lambda x: x[0])
    print(json.dumps({"decode": name, "B": B, "ctx": ctx, "best": [(round(t, 1), c) for t, c in res[:4]]}), flush=True)
    del fp4
    torch.cuda.empty_cache()


def extend(name, prefix, new, window=-1):
    hq, kvh, d = SHAPES[name]
    fp4, _ = kb._pools(prefix + 1, kvh, d)
    q = torch.randn(new, hq, d, device=DEV, dtype=torch.bfloat16)
    ke = torch.randn(new, kvh, d, device=DEV, dtype=torch.bfloat16)
    ve = torch.randn(new, kvh, d, device=DEV, dtype=torch.bfloat16)
    o = torch.empty_like(q)
    qo = torch.tensor([0, new], dtype=torch.int32, device=DEV)
    pe = min(prefix, window) if window > 0 else prefix
    kvp = torch.tensor([0, pe], dtype=torch.int32, device=DEV)
    idx = torch.arange(1 + prefix - pe, prefix + 1, device=DEV, dtype=torch.int64)
    g = hq // kvh
    gp = 1 << (g - 1).bit_length()
    res = []
    bts = sorted({max(1, x // gp) for x in (16, 32, 64, 128)})
    for bt, bn, dk, nw, ns in itertools.product(bts, (32, 64, 128), (64, 128), (4, 8), (1, 2, 3)):
        if dk > d:
            continue
        dv = min(d, 256)
        try:
            t = kb.graph_time_us(lambda: k4.nvfp4_extend_fwd(q, ke, ve, o, *fp4[:2], *fp4[2:], qo, kvp, idx, new,
                                                              d ** -0.5, sliding_window_size=window, block_t=bt,
                                                              block_n=bn, dk=dk, dv=dv, num_warps=nw,
                                                              num_stages=ns), reps=2, iters=2)
        except Exception:  # noqa: BLE001
            t = float("inf")
        res.append((t, dict(block_t=bt, block_n=bn, dk=dk, dv=dv, num_warps=nw, num_stages=ns)))
    res.sort(key=lambda x: x[0])
    print(json.dumps({"extend": name, "prefix": prefix, "new": new, "best": [(round(t, 1), c) for t, c in res[:4]]}),
          flush=True)
    del fp4
    torch.cuda.empty_cache()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "decode"
    if what == "decode":
        for name in ("qwen38", "qwen36", "gemma_global"):
            decode(name, 16, 104000)
            decode(name, 1, 104000, splits_list=(8, 32, 64))
        decode("gemma_sliding", 16, 1023)
    else:
        for name in ("qwen38", "qwen36", "gemma_global"):
            extend(name, 32768, 4096)
        extend("gemma_sliding", 32768, 4096, window=1023)
