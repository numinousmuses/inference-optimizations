"""CUDA-graph timing: 4-bit sparse decode (kv4pro_fp4.kv4_sparse_decode) vs FlashMLA FP8 on DeepSeek V4 Pro shapes.

    python3 operations/kernels/kv4pro_bench.py [--tokens 6,96,192,384] [--configs 128:8:2,32:4:1]

Per (query tokens T, layer kind) it builds the same random latent rows, stores them as FlashMLA 584-byte
FP8 cells and as 4-bit cells, gathers with random indices (SWA width 192 = DSpark verify table, c4 top-1024,
c128 = ceil64(46k / 128) = 384), and reports microseconds per call (graph of 20 calls, replayed, events) and
the relative output error of each path against the fp32 attention over the unquantized rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashinfer  # noqa: F401,E402  (engine ptxas)
import kv4pro_fp4 as K  # noqa: E402

H = 128
SCALE = 512 ** -0.5


def graph_us(fn, n=20, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / n)
    return best


def build(T, w_ext, ext_page, seed, n_swa=65536, n_ext=131072):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = "cuda"
    q = (torch.randn(T, H, 512, generator=g) * 0.05).to(dev, torch.bfloat16)
    swa_x = (torch.randn(n_swa, 512, generator=g) * 1.5).to(dev)
    ext_x = (torch.randn(n_ext, 512, generator=g) * 1.5).to(dev)
    swa_idx = torch.randint(0, n_swa, (T, 192), generator=g).to(dev, torch.int32)
    swa_len = torch.full((T,), 133, dtype=torch.int32, device=dev)
    ext_idx = torch.randint(0, n_ext, (T, w_ext), generator=g).to(dev, torch.int32)
    ext_len = torch.full((T,), w_ext if w_ext == 1024 else 360, dtype=torch.int32, device=dev)
    sink = torch.randn(H, generator=g).to(dev)
    return dict(q=q, swa_x=swa_x, ext_x=ext_x, swa_idx=swa_idx, swa_len=swa_len, ext_idx=ext_idx,
                ext_len=ext_len, sink=sink, ext_page=ext_page)


def flashmla_fn(c, swa8, ext8):
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

    sw = swa8[:, : 128 * 584].view(swa8.shape[0], 128, 1, 584).view(torch.float8_e4m3fn)
    ep = c["ext_page"]
    ex = ext8[:, : ep * 584].view(ext8.shape[0], ep, 1, 584).view(torch.float8_e4m3fn)
    meta = get_mla_metadata()[0]
    q4 = c["q"].unsqueeze(1)
    si, ei = c["swa_idx"].unsqueeze(1), c["ext_idx"].unsqueeze(1)

    def fn():
        return flash_mla_with_kvcache(q=q4, k_cache=sw, block_table=None, cache_seqlens=None, head_dim_v=512,
                                      tile_scheduler_metadata=meta, softmax_scale=SCALE, is_fp8_kvcache=True,
                                      indices=si, topk_length=c["swa_len"], attn_sink=c["sink"],
                                      extra_k_cache=ex, extra_indices_in_kvcache=ei,
                                      extra_topk_length=c["ext_len"])[0].squeeze(1)
    return fn


def ws_fn(c, swa_src, swa_is4, ext_src, ext_is4, rope):
    """4-bit -> FP8 workspace gather for the 4-bit pools, then the stock FlashMLA kernel (engine decode path)."""
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

    T = c["q"].shape[0]
    ep = c["ext_page"]
    ws_s = torch.empty(-(-T * 192 // 128), K.fp8_cell_page_bytes(128), dtype=torch.uint8, device="cuda")
    ws_e = torch.empty(-(-T * c["ext_idx"].shape[1] // ep), K.fp8_cell_page_bytes(ep), dtype=torch.uint8,
                       device="cuda")
    meta = get_mla_metadata()[0]
    q4 = c["q"].unsqueeze(1)

    def view(buf, page):
        return buf[:, : page * 584].view(buf.shape[0], page, 1, 584).view(torch.float8_e4m3fn)

    def fn():
        if swa_is4:
            si = K.kv4_to_fp8_workspace(swa_src, c["swa_idx"], c["swa_len"], ws_s, 128, rope)
            sw = view(ws_s, 128)
        else:
            si, sw = c["swa_idx"], view(swa_src, 128)
        if ext_is4:
            ei = K.kv4_to_fp8_workspace(ext_src, c["ext_idx"], c["ext_len"], ws_e, ep, rope)
            ex = view(ws_e, ep)
        else:
            ei, ex = c["ext_idx"], view(ext_src, ep)
        return flash_mla_with_kvcache(q=q4, k_cache=sw, block_table=None, cache_seqlens=None, head_dim_v=512,
                                      tile_scheduler_metadata=meta, softmax_scale=SCALE, is_fp8_kvcache=True,
                                      indices=si.unsqueeze(1), topk_length=c["swa_len"], attn_sink=c["sink"],
                                      extra_k_cache=ex, extra_indices_in_kvcache=ei.unsqueeze(1),
                                      extra_topk_length=c["ext_len"])[0].squeeze(1)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="6,96,192,384")
    ap.add_argument("--configs", default="128:8:2")
    ap.add_argument("--rope", default="fp8")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = []
    for T in [int(x) for x in a.tokens.split(",")]:
        for kind, w_ext, ext_page in (("c4", 1024, 64), ("c128", 384, 2)):
            c = build(T, w_ext, ext_page, seed=T + w_ext)
            swa8 = K.quantize_fp8_paged_ref(c["swa_x"], 128)
            ext8 = K.quantize_fp8_paged_ref(c["ext_x"], ext_page)
            swa4 = K.quantize_ref(c["swa_x"], a.rope)
            ext4 = K.quantize_ref(c["ext_x"], a.rope)
            ref, _ = K.sparse_decode_ref(c["q"].float(), None, c["swa_idx"], c["swa_len"], None, c["ext_idx"],
                                         c["ext_len"], attn_sink=c["sink"], softmax_scale=SCALE,
                                         kv_rows=(c["swa_x"], c["ext_x"]))
            rel = lambda o: ((o.float() - ref).norm() / ref.norm()).item()  # noqa: E731
            fm = flashmla_fn(c, swa8, ext8)
            row = {"T": T, "kind": kind, "flashmla_us": graph_us(fm), "flashmla_err": rel(fm())}
            # engine paths: mode all (SWA + c4 in 4 bit) and mode c4; c128 layers keep their FP8 pool
            ext4_used = kind == "c4"
            for name, swa_src, swa_is4 in (("ws_all", swa4, True), ("ws_c4", swa8, False)):
                f = ws_fn(c, swa_src, swa_is4, ext4 if ext4_used else ext8, ext4_used, a.rope)
                row[f"{name}_us"] = graph_us(f)
                row[f"{name}_err"] = rel(f())
            for cfg in a.configs.split(","):
                bh, nw, dv = (int(x) for x in cfg.split(":"))

                def mine(swa_cache, swa_fp8):
                    return lambda: K.kv4_sparse_decode(
                        c["q"], swa_cache, c["swa_idx"], c["swa_len"], c["sink"], SCALE, ext4, c["ext_idx"],
                        c["ext_len"], rope=a.rope, swa_fp8_page_size=128 if swa_fp8 else None, block_h=bh,
                        num_warps=nw, dv_split=dv)[0]
                try:
                    f_all = mine(swa4, False)
                    f_comp = mine(swa8, True)
                    row[f"all_{cfg}_us"] = graph_us(f_all)
                    row[f"all_{cfg}_err"] = rel(f_all())
                    row[f"comp_{cfg}_us"] = graph_us(f_comp)
                    row[f"comp_{cfg}_err"] = rel(f_comp())
                except Exception as e:  # report and keep going: a config may not fit this GPU
                    row[f"all_{cfg}_us"] = f"error {type(e).__name__}: {str(e)[:80]}"
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            del swa8, ext8, swa4, ext4
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
