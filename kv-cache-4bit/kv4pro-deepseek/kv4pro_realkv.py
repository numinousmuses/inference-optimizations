"""Quantization error of DeepSeek V4 Pro KV rows built from the real checkpoint (no model forward).

    python3 operations/kernels/kv4pro_realkv.py [--layers 0,2,3,30,31,59,60] [--tokens 2048]

Rows: real token embeddings -> RMSNorm(attn_norm) -> wkv (FP8 128x128 block weights, dequantized) -> RMSNorm(kv_norm)
("swa" rows, the SWA window cache content) and the same through the layer's compressor wkv + compressor norm without the
softmax pooling ("comp", an approximation of the c4/c128 rows; pooling averages 4 / 128 such rows). Layer 0's rows are
exact up to RoPE (its four mHC streams are copies of the embedding); deeper layers get embedding-derived inputs, so they
test the per-channel structure the norms and projections impose, not the residual stream's own statistics.
Reports relative L2 error of the 448 nope values for: the FP8 584-byte cell (UE8M0 per 64), the 4-bit cell (NVFP4
E4M3 per 16), and the 4-bit cell after a 64-wide Hadamard rotation (error measured back in the original basis), plus
the in-block dynamic range of the norm weights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv4pro_fp4 as K  # noqa: E402

CKPT = "/cache/deepseek-ai__DeepSeek-V4-Pro-0813"


def load(names):
    from safetensors import safe_open

    wm = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    out = {}
    by_file = {}
    for n in names:
        by_file.setdefault(wm[n], []).append(n)
    for f, ns in by_file.items():
        with safe_open(f"{CKPT}/{f}", framework="pt", device="cuda") as sf:
            for n in ns:
                out[n] = sf.get_tensor(n)
    return out


def deq_fp8(w, s):
    s = s.float()
    return (w.float().reshape(s.shape[0], 128, s.shape[1], 128) * s[:, None, :, None]).reshape(w.shape)


def rms(x, w, eps=1e-6):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def hadamard(n=64, device="cuda"):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def errs(x):
    x = x.float()
    nope = x[:, :448]
    rel = lambda y: ((y - nope).norm() / nope.norm()).item()  # noqa: E731
    e8 = rel(K.dequantize_fp8_paged_ref(K.quantize_fp8_paged_ref(x, 64), 64, x.shape[0])[:, :448])
    e4 = rel(K.dequantize_ref(K.quantize_ref(x))[:, :448])
    H = hadamard()
    xr = x.clone()
    xr[:, :448] = (nope.reshape(-1, 7, 64) @ H).reshape(-1, 448)
    back = K.dequantize_ref(K.quantize_ref(xr))[:, :448].reshape(-1, 7, 64) @ H.T
    e4h = rel(back.reshape(-1, 448))
    return e8, e4, e4h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,2,3,30,31,59,60")
    ap.add_argument("--tokens", type=int, default=2048)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    text = open(__file__).read() * 4
    ids = torch.tensor(tok(text)["input_ids"][: a.tokens], device="cuda")
    emb = load(["embed.weight"])["embed.weight"][ids]
    rows = []
    for L in [int(x) for x in a.layers.split(",")]:
        p = f"layers.{L}."
        names = [p + "attn_norm.weight", p + "attn.wkv.weight", p + "attn.wkv.scale", p + "attn.kv_norm.weight"]
        comp = [p + "attn.compressor.wkv.weight", p + "attn.compressor.norm.weight"]
        wm = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
        has_comp = all(n in wm for n in comp)
        t = load(names + (comp + ([p + "attn.compressor.wkv.scale"] if p + "attn.compressor.wkv.scale" in wm else [])
                          if has_comp else []))
        x = rms(emb, t[p + "attn_norm.weight"])
        wkv = t[p + "attn.wkv.weight"]
        wkv = deq_fp8(wkv, t[p + "attn.wkv.scale"]) if wkv.dtype == torch.float8_e4m3fn else wkv.float()
        kv = rms(x @ wkv.T, t[p + "attn.kv_norm.weight"])
        g = t[p + "attn.kv_norm.weight"].float()[:448].abs().reshape(28, 16)
        row = {"layer": L, "swa": errs(kv), "kv_norm_block_range_p90":
               torch.quantile((g.amax(1) / g.amin(1).clamp(min=1e-6)), 0.9).item(),
               "swa_row_absmax_over_rms": (kv[:, :448].abs().amax(1) / kv[:, :448].pow(2).mean(1).sqrt()).median().item()}
        if has_comp:
            cw = t[p + "attn.compressor.wkv.weight"]
            cs = t.get(p + "attn.compressor.wkv.scale")
            cw = deq_fp8(cw, cs) if cw.dtype == torch.float8_e4m3fn else cw.float()
            ckv = x @ cw.T
            ckv = rms(ckv[:, :512], t[p + "attn.compressor.norm.weight"])  # overlap c4 projects 2x; first half
            row["comp"] = errs(ckv)
        rows.append(row)
        print(json.dumps(row), flush=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    print(json.dumps({"gaussian": errs(torch.randn(a.tokens, 512, generator=g).cuda())}))


if __name__ == "__main__":
    main()
