"""Probe (1xH100 h100b): can FA4's CuTe-DSL SM90 forward (flash-attn-4 4.0.0b19, flash_attn/cute/flash_fwd_sm90.py)
run Gemma's hd-512 full-layer prefill when its head-dim gate (interface.py:102-105) is lifted and it gets a tile
that fits 227 KB of smem? rtxattn, coordinator 03:4x: "leave Triton, FA3-style wgmma + TMA, target <= 40 ms per 8K
chunk for the 10 full layers".

Plan (a): S = Q.K^T over 512 (32 k-steps of 16), output in two passes, one per 256-wide V half (head_dim_v 256). Q.K^T
is computed twice (1.5x FLOPs), but only the QK side generalizes past 256.
  - smem at tile (64, 48), 2 stages: Q 64x512 + K 2x48x512 + V 2x48x256 (bf16) = 64 + 96 + 48 KB = 208 KB.
  - registers per consumer warpgroup: S 64x48 + O 64x256 fp32 (as FA4's hdim 256).
Exactness vs an fp32 reference on a head subset; timing for an 8K chunk (32 q / 4 kv heads, causal).

    python3 operations/kernels/g31_fa4_hd512_probe.py
"""
import json
import statistics
import sys

import torch


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


def main():
    import flash_attn.cute.interface as fi
    fi._validate_head_dims = lambda *a, **k: None                       # lift the SM90 hdim <= 256 gate
    orig = fi._tile_size_fwd_sm90
    tiles = [(64, 48, True, False, False), (64, 48, True, True, False), (128, 32, True, False, False),
             (128, 32, True, True, False), (64, 48, True, False, True), (128, 32, True, False, True),
             (64, 48, True, True, True), (128, 32, True, True, True)]
    dev = "cuda"
    n, hq, hkv = 8192, 32, 4
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(1, n, hq, 512, device=dev, generator=g).to(torch.bfloat16)
    k = torch.randn(1, n, hkv, 512, device=dev, generator=g).to(torch.bfloat16)
    v = torch.randn(1, n, hkv, 512, device=dev, generator=g).to(torch.bfloat16)
    v_lo, v_hi = v[..., :256].contiguous(), v[..., 256:].contiguous()
    sm = 512 ** -0.5
    # fp32 reference on 2 q heads (kv head 0), the last 256 query rows
    rows = slice(n - 256, n)
    qs = q[0, rows, :2].float()
    ks, vs = k[0, :, 0].float(), v[0, :, 0].float()
    sc = torch.einsum("qhd,kd->hqk", qs, ks) * sm
    qpos = torch.arange(n - 256, n, device=dev)
    sc = sc.masked_fill(torch.arange(n, device=dev)[None, None, :] > qpos[None, :, None], float("-inf"))
    ref = torch.einsum("hqk,kd->qhd", torch.softmax(sc, -1), vs)
    out = []
    for tm, tn, rs, ov, pg in tiles:
        fi._tile_size_fwd_sm90 = (lambda tm=tm, tn=tn, rs=rs, ov=ov:
                                  lambda hd, hdv, c, l, sparse_block_size_q=None: (
                                      fi.FwdConfig(tm, tn, rs, ov) if hd > 256
                                      else orig(hd, hdv, c, l, sparse_block_size_q=sparse_block_size_q)))()
        row = {"tile": [tm, tn], "rs": rs, "overlap": ov, "pack_gqa": pg}
        try:
            def run():
                o_lo = fi.flash_attn_func(q, k, v_lo, softmax_scale=sm, causal=True, pack_gqa=pg)
                o_hi = fi.flash_attn_func(q, k, v_hi, softmax_scale=sm, causal=True, pack_gqa=pg)
                o_lo = o_lo[0] if isinstance(o_lo, tuple) else o_lo
                o_hi = o_hi[0] if isinstance(o_hi, tuple) else o_hi
                return o_lo, o_hi
            o_lo, o_hi = run()
            o = torch.cat([o_lo, o_hi], -1)
            got = o[0, rows, :2].float()
            row["rel_l2_vs_fp32"] = round(((got - ref).norm() / ref.norm()).item(), 6)
            row["ms_per_layer_2pass"] = round(timeit(run), 3)
            row["ms_10_layers"] = round(row["ms_per_layer_2pass"] * 10, 1)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(json.dumps(row), flush=True)
        out.append(row)
    json.dump(out, open("/root/g31_fa4_probe.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
