"""k3state drift study: how far a quantized KDA state drifts from fp32 over long sequences (GPU, no model load).

The KDA recurrence (SGLang IS_KDA op order, k3state_quant.kda_recurrent_ref) runs token by token for 12 heads (one
TP8 rank of K3) on synthetic inputs, with K3's gate: g = -5 * sigmoid(exp(A_log) * (a + dt_bias)) using the
checkpoint's own A_log / dt_bias for one KDA layer when the weights are on /cache (else a log-uniform stand-in).
Inputs are synthetic (q, k, v ~ N(0, 1), a ~ N(0, 1), beta = sigmoid(N(0, 1))); real activations are temporally
correlated, so this ranks formats and storage schemes, it does not replace the served accuracy gate.

Schemes (state quantized then dequantized at each event, recurrence otherwise fp32):
  two_tier_c<C>   k3state's layout: the state is read back from the quantized pool only at a prefill chunk / turn
                  boundary, every C tokens (C = 8192 and a pessimistic 2048)
  per_commit      the state requantized after every speculative commit (every 4 tokens, K3's DSpark accept ~3-4):
                  what a plain quantized pool without the fp32 working state would do; bf16 here is exactly the
                  stock --mamba-ssm-dtype bfloat16 behaviour
Metrics at 8k / 32k / 128k tokens: output relative error ||o - o_ref|| / ||o_ref|| over the 512 tokens before the
point ("before") and over the 256 tokens right after it ("after": for two_tier the state was just read back from the
quantized pool at exactly that point, so this is the worst window), and the state relative error at the point.
--gate-shift s adds s to the gate pre-activation: 0 uses a ~ N(0, 1); a negative shift (e.g. -4) makes most channels
decay slowly (long memory), the adversarial case for accumulated error.

    python3 operations/kernels/k3state_drift.py --tokens 131072 --out /root/k3state_drift.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3state_quant as kq  # noqa: E402

CKPT = "/cache/moonshotai__Kimi-K3"


def _gate_params(heads: int, layer: int, device):
    """A_log, dt_bias for the first `heads` heads of one KDA layer from the checkpoint, else a stand-in."""
    try:
        import glob

        from safetensors import safe_open
        idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
        want = {}
        for name, f in idx.items():
            if f".layers.{layer}." in name and (name.endswith("A_log") or name.endswith("dt_bias")):
                want[name] = f
        out = {}
        for name, f in want.items():
            with safe_open(os.path.join(CKPT, f), "pt") as sf:
                out["A_log" if name.endswith("A_log") else "dt_bias"] = sf.get_tensor(name).float()
        a_log = out["A_log"].reshape(-1)[:heads].to(device)            # per head
        dt = out["dt_bias"].reshape(-1, 128)[:heads].to(device)         # per head, per K
        return a_log, dt, f"checkpoint layer {layer} ({', '.join(sorted(want))})"
    except Exception as e:  # noqa: BLE001
        g = torch.Generator().manual_seed(layer)
        a_log = torch.log(torch.logspace(-2, 1, heads)).to(device)
        dt = (torch.randn(heads, 128, generator=g) * 0.5).to(device)
        return a_log, dt, f"stand-in ({type(e).__name__}: {str(e)[:80]})"


def _roundtrip(h, fmt):
    if fmt == "bf16":
        return h.to(torch.bfloat16).float()
    return kq.ref_dequant(kq.ref_quant(h, fmt), fmt)


def run(tokens: int, heads: int, layer: int, seed: int, gate_shift: float = 0.0, device="cuda") -> dict:
    a_log, dt_bias, src = _gate_params(heads, layer, device)
    checkpoints = [c for c in (8192, 32768, 131072) if c + 256 <= tokens]
    schemes = [("two_tier_c8192", 8192), ("two_tier_c2048", 2048), ("per_commit", 4)]
    fmts = ["bf16", "fp8", "fp4"]
    runs = {"fp32": None}
    for s, _ in schemes:
        for f in fmts:
            runs[f"{s}/{f}"] = (f, dict(schemes)[s])
    names = list(runs)
    n = len(names)
    g = torch.Generator(device=device).manual_seed(seed)
    h = torch.zeros(n, heads, 128, 128, device=device)
    scale = 128 ** -0.5
    out = {"tokens": tokens, "heads": heads, "gate_params": src, "seed": seed, "gate_shift": gate_shift,
           "checkpoints": checkpoints, "results": {k: {} for k in names if k != "fp32"}}
    win, after = 512, 256
    acc = {w: (torch.zeros(n, device=device), torch.zeros(n, device=device)) for w in ("before", "after")}
    decay_med = None
    t0 = time.time()
    for t in range(1, tokens + 1):
        q = torch.randn(heads, 128, generator=g, device=device)
        k = torch.randn(heads, 128, generator=g, device=device)
        v = torch.randn(heads, 128, generator=g, device=device)
        a = torch.randn(heads, 128, generator=g, device=device)
        beta = torch.sigmoid(torch.randn(heads, generator=g, device=device))
        gk = -5.0 * torch.sigmoid(torch.exp(a_log)[:, None] * (a + gate_shift + dt_bias))
        if decay_med is None:
            decay_med = float(torch.exp(gk).median())
            out["median_step_decay"] = decay_med
        # every scheme sees identical inputs: broadcast over the scheme axis
        o, hn = kq.kda_recurrent_ref(h.reshape(n * heads, 128, 128), q.repeat(n, 1), k.repeat(n, 1),
                                     v.repeat(n, 1), gk.repeat(n, 1), beta.repeat(n), scale)
        h = hn.reshape(n, heads, 128, 128)
        o = o.reshape(n, heads, 128)
        if t > 0:
            for i, name in enumerate(names):
                if name == "fp32":
                    continue
                fmt, every = runs[name]
                if t % every == 0 or (name.startswith("two_tier") and t in checkpoints):
                    h[i] = _roundtrip(h[i], fmt)
        for w, hit in (("before", any(c - win < t <= c for c in checkpoints)),
                       ("after", any(c < t <= c + after for c in checkpoints))):
            if hit:
                d = (o - o[0:1]).flatten(1)
                acc[w][0].add_((d * d).sum(1))
                acc[w][1].add_((o[0:1].flatten(1) ** 2).sum(1))
        if t in checkpoints:
            for i, name in enumerate(names):
                if name != "fp32":
                    out["results"][name][str(t)] = {
                        "out_rel_err_before": float((acc["before"][0][i] / acc["before"][1][i]).sqrt()),
                        "state_rel_err": float((h[i] - h[0]).norm() / h[0].norm())}
            acc["before"][0].zero_()
            acc["before"][1].zero_()
        for c in checkpoints:
            if t == c + after:
                for i, name in enumerate(names):
                    if name != "fp32":
                        out["results"][name][str(c)]["out_rel_err_after"] = float(
                            (acc["after"][0][i] / acc["after"][1][i]).sqrt())
                acc["after"][0].zero_()
                acc["after"][1].zero_()
                print(f"t={c} ({time.time() - t0:.0f} s): " + ", ".join(
                    f"{k} {v[str(c)]['out_rel_err_before']:.1e}/{v[str(c)]['out_rel_err_after']:.1e}"
                    for k, v in out["results"].items()), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=131072 + 256)
    ap.add_argument("--gate-shift", type=float, default=0.0)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    res = run(a.tokens, a.heads, a.layer, a.seed, a.gate_shift)
    txt = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt[-3000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
