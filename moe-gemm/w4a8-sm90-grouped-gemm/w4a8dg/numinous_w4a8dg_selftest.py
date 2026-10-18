"""Boot-time self-test + autotune of the k3amd W4A8 grouped GEMM (operations/kernels/w4a8dg) inside a serving node.

Why: the operator's budget (2026-10-05 20:1x) is one final 8xH200 run per model and no single-GPU microbench, so
w4a8dg v4 can only be measured inside that run. This module makes carrying it safe and makes the run its own
microbench:

  parent (the server process, W4AFp8MoEMethod.process_weights_after_loading, hooked by glm_w4a8_tiles_hooks
  'selftest'): once per process, before the KV pool is allocated, runs this file as a SUBPROCESS on the rank's GPU
  with a timeout (a hung or crashing kernel only costs the timeout), and stores the decision in
  os.environ["NUMINOUS_W4A8_DG_DECISION"] for the dispatcher in the same process.

  child (python3 -m sglang.srt.layers.moe.numinous_w4a8dg_selftest --run ...): synthetic int4 weights and FP8
  activations of the server's exact MoE shapes (E_local, w13 N x K, w2 K x N/2, top-k, EP size), the whole patched
  cutlass_w4a8_moe call on three paths:
    tiles     the served tile rules (no NUMINOUS_W4A8_DG)
    dg        both GEMMs on w4a8dg
    dg_silu   + GEMM1 with the fused silu*up -> FP8 epilogue
  exactness: dg must be bit-identical to tiles (both compute the exact per-group sums; measured rel_vs_stock 0.0 for
  v0-v3 on 1xH200); dg_silu within SILU_TOL of tiles (the served silu kernel's FP8 rounding, ex2.approx);
  autotune: CUDA-graph us per call at each m; min_m = the smallest m from which the path wins at every larger m.

The decision (and every timing) goes to stderr / the server log as one JSON line: '[numinous w4a8dg selftest] {...}'.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ENV = "NUMINOUS_W4A8_DG"
DECISION = "NUMINOUS_W4A8_DG_DECISION"
SILU_TOL = 2e-3
MS = (2048, 4096, 8192, 16384)
TAG = "[numinous w4a8dg selftest]"


def parse_spec(text: str) -> tuple[dict, dict]:
    kv, shapes = {}, {}
    for part in filter(None, (p.strip() for p in (text or "").split(";"))):
        key, val = part.split("=", 1)
        if "x" in key and key.replace("x", "").isdigit():
            n, k = (int(x) for x in key.split("x"))
            shapes[(n, k)] = int(val)
        else:
            kv[key] = val
    return kv, shapes


def child_spec(text: str) -> str:
    """The parent's spec for the child's dg runs: no selftest (no recursion), min_m=1 (every measured m)."""
    kv, shapes = parse_spec(text)
    kv.pop("selftest", None)
    kv.pop("selftest_timeout", None)
    kv["min_m"] = "1"
    return ";".join([f"{k}={v}" for k, v in kv.items()] + [f"{n}x{k}={c}" for (n, k), c in shapes.items()])


def path_spec(spec: str, dg: bool, silu: bool) -> str | None:
    """The child's NUMINOUS_W4A8_DG for one path: None = the tiles path; dg without silu drops the silu key."""
    if not dg:
        return None
    kv, shapes = parse_spec(spec)
    if not silu:
        kv.pop("silu", None)
    return ";".join([f"{k}={v}" for k, v in kv.items()] + [f"{n}x{k}={c}" for (n, k), c in shapes.items()])


def decide(us: dict, exact: bool, silu_rel: float | None, silu_on: bool) -> dict:
    """us[m] = {"tiles": .., "dg": .., "dg_silu": ..} (missing / None = failed). Pure: CPU-tested."""
    if not exact:
        return {"ok": False, "reason": "dg output is not bit-identical to the tiles path", "min_m": None,
                "silu": False}
    ms = sorted(us)

    def min_m_for(key):
        win = [us[m].get(key) is not None and us[m].get("tiles") is not None and us[m][key] < us[m]["tiles"]
               for m in ms]
        for i, m in enumerate(ms):
            if all(win[i:]):
                return m
        return None
    silu_ok = silu_on and silu_rel is not None and silu_rel <= SILU_TOL
    m_dg = min_m_for("dg")
    m_silu = min_m_for("dg_silu") if silu_ok else None
    use_silu = m_silu is not None and all(us[m]["dg_silu"] <= (us[m].get("dg") or float("inf"))
                                          for m in ms if m >= m_silu)
    min_m = m_silu if use_silu else m_dg
    return {"ok": min_m is not None, "min_m": min_m, "silu": bool(use_silu), "min_m_dg": m_dg, "min_m_silu": m_silu,
            "silu_rel": silu_rel, "reason": None if min_m is not None else "the tiles path wins at every m"}


def tokens_for(rows: int | None, ep: int, topk: int) -> int | None:
    """The dispatcher compares min_m with TOKENS (cutlass_w4a8_moe_mm: a.shape[0] // topk); the child measures at a
    number of local expert ROWS (tokens x topk / ep). Convert the decision to the dispatcher's unit."""
    return None if rows is None else max(1, rows * ep // max(1, topk))


def physical_gpu(index: int) -> str:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        ids = [v.strip() for v in vis.split(",") if v.strip()]
        if index < len(ids):
            return ids[index]
    return str(index)


def selftest_once(layer) -> dict | None:
    """Parent side (hooked into W4AFp8MoEMethod.process_weights_after_loading): run the child once per process when
    NUMINOUS_W4A8_DG has selftest=1. Never raises: any failure leaves a not-ok decision (the dispatcher keeps tiles)."""
    spec = os.environ.get(ENV, "")
    kv, shapes = parse_spec(spec)
    if kv.get("selftest") != "1" or DECISION in os.environ:
        return None
    t0 = time.time()
    try:
        w13, w2 = layer.w13_weight, layer.w2_weight
        e, n13, k = int(w13.shape[0]), int(w13.shape[1]), int(w13.shape[2]) * 2
        n2 = int(w2.shape[2]) * 2
        topk = int(getattr(layer, "top_k", 8) or 8)
        ep = int(getattr(layer, "moe_ep_size", 1) or 1)
        dev = w13.device.index if w13.device.index is not None else 0
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=physical_gpu(dev), PYTHONPATH=os.pathsep.join(sys.path),
                   **{ENV: child_spec(spec)})
        env.pop(DECISION, None)
        cmd = [sys.executable, "-m", "sglang.srt.layers.moe.numinous_w4a8dg_selftest", "--run", "--experts", str(e),
               "--n13", str(n13), "--k", str(k), "--n2", str(n2), "--topk", str(topk), "--ep", str(ep),
               "--silu", "1" if kv.get("silu") else "0"]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=int(kv.get("selftest_timeout", "240")))
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        dec = json.loads(lines[-1]) if lines else {"ok": False, "reason": f"child rc={p.returncode}",
                                                   "stderr": p.stderr[-1500:]}
    except subprocess.TimeoutExpired:
        dec = {"ok": False, "reason": "child timed out (kernel hang?)"}
    except Exception as ex:  # noqa: BLE001
        dec = {"ok": False, "reason": f"{type(ex).__name__}: {str(ex)[:300]}"}
    dec["elapsed_s"] = round(time.time() - t0, 1)
    os.environ[DECISION] = json.dumps(dec)
    print(TAG, summary_line(dec, os.environ.get("CUDA_VISIBLE_DEVICES", "")), file=sys.stderr, flush=True)
    print(TAG, "full", json.dumps(dec), file=sys.stderr, flush=True)
    return dec


def summary_line(dec: dict, gpu: str = "") -> str:
    """One compact line per rank for the gate's numinous-log capture: decision + us per path at each m."""
    us = dec.get("us") or {}
    cols = " ".join(f"m{m}=" + "/".join(str((us[m] or {}).get(k)) for k in ("tiles", "dg", "dg_silu"))
                    for m in sorted(us, key=lambda x: int(x)))
    return (f"gpu={dec.get('gpu', gpu)} ok={dec.get('ok')} min_m={dec.get('min_m')} silu={dec.get('silu')} "
            f"exact={dec.get('exact')} silu_rel={dec.get('silu_rel')} {cols} reason={dec.get('reason')} "
            f"t={dec.get('elapsed_s')}s")


def _timeit(fn, iters: int = 10) -> float:
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def child(a) -> dict:
    import types
    import torch
    import sglang.srt.layers.moe.cutlass_w4a8_moe as cm
    spec = os.environ.get(ENV, "")
    cm.get_parallel = lambda: types.SimpleNamespace(moe_ep_size=a.ep)
    dev, e, n13, k, n2 = "cuda", a.experts, a.n13, a.k, a.n2
    gen = torch.Generator(device=dev).manual_seed(97)
    w13 = torch.randint(-128, 128, (e, n13, k // 2), dtype=torch.int8, device=dev, generator=gen)
    w2 = torch.randint(-128, 128, (e, k, n2 // 2), dtype=torch.int8, device=dev, generator=gen)

    def scales(n, kk):     # alignment 1: [E, K/128, N]
        raw = (torch.rand(e, n, kk // 128, device=dev, generator=gen) * 0.004 + 0.001).to(torch.bfloat16)
        return raw.permute(0, 2, 1).contiguous()
    s13, s2 = scales(n13, k), scales(k, n2)
    full = lambda v: torch.full((e, 3), v, dtype=torch.int64, device=dev)  # noqa: E731
    a1, c1, a2_, c2 = full(k), full(n13), full(n2), full(k)
    offs = torch.empty(e + 1, dtype=torch.int32, device=dev)
    ps1 = torch.empty((e, 3), dtype=torch.int32, device=dev)
    ps2 = torch.empty((e, 3), dtype=torch.int32, device=dev)
    glob = e * a.ep

    def set_path(dg: bool, silu: bool):
        val = path_spec(spec, dg, silu)
        if val is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = val
        cm._nm_dg = None
    us, exact, silu_rel, out = {}, True, None, {}
    for m in a.ms:
        tokens = max(1, m * a.ep // a.topk)     # ~m local rows: tokens x topk / ep
        x = (torch.randn(tokens, k, device=dev, generator=gen) * 0.5).to(torch.bfloat16)
        logits = torch.randn(tokens, glob, device=dev, generator=gen)
        w, ids = torch.topk(torch.softmax(logits, -1), a.topk, dim=-1)
        w = (w / w.sum(-1, keepdim=True)).float()
        ids = ids.to(torch.int32)
        if a.ep > 1:
            ids = torch.where(ids < e, ids, torch.full_like(ids, -1))
        in13 = torch.tensor([float(x.abs().max()) / 448.0], dtype=torch.float32, device=dev)
        in2 = torch.tensor([0.05], dtype=torch.float32, device=dev)
        call = lambda: cm.cutlass_w4a8_moe(x, w13, w2, s13, s2, w, ids, a1, a1, c1, a2_, a2_, c2, c1, c2,  # noqa
                                           offs, ps1, ps2, in13, in2, routed_scaling_factor=1.0)
        row = {}
        for name, dg, silu in (("tiles", False, False), ("dg", True, False), ("dg_silu", True, True)):
            if silu and not a.silu:
                continue
            set_path(dg, silu)
            try:
                y = call().float()
                torch.cuda.synchronize()
                out[name] = y
                row[name] = round(_timeit(call), 1)
            except Exception as ex:  # noqa: BLE001
                row[name] = None
                row[f"{name}_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
        if "dg" in out and "tiles" in out:
            exact = exact and bool(torch.equal(out["dg"], out["tiles"]))
        elif row.get("dg") is None:
            exact = False
        if "dg_silu" in out and "tiles" in out:
            r = float((out["dg_silu"] - out["tiles"]).norm() / (out["tiles"].norm() + 1e-12))
            silu_rel = r if silu_rel is None else max(silu_rel, r)
        us[m] = row
        out.clear()
    set_path(False, False)
    dec = decide(us, exact, silu_rel, bool(a.silu))
    dec["min_m_rows"] = dec["min_m"]                    # us is keyed by local rows; the dispatcher wants tokens
    dec["min_m"] = tokens_for(dec["min_m"], a.ep, a.topk)
    dec.update(us=us, exact=exact, gpu=torch.cuda.get_device_name(), shapes={"experts": e, "w13": [n13, k],
                                                                              "w2": [k, n2], "topk": a.topk,
                                                                              "ep": a.ep})
    return dec


def main(argv: list) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--n13", type=int, default=4096)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--n2", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--silu", type=int, default=1)
    ap.add_argument("--ms", default=",".join(str(m) for m in MS))
    a = ap.parse_args(argv)
    a.ms = [int(x) for x in a.ms.split(",")]
    print(json.dumps(child(a)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
