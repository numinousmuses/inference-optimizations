"""Engine-path smoke of the SM120 DSA attention on ONE RTX PRO 6000 with a dummy-weight, 8-layer GLM-5.3-Flash.

The real checkpoint is 203 GB (does not fit one 96 GB card); the attention path under test does not need real
weights: the real config.json with 8 layers (KDA 0-2, DSA 3, KDA 4-6, DSA 7, + the MTP layer), 16 routed experts
instead of 288, bf16 (no quantization_config), `--load-format dummy` (seeded, identical weights every boot).
Every DSA dim is the served one: 64 heads, kv_lora 512, rope 0, indexer 32 x 128, top-k 2048, kpool 4.

Arms (one server each, sequential, same port): bf16 (Triton), fp8 fi_jit (FlashInfer widened), fp8 fi_stock
(FlashInfer 2048 + Triton tail), fp8 triton, fp4 triton; plus control_tilelang (stock tilelang, bf16) which is
EXPECTED to fail on SM120 (shared memory) and is recorded, not gated.
Per arm: boot log checks (engine line, pool line, CUDA graphs captured), 8 prompts of 300 / 3000 / 6000 tokens
(> 2048 exercises pooled selection and the 3-token kpool tail), greedy 48 tokens, top-20 logprobs of the first
position, then 32 concurrent 6000-token requests (decode batches under CUDA graphs), with MTP (EAGLE 3/1/4 over the
checkpoint's own NextN layer) in a second pass.

    python3 operations/kernels/glmflash_sm120_smoke.py prep  --out /root/flash_tiny     # config + tokenizer only
    python3 operations/kernels/glmflash_sm120_smoke.py run   --model /root/flash_tiny --arms all --out /root/rtxattn_smoke.json

Gates (preregistered, research/rtx-flash/attention.md section 8): every non-control arm boots with its engine line;
no NaN / crash / empty output; fi_jit vs fi_stock vs fp8 triton (same KV bytes, different kernels): first-position
top-20 logprob max |diff| <= 0.15 (greedy agreement recorded, not gated: dummy weights are uniform +-1e-3, logits are
near-uniform and argmax is tie noise); fp8/fp4 vs bf16 recorded; a repeat boot of fi_jit within 1e-2 of itself.
This smoke proves the engine path (hooks, pools, writers, CUDA graphs, kpool width, MTP) end to end; kernel numerics
are gated by test_glmflash_sm120_gpu.py (V4), not here.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import random
import signal
import subprocess
import sys
import time
import urllib.request

REPO_ID = "RadixArk/GLM-5.3-Flash-NVFP4"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
                   "special_tokens_map.json")
PORT = 30555
COMMON = ["--load-format", "dummy", "--tp", "1", "--mem-fraction-static", "0.80", "--context-length", "16384",
          "--max-running-requests", "32", "--chunked-prefill-size", "8192", "--mamba-ssm-dtype", "bfloat16",
          "--linear-attn-decode-backend", "triton", "--random-seed", "0", "--trust-remote-code",
          "--dsa-prefill-backend", "flashinfer_sparse_mla", "--dsa-decode-backend", "flashinfer_sparse_mla"]
MTP = ["--speculative-algorithm", "EAGLE", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
       "--speculative-num-draft-tokens", "4", "--max-mamba-cache-size", "200"]
ARMS = {
    "bf16": (["--kv-cache-dtype", "bfloat16"], {}),
    "fp8_fi_jit": (["--kv-cache-dtype", "fp8_e4m3"], {"NUMINOUS_RTXFLASH_FP8_KERNEL": "fi_jit"}),
    "fp8_fi_stock": (["--kv-cache-dtype", "fp8_e4m3"], {"NUMINOUS_RTXFLASH_FP8_KERNEL": "fi_stock"}),
    "fp8_triton": (["--kv-cache-dtype", "fp8_e4m3"], {"NUMINOUS_RTXFLASH_FP8_KERNEL": "triton"}),
    "fp4": (["--kv-cache-dtype", "fp8_e4m3"], {"NUMINOUS_RTXFLASH_KV": "fp4"}),
}
CONTROL = {"control_tilelang": ["--kv-cache-dtype", "bfloat16", "--dsa-prefill-backend", "tilelang",
                                "--dsa-decode-backend", "tilelang"]}


def trim_config(cfg: dict, layers: int = 8, experts: int = 16) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg.pop("quantization_config", None)
    t = cfg["text_config"]
    t["num_hidden_layers"] = layers
    for key in ("layer_types", "indexer_types", "mlp_layer_types"):
        if key in t:
            t[key] = t[key][:layers]
    lac = t["linear_attn_config"]
    lac["full_attn_layers"] = [i for i in lac["full_attn_layers"] if i < layers]
    lac["kda_layers"] = [i for i in lac["kda_layers"] if i < layers]
    assert lac["full_attn_layers"] == [3, 7], lac
    t["n_routed_experts"] = experts
    t["num_experts_per_tok"] = min(t["num_experts_per_tok"], experts)
    t["max_position_embeddings"] = 262144
    return cfg


def prep(out: pathlib.Path) -> None:
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(hf_hub_download(REPO_ID, "config.json")))
    (out / "config.json").write_text(json.dumps(trim_config(cfg), indent=1))
    for f in TOKENIZER_FILES:
        try:
            p = hf_hub_download(REPO_ID, f)
            (out / f).write_bytes(pathlib.Path(p).read_bytes())
        except Exception as e:  # optional files
            print(f"skip {f}: {e}")
    print("PREP_OK", out)


def _post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _wait(proc, log, limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def _prompts(n_tokens, k, seed):
    rng = random.Random(seed)
    return [[rng.randrange(1000, 150000) for _ in range(n_tokens)] for _ in range(k)]


def run_arm(model, name, flags, env_extra, mtp=False, logdir="/root"):
    env = {**os.environ, "NUMINOUS_RTXFLASH_ATTN": "1", **env_extra}
    log = pathlib.Path(logdir) / f"rtxattn_smoke_{name}{'_mtp' if mtp else ''}.log"
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--port", str(PORT), *COMMON, *flags,
           *(MTP if mtp else [])]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    rec = {"arm": name, "mtp": mtp, "cmd": " ".join(cmd), "log": str(log)}
    try:
        up = _wait(proc, log)
        text = log.read_text(errors="replace")
        rec["booted"] = up
        rec["engine_line"] = next((ln for ln in text.splitlines() if "numinous rtxflash attention" in ln), None)
        rec["pool_line"] = next((ln for ln in text.splitlines() if "numinous rtxflash pool" in ln), None)
        rec["graphs"] = "cuda graph: True" in text  # v0.5.20 batch log lines
        if not up:
            rec["tail"] = text[-3000:]
            return rec
        outs = []
        for n, seed in ((300, 1), (3000, 2), (6000, 3)):
            for ids in _prompts(n, 2, seed):
                r = _post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 48, "temperature": 0},
                                        "return_logprob": True, "top_logprobs_num": 20, "logprob_start_len": len(ids) - 1})
                meta = r["meta_info"]
                outs.append({"n": n, "tokens": r.get("output_ids") or meta.get("output_token_logprobs"),
                             "first_top": meta.get("output_top_logprobs", [None])[0], "text": r.get("text", "")[:60]})
        rec["outputs"] = outs
        big = _prompts(6000, 32, 9)
        t0 = time.time()
        with cf.ThreadPoolExecutor(32) as ex:
            rs = list(ex.map(lambda ids: _post("/generate", {"input_ids": ids, "sampling_params": {
                "max_new_tokens": 128, "temperature": 0}}), big))
        rec["concurrent32_s"] = round(time.time() - t0, 1)
        rec["concurrent32_ok"] = all(len(r.get("output_ids", r.get("text", ""))) > 0 for r in rs)
        text = log.read_text(errors="replace")
        rec["graphs"] = "cuda graph: True" in text
        rec["nan_in_log"] = "nan" in text.lower() and "NaN" in text
        rec["accept_len"] = [ln[-80:] for ln in text.splitlines() if "accept len" in ln.lower()][-3:]
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(60)
        except ProcessLookupError:  # the server already exited (e.g. the tilelang control's expected crash)
            pass
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        time.sleep(5)
    return rec


def _tokens(rec):
    return [o["tokens"] for o in rec.get("outputs", [])]


def compare(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return None
    agree = [sum(x == y for x, y in zip(p, q)) / max(1, len(p)) for p, q in zip(ta, tb)]
    diffs = []
    for oa, ob in zip(a["outputs"], b["outputs"]):
        if oa["first_top"] and ob["first_top"]:
            da = {t[1]: t[0] for t in oa["first_top"]}
            db = {t[1]: t[0] for t in ob["first_top"]}
            common = set(da) & set(db)
            diffs.append(max(abs(da[k] - db[k]) for k in common) if common else 99.0)
    return {"greedy_agreement": round(sum(agree) / len(agree), 3), "top20_max_abs_diff": round(max(diffs), 3)
            if diffs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "finish"])
    ap.add_argument("--out", default="/root/rtxattn_smoke.json")
    ap.add_argument("--model", default="/root/flash_tiny")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--no-mtp", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(pathlib.Path(a.out))
        return 0
    names = list(ARMS) if a.arms == "all" else a.arms.split(",")
    recs = {}
    if a.cmd == "finish":  # resume from <out>.partial: keep every finished arm, run only what is missing
        recs = json.loads(pathlib.Path(a.out + ".partial").read_text())
    partial = pathlib.Path(a.out + ".partial")

    def keep(key, rec):
        recs[key] = rec
        print(json.dumps({k: v for k, v in rec.items() if k not in ("outputs", "tail")}), flush=True)
        partial.write_text(json.dumps(recs))  # nothing is lost if a later arm dies

    for n in names:
        if n not in recs:
            keep(n, run_arm(a.model, n, *ARMS[n]))
    if "fp8_fi_jit" in recs and "fp8_fi_jit_repeat" not in recs:
        keep("fp8_fi_jit_repeat", run_arm(a.model, "fp8_fi_jit_repeat", *ARMS["fp8_fi_jit"]))
    if not a.no_mtp:
        for n in ("fp8_fi_jit", "fp4"):
            if n in names and n + "_mtp" not in recs:
                keep(n + "_mtp", run_arm(a.model, n, *ARMS[n], mtp=True))
    for n, flags in CONTROL.items():
        if n not in recs:
            keep(n, run_arm(a.model, n, flags, {"NUMINOUS_RTXFLASH_ATTN": "0"}))
    cmp_ = {}
    pairs = [("fp8_fi_jit", "fp8_fi_stock"), ("fp8_fi_jit", "fp8_triton"), ("fp8_fi_jit", "fp8_fi_jit_repeat"),
             ("bf16", "fp8_fi_jit"), ("bf16", "fp4"), ("bf16", "fp8_triton")]
    for x, y in pairs:
        if x in recs and y in recs:
            cmp_[f"{x}~{y}"] = compare(recs[x], recs[y])
    gates = {}
    for n in names + [k for k in recs if k.endswith("_mtp") or k.endswith("_repeat")]:
        r = recs[n]
        gates[f"{n}:boot+engine"] = bool(r.get("booted") and r.get("engine_line"))
        gates[f"{n}:outputs"] = bool(r.get("outputs")) and r.get("concurrent32_ok", False) and not r.get("nan_in_log")
    # Dummy weights are uniform +-1e-3 (weight_utils.initialize_dummy_weights): logits are near-uniform, so greedy
    # argmax is near-tie noise and only recorded; the gate is the first-position top-20 logprob distance.
    for key in ("fp8_fi_jit~fp8_fi_stock", "fp8_fi_jit~fp8_triton"):
        c = cmp_.get(key)
        gates[key] = bool(c and c["top20_max_abs_diff"] is not None and c["top20_max_abs_diff"] <= 0.15)
    c = cmp_.get("fp8_fi_jit~fp8_fi_jit_repeat")
    gates["fi_jit_repeatable"] = bool(c and c["top20_max_abs_diff"] is not None and c["top20_max_abs_diff"] <= 1e-2)
    doc = {"records": recs, "compare": cmp_, "gates": gates, "pass": all(gates.values())}
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"compare": cmp_, "gates": gates}, indent=1))
    print("SMOKE_PASS" if doc["pass"] else "SMOKE_FAIL")
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
