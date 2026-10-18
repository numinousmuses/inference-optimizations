"""End-to-end engine smoke for the K2.7 4-bit MLA KV hooks on one GPU with a small DeepseekV3-shaped dummy model.

The real checkpoint needs 8 GPUs; this boots SGLang v0.5.20 (patched by operations/patches/k27_kv4_hooks.py) on a
2-layer DeepseekV3ForCausalLM with the same attention geometry K2.7 has per TP8 rank (8 q heads, kv_lora 512, rope 64,
nope 128, v 128), random N(0, 0.02) weights written once (seed 1234; the engine's dummy loader draws +-1e-3 and
attention then leaves no trace in the logits), tokenspeed_mla, FP8 KV, page 64, decode CUDA graphs (prefill graph off),
and compares NUMINOUS_K27_KV4=0 vs 1 on the same prompts: pool size (log line), teacher-forced logprobs of a fixed
continuation (KL, top-1 agreement over positions), and greedy decode agreement. With random weights the numbers are not
the model's accuracy; they prove the FP4 pool + native dense kernel run inside the engine's prefill (MHA prefix
path), decode (graphs) and chunked-prefill (prefix > chunk) loops and stay close to FP8.

    python3 operations/kernels/k27_kv4_smoke.py --root /tmp/pkg/sglang   # a patched copy of the installed package
    python3 operations/kernels/k27_kv4_smoke.py --spec                  # + EAGLE3-shaped verify path (draft = same tiny model)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

TOK_SRC = "/cache/moonshotai__Kimi-K2.7-Code"
PORT = 31337


def make_model(dst: str, layers: int = 2, hidden: int = 1024, experts: int = 8, topk: int = 2, inter: int = 512) -> str:
    """A DeepseekV3 text config with K2.7's per-rank attention geometry; tokenizer files copied from the checkpoint."""
    os.makedirs(dst, exist_ok=True)
    cfg = {
        "architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3", "torch_dtype": "bfloat16",
        "hidden_size": hidden, "intermediate_size": 4 * hidden, "moe_intermediate_size": inter,
        "num_hidden_layers": layers, "first_k_dense_replace": 1, "moe_layer_freq": 1,
        "n_routed_experts": experts, "n_shared_experts": 1, "num_experts_per_tok": topk, "n_group": 1, "topk_group": 1,
        "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True,
        "num_attention_heads": 8, "num_key_value_heads": 8, "kv_lora_rank": 512, "q_lora_rank": 512,
        "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
        "vocab_size": 8192, "max_position_embeddings": 262144, "rms_norm_eps": 1e-6,
        "rope_theta": 50000.0, "rope_scaling": {"type": "yarn", "factor": 32, "original_max_position_embeddings": 4096,
                                                "beta_fast": 1.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0},
        "attention_bias": False, "hidden_act": "silu", "tie_word_embeddings": False, "bos_token_id": 0,
        "eos_token_id": 1, "initializer_range": 0.02, "num_nextn_predict_layers": 0,
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    if not os.path.exists(os.path.join(dst, "model.safetensors")):
        make_weights(dst, cfg)
    for fn in ("tokenizer_config.json", "tiktoken.model", "tokenization_kimi.py", "tool_declaration_ts.py",
               "special_tokens_map.json", "chat_template.jinja"):
        p = os.path.join(TOK_SRC, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, fn))
    return dst


def make_weights(dst: str, cfg: dict, std: float = 0.02, seed: int = 1234) -> None:
    """Random N(0, std) weights in the HF DeepseekV3 layout (norms = 1), so attention carries real signal: SGLang's
    dummy loader draws uniform(-1e-3, 1e-3), under which attention changes nothing the logits can show."""
    import torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    h, v = cfg["hidden_size"], cfg["vocab_size"]
    nh, ql, kl, rope, nope, vd = (cfg["num_attention_heads"], cfg["q_lora_rank"], cfg["kv_lora_rank"],
                                  cfg["qk_rope_head_dim"], cfg["qk_nope_head_dim"], cfg["v_head_dim"])

    def w(*shape):
        return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16)

    t = {"model.embed_tokens.weight": w(v, h), "lm_head.weight": w(v, h), "model.norm.weight": torch.ones(h, dtype=torch.bfloat16)}
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        t[p + "input_layernorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
        t[p + "post_attention_layernorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
        t[p + "self_attn.q_a_proj.weight"] = w(ql, h)
        t[p + "self_attn.q_a_layernorm.weight"] = torch.ones(ql, dtype=torch.bfloat16)
        t[p + "self_attn.q_b_proj.weight"] = w(nh * (nope + rope), ql)
        t[p + "self_attn.kv_a_proj_with_mqa.weight"] = w(kl + rope, h)
        t[p + "self_attn.kv_a_layernorm.weight"] = torch.ones(kl, dtype=torch.bfloat16)
        t[p + "self_attn.kv_b_proj.weight"] = w(nh * (nope + vd), kl)
        t[p + "self_attn.o_proj.weight"] = w(h, nh * vd)
        if L < cfg["first_k_dense_replace"]:
            inter = cfg["intermediate_size"]
            t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"], t[p + "mlp.down_proj.weight"] = w(inter, h), w(inter, h), w(h, inter)
        else:
            mi, e = cfg["moe_intermediate_size"], cfg["n_routed_experts"]
            t[p + "mlp.gate.weight"] = w(e, h)
            t[p + "mlp.gate.e_score_correction_bias"] = torch.zeros(e, dtype=torch.float32)
            for i in range(e):
                q = p + f"mlp.experts.{i}."
                t[q + "gate_proj.weight"], t[q + "up_proj.weight"], t[q + "down_proj.weight"] = w(mi, h), w(mi, h), w(h, mi)
            si = mi * cfg["n_shared_experts"]
            q = p + "mlp.shared_experts."
            t[q + "gate_proj.weight"], t[q + "up_proj.weight"], t[q + "down_proj.weight"] = w(si, h), w(si, h), w(h, si)
    save_file(t, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})


def boot(model: str, root: str, kv4: bool, log: str, extra: list[str], kv_dtype: str = "fp8_e4m3") -> subprocess.Popen:
    env = {**os.environ, "NUMINOUS_K27_KV4": "1" if kv4 else "0", "PYTHONPATH": str(pathlib.Path(root).parent)}
    # both arms without the prefill CUDA graph: its flashinfer fallback reads the pool as bf16 rows (see k27_kv4_engine)
    extra = ["--cuda-graph-backend-prefill", "disabled"] + list(extra)
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", model,
           "--trust-remote-code", "--tp", "1", "--kv-cache-dtype", kv_dtype,
           # tokenspeed_mla requires FP8 KV; the bf16 reference goes through its parent backend trtllm_mla
           "--attention-backend", "tokenspeed_mla" if kv_dtype == "fp8_e4m3" else "trtllm_mla",
           "--mem-fraction-static", "0.5", "--max-running-requests", "16", "--max-total-tokens", "262144", "--chunked-prefill-size", "512",
           "--context-length", "16384", "--port", str(PORT), "--host", "127.0.0.1", "--skip-server-warmup",
           "--cuda-graph-max-bs-decode", "16", "--dtype", "bfloat16"] + extra
    return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)


def wait_ready(p: subprocess.Popen, log: str, timeout: int = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            raise SystemExit(f"server died: see {log}\n" + open(log).read()[-4000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"server not ready in {timeout}s: {log}")


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def logprobs(ids: list[int], n_ctx: int) -> list[float]:
    """Teacher-forced logprob of every token after position n_ctx (return_logprob with logprob_start_len)."""
    r = post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 0, "temperature": 0},
                           "return_logprob": True, "logprob_start_len": n_ctx})
    lp = r["meta_info"]["input_token_logprobs"]
    return [x[0] for x in lp if x[0] is not None]


def top1(ids: list[int], n_ctx: int, k: int = 5) -> list[int]:
    r = post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 0, "temperature": 0},
                           "return_logprob": True, "logprob_start_len": n_ctx, "top_logprobs_num": k})
    tl = r["meta_info"]["input_top_logprobs"]
    return [t[0][1] for t in tl if t]


def greedy(ids: list[int], n: int) -> list[int]:
    r = post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": n, "temperature": 0}})
    return r["output_ids"] if "output_ids" in r else r["meta_info"].get("output_token_ids", [])


def run_arm(model: str, root: str, kv4: bool, prompts: list[list[int]], extra: list[str], out: dict,
            kv_dtype: str = "fp8_e4m3", name: str = "") -> None:
    name = name or ("kv4" if kv4 else "fp8")
    log = f"/root/k27_smoke_{name}.log"
    p = boot(model, root, kv4, log, extra, kv_dtype)
    try:
        wait_ready(p, log)
        text = open(log).read()
        pool = [ln for ln in text.splitlines() if "KV Cache is allocated" in ln or "numinous k27_kv4" in ln
                or "#token:" in ln or "max_total_num_tokens" in ln]
        res = {"pool_lines": pool[:6], "lp": [], "top1": [], "greedy": []}
        for ids in prompts:
            n_ctx = len(ids) - 64
            res["lp"].append(logprobs(ids, n_ctx))
            res["top1"].append(top1(ids, n_ctx))
            res["greedy"].append(greedy(ids[:n_ctx], 24))
        # a second pass on the same prompts: the prefix is cached now (radix hit -> prefix read door)
        res["lp_cached"] = [logprobs(ids, len(ids) - 64) for ids in prompts]
        res["activation"] = [ln for ln in open(log).read().splitlines() if "numinous k27_kv4" in ln][:4]
        out[name] = res
    finally:
        p.terminate()
        try:
            p.wait(30)
        except Exception:
            p.kill()
        subprocess.run("pkill -f 'sglang.launch_server' ; sleep 3", shell=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="patched sglang package dir (…/sglang)")
    ap.add_argument("--model", default="/root/k27_smoke_model")
    ap.add_argument("--spec", action="store_true", help="also run EAGLE3-shaped speculative decoding (draft = the same model)")
    ap.add_argument("--out", default="/root/k27_smoke.json")
    a = ap.parse_args()
    model = make_model(a.model)
    import random
    rng = random.Random(7)
    # prompts of 700 / 2100 / 5000 tokens: several 64-token pages, one crossing the 512 chunk many times
    prompts = [[rng.randrange(2, 8192) for _ in range(n)] for n in (700, 2100, 5000)]
    extra = []
    if a.spec:
        extra = ["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", model,
                 "--speculative-num-steps", "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"]
    out: dict = {"prompts": [len(p) for p in prompts], "spec": a.spec}
    # bf16 KV = the reference; fp8 = the served control; kv4 = the FP4 pool + native kernel
    run_arm(model, a.root, False, prompts, extra, out, kv_dtype="bf16", name="bf16")
    run_arm(model, a.root, False, prompts, extra, out)
    run_arm(model, a.root, True, prompts, extra, out)

    def compare(ref: str, arm: str) -> dict:
        r: dict = {}
        for key in ("lp", "lp_cached"):
            d, n = [], 0
            for a8, a4 in zip(out[ref][key], out[arm][key]):
                m = min(len(a8), len(a4))
                d += [abs(x - y) for x, y in zip(a8[:m], a4[:m])]   # |dlogprob| of the teacher-forced token per position
                n += m
            r[key] = {"positions": n, "mean_abs_dlogprob": sum(d) / max(n, 1), "max_abs_dlogprob": max(d) if d else None}
        agree = sum(int(x == y) for a8, a4 in zip(out[ref]["top1"], out[arm]["top1"]) for x, y in zip(a8, a4))
        tot = sum(min(len(a8), len(a4)) for a8, a4 in zip(out[ref]["top1"], out[arm]["top1"]))
        r["top1_agreement"] = agree / max(tot, 1)
        r["greedy_agreement_per_prompt"] = [sum(int(x == y) for x, y in zip(a8, a4)) / max(len(a8), 1)
                                            for a8, a4 in zip(out[ref]["greedy"], out[arm]["greedy"])]
        return r

    summary = {"fp8_vs_bf16": compare("bf16", "fp8"), "kv4_vs_bf16": compare("bf16", "kv4"), "kv4_vs_fp8": compare("fp8", "kv4"),
               "fp8_pool_lines": out["fp8"]["pool_lines"], "kv4_pool_lines": out["kv4"]["pool_lines"],
               "kv4_activation": out["kv4"]["activation"]}
    out["summary"] = summary
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
