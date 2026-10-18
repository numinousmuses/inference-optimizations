"""Accuracy gate for numerics arms: is an arm's output distribution the control's, within run-to-run noise?

Two measurements against SGLang's native /generate, each recorded per endpoint into a JSON artifact, then compared:

1. Teacher-forced logprobs. >= 200 fixed prompts (real coding-agent sessions from data/real_sessions*.json cut at an
   assistant turn; GSM8K question + reference solution when no session file is present), identical input_ids on both
   endpoints, `return_logprob` + `logprob_start_len` + `top_logprobs_num` over the last SCORE_TOKENS positions.
   Per scored position: KL(control || arm) with a top-k + tail-bucket approximation, top-1 agreement, and
   |delta logprob| of the actual next token. Reported as mean / p99 / max over all positions.
2. Exact match on 200 GSM8K test items (openai/grade-school-math, fetched once, sha256-pinned, cached), greedy,
   temperature 0, answers after "####". Paired comparison: discordant counts and a one-sided exact McNemar test.

Thresholds are constants (THRESHOLDS below, reasoning next to each). A control-vs-control artifact pair (`calibrate`)
measures run-to-run noise; when given, each threshold widens to NOISE_MULT x the noise if that is larger, because a
difference smaller than the engine's own run-to-run spread cannot be attributed to the arm.

    # live, both endpoints up
    python3 accuracy_gate.py run --control-url http://ctl:30000 --arm-url http://arm:30000 --model-dir /cache/<repo>
    # one endpoint at a time (how cell_gate boots arms), then compare
    python3 accuracy_gate.py record  --url http://127.0.0.1:30000 --model-dir /cache/<repo> --out control.json
    python3 accuracy_gate.py record  --url http://127.0.0.1:30000 --model-dir /cache/<repo> --out control2.json
    python3 accuracy_gate.py calibrate --a control.json --b control2.json --out noise.json
    python3 accuracy_gate.py compare --control control.json --arm arm.json [--noise noise.json]   # exit 1 = fail

cell_gate / verify entry point: `gate_arm(url, model_dir, arm, out_dir)` records `<out_dir>/accuracy_<arm>.json`,
then compares against `accuracy_control.json` (arms named control* other than control give the noise artifact).
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import math
import os
import pathlib
import random
import re
import sys
import time
import urllib.request
from typing import Callable, Optional

HERE = pathlib.Path(__file__).resolve().parent

N_PROMPTS = 200
SCORE_TOKENS = 256        # teacher-forced positions scored per prompt (the tail of an assistant turn)
MAX_PROMPT_TOKENS = 16384  # long enough to exercise sparse attention (GLM DSA top-2048, V4 compression) and FP8 KV
HEAD_TOKENS = 256          # kept from the start when a prompt is cut to MAX_PROMPT_TOKENS (template + task statement)
TOP_K = 20
GSM8K_N = 200
GSM8K_MAX_NEW = 512
CONCURRENCY = 32
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"  # 1319 items
GSM8K_INSTR = "\nSolve the problem step by step. End with a final line of the form '#### <number>'."
EPS = 1e-9

# Pass/fail thresholds (assumed priors; widened by measured noise, see effective_thresholds).
THRESHOLDS = {
    # Mean KL over scored tokens, nats. Benign quantization (FP8 weights / KV vs bf16) is reported around 1e-3 to
    # 5e-3 per token and batch-variant control-vs-control around 1e-4 to 1e-3; 0.01 flags a change an order of
    # magnitude outside the benign range while tolerating FP8-class arms.
    "kl_mean_max": 0.01,
    # p99 KL: the tail catches arms that are fine on average but break specific contexts (long-range attention,
    # outlier channels). A near-tie flipping costs up to ~0.1-0.2 nats; 0.25 allows that, not systematic damage.
    "kl_p99_max": 0.25,
    # Greedy top-1 agreement: near-ties flip under any numerics change; 97% is the floor seen for FP8-class changes.
    "top1_agree_min": 0.97,
    # p99 |delta logprob| of the actual next token (max is reported, not gated: one near-tie dominates it).
    "dlp_p99_max": 0.5,
    # Exact match: fail when the arm is worse by more than this many points AND the paired one-sided McNemar test
    # says the drop is unlikely under no difference (p < em_p_max). 2 points = 4 of 200 items.
    "em_drop_max_pts": 2.0,
    "em_p_max": 0.10,
    # ...and fail regardless of the test beyond this drop (10 items of 200).
    "em_drop_hard_pts": 5.0,
}
NOISE_MULT = 3.0


# ------------------------------------------------------------------------------------------------ rendering
def make_renderer(model_dir: str) -> Callable[[list, bool], list]:
    """messages, add_generation_prompt -> input_ids, in the model's own prompt format (DeepSeek V4's reference
    encoder in chat mode when the checkpoint ships it, else the chat template with thinking disabled)."""
    # V4 Pro ships encoding/encoding_dsv4.py, V4.1 Flash encoding/encoding.py (same encode_messages API)
    enc = next((p for p in (os.path.join(model_dir, "encoding", "encoding_dsv4.py"),
                            os.path.join(model_dir, "encoding", "encoding.py")) if os.path.exists(p)), "")
    if enc:
        import importlib.util
        spec = importlib.util.spec_from_file_location("encoding_dsv4", enc)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
        return lambda msgs, gen: tok(mod.encode_messages(msgs, thinking_mode="chat"),
                                     add_special_tokens=False)["input_ids"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def render(msgs, gen):
        text = tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"]
    return render


def ids_hash(ids: list) -> str:
    return hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:16]


def cut(ids: list, max_tokens: int = MAX_PROMPT_TOKENS, head: int = HEAD_TOKENS) -> list:
    return ids if len(ids) <= max_tokens else ids[:head] + ids[len(ids) - (max_tokens - head):]


# ------------------------------------------------------------------------------------------------ data
def sessions_file() -> Optional[str]:
    # ACC_SESSIONS pins the teacher-forced prompt source: the glob below takes the alphabetically first
    # /root/real_sessions*.json, which changes when other session files land on a box (MI355X, 2026-09-30: the 524K
    # file displaced real_sessions_t8_300.json and a kv4 arm was scored on prompts its control never saw)
    if os.environ.get("ACC_SESSIONS"):
        return os.environ["ACC_SESSIONS"]
    for pat in (str(HERE / "data" / "real_sessions.json"), "/root/real_sessions.json",
                str(HERE / "data" / "real_sessions*.json"), "/root/real_sessions*.json"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def sanitize_messages(messages: list, end: int) -> Optional[list]:
    """messages[:end+1] trimmed to start at the first user turn, so every chat template has a user query to anchor
    on (Qwen's raises `No user query found in messages` on an assistant-only slice; real coding sessions can begin
    with an assistant turn). Returns None when there is no user turn at or before `end` (the slice is unusable)."""
    first_user = next((k for k, m in enumerate(messages[:end + 1]) if m["role"] == "user"), None)
    if first_user is None or first_user > end:
        return None
    return [{"role": m["role"], "content": m["content"]} for m in messages[first_user:end + 1]]


def session_prompts(path: str, n: int = N_PROMPTS, seed: int = 0) -> list:
    """(id, messages ending in an assistant turn) pairs, spread round-robin over sessions, deterministic. Each slice
    is sanitized to start at a user turn (see sanitize_messages); assistant turns with no preceding user are skipped."""
    sessions = json.load(open(path))
    rng = random.Random(seed)
    per = []
    for si, s in enumerate(sessions):
        idx = [i for i, m in enumerate(s["messages"])
               if m["role"] == "assistant" and isinstance(m["content"], str) and len(m["content"]) >= 400
               and any(mm["role"] == "user" for mm in s["messages"][:i])]
        rng.shuffle(idx)
        per.append((si, s.get("id", str(si)), idx))
    out, r = [], 0
    while len(out) < n and any(len(ix) > r for _, _, ix in per):
        for si, sid, ix in per:
            if len(ix) > r and len(out) < n:
                i = ix[r]
                msgs = sanitize_messages(sessions[si]["messages"], i)
                if msgs is not None:
                    out.append((f"{sid}#{i}", msgs))
        r += 1
    return out


def gsm8k_items(cache_dir: Optional[str] = None, n: int = GSM8K_N) -> list:
    """First n items of the GSM8K test split, downloaded once to the cache and checked against GSM8K_SHA256."""
    cache_dir = cache_dir or os.environ.get("ACCURACY_GATE_CACHE") or \
        os.path.join(os.environ.get("GATE_CACHE", "/cache"), "accuracy_gate")
    path = os.path.join(cache_dir, "gsm8k_test.jsonl")
    if not os.path.exists(path):
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            path = os.path.join(os.path.expanduser("~/.cache/numinous/accuracy_gate"), "gsm8k_test.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with urllib.request.urlopen(GSM8K_URL, timeout=120) as r:
                raw = r.read()
            if hashlib.sha256(raw).hexdigest() != GSM8K_SHA256:
                raise ValueError("GSM8K download does not match the pinned sha256")
            with open(path + ".tmp", "wb") as f:
                f.write(raw)
            os.replace(path + ".tmp", path)
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != GSM8K_SHA256:
        raise ValueError(f"{path} does not match the pinned sha256")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return [{"id": i, "question": r["question"], "solution": r["answer"], "answer": gold_answer(r["answer"])}
            for i, r in enumerate(rows[:n])]


def gold_answer(solution: str) -> str:
    return normalize_number(solution.split("####")[-1])


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_number(s: str) -> Optional[str]:
    m = NUM.findall(s.replace("$", ""))
    if not m:
        return None
    v = m[0].replace(",", "")
    try:
        f = float(v)
    except ValueError:
        return None
    if not math.isfinite(f):  # a runaway digit string (>308 digits) floats to inf: int(inf) raised and lost a whole arm's record
        return None
    return str(int(f)) if f == int(f) else repr(f)


def extract_answer(text: str) -> Optional[str]:
    """Number after the last '####'; else the last number in the text."""
    if "####" in text:
        got = normalize_number(text.rsplit("####", 1)[1])
        if got is not None:
            return got
    nums = NUM.findall(text.replace("$", ""))
    return normalize_number(nums[-1]) if nums else None


def tf_prompts(render, n: int = N_PROMPTS, max_tokens: int = MAX_PROMPT_TOKENS, score: int = SCORE_TOKENS,
               cache_dir: Optional[str] = None, sessions: Optional[str] = "auto") -> tuple[str, list]:
    """(source, [{"id", "ids", "start"}]) for the teacher-forced comparison."""
    path = sessions_file() if sessions == "auto" else sessions
    if path:
        src, pairs = f"sessions:{os.path.basename(path)}", session_prompts(path, n)
    else:
        items = gsm8k_items(cache_dir, max(n, GSM8K_N) + n)[-n:]  # disjoint from the EM items when n <= 1119
        src = "gsm8k_reference_solutions"
        pairs = [(f"gsm8k#{it['id']}", [{"role": "user", "content": it["question"] + GSM8K_INSTR},
                                        {"role": "assistant", "content": it["solution"]}]) for it in items]
    out = []
    for pid, msgs in pairs:
        try:
            rendered = render(msgs, False)
        except Exception as e:  # noqa: BLE001  a template that still rejects a slice must not fail the whole gate
            print(f"[accuracy] skipped prompt {pid}: render failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            continue
        ids = cut(rendered, max_tokens)
        s = min(score, len(ids) - 2)
        # SGLang returns entries for tokens start.. with the first one empty (no logits before it), so scoring the
        # last s tokens needs start = len - s - 1
        out.append({"id": pid, "ids": ids, "start": len(ids) - s - 1})
    return src, out


# ------------------------------------------------------------------------------------------------ recording
async def _post(http, url: str, body: dict, sem, retries: int = 2) -> dict:
    import aiohttp
    async with sem:
        for attempt in range(retries + 1):
            try:
                async with http.post(url + "/generate", json=body) as r:
                    if r.status != 200:
                        raise RuntimeError(f"http {r.status}: {(await r.text())[:300]}")
                    return await r.json()
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if attempt == retries:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _pairs(entries) -> list:
    """SGLang logprob entries ([lp, id, text?] or None) -> [(id, lp)] (None -> [])."""
    return [(int(e[1]), float(e[0])) for e in (entries or []) if e is not None and e[0] is not None]


async def _tf_one(http, url, p, top_k, sem) -> dict:
    body = {"input_ids": p["ids"], "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "logprob_start_len": p["start"], "top_logprobs_num": top_k}
    mi = (await _post(http, url, body, sem))["meta_info"]
    tok = mi["input_token_logprobs"]
    top = mi.get("input_top_logprobs") or [None] * len(tok)
    rec = {"id": p["id"], "hash": ids_hash(p["ids"]), "len": len(p["ids"]), "start": p["start"],
           "t": [], "lp": [], "top": []}
    for e, tp in zip(tok, top):
        if e is None or e[0] is None or not tp:
            continue
        rec["t"].append(int(e[1]))
        rec["lp"].append(round(float(e[0]), 6))
        rec["top"].append([[i, round(lp, 6)] for i, lp in _pairs(tp)])
    expect = p["ids"][p["start"]:]
    if rec["t"] and rec["t"] != expect[len(expect) - len(rec["t"]):]:
        raise RuntimeError(f"{p['id']}: server scored different tokens than sent (logprob alignment changed?)")
    return rec


async def _em_one(http, url, it, ids, sem, max_new) -> dict:
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}
    text = (await _post(http, url, body, sem)).get("text", "")
    pred = extract_answer(text)
    return {"id": it["id"], "hash": ids_hash(ids), "gold": it["answer"], "pred": pred,
            "correct": pred is not None and pred == it["answer"], "text": text[-300:]}


# ---------------------------------------------------------------------------------------------- vLLM endpoint
# vLLM has no /generate; teacher-forced logprobs come from /v1/completions with echo + prompt_logprobs, and
# generation from /v1/completions. `prompt` is sent as raw token ids (vLLM accepts List[int]) so the input matches
# the sglang path byte-for-byte. DiffusionGemma is not autoregressive, so teacher-forced logprobs are undefined for
# it (record() sets diffusion=True -> EM task check only).
def detect_engine(url: str) -> str:
    """'sglang' if the server answers /get_model_info, else 'vllm'."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/get_model_info", timeout=15) as r:
            if r.status == 200:
                return "sglang"
    except Exception:  # noqa: BLE001
        pass
    return "vllm"


def vllm_model_name(url: str) -> str:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=15) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:  # noqa: BLE001
        return "m"  # cell_gate launches vLLM with --served-model-name m


def is_diffusion(model_dir: str) -> bool:
    return "diffusion" in os.path.basename(str(model_dir).rstrip("/")).lower()


def parse_vllm_prompt_logprobs(choice: dict, ids: list, start: int, top_k: int) -> dict:
    """A vLLM /v1/completions choice (echo + prompt_logprobs) -> the sglang-shaped rec (t / lp / top). Pure, so it is
    unit-testable without a server. prompt_logprobs[pos] is the distribution for prompt token `pos` given 0..pos-1
    ({token_id_str: {logprob, rank, decoded_token}}); [0] is null."""
    plp = choice.get("prompt_logprobs") or []
    rec: dict = {"t": [], "lp": [], "top": []}
    for pos in range(start, len(ids)):
        entry = plp[pos] if pos < len(plp) else None
        if not entry or str(ids[pos]) not in entry:
            continue
        rec["t"].append(int(ids[pos]))
        rec["lp"].append(round(float(entry[str(ids[pos])]["logprob"]), 6))
        top = sorted(((int(k), float(v["logprob"])) for k, v in entry.items()), key=lambda x: -x[1])[:top_k]
        rec["top"].append([[i, round(lp, 6)] for i, lp in top])
    return rec


async def _post_vllm(http, url: str, path: str, body: dict, sem, retries: int = 2) -> dict:
    import aiohttp
    async with sem:
        for attempt in range(retries + 1):
            try:
                async with http.post(url + path, json=body) as r:
                    if r.status != 200:
                        raise RuntimeError(f"http {r.status}: {(await r.text())[:300]}")
                    return await r.json()
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if attempt == retries:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


async def _vllm_tf_one(http, url, p, top_k, sem, model) -> dict:
    body = {"model": model, "prompt": p["ids"], "max_tokens": 1, "temperature": 0, "echo": True,
            "prompt_logprobs": top_k}
    ch = (await _post_vllm(http, url, "/v1/completions", body, sem))["choices"][0]
    rec = parse_vllm_prompt_logprobs(ch, p["ids"], p["start"], top_k)
    rec.update(id=p["id"], hash=ids_hash(p["ids"]), len=len(p["ids"]), start=p["start"])
    expect = p["ids"][p["start"]:]
    if rec["t"] and rec["t"] != expect[len(expect) - len(rec["t"]):]:
        raise RuntimeError(f"{p['id']}: vLLM scored different tokens than sent (prompt_logprobs alignment?)")
    return rec


async def _vllm_em_one(http, url, it, ids, sem, max_new, model, diffusion: bool = False) -> dict:
    # diffusion servers reject temperature/min_p/seed/etc (BadRequest); send only model+prompt+max_tokens for them.
    body = {"model": model, "prompt": ids, "max_tokens": max_new}
    if not diffusion:
        body["temperature"] = 0
    text = (await _post_vllm(http, url, "/v1/completions", body, sem))["choices"][0].get("text", "")
    pred = extract_answer(text)
    return {"id": it["id"], "hash": ids_hash(ids), "gold": it["answer"], "pred": pred,
            "correct": pred is not None and pred == it["answer"], "text": text[-300:]}


async def _record(url: str, prompts: list, em: list, top_k: int, concurrency: int, max_new: int,
                  engine: str = "sglang", model_name: str = "m", do_tf: bool = True,
                  diffusion: bool = False) -> tuple:
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        if not do_tf:
            tf = []
        elif engine == "vllm":
            tf = await asyncio.gather(*[_vllm_tf_one(http, url, p, top_k, sem, model_name) for p in prompts])
        else:
            tf = await asyncio.gather(*[_tf_one(http, url, p, top_k, sem) for p in prompts])
        if engine == "vllm":
            ems = await asyncio.gather(*[_vllm_em_one(http, url, it, ids, sem, max_new, model_name, diffusion)
                                         for it, ids in em])
        else:
            ems = await asyncio.gather(*[_em_one(http, url, it, ids, sem, max_new) for it, ids in em])
    return tf, ems


def _server_info(url: str) -> dict:
    info = {}
    for path in ("/get_model_info", "/get_server_info"):
        try:
            with urllib.request.urlopen(url + path, timeout=30) as r:
                j = json.loads(r.read())
            info[path.strip("/")] = {k: j.get(k) for k in ("model_path", "served_model_name", "quantization",
                                                         "kv_cache_dtype", "moe_runner_backend", "version") if k in j}
        except Exception as e:  # noqa: BLE001  informational only
            info[path.strip("/")] = f"unavailable: {type(e).__name__}"
    return info


def record(url: str, render, out: Optional[str] = None, n_prompts: int = N_PROMPTS, gsm8k_n: int = GSM8K_N,
           top_k: int = TOP_K, max_prompt_tokens: int = MAX_PROMPT_TOKENS, score_tokens: int = SCORE_TOKENS,
           concurrency: int = CONCURRENCY, cache_dir: Optional[str] = None, sessions: Optional[str] = "auto",
           label: str = "", max_new: int = GSM8K_MAX_NEW, engine: Optional[str] = None,
          diffusion: bool = False, skip_tf: str = "") -> dict:
    """Run the measurements against one endpoint; returns (and optionally writes) the artifact. `engine` auto-detects
    (sglang/vllm) from the server when None; `diffusion` skips teacher-forced logprobs (undefined for a diffusion
    model) and records the EM task check only. `skip_tf` (a reason) does the same for endpoints that cannot return
    prompt logprobs, e.g. an in-box PD front: the prompt is scored on the prefill server, the response is the decode
    server's."""
    t0 = time.time()
    url = url.rstrip("/")
    engine = engine or detect_engine(url)
    model_name = vllm_model_name(url) if engine == "vllm" else "m"
    do_tf = not (diffusion or skip_tf)
    if do_tf:
        src, prompts = tf_prompts(render, n_prompts, max_prompt_tokens, score_tokens, cache_dir, sessions)
    else:
        src, prompts = ("skipped:diffusion" if diffusion else f"skipped:{skip_tf}"), []
    items = gsm8k_items(cache_dir, gsm8k_n) if gsm8k_n else []
    em = [(it, render([{"role": "user", "content": it["question"] + GSM8K_INSTR}], True)) for it in items]
    tf, ems = asyncio.run(_record(url, prompts, em, top_k, concurrency, max_new, engine, model_name, do_tf,
                                  diffusion))
    art = {"meta": {"label": label, "url": url, "engine": engine,
                    "tf_skipped": ("diffusion: teacher-forced logprobs are undefined" if diffusion else (skip_tf or None)),
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "prompts_source": src, "n_prompts": len(tf), "top_k": top_k, "score_tokens": score_tokens,
                    "max_prompt_tokens": max_prompt_tokens, "gsm8k_n": len(ems), "server": _server_info(url),
                    "seconds": round(time.time() - t0, 1)},
           "tf": tf, "em": ems}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out + ".tmp", "w") as f:
            json.dump(art, f, separators=(",", ":"))
        os.replace(out + ".tmp", out)
    return art


# ------------------------------------------------------------------------------------------------ comparison
def token_kl(top_p: list, top_q: list) -> float:
    """KL(P || Q) from top-k lists [[id, logprob]]: exact over P's top-k, with P's and Q's remaining mass in one tail
    bucket. A P-top-k token missing from Q's top-k gets Q's k-th logprob (an upper bound on its true probability),
    so the estimate is conservative on the tail bucket and never overstates a missing token's penalty."""
    q = {i: lp for i, lp in top_q}
    q_floor = min(q.values()) if q else math.log(EPS)
    kl, p_mass, q_mass = 0.0, 0.0, 0.0
    for i, lp in top_p:
        p = math.exp(lp)
        lq = q.get(i, q_floor)
        kl += p * (lp - lq)
        p_mass += p
        q_mass += math.exp(lq)
    p_tail, q_tail = max(1.0 - p_mass, EPS), max(1.0 - q_mass, EPS)
    kl += p_tail * math.log(p_tail / q_tail)
    return max(kl, 0.0)


def _quantile(v: list, q: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(X >= b) for X ~ Binomial(b + c, 1/2): chance of at least b control-only wins if neither side is better."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def compare_artifacts(a: dict, b: dict) -> dict:
    """Metrics of B against A (A = control)."""
    ta = {r["id"]: r for r in a["tf"]}
    kls, agree, dlp, n_pos, mismatched = [], 0, [], 0, []
    for rb in b["tf"]:
        ra = ta.get(rb["id"])
        if ra is None or ra["hash"] != rb["hash"] or ra["t"] != rb["t"]:
            mismatched.append(rb["id"])
            continue
        for pa, pb, la, lb in zip(ra["top"], rb["top"], ra["lp"], rb["lp"]):
            if not pa or not pb:
                continue
            kls.append(token_kl(pa, pb))
            agree += int(max(pa, key=lambda x: x[1])[0] == max(pb, key=lambda x: x[1])[0])
            dlp.append(abs(la - lb))
            n_pos += 1
    ea = {r["id"]: r for r in a["em"]}
    both = [(ea[r["id"]], r) for r in b["em"] if r["id"] in ea and ea[r["id"]]["hash"] == r["hash"]]
    n_em = len(both)
    acc_a = sum(x["correct"] for x, _ in both) / n_em if n_em else float("nan")
    acc_b = sum(y["correct"] for _, y in both) / n_em if n_em else float("nan")
    ctl_only = sum(x["correct"] and not y["correct"] for x, y in both)
    arm_only = sum(y["correct"] and not x["correct"] for x, y in both)
    return {
        "tf": {"prompts": len(b["tf"]) - len(mismatched), "positions": n_pos, "mismatched_prompts": mismatched[:10],
               "kl_mean": sum(kls) / n_pos if n_pos else float("nan"), "kl_p99": _quantile(kls, 0.99),
               "kl_max": max(kls) if kls else float("nan"),
               "top1_agree": agree / n_pos if n_pos else float("nan"),
               "dlp_mean": sum(dlp) / n_pos if n_pos else float("nan"), "dlp_p99": _quantile(dlp, 0.99),
               "dlp_max": max(dlp) if dlp else float("nan")},
        "em": {"items": n_em, "acc_control": acc_a, "acc_arm": acc_b,
               "drop_pts": 100 * (acc_a - acc_b) if n_em else float("nan"),
               "control_only_correct": ctl_only, "arm_only_correct": arm_only,
               "answer_changed": sum(x["pred"] != y["pred"] for x, y in both),
               "mcnemar_p_arm_worse": mcnemar_one_sided(ctl_only, arm_only)},
    }


def _noise_metrics(noise: Optional[dict]) -> Optional[dict]:
    """The control-vs-control metrics from either a full compare output (metrics nested) or the bare metrics."""
    if not noise:
        return None
    return noise.get("metrics", noise)


def effective_thresholds(noise: Optional[dict] = None) -> dict:
    """THRESHOLDS widened to NOISE_MULT x the control-vs-control metric where that is looser."""
    t = dict(THRESHOLDS)
    if not noise:
        return t
    noise = _noise_metrics(noise)
    tf, em = noise["tf"], noise["em"]
    t["kl_mean_max"] = max(t["kl_mean_max"], NOISE_MULT * tf["kl_mean"])
    t["kl_p99_max"] = max(t["kl_p99_max"], NOISE_MULT * tf["kl_p99"])
    t["top1_agree_min"] = min(t["top1_agree_min"], 1.0 - NOISE_MULT * (1.0 - tf["top1_agree"]))
    t["dlp_p99_max"] = max(t["dlp_p99_max"], NOISE_MULT * tf["dlp_p99"])
    t["em_drop_max_pts"] = max(t["em_drop_max_pts"], NOISE_MULT * abs(em["drop_pts"]))
    return t


def verdict(metrics: dict, noise: Optional[dict] = None, tf_skipped: bool = False) -> dict:
    noise = _noise_metrics(noise)
    t = effective_thresholds(noise)
    tf, em = metrics["tf"], metrics["em"]
    checks: dict = {}
    note = None
    if tf_skipped:
        note = "teacher-forced logprobs are undefined for a diffusion model; EM task check only"
    else:
        checks.update({
            "inputs_identical": not tf["mismatched_prompts"] and tf["positions"] > 0,
            "kl_mean": tf["kl_mean"] <= t["kl_mean_max"],
            "kl_p99": tf["kl_p99"] <= t["kl_p99_max"],
            "top1_agree": tf["top1_agree"] >= t["top1_agree_min"],
            "dlp_p99": tf["dlp_p99"] <= t["dlp_p99_max"],
        })
    if em["items"]:
        significant = em["drop_pts"] > t["em_drop_max_pts"] and em["mcnemar_p_arm_worse"] < t["em_p_max"]
        checks["em_paired"] = not significant and em["drop_pts"] <= t["em_drop_hard_pts"]
    within = None
    if noise and not tf_skipped:
        within = (tf["kl_mean"] <= 1.5 * max(noise["tf"]["kl_mean"], EPS)
                  and tf["top1_agree"] >= 1.0 - 1.5 * (1.0 - noise["tf"]["top1_agree"]))
    return {"pass": bool(checks) and all(checks.values()), "checks": checks, "thresholds": t, "within_noise": within,
            "noise_given": noise is not None, "tf_skipped": tf_skipped, "note": note}


def compare(control: dict, arm: dict, noise: Optional[dict] = None) -> dict:
    m = compare_artifacts(control, arm)
    tf_skipped = bool(arm["meta"].get("tf_skipped")) or bool(control["meta"].get("tf_skipped"))
    return {"control": control["meta"].get("label") or control["meta"]["url"],
            "arm": arm["meta"].get("label") or arm["meta"]["url"], "metrics": m,
            "verdict": verdict(m, noise, tf_skipped=tf_skipped)}


def calibrate(a: dict, b: dict) -> dict:
    """Control-vs-control metrics: the noise floor the thresholds widen to."""
    m = compare_artifacts(a, b)
    return {"a": a["meta"].get("label"), "b": b["meta"].get("label"), **m,
            "suggested_thresholds": effective_thresholds(m)}


# ------------------------------------------------------------------------------------------------ cell_gate hook
def gate_arm(url: str, model_dir: str, arm: str, out_dir: str, control_arm: str = "control", **kw) -> dict:
    """Record this arm; compare with the control artifact in out_dir if present.

    Arms named like the control (control2, control_b) are compared as noise and write accuracy_noise.json, which later
    comparisons pick up. Returns a small dict for the gate's per-arm result."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"accuracy_{arm}.json")
    art = record(url, make_renderer(model_dir), path, label=arm, diffusion=is_diffusion(model_dir), **kw)
    res = {"artifact": path, "seconds": art["meta"]["seconds"], "prompts_source": art["meta"]["prompts_source"],
           "gsm8k_acc": (sum(r["correct"] for r in art["em"]) / len(art["em"])) if art["em"] else None}
    ctl_path = os.path.join(out_dir, f"accuracy_{control_arm}.json")
    if arm == control_arm or not os.path.exists(ctl_path):
        return res
    ctl = json.load(open(ctl_path))
    noise_path = os.path.join(out_dir, "accuracy_noise.json")
    if arm.startswith(control_arm):
        noise = calibrate(ctl, art)
        json.dump(noise, open(noise_path, "w"), indent=1)
        res["noise"] = {k: noise[k] for k in ("tf", "em")}
        return res
    noise = json.load(open(noise_path)) if os.path.exists(noise_path) else None
    cmp = compare(ctl, art, noise)
    json.dump(cmp, open(os.path.join(out_dir, f"accuracy_compare_{arm}.json"), "w"), indent=1)
    res.update(verdict=cmp["verdict"], metrics=cmp["metrics"])
    return res


# ------------------------------------------------------------------------------------------------ CLI
def _load(p):
    return json.load(open(p)) if p else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-dir", required=True)
    common.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    common.add_argument("--gsm8k-n", type=int, default=GSM8K_N)
    common.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    common.add_argument("--score-tokens", type=int, default=SCORE_TOKENS)
    common.add_argument("--concurrency", type=int, default=CONCURRENCY)
    common.add_argument("--cache-dir", default=None)
    common.add_argument("--sessions", default="auto", help="session file, 'auto', or 'none' (GSM8K prompts)")
    r = sub.add_parser("record", parents=[common])
    r.add_argument("--url", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="")
    rn = sub.add_parser("run", parents=[common])
    rn.add_argument("--control-url", required=True)
    rn.add_argument("--arm-url", required=True)
    rn.add_argument("--out-dir", default=".")
    rn.add_argument("--noise", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--control", required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--noise", default=None)
    c.add_argument("--out", default=None)
    k = sub.add_parser("calibrate")
    k.add_argument("--a", required=True)
    k.add_argument("--b", required=True)
    k.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd in ("record", "run"):
        kw = dict(n_prompts=a.n_prompts, gsm8k_n=a.gsm8k_n, max_prompt_tokens=a.max_prompt_tokens,
                  score_tokens=a.score_tokens, concurrency=a.concurrency, cache_dir=a.cache_dir,
                  sessions=None if a.sessions == "none" else a.sessions)
        render = make_renderer(a.model_dir)
    if a.cmd == "record":
        art = record(a.url, render, a.out, label=a.label, **kw)
        print(json.dumps(art["meta"], indent=1))
        return 0
    if a.cmd == "run":
        ctl = record(a.control_url, render, os.path.join(a.out_dir, "accuracy_control.json"), label="control", **kw)
        arm = record(a.arm_url, render, os.path.join(a.out_dir, "accuracy_arm.json"), label="arm", **kw)
        res = compare(ctl, arm, _load(a.noise))
    elif a.cmd == "compare":
        res = compare(_load(a.control), _load(a.arm), _load(a.noise))
    else:
        res = calibrate(_load(a.a), _load(a.b))
    if getattr(a, "out", None):
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0 if res.get("verdict", {"pass": True})["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
