"""Gate a candidate against a shipped configuration, on real agent traffic at the shape the lane is sold at.

Every arm starts from one registry cell exactly as a node runs it (image digest, patches, flags, env: see
registry_image.py) and adds one delta: flags added or removed, environment variables, or kernel overlay files
installed over the engine package. The control arm adds nothing. So a result is always "the shipped lane" against
"the shipped lane plus this one change", which is the only comparison that says what a customer would get.

Traffic is recorded coding-agent sessions (Harbor trajectories) at the sold shape, three growing turns, rendered
by the model's own prompt format (DeepSeek V4's reference encoder for Pro, the chat template otherwise) and sent as
token ids with the model card's sampling. On data-parallel cells each session is pinned to one rank for all its
turns, as the gateway does. Per concurrency step: a warm-up round on sessions the measurement never uses, a cache
flush, then the measured round; warm figures are turns 2 and 3. Each arm repeats the whole ladder on fresh
sessions (at least three times): one run per arm is never enough, the GLM control moved 11% between runs.

    modal run operations/modal/cell_gate.py --cell deepseek-v4-pro/standard --arms control,shared_local \
        --ladder 64,128 --repeats 3
"""

from __future__ import annotations

import asyncio
import importlib.util
import glob
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import registry_image as ri  # noqa: E402

# GATE_REGISTRY: an extended registry (shipped cells + candidate cells for models the platform does not serve yet, e.g.
# operations/modal/dsv41flash_registry.json); it travels to the container as /root/serving_registry.json
REGISTRY = next((c for c in ([pathlib.Path(os.environ["GATE_REGISTRY"])] if os.environ.get("GATE_REGISTRY") else [])
                 + [ROOT / "configs/serving_registry.json", pathlib.Path("/root/serving_registry.json")]
                 if c.exists()), ROOT / "configs/serving_registry.json")
# GATE_REGISTRY: a gate-only registry for cells that are not served (e.g. configs/k3state_registry.json, Kimi K3); it is
# shipped to the container as /root/serving_registry.json like the real one
if os.environ.get("GATE_REGISTRY"):
    REGISTRY = pathlib.Path(os.environ["GATE_REGISTRY"]).resolve()
# GATE_LOCAL=1 (gate_local.py): arms run on this machine (a Daytona box), no Modal functions or volume
LOCAL = os.environ.get("GATE_LOCAL") == "1"
CACHE = os.environ.get("GATE_CACHE", "/cache")


class _NoVolume:
    def commit(self) -> None:
        pass


WEIGHTS = _NoVolume() if LOCAL else modal.Volume.from_name("weights-cache", create_if_missing=True)
# GATE_PORT: two gates can share one box on disjoint GPUs (CUDA_VISIBLE_DEVICES) only on distinct port ranges
PORT = int(os.environ.get("GATE_PORT", "30000"))
# Output cap per turn. Default 400 keeps every earlier gate comparable. Measured on the V4.1 46K-session ladder
# (rtxv41, 2026-10-01): mean_out_tokens 236-256, so the ladder ran r = 15-17 prefill tokens per output token, ~4x
# the OpenRouter mix (r = 4.25, 988 out), and its tok/s under-reported market throughput ~3-4x.
# OR-shaped runs: GATE_MAX_OUT=988 GATE_IGNORE_EOS=1 (every turn decodes exactly the OR mean).
MAX_OUT = int(os.environ.get("GATE_MAX_OUT", "400"))
IGNORE_EOS = os.environ.get("GATE_IGNORE_EOS") == "1"
app = modal.App(os.environ.get("GATE_APP", "cell-gate-7f3a"))

SAMPLING = {  # the model cards' recommended sampling
    "deepseek-v4-pro": {"temperature": 0.6, "top_p": 0.95},
    "deepseek-v4.1-flash": {"temperature": 1.0, "top_p": 0.95},   # V4.1 card: reasoning evals at 1.0 / 0.95
    "glm-5.3": {"temperature": 0.6, "top_p": 0.95},
    "qwen3.8-27b": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "qwen3.6-35b-a3b": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "gemma-4-26b-a4b": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "kimi-k2.7-code": {"temperature": 1.0, "top_p": 0.95},   # K2.7 Code card (lab cell, operations/modal/k27_registry.py)
}
SESSIONS = {  # sessions sized to each lane's sold shape in that model's tokens
    "deepseek-v4-pro/standard": "data/real_sessions_dsv4_46k.json",   # OpenRouter shape, 46k median
    "deepseek-v4-pro/fast": "data/real_sessions_dsv4_83k.json",       # the fast lane's 65k cell and coding turns
    "deepseek-v4.1-flash/standard": "data/real_sessions_dsv4_46k.json",  # same 129,280 vocab as Pro (dsv41flash)
    "deepseek-v4.1-flash/fast": "data/real_sessions_dsv4_83k.json",
    "diffusiongemma-26b-a4b/standard": "data/real_sessions_gemma_28k.json",   # the lane's 28k shape, 28.9k median
    "diffusiongemma-26b-a4b/fast": "data/real_sessions_gemma_28k.json",
    "default": "data/real_sessions_240.json",                          # ~60-75k in Qwen/Gemma tokens
}

# ---- candidate deltas. Each is one change against the shipped cell. -------------------------------------------
from gate_arms import ARMS, compose_flags  # noqa: E402  (candidate deltas and flag composition, shared with verify_registry and preflight)


def cells() -> list[dict]:
    return json.loads(REGISTRY.read_text())["cells"]


def cell_by_key(key: str) -> dict:
    m, lane = key.split("/")
    return next(c for c in cells() if c["model"] == m and c["lane"] == lane)


# ---- traffic -------------------------------------------------------------------------------------------------
def _renderer(model_dir: str):
    # DeepSeek's reference encoders: V4 Pro ships encoding/encoding_dsv4.py, V4.1 Flash encoding/encoding.py (same
    # encode_messages(messages, thinking_mode) API, chat mode = thinking off)
    enc = next((p for p in (os.path.join(model_dir, "encoding", "encoding_dsv4.py"),
                            os.path.join(model_dir, "encoding", "encoding.py")) if os.path.exists(p)), "")
    if enc:
        spec = importlib.util.spec_from_file_location("encoding_dsv4", enc)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
        return lambda msgs: tok(mod.encode_messages(msgs, thinking_mode="chat"), add_special_tokens=False)["input_ids"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return lambda msgs: tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                            add_special_tokens=False)["input_ids"]


async def _request(http, url: str, engine: str, ids: list[int], rank, sampling: dict) -> dict:
    """One request; a disconnect before the server answered is retried once (see _request_once)."""
    import aiohttp
    try:
        return await _request_once(http, url, engine, ids, rank, sampling)
    except aiohttp.ServerDisconnectedError as e:
        if getattr(e, "_numinous_started", False):
            raise
        return await _request_once(http, url, engine, ids, rank, sampling)


async def _stream(resp):
    async for raw in resp.content:
        yield raw


async def _request_once(http, url: str, engine: str, ids: list[int], rank, sampling: dict) -> dict:
    t0 = time.perf_counter()
    ttft, usage = None, {}
    if sampling.get("_diffusion"):
        # block-diffusion samplers answer whole canvases and reject temperature/top_p/top_k: non-streaming,
        # max_tokens only, and the speed is the whole request's (how the diffusion lanes are sold)
        body = {"model": os.environ.get("GATE_MODEL_NAME", "m"), "prompt": ids, "max_tokens": MAX_OUT}
        async with http.post(url + "/v1/completions", json=body) as resp:
            if resp.status != 200:
                raise RuntimeError(f"http {resp.status}: {(await resp.text())[:300]}")
            j = await resp.json()
        wall = time.perf_counter() - t0
        u = j.get("usage") or {}
        n = int(u.get("completion_tokens") or 0)
        return {"prompt": int(u.get("prompt_tokens") or len(ids)),
                "cached": int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0), "out": n,
                "ttft": wall, "wall": wall, "t_start": t0, "t_end": t0 + wall,
                "decode_tok_s": n / wall if n and wall > 0 else None}
    if engine == "vllm":
        body = {"model": os.environ.get("GATE_MODEL_NAME", "m"), "prompt": ids, "max_tokens": MAX_OUT, "stream": True,
                "stream_options": {"include_usage": True}, **sampling}
        if IGNORE_EOS:
            body["ignore_eos"] = True
        path, headers = "/v1/completions", ({"X-data-parallel-rank": str(rank)} if isinstance(rank, int) else {})
    else:
        body = {"input_ids": ids, "stream": True, "sampling_params": {"max_new_tokens": MAX_OUT, **sampling}}
        if IGNORE_EOS:
            body["sampling_params"]["ignore_eos"] = True
        if isinstance(rank, str):
            # cold-turn balancing (GATE_PD_COLD_BALANCE): the PD proxy picks the session's rank on its first turn
            body["session_key"] = rank
        elif rank is not None:
            body["routed_dp_rank"] = rank
        path, headers = "/generate", {}
    import aiohttp
    async with http.post(url + path, json=body, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError(f"http {resp.status}: {(await resp.text())[:300]}")
        try:
            async for raw in _stream(resp):
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if engine == "vllm":
                    if ttft is None and chunk.get("choices") and chunk["choices"][0].get("text"):
                        ttft = time.perf_counter() - t0
                    usage = chunk.get("usage") or usage
                else:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    usage = chunk.get("meta_info") or usage
        except aiohttp.ServerDisconnectedError as e:
            e._numinous_started = ttft is not None or bool(usage)   # noqa: SLF001  mid-stream: not safe to retry
            raise
    wall = time.perf_counter() - t0
    n = int(usage.get("completion_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or len(ids))
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or usage.get("cached_tokens") or 0)
    return {"prompt": prompt, "cached": cached, "out": n, "ttft": ttft or wall, "wall": wall, "t_start": t0,
            "t_end": t0 + wall, "decode_tok_s": (n - 1) / (wall - ttft) if ttft and n > 1 and wall > ttft else None}


def _rung(url: str, engine: str, dp: int, prompts: list[list[list[int]]], sampling: dict,
          stagger_s: float = 0.0) -> dict:
    """stagger_s spreads the sessions' first turns uniformly over that many seconds. With 0 every session starts
    its cold first turn at the same instant, a synchronized herd production never sees: early sessions' warm turns
    then queue behind later sessions' cold prefills on the same rank and TTFT p99 reads 40-70 s."""
    import aiohttp

    nonce = os.urandom(4).hex()   # session keys are unique per rung (the proxy pins keys for its lifetime)

    async def one(http, i, turns):
        # GATE_NO_PIN=1: requests carry no routed_dp_rank, so the engine's DP controller places them (e.g. the
        # cache-aware balancer, operations/patches/dp_prefill_balance_hooks.py); default keeps the i % dp pin
        rank = i % dp if dp > 1 and os.environ.get("GATE_NO_PIN") != "1" else None
        if dp > 1 and os.environ.get("GATE_PD_COLD_BALANCE") == "1":
            # in-box PD with gateway_pd_proxy --cold-balance: the session's first turn goes to the least-loaded
            # prefill rank and its later turns stay there (amdpro 2026-10-05; P4: a pinned rank ran dry)
            rank = f"{nonce}-{i}"
        if stagger_s > 0:
            await asyncio.sleep(stagger_s * i / max(1, len(prompts)))
        return [await _request(http, url, engine, ids, rank, sampling) for ids in turns]

    async def go():
        timeout = aiohttp.ClientTimeout(total=3600, sock_read=3600)
        # force_close: one connection per request. aiohttp keeps idle connections 15 s, uvicorn closes them after 5 s,
        # so a session's next turn could reuse a connection the server had already closed and fail with
        # ServerDisconnectedError at high load, which read as a server crash.
        async with aiohttp.ClientSession(timeout=timeout,
                                         connector=aiohttp.TCPConnector(limit=0, force_close=True)) as http:
            return await asyncio.gather(*(one(http, i, t) for i, t in enumerate(prompts)))

    per = asyncio.run(go())
    rows = [r for s in per for r in s]
    later = [r for s in per for r in s[1:]]
    med = lambda xs: round(statistics.median(xs), 2) if xs else None  # noqa: E731
    wspan = (max(r["t_end"] for r in later) - min(r["t_start"] for r in later)) if later else 0
    # warm_agg divides by first-start..last-end, so one session stuck behind a long prefill moves it by tens of
    # percent on identical sessions. Two tail-robust companions: steady = output of requests completing between the
    # 10th and 90th percentile of completion times over that interval (all turns); session_rate_sum = sum over
    # sessions of their own turn-2..n output rate.
    ends = sorted(r["t_end"] for r in rows)
    steady = None
    if len(ends) >= 10:
        lo, hi = ends[len(ends) // 10], ends[(len(ends) * 9) // 10]
        if hi > lo:
            steady = round(sum(r["out"] for r in rows if lo < r["t_end"] <= hi) / (hi - lo), 1)
    rate_sum = 0.0
    for sess in per:
        tail = sess[1:]
        if tail:
            dur = max(r["t_end"] for r in tail) - min(r["t_start"] for r in tail)
            if dur > 0:
                rate_sum += sum(r["out"] for r in tail) / dur
    def pct(xs, q):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None
    walls = [r["wall"] for r in later]
    ttfts = [r["ttft"] for r in later]
    tail = {"ttft_p90_s": pct(ttfts, 0.9), "ttft_p99_s": pct(ttfts, 0.99), "ttft_max_s": round(max(ttfts), 2) if ttfts else None,
            "turn_wall_p50_s": pct(walls, 0.5), "turn_wall_p90_s": pct(walls, 0.9), "turn_wall_p99_s": pct(walls, 0.99),
            # how long the last tenth of completions took: a long tail shows here and in warm_agg, not in steady
            "tail_last10pct_s": round(ends[-1] - ends[(len(ends) * 9) // 10], 1) if len(ends) >= 10 else None}
    # per-turn prefix reuse (turn 1 is cold after the flush; a healthy cache reads ~0.9 from turn 2 on) and the warm
    # requests that reused almost nothing: the client-side face of an eviction problem
    nturns = max((len(s) for s in per), default=0)
    hit_by_turn = [round(sum(s[t]["cached"] for s in per if len(s) > t) /
                         max(1, sum(s[t]["prompt"] for s in per if len(s) > t)), 4) for t in range(nturns)]
    tail["hit_by_turn"] = hit_by_turn
    tail["warm_low_hit_requests"] = sum(1 for r in later if r["cached"] < 0.1 * r["prompt"])
    tail["warm_requests"] = len(later)
    return {"warm_per_user_tok_s": med([r["decode_tok_s"] for r in later if r["decode_tok_s"]]),
            "warm_agg_out_tok_s": round(sum(r["out"] for r in later) / wspan, 1) if wspan > 0 else None,
            "steady_agg_out_tok_s": steady,
            **tail,
            "warm_session_rate_sum": round(rate_sum, 1) if rate_sum else None,
            "warm_ttft_p50_s": med([r["ttft"] for r in later]),
            "hit_rate": round(sum(r["cached"] for r in later) / max(1, sum(r["prompt"] for r in later)), 4),
            "mean_prompt_tokens": round(sum(r["prompt"] for r in rows) / len(rows)),
            "mean_out_tokens": round(sum(r["out"] for r in rows) / len(rows), 1)}


# Production-shape row (info only; amdglm findings 9g, coordinator order; the gate stays warm_agg on the OR rungs).
# The OR rungs start N sessions together after a flush, so every cold turn-1 prefill lands in one wave and the
# per-session prefill load is ~1.8x the OpenRouter mix per output token (GLM pool ~11.6 uncached tokens per output
# token vs 6.39 on OR). This row keeps N sessions in flight (closed loop, each finished session replaced by a fresh
# one drawn from the pool), with the pool tilted toward the OR mix, and reads output tok/s over a fixed window after a
# warm-up: the rate the price assumes at concurrency N.
OR_MIX = {"glm-5.3": (6.39, 51.2), "glm-5.3-flash": (6.95, 34.77)}   # (uncached, cached) prompt tokens per output token


def session_mix(s: dict) -> tuple[float, float, float]:
    """(uncached, cached, output) characters of one recorded session replayed turn by turn with a warm prefix cache:
    turn 1 is all uncached; turn t > 1 reuses turn t-1's prompt and adds the recorded answer + the new user message;
    the recorded assistant turns stand in for the generated outputs. Characters, not tokens: only ratios are used."""
    msgs, ends = s["messages"], s["turn_ends"]
    chars = [len(m.get("content") or "") for m in msgs]
    pref = [sum(chars[:e]) for e in ends]
    u = pref[0] + sum(pref[t] - pref[t - 1] for t in range(1, len(pref)))
    c = sum(pref[t - 1] for t in range(1, len(pref)))
    o = sum(chars[i] for i, m in enumerate(msgs) if m.get("role") == "assistant") or 1
    return float(u), float(c), float(o)


def mix_weights(stats: list, target_u: float) -> tuple[list, dict]:
    """Weights w_i ~ exp(lam * u_i / o_i) (a 1-D exponential tilt on the uncached-per-output ratio, the term that
    sets the prefill load) so the weighted pool's sum(w u) / sum(w o) meets target_u; bisection on lam. Returns the
    weights and the achieved (uncached, cached) per output for the record (cached follows, it is not fitted)."""
    import math
    ratios = [u / o for u, _, o in stats]

    def achieved(lam):
        m = max(lam * r for r in ratios)
        w = [math.exp(lam * r - m) for r in ratios]
        su = sum(wi * st[0] for wi, st in zip(w, stats))
        sc = sum(wi * st[1] for wi, st in zip(w, stats))
        so = sum(wi * st[2] for wi, st in zip(w, stats))
        return w, su / so, sc / so
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if achieved(mid)[1] > target_u:
            hi = mid
        else:
            lo = mid
    w, au, ac = achieved((lo + hi) / 2)
    tot = sum(w)
    return [x / tot for x in w], {"uncached_per_out": round(au, 3), "cached_per_out": round(ac, 3),
                                  "pool_uncached_per_out": round(sum(st[0] for st in stats)
                                                                 / sum(st[2] for st in stats), 3)}


def _prod_rung(url: str, engine: str, dp: int, prompt_of, weights: list, n: int, warmup_s: float, window_s: float,
               sampling: dict, stagger_s: float = 0.0, seed: int = 7, request=None) -> dict:
    """Closed loop: n session slots; a slot runs one drawn session's turns back to back, then draws the next (with
    replacement, by weight). Starts are staggered over stagger_s. Counted: requests completing inside
    [warmup_s, warmup_s + window_s] after the first start; slots stop drawing new sessions at the window end and the
    in-flight requests finish (not counted). prompt_of(i) -> the session's rendered turns."""
    import random
    import aiohttp
    request = request or _request
    rng = random.Random(seed)
    t0 = time.perf_counter()
    stop = t0 + warmup_s + window_s
    done: list = []
    started = {"sessions": 0}

    async def slot(http, k):
        rank = k % dp if dp > 1 and os.environ.get("GATE_NO_PIN") != "1" else None
        if stagger_s > 0:
            await asyncio.sleep(stagger_s * k / max(1, n))
        while time.perf_counter() < stop:
            i = rng.choices(range(len(weights)), weights=weights)[0]
            started["sessions"] += 1
            for t, ids in enumerate(prompt_of(i)):
                if time.perf_counter() >= stop:
                    break
                r = await request(http, url, engine, ids, rank, sampling)
                r["turn"] = t
                done.append(r)

    async def go():
        timeout = aiohttp.ClientTimeout(total=7200, sock_read=3600)
        async with aiohttp.ClientSession(timeout=timeout,
                                         connector=aiohttp.TCPConnector(limit=0, force_close=True)) as http:
            await asyncio.gather(*(slot(http, k) for k in range(n)))

    asyncio.run(go())
    w0, w1 = t0 + warmup_s, stop
    inwin = [r for r in done if w0 <= r["t_end"] <= w1]
    out = sum(r["out"] for r in inwin)
    rates = sorted(r["decode_tok_s"] for r in inwin if r.get("decode_tok_s"))
    ttfts = sorted(r["ttft"] for r in inwin)
    pct = lambda xs, q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None  # noqa: E731
    prompt = sum(r["prompt"] for r in inwin)
    cached = sum(r["cached"] for r in inwin)
    return {"prod_sessions": n, "prod_window_s": window_s, "prod_warmup_s": warmup_s,
            "prod_out_tok_s": round(out / window_s, 1) if window_s > 0 else None,
            "prod_per_user_tok_s": pct(rates, 0.5), "prod_ttft_p50_s": pct(ttfts, 0.5),
            "prod_ttft_p90_s": pct(ttfts, 0.9), "prod_requests": len(inwin),
            "prod_cold_share": round(sum(1 for r in inwin if r["turn"] == 0) / max(1, len(inwin)), 3),
            "prod_sessions_started": started["sessions"],
            "prod_uncached_per_out": round((prompt - cached) / max(1, out), 3),
            "prod_cached_per_out": round(cached / max(1, out), 3)}


def _post(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=120)
        return True
    except Exception:  # noqa: BLE001
        return False


def expert_load_summary(logical_count, ep: int) -> dict:
    """Per-layer EP load skew from SGLang's expert-distribution dump (stat mode: logical_count [layers, experts],
    token-expert assignments summed over the recorded steps). With the trivial placement expert e lives on rank
    e // (experts / ep); a rank's load is the sum over its experts. skew = max rank load / mean rank load (1.0 =
    balanced). The MegaMoE kernel ends when the most loaded rank's experts finish, so the skew bounds what a
    balanced placement (EPLB redundant experts) can save in MoE time."""
    rows = [list(map(float, r)) for r in logical_count]
    rows = [r for r in rows if sum(r) > 0]
    if not rows:
        return {"layers": 0}
    per = len(rows[0]) // ep
    skews, hot = [], []
    for r in rows:
        loads = [sum(r[k * per:(k + 1) * per]) for k in range(ep)]
        mean = sum(loads) / ep
        skews.append(max(loads) / mean if mean else 1.0)
        hot.append(max(r) / (sum(r) / len(r)) if sum(r) else 1.0)
    q = sorted(skews)
    return {"layers": len(rows), "rank_skew_mean": round(sum(q) / len(q), 3),
            "rank_skew_p50": round(q[len(q) // 2], 3), "rank_skew_max": round(q[-1], 3),
            "hot_expert_over_mean_mean": round(sum(hot) / len(hot), 2)}


def _expert_record_dump(url: str, out_dir: str, ep: int) -> dict:
    """Stop the expert-distribution recorder, dump it (the server writes a .pt into
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR) and summarise the newest dump."""
    import glob
    _post(url + "/stop_expert_distribution_record")
    if not _post(url + "/dump_expert_distribution_record"):
        return {"error": "dump failed"}
    time.sleep(5)
    fs = sorted(glob.glob(f"{out_dir}/*.pt"), key=os.path.getmtime)
    if not fs:
        return {"error": f"no dump in {out_dir}"}
    import torch
    data = torch.load(fs[-1], weights_only=True, map_location="cpu")
    lc = data.get("logical_count")
    if lc is None:
        return {"error": f"no logical_count in {fs[-1]}", "keys": list(data)}
    if lc.dim() == 3:   # per-step buffer: sum the steps
        lc = lc.sum(0)
    try:
        WEIGHTS.commit()          # the dump on the volume is the input of a static EPLB placement (--init-expert-location)
    except Exception:  # noqa: BLE001
        pass
    return {"file": fs[-1], **expert_load_summary(lc.tolist(), ep)}


def _flush(url: str, tries: int = 6) -> bool:
    """Flush the prefix cache and confirm it. SGLang refuses (HTTP 400) while any request or cache op is pending,
    e.g. HiCache write-through still draining after a round: the next round then measures warm prefixes it should not
    have (q38c_hicache 2026-10-02: turn-1 hit 0.74 at the second rung). Retry until it reports success."""
    for i in range(tries):
        if _post(url):
            return True
        time.sleep(5 * (i + 1))
    return False


def _cause(text: str) -> list[str]:
    """The failure's own traceback: the last one not logged as a warning (vLLM logs optional-import tracebacks as
    warnings long before anything fails), plus the log's final lines."""
    lines = [x for x in text.splitlines() if x.strip()]
    starts = [i for i, x in enumerate(lines) if "Traceback" in x and "WARNING" not in x]
    # vLLM's engine core runs in its own process and logs its failure before the API server's wrapper raises, so
    # the first engine-core traceback is usually the real one
    core = [i for i, x in enumerate(lines) if "EngineCore" in x and ("Traceback" in x or "ERROR" in x)]
    core_block = lines[core[0]:core[0] + 45] if core else []
    block = lines[starts[-1]:starts[-1] + 25] if starts else []
    errs = [x for x in lines if re.search(r"(Error|Exception)[:(]", x) and "WARNING" not in x][-10:]
    return core_block + ["..."] + block + ["... error lines:"] + errs + ["... tail:"] + lines[-8:]


ACCEPT = re.compile(r"accept[ _]len(?:gth)?[:=]\s*([\d.]+)|Mean acceptance length:\s*([\d.]+)")
POOL = re.compile(r"max_total_num_tokens=(\d+)|KV cache size:\s*([\d,]+)\s*tokens")
RADIX_TAG = "numinous radix stats "


def _radix_snaps(text: str) -> list:
    """The radix accounting lines (operations/kernels/q38_radix_stats.py, NUMINOUS_RADIX_STATS=1) in a log chunk."""
    out = []
    for ln in text.splitlines():
        i = ln.find(RADIX_TAG + "{")
        if i >= 0:
            try:
                out.append(json.loads(ln[i + len(RADIX_TAG):]))
            except json.JSONDecodeError:
                pass
    return out


def _radix_delta(head: str, seg: str) -> dict | None:
    """Counters accumulated during one measured round: the last snapshot in the round minus the last one before it
    (the flush that opened the round), with the shares of prompt tokens reused / lost to a missing GDN state / lost
    to evicted KV."""
    after = _radix_snaps(seg)
    if not after:
        return None
    before = (_radix_snaps(head[-200000:]) or [{}])[-1]
    a = after[-1]
    d = {k: (v - before.get(k, 0)) if isinstance(v, (int, float)) and not k.endswith("_now") else v for k, v in a.items()}
    if d.get("requests", 0) < 0:
        d = dict(a)
    p = d.get("prompt_tokens") or 0
    if p > 0:
        for k in ("usable", "mamba_gap", "kv_absent"):
            d[f"{k}_share"] = round(d.get(f"{k}_tokens", 0) / p, 4)
    return d


def _probes(url: str, engine: str, render, sampling: dict) -> dict:
    """Six long-document retrieval questions with checkable answers (dc_bench.probe_prompts), greedy where the engine
    allows it. GATE_PROBE_REPEATS asks each question that many times: a diffusion sampler is not deterministic, so one
    miss in six is only a regression if it repeats. Keeps which probe missed and what the model said."""
    import urllib.request
    sys.path.insert(0, "/root")
    import dc_bench
    repeats = int(os.environ.get("GATE_PROBE_REPEATS", "1"))
    probes = dc_bench.probe_prompts(n_prompts=6, target_chars=60000)
    per_probe, misses = [], []
    for k, pr in enumerate(probes):
        ids = render([{"role": "user", "content": pr["text"]}])
        if sampling.get("_diffusion"):
            body = {"model": os.environ.get("GATE_MODEL_NAME", "m"), "prompt": ids, "max_tokens": 512}   # diffusion samplers reject temperature
            path = "/v1/completions"
        elif engine == "vllm":
            body = {"model": os.environ.get("GATE_MODEL_NAME", "m"), "prompt": ids, "max_tokens": 512, "temperature": 0}
            path = "/v1/completions"
        else:
            body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 512, "temperature": 0}}
            path = "/generate"
        hits = 0
        for _ in range(repeats):
            req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                j = json.loads(r.read())
            text = j["choices"][0]["text"] if (engine == "vllm" or sampling.get("_diffusion")) else j.get("text", "")
            hit = pr["expect"].strip().lower() in text.lower()
            hits += int(hit)
            if not hit:
                misses.append({"probe": k, "expect": pr["expect"], "said": text[:300]})
        per_probe.append(hits)
    ok, total = sum(per_probe), len(probes) * repeats
    return {"score": f"{ok}/{total}", "per_probe": per_probe, "misses": misses[:12]}


NSYS_DIR = "/root/nsys"


def nsys_wrap(cmd: list) -> list:
    """The server under Nsight Systems: capture only between cudaProfilerStart/Stop (SGLang's CUDA_PROFILER
    activity, called by the first GPU's scheduler), one report per capture, kernels inside CUDA graphs traced."""
    import shutil
    os.makedirs(NSYS_DIR, exist_ok=True)
    import re as _re
    found = glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys")
    newest = max(found, key=lambda f: [int(x) for x in _re.findall(r"\d+", f.split("/")[-3])]) if found else None
    exe = newest or shutil.which("nsys") or "/usr/local/cuda/bin/nsys"
    return [exe, "profile", "-t", "cuda", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
            "--capture-range-end=repeat", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
            "-o", f"{NSYS_DIR}/rep"] + cmd


def nsys_summary(rep: str, phase: str) -> dict:
    """Kernel totals from one nsys report (the first GPU's process), in trace_summary.summarize's shape."""
    import csv
    import io
    import trace_summary
    exe = nsys_wrap([])[0]
    r = subprocess.run([exe, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--force-export=true",
                        "--output", "-", rep], capture_output=True, text=True, timeout=1800)
    text = r.stdout[r.stdout.find("Time"):] if "Time" in r.stdout else r.stdout
    rows = list(csv.DictReader(io.StringIO(text)))
    top, fams = [], {}
    for row in rows:
        name = row.get("Name") or ""
        ns = float(row.get("Total Time (ns)") or 0)
        cnt = int(float(row.get("Instances") or 0))
        fam = trace_summary.family(name)
        top.append({"name": name, "family": fam, "ms": ns / 1e6, "count": cnt})
        fams[fam] = fams.get(fam, 0.0) + ns / 1e6
    tot = sum(fams.values()) or 1.0
    top.sort(key=lambda k: -k["ms"])
    print(f"nsys {phase}: {len(rows)} kernels from {os.path.basename(rep)} ({r.returncode}; {r.stderr[-200:]!r})",
          flush=True)
    return {"families": {k: round(v / tot, 3) for k, v in sorted(fams.items(), key=lambda kv: -kv[1])},
            "top": top[:120], "ranks": [{"file": os.path.basename(rep), "kernels": len(rows)}], "source": "nsys"}


def _phase_profile(url: str, dp: int, prompts_by_session: list, n: int, steps: int, phase: str, arm: str) -> dict:
    """Profile one clean phase so every kernel can be priced against its own roofline.

    decode:  n sessions' first turns (the lane's sold context) are prefilled, then decode ignore_eos for long outputs;
             once every request is decoding, the profiler records `steps` forward passes.
    prefill: a feeder keeps 4 x dp fresh (uncached) first turns in flight with max_new_tokens 1, so every rank is
             prefilling full chunks; the profiler records `steps` passes.
    Returns the trace summary plus per-kernel microseconds per step per rank."""
    import glob
    import threading
    import urllib.request
    import aiohttp
    outdir = f"/root/prof_{phase}"
    counts: dict[int, int] = {}
    stop = threading.Event()

    async def one(http, i, ids, max_new):
        body = {"input_ids": ids, "stream": True, "sampling_params": {
            "max_new_tokens": max_new, "ignore_eos": True, "temperature": 0.6, "top_p": 0.95}}
        if dp > 1:
            body["routed_dp_rank"] = i % dp
        try:
            async with http.post(url + "/generate", json=body) as resp:
                async for raw in resp.content:
                    if raw.startswith(b"data:") and b"[DONE]" not in raw:
                        counts[i] = counts.get(i, 0) + 1
                    if stop.is_set():
                        break
        except Exception:  # noqa: BLE001
            pass

    async def run_decode():
        timeout = aiohttp.ClientTimeout(total=3600, sock_read=3600)
        async with aiohttp.ClientSession(timeout=timeout,
                                         connector=aiohttp.TCPConnector(limit=0, force_close=True)) as http:
            await asyncio.gather(*(one(http, i, prompts_by_session[i], 4096) for i in range(n)))

    async def run_prefill():
        timeout = aiohttp.ClientTimeout(total=3600, sock_read=3600)
        sem = asyncio.Semaphore(4 * dp)
        async with aiohttp.ClientSession(timeout=timeout,
                                         connector=aiohttp.TCPConnector(limit=0, force_close=True)) as http:
            async def fed(i):
                async with sem:
                    if not stop.is_set():
                        await one(http, i, prompts_by_session[i], 1)
            await asyncio.gather(*(fed(i) for i in range(len(prompts_by_session))))

    th = threading.Thread(target=lambda: asyncio.run(run_decode() if phase == "decode" else run_prefill()), daemon=True)
    th.start()
    t0 = time.time()
    if phase == "decode":
        while time.time() - t0 < 1800 and sum(1 for i in range(n) if counts.get(i, 0) >= 24) < n:
            time.sleep(2)
        print(f"[{arm}] decode: {sum(1 for i in range(n) if counts.get(i, 0) >= 24)}/{n} requests decoding after "
              f"{time.time() - t0:.0f} s", flush=True)
    else:
        time.sleep(25)
    nsys = os.environ.get("GATE_NSYS_ACTIVE") == "1"
    reps_before = set(glob.glob(f"{NSYS_DIR}/*.nsys-rep"))
    body = json.dumps({"output_dir": outdir, "num_steps": steps, "activities": ["CUDA_PROFILER"] if nsys else ["GPU"],
                       "with_stack": False, "record_shapes": False}).encode()
    urllib.request.urlopen(urllib.request.Request(url + "/start_profile", data=body, method="POST",
                                                  headers={"content-type": "application/json"}), timeout=120)
    t_wait = time.time()
    fs: list = []
    while time.time() - t_wait < 1800:
        if nsys:
            fs = sorted(set(glob.glob(f"{NSYS_DIR}/*.nsys-rep")) - reps_before)
            if fs and time.time() - os.path.getmtime(fs[-1]) > 30:
                break
        else:
            fs = glob.glob(f"{outdir}/**/*.trace.json*", recursive=True)
            if len(fs) >= dp and time.time() - max(os.path.getmtime(f) for f in fs) > 20:
                break
        time.sleep(10)
    print(f"[{arm}] {phase} traces: {len(fs)} files after {time.time() - t_wait:.0f} s, MB "
          f"{[round(os.path.getsize(f) / 1e6, 1) for f in sorted(fs)]}", flush=True)
    if not fs:
        print(f"[{arm}] {phase} profile dir: {[str(x) for x in pathlib.Path(outdir).rglob('*')][:20]}", flush=True)
        if nsys:
            print(f"[{arm}] {phase} nsys dir: {[(x.name, x.stat().st_size) for x in pathlib.Path(NSYS_DIR).glob('*')][:20]}",
                  flush=True)
            try:
                for ln in open(os.environ.get("GATE_SERVER_LOG", "/dev/null"), errors="ignore"):
                    if any(k in ln for k in ("nsys", "Capture", "capture", "Generating", "qdstrm", "cudaProfiler",
                                             "Collecting", "Nsight")):
                        print(f"[{arm}] {phase} server: {ln.strip()[:220]}", flush=True)
            except OSError:
                pass
    stop.set()
    try:
        urllib.request.urlopen(urllib.request.Request(url + "/abort_request", data=json.dumps(
            {"rid": "", "abort_all": True}).encode(), method="POST", headers={"content-type": "application/json"}),
            timeout=60)
    except Exception:  # noqa: BLE001
        pass
    th.join(timeout=300)
    import trace_summary
    summ = nsys_summary(fs[-1], phase) if (nsys and fs) else trace_summary.summarize(outdir, top_n=120)
    ranks = max(1, len(summ.get("ranks") or []))
    per_step = [{"name": k["name"], "family": k["family"], "us_per_step_rank": round(k["ms"] * 1000 / steps / ranks, 1),
                 "calls_per_step_rank": round(k["count"] / steps / ranks, 2)} for k in summ.get("top", [])]
    print(f"[{arm}] {phase} profile: families {summ.get('families')}", flush=True)
    print(f"[{arm}] {phase} ranks: {[(r.get('file'), r.get('kernels')) for r in summ.get('ranks') or []]}", flush=True)
    if fs and not any(r.get("kernels") for r in summ.get("ranks") or []):
        import collections
        import gzip
        with (gzip.open if fs[0].endswith(".gz") else open)(fs[0], "rt") as fh:
            ev = json.load(fh)
        ev = ev.get("traceEvents", ev) if isinstance(ev, dict) else ev
        print(f"[{arm}] {phase} no kernels; event categories "
              f"{collections.Counter(e.get('cat') for e in ev if isinstance(e, dict)).most_common(12)}", flush=True)
    for k in per_step[:40]:
        print(f"[{arm} {phase}] {k['us_per_step_rank']:9.1f} us/step x{k['calls_per_step_rank']:6.2f}  {k['family']:10s} "
              f"{k['name'][:120]}", flush=True)
    return {"steps": steps, "sessions": n, "summary": summ, "per_step": per_step}


def _dp_distribution(text: str) -> dict:
    """Peak running requests per DP rank, from each scheduler's own "DPn ... #running-req: N" log lines: whether the
    sessions actually spread over the ranks the router assigned them to."""
    peak: dict[str, int] = {}
    for m in re.finditer(r"DP(\d+) TP\d+[^\]]*\] (?:Decode|Prefill) batch[^\n]*?#running-req: (\d+)", text):
        r, n = m.group(1), int(m.group(2))
        peak[r] = max(peak.get(r, 0), n)
    return dict(sorted(peak.items(), key=lambda kv: int(kv[0])))


_MONITOR_STARTED = False


def _monitor(arm: str, every_s: int = 20) -> None:
    """Every few seconds, one line of host memory, the engine's resident memory and GPU state: when a container is
    terminated from outside ("Runner terminated", no traceback), the last lines tell a host-memory kill from a
    preemption or a GPU fault."""
    def rss_gb() -> float:
        total = 0
        for pid in os.listdir("/proc"):
            if pid.isdigit():
                try:
                    for line in open(f"/proc/{pid}/status"):
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1])
                except OSError:
                    pass
        return total / 1e6

    prev: dict = {}
    tck = os.sysconf("SC_CLK_TCK")

    def top_cpu(n: int = 10) -> list:
        """(name, % of one core) for the busiest processes since the last sample: shows whether a scheduler is
        CPU-bound (a straggling DP rank leaves the other GPUs spinning in collectives)."""
        out, now = [], {}
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
                ticks = int(f[11]) + int(f[12])
                name = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()[:48] or pid
            except (OSError, IndexError, ValueError):
                continue
            now[pid] = ticks
            if pid in prev:
                out.append((round(100.0 * (ticks - prev[pid]) / tck / every_s), name))
        prev.clear()
        prev.update(now)
        return sorted(out, reverse=True)[:n]

    def loop() -> None:
        while True:
            try:
                load = os.getloadavg()
                print(f"[{arm} mon] load={load[0]:.1f}/{os.cpu_count()} top_cpu={top_cpu()}", flush=True)
                mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
                g = _gpu_monitor_rows()
                shm = os.statvfs("/dev/shm")
                print(f"[{arm} mon] host_used_gb={(mem['MemTotal'] - mem['MemAvailable']) / 1e6:.1f}/{mem['MemTotal'] / 1e6:.0f} "
                      f"rss_gb={rss_gb():.1f} shm_used_gb={(shm.f_blocks - shm.f_bfree) * shm.f_frsize / 1e9:.1f} "
                      f"gpu_mem_gb={[round(int(x[0]) / 1024) for x in g]} temp={[x[1] for x in g]} "
                      f"power_w={[round(float(x[2])) for x in g]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{arm} mon] {type(e).__name__}: {e}", flush=True)
            time.sleep(every_s)

    global _MONITOR_STARTED
    if _MONITOR_STARTED:
        return
    _MONITOR_STARTED = True
    threading.Thread(target=loop, daemon=True).start()



# GPU queries for both vendors (the same gate runs on NVIDIA through Modal and on AMD MI355X through gate_local /
# amd_gate): nvidia-smi where it exists, rocm-smi otherwise, nothing (and no crash) when neither does.
def _gpu_mem_used_mib() -> list[int]:
    import shutil
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        return [int(x) for x in out if x.strip().isdigit()]
    if shutil.which("rocm-smi"):
        r = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True)
        try:
            d = json.loads(r.stdout or "{}")
        except ValueError:
            return []
        used = []
        for card, v in sorted(d.items()):
            if isinstance(v, dict):
                b = v.get("VRAM Total Used Memory (B)")
                if b is not None:
                    used.append(int(int(b) / (1 << 20)))
        return used
    return []


def _gpu_ids() -> list[str]:
    import shutil
    if shutil.which("nvidia-smi"):
        return subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True,
                              text=True).stdout.split()
    if shutil.which("rocm-smi"):
        r = subprocess.run(["rocm-smi", "--showuniqueid", "--json"], capture_output=True, text=True)
        try:
            d = json.loads(r.stdout or "{}")
        except ValueError:
            return []
        return [str(v.get("Unique ID", k)) for k, v in sorted(d.items()) if isinstance(v, dict)]
    return []


def _gpu_monitor_rows() -> list[list[str]]:
    """[mem_used_mib, temp_c, power_w] per GPU for the monitor thread; [] where no tool is present."""
    import shutil
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return [x.split(", ") for x in out.strip().splitlines()]
    return [[str(m), "-", "0"] for m in _gpu_mem_used_mib()]

def _run_arm(cell: dict, arm: str, delta: dict, ladder: list[int], repeats: int, sessions_file: str,
             probes: bool = False, profile_steps: int = 0, rep_offset: int = 0) -> dict:
    cfg = cell["config"]
    t0 = time.time()
    repo = cfg["repo"]
    path = f"{CACHE}/{repo.replace('/', '__')}"
    if not (os.path.isdir(path) and os.listdir(path)):
        import huggingface_hub
        huggingface_hub.snapshot_download(repo, local_dir=path, max_workers=16)
        WEIGHTS.commit()
    flags = compose_flags(cfg["flags"], delta, path)
    # __SPS__: the DSpark step-cost table profiled for this model on this GPU (sps_profile arm, below)
    sps_path = f"{CACHE}/sps/{cell['model']}_{delta.get('_gpu_label', 'reg')}.json"
    flags = [f.replace("__SPS__", sps_path) for f in flags]
    # __LATEST_EPREC__:<dir>: the newest expert-distribution dump in <dir> (an earlier pro_eprec arm of this node run
    # wrote it), the input of a static EPLB placement (--init-expert-location)
    if any(f.startswith("__LATEST_EPREC__:") for f in flags):
        import glob
        for k, f in enumerate(flags):
            if f.startswith("__LATEST_EPREC__:"):
                try:
                    WEIGHTS.reload()
                except Exception:  # noqa: BLE001
                    pass
                dumps = sorted(glob.glob(f"{f.split(':', 1)[1]}/*.pt"), key=os.path.getmtime)
                if not dumps:
                    out = {"cell": f"{cell['model']}/{cell['lane']}", "arm": arm, "runs": [],
                           "error": f"no expert-distribution dump in {f.split(':', 1)[1]} (run the recorder arm first)"}
                    print(f"[{arm}] {out['error']}", flush=True)
                    return out
                flags[k] = dumps[-1]
                print(f"[{arm}] static EPLB placement from {dumps[-1]}", flush=True)
    if any(f == sps_path for f in flags) and not os.path.exists(sps_path):
        out["error"] = f"no DSpark SPS table at {sps_path}: run the sps_profile arm on this GPU first"
        print(f"[{arm}] {out['error']}", flush=True)
        return out
    peak = max(ladder)
    env = {**os.environ, **(cfg.get("env") or {}), **delta.get("env", {}), "VLLM_SERVER_DEV_MODE": "1"}
    if delta.get("_same_sessions"):
        os.environ["GATE_SAME_SESSIONS"] = "1"
    if delta.get("_probe_repeats"):
        os.environ["GATE_PROBE_REPEATS"] = str(delta["_probe_repeats"])
    if delta.get("_max_chars"):
        os.environ["GATE_MAX_CHARS"] = str(delta["_max_chars"])
    global MAX_OUT, IGNORE_EOS
    if delta.get("_max_out"):
        MAX_OUT = int(delta["_max_out"])
    if "_ignore_eos" in delta:
        IGNORE_EOS = bool(delta["_ignore_eos"])
    if cfg["engine"] == "vllm":
        cmd = ["vllm", "serve", path, "--served-model-name", "m", "--host", "0.0.0.0", "--port", str(PORT)] + flags
        health, flush = "/health", "/reset_prefix_cache"
    else:
        cmd = ["python3", "-m", "sglang.launch_server", "--model-path", path, "--host", "0.0.0.0", "--port",
               str(PORT)] + flags
        health, flush = "/health_generate", "/flush_cache"
    ladder = list(delta.get("_ladder") or ladder)   # an arm may run its own ladder (e.g. a cache probe past the cap)
    out = {"cell": f"{cell['model']}/{cell['lane']}", "arm": arm, "delta": delta, "config_digest": cell["config_digest"],
           "image": delta.get("image") or cfg["image"], "flags": flags, "ladder": ladder, "repeats": repeats, "sessions_file": sessions_file,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": [],
           "out_shape": {"max_out": MAX_OUT, "ignore_eos": IGNORE_EOS}}
    logf = f"/root/{arm}.log"
    os.environ["GATE_SERVER_LOG"] = logf
    if delta.get("_nsys"):
        cmd = nsys_wrap(cmd)
        os.environ["GATE_NSYS_ACTIVE"] = "1"
        print(f"[{arm}] under nsys: {' '.join(cmd[:9])} ...", flush=True)
    if delta.get("_pd"):
        # in-box prefill/decode disaggregation: two servers on disjoint GPUs + the gateway's pd_proxy on PORT; every
        # later step (health, accuracy, rungs, flush, profile) talks to the proxy exactly as to a single engine
        pd_cmds = pd_launch(path, flags, delta["_pd"])
        out["pd"] = {k: {"flags": v["flags"], "env": v["env"]} for k, v in pd_cmds.items() if k != "proxy"}
        print(f"[{arm}] in-box PD: prefill {' '.join(pd_cmds['prefill']['flags'])}", flush=True)
        print(f"[{arm}] in-box PD: decode {' '.join(pd_cmds['decode']['flags'])}", flush=True)
        proc = PDGroup(pd_cmds, logf, env)
    else:
        proc = subprocess.Popen(cmd, stdout=open(logf, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    ready = False
    while time.time() - t0 < 150 * 60:
        if proc.poll() is not None:
            break
        code = subprocess.run(f"curl -s -m 5 -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{PORT}{health}",
                              shell=True, capture_output=True, text=True).stdout.strip()
        if code == "200":
            ready = True
            break
        time.sleep(15)
    text = open(logf, errors="ignore").read()
    pools = [int((a or b).replace(",", "")) for a, b in POOL.findall(text)]
    if isinstance(proc, PDGroup):
        ptext = open(proc.logs["prefill"], errors="ignore").read()
        out["pd_pools"] = {"decode": pools[:8], "prefill": [int((a or b).replace(",", "")) for a, b in POOL.findall(ptext)][:8]}
        print(f"[{arm}] PD pools {out['pd_pools']}", flush=True)
        if not ready:
            text = text + "\n" + ptext + "\n" + open(proc.logs["proxy"], errors="ignore").read()
    out.update(booted=ready, boot_min=round((time.time() - t0) / 60, 1), kv_pool_tokens=pools[:8] or None,
               node=_node())
    print(f"[{arm}] booted={ready} in {out['boot_min']} min pool={pools[:2]}", flush=True)
    if ready and delta.get("_idle_sanity"):
        # idle-rank sanity (operations/modal/idle_sanity.py): the production sanity pair on the still-empty server,
        # single / pair / single at T 0 / single again; a DP-attention + EAGLE idle-rank corruption fails it
        # (2026-10-03: every load gate passed while production served '17!!!!')
        try:
            import idle_sanity
            out["idle_sanity"] = idle_sanity.run_idle_sanity(f"http://127.0.0.1:{PORT}")
        except Exception as e:  # noqa: BLE001
            out["idle_sanity"] = {"pass": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"[{arm}] idle sanity pass={out['idle_sanity'].get('pass')}: "
              f"{[(c['phase'], c['ok'], c['text'][:60]) for c in out['idle_sanity'].get('checks', [])]}", flush=True)
    if ready and delta.get("_acc_dir"):
        # accuracy gate (GATE_ACCURACY=1): teacher-forced logprobs + GSM8K EM recorded right after boot, before any
        # load, into a directory shared by every arm of this gate on the weights volume; compared at the end
        try:
            import accuracy_gate
            out["accuracy"] = accuracy_gate.gate_arm(
                f"http://127.0.0.1:{PORT}", path, arm, delta["_acc_dir"], control_arm=delta.get("_acc_control", "control"),
                **({"skip_tf": "in-box PD front: prompt logprobs are computed on the prefill server and not returned"}
                   if delta.get("_pd") or delta.get("_pd_front") else {}))
            WEIGHTS.commit()
            print(f"[{arm}] accuracy recorded: {out['accuracy']}", flush=True)
        except Exception as e:  # noqa: BLE001
            out["accuracy"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"[{arm}] accuracy gate failed: {out['accuracy']['error']}", flush=True)
        # decode-path check (amdglm 2026-10-04): the TF part above scores prefill logits only and is skipped for PD
        # fronts; decode_check scores greedy decode logprobs through the same endpoint. Runs only where a prompts file
        # for THIS model exists ($GATE_DECODE_PROMPTS_DIR/decode_prompts_<model dir name>.json); GATE_DECODE_CHECK=0
        # skips it. Never fails the gate: the artifact is compared offline (operations/amdpro/drift/decode_check.py).
        dp = os.path.join(os.environ.get("GATE_DECODE_PROMPTS_DIR", "/root/amdglm_pro"),
                          f"decode_prompts_{os.path.basename(str(path).rstrip('/'))}.json")
        if os.environ.get("GATE_DECODE_CHECK", "1") != "0" and os.path.exists(dp):
            try:
                sys.path.insert(0, str(HERE.parent / "amdpro" / "drift"))
                import decode_check
                dout = os.path.join(delta["_acc_dir"], f"decode_{arm}.json")
                out["decode_check"] = decode_check.record(f"http://127.0.0.1:{PORT}", "", "", dout, prompts_file=dp)
                print(f"[{arm}] decode check recorded: {dout} {out['decode_check']}", flush=True)
            except Exception as e:  # noqa: BLE001
                out["decode_check"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                print(f"[{arm}] decode check failed: {out['decode_check']['error']}", flush=True)
    if os.environ.get("GATE_MONITOR") == "1" or delta.get("_monitor"):
        _monitor(arm)
    if not ready:
        out["root_cause"] = _cause(text)
        for ln in out["root_cause"]:
            print(f"[{arm} cause] {ln[:250]}", flush=True)
        proc.kill()
        return out
    if delta.get("_sps_profile"):
        # SGLang's own DSpark step-cost profiler against this server (launched with SGLANG_SIMULATE_ACC_LEN=1.0, the
        # SPS record and static verify): steps/s as a function of verify rows per DP rank, fitted into the table
        # that --speculative-dspark-sps-table-path loads for the confidence-budgeted (compact) verify
        os.makedirs(os.path.dirname(sps_path), exist_ok=True)
        cmd_p = [sys.executable, "-m", "sglang.benchmark.dspark_sps_profiler", "all", "--base-url",
                 f"http://127.0.0.1:{PORT}", "--out", sps_path, "--max-batch-size",
                 str(delta.get("_sps_max_bs", 32)), "--round-timeout", "900", "--no-plot"]
        t1 = time.time()
        r = subprocess.run(cmd_p, capture_output=True, text=True, timeout=7200)
        tail = (r.stdout + r.stderr)[-6000:]
        out["sps_profile"] = {"rc": r.returncode, "minutes": round((time.time() - t1) / 60, 1), "path": sps_path,
                              "table": open(sps_path).read() if os.path.exists(sps_path) else None, "tail": tail}
        print(f"[{arm}] sps profile rc={r.returncode} in {out['sps_profile']['minutes']} min -> {sps_path}", flush=True)
        for ln in tail.splitlines()[-25:]:
            print(f"[{arm} sps] {ln[:240]}", flush=True)
        if os.path.exists(sps_path):
            WEIGHTS.commit()
        _stop(proc)
        return out
    try:
        render = _renderer(path)
        sessions = json.load(open(f"/root/{pathlib.Path(sessions_file).name}"))
        # a few recorded trajectories carry enormous tool outputs (one rendered to 1.38M GLM tokens); drop sessions
        # whose last turn could exceed any served context, so one outlier cannot fail a whole arm
        cap = int(os.environ.get("GATE_MAX_CHARS", "600000"))
        before = len(sessions)
        sessions = [x for x in sessions if sum(len(m.get("content") or "") for m in x["messages"][:x["turn_ends"][-1]]) <= cap]
        out["sessions_dropped_oversize"] = before - len(sessions)
        # an in-box PD cell has dpSize None: pin across its prefill DP ranks (inboxPd.dp), else the PD proxy picks a
        # random prefill rank per turn and every warm turn misses its radix/HiCache (arm P 2026-10-05: hit 0.44)
        url, dp = f"http://127.0.0.1:{PORT}", int(delta.get("dp_size") or cfg.get("dpSize")
                                                  or (cfg.get("inboxPd") or {}).get("dp") or 1)
        pin_dp = dp if not delta.get("no_pin") else 1   # no_pin: requests carry no routed_dp_rank
        sampling = ({"_diffusion": True} if "diffusion" in repo.lower()
                    else SAMPLING.get(cell["model"], {"temperature": 0.6, "top_p": 0.95}))
        cache: dict[int, list[list[int]]] = {}

        def prompts(idx: list[int], turns: int | None = None):
            res = []
            for i in idx:
                if i not in cache:
                    s = sessions[i]
                    cache[i] = [render(s["messages"][:e]) for e in s["turn_ends"]]
                res.append(cache[i] if turns is None else cache[i][:turns])
            return res

        n_sess = len(sessions)
        for rep in range(repeats):
            for n in ladder:
                if 2 * n > n_sess:
                    print(f"[{arm}] skip {n}: needs {2 * n} sessions, have {n_sess}", flush=True)
                    continue
                # GATE_SAME_SESSIONS=1: every repeat measures the same sessions, so the spread between repeats is the
                # engine's and not the traffic's (real sessions differ a lot in prompt and answer length)
                same = os.environ.get("GATE_SAME_SESSIONS") == "1"
                base = 0 if same else (rep * 97) % max(1, n_sess - 2 * n)
                measured = list(range(base, base + n))
                warm = [n_sess - 1 - k for k in range(n)]
                _flush(url + flush)
                _rung(url, cfg["engine"], pin_dp, prompts(warm, 1), sampling)
                flushed = _flush(url + flush)
                if delta.get("_expert_record"):
                    _post(url + "/start_expert_distribution_record")
                start = os.path.getsize(logf)
                row = {"repeat": rep + rep_offset, "sessions": n, "stagger_s": float(delta.get("_stagger_s") or 0),
                       "flush_ok": flushed,
                       **_rung(url, cfg["engine"], pin_dp, prompts(measured), sampling,
                               float(delta.get("_stagger_s") or 0))}
                if delta.get("_expert_record"):
                    row["expert_load"] = _expert_record_dump(
                        url, (delta.get("env") or {}).get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR", "/tmp"),
                        int(delta.get("_ep") or 8))
                with open(logf, "rb") as f:
                    raw = f.read()
                head, seg = raw[:start].decode(errors="ignore"), raw[start:].decode(errors="ignore")
                acc = [float(a or b) for a, b in ACCEPT.findall(seg)]
                row["accept_len"] = round(sum(acc) / len(acc), 2) if acc else None
                radix = _radix_delta(head, seg)
                if radix:
                    row["radix"] = radix
                out["runs"].append(row)
                print(f"[{arm}] {json.dumps(row)}", flush=True)
        prod = delta.get("_prod_shape") or (json.loads(os.environ["GATE_PROD_SHAPE"])
                                            if os.environ.get("GATE_PROD_SHAPE") else None)
        if prod:
            # info row (amdglm 9g): closed-loop N on the OR mix; not part of the gate (operator: no arm passes on it)
            stats = [session_mix(x) for x in sessions]
            tu = (OR_MIX.get(cell["model"]) or OR_MIX["glm-5.3"])[0]
            weights, mix = mix_weights(stats, tu)
            out["prod_shape"] = []
            for n in prod.get("sessions") or [ladder[-1]]:
                _flush(url + flush)
                row = {"mix": mix, "mix_target_uncached_per_out": tu,
                       **_prod_rung(url, cfg["engine"], pin_dp, lambda i: prompts([i])[0], weights, int(n),
                                    float(prod.get("warmup_s", 180)), float(prod.get("window_s", 600)), sampling,
                                    float(prod.get("stagger_s", 60)))}
                out["prod_shape"].append(row)
                print(f"[{arm}] prod_shape {json.dumps(row)}", flush=True)
        if profile_steps and 2 * ladder[-1] <= n_sess:
            # real-traffic profile: one more measured round at the largest step, profiler armed shortly after it
            # starts so it records warm turns; GPU kernels only (Python stacks inflate CPU time)
            import glob
            import threading
            import urllib.request
            n = ladder[-1]
            idx = list(range(n_sess - 2 * n, n_sess - n))
            _post(url + flush)
            _rung(url, cfg["engine"], pin_dp, prompts(idx, 1), sampling)   # warm the prefixes
            done = {}
            th = threading.Thread(target=lambda: done.update(
                _rung(url, cfg["engine"], pin_dp, prompts(idx), sampling)), daemon=True)
            th.start()
            time.sleep(8)
            body = json.dumps({"output_dir": "/root/prof", "num_steps": profile_steps, "activities": ["GPU"],
                               "with_stack": False, "record_shapes": False}).encode()
            urllib.request.urlopen(urllib.request.Request(url + "/start_profile", data=body, method="POST",
                                                          headers={"content-type": "application/json"}), timeout=120)
            th.join(timeout=3600)
            t_wait = time.time()
            while time.time() - t_wait < 600:
                fs = glob.glob("/root/prof/**/*.trace.json*", recursive=True)
                if len(fs) >= dp and time.time() - max(os.path.getmtime(f) for f in fs) > 20:
                    break
                time.sleep(10)
            import trace_summary
            summ = trace_summary.summarize("/root/prof")
            out["profile"] = {"sessions": n, "steps": profile_steps, "bench": done, "summary": summ}
            print(f"[{arm}] profile families {summ['families']} compute imbalance {summ['compute_imbalance']} "
                  f"compute by rank {summ['compute_ms_by_rank']}", flush=True)
            for r in summ["ranks"]:
                print(f"[{arm}]   rank {r['file'][-40:]} busy {r['busy_ms']} of {r['span_ms']} ms, compute {r['compute_ms']} ms, "
                      f"collective calls {r['collective_call_ms']}", flush=True)
            for k in summ["top"][:20]:
                print(f"[{arm}]   {k['share']:.3f} {k['family']:10s} {k['ms']:9.1f} ms x{k['count']:6d} {k['name'][:110]}", flush=True)
        if delta.get("_phase_profile"):
            # clean decode and prefill profiles at the sold context: the kernel budget per step, per phase
            n_dec = ladder[-1]
            first = [render(x["messages"][:x["turn_ends"][0]]) for x in sessions[:max(n_dec, 8 * dp * 6)]]
            out["phase_profile"] = {}
            for ph in delta["_phase_profile"]:
                _post(url + flush)
                out["phase_profile"][ph] = _phase_profile(
                    url, pin_dp, first if ph == "decode" else first[n_dec:], n_dec, int(delta.get("_phase_steps", 30)),
                    ph, arm)
        if probes:
            out["probes"] = _probes(url, cfg["engine"], render, sampling)
            print(f"[{arm}] probes {out['probes']['score']} per probe {out['probes']['per_probe']}", flush=True)
        if dp > 1:
            out["dp_peak_running"] = _dp_distribution(open(logf, errors="ignore").read())
            print(f"[{arm}] peak running requests per DP rank {out['dp_peak_running']}", flush=True)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        print(f"[{arm}] error {out['error']}", flush=True)
        out["root_cause"] = _cause(open(logf, errors="ignore").read())
        for ln in out["root_cause"]:
            print(f"[{arm} cause] {ln[:250]}", flush=True)
    finally:
        out["numinous_log"] = _numinous_lines(logf)
        out["spec_acceptance"] = _spec_acceptance(logf)
        for ln in out["numinous_log"]:
            print(f"[{arm} numinous] {ln[:250]}", flush=True)
        _stop(proc)
    return out


def _spec_acceptance(logf: str) -> dict:
    """Speculative decoding acceptance from the server's decode lines ("accept len: X, accept rate: Y"): the number
    that separates a draft problem from an engine problem (a collapsed draft halves throughput and looks like a slow
    kernel). Mean / p10 / p50 / p90 over every decode line of the run."""
    import re
    lens, rates = [], []
    try:
        for ln in open(logf, errors="ignore"):
            m = re.search(r"accept len: ([0-9.]+), accept rate: ([0-9.]+)", ln)
            if m:
                lens.append(float(m.group(1)))
                rates.append(float(m.group(2)))
    except OSError:
        return {}
    if not lens:
        return {"samples": 0}
    q = sorted(lens)

    def pct(v):
        return round(q[min(len(q) - 1, int(v * len(q)))], 3)
    return {"samples": len(lens), "accept_len_mean": round(sum(lens) / len(lens), 3), "accept_len_p10": pct(0.1),
            "accept_len_p50": pct(0.5), "accept_len_p90": pct(0.9), "accept_rate_mean": round(sum(rates) / len(rates), 3)}


def _numinous_lines(logf: str, limit: int = 12) -> list:
    """Our kernels' own log lines from the server log (activation, declines, prepared buffers): the evidence that
    an overlay arm actually ran its path, recorded with every arm."""
    try:
        seen, out = set(), []
        for ln in open(logf, errors="ignore"):
            if "numinous" in ln.lower():
                key = ln.split("] ", 1)[-1].strip()[:160]
                if key not in seen:
                    seen.add(key)
                    out.append(ln.strip())
        return out[:limit]
    except OSError:
        return []


PD_PREFILL_PORT, PD_DECODE_PORT, PD_BOOTSTRAP_PORT = PORT + 100, PORT + 200, 8998


def _drop_all(flags: list[str], names: set) -> list[str]:
    """flags without every occurrence of the named flags (and their values)."""
    out, i = [], 0
    while i < len(flags):
        if flags[i] in names:
            i += 2 if i + 1 < len(flags) and not flags[i + 1].startswith("--") else 1
            continue
        out.append(flags[i])
        i += 1
    return out


def pd_launch(path: str, flags: list[str], pd: dict) -> dict:
    """Commands of an in-box PD arm (delta key _pd). The arm's composed flags are the base of both roles; each role
    applies its own flags_remove / flags_add / env on top, then the disaggregation flags: prefill on GPUs
    0..base-1 (--base-gpu-id 0) with the bootstrap port, decode on GPUs base.. (--base-gpu-id base). Ports are
    PORT+100 / PORT+200 (SGLang binds ZMQ ports at --port + 234.., so two engines need ~100 apart; serving.ts
    ENGINE_PORT_STRIDE). The front is the gateway's own pd_proxy.py (rooms congruent to the session's DP rank)."""
    backend = pd.get("transfer", "mooncake")
    base = int(pd.get("decode_base", 4))
    out = {}
    for role, port, gpu in (("prefill", PD_PREFILL_PORT, 0), ("decode", PD_DECODE_PORT, base)):
        r = pd.get(role) or {}
        # every occurrence of a removed flag goes (the arm's composition can repeat a flag, e.g. --ep-size), then the
        # role's additions
        drop = set(r.get("flags_remove", [])) | {"--disaggregation-mode", "--disaggregation-transfer-backend",
                                                 "--disaggregation-bootstrap-port", "--base-gpu-id"}
        f = _drop_all(compose_flags(flags, {}, path), drop) + list(r.get("flags_add", []))
        f += ["--disaggregation-mode", role, "--disaggregation-transfer-backend", backend, "--base-gpu-id", str(gpu)]
        if role == "prefill":
            f += ["--disaggregation-bootstrap-port", str(PD_BOOTSTRAP_PORT)]
        out[role] = {"port": port, "flags": f, "env": dict(r.get("env") or {}),
                     "cmd": ["python3", "-m", "sglang.launch_server", "--model-path", path, "--host", "127.0.0.1",
                             "--port", str(port)] + f}
    out["proxy"] = {"cmd": [sys.executable, str(HERE / "gateway_pd_proxy.py") if LOCAL else "/root/gateway_pd_proxy.py",
                            "--host", "0.0.0.0", "--port", str(PORT), "--prefill-port", str(PD_PREFILL_PORT),
                            "--decode-port", str(PD_DECODE_PORT), "--bootstrap-port", str(PD_BOOTSTRAP_PORT),
                            "--dp", str(int(pd.get("dp", 4)))]}
    return out


class PDGroup:
    """The prefill server, the decode server and the proxy as one 'process' for _run_arm / _stop: poll() reports the
    first that died (a dead role fails the boot at once instead of waiting out the health timeout), kill() and
    _stop() take down all three process groups. The decode log is the arm's server log (accept lengths, decode
    batches); the prefill log sits next to it."""

    def __init__(self, cmds: dict, logf: str, env: dict):
        self.logs = {"decode": logf, "prefill": logf.replace(".log", "_prefill.log"),
                     "proxy": logf.replace(".log", "_proxy.log")}
        self.procs = []
        for role in ("prefill", "decode"):
            c = cmds[role]
            self.procs.append(subprocess.Popen(c["cmd"], stdout=open(self.logs[role], "w"), stderr=subprocess.STDOUT,
                                               env={**env, **c["env"]}, start_new_session=True))
        self.procs.append(subprocess.Popen(cmds["proxy"]["cmd"], stdout=open(self.logs["proxy"], "w"),
                                           stderr=subprocess.STDOUT, env=env, start_new_session=True))
        self.pid = self.procs[0].pid

    def poll(self):
        for p in self.procs:
            rc = p.poll()
            if rc is not None:
                return rc
        return None

    def kill(self) -> None:
        for p in self.procs:
            try:
                p.kill()
            except ProcessLookupError:
                pass

    def wait(self, timeout=None):
        for p in self.procs:
            p.wait(timeout=timeout)


class DualGroup(PDGroup):
    """2xDP4 (amdpro 2026-10-05): two independent monolithic servers on disjoint GPUs plus operations/modal/dual_router.py
    as one 'process', the same surface as PDGroup (poll / kill / wait; _stop takes it down as a PDGroup). Server 1's
    log is the arm's server log; server 0's and the router's sit next to it."""

    def __init__(self, cmds: dict, logf: str, env: dict):  # noqa: D107  (PDGroup.__init__ is replaced, not extended)
        self.logs = {"server1": logf, "server0": logf.replace(".log", "_server0.log"),
                     "proxy": logf.replace(".log", "_proxy.log")}
        self.procs = []
        for role in ("server0", "server1"):
            c = cmds[role]
            self.procs.append(subprocess.Popen(c["cmd"], stdout=open(self.logs[role], "w"), stderr=subprocess.STDOUT,
                                               env={**env, **c["env"]}, start_new_session=True))
        self.procs.append(subprocess.Popen(cmds["proxy"]["cmd"], stdout=open(self.logs["proxy"], "w"),
                                           stderr=subprocess.STDOUT, env=env, start_new_session=True))
        self.pid = self.procs[0].pid


def _stop(proc) -> None:
    """Kill the server's whole process group (schedulers, detokenizer) and wait until the GPUs are free, so the next
    arm in a sequential run boots on a clean device."""
    import signal
    if isinstance(proc, PDGroup):
        for p in proc.procs:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(p.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                if sig == signal.SIGTERM:
                    try:
                        p.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        pass
        proc = proc.procs[0]
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    for _ in range(60):
        used = _gpu_mem_used_mib()
        if not used or max(used) < 2048:
            break
        time.sleep(5)
    _kill_leftovers()
    busy = _wait_ports_free(PORT, PORT + 400)
    if busy:
        print(f"[stop] ports still bound after teardown: {busy[:8]}", flush=True)


def _kill_leftovers(marker: str = "sglang") -> list:
    """SIGKILL every process whose command line names the engine and is not this one: scheduler / detokenizer children
    that left the server's process group kept their RPC port, and the next sequential boot failed with "rpc_port at
    30236 is not available in 30 seconds" (gate-profast-7f3a)."""
    import signal
    me, killed = os.getpid(), []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if marker in cmd and "cell_gate" not in cmd and "pytest" not in cmd:
            # a box shared by two gates on disjoint GPUs: only kill engine processes started with our GPU set
            mine = os.environ.get("CUDA_VISIBLE_DEVICES")
            if mine is not None:
                try:
                    penv = open(f"/proc/{d}/environ", "rb").read().split(b"\0")
                except OSError:
                    continue
                theirs = next((e.split(b"=", 1)[1].decode() for e in penv if e.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
                if theirs != mine:
                    continue
            try:
                os.kill(int(d), signal.SIGKILL)
                killed.append(int(d))
            except (ProcessLookupError, PermissionError):
                pass
    return killed


def _wait_ports_free(lo: int = 30000, hi: int = 30400, timeout_s: float = 120.0) -> list:
    """Wait until every port the server family uses (HTTP 30000 and the RPC / ZMQ ports above it) can be bound again;
    returns the ports still busy at the deadline."""
    import socket
    deadline = time.time() + timeout_s
    while True:
        busy = []
        for port in range(lo, hi):
            sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sk.bind(("", port))
            except OSError:
                busy.append(port)
            finally:
                sk.close()
        if not busy or time.time() > deadline:
            return busy
        time.sleep(2)


def _node() -> dict:
    """Who the node is: B300:8 nodes differ by ~33% on identical launches and sessions (rtxattn 2026-10-03: shipped Pro
    @8xB300, 79.4 vs 59.6 per user at 64 sessions on two nodes), so every arm records what could tell them apart."""
    import socket
    out = {"host": socket.gethostname(), "gpus": _gpu_ids(), "nproc": os.cpu_count()}

    def sh(cmd: str, limit: int = 4000) -> str:
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()[:limit]
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"
    out["gpu_info"] = sh("nvidia-smi --query-gpu=index,name,driver_version,vbios_version,pci.bus_id,power.limit,"
                         "power.max_limit,clocks.max.sm,clocks.max.mem,ecc.mode.current,pcie.link.gen.current,"
                         "pcie.link.width.current --format=csv,noheader")
    out["topo"] = sh("nvidia-smi topo -m", 3000)
    out["cpu"] = sh("grep -m1 'model name' /proc/cpuinfo; grep -c ^processor /proc/cpuinfo; uname -r")
    out["cgroup_cpu"] = sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null")
    return out


def _run_seq(cell: dict, arms: list, ladder: list[int], repeats: int, sessions_file: str, probes: bool = False,
             profile_steps: int = 0) -> dict:
    """Every arm of the group on this one node, one after another, alternating the order each repeat (A B C, C B A):
    separate containers land on different nodes, and a slow node moved identical-session results by ~35%."""
    results: dict = {}
    if repeats == 0:
        # one-shot arms (phase profiles, SPS tables): each arm once on this node, no ladder repeats
        for label, delta in arms:
            results[label] = _run_arm(cell, label, delta, ladder, 0, sessions_file, probes=probes,
                                      profile_steps=profile_steps)
        return results
    for rep in range(repeats):
        order = arms if rep % 2 == 0 else list(reversed(arms))
        for label, delta in order:
            r = _run_arm(cell, label, delta, ladder, 1, sessions_file,
                         probes=probes and rep == repeats - 1, profile_steps=profile_steps if rep == 0 else 0,
                         rep_offset=rep)
            if label not in results:
                results[label] = r
            else:
                results[label]["runs"] += r.get("runs", [])
                for k in ("probes", "error", "root_cause"):
                    if r.get(k):
                        results[label][k] = r[k]
    return results


GATE_NSYS = os.environ.get("GATE_NSYS") == "1"
GATE_CPU = float(os.environ.get("GATE_CPU", "32"))   # container vCPUs (8 schedulers, NCCL proxies, tokenizer, client)
# Nsight Systems CLI: the B300 path for kernel profiles (the torch profiler's CUPTI records nothing on SM103)
# the newest Nsight Systems in NVIDIA's repo, not the CUDA 12.9 bundle (2025.1.3 recorded nothing on B300): SM103
# support arrived in later CUPTI releases
NSYS_INSTALL = ("apt-get update -qq && apt-cache search --names-only '^nsight-systems-20' | sort -V | tail -3 && "
                "apt-get install -y -qq --no-install-recommends $(apt-cache search --names-only '^nsight-systems-20' "
                "| sort -V | tail -1 | cut -d' ' -f1) && ls -d /opt/nvidia/nsight-systems/*/ && "
                "$(ls -d /opt/nvidia/nsight-systems/*/ | sort -V | tail -1)bin/nsys --version")


def image_for_arm(cell: dict, delta: dict, sessions_file: str) -> modal.Image:
    # an arm may move the cell to another engine image (the v0.5.20 ports) and replace its patch series
    cfg = {**cell["config"], **({"patches": delta["patches"]} if "patches" in delta else {})}
    img = ri.with_patches(ri.base_image(delta.get("image") or cell["config"]["image"]), ri.patch_key(cfg))
    if GATE_NSYS:
        img = img.run_commands(NSYS_INSTALL)
    # kernel overlays: files installed over the engine package after the cell's own patches, then hook patchers
    pkg = "vllm" if cell["config"]["engine"] == "vllm" else "sglang"
    img = ri.with_overlays(img, tuple((delta.get("overlay") or {}).items()), pkg=pkg)
    return (img.add_local_file(str(HERE / sessions_file), f"/root/{pathlib.Path(sessions_file).name}")
            .add_local_python_source("registry_image", "trace_summary", "gate_arms", "accuracy_gate")
            .add_local_file(str(HERE / "dc_bench.py"), "/root/dc_bench.py")
            .add_local_file(str(HERE / "gateway_pd_proxy.py"), "/root/gateway_pd_proxy.py")
            .add_local_file(str(REGISTRY), "/root/serving_registry.json"))


# One function per (cell image, patches, overlay set, sessions file, GPU): Modal cannot swap images at call time.
GATE_CELL = os.environ.get("GATE_CELL", "deepseek-v4-pro/standard")
GATE_GPU = os.environ.get("GATE_GPU", "")
GATE_ARMS = [a for a in os.environ.get("GATE_ARMS", "control").split(",") if a in ARMS]
# GATE_BASE: arms merged into every arm, control included, once a delta has been accepted into the kept recipe (the
# next candidates are "kept + accepted + one delta"). Env and overlays merge, flag lists concatenate.
GATE_BASE = [a for a in os.environ.get("GATE_BASE", "").split(",") if a in ARMS]


def _compose(names: list[str]) -> dict:
    out: dict = {}
    for n in names:
        for k, v in ARMS[n].items():
            if k in ("env", "overlay"):
                out[k] = {**out.get(k, {}), **v}
            elif k in ("flags_add", "flags_remove"):
                out[k] = out.get(k, []) + list(v)
            else:
                out[k] = v
    return out


def _image_key(delta: dict) -> str:
    """Arms share a node image only when overlays, engine image and patch series all match."""
    return json.dumps({k: delta.get(k) for k in ("overlay", "image", "patches")}, sort_keys=True)


DELTAS = {a: _compose(GATE_BASE + [a]) for a in GATE_ARMS}
LABELS = {a: "+".join(GATE_BASE + ([a] if a != "control" or not GATE_BASE else [])) or a for a in GATE_ARMS}
FUNCS: dict[str, object] = {}
# GATE_SEQUENTIAL=1: arms that share an overlay set run on one node one after another (see _run_seq)
GATE_SEQUENTIAL = os.environ.get("GATE_SEQUENTIAL") == "1"
SEQ_FUNCS: dict[str, object] = {}
# inside a container the local GATE_* env is absent: GATE_CELL falls back to its default, which a gate-only registry
# (GATE_REGISTRY, shipped as /root/serving_registry.json) does not hold. The container only needs this module to
# import (it runs the module-level _run_arm / _run_seq), so a cell it cannot find skips the function definitions
# (k3amd flashb300 2026-10-03: StopIteration at import killed the B300:2 run after 58 min in the queue)
_CELL_FOUND = any(f"{c['model']}/{c['lane']}" == GATE_CELL for c in (cells() if REGISTRY.exists() else []))
if REGISTRY.exists() and not LOCAL and _CELL_FOUND:
    _cell = cell_by_key(GATE_CELL)
    # GATE_SESSIONS overrides the lane's sessions file (e.g. the 8-turn builds for production-like warm-turn mixes)
    _sessions = os.environ.get("GATE_SESSIONS") or SESSIONS.get(GATE_CELL, SESSIONS["default"])
    _gpu = ri.gpu_for(_cell, GATE_GPU)
    for _arm in GATE_ARMS:
        _key = _image_key(DELTAS[_arm])
        if _key not in FUNCS:
            FUNCS[_key] = app.function(gpu=_gpu, timeout=6 * 3600, volumes={CACHE: WEIGHTS}, cpu=GATE_CPU,
                                       memory=786432 if ":" in _gpu else 262144, ephemeral_disk=524288,
                                       image=image_for_arm(_cell, DELTAS[_arm], _sessions),
                                       name=f"gate_{len(FUNCS)}")(_run_arm)
        if GATE_SEQUENTIAL and _key not in SEQ_FUNCS:
            # all arms sharing this overlay set run on one node, sequentially (24 h: every arm boots per repeat)
            SEQ_FUNCS[_key] = app.function(gpu=_gpu, timeout=24 * 3600, volumes={CACHE: WEIGHTS}, cpu=GATE_CPU,
                                           memory=786432 if ":" in _gpu else 262144, ephemeral_disk=524288,
                                           image=image_for_arm(_cell, DELTAS[_arm], _sessions),
                                           name=f"gate_seq_{len(SEQ_FUNCS)}")(_run_seq)


if not LOCAL:
    # the function lives in this module, so the container imports cell_gate and everything it imports at top level
    ACC_IMAGE = (modal.Image.debian_slim(python_version="3.12")
                 .add_local_python_source("accuracy_gate", "registry_image", "trace_summary", "gate_arms")
                 .add_local_file(str(REGISTRY), "/root/serving_registry.json"))

    @app.function(image=ACC_IMAGE, volumes={CACHE: WEIGHTS}, timeout=900)
    def acc_compare(acc_dir: str, control: str, arms: list) -> dict:
        """Every arm's accuracy record vs the control's (both on the weights volume): accuracy_gate's verdict."""
        import accuracy_gate as ag
        WEIGHTS.reload()

        def load(label):
            path = os.path.join(acc_dir, f"accuracy_{label}.json")
            return json.load(open(path)) if os.path.exists(path) else None

        ctl, noise_p = load(control), os.path.join(acc_dir, "accuracy_noise.json")
        noise = json.load(open(noise_p)) if os.path.exists(noise_p) else None
        out = {"acc_dir": acc_dir, "control": control, "control_found": ctl is not None, "arms": {}}
        for a in arms:
            rec = load(a)
            if ctl is None or rec is None:
                out["arms"][a] = {"error": "missing record", "control": ctl is not None, "arm": rec is not None}
                continue
            out["arms"][a] = ag.compare(ctl, rec, noise)
        return out


@app.local_entrypoint()
def main(ladder: str = "64,128", repeats: int = 3, probes: bool = False, profile_steps: int = 0):
    cell = cell_by_key(GATE_CELL)
    sessions_file = os.environ.get("GATE_SESSIONS") or SESSIONS.get(GATE_CELL, SESSIONS["default"])
    steps = [int(x) for x in ladder.split(",")]
    print(f"gate {GATE_CELL} on {ri.gpu_for(cell, GATE_GPU)}: arms {GATE_ARMS}, ladder {steps}, {repeats} repeats, "
          f"sessions {sessions_file}", flush=True)
    same = os.environ.get("GATE_SAME_SESSIONS") == "1"
    if GATE_BASE:
        print(f"base (in every arm): {GATE_BASE}", flush=True)
    extra = {"_same_sessions": same, "_probe_repeats": int(os.environ.get("GATE_PROBE_REPEATS", "1")),
             "_monitor": os.environ.get("GATE_MONITOR") == "1",
             "_max_chars": int(os.environ.get("GATE_MAX_CHARS", "600000")), "_gpu_label": GATE_GPU or "reg",
             "_stagger_s": float(os.environ.get("GATE_STAGGER_S", "0")), "_nsys": GATE_NSYS,
             # the output shape travels with the arm: the container does not see this process's env (a launch with
             # GATE_MAX_OUT / GATE_IGNORE_EOS set locally measured the default 400 / EOS shape before 2026-10-02)
             "_max_out": MAX_OUT, "_ignore_eos": IGNORE_EOS}
    # GATE_ACCURACY=1: every arm records the accuracy gate after boot; the first arm is the control
    acc = os.environ.get("GATE_ACCURACY") == "1"
    if acc:
        acc_dir = f"{CACHE}/accuracy/{cell['model']}_{cell['lane']}_{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
        extra.update(_acc_dir=acc_dir, _acc_control=LABELS[GATE_ARMS[0]])
        print(f"accuracy gate: control {LABELS[GATE_ARMS[0]]}, records in {acc_dir}", flush=True)

    def _accuracy_report() -> None:
        if not acc:
            return
        res = acc_compare.remote(acc_dir, LABELS[GATE_ARMS[0]], [LABELS[a] for a in GATE_ARMS[1:]])
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        path = ROOT / f"research/gpu-evidence/gate/{cell['model']}_{cell['lane']}_accuracy_{stamp}.json"
        path.write_text(json.dumps(res, indent=1, default=str))
        for a, r in res["arms"].items():
            v = r.get("verdict", r)
            print(f"[accuracy] {a} vs {res['control']}: {json.dumps(v, default=str)[:400]}", flush=True)
        print(f"[accuracy] evidence: {path}", flush=True)

    if GATE_SEQUENTIAL:
        groups: dict[str, list[str]] = {}
        for a in GATE_ARMS:
            groups.setdefault(_image_key(DELTAS[a]), []).append(a)
        print(f"sequential: {len(groups)} node(s), arms per node {list(groups.values())}", flush=True)
        seq = {k: SEQ_FUNCS[k].spawn(cell, [(LABELS[a], {**DELTAS[a], **extra}) for a in g], steps, repeats,
                                     sessions_file, probes, profile_steps) for k, g in groups.items()}
        by_label = {}
        for k, h in seq.items():
            try:
                by_label.update(h.get())
            except Exception as e:  # noqa: BLE001
                for a in groups[k]:
                    by_label[LABELS[a]] = {"arm": LABELS[a], "error": f"{type(e).__name__}: {str(e)[:300]}"}
        _report(steps, {a: by_label.get(LABELS[a], {"arm": LABELS[a], "error": "missing"}) for a in GATE_ARMS})
        _accuracy_report()
        return
    # one function per image key (overlay + engine image + patch series), the same key the functions were built with
    handles = {a: FUNCS[_image_key(DELTAS[a])].spawn(cell, LABELS[a], {**DELTAS[a], **extra}, steps, repeats,
                                                     sessions_file, probes, profile_steps)
               for a in GATE_ARMS}
    results = {}
    for a, h in handles.items():
        try:
            results[a] = h.get()
        except Exception as e:  # noqa: BLE001
            results[a] = {"arm": a, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    _report(steps, results)
    _accuracy_report()


def _report(steps: list[int], results: dict) -> None:
    """Evidence per arm (research/gpu-evidence/gate/) and the summary table."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "research/gpu-evidence/gate"
    outdir.mkdir(parents=True, exist_ok=True)
    for a, r in results.items():
        (outdir / f"{GATE_CELL.replace('/', '_')}_{GATE_GPU or 'reg'}_{LABELS.get(a, a)}_{stamp}.json").write_text(
            json.dumps(r, indent=1) + "\n")
    print(f"\n{'arm':28s} " + "  ".join(f"{n:>22d}" for n in steps) + "   (warm per-user tok/s | aggregate, mean of repeats)")
    for a, r in results.items():
        a = LABELS.get(a, a)
        cells_txt = []
        for n in steps:
            rows = [x for x in r.get("runs", []) if x["sessions"] == n]
            pu = [x["warm_per_user_tok_s"] for x in rows if x.get("warm_per_user_tok_s")]
            ag = [x["warm_agg_out_tok_s"] for x in rows if x.get("warm_agg_out_tok_s")]
            st = [x["steady_agg_out_tok_s"] for x in rows if x.get("steady_agg_out_tok_s")]
            cells_txt.append(f"{statistics.mean(pu):6.1f}+-{(statistics.stdev(pu) if len(pu) > 1 else 0):4.1f} | "
                             f"{statistics.mean(ag):6.0f}" + (f" st{statistics.mean(st):5.0f}" if st else "")
                             if pu and ag else f"{'-':>22s}")
        print(f"{a:28s} " + "  ".join(f"{c:>22s}" for c in cells_txt) + (f"  probes {r['probes']['score'] if isinstance(r['probes'], dict) else r['probes']}" if r.get("probes") else "")
              + (f"  {r.get('error', '')[:60]}" if r.get("error") else ""))


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
