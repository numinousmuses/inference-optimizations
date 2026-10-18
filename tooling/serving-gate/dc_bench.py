"""Session bench for datacenter cells: N concurrent sessions, each with its own prefix, several turns.

Why not sglang.bench_serving's generated-shared-prefix: it shares one prefix across a group of requests, so
"running requests" overstates how many distinct sessions the KV pool holds, and it cannot pin a session to a
data-parallel rank. Real traffic is N users with N different contexts; consecutive turns of one user must land
on the rank that holds that user's KV.

Protocol per session (all sessions run concurrently):
  turn 1: prompt = prefix (P unique random token ids), generate O tokens          [cold prefill]
  turn t: prompt = previous prompt + previous output + Q new token ids, generate O [warm: prefix cached]
Every request goes to SGLang's /generate with input_ids (exact lengths, no tokenizer) and, when dp_size > 1,
routed_dp_rank = session % dp_size so a session's turns share a rank and its radix cache.

Metrics (warm turns only, unless stated): aggregate output tok/s over the warm window, per-user decode tok/s
p50/p99 (output tokens / (request wall - TTFT)), TTFT p50/p99, hit rate = sum(cached_tokens) / sum(prompt
tokens) from the server's meta_info, sessions completed, errors. The cold turn reports prefill tok/s.

Pure asyncio + aiohttp; the summary functions are pure so they are unit-tested on CPU.
"""
from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TurnResult:
    session: int
    turn: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    ttft_s: float
    wall_s: float
    t_start: float
    t_end: float
    error: Optional[str] = None

    @property
    def decode_tok_s(self) -> Optional[float]:
        d = self.wall_s - self.ttft_s
        if self.completion_tokens <= 1 or d <= 0:
            return None
        return (self.completion_tokens - 1) / d


@dataclass
class BenchSpec:
    sessions: int
    prefix_tokens: int
    turns: int = 3
    question_tokens: int = 2048
    output_tokens: int = 1000
    dp_size: int = 1
    vocab: int = 100000
    seed: int = 1
    ramp_s: float = 0.0          # stagger session starts across this many seconds (0 = all at once)
    backend: str = "sglang"      # "sglang": /generate with input_ids; "vllm": /v1/completions with a token-id prompt
    model: str = ""              # vllm needs the served model name
    stream: bool = True          # diffusion samplers answer whole canvases: use stream=False and take TTFT = wall
    think_s: float = 0.0         # pause between a session's turns; long enough and its KV is evicted before it returns
    concurrency: int = 0         # cap on sessions in flight at once (0 = all). A pool larger than the cap models
                                 # real agent traffic: many live sessions, few talking at any moment.


def _pct(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[TurnResult], spec: BenchSpec) -> dict:
    """Pure summary. Warm = turns >= 2. Aggregate rate uses the wall window of the warm turns only."""
    ok = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]
    cold = [r for r in ok if r.turn == 1]
    warm = [r for r in ok if r.turn >= 2]

    def agg(rows):
        if not rows:
            return None
        t0, t1 = min(r.t_start for r in rows), max(r.t_end for r in rows)
        out = sum(r.completion_tokens for r in rows)
        return round(out / (t1 - t0), 1) if t1 > t0 else None

    def per_user(rows):
        xs = [r.decode_tok_s for r in rows if r.decode_tok_s]
        return {"p50": round(_pct(xs, 0.5), 1) if xs else None, "p99": round(_pct(xs, 0.01), 1) if xs else None}

    def ttft(rows):
        xs = [r.ttft_s for r in rows]
        return {"p50_s": round(_pct(xs, 0.5), 2) if xs else None, "p99_s": round(_pct(xs, 0.99), 2) if xs else None}

    def hit(rows):
        p = sum(r.prompt_tokens for r in rows)
        c = sum(r.cached_tokens for r in rows)
        return round(c / p, 4) if p else None

    prefill_tok_s = None
    if cold:
        t0, t1 = min(r.t_start for r in cold), max(r.t_start + r.ttft_s for r in cold)
        new = sum(r.prompt_tokens - r.cached_tokens for r in cold)
        prefill_tok_s = round(new / (t1 - t0), 1) if t1 > t0 else None

    return {
        "spec": asdict(spec),
        "sessions_completed": len({r.session for r in ok if r.turn == spec.turns}),
        "requests_ok": len(ok), "requests_failed": len(errors),
        "error_sample": sorted({(e.error or "")[:160] for e in errors})[:3],
        "warm": {"agg_out_tok_s": agg(warm), "per_user_tok_s": per_user(warm), "ttft": ttft(warm), "hit_rate": hit(warm),
                 "prompt_tokens_mean": round(statistics.mean(r.prompt_tokens for r in warm)) if warm else None},
        "cold": {"prefill_tok_s": prefill_tok_s, "ttft": ttft(cold), "hit_rate": hit(cold)},
        "all": {"agg_out_tok_s": agg(ok)},
    }


def make_prefix(rng: random.Random, n: int, vocab: int) -> list[int]:
    # token ids in [1000, vocab) avoid specials at the bottom of most vocabularies
    return [rng.randrange(1000, vocab) for _ in range(n)]


def _request(spec: BenchSpec, input_ids: list[int], max_new: int, dp_rank: Optional[int]) -> tuple[str, dict]:
    """(path, body) for the backend. Both stream so TTFT is observable; both return token accounting in the last chunk."""
    if spec.backend == "vllm":
        body = {"model": spec.model, "prompt": input_ids, "max_tokens": max_new, "stream": spec.stream}
        if spec.stream:
            # autoregressive engines: pin the length and greedy decode
            body.update({"min_tokens": max_new, "temperature": 0.0, "ignore_eos": True, "stream_options": {"include_usage": True}})
        # non-streaming is used for diffusion samplers, which reject temperature / min_tokens / seed: max_tokens only
        return "/v1/completions", body
    body = {"input_ids": input_ids, "stream": True,
            "sampling_params": {"max_new_tokens": max_new, "min_new_tokens": max_new, "temperature": 0.0, "ignore_eos": True}}
    if dp_rank is not None:
        body["routed_dp_rank"] = dp_rank
    return "/generate", body


def _meta_from_chunk(spec: BenchSpec, chunk: dict, meta: dict) -> dict:
    if spec.backend == "vllm":
        u = chunk.get("usage")
        if u:
            det = u.get("prompt_tokens_details") or {}
            return {"prompt_tokens": u.get("prompt_tokens"), "cached_tokens": det.get("cached_tokens", 0), "completion_tokens": u.get("completion_tokens")}
        return meta
    return chunk.get("meta_info", meta)


async def _generate(session_http, base_url: str, spec: BenchSpec, input_ids: list[int], max_new: int, dp_rank: Optional[int]) -> tuple[dict, float, float]:
    """Streamed request; returns (final meta, ttft_s, wall_s)."""
    path, body = _request(spec, input_ids, max_new, dp_rank)
    t0 = time.perf_counter()
    ttft = None
    meta: dict = {}
    if not spec.stream:
        if spec.backend != "vllm":
            body["stream"] = False
        async with session_http.post(base_url + path, json=body) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"http {resp.status}: {txt[:200]}")
            j = await resp.json()
        wall = time.perf_counter() - t0
        if "error" in j and "choices" not in j:
            raise RuntimeError(f"server error: {json.dumps(j)[:200]}")
        meta = _meta_from_chunk(spec, j, {})
        if not meta.get("completion_tokens"):
            raise RuntimeError(f"no completion accounting: {json.dumps(j)[:200]}")
        return meta, wall, wall
    async with session_http.post(base_url + path, json=body) as resp:
        if resp.status != 200:
            txt = await resp.text()
            raise RuntimeError(f"http {resp.status}: {txt[:200]}")
        async for raw in resp.content:
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
            if "error" in chunk and "choices" not in chunk:
                raise RuntimeError(f"server error: {json.dumps(chunk)[:200]}")
            # vLLM's usage-only final chunk has an empty choices list and must not count as the first token
            if ttft is None and (spec.backend != "vllm" or chunk.get("choices")):
                ttft = time.perf_counter() - t0
            meta = _meta_from_chunk(spec, chunk, meta)
            if "_sample" not in meta and chunk.get("choices"):
                c0 = chunk["choices"][0]
                meta["_sample"] = (c0.get("text") or (c0.get("delta") or {}).get("content") or "")[:80]
    wall = time.perf_counter() - t0
    if not meta.get("completion_tokens"):
        # no accounting came back: do not invent one; report what the server said
        raise RuntimeError(f"no completion accounting; sample={meta.get('_sample')!r} keys={sorted(k for k in meta if k != '_sample')}")
    return meta, (ttft if ttft is not None else wall), wall


async def run_session(session_http, base_url: str, sid: int, spec: BenchSpec, results: list[TurnResult], t_origin: float,
                      out_ids_fn=None, gate=None) -> None:
    rng = random.Random(spec.seed * 1_000_003 + sid)
    prompt = make_prefix(rng, spec.prefix_tokens, spec.vocab)
    dp_rank = (sid % spec.dp_size) if spec.dp_size > 1 else None
    if spec.ramp_s > 0:
        await asyncio.sleep(rng.random() * spec.ramp_s)
    for turn in range(1, spec.turns + 1):
        if spec.think_s and turn > 1:
            # the session thinks; if the pause outlives its KV in GPU memory the next turn is a cache miss unless
            # the engine can fetch the prefix back from somewhere cheaper than recomputing it
            await asyncio.sleep(spec.think_s * (0.5 + rng.random()))
        if gate is not None:
            await gate.acquire()
        t_start = time.perf_counter() - t_origin
        try:
            meta, ttft, wall = await _generate(session_http, base_url, spec, prompt, spec.output_tokens, dp_rank)
            r = TurnResult(sid, turn, int(meta.get("prompt_tokens") or len(prompt)), int(meta.get("cached_tokens") or 0),
                           int(meta.get("completion_tokens") or spec.output_tokens), ttft, wall, t_start, t_start + wall)
        except Exception as e:  # noqa: BLE001 - every failure is data here
            results.append(TurnResult(sid, turn, len(prompt), 0, 0, 0.0, 0.0, t_start, time.perf_counter() - t_origin, error=f"{type(e).__name__}: {e}"))
            return
        finally:
            if gate is not None:
                gate.release()
        results.append(r)
        # the model's actual output ids are not returned by /generate without return_logprob; stand in with fresh
        # random ids of the same length, which is what the next turn's prefix extension looks like to the cache
        gen_ids = out_ids_fn(rng, r.completion_tokens) if out_ids_fn else make_prefix(rng, r.completion_tokens, spec.vocab)
        prompt = prompt + gen_ids + make_prefix(rng, spec.question_tokens, spec.vocab)


async def run_bench(base_url: str, spec: BenchSpec, timeout_s: float = 3600) -> dict:
    import aiohttp
    results: list[TurnResult] = []
    t_origin = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_read=timeout_s)
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    timed_out = False
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        gate = asyncio.Semaphore(spec.concurrency) if spec.concurrency else None
        tasks = [asyncio.create_task(run_session(http, base_url, sid, spec, results, t_origin, gate=gate)) for sid in range(spec.sessions)]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_s)
        except asyncio.TimeoutError:
            # keep what finished: a slow row is still a measurement of that configuration
            timed_out = True
            for t in tasks:
                t.cancel()
    summary = summarize(results, spec)
    summary["wall_s"] = round(time.perf_counter() - t_origin, 1)
    summary["timed_out"] = timed_out
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--prefix", type=int, required=True)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--question", type=int, default=2048)
    ap.add_argument("--output", type=int, default=1000)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--ramp", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--backend", default="sglang", choices=["sglang", "vllm"])
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    spec = BenchSpec(a.sessions, a.prefix, a.turns, a.question, a.output, a.dp, a.vocab, ramp_s=a.ramp, backend=a.backend, model=a.model)
    print(json.dumps(asyncio.run(run_bench(a.url, spec, a.timeout)), indent=1))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------------
# Quality probe: deterministic long prompts, greedy decode, outputs stored so arms that change numerics (state dtype,
# KV dtype, snapshot precision) can be compared against the control on the same prompts.
# ---------------------------------------------------------------------------------------------------------------------
_FACTS = [
    "The reservoir at Kellan Ridge held {a} million cubic metres in March and {b} million by July.",
    "Route {c} carries about {d} thousand vehicles a day; the bypass carries {e} thousand.",
    "The foundry poured {f} tonnes of grey iron in the second quarter and {g} tonnes in the third.",
    "Ward {h} recorded {i} admissions in winter against {j} in summer.",
    "The orchard's {k} hectares yielded {l} tonnes of fruit, of which {m} tonnes were exported.",
    "The library lent {n} thousand volumes in the year; {o} thousand were returned late.",
]


def probe_prompts(n_prompts: int = 6, target_chars: int = 60000, seed: int = 11) -> list[dict]:
    """Long natural-language documents with embedded numbers and a question whose answer depends on them."""
    rng = random.Random(seed)
    prompts = []
    for i in range(n_prompts):
        lines, answers = [], {}
        while sum(len(x) for x in lines) < target_chars:
            tpl = rng.choice(_FACTS)
            vals = {k: rng.randint(10, 990) for k in "abcdefghijklmno"}
            lines.append(tpl.format(**vals))
            answers[len(lines)] = vals
        pick = rng.randrange(max(1, len(lines) // 2), len(lines))
        doc = "\n".join(f"{j+1}. {t}" for j, t in enumerate(lines))
        q = f"\n\nQuestion: Repeat sentence {pick} exactly, then state every number in it as a comma-separated list."
        prompts.append({"id": i, "text": doc + q, "expect": lines[pick - 1]})
    return prompts


async def run_quality_probe(base_url: str, backend: str = "sglang", model: str = "m", max_new: int = 160, n_prompts: int = 6, target_chars: int = 60000) -> dict:
    """Chat-templated greedy answers (raw /generate without the template degenerates on instruct models) plus a
    teacher-forced logprob pass on the same long documents, so two servers can be compared numerically."""
    import aiohttp
    out = {"prompts": [], "exact": 0, "n": 0, "logprob": []}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800), connector=aiohttp.TCPConnector(force_close=True)) as http:
        for pr in probe_prompts(n_prompts, target_chars):
            body = {"model": model, "messages": [{"role": "user", "content": pr["text"]}], "max_tokens": max_new}
            if backend != "vllm_diffusion":
                body["temperature"] = 0.0   # diffusion samplers reject temperature
            async with http.post(f"{base_url}/v1/chat/completions", json=body) as r:
                j = await r.json()
            try:
                text = j["choices"][0]["message"].get("content") or j["choices"][0]["message"].get("reasoning_content") or ""
            except (KeyError, IndexError, TypeError):
                text = json.dumps(j)[:300]
            if backend not in ("vllm", "vllm_diffusion"):
                # teacher-forced: logprobs of the expected sentence appended to the document
                forced = pr["text"] + "\n\nAnswer: " + pr["expect"]
                lp_body = {"text": forced, "return_logprob": True, "logprob_start_len": max(0, len(pr["text"]) // 6),
                           "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}
                try:
                    async with http.post(f"{base_url}/generate", json=lp_body) as r:
                        lj = await r.json()
                    toks = lj.get("meta_info", {}).get("input_token_logprobs") or []
                    out["logprob"].append({"id": pr["id"], "tail": [(round(t[0], 4), t[1]) for t in toks[-48:] if t and t[0] is not None]})
                except Exception as e:  # noqa: BLE001
                    out["logprob"].append({"id": pr["id"], "error": str(e)[:120]})
            hit = pr["expect"].split(" held ")[0].split(" carries ")[0][:40] in text
            digits = [d for d in pr["expect"].replace(",", " ").split() if d.isdigit()]
            got = sum(1 for d in digits if d in text)
            out["prompts"].append({"id": pr["id"], "output": text[:600], "expect": pr["expect"], "sentence_found": hit, "numbers_found": f"{got}/{len(digits)}"})
            out["exact"] += int(hit and got == len(digits))
            out["n"] += 1
    return out


def compare_outputs(a: dict, b: dict) -> dict:
    """Agreement between two probe results on the same prompts: shared output prefix, sentence/number recovery, and
    on the teacher-forced tail the mean absolute logprob difference and the fraction of identical token ids."""
    rows = []
    for x, y in zip(a["prompts"], b["prompts"]):
        n = 0
        for c1, c2 in zip(x["output"], y["output"]):
            if c1 != c2:
                break
            n += 1
        rows.append({"id": x["id"], "shared_prefix_chars": n, "a_ok": x["sentence_found"], "b_ok": y["sentence_found"]})
    lp = []
    for x, y in zip(a.get("logprob", []), b.get("logprob", [])):
        ta, tb = x.get("tail") or [], y.get("tail") or []
        pairs = [(p, q) for p, q in zip(ta, tb) if p[1] == q[1]]
        if ta and tb:
            lp.append({"id": x["id"], "tokens": len(pairs), "same_token_ids": round(len(pairs) / max(len(ta), 1), 3),
                       "mean_abs_dlogprob": round(sum(abs(p[0] - q[0]) for p, q in pairs) / max(len(pairs), 1), 4)})
    return {"rows": rows, "a_exact": a["exact"], "b_exact": b["exact"], "n": a["n"], "logprob": lp}
