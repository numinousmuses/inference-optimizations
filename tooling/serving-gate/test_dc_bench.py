"""CPU tests for the session bench: summary math, routing, and an end-to-end run against a fake SGLang."""
import asyncio
import json
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dc_bench import BenchSpec, TurnResult, make_prefix, run_bench, summarize  # noqa: E402


def _r(sid, turn, prompt, cached, out, ttft, wall, t0, err=None):
    return TurnResult(sid, turn, prompt, cached, out, ttft, wall, t0, t0 + wall, err)


def test_summary_warm_only_and_hit_rate():
    spec = BenchSpec(sessions=2, prefix_tokens=1000, turns=2, question_tokens=100, output_tokens=100)
    rows = [
        _r(0, 1, 1000, 0, 100, 1.0, 3.0, 0.0),        # cold: 1000 new tokens over 1 s TTFT
        _r(1, 1, 1000, 0, 100, 1.0, 3.0, 0.0),
        _r(0, 2, 1200, 1100, 100, 0.2, 2.2, 3.0),      # warm: 99 tokens in 2.0 s = 49.5 tok/s per user
        _r(1, 2, 1200, 1100, 100, 0.2, 2.2, 3.0),
    ]
    s = summarize(rows, spec)
    assert s["sessions_completed"] == 2 and s["requests_failed"] == 0
    assert s["warm"]["hit_rate"] == pytest.approx(2200 / 2400, abs=1e-4)
    assert s["warm"]["per_user_tok_s"]["p50"] == pytest.approx(49.5, abs=0.1)
    assert s["warm"]["agg_out_tok_s"] == pytest.approx(200 / 2.2, abs=0.1)   # 200 tokens in the 2.2 s warm window
    assert s["cold"]["prefill_tok_s"] == pytest.approx(2000 / 1.0, abs=1)
    assert s["cold"]["hit_rate"] == 0.0


def test_summary_counts_errors_and_incomplete_sessions():
    spec = BenchSpec(sessions=2, prefix_tokens=10, turns=2)
    rows = [_r(0, 1, 10, 0, 5, 0.1, 0.5, 0.0), _r(0, 2, 20, 10, 5, 0.1, 0.5, 0.5), _r(1, 1, 10, 0, 0, 0, 0, 0.0, err="RuntimeError: http 500")]
    s = summarize(rows, spec)
    assert s["sessions_completed"] == 1 and s["requests_failed"] == 1
    assert s["error_sample"] == ["RuntimeError: http 500"]


def test_prefix_is_deterministic_and_in_range():
    import random
    a, b = make_prefix(random.Random(7), 50, 5000), make_prefix(random.Random(7), 50, 5000)
    assert a == b and all(1000 <= t < 5000 for t in a)


class FakeSGLang:
    """Minimal /generate that streams two chunks, records routed_dp_rank and remembers prefixes per rank
    to report cached_tokens like a radix cache would (longest common prefix with anything seen on that rank)."""

    def __init__(self):
        self.seen: dict[int, list[list[int]]] = {}
        self.bodies = []

    async def handle(self, request):
        from aiohttp import web
        body = await request.json()
        self.bodies.append(body)
        ids = body["input_ids"]
        rank = body.get("routed_dp_rank", 0)
        cached = 0
        for prev in self.seen.get(rank, []):
            n = 0
            for x, y in zip(prev, ids):
                if x != y:
                    break
                n += 1
            cached = max(cached, n)
        self.seen.setdefault(rank, []).append(ids)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await asyncio.sleep(0.01)
        await resp.write(b'data: {"text": "a", "meta_info": {"prompt_tokens": %d, "cached_tokens": %d, "completion_tokens": 1}}\n\n' % (len(ids), cached))
        await asyncio.sleep(0.02)
        n = body["sampling_params"]["max_new_tokens"]
        await resp.write(b'data: {"text": "ab", "meta_info": {"prompt_tokens": %d, "cached_tokens": %d, "completion_tokens": %d}}\n\n' % (len(ids), cached, n))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp


@pytest.mark.asyncio
async def test_end_to_end_against_fake_server_routes_sessions_by_rank():
    from aiohttp import web
    fake = FakeSGLang()
    app = web.Application()
    app.router.add_post("/generate", fake.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        spec = BenchSpec(sessions=4, prefix_tokens=64, turns=3, question_tokens=8, output_tokens=4, dp_size=2, vocab=3000)
        s = await run_bench(f"http://127.0.0.1:{port}", spec, timeout_s=30)
    finally:
        await runner.cleanup()
    assert s["requests_failed"] == 0 and s["sessions_completed"] == 4
    # sessions 0,2 -> rank 0; 1,3 -> rank 1; every turn of a session carries the same rank
    ranks = {b["routed_dp_rank"] for b in fake.bodies}
    assert ranks == {0, 1}
    # warm turns hit their own earlier prefix: prompt grows by output + question each turn, all of the old prompt is cached
    assert s["warm"]["hit_rate"] > 0.8
    assert s["cold"]["hit_rate"] == 0.0
    assert json.dumps(s)  # serialisable


@pytest.mark.asyncio
async def test_no_dp_omits_routing_field():
    from aiohttp import web
    fake = FakeSGLang()
    app = web.Application()
    app.router.add_post("/generate", fake.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        await run_bench(f"http://127.0.0.1:{port}", BenchSpec(sessions=1, prefix_tokens=16, turns=1, output_tokens=2, vocab=2000), timeout_s=30)
    finally:
        await runner.cleanup()
    assert all("routed_dp_rank" not in b for b in fake.bodies)


class FakeVLLM:
    """OpenAI completions with a token-id prompt; usage arrives in a final choices-less chunk, cached_tokens in
    prompt_tokens_details, computed like a prefix cache over everything seen (single engine, no ranks)."""

    def __init__(self):
        self.seen: list[list[int]] = []
        self.bodies = []

    async def handle(self, request):
        from aiohttp import web
        body = await request.json()
        self.bodies.append(body)
        ids = body["prompt"]
        cached = 0
        for prev in self.seen:
            n = 0
            for x, y in zip(prev, ids):
                if x != y:
                    break
                n += 1
            cached = max(cached, n)
        self.seen.append(ids)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await asyncio.sleep(0.01)
        await resp.write(b'data: {"choices": [{"text": "a", "index": 0}]}\n\n')
        # a wide gap before the rest: TTFT must come from the first chunk (~10 ms), and the gap keeps that distinguishable
        # from the usage chunk (~310 ms) even when the event loop is delayed by a loaded machine (a 20 ms gap was flaky)
        await asyncio.sleep(0.3)
        await resp.write(b'data: {"choices": [{"text": "b", "index": 0}]}\n\n')
        n = body["max_tokens"]
        await resp.write(('data: {"choices": [], "usage": {"prompt_tokens": %d, "completion_tokens": %d, "prompt_tokens_details": {"cached_tokens": %d}}}\n\n' % (len(ids), n, cached)).encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp


@pytest.mark.asyncio
async def test_vllm_backend_reads_usage_and_streams_token_ids():
    from aiohttp import web
    fake = FakeVLLM()
    app = web.Application()
    app.router.add_post("/v1/completions", fake.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        spec = BenchSpec(sessions=3, prefix_tokens=64, turns=2, question_tokens=8, output_tokens=5, vocab=3000, backend="vllm", model="m")
        s = await run_bench(f"http://127.0.0.1:{port}", spec, timeout_s=30)
    finally:
        await runner.cleanup()
    assert s["requests_failed"] == 0 and s["sessions_completed"] == 3
    b = fake.bodies[0]
    assert b["model"] == "m" and isinstance(b["prompt"][0], int) and b["stream_options"] == {"include_usage": True} and "routed_dp_rank" not in b
    assert s["warm"]["hit_rate"] > 0.8 and s["cold"]["hit_rate"] == 0.0
    # TTFT came from the first choices chunk (~10 ms), not the usage chunk (~310 ms)
    assert s["warm"]["ttft"]["p50_s"] < 0.2


@pytest.mark.asyncio
async def test_timeout_keeps_completed_turns():
    from aiohttp import web

    async def slow(request):
        body = await request.json()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"text": "a", "meta_info": {"prompt_tokens": %d, "cached_tokens": 0, "completion_tokens": 1}}\n\n' % len(body["input_ids"]))
        await asyncio.sleep(5)          # never finishes within the bench timeout
        await resp.write(b"data: [DONE]\n\n")
        return resp

    fast_first = {"n": 0}

    async def handler(request):
        fast_first["n"] += 1
        if fast_first["n"] <= 2:        # the first two requests complete instantly, the rest hang
            body = await request.json()
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(b'data: {"text": "a", "meta_info": {"prompt_tokens": %d, "cached_tokens": 0, "completion_tokens": 2}}\n\n' % len(body["input_ids"]))
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return await slow(request)

    app = web.Application()
    app.router.add_post("/generate", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        s = await run_bench(f"http://127.0.0.1:{port}", BenchSpec(sessions=2, prefix_tokens=8, turns=2, output_tokens=2, vocab=2000), timeout_s=0.5)
    finally:
        await runner.cleanup()
    assert s["timed_out"] is True
    assert s["requests_ok"] == 2 and s["sessions_completed"] == 0


def test_probe_prompts_are_deterministic_and_long():
    from dc_bench import probe_prompts, compare_outputs
    a, b = probe_prompts(3, 20000), probe_prompts(3, 20000)
    assert [x["text"] for x in a] == [x["text"] for x in b]
    assert all(len(x["text"]) >= 20000 for x in a)
    assert all(x["expect"] in x["text"] for x in a)
    ra = {"prompts": [{"id": 0, "output": "hello world", "sentence_found": True}], "exact": 1, "n": 1}
    rb = {"prompts": [{"id": 0, "output": "hello there", "sentence_found": False}], "exact": 0, "n": 1}
    c = compare_outputs(ra, rb)
    assert c["rows"][0]["shared_prefix_chars"] == 6 and c["a_exact"] == 1 and c["b_exact"] == 0
