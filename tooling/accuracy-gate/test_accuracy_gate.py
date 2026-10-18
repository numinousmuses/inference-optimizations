"""CPU tests for accuracy_gate.py against a mock SGLang /generate (aiohttp server on a background thread).

The mock scores tokens with a deterministic toy model (log-softmax of seeded logits per position) and returns SGLang
v0.5.20's logprob layout: entries for tokens logprob_start_len.. with the first entry empty. An arm is the same model
plus seeded logit noise of a chosen size; GSM8K answers come from a table keyed by the request's input ids.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import math
import os
import random
import socket
import sys
import threading

import pytest

pytest.importorskip("aiohttp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accuracy_gate as ag  # noqa: E402

V = 50


@functools.lru_cache(maxsize=None)
def _word(w):
    return 4 + int(hashlib.md5(w.encode()).hexdigest(), 16) % (V - 4)


def render(msgs, gen):
    """Toy tokenizer: role marker + one token per word (hash mod V), generation marker if asked."""
    ids = [1]
    for m in msgs:
        ids.append(2 if m["role"] == "user" else 3)
        ids += [_word(w) for w in m["content"].split()]
    if gen:
        ids.append(3)
    return ids


@functools.lru_cache(maxsize=None)
def logprobs(prev: int, pos: int, noise: float, arm_seed: int):
    rng = random.Random(prev * 131 + pos % 7)
    logits = [rng.gauss(0, 2.0) for _ in range(V)]
    if noise:
        nr = random.Random(arm_seed * 1_000_003 + prev * 131 + pos)
        logits = [x + nr.gauss(0, noise) for x in logits]
    m = max(logits)
    z = math.log(sum(math.exp(x - m) for x in logits)) + m
    return [x - z for x in logits]


@functools.lru_cache(maxsize=None)
def _top(prev, pos, noise, arm_seed, k):
    lp = logprobs(prev, pos, noise, arm_seed)
    return tuple(sorted(range(V), key=lambda i: -lp[i])[:k])


class Mock:
    def __init__(self, noise=0.0, arm_seed=1, answers=None, shift=False):
        self.noise, self.arm_seed, self.answers, self.shift = noise, arm_seed, answers or {}, shift
        self.calls = 0

    async def generate(self, request):
        from aiohttp import web
        body = await request.json()
        self.calls += 1
        ids = body["input_ids"]
        if body.get("return_logprob"):
            s, k = body["logprob_start_len"], body["top_logprobs_num"]
            tok, top = [[None, ids[s], None]], [None]
            for j in range(s + 1, len(ids)):
                lp = logprobs(ids[j - 1], j, self.noise, self.arm_seed)
                t = ids[j] if not self.shift else ids[j - 1]
                tok.append([lp[ids[j]], t, None])
                best = _top(ids[j - 1], j, self.noise, self.arm_seed, k)
                top.append([[lp[i], i, None] for i in best])
            return web.json_response({"text": "x", "meta_info": {"input_token_logprobs": tok,
                                                                 "input_top_logprobs": top}})
        return web.json_response({"text": self.answers.get(ag.ids_hash(ids), "no idea"), "meta_info": {}})

    async def info(self, request):
        from aiohttp import web
        return web.json_response({"model_path": "/cache/toy", "version": "mock"})


@pytest.fixture
def serve():
    """serve(mock) -> base url; servers stop at teardown."""
    from aiohttp import web
    stops = []

    def start(mock):
        loop = asyncio.new_event_loop()
        app = web.Application(client_max_size=1 << 26)
        app.router.add_post("/generate", mock.generate)
        app.router.add_get("/get_model_info", mock.info)
        runner = web.AppRunner(app)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
            ready.set()
            loop.run_forever()

        threading.Thread(target=run, daemon=True).start()
        ready.wait(10)
        stops.append((loop, runner))
        return f"http://127.0.0.1:{port}"

    yield start
    for loop, runner in stops:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Fake GSM8K cache (sha pin patched) and a session file with enough assistant turns for 200 prompts."""
    rng = random.Random(0)
    rows = []
    for i in range(420):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        rows.append({"question": f"What is {a} plus {b} q{i} ?", "answer": f"{a} + {b} = {a + b}\n#### {a + b:,}"})
    raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "gsm8k_test.jsonl").write_bytes(raw)
    monkeypatch.setattr(ag, "GSM8K_SHA256", hashlib.sha256(raw).hexdigest())
    words = "alpha beta gamma delta eps zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma".split()
    sessions = []
    for s in range(12):
        msgs = []
        for t in range(40):
            role = "user" if t % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": " ".join(rng.choice(words) for _ in range(90 + 7 * t))})
        sessions.append({"id": f"s{s}", "messages": msgs})
    sess = tmp_path / "real_sessions.json"
    sess.write_text(json.dumps(sessions))
    return {"cache": str(cache), "sessions": str(sess), "rows": rows}


def answers_for(rows, n, wrong=()):
    """input-ids hash -> reply text, for the first n GSM8K items; items in `wrong` answer off by one."""
    out = {}
    for i, r in enumerate(rows[:n]):
        ids = render([{"role": "user", "content": r["question"] + ag.GSM8K_INSTR}], True)
        gold = int(ag.gold_answer(r["answer"]))
        out[ag.ids_hash(ids)] = f"Adding gives the result.\n#### {gold + (1 if i in wrong else 0)}"
    return out


def rec(url, data, **kw):
    kw = {"sessions": data["sessions"], "max_prompt_tokens": 600, "score_tokens": 16, "top_k": 8, "concurrency": 8,
          **kw}
    return ag.record(url, render, cache_dir=data["cache"], **kw)


# ------------------------------------------------------------------------------------------------ units
def test_token_kl_and_mcnemar():
    p = [[0, math.log(0.7)], [1, math.log(0.2)]]
    assert ag.token_kl(p, p) == pytest.approx(0.0, abs=1e-12)
    q = [[0, math.log(0.5)], [1, math.log(0.4)]]
    want = 0.7 * math.log(0.7 / 0.5) + 0.2 * math.log(0.2 / 0.4) + 0.1 * math.log(0.1 / 0.1)
    assert ag.token_kl(p, q) == pytest.approx(want, rel=1e-9)
    # a P-top token missing from Q's top-k is charged at Q's k-th probability
    q2 = [[0, math.log(0.7)], [5, math.log(0.25)]]
    assert ag.token_kl(p, q2) == pytest.approx(0.2 * math.log(0.2 / 0.25) + 0.1 * math.log(0.1 / 0.05), rel=1e-9)
    assert ag.mcnemar_one_sided(0, 0) == 1.0
    assert ag.mcnemar_one_sided(5, 0) == pytest.approx(1 / 32)
    assert ag.mcnemar_one_sided(3, 3) == pytest.approx(sum(math.comb(6, k) for k in range(3, 7)) / 64)


def test_answer_extraction():
    assert ag.extract_answer("so 3 + 4 = 7\n#### 7") == "7"
    assert ag.extract_answer("#### $1,234.00") == "1234"
    assert ag.extract_answer("the answer is 12, then 15.5") == "15.5"
    assert ag.extract_answer("#### -3") == "-3"
    assert ag.extract_answer("nothing") is None
    assert ag.gold_answer("blah\n#### 1,000") == "1000"
    assert ag.extract_answer("#### " + "9" * 400) is None  # floats to inf; was OverflowError (q38kv4 bf16 arm)


def test_prompts_cut_and_sessions(data):
    ids = list(range(1000))
    c = ag.cut(ids, 300, 50)
    assert len(c) == 300 and c[:50] == ids[:50] and c[-250:] == ids[-250:]
    ps = ag.session_prompts(data["sessions"], 200)
    assert len(ps) == 200 and len({p[0] for p in ps}) == 200
    assert all(m[-1]["role"] == "assistant" for _, m in ps)
    assert ps == ag.session_prompts(data["sessions"], 200)  # deterministic
    src, tf = ag.tf_prompts(render, 200, 600, 64, data["cache"], data["sessions"])
    assert src.startswith("sessions:") and all(len(p["ids"]) - p["start"] == 65 for p in tf)
    src2, tf2 = ag.tf_prompts(render, 50, 600, 64, data["cache"], None)
    assert src2 == "gsm8k_reference_solutions" and len(tf2) == 50


def test_gsm8k_cache_integrity(data, monkeypatch):
    assert len(ag.gsm8k_items(data["cache"], 200)) == 200
    monkeypatch.setattr(ag, "GSM8K_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        ag.gsm8k_items(data["cache"], 200)


# ------------------------------------------------------------------------------------------------ end to end
def test_identical_servers_pass(serve, data):
    ans = answers_for(data["rows"], 200)
    a = rec(serve(Mock(answers=ans)), data, label="control")
    b = rec(serve(Mock(answers=ans)), data, label="arm")
    assert a["meta"]["n_prompts"] == 200 and a["meta"]["gsm8k_n"] == 200
    assert a["meta"]["server"]["get_model_info"]["model_path"] == "/cache/toy"
    r = ag.compare(a, b)
    m = r["metrics"]
    assert m["tf"]["positions"] == 200 * 16 and m["tf"]["kl_mean"] == pytest.approx(0, abs=1e-9)
    assert m["tf"]["top1_agree"] == 1.0 and m["tf"]["dlp_max"] == 0.0
    assert m["em"]["acc_control"] == m["em"]["acc_arm"] == 1.0
    assert r["verdict"]["pass"], r["verdict"]


def test_small_noise_passes_large_noise_fails(serve, data):
    ans = answers_for(data["rows"], 200)
    ctl = rec(serve(Mock(answers=ans)), data)
    small = ag.compare(ctl, rec(serve(Mock(noise=0.02, answers=ans)), data))
    big = ag.compare(ctl, rec(serve(Mock(noise=1.0, answers=ans)), data))
    assert small["verdict"]["pass"], small
    assert 0 < small["metrics"]["tf"]["kl_mean"] < big["metrics"]["tf"]["kl_mean"]
    assert not big["verdict"]["pass"]
    assert not big["verdict"]["checks"]["kl_mean"] and not big["verdict"]["checks"]["top1_agree"]


def test_em_paired_regression(serve, data):
    rows = data["rows"]
    ctl = rec(serve(Mock(answers=answers_for(rows, 200))), data, n_prompts=20)
    few = ag.compare(ctl, rec(serve(Mock(answers=answers_for(rows, 200, wrong={3, 9}))), data, n_prompts=20))
    many = ag.compare(ctl, rec(serve(Mock(answers=answers_for(rows, 200, wrong=set(range(0, 200, 20))))), data,
                               n_prompts=20))
    assert few["metrics"]["em"]["control_only_correct"] == 2 and few["verdict"]["checks"]["em_paired"]
    em = many["metrics"]["em"]
    assert em["control_only_correct"] == 10 and em["drop_pts"] == pytest.approx(5.0)
    assert em["mcnemar_p_arm_worse"] == pytest.approx(1 / 1024)
    assert not many["verdict"]["checks"]["em_paired"] and not many["verdict"]["pass"]


def test_noise_widens_thresholds(serve, data):
    ans = answers_for(data["rows"], 200)
    ctl = rec(serve(Mock(answers=ans)), data, n_prompts=40)
    ctl2 = rec(serve(Mock(noise=0.5, arm_seed=7, answers=ans)), data, n_prompts=40)
    arm = rec(serve(Mock(noise=0.5, arm_seed=9, answers=ans)), data, n_prompts=40)
    noise = ag.calibrate(ctl, ctl2)
    strict, lenient = ag.compare(ctl, arm), ag.compare(ctl, arm, noise)
    assert not strict["verdict"]["pass"] and lenient["verdict"]["pass"], (strict["verdict"], lenient["verdict"])
    assert lenient["verdict"]["thresholds"]["kl_mean_max"] == pytest.approx(3 * noise["tf"]["kl_mean"])
    assert lenient["verdict"]["within_noise"]


def test_misaligned_server_is_refused(serve, data):
    with pytest.raises(RuntimeError, match="different tokens"):
        rec(serve(Mock(shift=True)), data, n_prompts=5, gsm8k_n=0)


def test_changed_inputs_fail(serve, data):
    url = serve(Mock(answers=answers_for(data["rows"], 200)))
    a = rec(url, data, n_prompts=20, gsm8k_n=0)
    b = rec(url, data, n_prompts=20, gsm8k_n=0, max_prompt_tokens=500)
    r = ag.compare(a, b)
    assert r["metrics"]["tf"]["mismatched_prompts"] and not r["verdict"]["checks"]["inputs_identical"]


def test_gate_arm_flow_and_cli(serve, data, tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "make_renderer", lambda model_dir: render)
    ans = answers_for(data["rows"], 200)
    out = str(tmp_path / "gate")
    kw = dict(cache_dir=data["cache"], sessions=data["sessions"], n_prompts=30, max_prompt_tokens=600,
              score_tokens=16, top_k=8, concurrency=8)
    r0 = ag.gate_arm(serve(Mock(answers=ans)), "/m", "control", out, **kw)
    assert r0["gsm8k_acc"] == 1.0 and "verdict" not in r0
    r1 = ag.gate_arm(serve(Mock(noise=0.01, arm_seed=3, answers=ans)), "/m", "control2", out, **kw)
    assert "noise" in r1 and os.path.exists(os.path.join(out, "accuracy_noise.json"))
    r2 = ag.gate_arm(serve(Mock(noise=1.0, answers=ans)), "/m", "fp8_attn", out, **kw)
    assert r2["verdict"]["noise_given"] and not r2["verdict"]["pass"]
    assert os.path.exists(os.path.join(out, "accuracy_compare_fp8_attn.json"))
    # CLI compare exits 1 on a failing arm, 0 on the noise control
    base = os.path.join(out, "accuracy_")
    assert ag.main(["compare", "--control", base + "control.json", "--arm", base + "fp8_attn.json"]) == 1
    assert ag.main(["compare", "--control", base + "control.json", "--arm", base + "control2.json"]) == 0
    assert ag.main(["calibrate", "--a", base + "control.json", "--b", base + "control2.json",
                    "--out", str(tmp_path / "n.json")]) == 0


def test_noise_accepts_a_full_compare_output():
    """accuracy_gate compare --noise takes the saved output of a control-vs-control compare (metrics nested)."""
    metrics = {"tf": {"kl_mean": 0.013, "kl_p99": 0.18, "top1_agree": 0.979, "dlp_p99": 0.99},
               "em": {"drop_pts": -1.5}}
    bare = ag.effective_thresholds(metrics)
    nested = ag.effective_thresholds({"metrics": metrics, "verdict": {"pass": False}})
    assert bare == nested
    assert nested["kl_mean_max"] == ag.NOISE_MULT * 0.013
