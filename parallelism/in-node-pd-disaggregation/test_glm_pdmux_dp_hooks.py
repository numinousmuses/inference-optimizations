"""CPU tests for the DP pdmux scheduler side (amdglm): glm_pdmux_dp_hooks.py + numinous_pdmux_dp.py.

1. Anchors resolve once on SGLang v0.5.20, apply, compile, idempotent; the module installs.
2. Pure rules: split_forward_count (global max), choose_stream_idx (global state), global_decode_state.
3. THE lockstep property: the real event_loop_pdmux_dp runs on N gloo ranks (fake scheduler, fake streams, real CPU
   all_gather / allreduce for every scheduler sync) under random rank-skewed arrivals, chunked prompts and EAGLE.
   Every rank must issue the same sequence of collective-bearing operations (plan sync, decode sync, decode forward
   or IDLE, split forward over the same layer range, done-flag allreduce, the hand-off draft-extend), and every
   request must finish.
4. finish_split: the EAGLE hand-off stashes the target result and runs the spec path in EXTEND (IDLE on idle-split
   ranks); the unfinished chunked request stays out of the decode batch; merge vs adopt.
5. numinous_preflight: the lever's incompatible flags are refused, inert when off.
"""
from __future__ import annotations

import ast
import contextlib
import os
import pathlib
import random
import shutil
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm_pdmux_dp_hooks as H  # noqa: E402
import numinous_pdmux_dp as M  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python")) / "sglang"
needs_src = pytest.mark.skipif(not (SRC / H.MIX).exists(), reason="SGLang v0.5.20 source absent")


# ---------------------------------------------------------------------------------------------------------- 1
def _copy(root):
    pkg = root / "sglang"
    for rel in (H.MIX, H.VAL, H.TW, H.MR, H.ER, H.CTX):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / rel, pkg / rel)
    return pkg


@needs_src
def test_apply_once_compile_idempotent(tmp_path):
    pkg = _copy(tmp_path)
    chk = H.apply(str(pkg), check=True)
    assert all(x.endswith("anchor ok") or x.endswith("install ok") for x in chk), chk
    out = H.apply(str(pkg))
    assert sum("applied to" in x for x in out) == 11 and "module: installed srt/multiplex/numinous_pdmux_dp.py" in out
    again = H.apply(str(pkg))
    assert all("already" in x for x in again), again
    for rel in (H.MIX, H.VAL, H.TW, H.MR, H.ER, H.CTX, H.MODULE_DST):
        compile((pkg / rel).read_text(), rel, "exec")


@needs_src
def test_stash_return_is_the_first_statement(tmp_path):
    pkg = _copy(tmp_path)
    H.apply(str(pkg))
    tree = ast.parse((pkg / H.TW).read_text())
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward_batch_generation"
           and "_numinous_split_result" in ast.unparse(n)]
    assert len(fns) == 1
    first = fns[0].body[0]
    assert isinstance(first, ast.If) and "_numinous_split_result" in ast.unparse(first.test)
    assert isinstance(first.body[-1], ast.Return)


@needs_src
def test_idle_split_never_takes_the_decode_graph(tmp_path):
    pkg = _copy(tmp_path)
    H.apply(str(pkg))
    src = (pkg / H.MR).read_text()
    i_graph, i_disp = src.index('not getattr(forward_batch, "numinous_idle_split", False)'), src.index(
        'forward_batch.forward_mode.is_split_prefill() or getattr(')
    assert i_graph < i_disp


# ---------------------------------------------------------------------------------------------------------- 2
def test_split_forward_count_is_global():
    assert M.split_forward_count([2048, 0, 512], 65536, 78, 0) == 32
    assert M.split_forward_count([2048, 0, 512], 65536, 78, 64) == 14       # clipped at the last layer
    assert M.split_forward_count([0, 0], 65536, 78, 0) == 78                # 0 rows: every layer at once
    assert M.split_forward_count([100000], 65536, 78, 10) == 1              # at least one layer


def test_choose_stream_idx_global():
    assert M.choose_stream_idx(3, True, True, 4, [], 36) == 1
    assert M.choose_stream_idx(3, False, True, 4, [], 36) == 2
    assert M.choose_stream_idx(3, True, False, 0, [], 36) == 0
    assert M.choose_stream_idx(3, False, False, 0, [], 36) == 0
    assert M.choose_stream_idx(5, True, True, 40, [[96, 36, 0], [80, 52, 8], [64, 68, 32]], 36) == 3
    assert M.choose_stream_idx(5, True, True, 2, [[96, 36, 4], [80, 52, 8], [64, 68, 32]], 36) == 1   # stock: undefined


def test_global_decode_state():
    assert M.global_decode_state(None) == (False, 0)
    b = types.SimpleNamespace(global_num_tokens=[0, 3, 1], forward_mode=types.SimpleNamespace(is_idle=lambda: True))
    assert M.global_decode_state(b) == (True, 3)


# ---------------------------------------------------------------------------------------------------------- 3
class _Mode:
    def __init__(self, name):
        self.name = name

    def is_idle(self):
        return self.name == "IDLE"

    def is_extend(self):
        return self.name in ("EXTEND", "SPLIT_PREFILL")

    def is_decode(self):
        return self.name == "DECODE"

    def is_split_prefill(self):
        return self.name == "SPLIT_PREFILL"


class _FM:
    IDLE, EXTEND, DECODE, SPLIT_PREFILL = _Mode("IDLE"), _Mode("EXTEND"), _Mode("DECODE"), _Mode("SPLIT_PREFILL")


class _Spec:
    def is_none(self):
        return False


class _Req:
    def __init__(self, rid, prompt, out_len, chunk):
        self.rid, self.prompt, self.out_len, self.chunk = rid, prompt, out_len, chunk
        self.done_prompt, self.generated = 0, 0
        self.extend_range, self.prefix_indices = types.SimpleNamespace(end=0), []     # stash_chunked's view


class _Batch:
    def __init__(self, reqs, mode, tokens=0):
        self.reqs, self.forward_mode, self.extend_num_tokens = list(reqs), mode, tokens
        self.spec_algorithm, self.global_num_tokens, self.chunked_req = _Spec(), None, None
        self.spec_info = None

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def merge_batch(self, other):
        self.reqs += other.reqs

    def filter_batch(self, chunked_req_to_exclude=()):
        self.reqs = [r for r in self.reqs if r not in chunked_req_to_exclude]


class _Stop(Exception):
    pass


class _Stream:
    def synchronize(self):
        pass

    def record_event(self):
        return types.SimpleNamespace(query=lambda: True)

    def wait_event(self, e):
        pass


class _ThreadColl:
    """In-process collectives for N rank threads: every call carries a label, and all ranks must call the same label
    at the same position (a stricter check than shapes: a mismatched op type fails at once instead of hanging)."""

    def __init__(self, world):
        import threading
        self.world, self.slots, self.errors = world, [None] * world, []
        self.b1, self.b2 = threading.Barrier(world, timeout=60), threading.Barrier(world, timeout=60)

    def call(self, rank, label, value, op):
        self.slots[rank] = (label, value)
        self.b1.wait()
        labels = {x[0] for x in self.slots}
        vals = [x[1] for x in self.slots]
        self.b2.wait()
        if len(labels) != 1:
            self.errors.append(f"rank {rank}: collective label mismatch {sorted(labels)}")
            raise AssertionError(self.errors[-1])
        return vals if op == "gather" else sum(vals)


class _GlooColl:
    def __init__(self, world):
        self.world = world

    def call(self, rank, label, value, op):
        import torch
        import torch.distributed as dist
        t = torch.tensor([value], dtype=torch.int64)
        if op == "gather":
            out = [torch.zeros(1, dtype=torch.int64) for _ in range(self.world)]
            dist.all_gather(out, t)
            return [int(x.item()) for x in out]
        dist.all_reduce(t)
        return int(t.item())


_TLS = None


def _ctx():
    return _TLS.ctx


def _install_stubs():
    """Module stubs + module-level fakes that dispatch to the calling rank's context (thread-local)."""
    import threading

    import torch
    global _TLS
    if _TLS is not None:
        return
    _TLS = threading.local()
    stub_ps = types.ModuleType("sglang.srt.distributed.parallel_state")
    stub_ps.set_pdmux_status = lambda on: None
    stub_ctx = types.ModuleType("sglang.srt.multiplex.pdmux_context")
    stub_ctx.get_current_stream_idx = lambda: 0
    stub_ctx.set_current_stream_idx = lambda i: _ctx()["log"].append(("group", i))
    stub_fbi = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
    stub_fbi.ForwardMode = _FM
    for name, mod in (("sglang.srt.distributed.parallel_state", stub_ps),
                      ("sglang.srt.multiplex.pdmux_context", stub_ctx),
                      ("sglang.srt.model_executor.forward_batch_info", stub_fbi)):
        sys.modules[name] = mod
    torch.cuda.stream = lambda s: contextlib.nullcontext()
    torch.cuda.empty_cache = lambda: None

    def fake_dp_sync(sched, batch):
        c = _ctx()
        local = 0 if batch is None or batch.forward_mode.is_idle() else (
            batch.batch_size() if batch.forward_mode.is_decode() else batch.extend_num_tokens)
        kind = "plan" if sched._in_plan else "decode"
        g = c["coll"].call(c["rank"], f"sync:{kind}", local, "gather")
        c["log"].append(("sync", kind))
        if max(g) == 0:
            return None
        if batch is None:
            batch = _Batch([], _FM.IDLE)
        batch.global_num_tokens = g
        return batch

    stock_plan = M.plan_split

    def plan_wrap(sched, running):
        sched._in_plan = True
        try:
            return stock_plan(sched, running)
        finally:
            sched._in_plan = False

    M.dp_sync, M.plan_split = fake_dp_sync, plan_wrap
    stock_merge = _Batch.merge_batch

    def merge(self, other):
        stock_merge(self, other)
        self.forward_mode = _FM.DECODE

    _Batch.merge_batch = merge


def _run_rank(rank, world, iters, seed, coll):
    """One rank's scheduler on the REAL event_loop_pdmux_dp. Returns (log, unfinished request ids, request count)."""
    import torch
    _install_stubs()
    log = []
    _TLS.ctx = {"rank": rank, "coll": coll, "log": log}
    rng = random.Random(seed * 1000 + rank)
    arrivals = {}
    for t in range(iters - 170):                     # per-rank arrivals, skewed by rank
        if rng.random() < 0.02 + 0.012 * rank:
            arrivals.setdefault(t, []).append(_Req(f"r{rank}-{t}", rng.choice([300, 2500, 6000]),
                                                   rng.randint(3, 25), 2048))
    s = types.SimpleNamespace()
    s._in_plan, s.t = False, 0
    s.waiting, s.running_batch, s.split_prefill_batch, s.chunked_req = [], _Batch([], _FM.DECODE), None, None
    s.finished = []
    s.ps = types.SimpleNamespace(tp_size=world, tp_rank=rank, gpu_id=rank)
    s.model_config = types.SimpleNamespace(num_hidden_layers=78)
    s.pdmux_config = types.SimpleNamespace(split_forward_token_budget=65536, manual_divisions=[], decode_bs_divisor=36)
    s.stream_groups = [(_Stream(), _Stream()) for _ in range(3)]

    def ingest():
        if s.t >= iters:
            raise _Stop()
        s.waiting += arrivals.get(s.t, [])
        s.t += 1

    def get_new_batch_prefill(running):
        r = s.chunked_req if s.chunked_req is not None else (s.waiting.pop(0) if s.waiting else None)
        batch = None
        if r is not None:
            n = min(r.chunk, r.prompt - r.done_prompt)
            r.done_prompt += n
            s.chunked_req = r if r.done_prompt < r.prompt else None
            batch = _Batch([r], _FM.EXTEND, n)
            batch.chunked_req = s.chunked_req
        return types.SimpleNamespace(batch_to_run=batch, running_batch=running)

    def update_running(b):
        b.reqs = [r for r in b.reqs if r.generated < r.out_len]
        b.forward_mode = _FM.DECODE                  # prepare_for_decode
        return b

    def run_batch(b):
        stashed = getattr(b, "_numinous_split_result", None) is not None
        b._numinous_split_result = None              # the tp_worker hook consumes the stash once
        tag = ("draft_extend" if stashed else "decode_fwd", "IDLE" if b.forward_mode.is_idle() else
               ("EXTEND" if stashed else "DECODE"))
        coll.call(rank, f"fwd:{tag[0]}", 0, "gather")       # the forward's DP collectives
        log.append(tag)
        return types.SimpleNamespace()

    def process(b, res):
        if b.forward_mode.is_decode():
            for r in b.reqs:
                r.generated += 1
                if r.generated >= r.out_len:
                    s.finished.append(r.rid)

    def split_fwd(b):
        coll.call(rank, f"split:{b.split_index}:{b.split_forward_count}", 0, "gather")
        log.append(("split", b.split_index, b.split_forward_count, bool(getattr(b, "_numinous_idle_split", False))))
        return types.SimpleNamespace()

    class CPUGroup:
        def allreduce(self, t, op):
            t.fill_(coll.call(rank, "done_allreduce", int(t.item()), "sum"))
            log.append(("done_allreduce", int(t.item())))
            return types.SimpleNamespace(wait=lambda: None)

    s.ingest_requests, s.get_new_batch_prefill = ingest, get_new_batch_prefill
    s.n_abort = s.n_hicache = 0

    def abort_check():
        s.n_abort += 1

    def hicache():                                   # TP-wide consensus in the engine: a labelled collective here
        s.n_hicache += 1
        coll.call(rank, "hicache_drain", 0, "gather")

    s.process_pending_chunked_abort, s._process_hicache_events = abort_check, hicache
    s.stash_chunked_request = lambda r: None
    s.update_running_batch, s.run_batch, s.process_batch_result = update_running, run_batch, process
    s.tp_worker = types.SimpleNamespace(forward_batch_split_prefill=split_fwd,
                                        model_runner=types.SimpleNamespace(update_decode_attn_backend=lambda i: None))
    s.tp_cpu_group, s.on_idle = CPUGroup(), (lambda: None)
    try:
        M.event_loop_pdmux_dp(s)
    except _Stop:
        pass
    left = [r.rid for rs in arrivals.values() for r in rs if r.rid not in s.finished]
    assert s.n_abort == s.n_hicache == iters, (s.n_abort, s.n_hicache, iters)   # once per iteration, every rank
    return log, left, sum(len(v) for v in arrivals.values())


def _check(res, seed):
    def op(x):          # what must agree across ranks: the op, its layer range, the allreduce value (not idle/real)
        return x[:3] if x[0] == "split" else (x[0],) if x[0] in ("decode_fwd", "draft_extend") else x
    seqs = [[op(x) for x in log if x[0] != "group"] for log, _, _ in res]
    assert all(seq == seqs[0] for seq in seqs), f"seed {seed}: ranks diverge"
    groups = [[x for x in log if x[0] == "group"] for log, _, _ in res]
    assert all(g == groups[0] for g in groups), f"seed {seed}: SM partitions diverge"
    for rank, (_log, left, n) in enumerate(res):
        assert n > 0 and not left, f"seed {seed} rank {rank}: {len(left)} of {n} requests never finished"
    return (sum(1 for log, _, _ in res for x in log if x[0] == "split" and x[3]),
            sum(1 for log, _, _ in res for x in log if x == ("draft_extend", "IDLE")))


def test_every_rank_issues_the_same_collective_sequence_threads():
    """4 rank threads, labelled in-process collectives, 6 arrival seeds."""
    pytest.importorskip("torch")
    import threading
    world, iters = 4, 450
    idle_splits = idle_drafts = 0
    for seed in range(1, 7):
        coll = _ThreadColl(world)
        res, errs = [None] * world, []

        def go(r):
            try:
                res[r] = _run_rank(r, world, iters, seed, coll)
            except BaseException as e:  # noqa: BLE001
                errs.append(repr(e))

        ts = [threading.Thread(target=go, args=(r,)) for r in range(world)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=240)
        assert not errs and not coll.errors, (errs, coll.errors)
        a, b = _check(res, seed)
        idle_splits, idle_drafts = idle_splits + a, idle_drafts + b
    assert idle_splits > 0 and idle_drafts > 0, "the arrival skew never produced idle splits / idle draft-extends"


def _gloo_main(rank, world, port, iters, seed, q):
    import torch
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    q.put((rank,) + _run_rank(rank, world, iters, seed, _GlooColl(world)))
    dist.destroy_process_group()


def test_every_rank_issues_the_same_collective_sequence_gloo():
    """The same on 4 processes with REAL gloo collectives (one seed: ~3 min on a loaded box)."""
    pytest.importorskip("torch")
    import torch.multiprocessing as mp
    world, iters, seed = 4, 450, 11
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29700 + random.randint(0, 900)
    ps = [ctx.Process(target=_gloo_main, args=(r, world, port, iters, seed, q)) for r in range(world)]
    for p in ps:
        p.start()
    res = []
    for _ in range(world):
        try:
            res.append(q.get(timeout=280))
        except Exception:   # noqa: BLE001 - a crashed rank: report exit codes, never hang the suite
            break
    for p in ps:
        p.join(timeout=60)
    assert len(res) == world, f"rank exit codes {[p.exitcode for p in ps]}"
    res.sort()
    _check([r[1:] for r in res], seed)


# ---------------------------------------------------------------------------------------------------------- 4
def test_finish_split_spec_handoff_chunk_and_merge(monkeypatch):
    stub_fbi = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
    stub_fbi.ForwardMode = _FM
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.forward_batch_info", stub_fbi)
    calls = []
    sched = types.SimpleNamespace(chunked_req=None)
    sched.run_batch = lambda b: calls.append(("run", b.forward_mode.name, b._numinous_split_result)) or "eagle"
    sched.process_batch_result = lambda b, r: calls.append(("process", r))
    a, c = _Req("a", 10, 3, 2048), _Req("c", 9000, 3, 2048)
    split = _Batch([a, c], _FM.SPLIT_PREFILL, 4000)
    sched.chunked_req = c
    running = _Batch([_Req("x", 1, 3, 1)], _FM.DECODE)
    out = M.finish_split(sched, running, split, "target", None)
    assert calls[0] == ("run", "EXTEND", "target") and calls[1] == ("process", "eagle")
    assert out is running and [r.rid for r in running.reqs] == ["x", "a"]
    calls.clear()
    idle = _Batch([], _FM.IDLE)
    idle._numinous_idle_split = True
    assert M.finish_split(sched, running, idle, "empty", None) is running
    assert calls == [("run", "IDLE", "empty")]
    calls.clear()
    sched.chunked_req = None
    solo = _Batch([_Req("b", 5, 3, 2048)], _FM.SPLIT_PREFILL, 5)
    assert M.finish_split(sched, _Batch([], _FM.DECODE), solo, "t", None) is solo


# ---------------------------------------------------------------------------------------------------------- 5
def test_preflight():
    ok = types.SimpleNamespace(enable_pdmux=True, enable_dp_attention=True, disable_overlap_schedule=True,
                               enable_mixed_chunk=False, enable_prefill_delayer=False, chunked_prefill_size=16384,
                               enable_two_batch_overlap=False, pp_size=1)
    env = {H.ENV: "1", "NUMINOUS_PDMUX_CHUNKED": "1"}
    assert H.numinous_preflight(types.SimpleNamespace(env=env, args=ok)) == []
    assert H.numinous_preflight(types.SimpleNamespace(env={}, args=ok)) == []
    for k, v in (("enable_pdmux", False), ("disable_overlap_schedule", False), ("enable_mixed_chunk", True),
                 ("enable_prefill_delayer", True), ("enable_two_batch_overlap", True)):
        bad = types.SimpleNamespace(**{**vars(ok), k: v})
        assert H.numinous_preflight(types.SimpleNamespace(env=env, args=bad)), k
    assert H.numinous_preflight(types.SimpleNamespace(env={H.ENV: "1"}, args=ok))          # chunked without the env
    assert H.numinous_preflight(types.SimpleNamespace(env={**env, "NUMINOUS_DP_PREFILL_PACE": "1"}, args=ok))
    # hook_preflight passes the resolved args as a dict
    assert H.numinous_preflight(types.SimpleNamespace(env=env, args=dict(vars(ok)))) == []
    assert H.numinous_preflight(types.SimpleNamespace(env=env, args={**vars(ok), "enable_mixed_chunk": True}))


# ---------------------------------------------------------------------------------------------------------- 6
def test_loop_trace_and_idle_readout(tmp_path, monkeypatch):
    """The loop trace (NUMINOUS_SCHED_TRACE=1) writes one parseable pdmux_rank<tp>.jsonl per rank from the real loop,
    and pdmux_idle_readout turns it into per-rank and rank-time-weighted idle shares."""
    pytest.importorskip("torch")
    import threading
    sys.path.insert(0, str(HERE.parent / "hopper"))
    import pdmux_idle_readout as R
    monkeypatch.setenv("NUMINOUS_SCHED_TRACE", "1")
    monkeypatch.setenv("NUMINOUS_SCHED_TRACE_DIR", str(tmp_path))
    world, iters, seed = 3, 400, 5
    coll, res, errs = _ThreadColl(world), [None] * world, []
    orig = _run_rank

    def go(r):
        try:
            res[r] = orig(r, world, iters, seed, coll)
        except BaseException as e:  # noqa: BLE001
            errs.append(repr(e))

    ts = [threading.Thread(target=go, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=240)
    assert not errs, errs
    files = sorted(tmp_path.glob("pdmux_rank*.jsonl"))
    assert [f.name for f in files] == [f"pdmux_rank{r}.jsonl" for r in range(world)]
    assert all(R.load(str(f)) for f in files)
    recs = R.load(str(files[0]))
    assert any(r.get("dmode") == "DECODE" for r in recs) and any(r.get("split") for r in recs)
    assert any(r.get("finish_s") is not None for r in recs)
    out = R.readout([str(f) for f in files])
    assert out["idle_share_rank_time_weighted"] is not None and 0.0 <= out["idle_share_rank_time_weighted"] <= 1.0


def test_idle_readout_arithmetic(tmp_path):
    sys.path.insert(0, str(HERE.parent / "hopper"))
    import json as _json

    import pdmux_idle_readout as R
    recs = ([{"it": i, "t0": i * 0.06, "dur": 0.06, "running": 4, "dmode": "DECODE", "d_s": 0.058, "d_gpu_ms": 45.0}
             for i in range(10)] + [{"idle_span": 5, "idle_s": 1.0}]
            + [{"it": 20, "t0": 2.0, "dur": 0.1, "running": 0, "split": [0, 32, False], "split_s": 0.02}])
    p = tmp_path / "pdmux_rank3.jsonl"
    p.write_text("".join(_json.dumps(r) + "\n" for r in recs))
    r = R.rank_readout(R.load(str(p)))
    assert r["decode_holding_wall_s"] == 0.6 and r["decode_steps"] == 10 and r["idle_share"] == 0.25
    assert r["decode_step_gpu_ms_p50"] == 45.0 and r["split_enqueue_ms_p50"] == 20.0
    assert R.readout([str(p)])["closed_vs_M5_spec314"] == round(0.302 - 0.25, 4)


# ---------------------------------------------------------------------------------------------------------- 7
@needs_src
def test_every_scheduler_attribute_exists_on_the_v0520_scheduler(tmp_path):
    """The check that would have caught the final run's GX0 boot crash (stock init_pdmux reading self.gpu_id, which
    the v0.5.20 Scheduler keeps on self.ps): every self.X in the patched multiplexing_mixin and every sched.X in
    numinous_pdmux_dp must be assigned or defined somewhere in the Scheduler's class hierarchy (scheduler.py and
    its mixins), read statically from the v0.5.20 tree."""
    import re
    pkg = _copy(tmp_path)
    H.apply(str(pkg))
    srt = SRC / "srt"
    classes = {}
    for f in srt.rglob("*.py"):
        try:
            tree = ast.parse(f.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                classes.setdefault(n.name, []).append(n)
    attrs, meths, seen, stack = set(), set(), set(), ["Scheduler"]
    while stack:
        nm = stack.pop()
        if nm in seen or nm not in classes:
            continue
        seen.add(nm)
        for c in classes[nm]:
            stack += [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in c.bases]
            for b in c.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    meths.add(b.name)
                    for x in ast.walk(b):
                        if (isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == "self"
                                and isinstance(x.ctx, ast.Store)):
                            attrs.add(x.attr)
    assert {"Scheduler", "SchedulerMultiplexMixin"} <= seen
    mix = set(re.findall(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)", (pkg / H.MIX).read_text()))
    mine = set(re.findall(r"\bsched\.([A-Za-z_][A-Za-z0-9_]*)", (pkg / H.MODULE_DST).read_text()))
    known_ok = {"_numinous_idle_split"}           # set by the loop itself on batches, not on the scheduler
    missing = sorted(a for a in (mix | mine) - known_ok if a not in attrs and a not in meths)
    assert missing == [], f"not on the v0.5.20 Scheduler: {missing}"
