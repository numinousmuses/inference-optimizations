"""The scheduler trace hook and tracer (tracker 3.6).

CPU (runs anywhere): the tracer on stand-in scheduler classes, including four forked "ranks" that meet at a barrier
in place of the MLP-sync collective, with one rank made late on purpose, end to end through the report; the patcher
on a stand-in package; the wrapped method names present in v0.5.18 and v0.5.20 source (local SGLang checkout).
Box (skips without sglang): the hook applied to a copy of the installed package and imported in a clean interpreter
with and without NUMINOUS_SCHED_TRACE=1. GPU (skips without CUDA): forward timing from events on the forward stream
without blocking the loop.
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "operations/modal"))

import sched_trace_hooks as H  # noqa: E402
import sched_trace_report as R  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins with the method names the tracer wraps. A step: intake, schedule (prefill build with a delayer
# negotiation, then the MLP sync), run, process the result.
# ---------------------------------------------------------------------------------------------------------------
class FakeMode:
    def __init__(self, name):
        self.name = name

    def is_extend(self):
        return self.name == "EXTEND"


class FakeBatch:
    def __init__(self, mode, n, tokens):
        self.forward_mode = FakeMode(mode)
        self.reqs = list(range(n))
        self.extend_num_tokens = tokens
        self.global_num_tokens = None

    def batch_size(self):
        return len(self.reqs)


class FakeNegotiate:
    output_allow = True
    output_reason = "no_wait"


def make_classes():
    class PS:
        def __init__(self, rank):
            self.tp_rank = rank
            self.attn_dp_rank = rank

    class Delayer:
        def _negotiate_should_allow_prefill(self, local_prefillable, token_usage, **k):
            return FakeNegotiate()

    class Adapter:
        barrier = None

        def prepare_mlp_sync_batch(self, batch):
            if self.barrier is not None:
                self.barrier.wait(timeout=30)
            if batch is not None:
                batch.global_num_tokens = [batch.extend_num_tokens or batch.batch_size()] * 4
            return batch

    class Scheduler:
        def __init__(self, rank, plan):
            self.ps = PS(rank)
            self.plan = plan  # step -> (mode, n_reqs, tokens) or None
            self.step = 0
            self.forward_ct = 0
            self.dp_attn_adapter = Adapter()
            self.delayer = Delayer()
            self.running_batch = FakeBatch("DECODE", 3, 0)
            self.waiting_queue = [1, 2]
            self.chunked_req = None
            self.slow_result_at = None

        def ingest_requests(self):
            self.step += 1
            return []

        def get_new_batch_prefill(self, running):
            self.delayer._negotiate_should_allow_prefill(True, 0.5)
            p = self.plan.get(self.step)
            return FakeBatch(*p) if isinstance(p, tuple) else None

        def get_next_batch_to_run(self):
            b = self.get_new_batch_prefill(self.running_batch)
            if self.plan.get(self.step, "skip") == "skip":
                return None  # an idle loop: no batch, no sync
            return self.dp_attn_adapter.prepare_mlp_sync_batch(b or FakeBatch("IDLE", 0, 0))

        def run_batch(self, batch):
            self.forward_ct += 1
            time.sleep(0.001)
            return "result"

        def process_batch_result(self, batch, result):
            if self.slow_result_at == self.step:
                time.sleep(0.030)

        def on_idle(self):
            pass

    return Scheduler, Adapter, Delayer


def loop(s, steps: int):
    """event_loop_normal's shape."""
    for _ in range(steps):
        s.ingest_requests()
        batch = s.get_next_batch_to_run()
        if batch:
            s.process_batch_result(batch, s.run_batch(batch))
        else:
            s.on_idle()


def reset():
    H._TRACER = None
    H._WRAPPED.clear()


def rank_main(rank: int, out_dir: str, barrier, plan: dict, slow_rank: int, slow_step: int, steps: int):
    os.environ["NUMINOUS_SCHED_TRACE_DIR"] = out_dir
    reset()
    H._cuda_events = lambda s: (None, None, None)  # a forked rank must not touch CUDA
    Scheduler, Adapter, Delayer = make_classes()
    assert H.install(Scheduler, Adapter, Delayer) == []
    Adapter.barrier = barrier
    s = Scheduler(rank, plan)
    if rank == slow_rank:
        s.slow_result_at = slow_step
    loop(s, steps)
    H._TRACER.close()


def test_four_ranks_through_the_report_find_the_late_rank_and_why(tmp_path):
    out = tmp_path / "trace"
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(4)
    plan = {i: ("DECODE", 4, 0) for i in range(1, 41)}
    plan.update({25: ("EXTEND", 1, 4096)})
    procs = [ctx.Process(target=rank_main, args=(r, str(out), barrier, plan, 2, 20, 40)) for r in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    files = sorted(os.listdir(out))
    assert files == [f"rank{r}.jsonl" for r in range(4)]
    lines = [json.loads(x) for x in (out / "rank2.jsonl").read_text().splitlines()]
    assert "meta" in lines[0] and "Scheduler.run_batch" in lines[0]["meta"]["wrapped"]
    step = lines[5]
    assert set(step["phases"]) >= {"input", "schedule", "batch_build", "delayer", "mlp_sync", "launch", "result"}
    assert step["mode"] == "DECODE" and step["n_reqs"] == 4 and step["running"] == 3 and step["waiting"] == 2
    assert step["global_tokens"] == [4, 4, 4, 4] and step["delay_allow"] is True
    assert "gpu_ms" not in step  # no CUDA events in these ranks
    rep = R.report(R.load([str(out)]))
    assert rep["steps"] == 40 and rep["align"] == "sync"
    # rank 2's result processing at step 20 delays its arrival at the step-21 sync by ~30 ms
    s21 = next(s for s in rep["worst"] if s["step"] == 21)
    assert s21["late_rank"] == 2 and s21["max_lag_ms"] > 25.0, s21
    assert s21["late_phase"] == "prev.result" and s21["late_phase_excess_ms"] > 25.0
    # real processes add their own noise (start-up skew at step 1, scheduler jitter); the planted fault must stand out
    assert rep["late_rank_counts"].get(2, 0) >= 1
    # the others waited for it at the sync
    assert min(v for k, v in s21["sync_wait_ms"].items() if k != 2) > 20.0
    assert rep["prefill_steps"] == 1


def pd_rank_main(role: str, rank: int, out_dir: str, barrier, plan: dict, steps: int):
    """One rank of an in-box PD engine: the scheduler carries its disaggregation mode (v0.5.20 DisaggregationMode)."""
    os.environ["NUMINOUS_SCHED_TRACE_DIR"] = out_dir
    os.environ.pop("NUMINOUS_SCHED_TRACE_TAG", None)
    reset()
    H._cuda_events = lambda s: (None, None, None)
    Scheduler, Adapter, Delayer = make_classes()
    assert H.install(Scheduler, Adapter, Delayer) == []
    Adapter.barrier = barrier
    s = Scheduler(rank, plan)
    s.disaggregation_mode = type("DisaggregationMode", (), {"value": role})()
    loop(s, steps)
    H._TRACER.close()


def test_pd_roles_write_separate_files_and_report_separately(tmp_path):
    """In-box PD (arm P): a prefill engine and a decode engine with tp ranks 0..1 each share one trace dir. Each
    process must own its file (two engines appending to one rank<tp>.jsonl interleave records and cut lines), and the
    report aligns ranks within a role, never across roles."""
    out = tmp_path / "trace"
    ctx = mp.get_context("fork")
    pre_plan = {i: ("EXTEND", 1, 4096) for i in range(1, 21)}
    dec_plan = {i: ("DECODE", 8, 0) for i in range(1, 31)}
    b_pre, b_dec = ctx.Barrier(2), ctx.Barrier(2)
    procs = [ctx.Process(target=pd_rank_main, args=("prefill", r, str(out), b_pre, pre_plan, 20)) for r in range(2)]
    procs += [ctx.Process(target=pd_rank_main, args=("decode", r, str(out), b_dec, dec_plan, 30)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert sorted(os.listdir(out)) == ["decode_rank0.jsonl", "decode_rank1.jsonl",
                                       "prefill_rank0.jsonl", "prefill_rank1.jsonl"]
    for role, mode, n in (("prefill", "EXTEND", 20), ("decode", "DECODE", 30)):
        lines = [json.loads(x) for x in (out / f"{role}_rank1.jsonl").read_text().splitlines()]
        assert lines[0]["meta"]["role"] == role
        steps = [x for x in lines[1:] if "mode" in x]
        assert len(steps) == n and {x["mode"] for x in steps} == {mode}
    assert set(R.roles([str(out)])) == {"prefill", "decode"}
    with pytest.raises(ValueError):
        R.load([str(out)])  # several roles: never merged silently
    rp = R.report(R.load([str(out)], role="prefill"))
    rd = R.report(R.load([str(out)], role="decode"))
    assert rp["steps"] == 20 and rp["prefill_steps"] == 20
    assert rd["steps"] == 30 and rd["prefill_steps"] == 0
    assert R.main([str(out), "--json", str(tmp_path / "rep.json")]) == 0
    assert set(json.loads((tmp_path / "rep.json").read_text())) == {"prefill", "decode"}


def test_trace_role_tag_and_monolithic_default(monkeypatch):
    class S:
        pass
    s = S()
    monkeypatch.delenv("NUMINOUS_SCHED_TRACE_TAG", raising=False)
    assert H.trace_role(s) == ""                       # monolithic: rank<tp>.jsonl as before
    s.disaggregation_mode = type("M", (), {"value": "null"})()
    assert H.trace_role(s) == ""
    s.server_args = type("SA", (), {"disaggregation_mode": "decode"})()
    s.disaggregation_mode = None
    assert H.trace_role(s) == "decode"                 # falls back to server_args
    monkeypatch.setenv("NUMINOUS_SCHED_TRACE_TAG", "replica 1/a")
    assert H.trace_role(s) == "replica-1-a"            # explicit tag wins, made file-name safe


def test_idle_loops_are_coalesced_and_install_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMINOUS_SCHED_TRACE_DIR", str(tmp_path))
    reset()
    Scheduler, Adapter, Delayer = make_classes()
    assert H.install(Scheduler, Adapter, Delayer) == []
    n = len(H._WRAPPED)
    assert H.install(Scheduler, Adapter, Delayer) == [] and len(H._WRAPPED) == n, "no double wrapping"
    plan = {1: ("DECODE", 2, 0), 5: ("EXTEND", 1, 100)}
    plan.update({i: "skip" for i in (2, 3, 4)})
    s = Scheduler(0, plan)
    loop(s, 6)
    H._TRACER.close()
    rows = [json.loads(x) for x in (tmp_path / "rank0.jsonl").read_text().splitlines()]
    kinds = ["meta" if "meta" in r else "idle" if "idle_span" in r else r.get("mode") for r in rows]
    assert kinds == ["meta", "DECODE", "idle", "EXTEND", "idle"], kinds
    assert rows[2]["idle_span"] == 3 and rows[4]["idle_span"] == 1, "a trailing idle span is written at close"
    assert rows[3]["extend_tokens"] == 100 and rows[1]["extend_tokens"] == 0
    for r in rows[1:]:
        if "phases" in r:
            assert r["other"] >= 0 and r["dur"] >= sum(r["phases"].get(k, 0) for k in H.TOP_LEVEL) - 1e-6


def test_install_reports_what_it_cannot_find():
    reset()

    class Bare:
        def run_batch(self, b):
            return None

    missing = H.install(Bare, dp_adapter_cls=type("A", (), {}), delayer_cls=type("D", (), {}))
    assert "Scheduler.<boundary>" in missing and "Scheduler.get_next_batch_to_run" in missing
    assert "SchedulerDPAttnAdapter.prepare_mlp_sync_batch" in missing
    assert "Scheduler.run_batch" not in missing


def test_trace_file_size_is_capped(tmp_path):
    reset()
    Scheduler, _, _ = make_classes()
    tr = H.Tracer(Scheduler(0, {}), out_dir=str(tmp_path), max_mb=0.001)
    for i in range(200):
        tr._write({"rank": 0, "pad": "x" * 100, "i": i})
    tr.close()
    text = (tmp_path / "rank0.jsonl").read_text()
    assert len(text) < 2000 and text.rstrip().endswith('"truncated": true}')


# ---------------------------------------------------------------------------------------------------------------
# Patcher
# ---------------------------------------------------------------------------------------------------------------
def fake_package(tmp_path, body: str) -> pathlib.Path:
    root = tmp_path / "sglang"
    (root / "srt/managers").mkdir(parents=True)
    (root / "srt/managers/scheduler.py").write_text(body)
    return root


def test_patcher_appends_once_copies_the_tracer_and_fails_closed(tmp_path):
    root = fake_package(tmp_path, "import os\nimport logging\nlogger = logging.getLogger(__name__)\n\nclass Scheduler(object):\n    pass\n")
    assert H.apply(str(root), check=True) == ["sched_trace: anchor ok"]
    assert H.apply(str(root)) == ["sched_trace: applied to srt/managers/scheduler.py", f"module: {H.MODULE_DEST}"]
    src = (root / "srt/managers/scheduler.py").read_text()
    assert src.count("# numinous-hook:sched_trace") == 1 and 'os.environ.get("NUMINOUS_SCHED_TRACE") == "1"' in src
    assert (root / H.MODULE_DEST).read_text() == pathlib.Path(H.__file__).read_text()
    assert H.apply(str(root))[0] == "sched_trace: already applied"
    bad = fake_package(tmp_path / "b", "class Scheduler:\n    pass\n")
    with pytest.raises(SystemExit, match="has no 'import os'"):
        H.apply(str(bad))
    assert "numinous-hook" not in (bad / "srt/managers/scheduler.py").read_text()


FORK = pathlib.Path("/home/ubuntu/sglang-fork")


@pytest.mark.skipif(not (FORK / ".git").exists(), reason="needs the local SGLang checkout")
@pytest.mark.parametrize("tag", ["v0.5.18", "v0.5.20"])
def test_wrapped_names_exist_in_both_engine_versions(tag):
    """v0.5.18 is the GLM production image; v0.5.20 is the dev boxes. Both must trace fully."""
    show = lambda p: subprocess.run(["git", "-C", str(FORK), "show", f"{tag}:python/sglang/{p}"], capture_output=True,
                                    text=True, check=True, timeout=60).stdout
    sched = show("srt/managers/scheduler.py")
    for name in ("process_input_requests", *H.SCHEDULER_PHASES):
        assert f"    def {name}(" in sched, f"{tag}: Scheduler.{name}"
    for needle in ("import os\n", "logger = logging.getLogger(__name__)", "class Scheduler("):
        assert needle in sched
    assert "    def prepare_mlp_sync_batch(" in show("srt/managers/scheduler_components/dp_attn.py")
    assert "    def _negotiate_should_allow_prefill(" in show("srt/managers/prefill_delayer.py")


# ---------------------------------------------------------------------------------------------------------------
# Box: the installed engine
# ---------------------------------------------------------------------------------------------------------------
sglang_spec = importlib.util.find_spec("sglang")
needs_sglang = pytest.mark.skipif(sglang_spec is None, reason="needs an installed sglang (dev box)")


@pytest.fixture(scope="module")
def patched(tmp_path_factory) -> pathlib.Path:
    src = list(sglang_spec.submodule_search_locations)[0]
    base = tmp_path_factory.mktemp("sgl")
    shutil.copytree(src, base / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
    H.apply(str(base / "sglang"))
    return base


@needs_sglang
def test_anchor_resolves_on_the_installed_sglang():
    assert H.apply(list(sglang_spec.submodule_search_locations)[0], check=True)[0] in (
        "sched_trace: anchor ok", "sched_trace: already applied")


@needs_sglang
def test_patched_scheduler_traces_only_under_the_flag(patched, tmp_path):
    code = ("import sglang, sys, json; assert sglang.__file__.startswith(sys.argv[1]), sglang.__file__\n"
            "import sglang.srt.managers.scheduler as m\n"
            "from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter as A\n"
            "from sglang.srt.managers.prefill_delayer import PrefillDelayer as D\n"
            "from sglang.srt.managers import numinous_sched_trace as t\n"
            "S = m.Scheduler\n"
            "names = ['ingest_requests' if hasattr(S, 'ingest_requests') else 'process_input_requests', *t.SCHEDULER_PHASES]\n"
            "w = {n: getattr(getattr(S, n), '__numinous_trace__', False) for n in names}\n"
            "w['sync'] = getattr(A.prepare_mlp_sync_batch, '__numinous_trace__', False)\n"
            "w['delayer'] = getattr(D._negotiate_should_allow_prefill, '__numinous_trace__', False)\n"
            "print(json.dumps(w))")
    for flag, want in (("1", True), ("0", False)):
        env = {**os.environ, "PYTHONPATH": str(patched), "NUMINOUS_SCHED_TRACE": flag,
               "NUMINOUS_SCHED_TRACE_DIR": str(tmp_path)}
        r = subprocess.run([sys.executable, "-c", code, str(patched)], env=env, capture_output=True, text=True,
                           timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
        wrapped = json.loads(r.stdout.strip().splitlines()[-1])
        assert all(v is want for v in wrapped.values()), (flag, wrapped, r.stderr[-2000:])


# ---------------------------------------------------------------------------------------------------------------
# GPU: forward timing from events, without blocking the loop
# ---------------------------------------------------------------------------------------------------------------
def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
def test_gpu_forward_time_comes_from_events_on_the_forward_stream(tmp_path, monkeypatch):
    import torch

    monkeypatch.setenv("NUMINOUS_SCHED_TRACE_DIR", str(tmp_path))
    reset()
    Scheduler, Adapter, Delayer = make_classes()
    stream = torch.cuda.Stream()
    # spin-kernel cycles per ms, measured after warm-up (the first launches include lazy init and clock ramp)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        e0.record()
        torch.cuda._sleep(10_000_000)
        e1.record()
        e1.synchronize()
        cycles_per_ms = 10_000_000 / e0.elapsed_time(e1)

    def run_batch(self, batch):
        self.forward_ct += 1
        with torch.cuda.stream(stream):
            torch.cuda._sleep(int(20 * cycles_per_ms))  # ~20 ms of GPU work, launched asynchronously
        return "result"

    Scheduler.run_batch = run_batch
    assert H.install(Scheduler, Adapter, Delayer) == []
    s = Scheduler(0, {i: ("DECODE", 2, 0) for i in range(1, 6)})
    s.forward_stream = stream
    s.ingest_requests()
    b = s.get_next_batch_to_run()
    s.run_batch(b)
    t = time.perf_counter()
    s.ingest_requests()  # the next boundary: closes step 1 while its forward is still on the GPU
    boundary_ms = (time.perf_counter() - t) * 1e3
    assert boundary_ms < 5.0, f"the boundary blocked for {boundary_ms:.1f} ms"
    assert len(H._TRACER.pending) == 1, "step 1 waits for its events instead of syncing"
    torch.cuda.synchronize()
    s.run_batch(s.get_next_batch_to_run())  # finish step 2
    loop(s, 3)
    H._TRACER.close()
    rows = [json.loads(x) for x in (tmp_path / "rank0.jsonl").read_text().splitlines() if '"mode"' in x]
    assert len(rows) == 5
    for r in rows:
        assert 10.0 < r["gpu_ms"] < 40.0, (r["gpu_ms"], cycles_per_ms)
    assert all(r["phases"]["launch"] < 0.005 for r in rows), "launch is CPU time only"
