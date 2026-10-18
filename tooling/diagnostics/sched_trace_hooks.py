"""Per-rank scheduler loop tracing for SGLang (tracker 3.6: the intermittent GLM prefill straggler).

Two things in one file:

1. The patcher. One anchored hook appended to srt/managers/scheduler.py, plus this file copied into the package as
   srt/managers/numinous_sched_trace.py. The hook imports it and calls install(Scheduler) only when
   NUMINOUS_SCHED_TRACE=1, so a patched engine without the variable runs the stock loop.

       python3 sched_trace_hooks.py                 # patch the installed sglang package
       python3 sched_trace_hooks.py --root /tmp/sgl --check

2. The tracer. install() wraps scheduler methods at the class level, by name, so it holds across v0.5.18 (the GLM
   production image) and v0.5.20 without a per-release anchor inside the loop body. Every loop iteration of every
   rank becomes one JSON line in $NUMINOUS_SCHED_TRACE_DIR/rank<tp>.jsonl (default /tmp/numinous_sched_trace).
   Under PD the file is <role>_rank<tp>.jsonl (role = the scheduler's disaggregation mode, prefill or decode), and
   $NUMINOUS_SCHED_TRACE_TAG, if set, prefixes it the same way: an in-box PD node runs a prefill server and a decode
   server whose ranks share tp 0..n-1, and one rank<tp>.jsonl per pair of processes would interleave their records
   (64 KB buffered appends cut lines mid-record):

     it, t0 (wall, s), dur (s, boundary to next boundary), phases (s): input, schedule, batch_build, delayer,
     mlp_sync, launch, result, idle; other = dur minus the top-level phases (receive, loop glue)
     sync_step: the DP-attention MLP-sync count (every rank runs the same number, so it aligns ranks step by step)
     sync_t: wall time this rank entered the MLP sync (the last rank to arrive is the CPU straggler)
     delay_t / delay_allow / delay_reason: the prefill delayer negotiation this iteration, if any
     fwd: the scheduler's forward counter (aligns TP-only ranks), mode, bs, n_reqs, extend_tokens, global_tokens
     (every rank's token count after the sync), running, waiting, chunked
     gpu_ms: the forward's GPU time from CUDA events on the forward stream, read later with Event.query(), so tracing
     adds no host sync. On a straggler step the fast ranks' gpu_ms includes their wait inside the collectives.

   Under the overlap loop, `result` is the previous batch's result processing (it runs one iteration late).
   Idle iterations with no batch and no sync are coalesced into one {"idle_span": n, "idle_s": s} line.
   operations/modal/sched_trace_report.py turns the rank files into a per-step straggler report.

Cost (derived, not measured on a live engine): a handful of perf_counter calls and one json.dumps per iteration,
on the order of 20-40 us against 10-100 ms prefill passes; buffered writes flushed every 0.5 s.
"""
from __future__ import annotations

import argparse
import atexit
import collections
import functools
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------
# Patcher
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hook:
    name: str
    path: str  # relative to the sglang package directory
    anchor: str  # exact text; the insertion goes after it ("" = end of file)
    text: str

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


SCHED_TRACE = '''

# numinous-hook:sched_trace
# Numinous: per-rank scheduler loop timing and batch composition under NUMINOUS_SCHED_TRACE=1
# (srt/managers/numinous_sched_trace.py; operations/patches/sched_trace_hooks.py in numinous-inference).
if os.environ.get("NUMINOUS_SCHED_TRACE") == "1":
    try:
        from sglang.srt.managers import numinous_sched_trace as _numinous_sched_trace

        _numinous_sched_trace.install(Scheduler)
    except Exception as _e:  # noqa: BLE001
        logger.warning("numinous_sched_trace not installed: %s", _e)
'''

HOOKS = [Hook("sched_trace", "srt/managers/scheduler.py", "", SCHED_TRACE)]
MODULE_DEST = "srt/managers/numinous_sched_trace.py"


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook plus the module copy; raises SystemExit if an anchor is missing."""
    out, missing = [], []
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = open(path).read()
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        if h.anchor:
            n = src.count(h.anchor)
            if n != 1:
                missing.append(f"{h.name}: anchor found {n} times in {h.path}")
                continue
            new = src.replace(h.anchor, h.anchor + h.text, 1)
        else:
            new = src.rstrip("\n") + "\n" + h.text
        # the hook body leans on names scheduler.py already has at module level; without them, write nothing
        absent = [name for name in ("import os", "logger = ", "class Scheduler(") if name not in src]
        if absent:
            missing += [f"{h.name}: {h.path} has no '{name}'" for name in absent]
            continue
        if check:
            out.append(f"{h.name}: anchor ok")
            continue
        compile(new, path, "exec")  # never leave a file that does not parse
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("numinous sched_trace hooks: " + "; ".join(missing))
    dest = os.path.join(root, MODULE_DEST)
    if not check:
        shutil.copyfile(os.path.abspath(__file__), dest)
        out.append(f"module: {MODULE_DEST}")
    return out


# ----------------------------------------------------------------------------------------------------------------
# Tracer
# ----------------------------------------------------------------------------------------------------------------

# (method on Scheduler, phase). The boundary method opens an iteration; the first one present is used.
BOUNDARY = ("ingest_requests", "process_input_requests")
SCHEDULER_PHASES = {
    "get_next_batch_to_run": "schedule",
    "get_new_batch_prefill": "batch_build",
    "run_batch": "launch",
    "process_batch_result": "result",
    "on_idle": "idle",
}
TOP_LEVEL = ("input", "schedule", "launch", "result", "idle")


def _now() -> float:
    return time.perf_counter()


def trace_role(scheduler) -> str:
    """The file-name prefix of this scheduler's trace: $NUMINOUS_SCHED_TRACE_TAG, else its PD role ("prefill" or
    "decode"), else "" (a monolithic engine keeps rank<tp>.jsonl). Never raises."""
    tag = os.environ.get("NUMINOUS_SCHED_TRACE_TAG", "").strip()
    if tag:
        return "".join(c if c.isalnum() or c in "-." else "-" for c in tag)
    try:
        mode = getattr(scheduler, "disaggregation_mode", None)
        if mode is None:
            mode = getattr(getattr(scheduler, "server_args", None), "disaggregation_mode", None)
        mode = str(getattr(mode, "value", mode) or "").lower()
    except Exception:  # noqa: BLE001
        return ""
    return mode if mode in ("prefill", "decode") else ""


class Tracer:
    """One per scheduler process. Holds the open iteration, the pending GPU timings and the output file."""

    def __init__(self, scheduler, out_dir: str | None = None, max_mb: float | None = None):
        ps = getattr(scheduler, "ps", None)
        self.tp_rank = int(getattr(ps, "tp_rank", getattr(scheduler, "tp_rank", 0)) or 0)
        dp = getattr(ps, "attn_dp_rank", None)
        self.dp_rank = int(dp) if dp is not None else None
        out_dir = out_dir or os.environ.get("NUMINOUS_SCHED_TRACE_DIR", "/tmp/numinous_sched_trace")
        os.makedirs(out_dir, exist_ok=True)
        self.role = trace_role(scheduler)
        self.path = os.path.join(out_dir, f"{self.role}_rank{self.tp_rank}.jsonl" if self.role
                                 else f"rank{self.tp_rank}.jsonl")
        self.f = open(self.path, "a", buffering=1 << 16)
        self.max_bytes = int(float(max_mb or os.environ.get("NUMINOUS_SCHED_TRACE_MAX_MB", "512")) * 2**20)
        self.bytes = 0
        self.last_flush = _now()
        self.it = 0
        self.sync_step = 0
        self.cur: dict | None = None
        self.depth: dict[str, int] = collections.defaultdict(int)
        self.pending: collections.deque = collections.deque()
        self.idle_n = 0
        self.idle_s = 0.0
        self.scheduler = scheduler
        atexit.register(self.close)
        self._write({"meta": {"tp_rank": self.tp_rank, "dp_rank": self.dp_rank, "pid": os.getpid(), "t": time.time(),
                              "role": self.role, "wrapped": sorted(_WRAPPED)}})

    # -- iteration lifecycle ---------------------------------------------------------------------------------
    def boundary(self) -> None:
        t = _now()
        if self.cur is not None:
            self._close_iteration(t)
        self._drain_gpu(block=False)
        self.it += 1
        self.cur = {"rank": self.tp_rank, "dp": self.dp_rank, "it": self.it, "t0": time.time(), "_p0": t, "phases": {}}

    def _close_iteration(self, t: float) -> None:
        rec = self.cur
        self.cur = None
        rec["dur"] = t - rec.pop("_p0")
        top = sum(rec["phases"].get(k, 0.0) for k in TOP_LEVEL)
        rec["other"] = max(0.0, rec["dur"] - top)
        if "mode" not in rec and "sync_step" not in rec:
            self.idle_n += 1
            self.idle_s += rec["dur"]
            return
        if self.idle_n:
            self._write({"rank": self.tp_rank, "idle_span": self.idle_n, "idle_s": round(self.idle_s, 6)})
            self.idle_n, self.idle_s = 0, 0.0
        ev = rec.pop("_ev", None)
        if ev is None:
            self._emit(rec)
        else:
            self.pending.append((rec, ev))
            self._drain_gpu(block=False)

    def _drain_gpu(self, block: bool) -> None:
        while self.pending:
            rec, (e0, e1) = self.pending[0]
            try:
                if not block and not e1.query():
                    if len(self.pending) < 256:
                        return
                    rec["gpu_ms"] = None  # too far behind; never block the loop to find out
                else:
                    if block:
                        e1.synchronize()
                    rec["gpu_ms"] = round(e0.elapsed_time(e1), 4)
            except Exception:  # noqa: BLE001
                rec["gpu_ms"] = None
            self.pending.popleft()
            self._emit(rec)

    def _emit(self, rec: dict) -> None:
        rec["phases"] = {k: round(v, 6) for k, v in rec["phases"].items()}
        for k in ("dur", "other"):
            rec[k] = round(rec[k], 6)
        self._write(rec)

    def _write(self, rec: dict) -> None:
        if self.bytes > self.max_bytes:
            return
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        self.bytes += len(line)
        self.f.write(line)
        if self.bytes > self.max_bytes:
            self.f.write(json.dumps({"rank": self.tp_rank, "truncated": True}) + "\n")
        t = _now()
        if t - self.last_flush > 0.5:
            self.f.flush()
            self.last_flush = t

    def close(self) -> None:
        try:
            if self.cur is not None:
                self._close_iteration(_now())
            self._drain_gpu(block=True)
            if self.idle_n:
                self._write({"rank": self.tp_rank, "idle_span": self.idle_n, "idle_s": round(self.idle_s, 6)})
                self.idle_n, self.idle_s = 0, 0.0
            self.f.flush()
        except Exception:  # noqa: BLE001
            pass

    # -- phase accounting ------------------------------------------------------------------------------------
    def add(self, phase: str, dt: float) -> None:
        if self.cur is not None:
            ph = self.cur["phases"]
            ph[phase] = ph.get(phase, 0.0) + dt

    def note(self, **kv) -> None:
        if self.cur is not None:
            self.cur.update(kv)

    def on_sync(self, t_wall: float) -> None:
        # With speculative decoding under DP attention an iteration can sync twice (get_next_batch_to_run); every
        # rank does the same, so the count stays aligned. sync_t keeps the first entry: that is where a late rank
        # makes the others wait.
        self.sync_step += 1
        if self.cur is not None:
            self.cur["sync_step"] = self.sync_step
            self.cur["syncs"] = self.cur.get("syncs", 0) + 1
            self.cur.setdefault("sync_t", t_wall)

    def on_run_batch(self, scheduler, batch, e0, e1) -> None:
        if self.cur is None:
            return
        info = batch_info(scheduler, batch)
        self.cur.update(info)
        if e0 is not None:
            self.cur["_ev"] = (e0, e1)


def batch_info(scheduler, batch) -> dict:
    """Composition of the batch this rank runs, and the queue it leaves behind. Never raises."""
    out: dict = {}
    try:
        fm = getattr(batch, "forward_mode", None)
        out["mode"] = getattr(fm, "name", str(fm))
        out["fwd"] = getattr(scheduler, "forward_ct", None)
        reqs = getattr(batch, "reqs", None) or []
        out["n_reqs"] = len(reqs)
        try:
            out["bs"] = batch.batch_size()
        except Exception:  # noqa: BLE001
            out["bs"] = len(reqs)
        ext = getattr(batch, "extend_num_tokens", None)
        is_extend = bool(fm is not None and hasattr(fm, "is_extend") and fm.is_extend())
        out["extend_tokens"] = int(ext) if (ext and is_extend) else 0
        g = getattr(batch, "global_num_tokens", None)
        if g is not None:
            out["global_tokens"] = [int(x) for x in g]
        rb = getattr(scheduler, "running_batch", None)
        out["running"] = len(getattr(rb, "reqs", None) or [])
        wq = getattr(scheduler, "waiting_queue", None)
        out["waiting"] = len(wq) if wq is not None else None
        out["chunked"] = getattr(scheduler, "chunked_req", None) is not None
    except Exception as e:  # noqa: BLE001
        out["info_error"] = repr(e)[:200]
    return out


_TRACER: Tracer | None = None
_WRAPPED: list[str] = []


def tracer_for(scheduler) -> Tracer:
    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer(scheduler)
    return _TRACER


def _cuda_events(scheduler):
    """A pair of timing events and the stream the forward runs on, or (None, None, None) without CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None, None
        stream = getattr(scheduler, "forward_stream", None) or torch.cuda.current_stream()
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), stream
    except Exception:  # noqa: BLE001
        return None, None, None


def _wrap_boundary(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        tr = tracer_for(self)
        tr.boundary()
        t = _now()
        try:
            return fn(self, *a, **k)
        finally:
            tr.add("input", _now() - t)
    wrapper.__numinous_trace__ = True
    return wrapper


def _wrap_phase(fn, phase: str):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        tr = tracer_for(self)
        t = _now()
        try:
            return fn(self, *a, **k)
        finally:
            tr.add(phase, _now() - t)
    wrapper.__numinous_trace__ = True
    return wrapper


def _wrap_run_batch(fn):
    @functools.wraps(fn)
    def wrapper(self, batch, *a, **k):
        tr = tracer_for(self)
        e0, e1, stream = _cuda_events(self)
        if e0 is not None:
            e0.record(stream)
        t = _now()
        try:
            return fn(self, batch, *a, **k)
        finally:
            tr.add("launch", _now() - t)
            if e1 is not None:
                e1.record(stream)
            tr.on_run_batch(self, batch, e0, e1)
    wrapper.__numinous_trace__ = True
    return wrapper


def _wrap_sync(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        tr = _TRACER
        if tr is not None:
            tr.on_sync(time.time())
        t = _now()
        try:
            return fn(self, *a, **k)
        finally:
            if tr is not None:
                tr.add("mlp_sync", _now() - t)
    wrapper.__numinous_trace__ = True
    return wrapper


def _wrap_delayer(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        tr = _TRACER
        t_wall, t = time.time(), _now()
        out = fn(self, *a, **k)
        if tr is not None:
            tr.add("delayer", _now() - t)
            tr.note(delay_t=t_wall, delay_allow=bool(getattr(out, "output_allow", True)),
                    delay_reason=str(getattr(out, "output_reason", ""))[:40])
        return out
    wrapper.__numinous_trace__ = True
    return wrapper


def _patch(cls, name: str, make) -> bool:
    fn = cls.__dict__.get(name)
    if fn is None:
        for base in cls.__mro__[1:]:
            if name in base.__dict__:
                fn = base.__dict__[name]
                break
    if fn is None or getattr(fn, "__numinous_trace__", False):
        return fn is not None
    setattr(cls, name, make(fn))
    _WRAPPED.append(f"{cls.__name__}.{name}")
    return True


def install(scheduler_cls, dp_adapter_cls=None, delayer_cls=None) -> list[str]:
    """Wrap the loop methods; returns what could not be found (logged, never raised: tracing must not stop serving)."""
    missing = []
    boundary = next((b for b in BOUNDARY if hasattr(scheduler_cls, b)), None)
    if boundary is None or not _patch(scheduler_cls, boundary, _wrap_boundary):
        missing.append("Scheduler.<boundary>")
    for name, phase in SCHEDULER_PHASES.items():
        make = _wrap_run_batch if name == "run_batch" else (lambda f, p=phase: _wrap_phase(f, p))
        if not _patch(scheduler_cls, name, make):
            missing.append(f"Scheduler.{name}")
    if dp_adapter_cls is None:
        try:
            from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter as dp_adapter_cls
        except Exception:  # noqa: BLE001
            dp_adapter_cls = None
    if dp_adapter_cls is None or not _patch(dp_adapter_cls, "prepare_mlp_sync_batch", _wrap_sync):
        missing.append("SchedulerDPAttnAdapter.prepare_mlp_sync_batch")
    if delayer_cls is None:
        try:
            from sglang.srt.managers.prefill_delayer import PrefillDelayer as delayer_cls
        except Exception:  # noqa: BLE001
            delayer_cls = None
    if delayer_cls is None or not _patch(delayer_cls, "_negotiate_should_allow_prefill", _wrap_delayer):
        missing.append("PrefillDelayer._negotiate_should_allow_prefill")
    if missing:
        log.warning("numinous_sched_trace: not traced: %s", ", ".join(missing))
    log.info("numinous_sched_trace: wrapped %s", ", ".join(_WRAPPED))
    return missing


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
