"""Per-step scheduler host timestamps for DP-attention stall diagnosis (agent rtxv41, 2026-10-01).

Installed into the engine as srt/managers/numinous_step_trace.py by operations/patches/rtx_mhc_jit_hooks.py and
enabled with NUMINOUS_STEP_TRACE=<dir>. Wraps the scheduler loop's four phases (ingest_requests,
get_next_batch_to_run, run_batch, process_batch_result) and writes one JSON line per call to
<dir>/steps_dp<rank>_<pid>.jsonl:
  {"t": wall start (s), "ph": phase, "ms": duration, "mode": forward mode, "tok": extend tokens, "bs": batch size}
run_batch under the overlap scheduler only launches the forward. A long run_batch therefore means host work
before the launch: JIT compiles, metadata building, or a blocking collective in the DP metadata sync. A long
process_batch_result means waiting on the previous forward. Read the files side by side per rank to see which
rank is late and in which phase. operations/kernels/rtxflash_stall_diag.py summarizes them.
"""
from __future__ import annotations

import json
import os
import time

PHASES = ("ingest_requests", "get_next_batch_to_run", "run_batch", "process_batch_result")
_buf: list = []
_fh = None
_rank = None


def _batch_info(batch):
    if batch is None:
        return {}
    out = {}
    try:
        out["mode"] = str(getattr(batch, "forward_mode", "")).split(".")[-1]
        reqs = getattr(batch, "reqs", None)
        out["bs"] = len(reqs) if reqs is not None else None
        ext = getattr(batch, "extend_num_tokens", None)
        if ext is not None:
            out["tok"] = int(ext)
    except Exception:  # noqa: BLE001
        pass
    return out


def _flush(force=False):
    global _fh
    if not _buf or (len(_buf) < 256 and not force):
        return
    if _fh is None:
        d = os.environ.get("NUMINOUS_STEP_TRACE", "/tmp/numinous_step_trace")
        os.makedirs(d, exist_ok=True)
        rank = _rank if _rank is not None else os.environ.get("NUMINOUS_STEP_TRACE_RANK", "x")
        _fh = open(os.path.join(d, f"steps_dp{rank}_{os.getpid()}.jsonl"), "a", buffering=1)
    _fh.write("".join(json.dumps(r) + "\n" for r in _buf))
    _buf.clear()


def _wrap(name, fn):
    def wrapper(self, *args, **kwargs):
        global _rank
        if _rank is None:
            ps = getattr(self, "ps", None)
            _rank = getattr(ps, "dp_rank", None) if ps is not None else getattr(self, "dp_rank", None)
            _rank = "x" if _rank is None else _rank
        t = time.time()
        p0 = time.perf_counter()
        res = fn(self, *args, **kwargs)
        rec = {"t": round(t, 4), "ph": name, "ms": round((time.perf_counter() - p0) * 1e3, 3)}
        if name == "run_batch" and args:
            rec.update(_batch_info(args[0]))
        elif name == "get_next_batch_to_run":
            rec.update(_batch_info(getattr(res, "batch_to_run", None)))
        _buf.append(rec)
        _flush()
        return res
    wrapper.__wrapped__ = fn
    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


def install(cls) -> list:
    """Wrap the scheduler class's phase methods (idempotent). Returns the wrapped names."""
    done = []
    for name in PHASES:
        fn = getattr(cls, name, None)
        if fn is None or getattr(fn, "__wrapped__", None) is not None:
            continue
        setattr(cls, name, _wrap(name, fn))
        done.append(name)
    import atexit

    atexit.register(_flush, True)
    return done
