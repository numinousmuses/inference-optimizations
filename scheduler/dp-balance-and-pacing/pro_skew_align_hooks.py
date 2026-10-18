"""Prefill alignment under DP attention: hold mixed-mode prefills until enough
ranks want to prefill, so they go together and no single rank stalls the MoE barrier.

    python3 pro_skew_align_hooks.py                  # patch the installed sglang (v0.5.20)
    python3 pro_skew_align_hooks.py --root /tmp/sgl --check

Runtime: NUMINOUS_PREFILL_ALIGN=<min_ranks> (0 or unset = off; "1" = default 3).
When active and the delayer sees a "mixed" state (some ranks prefillable, some not):
- if fewer than min_ranks want to prefill, delay (hold for more ranks to accumulate
  waiting requests and become prefillable)
- if >= min_ranks want to prefill, release immediately (balanced multi-rank prefill)
- the max delay is still bounded by max_delay_passes and max_delay_ms

The complementary approach to the balanced budget (pro_skew_budget_hooks.py): instead of
shrinking the big rank's work, this makes more ranks do useful prefill work in the same
step, reducing the imbalance at the source.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

MODULE_PATH = "srt/managers/numinous_pro_skew_align.py"

MODULE_SOURCE = r'''"""Numinous prefill alignment for DP attention (v0.5.20): ride-along and min-ranks release.

Stock (source_code, prefill_delayer.py): in the "mixed" state (some ranks prefillable, some not) the delayer holds
every prefill up to --prefill-delayer-max-delay-passes (30) waiting for ALL ranks to become prefillable. A rank in
the middle of a chunked prefill extends every pass regardless of the verdict (PrefillAdder.add_chunked_req). So the
step is an extend step anyway, and the held ranks idle inside the MoE collectives for the whole extend (~274 ms vs
32 ms decode at 4096 tokens, ~50k context, 8xB200 MegaMoE). Measured on agent traffic: 676/736 extending ranks in
partial steps were such continuations, and in 69% of those steps a held rank had queued requests.

NUMINOUS_PREFILL_RIDE=1: when any rank is mid-chunk (gathered as prefillable value 2), every other prefillable rank
is released into that step (verdict allow). Idle ranks do useful prefill during a wait they pay anyway.
NUMINOUS_PREFILL_ALIGN=<n> ("1" = 3): in the mixed state, release as soon as >= n ranks are prefillable (earlier
than stock, which waits for all).
NUMINOUS_PREFILL_IDLE_RELEASE=1: in the mixed state, release when no non-prefillable rank has running requests. The
hold exists so a prefill does not stall other ranks' decode; with every other rank idle it only adds latency
(measured: a lone request waited max_delay_passes x ~83 ms idle pass, 20 s at 240 passes). Makes long holds
(--prefill-delayer-max-delay-passes 120+) safe at low load.

Mechanism (class-level wrappers, installed from the scheduler.py hook):
- PrefillAdder.add_chunked_req sets a module flag while it runs; the delayer call made inside it passes
  local_prefillable=2 into the gather (stock and pro_skew_budget gathers both write int(local_prefillable)).
- PrefillDelayer._gather_info is wrapped pass-through to keep the gathered tensor of this call.
- PrefillDelayer._negotiate_should_allow_prefill_pure applies the overrides. Every rank sees the same gathered tensor,
  so every rank takes the same decision. Overrides report output_reason "wait_success" (an existing value: the
  stock debug assert and the metrics label set stay valid); counts are logged by this module.
"""
from __future__ import annotations

import functools
import logging
import os

log = logging.getLogger(__name__)

DEFAULT_MIN_RANKS = 3
FORCED = 2  # gathered prefillable value of a rank that extends this pass regardless of the verdict
_WRAPPED: list[str] = []
_min_ranks: int = 0
_ride: bool = False
_idle_release: bool = False
_last_running = None  # gathered running-batch column of the latest delayer gather
_local_running = 0  # this rank's running requests at the start of get_new_batch_prefill (this pass)
_in_chunked: bool = False
_last_col0 = None  # gathered prefillable column of the latest delayer gather (list of ints)
_counts = {"ride": 0, "align": 0, "idle": 0, "passes": 0}


def parse_min_ranks(value) -> int:
    if value is None:
        return 0
    value = str(value).strip()
    if value in ("", "0"):
        return 0
    n = int(value)
    if n < 0:
        raise ValueError(f"NUMINOUS_PREFILL_ALIGN must be >= 0, got {value}")
    return DEFAULT_MIN_RANKS if n == 1 else n


def parse_ride(value) -> bool:
    return str(value or "").strip() not in ("", "0")


def decide(col0, local_prefillable, stock_allow: bool, stock_reason: str, status: str, min_ranks: int,
           ride: bool, running=None, idle_release: bool = False):
    """Pure decision. col0: gathered prefillable values per rank (0, 1, or FORCED); running: gathered running
    requests per rank. Returns the override reason ("ride" / "align" / "idle") or None to keep the stock verdict."""
    if stock_allow or stock_reason != "delay" or status != "mixed" or col0 is None:
        return None
    lp = int(local_prefillable)
    if lp != 1:  # not prefillable, or this rank is the forced one (it extends anyway)
        return None
    if ride and any(int(v) == FORCED for v in col0):
        return "ride"
    if min_ranks > 0 and sum(1 for v in col0 if int(v) > 0) >= min_ranks:
        return "align"
    if idle_release and running is not None and len(running) == len(col0) and all(
            int(r) == 0 for v, r in zip(col0, running) if int(v) == 0):
        return "idle"
    return None


def _patch(cls, name, make) -> bool:
    fn = cls.__dict__.get(name)
    if fn is None:
        for base in cls.__mro__[1:]:
            if name in base.__dict__:
                fn = base.__dict__[name]
                break
    if fn is None or getattr(fn, "__numinous_align__", False):
        return fn is not None
    setattr(cls, name, make(fn))
    _WRAPPED.append(f"{cls.__name__}.{name}")
    return True


def _wrap_add_chunked_req(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        global _in_chunked
        _in_chunked = True
        try:
            return fn(self, *a, **k)
        finally:
            _in_chunked = False
    wrapper.__numinous_align__ = True
    return wrapper


def _wrap_get_new_batch_prefill(fn):
    """Record this rank's running requests for the pass. A rank that is not prefillable reaches the delayer only
    through PrefillDelayerSinglePassExecutor.finalize(), which negotiates with running_batch=0: the gathered running
    column is then 0 for exactly the ranks idle-release must inspect (measured: idle_d120 -48%, every mixed pass
    released)."""
    @functools.wraps(fn)
    def wrapper(self, running_batch, *a, **k):
        global _local_running
        try:
            _local_running = int(running_batch.batch_size())
        except Exception:  # noqa: BLE001
            _local_running = len(getattr(running_batch, "reqs", None) or [])
        return fn(self, running_batch, *a, **k)
    wrapper.__numinous_align__ = True
    return wrapper


def _wrap_gather_info(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        global _last_col0, _last_running
        out = fn(self, *a, **k)
        try:
            _last_col0 = [int(v) for v in out[:, 0].tolist()]
            _last_running = [int(v) for v in out[:, 2].tolist()]
        except Exception:  # noqa: BLE001
            _last_col0 = _last_running = None
        return out
    wrapper.__numinous_align__ = True
    return wrapper


def _wrap_negotiate_pure(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        global _last_col0, _last_running
        lp = k.get("local_prefillable")
        if _ride and _in_chunked and lp:
            k["local_prefillable"] = FORCED
        if _idle_release:
            # report the true running count (finalize() passes 0); only the "all" branch reads this column in
            # stock, and there every rank already passes its running_batch, so stock decisions are unchanged
            k["running_batch"] = max(int(k.get("running_batch") or 0), _local_running)
        _last_col0 = _last_running = None
        result = fn(self, *a, **k)
        col0 = _last_col0
        local = k.get("local_prefillable", lp)
        why = decide(col0, local if local is not None else 0, result.output_allow, result.output_reason,
                     result.input_estimation, _min_ranks, _ride, _last_running, _idle_release)
        if col0 is not None:
            # the FORCED value must not inflate the prefillable count reported to metrics
            result = result._replace(num_prefillable=sum(1 for v in col0 if v > 0))
        _counts["passes"] += 1
        if why is not None:
            _counts[why] += 1
            prev = k.get("prev_state", a[0] if a else None)
            result = result._replace(next_state=None, output_allow=True, output_reason="wait_success",
                                     wait_forward_passes=prev.delayed_count if prev is not None else 0)
        if _counts["passes"] % 5000 == 0:
            log.info("numinous_pro_skew_align: %s", _counts)
        return result
    wrapper.__numinous_align__ = True
    return wrapper


_logged = False


def install(scheduler_cls=None) -> list[str]:
    """Called from the scheduler.py hook when NUMINOUS_PREFILL_ALIGN / _RIDE / _IDLE_RELEASE is set."""
    global _min_ranks, _ride, _idle_release, _logged
    _min_ranks = parse_min_ranks(os.environ.get("NUMINOUS_PREFILL_ALIGN"))
    _ride = parse_ride(os.environ.get("NUMINOUS_PREFILL_RIDE"))
    _idle_release = parse_ride(os.environ.get("NUMINOUS_PREFILL_IDLE_RELEASE"))
    if _min_ranks <= 0 and not _ride and not _idle_release:
        return ["disabled"]
    missing = []
    try:
        from sglang.srt.managers.prefill_delayer import PrefillDelayer
        from sglang.srt.managers.schedule_policy import PrefillAdder
        for cls, name, make in ((PrefillDelayer, "_negotiate_should_allow_prefill_pure", _wrap_negotiate_pure),
                                (PrefillDelayer, "_gather_info", _wrap_gather_info),
                                (PrefillAdder, "add_chunked_req", _wrap_add_chunked_req)):
            if not _patch(cls, name, make):
                missing.append(f"{cls.__name__}.{name}")
        if _idle_release and scheduler_cls is not None and not _patch(
                scheduler_cls, "get_new_batch_prefill", _wrap_get_new_batch_prefill):
            missing.append("Scheduler.get_new_batch_prefill")
    except ImportError as e:
        missing.append(f"import: {e}")
    if missing:
        log.warning("numinous_pro_skew_align: not patched: %s", ", ".join(missing))
    if not _logged:
        _logged = True
        log.info("numinous_pro_skew_align: ride=%s idle_release=%s min_ranks=%d, wrapped %s", _ride, _idle_release,
                 _min_ranks, ", ".join(_WRAPPED))
    return missing
'''

SCHEDULER_HOOK = '''

# numinous-hook:pro_skew_align_install
# Numinous prefill alignment under NUMINOUS_PREFILL_ALIGN=<min_ranks> / NUMINOUS_PREFILL_RIDE=1
if (os.environ.get("NUMINOUS_PREFILL_ALIGN") or os.environ.get("NUMINOUS_PREFILL_RIDE")
        or os.environ.get("NUMINOUS_PREFILL_IDLE_RELEASE")):
    try:
        from sglang.srt.managers import numinous_pro_skew_align as _numinous_psa_install

        _numinous_psa_install.install(Scheduler)
    except Exception as _e:  # noqa: BLE001
        logger.warning("numinous_pro_skew_align not installed: %s", _e)
'''


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


HOOKS = [
    Hook("pro_skew_align_install", "srt/managers/scheduler.py", "", SCHEDULER_HOOK),
]


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def _write_file(path: str, text: str) -> None:
    compile(text, path, "exec")
    tmp = f"{path}.numinous.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def apply(root: str, check: bool = False) -> list[str]:
    out, missing = [], []
    for h in HOOKS:
        path = os.path.join(root, h.path)
        if not os.path.exists(path):
            missing.append(f"{h.name}: {h.path} not found")
            continue
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
        if check:
            out.append(f"{h.name}: anchor ok")
            continue
        compile(new, path, "exec")
        _write_file(path, new)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("pro_skew_align hooks: " + "; ".join(missing))
    mod = os.path.join(root, MODULE_PATH)
    if check:
        out.append(f"module: {'up to date' if os.path.exists(mod) and open(mod).read() == MODULE_SOURCE else 'would write'}")
        return out
    _write_file(mod, MODULE_SOURCE)
    out.append(f"module: wrote {MODULE_PATH}")
    return out


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
