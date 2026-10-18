"""Balanced prefill budget under DP attention: cap a rank's prefill chunk to
max(floor, max_other_ranks_extend) so a lone prefill does not stall every DP rank
behind MegaMoE's cross-rank barrier for 61 layers.

    python3 pro_skew_budget_hooks.py                  # patch the installed sglang (v0.5.20)
    python3 pro_skew_budget_hooks.py --root /tmp/sgl --check

Runtime: NUMINOUS_PREFILL_BALANCE=<floor_tokens> (0 or unset = off; "1" = default 1024).
When active and this rank would extend while other ranks decode:
- the chunk budget is capped to max(floor, max_other_extends) -- so balanced multi-rank
  prefills run uncapped, and a lone prefill does smaller chunks
- the cap is page/alignment-safe (rounded to lcm of page_size, truncation_align_size)
- the cap never drops below one page

Origin: ported from the v0.5.18 balanced-budget overlay (agents/status/megamoe.md); that overlay
modified prefill_delayer.py and schedule_policy.py inline. This v0.5.20 port is an anchored hook
that installs a module (monkey-patches PrefillDelayer and the scheduler's chunk-size path) when the
env var is set.

Why re-check on v0.5.20: v0.5.20 has breakable prefill CUDA graphs and the decode-to-extend
conversion (dp_attn.py: maybe_convert_decode_to_extend). On v0.5.18 the cap created extend steps
that fell back to eager execution (no decode CUDA graphs); on v0.5.20 the decode ranks convert to
1-token extend views and replay breakable graphs, so the CUDA-graph cost of additional extend steps
is lower. The balanced budget should therefore lose less than it did on v0.5.18.

Measured outcome (8xB200, MegaMoE W4A4 + balance, 8-turn Pro traffic, 2 repeats, agent proskew 2026-09-28):
floor 512: -0.1% @64 / +0.4% @128; floor 1024: +3.3% / +1.8% vs the same gate's control (control repeat spread up
to ~5%); accuracy within the boot-to-boot floor. Neutral: on this traffic the partial-extend steps are chunk
continuations whose cost is the extending rank's own attention at ~50k context (lone 4096 extend 274 ms, 4x1024
~358 ms), so shrinking the chunk trades prefill time for decode steps at about par. See research/proskew/README.md.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# MODULE SOURCE -- installed as srt/managers/numinous_pro_skew_budget.py
# ---------------------------------------------------------------------------

MODULE_PATH = "srt/managers/numinous_pro_skew_budget.py"

MODULE_SOURCE = r'''"""Numinous balanced prefill budget for DP attention (v0.5.20).

Under DP attention every layer's MoE collective (MegaMoE barriers, TP-MoE all-reduce, EP gather)
makes each rank wait for the slowest rank. A rank prefilling a 4,096-token chunk runs ~10-40x more
attention work per layer than a decode rank, and the decode ranks idle. This module caps a rank's
prefill chunk to max(floor, max_other_ranks_extend) so a lone prefill does smaller chunks while
balanced multi-rank prefills are uncapped.

Mechanism:
1. install() monkey-patches PrefillDelayer to gather planned extend tokens per rank
   (one extra int64 per rank in the existing all_gather) and compute a balanced cap.
2. install() monkey-patches the scheduler's chunk-size determination to apply the cap.
3. A per-process global holds the cap between the delayer's negotiate call (where the
   cross-rank gather runs) and the scheduler's chunk-size point (where it is consumed).

Controlled by NUMINOUS_PREFILL_BALANCE=<floor_tokens> (0 or unset = off; "1" = default 1024).
"""
from __future__ import annotations

import functools
import logging
import math
import os
from typing import Optional

import torch

log = logging.getLogger(__name__)

DEFAULT_FLOOR = 1024
_WRAPPED: list[str] = []

# ---- per-process state ---------------------------------------------------
_floor: int = 0
_cap: Optional[int] = None     # the balanced cap for the current scheduling pass
_planned_tokens: int = 0       # this rank's planned extend tokens for the current pass
_last_gather: Optional[torch.Tensor] = None  # last gather result (dp_size x fields)


def parse_floor(value) -> int:
    if value is None:
        return 0
    value = str(value).strip()
    if value in ("", "0"):
        return 0
    n = int(value)
    if n < 0:
        raise ValueError(f"NUMINOUS_PREFILL_BALANCE must be >= 0, got {value}")
    return DEFAULT_FLOOR if n == 1 else n


def get_balanced_cap() -> Optional[int]:
    """Return the balanced cap computed by the most recent negotiate call, or None."""
    return _cap


def set_planned_tokens(n: int) -> None:
    """Set this rank's planned extend tokens for the current pass."""
    global _planned_tokens
    _planned_tokens = max(0, int(n))


def compute_cap(gathered_planned: list[int], this_rank_planned: int,
                floor: int, align: int = 1) -> Optional[int]:
    """Cap for this rank: max(floor, max_of_other_ranks_planned).

    Returns None if no cap should be applied (all ranks extending similarly,
    or no other rank is decoding).
    """
    if floor <= 0 or this_rank_planned <= 0:
        return None
    others = list(gathered_planned)
    # Remove one occurrence of our own planned tokens
    if this_rank_planned in others:
        others.remove(this_rank_planned)
    # If no other rank is active (all planned 0), no skew to fix
    has_decode = any(p == 0 for p in others)
    if not has_decode:
        return None
    max_other_extend = max(others) if others else 0
    cap = max(floor, max_other_extend)
    if align > 1:
        cap = max(align, -(-cap // align) * align)  # ceiling alignment
    if cap >= this_rank_planned:
        return None  # cap would not reduce the budget
    return cap


def scheduler_chunk_size(base, page_size, truncation_align_size):
    """Called by the scheduler right after its chunk-size determination.
    Applies the balanced cap (if any) and returns the new chunk size."""
    global _cap
    cap = _cap
    if cap is None or base is None or base <= 0:
        return base
    align = math.lcm(max(1, int(page_size or 1)), max(1, int(truncation_align_size or 1)))
    cap = max(align, -(-cap // align) * align)  # ceiling alignment
    out = min(base, cap)
    return out


# ---- monkey-patching -----------------------------------------------------

def _patch(cls, name, make) -> bool:
    fn = cls.__dict__.get(name)
    if fn is None:
        for base in cls.__mro__[1:]:
            if name in base.__dict__:
                fn = base.__dict__[name]
                break
    if fn is None or getattr(fn, "__numinous_skew__", False):
        return fn is not None
    setattr(cls, name, make(fn))
    _WRAPPED.append(f"{cls.__name__}.{name}")
    return True


def _wrap_delayer_init(fn):
    """Wrap PrefillDelayer.__init__ to allocate a 6-column gather buffer (instead of 5)."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        # Expand the gather buffer: add a 6th column for planned_tokens
        if hasattr(self, "_global_info_buffer") and self._global_info_buffer.shape[-1] == 5:
            shape = list(self._global_info_buffer.shape)
            shape[-1] = 6
            self._global_info_buffer = torch.empty(
                shape, dtype=self._global_info_buffer.dtype,
                device=self._global_info_buffer.device,
            )
    wrapper.__numinous_skew__ = True
    return wrapper


def _wrap_gather_info(fn):
    """Wrap PrefillDelayer._gather_info to include planned_tokens as the 6th field."""
    @functools.wraps(fn)
    def wrapper(self, local_prefillable, local_token_watermark_force_allow,
                running_batch=0, max_prefill_bs=0, waiting_queue_len=0):
        global _last_gather
        local_info = torch.tensor(
            [
                int(local_prefillable),
                int(local_token_watermark_force_allow),
                running_batch,
                max_prefill_bs,
                waiting_queue_len,
                _planned_tokens,
            ],
            device=self._gather_device,
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self._global_info_buffer.flatten(),
            local_info,
            group=self._gather_group,
        )
        tp0_info = self._global_info_buffer[:, 0, :]
        _last_gather = tp0_info
        return tp0_info
    wrapper.__numinous_skew__ = True
    return wrapper


def _wrap_negotiate(fn):
    """Wrap PrefillDelayer._negotiate_should_allow_prefill to compute balanced_cap."""
    @functools.wraps(fn)
    def wrapper(self, local_prefillable, token_usage, running_batch=0,
                max_prefill_bs=0, max_running_requests=0, waiting_queue_len=0):
        global _cap
        result = fn(self, local_prefillable, token_usage, running_batch=running_batch,
                    max_prefill_bs=max_prefill_bs, max_running_requests=max_running_requests,
                    waiting_queue_len=waiting_queue_len)
        # Compute balanced cap after the gather
        if result.output_allow and _floor > 0 and _last_gather is not None:
            planned = _last_gather[:, 5].tolist() if _last_gather.shape[-1] > 5 else []
            _cap = compute_cap([int(p) for p in planned], _planned_tokens, _floor)
        else:
            _cap = None
        return result
    wrapper.__numinous_skew__ = True
    return wrapper


def _wrap_negotiate_single_pass(fn):
    """Wrap PrefillDelayerSinglePassExecutor.negotiate_should_allow_prefill
    to pass planned_tokens before the inner call."""
    @functools.wraps(fn)
    def wrapper(self, local_prefillable, running_batch=0, max_prefill_bs=0,
                max_running_requests=0, waiting_queue_len=0,
                planned_tokens=0):
        # Store planned tokens before the gather runs
        set_planned_tokens(planned_tokens if local_prefillable else 0)
        return fn(self, local_prefillable, running_batch=running_batch,
                  max_prefill_bs=max_prefill_bs,
                  max_running_requests=max_running_requests,
                  waiting_queue_len=waiting_queue_len)
    wrapper.__numinous_skew__ = True
    return wrapper


_logged = False


def install(scheduler_cls) -> list[str]:
    """Monkey-patch the prefill delayer and scheduler for balanced prefill budget.
    Called from the anchored hook in scheduler.py when NUMINOUS_PREFILL_BALANCE is set."""
    global _floor, _logged
    _floor = parse_floor(os.environ.get("NUMINOUS_PREFILL_BALANCE"))
    if _floor <= 0:
        return ["disabled"]
    missing = []

    # Patch PrefillDelayer
    try:
        from sglang.srt.managers.prefill_delayer import (
            PrefillDelayer,
            PrefillDelayerSinglePassExecutor,
        )
        if not _patch(PrefillDelayer, "__init__", _wrap_delayer_init):
            missing.append("PrefillDelayer.__init__")
        if not _patch(PrefillDelayer, "_gather_info", _wrap_gather_info):
            missing.append("PrefillDelayer._gather_info")
        if not _patch(PrefillDelayer, "_negotiate_should_allow_prefill", _wrap_negotiate):
            missing.append("PrefillDelayer._negotiate_should_allow_prefill")
        if not _patch(PrefillDelayerSinglePassExecutor, "negotiate_should_allow_prefill",
                      _wrap_negotiate_single_pass):
            missing.append("PrefillDelayerSinglePassExecutor.negotiate_should_allow_prefill")
    except ImportError as e:
        missing.append(f"prefill_delayer import: {e}")

    if missing:
        log.warning("numinous_pro_skew_budget: not patched: %s", ", ".join(missing))
    if not _logged:
        _logged = True
        log.info("numinous_pro_skew_budget: floor=%d, wrapped %s", _floor, ", ".join(_WRAPPED))
    return missing
'''

# ---------------------------------------------------------------------------
# HOOKS
# ---------------------------------------------------------------------------

SCHEDULER_HOOK = '''
        # numinous-hook:pro_skew_budget_chunk
        # Numinous balanced prefill budget: cap chunk to max(floor, max_other_extends)
        # (srt/managers/numinous_pro_skew_budget.py; operations/patches/pro_skew_budget_hooks.py).
        if self.require_mlp_sync:
            from sglang.srt.managers import numinous_pro_skew_budget as _numinous_psb

            # Estimate planned extend tokens so the delayer gather includes them
            if self.chunked_req is not None:
                _psb_est = len(self.chunked_req.full_untruncated_fill_ids) - len(self.chunked_req.prefix_indices)
            elif self.waiting_queue:
                _psb_est = len(self.waiting_queue[0].full_untruncated_fill_ids) - len(self.waiting_queue[0].prefix_indices)
            else:
                _psb_est = 0
            _numinous_psb.set_planned_tokens(min(_psb_est, chunked_prefill_size or _psb_est))
            # Apply the cap (computed by the previous pass's gather)
            chunked_prefill_size = _numinous_psb.scheduler_chunk_size(
                chunked_prefill_size,
                self.page_size,
                getattr(self, "truncation_align_size", None),
            )
'''

# Hook that injects planned_tokens into the negotiate call from PrefillAdder.add_chunked_req
ADDER_CHUNKED_HOOK = '''
            # numinous-hook:pro_skew_budget_chunked
            from sglang.srt.managers import numinous_pro_skew_budget as _numinous_psb_c
            # v0.5.20 computes cand_extend_input_len only after this block: derive it here
            _numinous_psb_c.set_planned_tokens(
                min(len(req.full_untruncated_fill_ids) - len(req.prefix_indices), _rem_tokens))
'''

# Hook that injects planned_tokens into the negotiate call from PrefillAdder.__init__
# We need to set planned_tokens before the first negotiate call in the PrefillAdder.__init__
# The best place is inside the PrefillAdder when it first calls negotiate, but that's in __init__
# We'll use a different approach: wrap the first add_one_req call.

SCHEDULER_INSTALL_HOOK = '''

# numinous-hook:pro_skew_budget_install
# Numinous balanced prefill budget under NUMINOUS_PREFILL_BALANCE=<floor>
if os.environ.get("NUMINOUS_PREFILL_BALANCE"):
    try:
        from sglang.srt.managers import numinous_pro_skew_budget as _numinous_psb_install

        _numinous_psb_install.install(Scheduler)
    except Exception as _e:  # noqa: BLE001
        logger.warning("numinous_pro_skew_budget not installed: %s", _e)
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
    # Install hook at the end of scheduler.py (like sched_trace)
    Hook("pro_skew_budget_install", "srt/managers/scheduler.py", "", SCHEDULER_INSTALL_HOOK),
    # Apply the cap right after the chunk size determination
    Hook("pro_skew_budget_chunk", "srt/managers/scheduler.py",
         "        # Determine chunked_prefill_size for this batch\n"
         "        chunked_prefill_size = self.chunked_prefill_size\n"
         "        if self.chunked_req is not None and self.dynamic_chunk_sizer is not None:\n"
         "            history_len = len(self.chunked_req.prefix_indices)\n"
         "            dynamic_size = self.dynamic_chunk_sizer.predict(history_len)\n"
         "            if dynamic_size is not None:\n"
         "                chunked_prefill_size = dynamic_size\n",
         SCHEDULER_HOOK),
    # Set planned_tokens before the negotiate call in add_chunked_req
    Hook("pro_skew_budget_chunked", "srt/managers/schedule_policy.py",
         "        # A mid-chunk rank prefills this pass regardless of the delayer\n"
         "        # verdict, so report prefillable=True and ignore the result.\n"
         "        if self.prefill_delayer_single_pass is not None:\n",
         ADDER_CHUNKED_HOOK),
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
    """One status line per hook plus the module copy; raises SystemExit if an anchor is missing."""
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
        raise SystemExit("pro_skew_budget hooks: " + "; ".join(missing))
    # Write the module
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
