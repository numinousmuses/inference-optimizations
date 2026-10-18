"""DP-attention prefill pacing + peer release for a monolithic EAGLE + DP-attention server (SGLang v0.5.20), amdpro
2026-10-05, coordinator 17:4x: GLM arm M's decode-holding ranks sit idle behind other ranks' prefill passes.

source_code (v0.5.20 and the hopper decode-burst patch):
  - Under spec + DP attention any rank's new prefill batch turns every other rank into an IDLE batch for that pass
    (scheduler.py:3658-3670; dp_attn.py:489 need_idle_batch = max(global_num_tokens) > 0).
  - A chunked continuation is added before the delayer can say no and ignores its verdict (schedule_policy.py:1154-1163
    "A mid-chunk rank prefills this pass regardless of the delayer verdict").
  - The decode-burst rule never holds one either (decode_burst.py hold_prefill: chunked_prefill_pending -> False).
  - So a long prompt's chunks run back to back while the other ranks' decodes wait.
  - The delayer's "mixed" branch (prefill_delayer.py:323-350) meanwhile holds other ranks' queued requests for up to
    max_delay_passes, although the pass runs a forward anyway.
measured (M5-spec314 sched traces, 8 ranks, 2,903 s of forward wall; operations/hopper/m_decode_idle_readout.py ->
research/gpu-evidence/amdpro/m5spec314_decode_idle.json):
  - decode-holding ranks idle in 29.9% of rank-time; 92% of consecutive prefill passes have 0 decode steps between;
  - 662 lone-continuation passes (<= 2 ranks, empty queues) = 251 s with 6.3 decode-holding ranks idle per pass;
  - 1,826 continuation passes (879 s) held 2.7 ranks with queued requests per pass.

What the hooks change (NUMINOUS_DP_PREFILL_PACE=1; unset = stock):
  mark     scheduler.get_new_batch_prefill hands its delayer three local facts for this pass: continuation pending,
           running decodes, waiting requests.
  gather   PrefillDelayer._gather_info runs one more lockstep 3-int all-gather of those facts. The stock 5-column and
           the decode-burst 6-column info rows stay untouched.
  decide   _negotiate_should_allow_prefill applies numinous_dp_pace.decide (installed module):
             park     -> verdict delay; reason numinous_pace; the continuation parks;
             release  -> verdict allow; reason numinous_peer_forward.
           Every rank sees the same rows and streak, so all agree.
  park     PrefillAdder.add_chunked_req negotiates first, then returns the request unadded when the delayer parked
           this pass (the SWA park path's contract: chunked_req stays set and runs on a later pass).
  assert   the delayer's debug-log reason set accepts the two new reasons.
Knobs: NUMINOUS_DP_PACE_STEPS (default 2; 0 = release only), NUMINOUS_DP_PACE_K (2), NUMINOUS_DP_PACE_QMAX (0),
NUMINOUS_DP_PEER_RELEASE (1).

    python3 m_dp_pace_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ENV = "NUMINOUS_DP_PREFILL_PACE"
SCHED = "srt/managers/scheduler.py"
DELAYER = "srt/managers/prefill_delayer.py"
POLICY = "srt/managers/schedule_policy.py"
MODULE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numinous_dp_pace.py")
MODULE_DST = "srt/managers/numinous_dp_pace.py"

A_MARK = '''            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )
'''
T_MARK = '''            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )
            # numinous-hook:m_dp_pace_mark (amdpro): this rank's facts for the pacing gather
            self.prefill_delayer._numinous_local = (
                int(self.chunked_req is not None),
                int(running_batch.batch_size()) if running_batch is not None else 0,
                len(self.waiting_queue),
            )
'''
A_GATHER = '''        tp0_info = self._global_info_buffer[:, 0, :]
        return tp0_info
'''
T_GATHER = '''        tp0_info = self._global_info_buffer[:, 0, :]
        # numinous-hook:m_dp_pace_gather (amdpro): one more lockstep 3-int all-gather (continuation, running, waiting)
        if __import__("os").environ.get("NUMINOUS_DP_PREFILL_PACE", "0") == "1":
            _s = self._global_info_buffer.shape
            _b = getattr(self, "_numinous_pace_buf", None)
            if _b is None:
                _b = self._numinous_pace_buf = torch.empty(
                    (_s[0] * _s[1], 3), dtype=torch.int64, device=self._gather_device
                )
            torch.distributed.all_gather_into_tensor(
                _b.flatten(),
                torch.tensor(
                    list(getattr(self, "_numinous_local", (0, 0, 0))),
                    dtype=torch.int64,
                    device=self._gather_device,
                ),
                group=self._gather_group,
            )
            _rows = _b.view(_s[0], _s[1], 3)[:, 0, :].tolist()
            self._numinous_rows = (tp0_info[:, 0].tolist(), _rows)
        return tp0_info
'''
A_NEG = '''        self._curr_state = out.next_state
        return out
'''
T_NEG = '''        # numinous-hook:m_dp_pace_decide (amdpro): park lone continuations so decode runs; release held work into a
        # pass that runs a forward anyway (operations/patches/numinous_dp_pace.py)
        self._numinous_park = False
        if __import__("os").environ.get("NUMINOUS_DP_PREFILL_PACE", "0") == "1" and getattr(
            self, "_numinous_rows", None
        ):
            from sglang.srt.managers import numinous_dp_pace as _ndp

            _pre, _rows = self._numinous_rows
            self._numinous_rows = None
            _action, self._numinous_pace_streak = _ndp.decide(
                _pre,
                [r[0] for r in _rows],
                [r[1] for r in _rows],
                [r[2] for r in _rows],
                getattr(self, "_numinous_pace_streak", 0),
                **_ndp.params(),
            )
            if _action == "park":
                self._numinous_park = True
                out = out._replace(output_allow=False, output_reason="numinous_pace")
                self._numinous_parks = getattr(self, "_numinous_parks", 0) + 1
            elif _action == "release" and not out.output_allow:
                _prev = self._curr_state
                out = out._replace(
                    next_state=None,
                    output_allow=True,
                    output_reason="numinous_peer_forward",
                    wait_forward_passes=_prev.delayed_count if _prev else 0,
                    wait_seconds=(time.perf_counter() - _prev.start_time) if _prev else 0.0,
                )
                self._numinous_releases = getattr(self, "_numinous_releases", 0) + 1
        self._curr_state = out.next_state
        return out
'''
A_ASSERT = '''                "delay",
            }
'''
T_ASSERT = '''                "delay",
                "numinous_pace",
                "numinous_peer_forward",
            }
'''
A_PARK = '''        # A mid-chunk rank prefills this pass regardless of the delayer
        # verdict, so report prefillable=True and ignore the result.
'''
T_PARK = '''        # numinous-hook:m_dp_pace_park (amdpro): negotiate first; when the delayer parked this pass (every rank agrees),
        # the continuation is not added and stays chunked_req for a later pass (the SWA park path's contract above)
        if self.prefill_delayer_single_pass is not None and __import__("os").environ.get(
            "NUMINOUS_DP_PREFILL_PACE", "0"
        ) == "1":
            self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
                **(
                    {"decode_streak": self.decode_streak, "decode_burst": self.decode_burst}
                    if hasattr(self, "decode_burst")
                    else {}
                ),
            )
            if getattr(self.prefill_delayer_single_pass._prefill_delayer, "_numinous_park", False):
                return req
        # A mid-chunk rank prefills this pass regardless of the delayer
        # verdict, so report prefillable=True and ignore the result.
'''
HOOKS = [
    ("m_dp_pace_mark", SCHED, A_MARK, T_MARK),
    ("m_dp_pace_gather", DELAYER, A_GATHER, T_GATHER),
    ("m_dp_pace_decide", DELAYER, A_NEG, T_NEG),
    ("m_dp_pace_assert", DELAYER, A_ASSERT, T_ASSERT),
    ("m_dp_pace_park", POLICY, A_PARK, T_PARK),
]
CONFLICTS = ("operations/patches/pd_delayer_peer_hooks.py",)   # same anchors; this hook subsumes its release rule


def numinous_preflight(ctx) -> list:
    """CPU preflight (operations/hopper/hook_preflight.py): where NUMINOUS_DP_PREFILL_PACE=1 is inert or misplaced."""
    env = ctx.env
    if env.get(ENV, "0") != "1":
        return []
    a = ctx.args
    out = []
    if not a.get("enable_prefill_delayer"):
        out.append(f"{ENV}=1 without --enable-prefill-delayer: inert (the decision rides the delayer's negotiation)")
    if not a.get("enable_dp_attention") or int(a.get("dp_size") or 1) < 2:
        out.append(f"{ENV}=1 without DP attention (dp >= 2): nothing couples the ranks, inert")
    if (a.get("disaggregation_mode") or "null") == "decode":
        out.append(f"{ENV}=1 on a PD decode server: it never prefills, inert")
    if not a.get("speculative_algorithm"):
        out.append(f"{ENV}=1 without speculative decoding: v0.5.20 mixes prefill and decode across ranks there "
                   "(scheduler.py:3658-3670 only splits them under spec); the park only delays prompts")
    return out


def _root(arg: str) -> str:
    if arg:
        return arg
    import sglang

    return os.path.dirname(sglang.__file__)


def apply(root: str, check: bool = False) -> list[str]:
    out = []
    pending: dict[str, str] = {}
    for name, rel, anchor, text in HOOKS:
        path = os.path.join(root, rel)
        src = pending.get(path) or open(path).read()
        if text in src:
            out.append(f"{name}: already applied")
            continue
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"{name}: anchor found {n} times in {rel}")
        if check:
            out.append(f"{name}: anchor ok")
            continue
        pending[path] = src.replace(anchor, text, 1)
        out.append(f"{name}: applied to {rel}")
    for path, new in pending.items():
        compile(new, path, "exec")          # a patched file must compile before it replaces the original
        with open(path, "w") as f:
            f.write(new)
    dst = os.path.join(root, MODULE_DST)
    if not check:
        shutil.copyfile(MODULE_SRC, dst)
        out.append(f"module: installed {MODULE_DST}")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(_root(a.root), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
