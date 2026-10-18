"""Numinous speculation runtime for SGLang v0.5.20: load-adaptive DSpark verify rows (tracker 2.4) and DSpark
aux-hidden capture under two-batch overlap (tracker 2.7). Installed into the engine as
`sglang/srt/speculative/numinous_spec_policy.py`; entered through the anchored hooks in
`operations/patches/spec_hooks.py`. Pure Python + torch at import time (no sglang import), so the policy math is
CPU-testable.

Why this shape (source_code, SGLang v0.5.20, identical to the dev-box install):
- The official adaptive-speculation interface (#37274, `AdaptiveSpecPolicy` in speculative/adaptive_runtime_state.py)
  is wired only into the EAGLE/Standalone workers, and `adaptive_unsupported_reason` rejects DSPARK and DP attention.
  It is not a registration path for Pro's DSpark.
- DSpark changes its verify cost at runtime through the compact ragged-verify scheduler
  (`SGLANG_RAGGED_VERIFY_MODE=compact` + `--speculative-dspark-sps-table-path`): each step it picks a verify-token
  budget maximising (requests + sum of confidence survival) x steps/s(tokens). The draft width gamma stays fixed at
  launch (changing it needs one draft graph per width). So "block size by load" is a per-step cap on verify rows
  with gamma launched at the widest useful block. Pro 0813 ships the confidence head (`mtp.2.confidence_head`) and
  `dspark_block_size` 5 (checkpoint config.json).
- SGLang's `SpsCostTable.lookup` is a floor lookup over the profiled probes, so step time is a step function
  (tokens 97..143 cost the same as 96 on the 7-probe Pro tables) and flat past the last probe (192 tokens, which a
  gamma-7 window passes at 24 requests per rank). `StepCostTable.to_additive_json` emits SGLang's own
  `SpsAdditiveCostTable` format (piecewise-linear interpolation), extended linearly past the last probe.

What the policy does: rows*(n) = argmax_r n * E[tokens | r rows] / t(n * r) over r in 1..gamma+1, where n is the
running requests on this DP rank, E comes from an acceptance model and t from the step-cost table. The hook caps the
scheduler's verify-token budget (tokens above one row per request) at n * (rows*(n) - 1). Per-row lengths stay the
confidence scheduler's choice; the cap bounds the step's total verify tokens, which is what sets step time.

Defaults (derived): B200 table `/cache/sps/deepseek-v4-pro_reg.json` (measured 2026-09-27, static verify, W=6,
dp8/tp8) and accept length 4.27 at gamma 5 (measured, real 46k-token agent sessions, B200 and B300 gate runs
2026-09-27), fitted as a geometric per-position acceptance p = 0.862. Positions past the checkpoint's trained block
(5) are assumed to keep p (tail_decay = 1.0) until measured; with that assumption a gamma-7 launch verifies 8 rows up
to 20 requests per rank, 7 at 21-29, 6 at 30-43 and 5 from 44 (derived, see test_numinous_spec_policy.py).

Env: NUMINOUS_DSPARK_ADAPTIVE=1 turns the cap on (inert unless the compact scheduler runs); NUMINOUS_DSPARK_POLICY
points at a JSON built by `python3 numinous_spec_policy.py build ...` (else the embedded defaults);
NUMINOUS_TBO_DSPARK=1 lets DeepSeek V4 prefill run two-batch overlap while DSpark captures aux hidden states.
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# measured: /cache/sps/deepseek-v4-pro_reg.json (8xB200, dp8 tp8, static verify W=6, 2026-09-27T07:41Z)
PRO_B200_SPS = {"sample_batch_tokens": [6, 12, 24, 48, 96, 144, 192],
                "sample_steps_per_sec": [38.962734092976774, 38.483722107118474, 32.67249135506299,
                                         26.452778318773834, 19.51902201892076, 17.110012030477208,
                                         14.564283515989171],
                "max_batch_tokens": 192}
# measured: /cache/sps/deepseek-v4-pro_B300.json (8xB300, same settings)
PRO_B300_SPS = {"sample_batch_tokens": [6, 12, 24, 48, 96, 144, 192],
                "sample_steps_per_sec": [38.303077891702834, 37.69925232957817, 33.4909641378756,
                                         27.207700867653582, 20.15823458213177, 17.47892124030862,
                                         14.382716720798122],
                "max_batch_tokens": 192}
# measured: mean accept length (tokens per verify step, bonus included) at gamma 5 on real agent traffic
PRO_ACCEPT_LENGTH = 4.27
PRO_ACCEPT_GAMMA = 5
# derived: Pro serves at most 512 running requests over 8 DP ranks
PRO_MAX_PER_RANK = 64


# ----------------------------------------------------------------------------------------------------------------
# step cost
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StepCostTable:
    """Step time (seconds) by verify tokens on one DP rank; piecewise linear, linear extrapolation past the ends."""

    tokens: tuple[int, ...]
    seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < 2 or len(self.tokens) != len(self.seconds):
            raise ValueError("StepCostTable needs >= 2 probes with one time each")
        if list(self.tokens) != sorted(set(self.tokens)) or self.tokens[0] < 1:
            raise ValueError(f"tokens must be strictly increasing and >= 1, got {self.tokens}")
        if min(self.seconds) <= 0:
            raise ValueError("step times must be positive")

    @classmethod
    def from_sps(cls, sps: dict | str) -> "StepCostTable":
        """From SGLang's SpsCostTable JSON (dict or path): steps/s per probe -> seconds per step."""
        if isinstance(sps, str):
            with open(sps) as f:
                sps = json.load(f)
        if "bias_seconds" in sps:
            raise ValueError("already an additive table; build from the profiled SpsCostTable")
        return cls(tuple(int(x) for x in sps["sample_batch_tokens"]),
                   tuple(1.0 / float(s) for s in sps["sample_steps_per_sec"]))

    @classmethod
    def from_records(cls, lines: Iterable[str | dict]) -> "StepCostTable":
        """From the SPS profiler's per-step records (`*.records.jsonl`): median step_time per verify-token count."""
        by: dict[int, list[float]] = {}
        for line in lines:
            r = json.loads(line) if isinstance(line, str) else line
            if not r or "num_verify_tokens" not in r:
                continue
            by.setdefault(int(r["num_verify_tokens"]), []).append(float(r["step_time"]))
        keys = sorted(k for k in by if k >= 1)
        return cls(tuple(keys), tuple(statistics.median(by[k]) for k in keys))

    def seconds_at(self, tokens: float) -> float:
        xs, ys = self.tokens, self.seconds
        if tokens <= xs[0]:
            return ys[0]  # below the smallest probe the step is weight-read bound: never cheaper
        if tokens >= xs[-1]:
            lo, hi = len(xs) - 2, len(xs) - 1
        else:
            hi = bisect.bisect_right(xs, tokens)
            lo = hi - 1
        slope = (ys[hi] - ys[lo]) / (xs[hi] - xs[lo])
        return max(ys[lo] + slope * (tokens - xs[lo]), ys[0])

    def extended(self, max_tokens: int) -> "StepCostTable":
        """Adds a probe at max_tokens (linear continuation of the last segment); assumed, not measured."""
        if max_tokens <= self.tokens[-1]:
            return self
        return StepCostTable(self.tokens + (int(max_tokens),), self.seconds + (self.seconds_at(max_tokens),))

    def to_sps_json(self) -> dict:
        """SGLang SpsCostTable (floor lookup) at the probes; what the stock profiler writes."""
        return {"sample_batch_tokens": list(self.tokens), "sample_steps_per_sec": [1.0 / s for s in self.seconds],
                "max_batch_tokens": self.tokens[-1]}

    def to_additive_json(self) -> dict:
        """SGLang SpsAdditiveCostTable: step = bias + alpha(num_reqs) + theta(num_reqs + budget), both linearly
        interpolated. A static-verify profile ties requests to tokens (tokens = W x requests), so the per-request
        term is not identifiable: alpha = 0 and theta carries the whole curve (theta(m) = t(m) - t(first probe))."""
        bias = self.seconds[0]
        return {"bias_seconds": bias, "bs_probes": [1], "alpha_seconds": [0.0], "m_probes": list(self.tokens),
                "theta_seconds": [max(0.0, s - bias) for s in self.seconds]}


def additive_step_seconds(table: dict, num_reqs: int, budget: int) -> float:
    """SGLang's SpsAdditiveCostTable.step_time, re-implemented for CPU checks of emitted tables."""

    def interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        hi = bisect.bisect_right(xs, x)
        lo = hi - 1
        return ys[lo] + (x - xs[lo]) / (xs[hi] - xs[lo]) * (ys[hi] - ys[lo])

    return (table["bias_seconds"] + interp(table["bs_probes"], table["alpha_seconds"], num_reqs)
            + interp(table["m_probes"], table["theta_seconds"], num_reqs + budget))


# ----------------------------------------------------------------------------------------------------------------
# acceptance
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AcceptanceModel:
    """Conditional acceptance per draft position: q[i] = P(draft i+1 accepted | drafts 1..i accepted)."""

    q: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.q or any(not 0.0 <= x <= 1.0 for x in self.q):
            raise ValueError(f"conditional acceptances must be in [0, 1], got {self.q}")

    @classmethod
    def geometric(cls, p: float, positions: int, tail_from: int = 0, tail_decay: float = 1.0) -> "AcceptanceModel":
        """p per position; positions past `tail_from` (if > 0) multiply by tail_decay once per extra position."""
        q = []
        for i in range(positions):
            extra = max(0, i + 1 - tail_from) if tail_from else 0
            q.append(min(1.0, p * tail_decay ** extra))
        return cls(tuple(q))

    @classmethod
    def from_accept_length(cls, accept_length: float, gamma: int, positions: int,
                           tail_decay: float = 1.0) -> "AcceptanceModel":
        """Fits the geometric p for which a gamma-draft window yields accept_length tokens per step (bonus
        included, SGLang's accept-length metric), then extends to `positions` drafts."""
        if not 1.0 <= accept_length <= gamma + 1:
            raise ValueError(f"accept length {accept_length} outside [1, {gamma + 1}]")
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if _geom_tokens(mid, gamma + 1) < accept_length:
                lo = mid
            else:
                hi = mid
        return cls.geometric((lo + hi) / 2, positions, tail_from=gamma, tail_decay=tail_decay)

    def expected_tokens(self, rows: int) -> float:
        """Tokens emitted per verify step when a request verifies `rows` rows (anchor + rows - 1 drafts)."""
        if rows < 1:
            raise ValueError("rows >= 1")
        total, surv = 1.0, 1.0
        for i in range(min(rows - 1, len(self.q))):
            surv *= self.q[i]
            total += surv
        return total


def _geom_tokens(p: float, rows: int) -> float:
    return float(rows) if p >= 1.0 else (1.0 - p ** rows) / (1.0 - p)


# ----------------------------------------------------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class DSparkLoadPolicy:
    """rows_by_load[n - 1] = verify rows per request that maximise this rank's tokens/s at n running requests."""

    gamma: int
    rows_by_load: list[int]
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gamma < 1 or not self.rows_by_load:
            raise ValueError("gamma >= 1 and a non-empty table")
        if any(not 1 <= r <= self.gamma + 1 for r in self.rows_by_load):
            raise ValueError(f"rows must be in [1, gamma + 1 = {self.gamma + 1}]")

    @classmethod
    def build(cls, cost: StepCostTable, accept: AcceptanceModel, gamma: int, max_per_rank: int,
              source: Optional[dict] = None) -> "DSparkLoadPolicy":
        cost = cost.extended(max_per_rank * (gamma + 1))
        rows, cur = [], gamma + 1
        for n in range(1, max_per_rank + 1):
            # a heavier load never verifies more rows than a lighter one: guards against kinks of a piecewise
            # profile (e.g. tail_decay 0.8 would otherwise flip 6 -> 7 -> 6 rows between 13 and 23 requests)
            cur = min(cur, best_rows(n, cost, accept, gamma))
            rows.append(cur)
        return cls(gamma, rows, dict(source or {}))

    def rows(self, num_requests: int) -> int:
        if num_requests <= 0:
            return self.gamma + 1
        return self.rows_by_load[min(num_requests, len(self.rows_by_load)) - 1]

    def budget_cap(self, num_requests: int) -> int:
        """Verify tokens above one row per request that this load may spend."""
        return max(0, num_requests) * (self.rows(num_requests) - 1)

    def cap(self, num_requests: int, budget: int) -> int:
        return max(0, min(int(budget), self.budget_cap(num_requests)))

    def transitions(self) -> list[tuple[int, int]]:
        """(first load, rows) at every change of the table: the human-readable policy."""
        out, prev = [], None
        for n, r in enumerate(self.rows_by_load, start=1):
            if r != prev:
                out.append((n, r))
                prev = r
        return out

    def to_json(self) -> dict:
        return {"kind": "numinous-dspark-load-policy", "version": 1, "gamma": self.gamma,
                "rows_by_load": self.rows_by_load, "transitions": self.transitions(), "source": self.source}

    @classmethod
    def from_json(cls, d: dict | str) -> "DSparkLoadPolicy":
        if isinstance(d, str):
            with open(d) as f:
                d = json.load(f)
        if d.get("kind") != "numinous-dspark-load-policy":
            raise ValueError("not a numinous DSpark load policy")
        return cls(int(d["gamma"]), [int(x) for x in d["rows_by_load"]], dict(d.get("source") or {}))


def rank_tokens_per_second(n: int, rows: int, cost: StepCostTable, accept: AcceptanceModel) -> float:
    return n * accept.expected_tokens(rows) / cost.seconds_at(n * rows)


def best_rows(n: int, cost: StepCostTable, accept: AcceptanceModel, gamma: int) -> int:
    """Ties go to fewer rows (same speed, less verify compute)."""
    best, best_v = 1, -1.0
    for r in range(1, gamma + 2):
        v = rank_tokens_per_second(n, r, cost, accept)
        if v > best_v * (1 + 1e-9):
            best, best_v = r, v
    return best


def recommend_gamma(loads: Sequence[int], cost: StepCostTable, accept: AcceptanceModel,
                    candidates: Sequence[int] = (5, 6, 7)) -> dict:
    """Launch gamma for a lane whose running requests per rank are `loads` (equal weight): the adaptive cap lets a
    wide gamma serve each load at its own best rows, so a candidate's value is the mean best throughput it allows."""
    cost = cost.extended(max(loads) * (max(candidates) + 1))
    scores = {}
    for g in candidates:
        scores[g] = statistics.fmean(rank_tokens_per_second(n, best_rows(n, cost, accept, g), cost, accept)
                                     for n in loads)
    best = max(scores, key=lambda g: (round(scores[g], 6), -g))
    return {"gamma": best, "mean_rank_tokens_per_s": scores}


def default_policy(gamma: int, hardware: str = "b200") -> DSparkLoadPolicy:
    """Embedded measured defaults (module docstring); the served image should pass NUMINOUS_DSPARK_POLICY."""
    sps = PRO_B300_SPS if hardware.lower() == "b300" else PRO_B200_SPS
    accept = AcceptanceModel.from_accept_length(PRO_ACCEPT_LENGTH, PRO_ACCEPT_GAMMA, positions=max(gamma, 1))
    return DSparkLoadPolicy.build(StepCostTable.from_sps(sps), accept, gamma, PRO_MAX_PER_RANK,
                                  source={"sps": f"embedded {hardware} table", "accept_length": PRO_ACCEPT_LENGTH,
                                          "accept_gamma": PRO_ACCEPT_GAMMA, "tail_decay": 1.0})


# ----------------------------------------------------------------------------------------------------------------
# packaged tables (written into images by `spec_hooks.py --write-tables DIR`)
# ----------------------------------------------------------------------------------------------------------------
# GLM-5.3 EAGLE depth by load for TP (non-DP-attention) lanes, in SGLang's own --speculative-adaptive-config format
# (keys: decode batch size, padded to the CUDA-graph size; each slot adapts by EMA acceptance within its candidates).
# Derived from measured crossovers: 5/1/6 beats 3/1/4 up to ~16 decode requests per attention group and loses from
# ~24 (DP8 lane, 128 vs 192 users); the Fast lane measured 4/1/5 best at its loads. SGLang rejects adaptive under
# DP attention (adaptive_unsupported_reason), so DP lanes pick a static depth per lane load instead.
GLM_TP_ADAPTIVE_CONFIG = {
    "1": {"candidate_steps": [3, 4, 5, 6]},
    "16": {"candidate_steps": [3, 4, 5]},
    "24": {"candidate_steps": [3, 4]},
    "48": {"candidate_steps": [2, 3]},
}


def packaged_tables() -> dict[str, dict]:
    """File name -> JSON body for every table a recipe references (spec_recipes.TABLE_DIR)."""
    out: dict[str, dict] = {}
    for hw, sps in (("b200", PRO_B200_SPS), ("b300", PRO_B300_SPS)):
        cost = StepCostTable.from_sps(sps)
        for g in (5, 6, 7):
            out[f"pro_{hw}_sps_additive_g{g}.json"] = cost.extended(PRO_MAX_PER_RANK * (g + 1)).to_additive_json()
            out[f"pro_{hw}_policy_g{g}.json"] = default_policy(g, hw).to_json()
    out["glm_tp_eagle_adaptive.json"] = GLM_TP_ADAPTIVE_CONFIG
    return out


def write_tables(directory: str) -> list[str]:
    os.makedirs(directory, exist_ok=True)
    paths = []
    for name, body in packaged_tables().items():
        path = os.path.join(directory, name)
        with open(path, "w") as f:
            json.dump(body, f, indent=1)
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------------------------------------------
# engine glue (called from the hooks; every function tolerates being called each step)
# ----------------------------------------------------------------------------------------------------------------
_POLICY: dict[int, DSparkLoadPolicy] = {}
_LOGGED: set[str] = set()


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def adaptive_enabled() -> bool:
    return _env_on("NUMINOUS_DSPARK_ADAPTIVE")


def tbo_dspark_enabled() -> bool:
    return _env_on("NUMINOUS_TBO_DSPARK")


def _log_once(key: str, msg: str, *args: Any, level: int = logging.INFO) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.log(level, msg, *args)


def runtime_policy(gamma: int) -> DSparkLoadPolicy:
    pol = _POLICY.get(gamma)
    if pol is not None:
        return pol
    path = os.environ.get("NUMINOUS_DSPARK_POLICY", "")
    if path:
        pol = DSparkLoadPolicy.from_json(path)
        if pol.gamma != gamma:
            raise ValueError(f"NUMINOUS_DSPARK_POLICY {path} was built for gamma {pol.gamma}, the server runs "
                             f"gamma {gamma}; rebuild it (numinous_spec_policy.py build --gamma {gamma})")
    else:
        pol = default_policy(gamma, os.environ.get("NUMINOUS_DSPARK_POLICY_HW", "b200"))
    _POLICY[gamma] = pol
    _log_once(f"policy{gamma}", "numinous DSpark load policy (gamma %d, %s): rows by running requests per rank %s",
              gamma, path or "embedded defaults", pol.transitions())
    return pol


def on_planner_init(*, mode: str, gamma: int, tp_rank: int) -> None:
    """Startup check that the flag takes effect: the cap lives in the compact scheduler only."""
    if not adaptive_enabled():
        return
    if mode == "static":
        _log_once("inert", "NUMINOUS_DSPARK_ADAPTIVE=1 but SGLANG_RAGGED_VERIFY_MODE=static: every step verifies "
                  "all gamma + 1 rows and the load policy is inert. Set SGLANG_RAGGED_VERIFY_MODE=compact and "
                  "--speculative-dspark-sps-table-path.", level=logging.WARNING)
        return
    pol = runtime_policy(gamma)
    if tp_rank == 0:
        _log_once("active", "numinous DSpark load policy active (mode %s, gamma %d, transitions %s)", mode, gamma,
                  pol.transitions())


def cap_decision(decision: Any, *, num_requests: int, gamma: int) -> Any:
    """Caps a VerifyBudgetDecision's budget in place (msgspec Struct, mutable) and returns it."""
    capped = runtime_policy(gamma).cap(num_requests, decision.budget)
    if capped != decision.budget:
        decision.budget = capped
    return decision


# ----------------------------------------------------------------------------------------------------------------
# two-batch overlap with DSpark aux capture (DeepSeek V4)
# ----------------------------------------------------------------------------------------------------------------
class TboDSparkCapture:
    """A TBO op appended after each captured layer: records that layer's completed hidden state (hc-mean, the
    value the non-TBO loop captures) per ubatch; `merge` reassembles each layer's rows in parent token order."""

    def __init__(self, layer_ids: Sequence[int]):
        self.layer_ids = list(layer_ids)
        self.store: dict[int, list] = {0: [], 1: []}

    def op_numinous_dspark_capture(self, state, **kw):  # noqa: ARG002 (the operations engine passes state)
        self.store[kw["tbo_subbatch_index"]].append(kw["hidden_states"].mean(dim=1))
        return kw

    def merge(self, outputs_arr: Sequence[dict], original_len: int) -> list:
        import torch

        a, b = self.store[0], self.store[1]
        if len(a) != len(self.layer_ids) or len(b) != len(self.layer_ids):
            raise RuntimeError(f"TBO DSpark capture: expected {len(self.layer_ids)} layers per ubatch, got "
                               f"{len(a)} / {len(b)}")
        s0, t0 = outputs_arr[0]["forward_batch"].tbo_parent_token_range
        s1, t1 = outputs_arr[1]["forward_batch"].tbo_parent_token_range
        out = []
        for xa, xb in zip(a, b):
            res = torch.zeros((original_len, *xa.shape[1:]), dtype=xa.dtype, device=xa.device)
            res[s0:t0] = xa[: t0 - s0]
            res[s1:t1] = xb[: t1 - s1]
            out.append(res)
        self.store = {0: [], 1: []}
        return out


def tbo_strategy_with_dspark_capture(layers: Sequence[Any], start_layer: int, capture_layer_ids: Sequence[int],
                                     forward_mode: Any, strategy_cls: Any = None):
    """The upstream per-layer TBO strategy with a capture op after every layer DSpark reads.
    Returns (OperationsStrategy, TboDSparkCapture)."""
    if strategy_cls is None:
        from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy as strategy_cls
    wanted = set(int(i) for i in capture_layer_ids)
    ids = [start_layer + k for k in range(len(layers)) if start_layer + k in wanted]
    cap = TboDSparkCapture(ids)
    items = []
    for k, layer in enumerate(layers):
        s = strategy_cls.init_new_tbo([layer], forward_mode)
        if start_layer + k in wanted:
            s = strategy_cls(operations=list(s.operations) + [cap.op_numinous_dspark_capture],
                             deep_gemm_num_sms=s.deep_gemm_num_sms, tbo_delta_stages=s.tbo_delta_stages)
        items.append(s)
    if tbo_dspark_enabled():
        _log_once("tbo", "numinous: DeepSeek V4 prefill two-batch overlap with DSpark aux capture (layers %s)", ids)
    return strategy_cls.concat(items), cap


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
def _cost_from_args(a) -> StepCostTable:
    if a.records:
        with open(a.records) as f:
            return StepCostTable.from_records(f)
    if a.sps:
        return StepCostTable.from_sps(a.sps)
    return StepCostTable.from_sps(PRO_B300_SPS if a.hw == "b300" else PRO_B200_SPS)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "show"):
        p = sub.add_parser(name)
        p.add_argument("--sps", default="", help="SGLang SpsCostTable JSON (profiled)")
        p.add_argument("--records", default="", help="SPS profiler *.records.jsonl (overrides --sps)")
        p.add_argument("--hw", default="b200", choices=["b200", "b300"], help="embedded table when no file")
        p.add_argument("--accept-length", type=float, default=PRO_ACCEPT_LENGTH)
        p.add_argument("--accept-gamma", type=int, default=PRO_ACCEPT_GAMMA)
        p.add_argument("--tail-decay", type=float, default=1.0)
        p.add_argument("--gamma", type=int, default=7)
        p.add_argument("--max-per-rank", type=int, default=PRO_MAX_PER_RANK)
        if name == "build":
            p.add_argument("--out", required=True, help="policy JSON (NUMINOUS_DSPARK_POLICY)")
            p.add_argument("--additive-out", default="", help="SpsAdditiveCostTable JSON for "
                           "--speculative-dspark-sps-table-path")
    a = ap.parse_args(argv)
    cost = _cost_from_args(a)
    accept = AcceptanceModel.from_accept_length(a.accept_length, a.accept_gamma, positions=a.gamma,
                                                tail_decay=a.tail_decay)
    src = {"sps": a.records or a.sps or f"embedded {a.hw}", "accept_length": a.accept_length,
           "accept_gamma": a.accept_gamma, "tail_decay": a.tail_decay, "p": round(accept.q[0], 4)}
    pol = DSparkLoadPolicy.build(cost, accept, a.gamma, a.max_per_rank, source=src)
    ext = cost.extended(a.max_per_rank * (a.gamma + 1))
    rows = []
    for n in (1, 2, 4, 8, 16, 24, 32, 48, 64):
        if n > a.max_per_rank:
            break
        r = pol.rows(n)
        rows.append({"per_rank": n, "rows": r, "tok_s_rank": round(rank_tokens_per_second(n, r, ext, accept), 1),
                     "tok_s_rank_full": round(rank_tokens_per_second(n, a.gamma + 1, ext, accept), 1)})
    report = {"policy": pol.to_json(), "table": rows,
              "gamma_choice": recommend_gamma([n["per_rank"] for n in rows], cost, accept)}
    if a.cmd == "build":
        with open(a.out, "w") as f:
            json.dump(pol.to_json(), f, indent=1)
        if a.additive_out:
            with open(a.additive_out, "w") as f:
                json.dump(ext.to_additive_json(), f, indent=1)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
