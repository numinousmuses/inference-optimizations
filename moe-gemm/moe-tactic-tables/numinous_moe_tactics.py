"""Measured tactics for FlashInfer's TRT-LLM MXFP4 MoE kernel (DeepSeek V4 Pro routed experts).

With autotune off (the served recipe), FlashInfer resolves every call through resolveMoeTileAndConfig's fallback,
which returns the smallest tile of the candidate set around its own heuristic (std::set::begin): at decode, 12 tokens
per expert run in 8-token tiles and every expert weight tile is read twice. Its autotuner does not fix this either:
it buckets token counts to powers of two, so 768 tokens take the tactic tuned for 512.

This module serves the tactic measured fastest at each token count (operations/modal/moe_tactic_sweep.py: every
valid [tile_N, config] timed per count, per GPU). For a call with n tokens it uses the row with the smallest measured
count >= n (a tile sized for a few more tokens covers fewer without re-reading weights; a smaller one would not),
checks the tactic is valid for n, and otherwise leaves the choice to FlashInfer. Only this one op is touched, and
only when the runner's shape is the table's.

Tables (operations/kernels/moe_tactics/{stem}_{gpu}.json, gpu b200 | b300; graph-replay timed, FlashInfer 0.6.18):
    pro_mxfp4 (TP8), pro_mxfp4_ep8, pro_mxfp4_fs (TP8 with the shared expert fused as expert 384, top-k 7),
    glm_nvfp4 (TP8), glm_nvfp4_ep8, glm_nvfp4_tp4, glm_nvfp4_ep4 (one 4-GPU replica).
A table serves a runner only when its per-GPU shape is the runner's (shape_matches). Fused shared experts count in
the effective expert and top-k counts either way they reach FlashInfer: SGLang v0.5.20 appends the shared expert to
the routed weights (num_local_experts 385, top_k 7, num_fused_shared_experts 0); FlashInfer's own fusion keeps them
raw (384, 6) and records num_fused_shared_experts 1. Both enumerate the same tactic space (MoERunner.get_valid_tactics
keys on top_k + nfse and num_local_experts + nfse), so the _fs table covers both and never a plain 384/6 runner.
"""
from __future__ import annotations

import bisect
import json
import logging
import os
import pathlib
from typing import Optional

logger = logging.getLogger(__name__)

OP = "flashinfer::trtllm_fp4_block_scale_moe"
_TABLES: dict[str, dict] = {}
_INSTALLED = False
_SERVED: set = set()


def _table_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("NUMINOUS_MOE_TACTICS_DIR", pathlib.Path(__file__).with_name("moe_tactics")))


def gpu_key(capability: tuple[int, int]) -> str:
    """B200 is sm100, B300 is sm103."""
    return {(10, 0): "b200", (10, 3): "b300"}.get(tuple(capability), "")


def load_table(key: str, name: str = "pro_mxfp4") -> Optional[dict]:
    """One table: operations/kernels/moe_tactics/{name}_{gpu}.json (name: the file stem, e.g. pro_mxfp4 = TP8,
    pro_mxfp4_ep8, pro_mxfp4_fs, glm_nvfp4_tp4)."""
    cache_key = f"{name}_{key}"
    if cache_key in _TABLES:
        return _TABLES[cache_key]
    f = _table_dir() / f"{name}_{key}.json"
    if not key or not f.exists():
        return None
    t = json.loads(f.read_text())
    t["_tokens"] = [row["tokens"] for row in t["table"]]
    _TABLES[cache_key] = t
    return t


def load_tables(key: str) -> list[dict]:
    """Every table measured on this GPU (one per expert layout)."""
    if not key or not _table_dir().exists():
        return []
    names = sorted({f.name[:-len(f"_{key}.json")] for f in _table_dir().glob(f"*_{key}.json")})
    return [t for t in (load_table(key, n) for n in names) if t is not None]


MIN_GAIN = 1.03   # a row serves its tactic only if it beat FlashInfer's own choice by this much in the sweep


def lookup(table: dict, n: int) -> Optional[list]:
    """Tactic of the smallest measured token count >= n (the largest row past the end of the table), or None where
    the sweep found FlashInfer's own choice as fast (prefill-sized calls, >= ~12k tokens)."""
    toks = table["_tokens"]
    if not toks:
        return None
    row = table["table"][min(bisect.bisect_left(toks, n), len(toks) - 1)]
    if row.get("speedup", MIN_GAIN) < MIN_GAIN:
        return None
    return list(row["tactic"])


def effective_shape(runner) -> Optional[dict]:
    """The per-GPU shape the kernel runs: local experts and top-k including fused shared experts (see the module
    doc), global experts likewise when the runner records them (FlashInfer >= 0.6.18 MoERunner.num_experts)."""
    try:
        nfse = int(getattr(runner, "num_fused_shared_experts", 0) or 0)
        out = {"hidden": runner.hidden_size, "intermediate_per_rank": runner.intermediate_size,
               "experts": runner.num_local_experts + nfse, "top_k": runner.top_k + nfse}
    except (AttributeError, TypeError):
        return None
    ne = getattr(runner, "num_experts", None)
    if isinstance(ne, int):
        out["global_experts"] = ne + nfse
    return out


def shape_matches(runner, table: dict) -> bool:
    """True when the table was swept at this runner's per-GPU shape, fused shared experts included: a fused runner
    (385 local, top-k 7 either way) never matches a plain 384/6 table, and a plain runner never matches the _fs one."""
    s, r = table["shape"], effective_shape(runner)
    if r is None:
        return False
    if any(r[k] != s[k] for k in ("hidden", "intermediate_per_rank", "experts", "top_k")):
        return False
    return not ("global_experts" in r and "global_experts" in s and r["global_experts"] != s["global_experts"])


def choose(runner, inputs, table: dict, rows: int) -> Optional[list]:
    """The served tactic for this call, or None to defer to FlashInfer."""
    if not shape_matches(runner, table):
        return None
    t = lookup(table, rows)
    if t is None:
        return None
    try:
        valid = [list(v) for v in runner.get_valid_tactics(inputs, None)]
    except Exception:  # noqa: BLE001
        return None
    return t if t in valid else None


def _rows(inputs) -> Optional[int]:
    """Token count of the call: MoeRunnerInputs' hidden_states (index 4 in its flat order)."""
    try:
        from flashinfer.fused_moe.core import MoeRunnerInputs
        return int(MoeRunnerInputs.from_list(inputs).hidden_states.shape[0])
    except Exception:  # noqa: BLE001
        h = inputs[4] if len(inputs) > 4 else None
        return int(h.shape[0]) if hasattr(h, "shape") else None


def in_scope() -> bool:
    """NUMINOUS_MOE_TACTICS_SCOPE=graph (default): serve the table only for calls captured into CUDA graphs, i.e.
    pure decode steps, whose tokens come from many requests (close to the uniform routing the table was swept on).
    Eager calls (prefill, mixed and extend steps: one prompt's chunk concentrates on few experts) keep FlashInfer's
    choice; on real agent traffic the table applied to every call lost 11% on B200 (k4). =all serves every call."""
    if os.environ.get("NUMINOUS_MOE_TACTICS_SCOPE", "graph") == "all":
        return True
    import torch
    return bool(torch.cuda.is_current_stream_capturing())


_CAPTURED: dict = {}


def _capture(inputs, rows: int) -> None:
    """NUMINOUS_MOE_CAPTURE_DIR: save the packed top-k of a few real eager MoE calls per power-of-two token bucket
    (never inside CUDA-graph capture), so tactics can be swept on real routing skew instead of uniform routing."""
    d = os.environ.get("NUMINOUS_MOE_CAPTURE_DIR")
    if not d:
        return
    import torch
    if torch.cuda.is_current_stream_capturing():
        return
    bucket = 1 << max(0, int(rows) - 1).bit_length()
    per = int(os.environ.get("NUMINOUS_MOE_CAPTURE_PER_BUCKET", "4"))
    if _CAPTURED.get(bucket, 0) >= per:
        return
    try:
        from flashinfer.fused_moe.core import MoeRunnerInputs
        ids = MoeRunnerInputs.from_list(inputs).topk_ids
    except Exception:  # noqa: BLE001
        ids = inputs[2] if len(inputs) > 2 else None
    if ids is None or not hasattr(ids, "shape"):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:                               # TP experts: every rank routes the same gathered tokens
        return
    _CAPTURED[bucket] = _CAPTURED.get(bucket, 0) + 1
    os.makedirs(d, exist_ok=True)
    torch.save(ids.detach().cpu(), os.path.join(d, f"topk_r{rank}_n{int(rows)}_{_CAPTURED[bucket]}.pt"))


def install() -> bool:
    """Route FlashInfer's tactic choice for the FP4 MoE op through the measured tables (idempotent). No CUDA call
    here: the tables for this GPU are resolved on the first MoE call, inside a scheduler process (this runs at
    module import, which launcher and tokenizer processes also do)."""
    global _INSTALLED
    serve = os.environ.get("NUMINOUS_MOE_TACTICS", "1") == "1"
    capture = bool(os.environ.get("NUMINOUS_MOE_CAPTURE_DIR"))
    if _INSTALLED or not (serve or capture):
        return _INSTALLED
    from flashinfer.autotuner import AutoTuner

    orig = AutoTuner.choose_one
    state: dict = {}

    def tables() -> list:
        if "tables" not in state:
            import torch
            key = gpu_key(torch.cuda.get_device_capability()) if torch.cuda.is_available() else ""
            state["tables"] = load_tables(key)
            logger.warning("numinous_moe_tactics: %s tables for %s: %s", len(state["tables"]), key or "this GPU",
                           [(t.get("model", "pro"), t.get("layout", "tp8"), len(t["table"])) for t in state["tables"]])
        return state["tables"]

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kw):
        if custom_op == OP and not getattr(self, "is_tuning_mode", False) and runners:
            if os.environ.get("NUMINOUS_MOE_CAPTURE_DIR"):
                r0 = _rows(inputs)
                if r0:
                    _capture(inputs, r0)
            tabs = tables() if serve and in_scope() else []
            rows = _rows(inputs) if tabs else None
            table = next((t for t in tabs if shape_matches(runners[0], t)), None) if rows else None
            if table is not None:
                t = choose(runners[0], inputs, table, int(rows))
                if t is not None:
                    tag = f"{table.get('model', 'pro')}/{table.get('layout', 'tp8')}"
                    if tag not in _SERVED:
                        _SERVED.add(tag)
                        logger.warning("numinous_moe_tactics active: %s tactic %s for %d tokens (%s)", tag, t, rows,
                                       table.get("sm"))
                    return runners[0], t
        return orig(self, custom_op, runners, tuning_config, inputs, **kw)

    AutoTuner.choose_one = choose_one
    _INSTALLED = True
    return True
