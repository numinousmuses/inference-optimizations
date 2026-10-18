"""Preflight: what a cell + arms launch actually resolves to, from the engine's own argument code, before a GPU boot.

Runs inside the engine image (a dev box or a gate container): parses the composed launch flags with SGLang's
prepare_server_args (post-init included: checkpoint quant config, backend auto-selection, speculative defaults),
then reports
  - the resolved value of every flag given and of the settings that decide performance (KEY_FIELDS),
  - flags whose resolved value differs from what was passed (a flag that silently did nothing),
  - SGLANG_* / NUMINOUS_* env names that no code in the engine package or our overlays reads (a typo or a dead knob),
  - parse errors (unknown or removed flags).

    python3 operations/modal/preflight.py --cell glm-5.3/standard --arms glm_v20 [--json out.json]
    python3 operations/modal/preflight.py --cell deepseek-v4-pro/standard --arms v20_k,pro_k_dpmoe
    python3 operations/modal/preflight.py --cell deepseek-v4-pro/standard --arms v20 \
        --flags "--speculative-dspark-block-size 7" --env SGLANG_RAGGED_VERIFY_MODE=compact

--flags / --remove / --env add a one-off delta after the arms (spec_recipes.py drives run() with its recipes).
On v0.5.20 the ServerArgs record stays raw until ServerArgs.resolve_once() runs the resolution pipeline (DP chunk
division, backend and page-size selection, speculative defaults such as the checkpoint's DSpark gamma); preflight
runs it (--no-resolve reports the raw record, the pre-v0.5.20 behaviour).

Needs the model's config.json (a local /cache copy or a Hub download), never its weights.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from gate_arms import ARMS, compose_flags  # noqa: E402

KEY_FIELDS = [
    "tp_size", "dp_size", "ep_size", "enable_dp_attention", "moe_runner_backend", "moe_a2a_backend", "quantization",
    "kv_cache_dtype", "attention_backend", "prefill_attention_backend", "decode_attention_backend",
    "nsa_prefill_backend", "nsa_decode_backend", "dsa_prefill_backend", "dsa_decode_backend", "page_size",
    "chunked_prefill_size", "mem_fraction_static", "max_running_requests", "cuda_graph_max_bs",
    "speculative_algorithm", "speculative_num_steps", "speculative_eagle_topk", "speculative_num_draft_tokens",
    "speculative_dspark_block_size", "disable_shared_experts_fusion", "enforce_shared_experts_fusion",
    "swa_full_tokens_ratio", "enable_hierarchical_cache", "hicache_ratio", "enable_deepseek_v4_fp4_indexer",
    "enable_w4a4_mxfp4_megamoe", "enable_two_batch_overlap", "disable_cuda_graph", "enable_prefill_delayer",
    "num_continuous_decode_steps", "prefill_decode_interval",
    # speculation (spec_recipes.py): what DSpark / EAGLE / DFlash resolve to
    "speculative_draft_model_path", "speculative_dspark_sps_table_path",
    "speculative_dspark_align_verify_tokens_to_graph_tier", "speculative_adaptive", "speculative_adaptive_config",
    "speculative_draft_attention_backend", "speculative_dflash_block_size", "disable_overlap_schedule",
]
FOREIGN_ENV = re.compile(r"^(NCCL_|CUDA_|TORCH_|PYTORCH_|FLASHINFER_|TRTLLM_|DG_|HF_|VLLM_|NVSHMEM_|MC_|OMP_|GLOO_)")


def load_cell(key: str) -> dict:
    # GATE_REGISTRY: an extended registry with candidate cells (same override cell_gate.py takes)
    reg_path = pathlib.Path(os.environ.get("GATE_REGISTRY") or ROOT / "configs/serving_registry.json")
    reg = json.loads(reg_path.read_text())
    cells = reg["cells"] if isinstance(reg, dict) and "cells" in reg else reg
    model, lane = key.split("/")
    for c in cells:
        if c.get("model") == model and c.get("lane") == lane:
            return c
    raise SystemExit(f"no registry cell {key}")


def compose(arms: list[str]) -> dict:
    out: dict = {}
    for n in arms:
        for k, v in ARMS[n].items():
            if k in ("env", "overlay"):
                out[k] = {**out.get(k, {}), **v}
            elif k in ("flags_add", "flags_remove"):
                out[k] = out.get(k, []) + list(v)
            else:
                out[k] = v
    return out


def model_path(repo: str) -> str:
    local = f"/cache/{repo.replace('/', '__')}"
    return local if os.path.isfile(f"{local}/config.json") else repo


def env_tokens() -> set[str]:
    """Every SGLANG_* / NUMINOUS_* name mentioned in the engine package or our kernel/overlay sources."""
    import importlib.util
    roots = [list(importlib.util.find_spec("sglang").submodule_search_locations)[0],
             str(ROOT / "operations/kernels"), str(ROOT / "operations/patches")]
    pat = re.compile(r"\b((?:SGLANG|NUMINOUS)_[A-Z0-9_]+)\b")
    seen: set[str] = set()
    for r in roots:
        for f in pathlib.Path(r).rglob("*.py"):
            try:
                seen.update(pat.findall(f.read_text(errors="ignore")))
            except OSError:
                pass
    return seen


def given_values(flags: list[str]) -> dict[str, object]:
    """--flag value / bare --flag (True) as passed, keyed by the dataclass field name."""
    out, i = {}, 0
    while i < len(flags):
        f = flags[i]
        if f.startswith("--"):
            name = f[2:].replace("-", "_")
            if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
                out[name] = flags[i + 1]
                i += 2
                continue
            out[name] = True
        i += 1
    return out


def same(given: object, resolved: object) -> bool:
    if given is True:
        return resolved is True or resolved == 1
    return str(given) == str(resolved) or (isinstance(resolved, (int, float)) and _num(given) == resolved)


def _num(x: object):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def resolved_fields(sa) -> dict:
    """v0.5.20 ServerArgs is a msgspec Struct with resolved_dict; older ones are dataclasses."""
    rd = getattr(sa, "resolved_dict", None)
    if rd is not None:
        return dict(rd() if callable(rd) else rd)
    return {f.name: getattr(sa, f.name) for f in dataclasses.fields(sa)}


def checkpoint_quant(path: str) -> dict:
    """The checkpoint's own quantization declaration (config.json quantization_config, hf_quant_config.json)."""
    out: dict = {}
    for name in ("config.json", "hf_quant_config.json"):
        f = pathlib.Path(path) / name
        if not f.is_file():
            continue
        d = json.loads(f.read_text())
        q = d.get("quantization_config") or d.get("quantization") or {}
        for k in ("quant_method", "quant_algo", "kv_cache_quant_algo", "fmt"):
            if isinstance(q, dict) and q.get(k):
                out[k] = q[k]
    return out


def merge_delta(delta: dict, extra: dict | None) -> dict:
    """An extra delta (same keys as an arm) applied after the composed arms."""
    out = dict(delta)
    for k, v in (extra or {}).items():
        if k in ("env", "overlay"):
            out[k] = {**out.get(k, {}), **v}
        elif k in ("flags_add", "flags_remove"):
            out[k] = out.get(k, []) + list(v)
        else:
            out[k] = v
    return out


def run(cell_key: str, arms: list[str], extra: dict | None = None, resolve: bool = True) -> dict:
    cell = load_cell(cell_key)
    cfg = cell["config"]
    delta = merge_delta(compose(arms), extra)
    path = model_path(cfg["repo"])
    flags = compose_flags(cfg["flags"], delta, path)
    env = {**(cfg.get("env") or {}), **delta.get("env", {})}
    report = {"cell": cell_key, "arms": arms, "image": delta.get("image") or cfg["image"], "flags": flags,
              "env": env, "warnings": [], "resolved": {}}
    known = env_tokens()
    for name in env:
        if FOREIGN_ENV.match(name):
            continue
        if name not in known:
            report["warnings"].append(f"env {name}: no code in the engine or our overlays reads it")
    os.environ.update({k: str(v) for k, v in env.items()})
    argv = ["--model-path", path, "--port", "30000"] + [f for f in flags if f != "__SPS__"]
    try:
        from sglang.srt.server_args import prepare_server_args
        sa = prepare_server_args(argv)
    except SystemExit as e:  # argparse: unknown or removed flag
        report["warnings"].append(f"argument parsing failed (exit {e.code}); see stderr")
        return report
    except Exception as e:  # noqa: BLE001  post-init rejected the combination
        report["warnings"].append(f"post-init rejected the launch: {type(e).__name__}: {e}")
        return report
    if resolve and hasattr(sa, "resolve_once"):
        try:
            sa.resolve_once()
            report["resolution"] = "resolve_once"
        except Exception as e:  # noqa: BLE001  a resolution handler rejected the combination
            report["warnings"].append(f"post-init rejected the launch: resolution: {type(e).__name__}: {e}")
            return report
    else:
        report["resolution"] = "raw"
    resolved = resolved_fields(sa)
    fields = set(resolved)
    report["resolved"] = {k: resolved[k] for k in KEY_FIELDS if k in resolved}
    report["all"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                     for k, v in resolved.items()}
    ckpt = checkpoint_quant(path)
    if ckpt:
        report["checkpoint_quant"] = ckpt
        if report["resolved"].get("kv_cache_dtype") == "auto" and ckpt.get("kv_cache_quant_algo"):
            report["warnings"].append(f"kv_cache_dtype auto resolves to the checkpoint's "
                                      f"{ckpt['kv_cache_quant_algo']} KV at model load (pass it explicitly to change)")
    for name, val in given_values(flags).items():
        if name not in fields:
            continue
        report["resolved"].setdefault(name, resolved[name])
        if not same(val, resolved[name]):
            report["warnings"].append(f"--{name.replace('_', '-')} {val} resolved to {resolved[name]!r}")
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--arms", default="control")
    ap.add_argument("--json", default="")
    ap.add_argument("--flags", default="", help="extra launch flags appended after the arms (one string)")
    ap.add_argument("--remove", default="", help="comma list of flags to drop (with their values)")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument("--no-resolve", action="store_true", help="report the raw record (skip resolve_once)")
    a = ap.parse_args(argv)
    extra: dict = {}
    if a.flags:
        extra["flags_add"] = a.flags.split()
    if a.remove:
        extra["flags_remove"] = [x for x in a.remove.split(",") if x]
    if a.env:
        extra["env"] = dict(kv.split("=", 1) for kv in a.env)
    rep = run(a.cell, [x for x in a.arms.split(",") if x], extra or None, resolve=not a.no_resolve)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep.get(k) for k in ("cell", "arms", "image", "resolution", "env", "resolved", "warnings")},
                     indent=1, default=str))
    return 1 if any(w.startswith(("argument parsing", "post-init")) for w in rep["warnings"]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
