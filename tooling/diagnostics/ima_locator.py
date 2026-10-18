"""Sync-free locator for CUDA illegal-memory-access crashes in SGLang hybrid (mamba/KDA + DSA) serving.

Agent k3amd for rtxsys, 2026-10-01. Installed into the tree as sglang/srt/numinous_ima_locator.py by
operations/patches/ima_locator_hooks.py; inert unless NUMINOUS_IMA_LOCATOR=1.

Why: the GLM-5.3-Flash DP4/EP4 + EAGLE + extra_buffer IMA surfaces only through the NCCL watchdog (no kernel named),
and every serializing diagnostic tried so far (CUDA_LAUNCH_BLOCKING=1, SGLANG_MAMBA_DEBUG_ASSERTS=1) made it vanish.
This module checks the pool indices each consuming kernel is about to use with plain device ops (no .item(), no
synchronize, no pageable H2D copy), so host/stream timing stays as served. Per check:

    bad = (v < lo) | (v >= hi)   [optionally: v == -1 allowed, masked rows ignored]
    rec[site] = [count, first_bad_value, first_bad_pos, lo, hi, numel, step_at_first_hit]
    host_pinned[site] <- rec[site]   (non_blocking D2H on the SAME stream, i.e. ordered before the consumer)

The consumer that would fault runs after the D2H in stream order, so the record is in host memory before the IMA.
A daemon thread polls the pinned buffer (plain host reads, no CUDA call, works after the context dies) and dumps
new hits to stderr and NUMINOUS_IMA_LOCATOR_DIR/ima_locator_<pid>.json at once.

Detection only: indices are never clamped or rewritten. Skips inside CUDA-graph capture (graph-internal reads are
covered by the pre-replay metadata site).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Optional

import torch

ENABLED = os.environ.get("NUMINOUS_IMA_LOCATOR", "0") == "1"
OUT_DIR = os.environ.get("NUMINOUS_IMA_LOCATOR_DIR", "/tmp")
POLL_S = float(os.environ.get("NUMINOUS_IMA_LOCATOR_POLL_S", "0.1"))

# site name -> what it guards (the pool each index addresses)
SITES = {
    "fwd_req_pool_indices": "ForwardBatch.req_pool_indices vs req_to_token rows (model_runner._forward_raw)",
    "fwd_out_cache_loc": "ForwardBatch.out_cache_loc vs KV pool rows (model_runner._forward_raw)",
    "mamba_cow_src": "deferred COW source slot vs mamba pool / int8 ckpt pool (model_runner)",
    "mamba_cow_dst": "deferred COW destination slot vs mamba pool (model_runner)",
    "mamba_clear": "deferred clear slot vs mamba pool (model_runner)",
    "mamba_eager_state": "eager KDA/mamba state indices vs mamba pool (hybrid_linear_attn_backend._forward_metadata)",
    "mamba_eager_track": "eager track dst slots (masked rows) vs mamba pool (hybrid_linear_attn_backend)",
    "mamba_replay_state": "graph-replay state indices vs mamba pool (hybrid_linear_attn_backend._replay_metadata)",
    "verify_state": "post-verify commit state slots vs mamba pool (spec_utils.commit_mamba_states_after_verify)",
    "verify_track": "post-verify commit track slots (steps>=0 rows) vs mamba pool (spec_utils)",
    "verify_step": "post-verify last accepted step vs draft_token_num (spec_utils)",
    "dsa_page_table": "DSA sparse-MLA page_table_1 vs KV rows (dsa_backend._forward_flashinfer_sparse_mla)",
    "moe_topk_ids": "MoE topk_ids vs logical/physical experts, -1 allowed (moe/topk.select_experts)",
    "moe_topk_weights_nan": "MoE routing weights NaN flag (value 1 = NaN) (moe/topk.select_experts)",
}
SITE_IDS = {name: i for i, name in enumerate(SITES)}
FIELDS = ("count", "first_bad_value", "first_bad_pos", "lo", "hi", "numel", "step")
NF = len(FIELDS)


class _DeviceState:
    def __init__(self, device: torch.device):
        self.device = device
        self.rec = torch.zeros((len(SITES), NF), dtype=torch.int64, device=device)
        self.step = torch.zeros((), dtype=torch.int64, device=device)
        pin = device.type == "cuda"
        self.host = torch.zeros((len(SITES), NF), dtype=torch.int64, pin_memory=pin)


_STATES: dict = {}
_LOCK = threading.Lock()
_WATCHER: Optional[threading.Thread] = None
_SEEN: dict = {}


def _state(device: torch.device) -> _DeviceState:
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (device.type, device.index)
    st = _STATES.get(key)
    if st is None:
        with _LOCK:
            st = _STATES.get(key)
            if st is None:
                st = _DeviceState(device)
                _STATES[key] = st
                _start_watcher()
    return st


def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def tick(device) -> None:
    """Advance the per-device forward counter (call once per forward); device op, no sync."""
    if not ENABLED or _capturing():
        return
    _state(torch.device(device)).step.add_(1)


def kv_rows(pool) -> Optional[int]:
    """Addressable token rows of a KV pool (size + one padding page), None when the pool does not say."""
    size = getattr(pool, "size", None)
    if not isinstance(size, int):
        return None
    page = getattr(pool, "page_size", 1)
    return size + (page if isinstance(page, int) and page > 0 else 1)


def ckpt_rows(ckpt_pool) -> Optional[int]:
    """Slots of the int8 mamba checkpoint pool (conv[i] is [layers, slots, ...]); None when off."""
    conv = getattr(ckpt_pool, "conv", None) if ckpt_pool is not None else None
    if not conv:
        return None
    return int(conv[0].shape[1])


def token_rows(kv_cache: torch.Tensor) -> Optional[int]:
    """Token rows of a DSA/MLA cache tensor ([rows, D], [rows, 1, D] or [pages, page, D])."""
    if not torch.is_tensor(kv_cache) or kv_cache.dim() < 2 or kv_cache.shape[-1] == 0:
        return None
    return kv_cache.numel() // kv_cache.shape[-1]


def check(site: str, idx: Optional[torch.Tensor], lo: int, hi: Optional[int], *,
          mask: Optional[torch.Tensor] = None, allow_neg1: bool = False) -> None:
    """Record (without syncing) whether any idx outside [lo, hi) reaches the consumer guarded by `site`."""
    if not ENABLED or hi is None or idx is None or not torch.is_tensor(idx) or idx.numel() == 0 or _capturing():
        return
    sid = SITE_IDS[site]
    st = _state(idx.device)
    v = idx.reshape(-1)
    m = None
    if mask is not None:
        m = mask.reshape(-1)
        if m.dtype != torch.bool:
            m = m != 0
        n = min(m.numel(), v.numel())  # rows past the mask (e.g. a padded static buffer) are not live
        if n == 0:
            return
        v, m = v[:n], m[:n]
    if v.dtype != torch.int64:
        v = v.to(torch.int64)
    bad = (v < lo) | (v >= hi)
    if allow_neg1:
        bad = bad & (v != -1)
    if m is not None:
        bad = bad & m
    nbad = bad.sum()
    pos = bad.to(torch.int8).argmax().reshape(1)
    val = torch.gather(v, 0, pos).reshape(())
    row = st.rec[sid]
    first = (nbad > 0) & (row[0] == 0)
    upd = torch.stack((
        row[0] + nbad,
        torch.where(first, val, row[1]),
        torch.where(first, pos.reshape(()), row[2]),
        torch.where(first, torch.full_like(row[3], lo), row[3]),
        torch.where(first, torch.full_like(row[4], hi), row[4]),
        torch.where(first, torch.full_like(row[5], v.numel()), row[5]),
        torch.where(first, st.step, row[6]),
    ))
    row.copy_(upd)
    st.host[sid].copy_(row, non_blocking=True)


def snapshot() -> list[dict]:
    """Current hits across devices (host reads only)."""
    out = []
    for (dtype, dindex), st in list(_STATES.items()):
        h = st.host.numpy()
        for name, sid in SITE_IDS.items():
            if h[sid, 0] > 0:
                rec = {f: int(h[sid, i]) for i, f in enumerate(FIELDS)}
                rec.update(site=name, guards=SITES[name], device=f"{dtype}:{dindex}")
                out.append(rec)
    return out


def _dump(hits: list[dict]) -> None:
    payload = {"pid": os.getpid(), "time": time.time(), "hits": hits}
    line = "numinous ima_locator HIT " + json.dumps(payload)
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"ima_locator_{os.getpid()}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, path)
    except Exception:
        pass


def poll_once() -> list[dict]:
    """Report hits whose count changed since the last poll. Returns the new/changed records."""
    hits = snapshot()
    changed = [h for h in hits if _SEEN.get((h["device"], h["site"])) != h["count"]]
    if changed:
        for h in changed:
            _SEEN[(h["device"], h["site"])] = h["count"]
        _dump(hits)
    return changed


def _watch() -> None:
    while True:
        time.sleep(POLL_S)
        try:
            poll_once()
        except Exception:
            pass


def _start_watcher() -> None:
    global _WATCHER
    if _WATCHER is None and os.environ.get("NUMINOUS_IMA_LOCATOR_NO_THREAD", "0") != "1":
        _WATCHER = threading.Thread(target=_watch, name="numinous-ima-locator", daemon=True)
        _WATCHER.start()
