"""GLM-5.3-Flash KDA prefill on MoonshotAI FlashKDA past SGLang's caps, including tracked (extra_buffer) batches
(amdflash kernel program L4, 2026-10-05). Installed as srt/layers/attention/linear/numinous_flashkda.py by
operations/patches/flash_kda_flashkda_hooks.py.

Why (measured, 1x H200 at Flash's served shapes, research/gpu-evidence/amdflash/flash_kda_flashkda_bench_h200.json):
- FlashKDA takes 0.51-0.75x of SGLang's Triton chunk_kda:
  - 4,096 tokens: -7.6..-14.5 ms per prefill step (34 KDA layers);
  - 8,192 tokens: -15.2..-29.1 ms.
- Outputs agree with Triton at 5.9e-3; vs an fp64 recurrence, Triton is at 4.2e-3 and FlashKDA at 5.8e-3
  (flash_kda_flashkda_debug_h200.json).

Why stock SGLang never runs it for this config (source_code, v0.5.20):
- kda_flashkda.py:12-15 sends a batch to Triton when any sequence is > 2,048 or < 64 tokens.
- kda_flashkda.py:121-127 sends every tracked batch to Triton.
  - With --mamba-radix-cache-strategy extra_buffer, every request with an extend of >= 64 tokens is tracked
    (schedule_batch.py:2921, mask = extend length >= checkpoint grid, which is 64 here).
  - So every prefill batch is tracked.

What this module does for a tracked batch with unaligned rows (the state is needed at an interior 64-aligned point
p_i of request i, track_chunk_idx = p_i / 64):
- Pass A: one FlashKDA call over the batch, with each such request split at p_i into a head [s, s+p) and a tail
  [s+p, e). The tail runs from a zero state; its output is discarded.
  - Head final state = the snapshot (h_track_buf row).
  - Full segments' final states go to the pool.
- Pass B: the tails (< 64 tokens each) are gathered and run through a second, small FlashKDA call from the head
  states.
  - Their outputs overwrite the tail rows; their final states go to the pool.
  - Measured: FlashKDA handles 5- and 63-token segments at the same 5.9e-3. Triton for the tails cost a whole
    chunk_kda call's fixed overhead (e2e run 1: split batches 1.45-2.3x slower than stock).
- The index tensors of a plan come from one pinned host buffer with one async copy. They are built once per forward
  (same lens / track for its 34 KDA layers).
p_i comes from the CPU copy of track_chunk_idx that hybrid_linear_attn_backend._init_track_ssm_indices builds (stashed
by the hook, keyed by the device tensor's id), so nothing syncs.
Fallback to the stock Triton path: no stash match, lower_bound unset, spec forwards, or the package fails to load.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

import torch

ENV = "NUMINOUS_FLASHKDA"
DIR_ENV = "NUMINOUS_FLASHKDA_DIR"
CHUNK = 64
log = logging.getLogger(__name__)
_STASH: dict = {"id": None, "cpu": None}
_FK = {"mod": None, "err": None}
STATS = {"flashkda": 0, "split": 0, "fallback": 0}


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def stash_track_idx(dev: Optional[torch.Tensor], cpu: torch.Tensor):
    """Record the CPU track_chunk_idx built next to the device copy (hook in _init_track_ssm_indices)."""
    _STASH["id"], _STASH["cpu"] = (id(dev), cpu.tolist()) if dev is not None else (None, None)
    return dev


def track_idx_for(dev: Optional[torch.Tensor], n: int) -> Optional[list]:
    if dev is None or _STASH["id"] != id(dev) or _STASH["cpu"] is None or len(_STASH["cpu"]) != n:
        return None
    return _STASH["cpu"]


BAKED_DIR = "/opt/numinous/flash_kda-7afb9f45"          # built by flash_kda_flashkda_hooks.ensure_package


def package_dir() -> Optional[str]:
    """The first FlashKDA copy with a complete manifest: the staged volume dir (NUMINOUS_FLASHKDA_DIR), then the copy
    the hook series bakes into the image or node."""
    for d in (os.environ.get(DIR_ENV), BAKED_DIR):
        if not d or not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        man = json.load(open(os.path.join(d, "manifest.json")))
        if all(os.path.exists(os.path.join(d, f)) for f in man["files"]):
            return d
    return None


def load_flash_kda():
    """flash_kda from package_dir(), else the installed package; None (Triton stays) when neither imports."""
    if _FK["mod"] is not None or _FK["err"] is not None:
        return _FK["mod"]
    try:
        d = package_dir()
        if d and d not in sys.path:
            sys.path.insert(0, d)
        import flash_kda
        _FK["mod"] = flash_kda
    except Exception as e:  # noqa: BLE001
        _FK["err"] = f"{type(e).__name__}: {e}"
        log.warning("numinous_flashkda: flash_kda unavailable, Triton path stays (%s)", _FK["err"])
    return _FK["mod"]


def plan(lens: list, track: Optional[list]) -> tuple[list, list]:
    """Host plan. segs: (start, end, kind, req) in batch order, kind in full / head / tail. snap0: requests whose
    snapshot point is their extend start (p = 0: the snapshot is the initial state)."""
    segs, snap0, s = [], [], 0
    for i, n in enumerate(lens):
        p = CHUNK * track[i] if track is not None and track[i] is not None and track[i] >= 0 else -1
        if 0 < p < n:
            segs += [(s, s + p, "head", i), (s + p, s + n, "tail", i)]
        else:
            if p == 0:
                snap0.append(i)
            segs.append((s, s + n, "full", i))
        s += n
    return segs, snap0


def _fwd(fk, q, k, v, g, beta, init, cu, A_log, dt_bias, lower_bound):
    Kd = q.shape[3]
    out = torch.empty_like(v)
    final = torch.empty_like(init)
    fk.fwd(q, k, v, g, beta, Kd ** -0.5, out, A_log, dt_bias, lower_bound, initial_state=init, final_state=final,
           cu_seqlens=cu)
    return out, final


_PLAN: dict = {"key": None, "val": None}


def device_plan(lens: list, track: Optional[list], dev) -> dict:
    """Index tensors for one forward's plan, built once and shared by its 34 KDA layers (same lens / track).
    One pinned host buffer and one async H2D copy: a pageable copy per tensor would block the host every layer."""
    key = (tuple(lens), tuple(track) if track is not None else None, str(dev))
    if _PLAN["key"] == key:
        return _PLAN["val"]
    segs, snap0 = plan(lens, track)
    live = [j for j, sg in enumerate(segs) if sg[2] != "tail"]
    full = [j for j, sg in enumerate(segs) if sg[2] == "full"]
    tails = [j for j, sg in enumerate(segs) if sg[2] == "tail"]
    t_rows = [x for j in tails for x in range(segs[j][0], segs[j][1])]
    t_cu = [0]
    for j in tails:
        t_cu.append(t_cu[-1] + segs[j][1] - segs[j][0])
    parts = {
        "cu": [0] + [e for _, e, _, _ in segs], "live": live, "live_req": [segs[j][3] for j in live],
        "full": full, "full_req": [segs[j][3] for j in full], "heads": [j - 1 for j in tails],
        "tail_req": [segs[j][3] for j in tails], "rows": t_rows, "tcu": t_cu, "snap0": snap0,
    }
    flat, off = [], {}
    for name, vals in parts.items():
        off[name] = (len(flat), len(flat) + len(vals))
        flat += vals
    host = torch.tensor(flat, dtype=torch.int64)
    if dev.type == "cuda":
        host = host.pin_memory()
    buf = host.to(dev, non_blocking=True)
    val = {name: buf[a:b] for name, (a, b) in off.items()}
    val.update(n_seg=len(segs), n_tail=len(tails), tail_lens=[t_cu[x + 1] - t_cu[x] for x in range(len(tails))],
               host=host)                                                 # keep the pinned source alive
    _PLAN["key"], _PLAN["val"] = key, val
    return val


def extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, A_log=None, dt_bias=None,
           lower_bound=None, extend_seq_lens_cpu=None, is_spec_decode=False, beta_is_raw=False,
           return_intermediate_states=False, triton_extend=None, **kwargs):
    """Drop-in for FlashKDAKernel.extend (same return contract as TritonKDAKernel.extend)."""
    track_state = kwargs.get("track_state")
    lens = [int(x) for x in (extend_seq_lens_cpu.tolist() if torch.is_tensor(extend_seq_lens_cpu)
                             else (extend_seq_lens_cpu or []))]
    track = track_idx_for(kwargs.get("track_chunk_idx"), len(lens)) if track_state is not None else None
    fk = load_flash_kda() if (lower_bound is not None and not is_spec_decode and lens) else None
    if fk is None or (track_state is not None and track is None) or sum(lens) != q.shape[1] or min(lens) <= 0:
        STATS["fallback"] += 1
        return triton_extend(q, k, v, g, beta, ssm_states=ssm_states, cache_indices=cache_indices,
                             query_start_loc=query_start_loc, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                             extend_seq_lens_cpu=extend_seq_lens_cpu, is_spec_decode=is_spec_decode,
                             beta_is_raw=beta_is_raw, return_intermediate_states=return_intermediate_states, **kwargs)
    H = q.shape[2]
    dev = q.device
    qc, kc, vc, gc = (x.contiguous() for x in (q, k, v, g))
    fdt = torch.float64 if q.dtype == torch.float64 else torch.float32      # fp64 only in the CPU tests
    braw = beta if beta_is_raw else torch.logit(beta.to(fdt).clamp(1e-7, 1.0 - 1e-7))
    braw = braw.to(q.dtype).contiguous()                                     # bf16 in production (FlashKDA's dtype)
    a_log = A_log.reshape(-1).to(fdt).contiguous()
    dtb = dt_bias.reshape(H, -1).to(fdt).contiguous() if dt_bias is not None else None
    P = device_plan(lens, track, dev)
    ci = cache_indices.long()
    init = torch.zeros((P["n_seg"], *ssm_states.shape[1:]), dtype=ssm_states.dtype, device=dev)
    init[P["live"]] = ssm_states[ci[P["live_req"]]]
    if P["snap0"].numel():                                              # snapshot = state before this extend
        track_state[P["snap0"]] = ssm_states[ci[P["snap0"]]].to(track_state.dtype)
    out, final = _fwd(fk, qc, kc, vc, gc, braw, init, P["cu"], a_log, dtb, lower_bound)
    if P["full"].numel():
        ssm_states[ci[P["full_req"]]] = final[P["full"]]
    STATS["flashkda"] += 1
    if P["n_tail"]:
        STATS["split"] += 1
        hstate = final[P["heads"]]
        track_state[P["tail_req"]] = hstate.to(track_state.dtype)
        rows = P["rows"]
        sub = lambda x: x[:, rows].contiguous()                                  # noqa: E731
        o_t, f_t = _fwd(fk, sub(qc), sub(kc), sub(vc), sub(gc), sub(braw), hstate.contiguous(), P["tcu"], a_log,
                        dtb, lower_bound)
        out[:, rows] = o_t
        ssm_states[ci[P["tail_req"]]] = f_t
    return (out, None) if return_intermediate_states else out
