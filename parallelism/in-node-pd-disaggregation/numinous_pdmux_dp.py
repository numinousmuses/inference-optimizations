"""pdmux under DP attention: the scheduler side (amdglm; agents/findings/amdglm.md section 10, items S1-S4).
Installed by glm_pdmux_dp_hooks.py as sglang/srt/multiplex/numinous_pdmux_dp.py. The stock pdmux loop
(multiplex/multiplexing_mixin.py:100-228) delegates here when NUMINOUS_PDMUX_DP=1 and DP attention is on.

The rules that make two lanes safe across 8 DP ranks:
  1. Every collective sequence is identical on every rank. The prefill (P) lane plans only when no split is in flight
     (a global state). One all-gather on the CPU group decides whether a split starts, and a rank without a prefill
     gets an IDLE split. All ranks then advance the same layer range per iteration (forward count from the GLOBAL max
     tokens), finish in the same iteration (the stock done-flag allreduce), and hand off in that iteration.
  2. The decode (D) lane runs its own MLP sync every iteration (a decode batch, an IDLE batch, or nothing on every
     rank). It uses the CPU group as well, so no scheduler collective is enqueued on a CUDA stream.
  3. Lanes switch only between forwards (set_pdmux_status). rtxv41's glm_pdmux_lane_hooks keep each lane's
     communicators, DP buffers and W4A8 metadata apart.
  4. EAGLE hand-off (S4): the prefill draft-extend runs on the D lane at the finish point, after this iteration's
     decode step, on every rank (IDLE on ranks without a prefill). The draft runner's attention backends are single
     instances that the D lane uses every step, so the P lane never touches them.
  5. Cross-stream lifetimes: the D stream waits on the P stream's done event before the hand-off. A split batch
     adopted as the running batch has its tensors record_stream'ed on the D stream (merge_batch builds new D tensors).
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

ENV = "NUMINOUS_PDMUX_DP"
logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


# ---------------------------------------------------------------------------------------------------- pure rules
def split_forward_count(global_tokens: list, budget: int, num_layers: int, split_index: int) -> int:
    """Layers this iteration. Stock rule (multiplexing_mixin.py:170-178): budget // extend tokens, at least 1, and every
    layer at once for a 0-token batch. Here the tokens are the GLOBAL max, so every rank (idle splits included)
    runs the same [split_index, split_index + count)."""
    m = max(global_tokens) if global_tokens else 0
    count = max(1, budget // m) if m > 0 else num_layers
    return min(split_index + count, num_layers) - split_index


def choose_stream_idx(n_groups: int, split_active: bool, decode_active: bool, max_decode_bs: int,
                      manual_divisions: list, decode_bs_divisor: int) -> int:
    """S2: the stock partition rule (adjust_stream_groups, multiplexing_mixin.py:50-79) evaluated on GLOBAL state, so
    every rank picks the same group. Groups: 0 = prefill only (all SMs), n-1 = decode only (all SMs),
    1..n-2 = green-context splits. decode_active / max_decode_bs come from the D lane's gathered token counts.
    Stock leaves the index undefined when manual thresholds are all above the batch; this rule uses the first split."""
    if split_active and decode_active:
        if manual_divisions:
            idx = 1
            for i, (_p, _d, threshold) in enumerate(manual_divisions):
                if max_decode_bs >= threshold:
                    idx = i + 1
            return idx
        return max(1, min(n_groups - 2, max_decode_bs * (n_groups - 2) // max(1, decode_bs_divisor)))
    if decode_active:
        return n_groups - 1
    return 0


def global_decode_state(dbatch) -> tuple[bool, int]:
    """(any rank decodes, the largest per-rank decode token count) from a D-lane batch after the MLP sync."""
    if dbatch is None:
        return False, 0
    tokens = list(getattr(dbatch, "global_num_tokens", None) or [])
    if not tokens:
        n = 0 if dbatch.forward_mode.is_idle() else dbatch.batch_size()
        return n > 0, n
    return max(tokens) > 0, max(tokens)


# ------------------------------------------------------------------------------------------------- scheduler glue
def dp_sync(sched, batch):
    """The MLP sync of dp_attn.prepare_mlp_sync_batch_raw (dp_attn.py:356-519) on the CPU group. The stock non-overlap
    path gathers on the TP device group (:432-438), which would put a main-communicator NCCL op on whichever stream is
    current. pdmux has two streams, so the gather stays on gloo. Returns the local batch, an IDLE batch when another
    rank has work, or None when no rank has work. All ranks agree."""
    from sglang.srt.managers.scheduler_components import dp_attn as D
    from sglang.srt.runtime_context import get_parallel

    a = sched.dp_attn_adapter
    return D.prepare_mlp_sync_batch_raw(
        batch,
        model_runner=a.model_runner,
        dp_size=get_parallel().dp_size,
        attn_tp_size=a.ps.attn_tp_size,
        attn_cp_size=a.ps.attn_cp_size,
        tp_group=a.tp_group,
        get_idle_batch=a.get_idle_batch,
        disable_cuda_graph=D.cuda_graph_fully_disabled(),
        require_mlp_tp_gather=D.require_mlp_tp_gather(),
        disable_overlap_schedule=False,     # -> tp_group.cpu_group, device "cpu" (dp_attn.py:439-441)
        offload_tags=a.offload_tags,
        dwdp=get_parallel().dwdp_size > 1,
    )


def stash_chunked(sched) -> None:
    """S3(a), as get_next_batch_to_run does (and k3b300's K3 port 4.(a)): the previous chunk of the chunked request is
    cached (radix insert) before the next split is planned."""
    cr = sched.chunked_req
    if cr is not None and cr.extend_range.end > len(cr.prefix_indices):
        sched.stash_chunked_request(cr)


def plan_split(sched, running):
    """S1: plan the next split on every rank. A local proposal (get_new_batch_prefill) goes through one CPU-group sync.
    A rank with a prefill gets its batch as SPLIT_PREFILL, a rank without one an IDLE split
    (_numinous_idle_split = True; it runs forward_split_prefill with 0 rows over the same layers), and None means no
    rank prefills. Returns (running_batch, split or None)."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    stash_chunked(sched)
    plan = sched.get_new_batch_prefill(running)
    running = plan.running_batch
    local = plan.batch_to_run
    if local is not None and local.is_empty():
        local = None
    if local is not None and not local.forward_mode.is_extend():
        raise RuntimeError(f"numinous pdmux-dp: the prefill plan returned {local.forward_mode}; mixed chunk must be "
                           "off under pdmux (the decode requests would ride the split)")
    split = dp_sync(sched, local)
    if split is None:
        return running, None
    if split.forward_mode.is_idle():
        split._numinous_idle_split = True
    else:
        split.forward_mode = ForwardMode.SPLIT_PREFILL
        split._numinous_idle_split = False
    split.split_index = 0
    split.split_prefill_finished = False
    split.split_forward_batch = None
    return running, split


def adopt_on_stream(batch, stream) -> int:
    """record_stream(stream) on every CUDA tensor a batch (and its spec_info) holds. Returns the count. The caching
    allocator then keeps P-stream blocks alive until the D stream's queued work that reads them is done."""
    import torch

    n = 0
    for obj in (batch, getattr(batch, "spec_info", None), getattr(batch, "sampling_info", None)):
        if obj is None:
            continue
        for v in list(getattr(obj, "__dict__", {}).values()):
            if isinstance(v, torch.Tensor) and v.is_cuda:
                v.record_stream(stream)
                n += 1
    return n


def finish_split(sched, running, split, result, decode_stream):
    """S4 + S3(b) + the merge, on the D lane, on every rank in the same iteration.
    - Speculative: the finished target result is stashed on the split. The stock non-overlap spec path
      (scheduler.run_batch -> EAGLEWorkerV2.forward_batch_generation, eagle_worker_v2.py:1263-1309) then runs the
      prefill draft-extend unchanged: the target worker returns the stash instead of re-running the target
      (glm_pdmux_dp_hooks tp_worker hook). Idle-split ranks take the same path with an IDLE batch
      (is_extend_in_batch is set by the plan's sync), so the draft-extend's DP gathers line up.
    - Non-speculative: the split's own result is the batch result.
    - The unfinished chunked request never joins the decode batch (filter_batch(chunked_req_to_exclude)).
    Returns the new running batch."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    idle = bool(getattr(split, "_numinous_idle_split", False))
    if not split.spec_algorithm.is_none():
        split.forward_mode = ForwardMode.IDLE if idle else ForwardMode.EXTEND
        split._numinous_split_result = result
        result = sched.run_batch(split)                 # draft-extend for the prefill on the D lane
        split._numinous_split_result = None             # consumed by the tp_worker hook; never reused by a decode
    split.split_forward_batch = None                    # release the split's ForwardBatch (hidden states, logits)
    if idle:
        return running
    split.forward_mode = ForwardMode.EXTEND
    sched.process_batch_result(split, result)
    excl = [r for r in (sched.chunked_req, getattr(split, "chunked_req", None)) if r is not None]
    if excl:
        split.filter_batch(chunked_req_to_exclude=excl)
    if split.is_empty():
        return running
    if running is not None and not running.is_empty():
        running.merge_batch(split)
        return running
    adopt_on_stream(split, decode_stream)
    return split


# ---------------------------------------------------------------------------------------------------- the trace
class LoopTrace:
    """Per-rank pdmux loop trace (coordinator 2026-10-06: the run must show whether the decode-idle wall closed).
    On when NUMINOUS_SCHED_TRACE is 1/true; one JSON line per iteration in
    $NUMINOUS_SCHED_TRACE_DIR/pdmux_rank<tp>.jsonl. verify_registry gzips every *.jsonl of the trace dir into the
    cell's evidence; operations/hopper/pdmux_idle_readout.py reads it.
    Fields:
    - it, t0 (wall s), dur (s, to the next iteration);
    - running / waiting (local requests);
    - dmode: the D-lane batch, DECODE / IDLE / null;
    - d_s: the decode step from run_batch to its processed result (the D-stream wait included);
    - d_gpu_ms: the decode step's GPU time on the D stream (CUDA events, read after the loop's own sync);
    - split_s: the CPU time to enqueue this iteration's prefill split;
    - sync_s: the D-lane MLP sync;
    - plan_s: the P-lane plan sync, if this iteration planned;
    - split: [index, count, idle] if a split was enqueued;
    - finish_s: the hand-off, if it ran;
    - gdec: the global max decode tokens per rank;
    - group: the SM stream group.
    Idle iterations (no D batch, no split, nothing running) coalesce into {"idle_span": n, "idle_s": s}."""

    def __init__(self, path: Optional[str]):
        self.path, self.buf, self.last_flush, self.rec, self.idle_n, self.idle_s = path, [], 0.0, None, 0, 0.0

    @classmethod
    def open(cls, sched) -> "LoopTrace":
        if str(os.environ.get("NUMINOUS_SCHED_TRACE", "")).lower() not in ("1", "true"):
            return cls(None)
        d = os.environ.get("NUMINOUS_SCHED_TRACE_DIR") or "/tmp/numinous_sched_trace"
        os.makedirs(d, exist_ok=True)
        t = cls(os.path.join(d, f"pdmux_rank{int(sched.ps.tp_rank)}.jsonl"))   # v0.5.20: rank lives on ps
        import atexit
        atexit.register(t.flush, True)
        return t

    def begin(self, it: int, t0: float) -> None:
        if self.path is None:
            return
        self.end(t0)
        self.rec = {"it": it, "t0": round(t0, 6)}

    def put(self, **kv) -> None:
        if self.rec is not None:
            self.rec.update(kv)

    def end(self, t_next: float) -> None:
        """Close the open record at the next iteration's start (dur covers the whole iteration)."""
        r, self.rec = self.rec, None
        if r is None:
            return
        dur = t_next - r["t0"]
        if r.get("dmode") is None and r.get("split") is None and not r.get("running") and r.get("finish_s") is None:
            self.idle_n, self.idle_s = self.idle_n + 1, self.idle_s + dur
            return
        if self.idle_n:
            self.buf.append({"idle_span": self.idle_n, "idle_s": round(self.idle_s, 6)})
            self.idle_n, self.idle_s = 0, 0.0
        r["dur"] = round(dur, 6)
        self.buf.append(r)
        if t_next - self.last_flush > 0.5:
            self.flush()
            self.last_flush = t_next

    def flush(self, final: bool = False) -> None:
        if self.path is None or not self.buf:
            return
        import json
        with open(self.path, "a") as f:
            f.write("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in self.buf))
        self.buf = []


# ----------------------------------------------------------------------------------------------------- the loop
def event_loop_pdmux_dp(sched) -> None:
    """The pdmux loop for DP attention (replaces multiplexing_mixin.event_loop_pdmux when enabled())."""
    import torch

    from sglang.srt.distributed.parallel_state import set_pdmux_status
    from sglang.srt.multiplex.pdmux_context import get_current_stream_idx, set_current_stream_idx

    num_layers = sched.model_config.num_hidden_layers
    budget = sched.pdmux_config.split_forward_token_budget
    stream_idx = get_current_stream_idx()
    prefill_stream, decode_stream = sched.stream_groups[stream_idx]
    waiting = False
    prefill_result = None
    prefill_done_event = None
    regroup = True
    torch.cuda.empty_cache()
    logger.info("numinous pdmux-dp: DP-synchronized split-prefill loop (%d stream groups)", len(sched.stream_groups))
    import time
    clock = time.perf_counter
    trace = LoopTrace.open(sched)
    it = 0

    try:
        while True:
            trace.begin(it, clock())
            it += 1
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                sched.ingest_requests()
                # get_next_batch_to_run's first two steps (scheduler.py:3565-3566), which the stock pdmux loop never
                # runs: a pending chunked abort, and the HiCache drain (TP-wide consensus, so every rank at the same
                # point each iteration; without it HiCache load/write-back completions are never processed)
                sched.process_pending_chunked_abort()
                sched._process_hicache_events()
                running = sched.running_batch
            trace.put(running=0 if running is None else running.batch_size(), group=stream_idx,
                      waiting=len(getattr(sched, "waiting_queue", ()) or ()))

            created = False
            if sched.split_prefill_batch is None:
                t = clock()
                with torch.cuda.stream(prefill_stream):
                    set_pdmux_status(True)
                    running, split = plan_split(sched, running)
                sched.running_batch = running
                trace.put(plan_s=round(clock() - t, 6))
                if split is not None:
                    sched.split_prefill_batch = split
                    created = True

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                running = sched.update_running_batch(running)
                sched.running_batch = running
                t = clock()
                dbatch = dp_sync(sched, running if (running is not None and not running.is_empty()) else None)
                trace.put(sync_s=round(clock() - t, 6),
                          dmode=None if dbatch is None else ("IDLE" if dbatch.forward_mode.is_idle() else "DECODE"),
                          gdec=global_decode_state(dbatch)[1])

            if created or regroup:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                active, max_bs = global_decode_state(dbatch)
                stream_idx = choose_stream_idx(len(sched.stream_groups), sched.split_prefill_batch is not None, active,
                                               max_bs, sched.pdmux_config.manual_divisions,
                                               sched.pdmux_config.decode_bs_divisor)
                set_current_stream_idx(stream_idx)
                sched.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
                prefill_stream, decode_stream = sched.stream_groups[stream_idx]
                regroup = False

            t_dec = clock()
            ev = None
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                if dbatch is not None and trace.path is not None and torch.cuda.is_available():
                    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    ev[0].record(decode_stream)
                decode_result = sched.run_batch(dbatch) if dbatch is not None else None
                if ev is not None:
                    ev[1].record(decode_stream)

            split = sched.split_prefill_batch
            if split is not None and not waiting:
                with torch.cuda.stream(prefill_stream):
                    set_pdmux_status(True)
                    count = split_forward_count(list(split.global_num_tokens or [0]), budget, num_layers,
                                                split.split_index)
                    split.split_forward_count = count
                    t = clock()
                    result = sched.tp_worker.forward_batch_split_prefill(split)
                    trace.put(split=[split.split_index, count, bool(getattr(split, "_numinous_idle_split", False))],
                              split_s=round(clock() - t, 6))
                    split.split_index += count
                    if split.split_index >= num_layers:
                        split.split_prefill_finished = True
                        prefill_result = result
                        prefill_done_event = prefill_stream.record_event()

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_stream.synchronize()
                if dbatch is not None:
                    sched.process_batch_result(dbatch, decode_result)
                    running = sched.running_batch
                    trace.put(d_s=round(clock() - t_dec, 6))
                    if ev is not None:
                        trace.put(d_gpu_ms=round(ev[0].elapsed_time(ev[1]), 3))

            split = sched.split_prefill_batch
            if split is not None and split.split_prefill_finished:
                waiting = True
                flags = torch.ones(1, dtype=torch.int32) if prefill_done_event.query() else torch.zeros(1, dtype=torch.int32)
                sched.tp_cpu_group.allreduce(flags, torch.distributed.ReduceOp.SUM).wait()
                if int(flags.item()) == sched.ps.tp_size:
                    t = clock()
                    with torch.cuda.stream(decode_stream):
                        set_pdmux_status(False)
                        decode_stream.wait_event(prefill_done_event)
                        running = finish_split(sched, sched.running_batch, split, prefill_result, decode_stream)
                    trace.put(finish_s=round(clock() - t, 6))
                    sched.running_batch = running
                    sched.split_prefill_batch = None
                    prefill_result = prefill_done_event = None
                    waiting = False
                    regroup = True

            if dbatch is None and sched.split_prefill_batch is None and (
                    sched.running_batch is None or sched.running_batch.is_empty()):
                sched.on_idle()
    finally:                                         # a crash or a stop keeps the last records
        trace.end(clock())
        trace.flush(True)
