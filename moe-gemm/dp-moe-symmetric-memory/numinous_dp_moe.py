"""Fused DP-attention / TP-expert MoE layer for DeepSeek V4 Pro (and any DeepseekV2MoE with MXFP4 TRT-LLM experts).

What SGLang v0.5.18 does per MoE layer when attention is data-parallel and experts are tensor-parallel:

  1. gather every GPU's tokens in bf16 to every GPU (prefill: an all-reduce over a zero-filled buffer, because NCCL
     all-gather needs equal sizes and prefill steps are skewed; decode: an all-gather over MAX_LEN padding)
  2. on the full gathered buffer, on every GPU: the router GEMM + top-k, the replicated shared expert, the MXFP8
     quantization of the expert input (all 8x redundant)
  3. routed experts (each GPU holds 1/8 of every expert's intermediate dimension)
  4. scale the partial output by routed_scaling_factor, all-reduce it at full width, copy this GPU's rows out, add
     the shared expert output

What this module does instead:

  1. on this GPU's own tokens only: router + top-k, shared expert, MXFP8 quantization
  2. push the MXFP8 rows, their e8m0 scales and the top-k ids/weights (7,440 bytes per token instead of 14,336)
     straight into every peer's gathered buffer (symm_collectives.push_rows): variable lengths are free, so there
     is no zero-fill and no reduction on the gather side
  3. routed experts on the gathered, already-quantized rows, writing the partial output into a symmetric buffer
  4. pull this GPU's rows from all eight partials, scale, add the local shared output, write the layer output
     (symm_collectives.pull_reduce_rows): one pass over 1/8 of the data replaces the scale multiply, the full-width
     all-reduce, the scatter copy and the shared add

Per layer and GPU at a balanced 4,096-token prefill chunk that is ~0.21 GB + 0.41 GB of NVLink traffic instead of
~1.6 GB, and the shared expert computes 4,096 tokens instead of 32,768. The math is the same up to the order of
the TP partial sum (fixed peer order, identical on every rank).

Enabled with NUMINOUS_DP_MOE=1; any layer or step it does not cover falls back to the SGLang path.
NUMINOUS_DP_MOE_VERIFY=<n layers>: on eager steps, also run the layer's stock MoE forward on a stock-identical
gathered buffer and log per-layer fused-vs-stock differences (maybe_verify below; the NVFP4 module reuses it).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

try:  # the overlay installs both files into the same package
    from sglang.srt.layers.moe.symm_collectives import (pull_reduce_rows, pull_reduce_rows_ptrs, pull_reduce_v,
                                                         push_payloads, push_rows, push_rows_ptrs)
except ImportError:  # tests: run from operations/kernels
    from symm_collectives import (pull_reduce_rows, pull_reduce_rows_ptrs, pull_reduce_v, push_payloads, push_rows,
                                  push_rows_ptrs)

MXFP8_GROUP = 32


def enabled() -> bool:
    return os.environ.get("NUMINOUS_DP_MOE", "0") == "1"


# ---------------------------------------------------------------------------------------------------------------
# Symmetric buffers: one set per process, sized once for the largest step, rendezvoused over the TP group.
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class DpMoeBuffers:
    world: int
    rank: int
    xq: list          # per peer [rows_cap, hidden] uint8 (MXFP8 e4m3 bytes)
    xs: list          # per peer [rows_cap, hidden / 32] uint8 (e8m0 scales)
    ids: list         # per peer [rows_cap, topk] int32
    wts: list         # per peer [rows_cap, topk] float32
    part: list        # per peer [rows_cap, hidden] bfloat16 (TP-partial routed output)
    barrier: Callable[[], None]
    # GPU: each buffer's peer base-address array (symmetric memory buffer_ptrs_dev); Triton rejects a peer mapping
    # passed as a tensor argument, so the kernels load the peers' pointers themselves. Empty on CPU (tests).
    ptrs: Optional[dict] = None
    # the symmetric-memory handles themselves: `ptrs` holds raw device addresses of arrays each handle owns, so a
    # handle collected after buffers() returns leaves a dangling address (an illegal memory access in the first push)
    handles: tuple = ()
    # each buffer's NVLink multicast address (0 where the fabric has none). Measured per layer on 8xB200 with the
    # barrier that follows and FlashInfer loaded (gpu_test_symm_v.py, 2 runs): decode (96-row block) push 25.7 us
    # old -> 20.6 vectorized unicast (multicast 25.0), pull 39.1 -> 32.0 switch-reduced; prefill (4,096 rows) push
    # 462 -> 357 multicast, pull 660 old = best (the exact fp32 unicast sum stays). phase_push / phase_pull pick
    # accordingly by step type.
    mc: Optional[dict] = None

    @property
    def rows_cap(self) -> int:
        return self.xq[self.rank].shape[0]


_BUFFERS: Optional[DpMoeBuffers] = None


def buffers(group, rows_cap: int, hidden: int, topk: int, device: torch.device) -> DpMoeBuffers:
    """Allocate and rendezvous the symmetric buffers on first use (a collective: every TP rank reaches its first
    fused MoE layer together). Must not run inside CUDA-graph capture."""
    global _BUFFERS
    if _BUFFERS is not None:
        if _BUFFERS.rows_cap < rows_cap:
            raise RuntimeError(f"numinous_dp_moe: step needs {rows_cap} rows, buffers hold {_BUFFERS.rows_cap}")
        if _BUFFERS.ids[0].shape[1] != topk:
            raise RuntimeError(f"numinous_dp_moe: layer routes top-{topk}, buffers hold top-{_BUFFERS.ids[0].shape[1]}")
        return _BUFFERS
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("numinous_dp_moe: buffers must be allocated before CUDA-graph capture")
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    name = group.group_name
    if hasattr(symm_mem, "enable_symm_mem_for_group"):
        symm_mem.enable_symm_mem_for_group(name)
    world, rank = dist.get_world_size(group), dist.get_rank(group)

    # one symmetric allocation carved into the five buffers at 2 MiB-aligned offsets: one handle, one rendezvous,
    # one multicast object, so every peer and multicast address is base + an offset chosen here (separate small
    # allocations produced multicast stores that landed wrong for the 224-byte scale rows in composition)
    align = 2 << 20
    specs = (("xq", (rows_cap, hidden), torch.uint8), ("xs", (rows_cap, hidden // MXFP8_GROUP), torch.uint8),
             ("ids", (rows_cap, topk), torch.int32), ("wts", (rows_cap, topk), torch.float32),
             ("part", (rows_cap, hidden), torch.bfloat16))
    offs, total = {}, 0
    for nm, shape, dtype in specs:
        offs[nm] = total
        nbytes = shape[0] * shape[1] * torch.empty(0, dtype=dtype).element_size()
        total += (nbytes + align - 1) // align * align
    raw = symm_mem.empty((total,), dtype=torch.uint8, device=device)
    raw.zero_()
    # zeroed: in MAX_LEN steps each rank pushes only its real rows, so the padding rows of the gathered buffer keep
    # what memory held, and the expert kernel indexes experts by those top-k ids (uninitialised ids were an illegal
    # memory access in CUDA-graph warm-up). Zeros are expert 0 at weight 0; later, earlier real rows.
    h = symm_mem.rendezvous(raw, name)
    views, ptr_arrays = {}, {}
    for nm, shape, dtype in specs:
        esz = torch.empty(0, dtype=dtype).element_size()
        views[nm] = [h.get_buffer(r, shape, dtype, offs[nm] // esz) for r in range(world)]
        ptr_arrays[nm] = torch.tensor([int(h.buffer_ptrs[r]) + offs[nm] for r in range(world)], dtype=torch.int64,
                                      device=device)
    ptrs = {nm: int(t.data_ptr()) for nm, t in ptr_arrays.items()}
    use_mc = os.environ.get("NUMINOUS_DP_MOE_MC", "1") == "1"
    base_mc = int(getattr(h, "multicast_ptr", 0) or 0) if use_mc else 0
    mc = {nm: (base_mc + offs[nm] if base_mc else 0) for nm, _, _ in specs}
    _BUFFERS = DpMoeBuffers(world, rank, views["xq"], views["xs"], views["ids"], views["wts"], views["part"],
                            barrier=lambda: h.barrier(channel=0), ptrs=ptrs, handles=(h, raw, ptr_arrays), mc=mc)
    torch.cuda.synchronize(device)
    _BUFFERS.barrier()                          # every rank's zeros are in place before any peer pushes
    return _BUFFERS


def prepare(hidden: int, topk: int) -> None:
    """Allocate and rendezvous the symmetric buffers at model load, before any CUDA-graph capture: SGLang captures
    decode graphs before serving, so the first fused layer would otherwise run inside capture. Every TP rank loads
    weights together, which is what the rendezvous needs."""
    from sglang.srt.distributed import get_tp_group
    cap = int(os.environ.get("NUMINOUS_DP_MOE_ROWS", "40960"))
    buffers(get_tp_group().device_group, cap, hidden, topk, torch.device("cuda", torch.cuda.current_device()))
    log_once("prepared", f"numinous_dp_moe: symmetric buffers ready ({cap} rows x {hidden})")


# ---------------------------------------------------------------------------------------------------------------
# The three phases. The live forward runs them back to back with a barrier after push and after experts; tests run
# them for every simulated rank phase by phase.
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class LocalPrep:
    xq: torch.Tensor          # [n, hidden] float8_e4m3fn
    xs: torch.Tensor          # [n, hidden / 32] uint8 (e8m0)
    topk_ids: torch.Tensor    # [n, topk] int32
    topk_w: torch.Tensor      # [n, topk] float32
    shared: Optional[torch.Tensor]  # [n, hidden] bf16


def phase_push(prep: LocalPrep, buf: DpMoeBuffers, row_off: int, multicast: bool = True, max_ctas: int = 0) -> None:
    """multicast: prefill steps (large pushes) go through the multicast mapping where the fabric has one; decode
    blocks are faster as vectorized unicast (see DpMoeBuffers.mc)."""
    items = (("xq", prep.xq.view(torch.uint8), buf.xq), ("xs", prep.xs, buf.xs), ("ids", prep.topk_ids, buf.ids),
             ("wts", prep.topk_w, buf.wts))
    if buf.ptrs:  # GPU: every payload in one launch; 16-byte rows through the multicast mapping where there is one
        mc = buf.mc or {}
        pl = [(src, buf.ptrs[name], mc.get(name, 0), views[buf.rank].shape[1] * views[buf.rank].element_size())
              for name, src, views in items]
        push_payloads(pl, buf.world, prep.xq.shape[0], row_off, multicast=multicast and any(mc.values()),
                      max_ctas=max_ctas)
        return
    for name, src, views in items:
        if buf.ptrs:
            width = views[buf.rank].shape[1] * views[buf.rank].element_size()
            push_rows_ptrs(src, buf.ptrs[name], buf.world, width, row_off)
        else:
            push_rows(src, views, row_off)


def phase_experts(buf: DpMoeBuffers, rows: int, experts_fn: Callable, start: int = 0) -> None:
    """experts_fn(xq [rows, H] fp8, xs [rows, H/32] uint8, ids, weights, out [rows, H] bf16) writes this GPU's
    TP-partial routed output (unscaled) for gathered rows [start, start + rows)."""
    if rows == 0:
        return
    r, e = buf.rank, start + rows
    experts_fn(buf.xq[r][start:e].view(torch.float8_e4m3fn), buf.xs[r][start:e], buf.ids[r][start:e],
               buf.wts[r][start:e], buf.part[r][start:e])


_STREAMS: dict = {}


def overlapped(prep: LocalPrep, buf: DpMoeBuffers, counts: list, rank: int, experts_fn: Callable, out: torch.Tensor,
               scale: float) -> torch.Tensor:
    """Prefill (SUM_LEN, eager) with communication overlapped by expert compute. Every rank splits its rows in two;
    the gathered buffer holds all first halves (region A, packed by rank) then all second halves (region B):

        push A, barrier | experts(A) on the compute stream  || push B, barrier
        barrier (every rank finished experts A) | experts(B) || pull A
        barrier (every rank finished experts B) | pull B

    Same arithmetic as the sequential path (same rows, same fixed peer order in each pull)."""
    n = counts[rank]
    h0 = [c // 2 for c in counts]
    h1 = [c - a for c, a in zip(counts, h0)]
    rows_a, rows_b = sum(h0), sum(h1)
    off_a, off_b = sum(h0[:rank]), rows_a + sum(h1[:rank])
    dev = out.device
    comp = _STREAMS.setdefault(dev, torch.cuda.Stream(device=dev))
    cur = torch.cuda.current_stream(dev)

    def half(lo, hi):
        return LocalPrep(prep.xq[lo:hi], prep.xs[lo:hi], prep.topk_ids[lo:hi], prep.topk_w[lo:hi],
                         prep.shared[lo:hi] if prep.shared is not None else None)

    a, b = half(0, h0[rank]), half(h0[rank], n)
    ctas = int(os.environ.get("NUMINOUS_DP_MOE_COMM_CTAS", "32"))
    phase_push(a, buf, off_a, multicast=True)
    buf.barrier()
    comp.wait_stream(cur)
    with torch.cuda.stream(comp):
        phase_experts(buf, rows_a, experts_fn, start=0)
        done_a = torch.cuda.Event()
        done_a.record(comp)
    phase_push(b, buf, off_b, multicast=True, max_ctas=ctas)     # runs alongside experts(A)
    buf.barrier()
    comp.wait_stream(cur)
    with torch.cuda.stream(comp):
        phase_experts(buf, rows_b, experts_fn, start=rows_a)
        done_b = torch.cuda.Event()
        done_b.record(comp)
    cur.wait_event(done_a)
    buf.barrier()                                    # every rank's region-A partials are complete
    if h0[rank]:
        phase_pull(buf, off_a, h0[rank], out[:h0[rank]], scale, a.shared, max_ctas=ctas)  # alongside experts(B)
    cur.wait_event(done_b)
    buf.barrier()                                    # every rank's region-B partials are complete
    if h1[rank]:
        phase_pull(buf, off_b, h1[rank], out[h0[rank]:n], scale, b.shared)
    return out


def phase_pull(buf: DpMoeBuffers, row_off: int, n_local: int, out: torch.Tensor, scale: float,
               shared: Optional[torch.Tensor], valid: Optional[torch.Tensor] = None, max_ctas: int = 0) -> torch.Tensor:
    if buf.ptrs and max_ctas:  # overlapped prefill: a capped grid leaves the SMs to the expert GEMMs
        return pull_reduce_v(buf.ptrs["part"], buf.world, row_off, n_local, out, scale=scale, shared=shared,
                             valid=valid, max_ctas=max_ctas)
    if buf.ptrs:
        mc = (buf.mc or {}).get("part", 0)
        if mc and valid is not None:  # decode block (CUDA graph): the switch sums the eight partials
            return pull_reduce_v(mc, buf.world, row_off, n_local, out, scale=scale, shared=shared, valid=valid,
                                 multicast=True)
        return pull_reduce_rows_ptrs(buf.ptrs["part"], buf.world, row_off, n_local, out, scale=scale, shared=shared,
                                     valid=valid)
    return pull_reduce_rows(buf.part, row_off, n_local, out, scale=scale, shared=shared, valid=valid)


# ---------------------------------------------------------------------------------------------------------------
# In-engine verify mode (NUMINOUS_DP_MOE_VERIFY=<n layers>): after the fused path produced a step's output, rebuild
# the gathered bf16 buffer exactly as stock prepare_mlp would (equal-size all_gather, pads zeroed), run the layer's
# own stock MoE forward on it (SGLang's router, top-k / bypassed routing, expert kernel, TP all-reduce, shared
# expert: the layer's real weights), and log the difference for this rank's rows. Eager steps only: replayed CUDA
# graphs never re-enter Python and capture is skipped, so the mode sees the profile run and prefill / extend steps.
# Every TP rank takes the same branch (same env, same per-layer step counters), so the reference's collectives stay
# lockstep. First NUMINOUS_DP_MOE_VERIFY covered MoE layers in first-seen order, NUMINOUS_DP_MOE_VERIFY_STEPS
# (default 3) covered eager steps each.
# ---------------------------------------------------------------------------------------------------------------
_VERIFY_SEEN: dict = {}      # id(mlp) -> first-seen index
_VERIFY_DONE: dict = {}      # id(mlp) -> eager steps verified


def verify_layers() -> int:
    return int(os.environ.get("NUMINOUS_DP_MOE_VERIFY", "0") or "0")


def pack_gathered(parts: list, counts: list) -> torch.Tensor:
    """The SUM_LEN gathered layout from equal-padded per-rank tensors: each rank's first counts[r] rows, packed
    back to back in rank order (the row offsets layout() computes)."""
    return torch.cat([p[:c] for p, c in zip(parts, counts)], dim=0)


def diff_stats(fused: torch.Tensor, ref: torch.Tensor, parts: Optional[dict] = None) -> dict:
    """One rank's fused rows against the stock reference: worst element, l2 relative error, the least-squares
    scale alpha (a constant-factor bug reads directly: alpha 2.5 = routed scale applied twice with scale 2.5), and
    per named part (e.g. the shared expert output) the projection beta of the error onto it (+1: the error is one
    extra copy of that part; -1: one missing copy; 0: unrelated)."""
    f, r = fused.float(), ref.float()
    err = f - r
    out = {"rows": int(f.shape[0]), "ref_absmax": r.abs().max().item(), "err_absmax": err.abs().max().item(),
           "rel_l2": (err.norm() / r.norm().clamp_min(1e-30)).item(),
           "alpha": ((f * r).sum() / (r * r).sum().clamp_min(1e-30)).item()}
    for nm, p in (parts or {}).items():
        p = p.float()
        out[f"beta_{nm}"] = ((err * p).sum() / (p * p).sum().clamp_min(1e-30)).item()
    return out


def _verify_gather(x_local: torch.Tensor, n: int, forward_batch, rows: int) -> torch.Tensor:
    """The gathered buffer stock prepare_mlp would build, from this rank's first n real rows: MAX_LEN as world
    equal blocks (pads zeroed; row independence of the MoE keeps them harmless), SUM_LEN packed by rank."""
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group
    group = get_tp_group().device_group
    world = dist.get_world_size(group)
    flat = x_local.dim() == 1
    x2 = x_local.reshape(x_local.shape[0], 1) if flat else x_local
    if forward_batch.dp_padding_mode.is_max_len():
        block = rows // world
        send = x2.new_zeros(block, x2.shape[1])
        send[:min(n, block)] = x2[:min(n, block)]
        out = x2.new_empty(world * block, x2.shape[1])
        dist.all_gather_into_tensor(out, send, group=group)
    else:
        counts = [int(c) for c in forward_batch.global_num_tokens_cpu]
        m = max(counts + [1])
        send = x2.new_zeros(m, x2.shape[1])
        send[:n] = x2[:n]
        peers = [torch.empty_like(send) for _ in range(world)]
        dist.all_gather(peers, send, group=group)
        out = pack_gathered(peers, counts)
    return out.reshape(-1) if flat else out


def _stock_reference(mlp, gathered: torch.Tensor, input_ids_global: Optional[torch.Tensor]) -> torch.Tensor:
    """The stock MoE block on the gathered rows: DeepseekV2MoE.forward_normal (the eager path the layer runs when
    the fused path declines), bypassing only the mega / a2a / dual-stream dispatch the coverage checks exclude."""
    if getattr(mlp, "is_hash", False):
        return mlp.forward_normal(gathered, input_ids_global=input_ids_global)
    return mlp.forward_normal(gathered)


def maybe_verify(mlp, hidden_local: torch.Tensor, out_fused: torch.Tensor, forward_batch, row_off: int, n: int,
                 rows: int, dp_rank: int, input_ids: Optional[torch.Tensor] = None,
                 tag: str = "numinous_dp_moe") -> None:
    """Log max / mean difference of the fused output against the stock computation for this rank's real rows.
    No-op unless NUMINOUS_DP_MOE_VERIFY is set; skipped under CUDA-graph capture. All collectives (the gather, the
    reference's TP all-reduce) run before any rank-local early return, so ranks stay lockstep."""
    k = verify_layers()
    if not k:
        return
    if hidden_local.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    key = id(mlp)
    if _VERIFY_SEEN.setdefault(key, len(_VERIFY_SEEN)) >= k:
        return
    done = _VERIFY_DONE.get(key, 0)
    if done >= int(os.environ.get("NUMINOUS_DP_MOE_VERIFY_STEPS", "3")):
        return
    _VERIFY_DONE[key] = done + 1
    gathered = _verify_gather(hidden_local, n, forward_batch, rows)
    ids_global = None
    if input_ids is not None and getattr(mlp, "is_hash", False):
        ids_global = _verify_gather(input_ids, n, forward_batch, rows)
    ref = _stock_reference(mlp, gathered, ids_global)
    log = logging.getLogger("sglang.numinous")
    mode = "max_len" if forward_batch.dp_padding_mode.is_max_len() else "sum_len"
    real = n
    if mode == "max_len":
        cpu = getattr(forward_batch, "global_num_tokens_cpu", None)
        if cpu is not None:
            real = min(n, int(cpu[dp_rank]))
    lid = getattr(mlp, "layer_id", "?")
    if real == 0:
        log.warning("%s verify layer %s step %d rank %d %s: idle (0 real rows)", tag, lid, done, dp_rank, mode)
        return
    parts = {}
    if getattr(mlp, "shared_experts", None) is not None and not getattr(mlp, "num_fused_shared_experts", 0):
        shared = mlp._forward_shared_experts(hidden_local[:real])
        if shared is not None:
            ref_rows = ref[row_off:row_off + real]
            parts = {"shared": shared, "routed": ref_rows.float() - shared.float()}
    st = diff_stats(out_fused[:real], ref[row_off:row_off + real], parts)
    log.warning("%s verify layer %s step %d rank %d %s rows %d: rel_l2 %.3e err_absmax %.3e (ref absmax %.3e) "
                "alpha %.6f%s", tag, lid, done, dp_rank, mode, st["rows"], st["rel_l2"], st["err_absmax"],
                st["ref_absmax"], st["alpha"],
                "".join(f" beta_{nm} {st[f'beta_{nm}']:.4f}" for nm in ("shared", "routed") if f"beta_{nm}" in st))


# ---------------------------------------------------------------------------------------------------------------
# SGLang glue (GPU only): local routing and quantization, the pre-quantized TRT-LLM MXFP4 expert call, the hook.
# ---------------------------------------------------------------------------------------------------------------
_LOGGED: set = set()


def log_once(key, msg: str) -> None:
    """One line per process per decision, at warning level so it reaches the server log: a declined fused path
    must never pass for a measured one."""
    if key not in _LOGGED:
        _LOGGED.add(key)
        logging.getLogger("sglang.numinous").warning(msg)


def decline_reason(mlp, forward_batch) -> str:
    """Empty when the fused path covers this layer; otherwise why not."""
    try:
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import Mxfp4FlashinferTrtllmMoEMethod
    except ImportError:
        return "SGLang's MXFP4 TRT-LLM MoE method is unavailable"
    qm = getattr(getattr(mlp, "experts", None), "quant_method", None)
    if not isinstance(qm, Mxfp4FlashinferTrtllmMoEMethod):
        return f"experts use {type(qm).__name__}, not the MXFP4 TRT-LLM method"
    if getattr(qm, "flashinfer_mxfp4_moe_precision", "default") != "default":
        return f"MXFP4 MoE precision {qm.flashinfer_mxfp4_moe_precision}"
    if getattr(mlp, "shared_experts", None) is not None and not getattr(mlp, "_shared_expert_tp1", False):
        # a TP-sharded shared expert computes 1/tp of its output per GPU: adding it locally would be wrong
        return "the shared expert is TP-sharded (serve with SGLANG_SHARED_EXPERT_TP1=1)"
    if forward_batch.dp_padding_mode is None:
        return "the batch has no DP padding mode"
    return ""


def covers(mlp, forward_batch) -> bool:
    """The fused path handles DP attention over TP experts with SGLang's MXFP4 TRT-LLM method, tp == dp, and a
    replicated (TP1) shared expert."""
    reason = decline_reason(mlp, forward_batch)
    if reason:
        log_once(("declined", reason), f"numinous_dp_moe declined: {reason}")
        return False
    log_once("active", "numinous_dp_moe active: fused DP-MoE serves the MoE layers")
    return True


def _mxfp8(x: torch.Tensor):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import _MXFP8_QUANTIZE_BACKEND
    q, s = flashinfer_mxfp8_quantize(x, False, alignment=x.shape[-1], backend=_MXFP8_QUANTIZE_BACKEND)
    return q, s.view(torch.uint8).reshape(x.shape[0], -1)


def _experts_prequant(layer):
    """The body of Mxfp4FlashinferTrtllmMoEMethod.apply after its MXFP8 quantization, on gathered rows."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import trtllm_moe_enable_pdl
    try:  # v0.5.18 (FlashInfer 0.6.17 takes packed (expert_id << 16) | bf16 weight ids only)
        from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds

        pack = PackTopkIds.execute
    except ImportError:  # v0.5.20: FlashInfer 0.6.18 takes (ids, weights) unpacked, as upstream's apply passes them
        def pack(ids, wts):
            return (ids, wts)
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import RoutingMethodType as RMT
    from sglang.srt.utils.common import next_power_of_2

    qm = layer.quant_method
    w13, w2 = layer.w13_weight, layer.w2_weight
    w13_scale, w2_scale = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    inter = w2.shape[2] * 2 if w2.dtype == torch.uint8 else w2.shape[2]
    hidden = w13.shape[2] * 2 if w13.dtype == torch.uint8 else w13.shape[2]
    n_local = layer.num_local_experts
    if w13_scale.dim() == 2:
        w13_scale = w13_scale.reshape(n_local, 2 * inter, -1)
    if w2_scale.dim() == 2:
        w2_scale = w2_scale.reshape(n_local, hidden, -1)

    def run(xq, xs, ids, wts, out):
        rows = xq.shape[0]
        trtllm_fp4_block_scale_routed_moe(
            topk_ids=pack(ids, wts), routing_bias=None, hidden_states=xq,
            hidden_states_scale=xs.view(torch.float8_e4m3fn).reshape(rows, -1),
            gemm1_weights=w13, gemm1_weights_scale=w13_scale, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
            gemm1_clamp_limit=qm._gemm1_clamp_limit_tensor, gemm2_weights=w2, gemm2_weights_scale=w2_scale,
            gemm2_bias=None, output1_scale_scalar=layer.output1_scale_scalar,
            output1_scale_gate_scalar=layer.output1_scale_gate_scalar,
            output2_scale_scalar=layer.output2_scale_scalar, num_experts=layer.num_experts,
            top_k=ids.shape[1], n_group=1, topk_group=1, intermediate_size=inter,
            local_expert_offset=layer.moe_ep_rank * n_local, local_num_experts=n_local, routed_scaling_factor=1.0,
            routing_method_type=int(RMT.TopK), do_finalize=True, tune_max_num_tokens=next_power_of_2(rows),
            output=out, enable_pdl=trtllm_moe_enable_pdl(rows))
    return run


def layout(forward_batch, dp_rank: int, dp: int, n_local_rows: int) -> tuple[int, int, int, Optional[torch.Tensor]]:
    """(row_off, n_push, rows, valid) for this step, all Python ints except valid.

    MAX_LEN (decode, CUDA graphs): every rank owns a fixed block of `block` rows at dp_rank * block; the whole block
    travels (pad rows are computed and discarded, as SGLang does), and the real count is read on the device (valid)
    so a graph captured for the block stays correct on replay.
    SUM_LEN (prefill / extend, eager): ranks are packed back to back from the CPU token counts; only real rows move.
    """
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len, get_local_dp_buffer_len
    if forward_batch.dp_padding_mode.is_max_len():
        block = int(get_local_dp_buffer_len())
        return dp_rank * block, min(block, n_local_rows), dp * block, forward_batch.global_num_tokens_gpu[dp_rank:dp_rank + 1]
    counts = [int(c) for c in forward_batch.global_num_tokens_cpu]
    return sum(counts[:dp_rank]), counts[dp_rank], sum(counts), None


def routed_scale(mlp) -> float:
    """The factor SGLang applies after the experts: none when the top-k weights already carry it
    (experts.should_fuse_routed_scaling_factor_in_topk, e.g. NVFP4), routed_scaling_factor otherwise (MXFP4 TRT-LLM,
    where maybe_fuse_routed_scale_and_shared_add scales the routed sum)."""
    if getattr(getattr(mlp, "experts", None), "should_fuse_routed_scaling_factor_in_topk", False):
        return 1.0
    return float(mlp.routed_scaling_factor)


def local_topk(mlp, x: torch.Tensor, input_ids: Optional[torch.Tensor], n: int):
    """(ids int32 [n, k], weights float32 [n, k]) for this GPU's rows. An idle rank (n == 0, a prefill step where
    another rank holds every token) skips the router: dsv3_router_gemm rejects 0 tokens, and the rank still joins
    both barriers and the expert call. The width then comes from the buffers prepare() sized at load (routed top-k
    plus any fused shared column)."""
    if n == 0:
        if _BUFFERS is None:
            raise RuntimeError("numinous_dp_moe: an idle rank needs the buffers prepare() allocates at load")
        k = _BUFFERS.ids[0].shape[1]
        return (torch.empty(0, k, dtype=torch.int32, device=x.device),
                torch.empty(0, k, dtype=torch.float32, device=x.device))
    # hash-routed layers (V4's first layers) route by token id through HashTopK, which takes input_ids
    kwargs = {"input_ids": input_ids[:n]} if getattr(mlp, "is_hash", False) else {}
    tko = mlp.topk(x, mlp.gate(x), **kwargs)
    return tko.topk_ids.to(torch.int32), tko.topk_weights.to(torch.float32)


def forward(mlp, hidden_local: torch.Tensor, forward_batch, input_ids: torch.Tensor) -> torch.Tensor:
    """The fused layer. hidden_local: this GPU's rows [n (+ graph padding), hidden] bf16. Returns the MoE output for
    the same rows (routed, scaled, TP-summed, plus the shared expert); rows past the real count are zero, as SGLang's
    dp_scatter leaves them."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import get_attention_dp_rank, get_attention_dp_size

    tp = get_tp_group()
    dp_rank, dp = get_attention_dp_rank(), get_attention_dp_size()
    row_off, n, rows, valid = layout(forward_batch, dp_rank, dp, hidden_local.shape[0])
    hidden = hidden_local.shape[1]
    x = hidden_local[:n]
    ids, wts = local_topk(mlp, x, input_ids, n)
    topk = int(ids.shape[1])                     # TopK and HashTopK alike (a fused shared column included)
    cap = int(os.environ.get("NUMINOUS_DP_MOE_ROWS", "40960"))
    buf = buffers(tp.device_group, max(cap, rows), hidden, topk, hidden_local.device)
    if n:
        shared = mlp._forward_shared_experts(x) if getattr(mlp, "shared_experts", None) is not None else None
        xq, xs = _mxfp8(x)
    else:
        # an idle rank (no tokens this step) pushes nothing but still joins both barriers and the expert call
        shared = None
        xq = torch.empty(0, hidden, dtype=torch.float8_e4m3fn, device=x.device)
        xs = torch.empty(0, hidden // MXFP8_GROUP, dtype=torch.uint8, device=x.device)
    out = torch.zeros_like(hidden_local)
    # off by default: measured slower on 8xB200 (fused_path_bench, 4,096 rows per GPU): TP8 3,450 -> 4,104 us,
    # EP8 2,661 -> 3,271 at full comm grids; capping the comm kernels to 16 / 32 / 64 CTAs: 8,100 / 5,343 / 4,051.
    # SM-driven copies do not hide behind the expert GEMMs; overlap needs copy-engine (DMA) transfers.
    if valid is None and rows >= int(os.environ.get("NUMINOUS_DP_MOE_OVERLAP_ROWS", "8192")) \
            and os.environ.get("NUMINOUS_DP_MOE_OVERLAP", "0") == "1":
        counts = [int(c) for c in forward_batch.global_num_tokens_cpu]
        out = overlapped(LocalPrep(xq, xs, ids, wts, shared), buf, counts, dp_rank, _experts_prequant(mlp.experts),
                         out, routed_scale(mlp))
        maybe_verify(mlp, hidden_local, out, forward_batch, row_off, n, rows, dp_rank, input_ids=input_ids)
        return out
    phase_push(LocalPrep(xq, xs, ids, wts, shared), buf, row_off, multicast=valid is None)
    buf.barrier()
    phase_experts(buf, rows, _experts_prequant(mlp.experts))
    buf.barrier()
    if n:
        phase_pull(buf, row_off, n, out, routed_scale(mlp), shared, valid)
    maybe_verify(mlp, hidden_local, out, forward_batch, row_off, n, rows, dp_rank, input_ids=input_ids)
    return out
