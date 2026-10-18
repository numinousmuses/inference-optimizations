"""Fused DP-attention / TP-expert MoE layer for GLM-5.3 (NVFP4 experts through FlashInfer's TRT-LLM FP4 MoE).

What SGLang v0.5.20 does per GLM MoE layer when attention is data-parallel (attn_tp 1) and experts are TP8:

  1. prepare_mlp: fused residual add + RMSNorm on this GPU's rows, then dp_gather_replicate of the bf16 rows into
     the global buffer (12,288 bytes per token to every GPU)
  2. on the full gathered buffer, on every GPU: router GEMM, NVFP4 quantization of the expert input (8x redundant),
     TRT-LLM FP4 MoE with routing inside the kernel (TopK output "bypassed"), this GPU's TP slice of every expert
  3. postprocess_layer: reduce-scatter / dp_scatter of the full-width partial outputs back to each GPU's rows

What this module does instead (NUMINOUS_DP_MOE=1, hooked into DeepseekV2DecoderLayer.forward before prepare_mlp):

  1. the same add + RMSNorm on local rows; the router GEMM and the runner's own NVFP4 quantization on local rows
  2. push the NVFP4 rows (hidden / 2 bytes), their FP8 E4M3 block scales (hidden / 16 bytes, linear layout: the
     layout the runner hands the kernel) and the routing into every peer's gathered buffer
  3. the TRT-LLM FP4 MoE on the gathered, already-quantized rows (the runner's call, verbatim arguments), writing
     this GPU's TP-partial output into a symmetric buffer
  4. pull this GPU's rows from all partials (switch-reduced for decode blocks, exact fp32 sum for prefill), add a
     replicated shared expert if the layer has one, write the layer output; the gather, the scatter and both
     bf16 full-width collectives disappear

Routing payload, two modes (both SGLang's / the kernel's own routing code, no reimplementation):
  logits  fp32 router logits (4 E bytes); the experts run the runner's bypassed call (trtllm_fp4_block_scale_moe,
          in-kernel DeepSeekV3 routing with the correction bias) on the gathered logits: the stock routing
          exactly. 3,072 + 384 + 1,024 = 4,480 bytes per token at hidden 6144, E 256 (vs 12,288 bf16).
  ids     SGLang select_experts (BypassedTopKOutput.to_standard) on local rows, int32 ids + weights (8 k bytes);
          the experts run the runner's routed call (trtllm_fp4_block_scale_routed_moe). 3,528 bytes at top-9.
The default is logits for an unfused shared expert (the configuration stock v0.5.20 runs) and ids for a fused one
(the in-kernel routing has no fused shared column here); NUMINOUS_DP_MOE_ROUTE=ids forces ids.

Measured on B200 (FlashInfer 0.6.18, 256 experts top-8, sigmoid + bias, renormalize, scale 2.5), select_experts vs
the kernel's internal routing: the kernel applies the weights in bf16, so the weights travel bf16-rounded (in an fp32
lane); with that the routed kernel is bit-identical to the bypassed one except on near-tie rows, where the two
routings rank experts whose biased scores differ by ~1e-6 differently: 1 row in 1,024 to 16,384, 2 in 32,768 on
random data. One such row moves a layer output by ~1e-2 relative, which is why logits is the default where it can
run.

Shared expert: fused as expert 256 (num_fused_shared_experts 1, top-k 9) it is one more TP-sliced expert and needs
nothing here; unfused, a TP-sharded bf16 shared expert cannot be computed on local rows and the layer is declined
unless it is replicated (SGLANG_SHARED_EXPERT_TP1=1), in which case it runs on local rows and is added in the pull.
Note (measured): SGLang v0.5.20's bypassed FP4 call does not pass num_fused_shared_experts, so with a fused shared
expert FlashInfer rejects the 256-entry correction bias against 257 experts; the stock bypassed path only runs
unfused.

Validated (measured, 8xB200, operations/modal/fused_path_bench_nvfp4.py --local, GLM shapes, random NVFP4 weights,
stock reference = gathered bf16 rows through the runner's quantization and bypassed kernel, TP partials summed in
fp32): logits mode bit-exact in every case (MAX_LEN padded / idle / 1,024-row blocks, uneven SUM_LEN with an idle
rank, 16k and 32k-row prefill, CUDA-graph replay; TP8 and EP8) with NUMINOUS_DP_MOE_MC=0; with the default switch-
reduced decode pull the MAX_LEN cases are 2.9e-3 relative from exact, below stock's own NCCL bf16 reduce-scatter
(4.0e-3). Per layer, slowest rank, old (torch NCCL) -> fused: decode 96 rows/GPU 356 -> 260 us, prefill 2,048
rows/GPU 2,896 -> 1,994 us, 4,096 rows/GPU 4,767 -> 3,069 us (TP8).

World size: nothing here assumes 8 GPUs. The TP group (world = its size, 2..8: symm_collectives.MAX_WORLD) is
whatever the server runs, e.g. one TP4/DP4 replica of GLM Standard x4 on B300 (two independent 4-GPU servers, each
with its own symmetric allocation and multicast object); coverage needs attention TP 1 and TP == DP. The default
buffer (NUMINOUS_DP_MOE_ROWS 40,960 rows, 0.69 GB) is sized for 8 x 4,096-row chunks; a 4-GPU replica with chunk
16,384 (4,096 per rank) gathers at most 16,384 prefill rows, so 20,480 rows (0.34 GB) suffices there.

The pull, barrier, MAX_LEN / SUM_LEN layout, the CUDA-graph handling and the push kernels are the DeepSeek V4 Pro
module's (numinous_dp_moe, symm_collectives), reused by import; this module owns the NVFP4 widths, the routing
payload, local quantization, the NVFP4 expert calls and the decoder-layer glue. Any layer or step it does not cover
falls back to SGLang's path with one log line (logger sglang.numinous).

NUMINOUS_DP_MOE_VERIFY=<n layers> (numinous_dp_moe.maybe_verify, shared with the Pro module): on eager steps, also
run the layer's stock MoE forward on a stock-identical gathered buffer and log per-layer fused-vs-stock differences.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

try:  # the overlay installs these next to this file
    from sglang.srt.layers.moe.numinous_dp_moe import (DpMoeBuffers, enabled, layout, log_once, maybe_verify,
                                                        phase_experts, phase_pull, routed_scale)
    from sglang.srt.layers.moe.symm_collectives import push_payloads, push_rows
except ImportError:  # tests: run from operations/kernels
    from numinous_dp_moe import (DpMoeBuffers, enabled, layout, log_once, maybe_verify,  # noqa: F401
                                 phase_experts, phase_pull, routed_scale)
    from symm_collectives import push_payloads, push_rows

NVFP4_GROUP = 16
ALIGN = 2 << 20
DEFAULT_ROWS = 40960

__all__ = ["enabled", "Route", "route_of", "buffers", "buffer_specs", "prepare", "covers", "decline_reason",
           "forward", "forward_layer", "wire_bytes_per_token"]


@dataclass(frozen=True)
class Route:
    """The routing payload: mode "logits" (width = routed experts, fp32) or "ids" (width = top-k incl. a fused
    shared column; int32 ids + fp32-lane weights)."""
    mode: str
    width: int

    def specs(self, rows: int) -> tuple:
        if self.mode == "logits":
            return (("rt", (rows, self.width), torch.float32),)
        return (("ids", (rows, self.width), torch.int32), ("wts", (rows, self.width), torch.float32))

    def bytes(self) -> int:
        return 4 * self.width if self.mode == "logits" else 8 * self.width


def route_of(mlp) -> Route:
    """logits unless the layer fuses its shared expert (the in-kernel routing cannot append it) or
    NUMINOUS_DP_MOE_ROUTE=ids."""
    if int(getattr(mlp, "num_fused_shared_experts", 0) or 0) or os.environ.get("NUMINOUS_DP_MOE_ROUTE") == "ids":
        return Route("ids", int(mlp.topk.topk_config.top_k))
    return Route("logits", int(mlp.config.n_routed_experts))


def rows_cap() -> int:
    return int(os.environ.get("NUMINOUS_DP_MOE_ROWS", str(DEFAULT_ROWS)))


def wire_bytes_per_token(hidden: int, route: Route) -> int:
    """Bytes each token pushes to each peer: NVFP4 payload, E4M3 scales, the routing payload."""
    return hidden // 2 + hidden // NVFP4_GROUP + route.bytes()


def buffer_specs(rows: int, hidden: int, route: Route) -> list:
    """[(name, shape, dtype, byte offset)] of the one symmetric allocation, each part at a 2 MiB-aligned offset
    (one handle, one rendezvous, one multicast object), then ("total", ..., total bytes)."""
    specs = ((("xq", (rows, hidden // 2), torch.uint8), ("xs", (rows, hidden // NVFP4_GROUP), torch.uint8))
             + route.specs(rows) + (("part", (rows, hidden), torch.bfloat16),))
    out, off = [], 0
    for nm, shape, dtype in specs:
        out.append((nm, shape, dtype, off))
        nbytes = shape[0] * shape[1] * torch.empty(0, dtype=dtype).element_size()
        off += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    return out + [("total", (off,), torch.uint8, off)]


def make_buffers(world: int, rank: int, views: dict, route: Route, **kw) -> DpMoeBuffers:
    """DpMoeBuffers over per-peer views {name: [view per rank]}. In logits mode the logits stand in both routing
    slots (phase_experts hands them to the expert call as its ids argument; the weights argument is unused)."""
    ids, wts = (views["rt"], views["rt"]) if route.mode == "logits" else (views["ids"], views["wts"])
    buf = DpMoeBuffers(world, rank, views["xq"], views["xs"], ids, wts, views["part"], **kw)
    buf.route = route
    buf.views = views
    return buf


# ---------------------------------------------------------------------------------------------------------------
# Symmetric buffers (NVFP4 widths), same pattern and semantics as numinous_dp_moe.buffers.
# ---------------------------------------------------------------------------------------------------------------
_BUFFERS: Optional[DpMoeBuffers] = None


def buffers(group, rows: int, hidden: int, route: Route, device: torch.device) -> DpMoeBuffers:
    """Allocate and rendezvous on first use (a collective over the TP group); must not run inside CUDA-graph capture.
    Zero-filled: in MAX_LEN steps pad rows are never pushed and the expert kernel reads their routing (zero logits,
    or expert 0 at weight 0, or an earlier step's real rows)."""
    global _BUFFERS
    if _BUFFERS is not None:
        if _BUFFERS.rows_cap < rows:
            raise RuntimeError(f"numinous_dp_moe_nvfp4: step needs {rows} rows, buffers hold {_BUFFERS.rows_cap}")
        if _BUFFERS.route != route or _BUFFERS.part[0].shape[1] != hidden:
            raise RuntimeError(f"numinous_dp_moe_nvfp4: layer is hidden {hidden} {route}, buffers hold "
                               f"{_BUFFERS.part[0].shape[1]} {_BUFFERS.route}")
        return _BUFFERS
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("numinous_dp_moe_nvfp4: buffers must be allocated before CUDA-graph capture")
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    name = group.group_name
    if hasattr(symm_mem, "enable_symm_mem_for_group"):
        symm_mem.enable_symm_mem_for_group(name)
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    specs = buffer_specs(rows, hidden, route)
    total = specs[-1][3]
    specs = specs[:-1]
    raw = symm_mem.empty((total,), dtype=torch.uint8, device=device)
    raw.zero_()
    h = symm_mem.rendezvous(raw, name)
    views, ptr_arrays = {}, {}
    for nm, shape, dtype, off in specs:
        esz = torch.empty(0, dtype=dtype).element_size()
        views[nm] = [h.get_buffer(r, shape, dtype, off // esz) for r in range(world)]
        ptr_arrays[nm] = torch.tensor([int(h.buffer_ptrs[r]) + off for r in range(world)], dtype=torch.int64,
                                      device=device)
    ptrs = {nm: int(t.data_ptr()) for nm, t in ptr_arrays.items()}
    use_mc = os.environ.get("NUMINOUS_DP_MOE_MC", "1") == "1"
    base_mc = int(getattr(h, "multicast_ptr", 0) or 0) if use_mc else 0
    mc = {nm: (base_mc + off if base_mc else 0) for nm, _, _, off in specs}
    _BUFFERS = make_buffers(world, rank, views, route, barrier=lambda: h.barrier(channel=0), ptrs=ptrs,
                            handles=(h, raw, ptr_arrays), mc=mc)
    torch.cuda.synchronize(device)
    _BUFFERS.barrier()                          # every rank's zeros are in place before any peer pushes
    return _BUFFERS


def push(buf: DpMoeBuffers, payloads: list, row_off: int, multicast: bool) -> None:
    """Store this rank's rows of each (name, src) payload into rows row_off.. of every peer's buffer: one launch on
    the GPU (16-byte rows through the multicast mapping when multicast is set and the fabric has one), per-payload
    push_rows over plain per-peer tensors on the CPU (tests)."""
    n = payloads[0][1].shape[0]
    if n == 0:
        return
    if buf.ptrs:
        mc = buf.mc or {}
        mine = {nm: buf.views[nm][buf.rank] for nm, _ in payloads}
        pl = [(src, buf.ptrs[nm], mc.get(nm, 0), mine[nm].shape[1] * mine[nm].element_size()) for nm, src in payloads]
        push_payloads(pl, buf.world, n, row_off, multicast=multicast and any(mc.values()))
        return
    for nm, src in payloads:
        push_rows(src, buf.views[nm], row_off)


# ---------------------------------------------------------------------------------------------------------------
# Coverage. Static reasons (the layer, its quantization, the parallel layout) are fixed after load and cached on the
# decoder layer; dynamic reasons (this step's padding mode, row count) are evaluated per step. Every input is
# identical on every TP rank, so all ranks take the same path (the barriers need that).
# ---------------------------------------------------------------------------------------------------------------
def _moe_static_reason(mlp) -> str:
    """Empty when the fused path can serve this MoE block."""
    try:
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    except ImportError as e:
        return f"SGLang classes unavailable ({e})"
    if not isinstance(mlp, DeepseekV2MoE):
        return f"mlp is {type(mlp).__name__}, not DeepseekV2MoE"
    experts = getattr(mlp, "experts", None)
    qm = getattr(experts, "quant_method", None)
    if not isinstance(qm, ModelOptNvFp4FusedMoEMethod):
        return f"experts use {type(qm).__name__}, not ModelOptNvFp4FusedMoEMethod"
    backend = getattr(qm, "_moe_runner_backend", None)
    if backend is None or not (backend.is_flashinfer_trtllm() or backend.is_flashinfer_trtllm_routed()):
        return f"NVFP4 MoE runner backend {backend}, not flashinfer_trtllm"
    if not getattr(qm, "enable_flashinfer_trtllm_moe", False) or not hasattr(experts, "g1_scale_c"):
        return "the NVFP4 method is not on its TRT-LLM FP4 path (no g1_scale_c)"
    if getattr(qm.quant_config, "use_per_token_activation", False):
        return "per-token NVFP4 activation scales"
    if getattr(mlp, "_enable_a2a_moe", False):
        return "an A2A MoE backend dispatches this layer"
    if getattr(mlp, "is_hash", False):
        return "hash-routed MoE layer"
    if getattr(experts.moe_runner_config, "activation", "silu") not in ("silu", "relu2", "gelu", "situ"):
        return f"activation {experts.moe_runner_config.activation}"
    nfse = int(getattr(mlp, "num_fused_shared_experts", 0) or 0)
    if nfse == 0 and getattr(mlp, "shared_experts", None) is not None and not getattr(mlp, "_shared_expert_tp1", False):
        return "the shared expert is TP-sharded (fuse it, or serve with SGLANG_SHARED_EXPERT_TP1=1)"
    if getattr(experts, "_dwdp_bound", False):
        return "DWDP-bound experts"
    return ""


def _layer_static_reason(layer) -> str:
    reason = _moe_static_reason(getattr(layer, "mlp", None))
    if reason:
        return reason
    from sglang.srt.layers.communicator import CommunicateSummableTensorPairFn, CommunicateWithAllReduceAndLayerNormFn
    from sglang.srt.runtime_context import get_exec, get_parallel
    par = get_parallel()
    if not (par.attn_tp_size == 1 and par.attn_dp_size > 1 and par.tp_size == par.attn_dp_size):
        return (f"needs attention TP 1 and TP == DP > 1 (attn_tp {par.attn_tp_size}, dp {par.attn_dp_size}, "
                f"tp {par.tp_size})")
    comm = getattr(layer, "layer_communicator", None)
    if comm is None:
        return "the layer has no LayerCommunicator"
    norm_fn = getattr(comm._communicate_with_all_reduce_and_layer_norm_fn, "func",
                      comm._communicate_with_all_reduce_and_layer_norm_fn)
    if norm_fn is not CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual:
        return f"prepare_mlp is {getattr(norm_fn, '__name__', norm_fn)}, not the DP gather"
    if comm._communicate_summable_tensor_pair_fn is not CommunicateSummableTensorPairFn._scatter_hidden_states:
        return "postprocess_layer is not the DP scatter"
    if getattr(comm, "_sp_variant", None) is not None:
        return "LayerNorm sequence parallelism is configured"
    if get_exec().moe.enable_eplb:
        return "EPLB remaps expert ids"
    return ""


def decline_reason(layer, forward_batch) -> str:
    """Empty when the fused path covers this decoder layer in this step; otherwise why not."""
    reason = getattr(layer, "_numinous_ndm4_static", None)
    if reason is None:
        reason = _layer_static_reason(layer)
        layer._numinous_ndm4_static = reason
    if reason:
        return reason
    if getattr(forward_batch, "dp_padding_mode", None) is None:
        return "the batch has no DP padding mode"
    from sglang.srt.layers.communicator import get_attn_tp_context
    if get_attn_tp_context().input_scattered:
        return "attention input is TP-scattered"
    from sglang.srt.runtime_context import get_forward
    if getattr(get_forward(), "sp_active", False):
        return "LayerNorm sequence parallelism is active"
    if forward_batch.dp_padding_mode.is_max_len():
        from sglang.srt.layers.dp_attention import get_global_dp_buffer_len
        rows = int(get_global_dp_buffer_len())
    else:
        rows = sum(int(c) for c in forward_batch.global_num_tokens_cpu)
    if rows > rows_cap():
        return f"step of {rows} gathered rows exceeds NUMINOUS_DP_MOE_ROWS={rows_cap()}"
    return ""


def covers(layer, forward_batch) -> bool:
    reason = decline_reason(layer, forward_batch)
    if reason:
        log_once(("ndm4_declined", reason), f"numinous_dp_moe_nvfp4 declined: {reason}")
        return False
    log_once("ndm4_active", "numinous_dp_moe_nvfp4 active: fused DP-MoE (NVFP4 payload) serves the MoE layers")
    return True


def prepare(model) -> None:
    """At model load, before CUDA-graph capture (SGLang captures decode graphs before serving): allocate and
    rendezvous the symmetric buffers if any MoE layer of this model can be served. Every TP rank loads together.
    A second model in the process (the EAGLE draft) reuses the buffers when its layers fit them."""
    try:
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled
        if not is_dp_attention_enabled():
            return
    except ImportError:
        return
    layers = [m for m in model.modules() if hasattr(m, "layer_communicator") and hasattr(m, "mlp")
              and hasattr(m.mlp, "experts") and hasattr(m.mlp, "topk")]
    reasons = [_layer_static_reason(m) for m in layers]
    served = [m.mlp for m, r in zip(layers, reasons) if not r]
    if not served:
        if layers:
            log_once(("ndm4_prepare_none", type(model).__name__),
                     f"numinous_dp_moe_nvfp4: no MoE layer of {type(model).__name__} is covered ({reasons[0]})")
        return
    routes = {route_of(m) for m in served}
    if len(routes) != 1:
        raise RuntimeError(f"numinous_dp_moe_nvfp4: MoE layers route differently {routes}")
    route = routes.pop()
    from sglang.srt.distributed import get_tp_group
    hidden = int(served[0].config.hidden_size)
    buffers(get_tp_group().device_group, rows_cap(), hidden, route, torch.device("cuda", torch.cuda.current_device()))
    log_once("ndm4_prepared", f"numinous_dp_moe_nvfp4: symmetric buffers ready ({rows_cap()} rows x {hidden}, "
                              f"{route}, {wire_bytes_per_token(hidden, route)} B/token on the wire)")


# ---------------------------------------------------------------------------------------------------------------
# Local work: routing, quantization, the pre-quantized expert calls. All from SGLang's / the runner's own code.
# ---------------------------------------------------------------------------------------------------------------
def local_route(mlp, x: torch.Tensor, n: int, route: Route) -> list:
    """This GPU's routing payload as [(buffer name, tensor)]. logits: the router GEMM (MoEGate, fp32 out). ids:
    the router GEMM, then SGLang's select_experts with the layer's TopKConfig (BypassedTopKOutput.to_standard: the
    DeepSeekV3 routing, routed scale folded in, a fused shared column appended), weights rounded to bf16 as the
    kernel applies them (module doc). An idle rank (n == 0) skips the router (its GEMM kernels reject 0 rows) but
    still joins both barriers and the expert call."""
    dev = x.device
    if route.mode == "logits":
        if n == 0:
            return [("rt", torch.empty(0, route.width, dtype=torch.float32, device=dev))]
        return [("rt", mlp.gate(x).float().contiguous())]
    if n == 0:
        return [("ids", torch.empty(0, route.width, dtype=torch.int32, device=dev)),
                ("wts", torch.empty(0, route.width, dtype=torch.float32, device=dev))]
    from sglang.srt.layers.moe.topk import select_experts
    tko = select_experts(hidden_states=x, router_logits=mlp.gate(x), topk_config=mlp.topk.topk_config,
                         layer_id=getattr(mlp.topk, "layer_id", None))
    return [("ids", tko.topk_ids.to(torch.int32).contiguous()),
            ("wts", tko.topk_weights.to(torch.bfloat16).to(torch.float32).contiguous())]


def local_topk(mlp, x: torch.Tensor, n: int, width: Optional[int] = None):
    """(ids int32 [n, k], weights fp32 [n, k], bf16-rounded): the ids-mode routing of local_route."""
    k = width if width is not None else int(mlp.topk.topk_config.top_k)
    (_, ids), (_, wts) = local_route(mlp, x, n, Route("ids", k))
    return ids, wts


def quantize(experts, x: torch.Tensor):
    """The runner's NVFP4 quantization (quantize_hidden_states_fp4, the layer's input global scale, 16-element
    blocks, linear E4M3 scales): (packed uint8 [n, H/2], scales uint8 [n, H/16])."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import quantize_hidden_states_fp4
    q, s = quantize_hidden_states_fp4(x, experts.w13_input_scale_quant)
    return q.contiguous(), s.view(torch.uint8).reshape(x.shape[0], -1).contiguous()


def experts_prequant(mlp, route: Route):
    """experts_fn(xq, xs, rt_or_ids, wts, out) for phase_experts: fused_experts_none_to_flashinfer_trtllm_fp4 after
    its quantization, with the arguments ModelOptNvFp4FusedMoEMethod.apply builds (v0.5.20): the bypassed call
    (routing from the gathered logits, the TopKConfig's bias / groups / top-k, the runner config's routed scale)
    in logits mode, the routed call in ids mode."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe, trtllm_fp4_block_scale_routed_moe
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import get_activation_type, trtllm_moe_enable_pdl
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.utils.common import next_power_of_2

    experts = mlp.experts
    rc = experts.moe_runner_config
    weights = dict(
        gemm1_weights=experts.w13_weight.data,
        gemm1_weights_scale=experts.w13_weight_scale.data.view(torch.float8_e4m3fn), gemm1_bias=None,
        gemm2_weights=experts.w2_weight.data,
        gemm2_weights_scale=experts.w2_weight_scale.data.view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=experts.g1_scale_c.data, output1_scale_gate_scalar=experts.g1_alphas.data,
        output2_scale_scalar=experts.g2_alphas.data, per_token_scale=None, num_experts=experts.num_experts,
        intermediate_size=experts.intermediate_size_per_partition,
        local_expert_offset=experts.moe_ep_rank * experts.num_local_experts,
        local_num_experts=experts.num_local_experts, do_finalize=True,
        activation_type=get_activation_type(rc.activation, is_gated=rc.is_gated),
        **{nm: (getattr(experts, nm).data if getattr(experts, nm, None) is not None else None)
           for nm in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")})
    if route.mode == "logits":
        cfg = mlp.topk.topk_config
        rmt = getattr(experts, "routing_method_type", RoutingMethodType.Default)
        routing = dict(routing_bias=cfg.correction_bias, top_k=cfg.top_k, n_group=cfg.num_expert_group,
                       topk_group=cfg.topk_group, routed_scaling_factor=rc.routed_scaling_factor,
                       routing_method_type=rmt if rmt is not None else RoutingMethodType.Default)

        def run(xq, xs, logits, _unused, out):
            rows = xq.shape[0]
            trtllm_fp4_block_scale_moe(
                routing_logits=logits, hidden_states=xq.view(torch.uint8),
                hidden_states_scale=xs.view(torch.float8_e4m3fn).reshape(rows, -1),
                tune_max_num_tokens=next_power_of_2(rows), output=out, enable_pdl=trtllm_moe_enable_pdl(rows),
                **routing, **weights)
        return run

    def run(xq, xs, ids, wts, out):
        rows = xq.shape[0]
        trtllm_fp4_block_scale_routed_moe(
            topk_ids=(ids, wts), routing_bias=None, hidden_states=xq.view(torch.uint8),
            hidden_states_scale=xs.view(torch.float8_e4m3fn).reshape(rows, -1), top_k=ids.shape[1], n_group=0,
            topk_group=0, routed_scaling_factor=None, routing_method_type=1,
            tune_max_num_tokens=next_power_of_2(rows), output=out, enable_pdl=trtllm_moe_enable_pdl(rows), **weights)
    return run


def _run(mlp, hidden_local: torch.Tensor, row_off: int, n: int, rows: int, valid, buf: DpMoeBuffers,
         prefill: bool) -> torch.Tensor:
    """route + quantize local rows, push, barrier, experts, barrier, pull: the fused layer for one step here."""
    hidden = hidden_local.shape[1]
    x = hidden_local[:n]
    routing = local_route(mlp, x, n, buf.route)
    shared = None
    if n:
        if getattr(mlp, "shared_experts", None) is not None and not getattr(mlp, "num_fused_shared_experts", 0):
            shared = mlp._forward_shared_experts(x)        # replicated (TP1): the whole shared expert, local rows
        xq, xs = quantize(mlp.experts, x)
    else:
        xq = torch.empty(0, hidden // 2, dtype=torch.uint8, device=x.device)
        xs = torch.empty(0, hidden // NVFP4_GROUP, dtype=torch.uint8, device=x.device)
    out = torch.empty_like(hidden_local)
    if out.shape[0] > n:
        out[n:].zero_()
    push(buf, [("xq", xq), ("xs", xs)] + routing, row_off, multicast=prefill)
    buf.barrier()
    phase_experts(buf, rows, experts_prequant(mlp, buf.route))
    buf.barrier()
    if n:
        phase_pull(buf, row_off, n, out, routed_scale(mlp), shared, valid)
    return out


def forward(mlp, hidden_local: torch.Tensor, forward_batch) -> torch.Tensor:
    """The fused MoE block on this GPU's normed rows [n (+ graph padding), hidden] bf16. Returns the block's output
    for the same rows (routed + shared, TP-summed); rows past the real count are zero."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import get_attention_dp_rank, get_attention_dp_size

    tp = get_tp_group()
    dp_rank, dp = get_attention_dp_rank(), get_attention_dp_size()
    row_off, n, rows, valid = layout(forward_batch, dp_rank, dp, hidden_local.shape[0])
    buf = buffers(tp.device_group, max(rows_cap(), rows), hidden_local.shape[1], route_of(mlp), hidden_local.device)
    out = _run(mlp, hidden_local, row_off, n, rows, valid, buf, prefill=valid is None)
    maybe_verify(mlp, hidden_local, out, forward_batch, row_off, n, rows, dp_rank, tag="numinous_dp_moe_nvfp4")
    return out


def forward_layer(layer, hidden_states: torch.Tensor, residual: torch.Tensor, forward_batch, topk_indices):
    """DeepseekV2DecoderLayer.forward from prepare_mlp on: the communicator's local add + RMSNorm (skipped for an
    empty local batch, as _gather_hidden_states_and_residual does), the fused MoE block on local rows, and the
    decoder layer's return value; both the DP gather and the DP scatter are skipped."""
    if hidden_states.shape[0] != 0:
        hidden_states, residual = layer.layer_communicator.post_attention_layernorm(hidden_states, residual)
    return forward(layer.mlp, hidden_states, forward_batch), residual, topk_indices
