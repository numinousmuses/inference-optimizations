"""pdmux lanes under DP attention: the collective side of the GLM-5.3 8xH200 DP-lockstep fix (agent rtxv41).

Tree: SGLang v0.5.20 (94602c9, IMG_V0520), srt/distributed/parallel_state.py.
    python3 glm_pdmux_lane_hooks.py [--root <sglang pkg dir>] [--check]

Context (agents/findings/amdglm.md section 10). pdmux (--enable-pdmux, multiplex/multiplexing_mixin.py) runs prefill as
layer splits on one green-context stream (the P lane) while decode steps run on the other stream (the D lane). Decode
ranks then keep their own lockstep at decode pace instead of waiting in every layer's DP gather behind a rank's extend
(M5 profile: 18 ms of a 70.7 ms step on the seven waiting ranks; decode-holding ranks idle 42% of wall at OR 32).
Stock pdmux only duplicates the TP group. amdglm owns the scheduler and model split (S1-S5); this hook owns
"what a lane switch must carry". The interface between the two is the stock set_pdmux_status(True/False).

What this patches, each with the stock line it changes:
  1. Group getters (parallel_state.py:2129-2170). get_attn_tp_group, get_attn_cp_group, get_dcp_group,
     get_dcp_group_no_assert, get_moe_dp_group, get_moe_ep_group and get_moe_tp_group return the P-lane twin while the
     P lane is active, whenever the group is the main TP group. initialize_model_parallel aliases them to _TP when the
     widths match (:2690-2790): on the M layout (tp8 = dp8 = ep8) _MOE_EP IS _TP, and the EP8 MoE combine runs on it.
     Stock returns the main object, so a P-lane MoE collective and a D-lane one interleave on one communicator. Groups
     that are not the main TP group (attention TP of width 1 on the M layout) are returned unchanged.
  2. get_tp_group (:2119-2126). On the P lane it returns the twin, except inside a draft scope that installed a group of
     its own (patch_tensor_parallel_group with a non-main group, e.g. an EAGLE3 draft at attention-TP width).
  3. patch_tensor_parallel_group (:2936-2965), the EAGLE draft_tp_context. Two stock bugs under the P lane:
     (a) it saves get_tp_group(), which is the TWIN on the P lane, and restores it into _TP on exit. After one P-lane
         draft-extend (amdglm S4) the module global _TP is the twin, and every later D-lane collective runs on the
         P-lane communicator: cross-lane interleaving, i.e. a hang or wrong rows;
     (b) it overrides get_parallel().tp_group with the MAIN group even on the P lane.
     Fixed: the raw _TP is saved, and the scope's group resolves to its P-lane twin.
  4. set_pdmux_status (:2114-2116). A lane switch saves the per-forward state of the lane being left and restores the
     state of the lane being entered:
     - the _DpGatheredBufferWrapper sizing quintet (dp_attention.py:168-187: global/local buffer length, MAX_LEN flag,
       global_num_tokens and its GPU copy). Every ForwardBatch writes it (forward_batch_info.py:1442-1449), and a
       decode step between two prefill splits overwrites it;
     - the sticky is_extend_in_batch forward flag (dp_attention.py:330-335; an unscoped write, runtime_context.py:739);
     - the grow-only module-level buffer dicts: dp_attention._dp_gather_fp8_bufs (:613-634, the FP8 DP-gather payload
       and scales) and _TBO_PERSIST_BUF (:993), plus rtxattn's router piggyback wire buffers
       (numinous_dp_router_piggyback._bufs, installed by dp_router_before_gather_hooks). Shared, they are written by
       both streams at once: a P-lane SUM_LEN gather and an eager D-lane SUM_LEN decode (dp_attention.py:121-135
       picks SUM_LEN for an unbalanced decode). A grow on one lane would also free a buffer the other lane's kernels
       still read. Each lane gets its own dict object.
     The P-lane state starts empty: the first P-lane forward sets it (amdglm's split-0 prep runs
     _prepare_eager_forward_batch -> prepare_mlp_sync_batch).
  5. numinous_pdmux_lane() -> 0 (D lane) / 1 (P lane), for any later per-lane cache.
  6. W4AFP8 MoE routing metadata (srt/layers/quantization/w4afp8.py:245-253 + apply :330-366). Each MoE layer's quant
     method owns ONE expert_offsets [E+1] and problem_sizes1/2 [E, 3]. Every call rewrites them in place
     (cutlass_w4a8_moe.py: get_cutlass_w4a8_moe_mm_data), and the grouped GEMMs then read them (stock CUTLASS, k3amd
     tiles .so, k3amd w4a8dg). The D lane's CUDA graphs captured these exact tensors. A P-lane MoE call on the same
     layer rewrites them mid-replay: wrong expert row ranges, i.e. garbage rows or an illegal address. The P lane now
     gets its own triple per layer, allocated on its first P-lane call (never under capture). The D lane and non-pdmux
     servers keep the original tensors (graph pointers unchanged). The kernels themselves allocate their workspaces
     per call on the current stream (w4a8_grouped_mm_c3x.cuh:202-261; operations/kernels/w4a8dg/w4a8dg.cu:77,111),
     so they are lane-safe.

Contract for callers (amdglm's scheduler): switch lanes only between whole layers; build a P-lane ForwardBatch with
status True and a D-lane one with status False. Inert without --enable-pdmux: set_pdmux_status is called only by the
pdmux loop (multiplexing_mixin.py:116-205) and once with False by the pdmux graph capture
(decode_cuda_graph_runner.py:1043). NUMINOUS_PDMUX_LANE_STATE=0 turns the state swap off (items 1-3 stay).
Same contract as the other patchers: exact anchors, marker -> re-application is a no-op, patched files compile.
"""
from __future__ import annotations

import argparse
import os
import sys

NAME = "pdmux_lane"
MARKER = f"numinous-hook:{NAME}"
PS = "srt/distributed/parallel_state.py"

GETTERS = {
    # anchor (stock getter body) -> the group global it returns
    "def get_attn_tp_group() -> GroupCoordinator:\n    assert _ATTN_TP is not None, (\n"
    '        "attention tensor model parallel group is not initialized"\n    )\n    return _ATTN_TP\n': "_ATTN_TP",
    "def get_attn_cp_group() -> GroupCoordinator:\n    assert _ATTN_CP is not None, (\n"
    '        "attention context model parallel group is not initialized"\n    )\n    return _ATTN_CP\n': "_ATTN_CP",
    "def get_dcp_group_no_assert() -> Optional[GroupCoordinator]:\n    return _DCP\n": "_DCP",
    'def get_dcp_group() -> GroupCoordinator:\n    assert _DCP is not None, "decode context parallel group is not '
    'initialized"\n    return _DCP\n': "_DCP",
    'def get_moe_dp_group() -> GroupCoordinator:\n    assert _MOE_DP is not None, "moe data parallel group is not '
    'initialized"\n    return _MOE_DP\n': "_MOE_DP",
    'def get_moe_ep_group() -> GroupCoordinator:\n    assert _MOE_EP is not None, "expert model parallel group is not '
    'initialized"\n    return _MOE_EP\n': "_MOE_EP",
    'def get_moe_tp_group() -> GroupCoordinator:\n    assert _MOE_TP is not None, "expert model parallel group is not '
    'initialized"\n    return _MOE_TP\n': "_MOE_TP",
}

STATUS_ANCHOR = ("def set_pdmux_status(enable_prefill_multiplexing: bool):\n"
                 "    global _ENABLE_PDMUX_P_TP\n"
                 "    _ENABLE_PDMUX_P_TP = enable_prefill_multiplexing\n")
STATUS_TEXT = ("def set_pdmux_status(enable_prefill_multiplexing: bool):\n"
               "    global _ENABLE_PDMUX_P_TP\n"
               f"    # {MARKER}: a lane switch carries the per-forward DP state of each lane\n"
               "    if bool(enable_prefill_multiplexing) != bool(_ENABLE_PDMUX_P_TP):\n"
               "        _numinous_lane_switch(bool(enable_prefill_multiplexing))\n"
               "    _ENABLE_PDMUX_P_TP = enable_prefill_multiplexing\n")

TP_ANCHOR = ("    if _ENABLE_PDMUX_P_TP:\n"
             "        assert _PDMUX_PREFILL_TP_GROUP is not None, (\n"
             '            "tensor model parallel group for PD-Multiplexing Prefill is not initialized"\n'
             "        )\n"
             "        return _PDMUX_PREFILL_TP_GROUP\n"
             '    assert _TP is not None, "tensor model parallel group is not initialized"\n'
             "    return _TP\n")
TP_TEXT = ("    if _ENABLE_PDMUX_P_TP:\n"
           "        assert _PDMUX_PREFILL_TP_GROUP is not None, (\n"
           '            "tensor model parallel group for PD-Multiplexing Prefill is not initialized"\n'
           "        )\n"
           f"        # {MARKER}: a draft scope that installed its own (non-main) group keeps it on the P lane\n"
           "        if (\n"
           "            _TP is not None\n"
           "            and _NUMINOUS_LANE_MAIN_TP is not None\n"
           "            and _TP is not _NUMINOUS_LANE_MAIN_TP\n"
           "            and _TP is not _PDMUX_PREFILL_TP_GROUP\n"
           "        ):\n"
           "            return _TP\n"
           "        return _PDMUX_PREFILL_TP_GROUP\n"
           '    assert _TP is not None, "tensor model parallel group is not initialized"\n'
           "    return _TP\n")

DUP_ANCHOR = ('            group_name="pdmux_prefill_tp",\n'
              "            recovered_rank=recovered_rank,\n"
              "            rank_offset=rank_offset,\n"
              "            max_world_size=max_world_size,\n"
              "        )\n")
DUP_TEXT = DUP_ANCHOR + (f"        global _NUMINOUS_LANE_MAIN_TP  # {MARKER}: the main TP group the twins stand in for\n"
                         "        _NUMINOUS_LANE_MAIN_TP = _TP\n")

PATCH_ANCHOR = ("    _TP_STATE_PATCHED = True\n"
                "    old_tp_group = get_tp_group()\n"
                "    global _TP\n"
                "    _TP = tp_group\n")
PATCH_TEXT = ("    global _TP\n"
              "    _TP_STATE_PATCHED = True\n"
              f"    # {MARKER}: save the raw global (get_tp_group() is the twin on the P lane, and restoring it would\n"
              "    # leave the D lane on the P-lane communicator), and run the scope on the lane's own twin\n"
              "    old_tp_group = _TP\n"
              "    tp_group = _numinous_lane_twin(tp_group)\n"
              "    _TP = tp_group\n")

TAIL = f'''


# {MARKER} (rtxv41): per-lane state for pdmux under DP attention; see operations/patches/glm_pdmux_lane_hooks.py
_NUMINOUS_LANE_MAIN_TP: Optional[GroupCoordinator] = None
_NUMINOUS_LANE_STATE: Dict[bool, dict] = {{}}
# grow-only module-level buffer dicts the two lanes must not share: (module, attribute)
_NUMINOUS_LANE_DICTS = (
    ("sglang.srt.layers.dp_attention", "_dp_gather_fp8_bufs"),
    ("sglang.srt.layers.dp_attention", "_TBO_PERSIST_BUF"),
    ("sglang.srt.layers.numinous_dp_router_piggyback", "_bufs"),
)


def numinous_pdmux_lane() -> int:
    """0 on the decode lane (and without pdmux), 1 while the pdmux prefill lane is active."""
    return 1 if _ENABLE_PDMUX_P_TP else 0


def _numinous_lane_twin(group):
    """The P-lane stand-in for `group`: the prefill twin when `group` is the main TP group and the P lane is active."""
    if not _ENABLE_PDMUX_P_TP or group is None or _PDMUX_PREFILL_TP_GROUP is None:
        return group
    main = _NUMINOUS_LANE_MAIN_TP if _NUMINOUS_LANE_MAIN_TP is not None else _TP
    return _PDMUX_PREFILL_TP_GROUP if group is main else group


def _numinous_lane_capture() -> dict:
    import sys as _sys

    st = {{}}
    dpa = _sys.modules.get("sglang.srt.layers.dp_attention")
    if dpa is not None:
        w = dpa._DpGatheredBufferWrapper
        st["dp"] = (
            w._global_dp_buffer_len,
            w._local_dp_buffer_len,
            w._dp_max_padding,
            w._global_num_tokens,
            w._global_num_tokens_gpu,
        )
    from sglang.srt.runtime_context import get_forward

    st["is_extend_in_batch"] = get_forward().is_extend_in_batch
    dicts = {{}}
    for mod, attr in _NUMINOUS_LANE_DICTS:
        m = _sys.modules.get(mod)
        if m is not None and isinstance(getattr(m, attr, None), dict):
            dicts[(mod, attr)] = getattr(m, attr)
    st["dicts"] = dicts
    return st


def _numinous_lane_restore(st: dict) -> None:
    import sys as _sys

    dpa = _sys.modules.get("sglang.srt.layers.dp_attention")
    if dpa is not None:
        dpa._DpGatheredBufferWrapper.set_dp_buffer_len(*(st.get("dp") or (0, 0, False, None, None)))
    from sglang.srt.runtime_context import get_forward

    get_forward().set("is_extend_in_batch", bool(st.get("is_extend_in_batch", False)))
    saved = st.get("dicts") or {{}}
    for mod, attr in _NUMINOUS_LANE_DICTS:
        m = _sys.modules.get(mod)
        if m is not None and isinstance(getattr(m, attr, None), dict):
            # a lane that never held this dict starts with an empty one of its own
            setattr(m, attr, saved.get((mod, attr), {{}}))


def _numinous_lane_switch(to_prefill: bool) -> None:
    if os.environ.get("NUMINOUS_PDMUX_LANE_STATE", "1") == "0":
        return
    _NUMINOUS_LANE_STATE[not to_prefill] = _numinous_lane_capture()
    _numinous_lane_restore(_NUMINOUS_LANE_STATE.get(to_prefill) or {{}})
'''


W4 = "srt/layers/quantization/w4afp8.py"
W4_ANCHOR = ("            self.expert_offsets,\n            self.problem_sizes1,\n            self.problem_sizes2,\n"
             "            layer.w13_input_scale,\n            layer.w2_input_scale,\n            routed_scaling_factor=")
W4_TEXT = (f"            # {MARKER}: the active pdmux lane's own routing metadata (D lane: the graph-captured tensors)\n"
           "            *_numinous_w4a8_lane_meta(self),\n"
           "            layer.w13_input_scale,\n            layer.w2_input_scale,\n            routed_scaling_factor=")
W4_TAIL = f'''


# {MARKER} (rtxv41): per-lane W4AFP8 routing metadata; see operations/patches/glm_pdmux_lane_hooks.py item 6
def _numinous_w4a8_lane_meta(method):
    """(expert_offsets, problem_sizes1, problem_sizes2) for the active pdmux lane. cutlass_w4a8_moe rewrites them on
    every call, so the prefill lane must not share the decode lane's (graph-captured) tensors."""
    from sglang.srt.distributed.parallel_state import numinous_pdmux_lane

    if numinous_pdmux_lane() == 0:
        return method.expert_offsets, method.problem_sizes1, method.problem_sizes2
    meta = getattr(method, "_numinous_p_lane_meta", None)
    if meta is None:
        assert not torch.cuda.is_current_stream_capturing(), "pdmux P-lane W4A8 metadata requested during capture"
        meta = (
            torch.empty_like(method.expert_offsets),
            torch.empty_like(method.problem_sizes1),
            torch.empty_like(method.problem_sizes2),
        )
        method._numinous_p_lane_meta = meta
    return meta
'''


def _replace_once(src: str, anchor: str, text: str, path: str = PS) -> str:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"{NAME}: anchor found {n} times in {path}: {anchor[:80]!r}")
    return src.replace(anchor, text, 1)


def patch_w4afp8(src: str) -> str:
    return _replace_once(src, W4_ANCHOR, W4_TEXT, W4).rstrip("\n") + "\n" + W4_TAIL


def patch_source(src: str) -> str:
    out = src
    for anchor, var in GETTERS.items():
        ret = f"    return {var}\n"
        assert anchor.endswith(ret), anchor
        out = _replace_once(out, anchor, anchor[: -len(ret)] + f"    return _numinous_lane_twin({var})  # {MARKER}\n")
    out = _replace_once(out, STATUS_ANCHOR, STATUS_TEXT)
    out = _replace_once(out, TP_ANCHOR, TP_TEXT)
    out = _replace_once(out, DUP_ANCHOR, DUP_TEXT)
    out = _replace_once(out, PATCH_ANCHOR, PATCH_TEXT)
    return out.rstrip("\n") + "\n" + TAIL


def apply(root: str, check: bool = False) -> list[str]:
    """Both files or neither: every anchor is resolved and compiled before anything is written."""
    plan, msgs = [], []
    for rel, fn, what in ((PS, patch_source, f"{len(GETTERS)} getters, get_tp_group, set_pdmux_status, twin record, "
                                                 "patch_tensor_parallel_group"),
                          (W4, patch_w4afp8, "per-lane W4AFP8 routing metadata")):
        f = os.path.join(root, rel)
        src = open(f).read()
        if MARKER in src:
            msgs.append(f"{NAME}: already applied to {rel}")
            continue
        out = fn(src)                  # raises on any anchor that is missing or ambiguous
        compile(out, f, "exec")
        plan.append((f, out))
        msgs.append(f"{NAME}: {'anchors ok in' if check else 'applied to'} {rel} ({what})")
    if not check:
        for f, out in plan:
            open(f, "w").write(out)
    return msgs


def numinous_preflight(ctx) -> list:
    """CPU attach check (operations/hopper/hook_preflight.py). With --enable-pdmux, list every resolved configuration
    whose collectives this hook does not give a P-lane twin; [] when pdmux is off (the hook is inert)."""
    args = dict(getattr(ctx, "args", None) or {})
    env = dict(getattr(ctx, "env", None) or {})
    if not args.get("enable_pdmux"):
        return []
    problems = []
    tp = int(args.get("tp_size") or 1)
    dp = int(args.get("dp_size") or 1)
    attn_tp = tp // dp if args.get("enable_dp_attention") and dp else tp
    ep = int(args.get("ep_size") or 1)
    if args.get("enable_dp_attention") and attn_tp not in (1, tp):
        problems.append(f"pdmux lanes: attention-TP width {attn_tp} is a separate group of width > 1 with no P-lane "
                        "twin (only groups aliased to the main TP group, or width 1, are lane-safe)")
    if ep not in (1, tp):
        problems.append(f"pdmux lanes: --ep-size {ep} != tp {tp} builds a separate MoE-EP group with no P-lane twin")
    a2a = str(args.get("moe_a2a_backend") or "none")
    if a2a != "none":
        problems.append(f"pdmux lanes: --moe-a2a-backend {a2a} keeps single-instance dispatch buffers; only 'none' "
                        "(gather + reduce_scatterv on the twinned TP group) is lane-safe")
    if args.get("enable_two_batch_overlap"):
        problems.append("pdmux lanes: two-batch overlap shares its persistent buffers across the lanes")
    if int(args.get("pp_size") or 1) != 1:
        problems.append("pdmux lanes: pipeline parallelism is not supported by pdmux")
    if env.get("NUMINOUS_PDMUX_LANE_STATE") == "0":
        problems.append("pdmux lanes: NUMINOUS_PDMUX_LANE_STATE=0 under DP attention lets a decode step overwrite the "
                        "prefill split's DP buffer state")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = a.root
    if root is None:
        import importlib.util
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("sglang is not importable")
        root = list(spec.submodule_search_locations)[0]
    for ln in apply(root, a.check):
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
