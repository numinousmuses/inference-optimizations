"""CPU tests for glm_pdmux_lane_hooks (pdmux lanes under DP attention; agent rtxv41).

Fast tests exec the PATCHED v0.5.20 functions (cut from the patched parallel_state.py by AST) against stub groups, the
real _DpGatheredBufferWrapper class text and a stub ForwardFlags, so every assertion runs the exact patched source.
The stock functions are run the same way as negative controls (the patch_tensor_parallel_group twin bug reproduces).
    pytest -p no:cacheprovider operations/patches/test_glm_pdmux_lane_hooks.py
"""
from __future__ import annotations

import ast
import contextlib
import os
import pathlib
import subprocess
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm_pdmux_lane_hooks as hk  # noqa: E402

SGL = pathlib.Path(os.environ.get("SGLANG_FORK", "/home/ubuntu/sglang-fork"))
REF = "94602c9c2b"   # v0.5.20 release commit (IMG_V0520)


def _stock(rel: str) -> str:
    return subprocess.run(["git", "-C", str(SGL), "show", f"{REF}:python/sglang/{rel}"], capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture(scope="module")
def stock_ps() -> str:
    return _stock(hk.PS)


@pytest.fixture(scope="module")
def patched_ps(stock_ps) -> str:
    return hk.patch_source(stock_ps)


FUNCS = ["get_tp_group", "get_attn_tp_group", "get_attn_cp_group", "get_dcp_group_no_assert", "get_dcp_group",
         "get_moe_dp_group", "get_moe_ep_group", "get_moe_tp_group", "set_pdmux_status", "patch_tensor_parallel_group"]
TAIL_FUNCS = ["numinous_pdmux_lane", "_numinous_lane_twin", "_numinous_lane_capture", "_numinous_lane_restore",
              "_numinous_lane_switch"]


class G:
    """A stub GroupCoordinator."""

    def __init__(self, name, world_size=8, rank=0):
        self.name, self.world_size, self.rank_in_group = name, world_size, rank

    def __repr__(self):
        return f"G({self.name})"


class StubParallel:
    def __init__(self):
        self._overrides = {}

    @contextlib.contextmanager
    def override(self, **kw):
        saved = dict(self._overrides)
        self._overrides.update(kw)
        try:
            yield self
        finally:
            self._overrides = saved


class StubForward:
    def __init__(self):
        self.is_extend_in_batch = False

    def set(self, name, value):
        setattr(self, name, value)


def _wrapper_class():
    """The real v0.5.20 _DpGatheredBufferWrapper (dp_attention.py), exec'd with its few module names stubbed."""
    src = _stock("srt/layers/dp_attention.py")
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_DpGatheredBufferWrapper")
    ns = {"Optional": __import__("typing").Optional, "List": list, "torch": types.SimpleNamespace(Tensor=object,
          dtype=object, device=object), "GroupCoordinator": object, "get_flags": None, "use_symmetric_memory": None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "dp_attention.py", "exec"), ns)
    return ns["_DpGatheredBufferWrapper"]


def _load(src: str, names, with_tail: bool):
    """exec the named top-level functions (and the hook's tail) of `src` in one namespace with stub globals."""
    tree = ast.parse(src)
    want = set(names) | (set(TAIL_FUNCS) if with_tail else set())
    body = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in want:
            body.append(n)
        elif with_tail and isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgt = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if isinstance(tgt, ast.Name) and tgt.id.startswith("_NUMINOUS_LANE"):
                body.append(n)
    found = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    assert want <= found, sorted(want - found)
    sp = StubParallel()
    ns = {"os": os, "Optional": __import__("typing").Optional, "Dict": dict, "GroupCoordinator": G,
          "contextmanager": contextlib.contextmanager, "get_parallel": lambda: sp, "_TP_STATE_PATCHED": False,
          "_ENABLE_PDMUX_P_TP": False, "_PDMUX_PREFILL_TP_GROUP": None, "_TP": None, "_ATTN_TP": None, "_ATTN_CP": None,
          "_DCP": None, "_MOE_DP": None, "_MOE_EP": None, "_MOE_TP": None, "_NUMINOUS_LANE_MAIN_TP": None}
    exec(compile(ast.Module(body=body, type_ignores=[]), "parallel_state.py", "exec"), ns)
    ns["_sp"] = sp
    return ns


@pytest.fixture
def fake_modules(monkeypatch):
    """sys.modules stubs for the two modules the lane switch reads: runtime_context (get_forward) and dp_attention
    (the real wrapper class + the module-level buffer dicts) and the router piggyback module."""
    fwd = StubForward()
    rc = types.ModuleType("sglang.srt.runtime_context")
    rc.get_forward = lambda: fwd
    dpa = types.ModuleType("sglang.srt.layers.dp_attention")
    dpa._DpGatheredBufferWrapper = _wrapper_class()
    dpa._dp_gather_fp8_bufs = {"cuda:0": "D-lane fp8 buffers"}
    dpa._TBO_PERSIST_BUF = {}
    for name in ("sglang", "sglang.srt", "sglang.srt.layers"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", rc)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.dp_attention", dpa)
    monkeypatch.delenv("NUMINOUS_PDMUX_LANE_STATE", raising=False)
    return types.SimpleNamespace(fwd=fwd, dpa=dpa)


def _m_layout(ns):
    """Groups as initialize_model_parallel builds them on the M layout (tp8 = dp8 = ep8, attention TP width 1)."""
    main, twin = G("tp"), G("pdmux_prefill_tp")
    attn = G("attn_tp", world_size=1)
    moetp = G("moe_tp", world_size=1)
    dcp = G("dcp", world_size=1)
    ns.update(_TP=main, _PDMUX_PREFILL_TP_GROUP=twin, _NUMINOUS_LANE_MAIN_TP=main, _ATTN_TP=attn, _ATTN_CP=attn,
              _DCP=dcp, _MOE_DP=main, _MOE_EP=main, _MOE_TP=moetp)
    return main, twin, attn, moetp, dcp


# ------------------------------------------------------------------------------------------------ patcher contract
@pytest.fixture(scope="module")
def stock_w4() -> str:
    return _stock(hk.W4)


def _tree(tmp_path, ps_src, w4_src):
    root = tmp_path / "sglang"
    for rel, src in ((hk.PS, ps_src), (hk.W4, w4_src)):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(src)
    return root


def test_anchors_apply_compile_and_idempotent(tmp_path, stock_ps, stock_w4):
    root = _tree(tmp_path, stock_ps, stock_w4)
    f, w = root / hk.PS, root / hk.W4
    msgs = hk.apply(str(root), check=True)
    assert len(msgs) == 2 and all("anchors ok" in m for m in msgs), msgs
    assert f.read_text() == stock_ps and w.read_text() == stock_w4      # --check writes nothing
    assert all("applied to" in m for m in hk.apply(str(root)))
    out, wout = f.read_text(), w.read_text()
    compile(out, str(f), "exec")
    compile(wout, str(w), "exec")
    assert out.count(hk.MARKER) == len(hk.GETTERS) + 5    # 7 getters, status, get_tp_group, twin record, patch, tail
    assert wout.count(hk.MARKER) == 2 and "*_numinous_w4a8_lane_meta(self)," in wout
    assert all("already applied" in m for m in hk.apply(str(root)))
    assert f.read_text() == out and w.read_text() == wout


def test_missing_anchor_refuses_and_writes_nothing(tmp_path, stock_ps, stock_w4):
    bad_w4 = stock_w4.replace("            self.problem_sizes2,\n            layer.w13_input_scale,\n",
                              "            self.problem_sizes2,\n            layer.w13_input_scale,  # moved\n", 1)
    root = _tree(tmp_path, stock_ps, bad_w4)
    with pytest.raises(SystemExit, match="anchor found 0 times"):
        hk.apply(str(root))
    assert (root / hk.PS).read_text() == stock_ps         # all-or-nothing: parallel_state untouched
    root2 = _tree(tmp_path / "b", stock_ps.replace("    old_tp_group = get_tp_group()\n", "    old_tp_group = _TP\n"),
                  stock_w4)
    with pytest.raises(SystemExit, match="anchor found 0 times"):
        hk.apply(str(root2))


def _w4_helper(lane_fn, capturing=False):
    """exec the patched w4afp8 tail helper with stub torch and a stub parallel_state lane function."""
    import torch as _t
    tree = ast.parse(hk.W4_TAIL)
    ns = {"torch": types.SimpleNamespace(empty_like=_t.empty_like, cuda=types.SimpleNamespace(
        is_current_stream_capturing=lambda: capturing))}
    mod = types.ModuleType("sglang.srt.distributed.parallel_state")
    mod.numinous_pdmux_lane = lane_fn
    return tree, ns, mod


def test_w4a8_metadata_is_per_lane(monkeypatch):
    import torch as _t
    lane = {"v": 0}
    tree, ns, mod = _w4_helper(lambda: lane["v"])
    for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(tree, "w4afp8.py", "exec"), ns)
    m = types.SimpleNamespace(expert_offsets=_t.zeros(33, dtype=_t.int32), problem_sizes1=_t.zeros(32, 3,
                              dtype=_t.int32), problem_sizes2=_t.zeros(32, 3, dtype=_t.int32))
    d = ns["_numinous_w4a8_lane_meta"](m)
    assert d[0] is m.expert_offsets and d[1] is m.problem_sizes1 and d[2] is m.problem_sizes2
    lane["v"] = 1
    p = ns["_numinous_w4a8_lane_meta"](m)
    assert all(a is not b for a, b in zip(p, d))
    assert [tuple(x.shape) for x in p] == [(33,), (32, 3), (32, 3)] and all(x.dtype == _t.int32 for x in p)
    assert all(a is b for a, b in zip(ns["_numinous_w4a8_lane_meta"](m), p))     # allocated once per layer
    lane["v"] = 0
    assert ns["_numinous_w4a8_lane_meta"](m)[0] is m.expert_offsets


def test_w4a8_p_lane_metadata_never_allocated_under_capture(monkeypatch):
    import torch as _t
    tree, ns, mod = _w4_helper(lambda: 1, capturing=True)
    for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(tree, "w4afp8.py", "exec"), ns)
    m = types.SimpleNamespace(expert_offsets=_t.zeros(3, dtype=_t.int32), problem_sizes1=_t.zeros(2, 3,
                              dtype=_t.int32), problem_sizes2=_t.zeros(2, 3, dtype=_t.int32))
    with pytest.raises(AssertionError, match="during capture"):
        ns["_numinous_w4a8_lane_meta"](m)


def test_w4a8_apply_passes_the_lane_triple_in_the_stock_slots(stock_w4):
    """The patched call keeps the argument order of cutlass_w4a8_moe (..., s_strides2, expert_offsets, problem_sizes1,
    problem_sizes2, a1_scale, a2_scale)."""
    out = hk.patch_w4afp8(stock_w4)
    fn = next(n for n in ast.walk(ast.parse(out)) if isinstance(n, ast.FunctionDef) and n.name == "apply")
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "cutlass_w4a8_moe")
    srcs = [ast.unparse(a) for a in call.args]
    i = srcs.index("self.s_strides2")
    assert srcs[i + 1] == "*_numinous_w4a8_lane_meta(self)" and srcs[i + 2] == "layer.w13_input_scale"
    stock_fn = next(n for n in ast.walk(ast.parse(stock_w4)) if isinstance(n, ast.FunctionDef) and n.name == "apply")
    stock_call = next(n for n in ast.walk(stock_fn) if isinstance(n, ast.Call)
                      and getattr(n.func, "id", "") == "cutlass_w4a8_moe")
    assert len(stock_call.args) == len(call.args) + 2


def test_patched_file_has_no_stock_restore_of_the_twin(patched_ps):
    assert "old_tp_group = get_tp_group()" not in patched_ps
    assert patched_ps.count("_numinous_lane_twin(") >= len(hk.GETTERS) + 1


# ------------------------------------------------------------------------------------------------------- 1. getters
def test_getters_resolve_twin_only_on_the_p_lane(patched_ps):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    main, twin, attn, moetp, dcp = _m_layout(ns)
    d = {k: ns[k]() for k in ("get_tp_group", "get_attn_tp_group", "get_moe_ep_group", "get_moe_dp_group",
                              "get_moe_tp_group", "get_dcp_group", "get_dcp_group_no_assert", "get_attn_cp_group")}
    assert d == {"get_tp_group": main, "get_attn_tp_group": attn, "get_moe_ep_group": main, "get_moe_dp_group": main,
                 "get_moe_tp_group": moetp, "get_dcp_group": dcp, "get_dcp_group_no_assert": dcp,
                 "get_attn_cp_group": attn}
    ns["_ENABLE_PDMUX_P_TP"] = True
    assert ns["get_tp_group"]() is twin
    assert ns["get_moe_ep_group"]() is twin               # the EP8 MoE combine's group on the M layout
    assert ns["get_moe_dp_group"]() is twin
    assert ns["get_attn_tp_group"]() is attn              # width-1 groups have no communicator to share
    assert ns["get_moe_tp_group"]() is moetp
    assert ns["get_dcp_group"]() is dcp


def test_stock_getters_leak_the_main_group_onto_the_p_lane(stock_ps):
    """Negative control: stock get_moe_ep_group returns the main TP object on the P lane (cross-lane interleave)."""
    ns = _load(stock_ps, FUNCS, with_tail=False)
    main, twin, *_ = _m_layout(ns)
    ns["_ENABLE_PDMUX_P_TP"] = True
    assert ns["get_tp_group"]() is twin
    assert ns["get_moe_ep_group"]() is main


def test_attention_alias_of_tp_follows_the_lane(patched_ps):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    main, twin, *_ = _m_layout(ns)
    ns["_ATTN_TP"] = main                                 # attn_tp == tp (no DP attention): _ATTN_TP = _TP
    ns["_ENABLE_PDMUX_P_TP"] = True
    assert ns["get_attn_tp_group"]() is twin


# -------------------------------------------------------------------------- 2/3. draft scope (EAGLE draft_tp_context)
def test_draft_scope_on_p_lane_runs_on_twin_and_restores_main(patched_ps, fake_modules):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    main, twin, *_ = _m_layout(ns)
    ns["set_pdmux_status"](True)
    with ns["patch_tensor_parallel_group"](main):
        assert ns["get_tp_group"]() is twin
        assert ns["_sp"]._overrides["tp_group"] is twin
        assert ns["get_moe_ep_group"]() is twin
    assert ns["_TP"] is main
    ns["set_pdmux_status"](False)
    assert ns["get_tp_group"]() is main
    assert ns["get_moe_ep_group"]() is main


def test_stock_draft_scope_leaves_decode_lane_on_the_twin(stock_ps):
    """Negative control (source_code bug): stock saves get_tp_group() (the twin on the P lane) and restores it into _TP."""
    ns = _load(stock_ps, FUNCS, with_tail=False)
    main, twin, *_ = _m_layout(ns)
    ns["set_pdmux_status"](True)
    with ns["patch_tensor_parallel_group"](main):
        assert ns["_sp"]._overrides["tp_group"] is main  # (b): MAIN group answered on the P lane
    ns["set_pdmux_status"](False)
    assert ns["_TP"] is twin                              # (a): the D lane now issues on the P-lane communicator
    assert ns["get_tp_group"]() is twin


def test_draft_scope_on_decode_lane_is_stock(patched_ps, fake_modules):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    main, twin, *_ = _m_layout(ns)
    with ns["patch_tensor_parallel_group"](main):
        assert ns["get_tp_group"]() is main
        assert ns["_sp"]._overrides["tp_group"] is main
    assert ns["_TP"] is main


def test_draft_scope_with_its_own_group_keeps_it_on_both_lanes(patched_ps, fake_modules):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    main, twin, attn, *_ = _m_layout(ns)
    ns["set_pdmux_status"](True)
    with ns["patch_tensor_parallel_group"](attn):        # EAGLE3: draft at attention-TP width
        assert ns["get_tp_group"]() is attn
        assert ns["_sp"]._overrides["tp_group"] is attn
    assert ns["_TP"] is main


def test_nested_draft_scope_still_refused(patched_ps, fake_modules):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    main, *_ = _m_layout(ns)
    with ns["patch_tensor_parallel_group"](main):
        with pytest.raises(AssertionError):
            with ns["patch_tensor_parallel_group"](main):
                pass


# ----------------------------------------------------------------------------------------------- 4. lane state swap
def test_lane_switch_carries_dp_state_flag_and_buffers(patched_ps, fake_modules):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    _m_layout(ns)
    W, fwd, dpa = fake_modules.dpa._DpGatheredBufferWrapper, fake_modules.fwd, fake_modules.dpa
    gpu_d, gpu_p = object(), object()
    d_bufs = dpa._dp_gather_fp8_bufs
    # a decode step (MAX_LEN, 8 ranks x 6 verify rows)
    W.set_dp_buffer_len(48, 6, True, [6] * 8, gpu_d)
    fwd.set("is_extend_in_batch", False)
    # P lane: first entry starts empty; split 0's prep then writes the prefill state (SUM_LEN, one rank extends)
    ns["set_pdmux_status"](True)
    assert (W._global_dp_buffer_len, W._local_dp_buffer_len, W._dp_max_padding, W._global_num_tokens) == (0, 0, False,
                                                                                                         None)
    assert dpa._dp_gather_fp8_bufs == {} and dpa._dp_gather_fp8_bufs is not d_bufs
    p_sizes = [2048, 0, 0, 0, 0, 0, 0, 0]
    W.set_dp_buffer_len(2048, 2048, False, p_sizes, gpu_p)
    fwd.set("is_extend_in_batch", True)
    dpa._dp_gather_fp8_bufs["cuda:0"] = "P-lane fp8 buffers"
    p_bufs = dpa._dp_gather_fp8_bufs
    # D lane: the decode state is back, untouched by the prefill
    ns["set_pdmux_status"](False)
    assert (W._global_dp_buffer_len, W._local_dp_buffer_len, W._dp_max_padding, W._global_num_tokens,
            W._global_num_tokens_gpu) == (48, 6, True, [6] * 8, gpu_d)
    assert fwd.is_extend_in_batch is False
    assert dpa._dp_gather_fp8_bufs is d_bufs
    # the next decode step rewrites the D state (a different batch size)
    W.set_dp_buffer_len(64, 8, True, [8] * 8, gpu_d)
    # the next prefill split (no ForwardBatch rebuild) reads the P state
    ns["set_pdmux_status"](True)
    assert (W._global_dp_buffer_len, W._local_dp_buffer_len, W._dp_max_padding, W._global_num_tokens,
            W._global_num_tokens_gpu) == (2048, 2048, False, p_sizes, gpu_p)
    assert fwd.is_extend_in_batch is True
    assert dpa._dp_gather_fp8_bufs is p_bufs
    ns["set_pdmux_status"](False)
    assert (W._global_dp_buffer_len, W._local_dp_buffer_len) == (64, 8)


def test_same_lane_status_writes_do_not_swap(patched_ps, fake_modules):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    _m_layout(ns)
    W = fake_modules.dpa._DpGatheredBufferWrapper
    W.set_dp_buffer_len(48, 6, True, [6] * 8, None)
    ns["set_pdmux_status"](False)                         # the pdmux graph capture's set_pdmux_status(False)
    ns["set_pdmux_status"](False)
    assert W._global_dp_buffer_len == 48
    ns["set_pdmux_status"](True)
    W.set_dp_buffer_len(100, 100, False, [100] + [0] * 7, None)
    ns["set_pdmux_status"](True)
    assert W._global_dp_buffer_len == 100


def test_module_loaded_after_first_switch_gets_per_lane_dicts(patched_ps, fake_modules, monkeypatch):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    _m_layout(ns)
    ns["set_pdmux_status"](True)
    ns["set_pdmux_status"](False)
    rbg = types.ModuleType("sglang.srt.layers.numinous_dp_router_piggyback")
    rbg._bufs = {}
    monkeypatch.setitem(sys.modules, rbg.__name__, rbg)    # imported lazily by the first D-lane forward
    d = rbg._bufs
    d["k"] = "D wire"
    ns["set_pdmux_status"](True)
    assert rbg._bufs == {} and rbg._bufs is not d
    rbg._bufs["k"] = "P wire"
    ns["set_pdmux_status"](False)
    assert rbg._bufs is d and d["k"] == "D wire"


def test_lane_state_env_off(patched_ps, fake_modules, monkeypatch):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    _m_layout(ns)
    monkeypatch.setenv("NUMINOUS_PDMUX_LANE_STATE", "0")
    W = fake_modules.dpa._DpGatheredBufferWrapper
    W.set_dp_buffer_len(48, 6, True, [6] * 8, None)
    ns["set_pdmux_status"](True)
    assert W._global_dp_buffer_len == 48                  # stock: nothing carried
    assert ns["get_moe_ep_group"]() is ns["_PDMUX_PREFILL_TP_GROUP"]   # the twins stay on


def test_lane_index(patched_ps, fake_modules):
    ns = _load(patched_ps, FUNCS, with_tail=True)
    _m_layout(ns)
    assert ns["numinous_pdmux_lane"]() == 0
    ns["set_pdmux_status"](True)
    assert ns["numinous_pdmux_lane"]() == 1


# ------------------------------------------------------------------------------------------------------- preflight
M6_ARGS = {"tp_size": 8, "dp_size": 8, "enable_dp_attention": True, "ep_size": 8, "moe_a2a_backend": "none",
           "pp_size": 1, "enable_two_batch_overlap": False}


def test_preflight_inert_without_pdmux():
    assert hk.numinous_preflight(types.SimpleNamespace(args=dict(M6_ARGS), env={})) == []


def test_preflight_m_layout_with_pdmux_is_clean():
    assert hk.numinous_preflight(types.SimpleNamespace(args=dict(M6_ARGS, enable_pdmux=True), env={})) == []


@pytest.mark.parametrize("change,needle", [
    ({"moe_a2a_backend": "deepep"}, "single-instance dispatch buffers"),
    ({"ep_size": 4}, "separate MoE-EP group"),
    ({"dp_size": 4}, "attention-TP width 2"),
    ({"enable_two_batch_overlap": True}, "two-batch overlap"),
    ({"pp_size": 2}, "pipeline parallelism"),
])
def test_preflight_refusals(change, needle):
    probs = hk.numinous_preflight(types.SimpleNamespace(args=dict(M6_ARGS, enable_pdmux=True, **change), env={}))
    assert any(needle in p for p in probs), probs


def test_preflight_lane_state_off_refused():
    probs = hk.numinous_preflight(types.SimpleNamespace(args=dict(M6_ARGS, enable_pdmux=True),
                                                        env={"NUMINOUS_PDMUX_LANE_STATE": "0"}))
    assert any("NUMINOUS_PDMUX_LANE_STATE=0" in p for p in probs)
