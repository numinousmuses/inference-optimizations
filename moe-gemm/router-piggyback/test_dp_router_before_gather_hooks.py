"""Router before the DP gather: wire pack + fused dequant/split vs reference (TRITON_INTERPRET), the rank-uniform width
rule, the stash protocol, and the hooks (anchors, compile, idempotence, placement, composition with the other GLM
glue hooks) on the SGLang v0.5.20 tree."""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import types

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
pytest.importorskip("triton")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kernels"))
import dp_router_before_gather_hooks as H  # noqa: E402
import dp_router_piggyback as P  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python/sglang"))


def test_wire_roundtrip_matches_stock_dequant_and_carries_topk():
    g = torch.Generator().manual_seed(0)
    rows, hidden, group, k = 5, 384, 128, 8
    ng = hidden // group
    q = (torch.randn(rows, hidden, generator=g) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.rand(rows, ng, generator=g) + 0.01
    w = torch.rand(rows, k, generator=g)
    ids = torch.randint(0, 256, (rows, k), generator=g, dtype=torch.int32)
    wide = P.pack_scales(s, ng, w, ids)
    assert wide.shape == (rows, ng + 2 * P.KMAX) and torch.equal(wide[:, :ng], s)
    out = torch.empty(rows, hidden, dtype=torch.bfloat16)
    ow, oid = P.dequant_and_split(q.view(torch.uint8), wide, out, ng, group, k, True)
    exact = q.float() * s.repeat_interleave(group, dim=1)
    ref = exact.to(torch.bfloat16)
    # The Triton interpreter's fp32 -> bf16 cast truncates (the GPU rounds to nearest even), so allow one bf16 ulp
    # here. The GPU microbench checks bit equality against the stock _dequant_per_token_group_fp8_kernel.
    assert torch.allclose(out.float(), exact, rtol=2 ** -7, atol=0)
    assert torch.equal(ow, w) and torch.equal(oid, ids)
    out2 = torch.empty_like(out)
    assert P.dequant_and_split(q.view(torch.uint8), P.pack_scales(s, ng), out2, ng, group, 0, False) == (None, None)
    assert torch.equal(out2, out) and ref.shape == out.shape


def _mlp(top_k=8, fmt="STANDARD", fused=0, is_hash=False):
    from enum import Enum

    class F(Enum):
        STANDARD = 1
        BYPASSED = 2
    cfg = types.SimpleNamespace(top_k=top_k, output_format=F[fmt], num_fused_shared_experts=fused)
    return types.SimpleNamespace(topk=types.SimpleNamespace(topk_config=cfg), is_hash=is_hash), F


def test_static_width_is_config_only(monkeypatch):
    mlp, F = _mlp()
    fake = types.ModuleType("sglang.srt.layers.moe.topk")
    fake.TopKOutputFormat = F
    moe = types.ModuleType("sglang.srt.layers.moe")
    moe.get_moe_runner_backend = lambda: (_ for _ in ()).throw(AssertionError("explicit format must not consult it"))
    for name, mod in (("sglang", types.ModuleType("sglang")), ("sglang.srt", types.ModuleType("sglang.srt")),
                      ("sglang.srt.layers", types.ModuleType("sglang.srt.layers")), ("sglang.srt.layers.moe", moe),
                      ("sglang.srt.layers.moe.topk", fake)):
        monkeypatch.setitem(sys.modules, name, mod)
    assert P.static_topk_width(mlp) == 8
    assert P.static_topk_width(_mlp(fmt="BYPASSED")[0]) == 0
    assert P.static_topk_width(_mlp(top_k=9)[0]) == 0
    assert P.static_topk_width(_mlp(fused=1)[0]) == 0
    assert P.static_topk_width(_mlp(is_hash=True)[0]) == 0
    assert P.static_topk_width(None) == 0


def test_stash_protocol():
    a, b = object(), object()
    out = types.SimpleNamespace(topk_ids=torch.zeros(7, 8))
    P.set_pending(a)
    assert P.take_pending() is a and P.take_pending() is None
    P.stash(a, out)
    assert P.take(b, 7) is None and P.take(a, 7) is None          # wrong module consumed the read-once stash
    P.stash(a, out)
    assert P.take(a, 6) is None
    P.stash(a, out)
    assert P.take(a, 7) is out and P.take(a, 7) is None
    P.stash(a, out)
    P.set_pending(b)                                              # the next layer's mark drops a stale stash
    assert P.take(a, 7) is None


def _tree(tmp_path):
    if not (SRC / H.DSV2).exists():
        pytest.skip("SGLang v0.5.20 source absent")
    root = tmp_path / "sglang"
    for rel in (H.DP, H.DSV2, "srt/layers/moe/cutlass_w4a8_moe.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / rel, root / rel)
    return root


def test_hooks_apply_compile_place(tmp_path):
    root = _tree(tmp_path)
    assert all("anchor ok" in x for x in H.apply(str(root), check=True))
    out = H.apply(str(root))
    assert [x.split(":")[0] for x in out] == ["gather", "pending", "consume", "lazy_moe_entry", "lazy_normal_check",
                                              "lazy_runner", "module"]
    assert all("already applied" in x for x in H.apply(str(root), install_module=False))
    dp = (root / H.DP).read_text()
    ds = (root / H.DSV2).read_text()
    compile(dp, H.DP, "exec")
    compile(ds, H.DSV2, "exec")
    assert (root / H.MODULE_DST).exists()
    fn = dp[dp.index("def _dp_gather_via_all_gatherv_fp8("):]
    assert fn.index("gather_fp8_with_router") < fn.index("sglang_per_token_group_quant_fp8(")
    i_set = ds.index("_nm_rbg.set_pending(")
    i_prep = ds.index("self.layer_communicator.prepare_mlp(", i_set)
    assert i_set < i_prep < ds.index("_nm_rbg.clear_pending()", i_prep)
    fwd = ds[ds.index("    def forward_normal("):]
    assert fwd.index("_nm_rbg.take(self, hidden_states.shape[0])") < fwd.index(
        "router_logits = self.gate(hidden_states, gemm_output_zero_allocator)")


def test_composes_with_glue_hooks(tmp_path):
    root = _tree(tmp_path)
    import mla_q_cat_skip_hooks as Q
    import moe_shared_post_reorder_hooks as S
    rel = Q.REL
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / rel, root / rel)
    S.apply(str(root))
    Q.apply(str(root))
    H.apply(str(root))
    ds = (root / H.DSV2).read_text()
    compile(ds, H.DSV2, "exec")
    assert ds.count("numinous-hook:dp_router_before_gather (consume)") == 1
    assert "numinous-hook:moe_shared_post_reorder" in ds


def _quant_ref(x, group=128):
    """sglang_per_token_group_quant_fp8 semantics on CPU: s = amax/448 per group, q = e4m3(x/s)."""
    t, h = x.shape
    xg = x.float().view(t, h // group, group)
    s = (xg.abs().amax(-1) / 448.0).clamp(min=1e-10)
    q = (xg / s.unsqueeze(-1)).to(torch.float8_e4m3fn).view(t, h)
    return q, s.contiguous()


def test_gather_end_to_end_single_rank_with_passthrough(monkeypatch):
    """gather_fp8_with_router on CPU with a 1-rank group: bf16 buffer filled as the stock dequant would, topk stash
    equals the module's own router on the local rows, and rtxsys's passthrough sees the same (q, s)."""
    import importlib.util
    os.environ["NUMINOUS_DP_FP8_PASSTHROUGH"] = "1"
    spec = importlib.util.spec_from_file_location("numinous_fp8_passthrough",
                                                  HERE.parent / "kernels" / "dp_fp8_passthrough.py")
    nfp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nfp)
    monkeypatch.setattr(nfp, "_stream_id", lambda: 0)
    from collections import namedtuple
    from enum import Enum
    STO = namedtuple("StandardTopKOutput", "topk_weights topk_ids router_logits")

    class F(Enum):
        STANDARD = 1

    class Group:
        def all_gatherv(self, x, sizes, output):
            assert sizes == [x.shape[0]]
            output.copy_(x)

    mods = {
        "sglang": types.ModuleType("sglang"), "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.layers": types.ModuleType("sglang.srt.layers"),
        "sglang.srt.layers.moe": types.ModuleType("sglang.srt.layers.moe"),
        "sglang.srt.layers.moe.topk": types.ModuleType("sglang.srt.layers.moe.topk"),
        "sglang.srt.distributed": types.ModuleType("sglang.srt.distributed"),
        "sglang.kernels": types.ModuleType("sglang.kernels"), "sglang.kernels.ops": types.ModuleType("sglang.kernels.ops"),
        "sglang.kernels.ops.quantization": types.ModuleType("sglang.kernels.ops.quantization"),
        "sglang.kernels.ops.quantization.fp8_kernel": types.ModuleType("sglang.kernels.ops.quantization.fp8_kernel"),
    }
    mods["sglang.srt.layers.moe.topk"].StandardTopKOutput = STO
    mods["sglang.srt.layers.moe.topk"].TopKOutputFormat = F
    mods["sglang.srt.layers.moe"].get_moe_runner_backend = lambda: None
    mods["sglang.srt.distributed"].get_tp_group = lambda: Group()
    mods["sglang.kernels.ops.quantization.fp8_kernel"].sglang_per_token_group_quant_fp8 = _quant_ref
    mods["sglang.srt.layers"].numinous_fp8_passthrough = nfp
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.numinous_fp8_passthrough", nfp)

    g = torch.Generator().manual_seed(1)
    t, hidden, experts, k = 6, 256, 16, 4
    x = torch.randn(t, hidden, generator=g).bfloat16()
    wgate = torch.randn(experts, hidden, generator=g).bfloat16()

    class Gate:
        def __call__(self, h):
            return h.float() @ wgate.float().t()

    class TopK:
        topk_config = types.SimpleNamespace(top_k=k, output_format=F.STANDARD, num_fused_shared_experts=0)

        def __call__(self, h, logits):
            w, i = torch.topk(torch.sigmoid(logits), k, dim=-1)
            return STO(w / w.sum(-1, keepdim=True), i.to(torch.int32), logits)

    mlp = types.SimpleNamespace(gate=Gate(), topk=TopK(), is_hash=False)
    glob = torch.empty(t, hidden, dtype=torch.bfloat16)
    P.set_pending(mlp)
    nfp.clear()
    P.gather_fp8_with_router(glob, x, [t], 128)

    q, s = _quant_ref(x)
    exact = q.float() * s.repeat_interleave(128, dim=1)
    assert torch.allclose(glob.float(), exact, rtol=2 ** -7, atol=0)       # interpreter bf16 cast truncates
    got = P.take(mlp, t)
    want = mlp.topk(x, mlp.gate(x))
    assert got is not None and torch.equal(got.topk_ids, want.topk_ids) and torch.equal(got.topk_weights, want.topk_weights)
    pq, ps = nfp.for_linear(glob, 128)                                      # rtxsys's shared-expert consumer
    assert torch.equal(pq.view(torch.uint8), q.view(torch.uint8)) and torch.equal(ps, s)
    assert ps.stride() == (1, (t + 3) // 4 * 4)                             # column-major TMA-aligned, as DeepGEMM wants
    os.environ.pop("NUMINOUS_DP_FP8_PASSTHROUGH", None)


def test_preflights_flag_inert_configurations():
    import moe_post_reorder_shared as S
    cfg = types.SimpleNamespace(num_experts_per_tok=8)
    good = {"enable_dp_attention": True, "tp_size": 8, "dp_size": 8, "moe_runner_backend": "auto"}
    env = {"NUMINOUS_DP_ROUTER_BEFORE_GATHER": "1", "SGLANG_DP_USE_GATHERV": "1", "SGLANG_ENABLE_DP_GATHER_FP8": "1"}
    ctx = lambda args, e, q="w4afp8": types.SimpleNamespace(args=args, env=e, hf_config=cfg, checkpoint_quant=q)
    assert P.numinous_preflight(ctx(good, env)) == []
    assert P.numinous_preflight(ctx(good, {})) == []                                   # lever off: nothing to check
    assert any("FP8 gatherv" in m for m in P.numinous_preflight(ctx(good, {**env, "SGLANG_ENABLE_DP_GATHER_FP8": "0"})))
    assert any("tp == dp" in m for m in P.numinous_preflight(ctx({**good, "dp_size": 4}, env)))
    assert any("non-STANDARD" in m for m in P.numinous_preflight(ctx({**good, "moe_runner_backend": "flashinfer_trtllm"}, env)))
    assert any("eplb" in m for m in P.numinous_preflight(ctx({**good, "enable_eplb": True}, env)))
    senv = {"NUMINOUS_MOE_SHARED_IN_POST_REORDER": "1"}
    assert S.numinous_preflight(ctx(good, senv)) == []
    assert any("TP1" in m for m in S.numinous_preflight(ctx(good, {**senv, "SGLANG_SHARED_EXPERT_TP1": "1"})))
    assert any("W4A8" in m for m in S.numinous_preflight(ctx(good, senv, q="fp8")))


def test_pre_reorder_fp8_matches_stock_on_the_dequantized_buffer():
    """pre_reorder_fp8 (FP8 payload + wide group scales) vs the stock pre_reorder_triton_kernel_for_cutlass_moe on
    the bf16-dequantized buffer, both under TRITON_INTERPRET, against the fp32 reference."""
    if not (SRC / "kernels/ops/moe/ep_moe_kernels.py").exists():
        pytest.skip("SGLang v0.5.20 source absent")
    import triton
    import triton.language as tl
    text = (SRC / "kernels/ops/moe/ep_moe_kernels.py").read_text()
    i = text.index("@triton.jit\ndef pre_reorder_triton_kernel_for_cutlass_moe(")
    body = text[i:text.index("\n\n\ndef ", i)]
    # the stock kernel verbatim, except the int64 cast the interpreter cannot apply to its Python-int loop index
    assert body.count("src_idx = src_idx_int32.to(tl.int64)") == 1
    body = body.replace("src_idx = src_idx_int32.to(tl.int64)", "src_idx = src_idx_int32")
    ns = {"triton": triton, "tl": tl}
    exec(compile(body, "ep_moe_kernels.py:pre_reorder_triton_kernel_for_cutlass_moe", "exec"), ns)
    ep = types.SimpleNamespace(pre_reorder_triton_kernel_for_cutlass_moe=ns["pre_reorder_triton_kernel_for_cutlass_moe"])
    g = torch.Generator().manual_seed(3)
    t, hidden, group, topk, experts = 5, 256, 128, 4, 8
    ng = hidden // group
    q = (torch.randn(t, hidden, generator=g) * 60).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.rand(t, ng, generator=g) * 0.02 + 0.001
    wide = P.pack_scales(s, ng)
    topk_ids = torch.randint(0, experts, (t, topk), generator=g, dtype=torch.int32)
    topk_ids[1, 2] = experts                                                  # a non-local id (EP sentinel): skipped
    src2dst = torch.randperm(t * topk, generator=g).to(torch.int32).view(t, topk)
    a1 = torch.tensor([0.05])
    x_bf16 = (q.float() * s.repeat_interleave(group, dim=1)).to(torch.bfloat16)
    # fp32 destination buffers: both kernels cast to the destination dtype, and the Triton interpreter's fp32 -> e4m3
    # cast is not IEEE (1.94 -> 1.0), so the FP8 cast is left out here. On the GPU it is the same instruction in both
    # kernels. What this checks: indexing (src2dst, topk, the EP sentinel), the wide-scale stride, and the math.
    out_stock = torch.zeros(t * topk, hidden, dtype=torch.float32)
    out_new = torch.zeros_like(out_stock)
    grid, block = (t, 2), 128
    ep.pre_reorder_triton_kernel_for_cutlass_moe[grid](x_bf16, out_stock, src2dst, topk_ids, a1, experts, topk, t,
                                                        hidden, BLOCK_SIZE=block, NUM_STAGES=1)
    P.pre_reorder_fp8(q, wide, ng, group, out_new, src2dst, topk_ids, a1, experts, topk, t, hidden, grid=grid,
                      block_dim=block)
    ref = q.float() * s.repeat_interleave(group, dim=1) * (1.0 / a1)
    for i in range(t):
        for j in range(topk):
            d = int(src2dst[i, j])
            if int(topk_ids[i, j]) == experts:
                assert torch.all(out_new[d] == 0) and torch.all(out_stock[d] == 0)
                continue
            assert torch.allclose(out_new[d], ref[i], rtol=1e-6, atol=0), (i, j)      # exact fp32 math
            assert torch.allclose(out_stock[d], ref[i], rtol=2 ** -8, atol=0), (i, j)  # stock: one bf16 rounding


def _fake_env(monkeypatch, quant_method_name="W4AFp8MoEMethod"):
    """The 1-rank fake sglang of test_gather_end_to_end_single_rank_with_passthrough, returning (nfp, mlp, STO)."""
    import importlib.util
    from collections import namedtuple
    from enum import Enum
    monkeypatch.setenv("NUMINOUS_DP_FP8_PASSTHROUGH", "1")
    spec = importlib.util.spec_from_file_location("numinous_fp8_passthrough",
                                                  HERE.parent / "kernels" / "dp_fp8_passthrough.py")
    nfp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nfp)
    monkeypatch.setattr(nfp, "_stream_id", lambda: 0)
    STO = namedtuple("StandardTopKOutput", "topk_weights topk_ids router_logits")

    class F(Enum):
        STANDARD = 1

    class Group:
        def all_gatherv(self, x, sizes, output):
            output.copy_(x)

    names = ["sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.moe", "sglang.srt.layers.moe.topk",
             "sglang.srt.distributed", "sglang.kernels", "sglang.kernels.ops", "sglang.kernels.ops.quantization",
             "sglang.kernels.ops.quantization.fp8_kernel"]
    mods = {n: types.ModuleType(n) for n in names}
    mods["sglang.srt.layers.moe.topk"].StandardTopKOutput = STO
    mods["sglang.srt.layers.moe.topk"].TopKOutputFormat = F
    mods["sglang.srt.layers.moe"].get_moe_runner_backend = lambda: None
    mods["sglang.srt.distributed"].get_tp_group = lambda: Group()
    mods["sglang.kernels.ops.quantization.fp8_kernel"].sglang_per_token_group_quant_fp8 = _quant_ref
    mods["sglang.srt.layers"].numinous_fp8_passthrough = nfp
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.numinous_fp8_passthrough", nfp)
    hidden, experts, k = 256, 16, 4
    wgate = torch.randn(experts, hidden, generator=torch.Generator().manual_seed(5)).bfloat16()

    class TopK:
        topk_config = types.SimpleNamespace(top_k=k, output_format=F.STANDARD, num_fused_shared_experts=0)

        def __call__(self, h, logits):
            w, i = torch.topk(torch.sigmoid(logits), k, dim=-1)
            return STO(w / w.sum(-1, keepdim=True), i.to(torch.int32), logits)

    qm = type(quant_method_name, (), {})()
    mlp = types.SimpleNamespace(gate=lambda h: h.float() @ wgate.float().t(), topk=TopK(), is_hash=False,
                                experts=types.SimpleNamespace(quant_method=qm), _enable_a2a_moe=False,
                                _fuse_shared_experts_inside_sbo=False)
    return nfp, mlp


def test_lazy_dequant_protocol(monkeypatch):
    nfp, mlp = _fake_env(monkeypatch)
    monkeypatch.setenv("NUMINOUS_DP_LAZY_DEQUANT", "1")
    t, hidden = 6, 256
    x = torch.randn(t, hidden, generator=torch.Generator().manual_seed(9)).bfloat16()
    q, s = _quant_ref(x)
    want = (q.float() * s.repeat_interleave(128, dim=1))

    def gather():
        glob = torch.full((t, hidden), float("nan"), dtype=torch.bfloat16)        # poisoned: an unread lazy buffer
        P.set_pending(mlp)
        nfp.clear()
        P.gather_fp8_with_router(glob, x, [t], 128)
        return glob

    # 1. all readers take FP8: the buffer stays unwritten, pre_reorder gets the payload, the next layer is clean
    glob = gather()
    assert torch.isnan(glob.float()).all() and P.is_lazy(glob)
    P.lazy_check_normal(mlp, glob, skip_shared_experts=False)
    assert P.is_lazy(glob)                                                     # router stash + passthrough present
    assert P.take(mlp, t) is not None and nfp.for_linear(glob, 128) is not None
    lq, lw, ng, grp = P.lazy_take(glob)
    assert torch.equal(lq.view(torch.uint8), q.view(torch.uint8)) and torch.equal(lw[:, :ng], s) and grp == 128
    P.set_pending(None)                                                        # taken: no raise
    # 2. the passthrough already consumed (or not stashed): forward_normal's check materializes before any reader
    glob = gather()
    nfp.for_linear(glob, 128)                                                  # 'linear' used up
    P.lazy_check_normal(mlp, glob, skip_shared_experts=False)
    assert not P.is_lazy(glob) and torch.allclose(glob.float(), want, rtol=2 ** -7, atol=0)
    # 3. a router-stash mismatch materializes too
    glob = gather()
    P.take(mlp, t)                                                             # stash consumed elsewhere
    P.lazy_check_normal(mlp, glob, skip_shared_experts=False)
    assert not P.is_lazy(glob) and not torch.isnan(glob.float()).any()
    # 4. a deferred buffer nobody took raises at the next layer instead of being served
    glob = gather()
    with pytest.raises(RuntimeError, match="unlisted reader"):
        P.set_pending(None)
    # 5. the runner handed a different tensor while a record is pending: raise
    glob = gather()
    with pytest.raises(RuntimeError, match="not the gathered buffer"):
        P.lazy_take(glob.clone())
    P._tls.lazy = None
    # 6. not eligible (another routed runner): the gather dequantizes as before
    nfp2, mlp2 = _fake_env(monkeypatch, quant_method_name="Fp8MoEMethod")
    glob = torch.full((t, hidden), float("nan"), dtype=torch.bfloat16)
    P.set_pending(mlp2)
    P.gather_fp8_with_router(glob, x, [t], 128)
    assert not P.is_lazy(glob) and torch.allclose(glob.float(), want, rtol=2 ** -7, atol=0)
    P._tls.lazy = None
    P._tls.stash = None


def test_lazy_preflight():
    cfg = types.SimpleNamespace(num_experts_per_tok=8)
    good = {"enable_dp_attention": True, "tp_size": 8, "dp_size": 8, "moe_runner_backend": "auto"}
    env = {"NUMINOUS_DP_ROUTER_BEFORE_GATHER": "1", "SGLANG_DP_USE_GATHERV": "1", "SGLANG_ENABLE_DP_GATHER_FP8": "1",
           "NUMINOUS_DP_LAZY_DEQUANT": "1", "NUMINOUS_DP_FP8_PASSTHROUGH": "1"}
    ctx = lambda e, q="w4afp8": types.SimpleNamespace(args=good, env=e, hf_config=cfg, checkpoint_quant=q)  # noqa
    assert P.numinous_preflight(ctx(env)) == []
    assert any("PASSTHROUGH" in m for m in P.numinous_preflight(ctx({**env, "NUMINOUS_DP_FP8_PASSTHROUGH": "0"})))
    assert any("MARLIN" in m for m in P.numinous_preflight(ctx({**env, "NUMINOUS_W4A8_MARLIN": "dispatch"})))
    assert any("W4A8" in m for m in P.numinous_preflight(ctx(env, q="fp8")))
