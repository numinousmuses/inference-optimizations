"""The measured-tactic table: lookup policy, validity and shape guards (no GPU)."""
import json

import pytest

import numinous_moe_tactics as nmt

SHAPE = {"experts": 384, "hidden": 7168, "intermediate_per_rank": 384, "top_k": 6}


@pytest.fixture
def table(tmp_path, monkeypatch):
    rows = [{"tokens": 64, "tactic": [8, 3]}, {"tokens": 512, "tactic": [16, 5]}, {"tokens": 768, "tactic": [16, 7]},
            {"tokens": 1024, "tactic": [32, 2]}, {"tokens": 16384, "tactic": [192, 2], "speedup": 0.986},
            {"tokens": 32768, "tactic": [256, 1], "speedup": 1.2}]
    (tmp_path / "pro_mxfp4_b200.json").write_text(json.dumps({"sm": "sm100", "shape": SHAPE, "table": rows}))
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_DIR", str(tmp_path))
    nmt._TABLES.clear()
    return nmt.load_table("b200")


class Runner:
    hidden_size, intermediate_size, num_local_experts, top_k, num_fused_shared_experts = 7168, 384, 384, 6, 0

    def __init__(self, valid):
        self.valid = valid

    def get_valid_tactics(self, inputs, profile):
        return self.valid


def test_gpu_keys():
    assert nmt.gpu_key((10, 0)) == "b200" and nmt.gpu_key((10, 3)) == "b300" and nmt.gpu_key((9, 0)) == ""


@pytest.mark.parametrize("n,want", [(1, [8, 3]), (64, [8, 3]), (65, [16, 5]), (768, [16, 7]), (700, [16, 7]),
                                    (1000, [32, 2]), (40000, [256, 1]), (16384, None), (12000, None)])
def test_lookup_takes_the_smallest_measured_count_at_or_above(table, n, want):
    assert nmt.lookup(table, n) == want


def test_choose_serves_a_valid_tactic(table):
    assert nmt.choose(Runner([[8, 3], [16, 7], [32, 2]]), [], table, 768) == [16, 7]


def test_choose_defers_when_the_tactic_is_not_valid_at_this_size(table):
    assert nmt.choose(Runner([[8, 3], [32, 2]]), [], table, 768) is None


def test_choose_defers_for_another_shape(table):
    r = Runner([[16, 7]])
    r.intermediate_size = 256                       # e.g. a different TP split
    assert nmt.choose(r, [], table, 768) is None
    r = Runner([[16, 7]])
    r.num_fused_shared_experts = 1
    assert nmt.choose(r, [], table, 768) is None


def test_missing_table_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_DIR", str(tmp_path))
    nmt._TABLES.clear()
    assert nmt.load_table("b300") is None and nmt.load_table("") is None


def test_rows_that_did_not_beat_flashinfer_defer(table):
    assert nmt.lookup(table, 16000) is None
    assert nmt.choose(Runner([[192, 2]]), [], table, 16000) is None


def test_every_layout_table_for_the_gpu_loads(tmp_path, monkeypatch):
    for name, inter, experts in (("pro_mxfp4", 384, 384), ("pro_mxfp4_ep8", 3072, 48)):
        (tmp_path / f"{name}_b300.json").write_text(json.dumps({
            "layout": "ep8" if "ep8" in name else "tp8", "shape": dict(SHAPE, intermediate_per_rank=inter, experts=experts),
            "table": [{"tokens": 768, "tactic": [32, 1]}]}))
    (tmp_path / "pro_mxfp4_b200.json").write_text(json.dumps({"shape": SHAPE, "table": []}))
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_DIR", str(tmp_path))
    nmt._TABLES.clear()
    tables = nmt.load_tables("b300")
    assert sorted(t["layout"] for t in tables) == ["ep8", "tp8"]
    r = Runner([[32, 1]])
    r.intermediate_size, r.num_local_experts = 3072, 48
    assert [t["layout"] for t in tables if nmt.shape_matches(r, t)] == ["ep8"]


FS_SHAPE = {"experts": 385, "global_experts": 385, "hidden": 7168, "intermediate_per_rank": 384, "top_k": 7,
            "fused_shared": 1}


def _runner(local, top_k, nfse=0, num_experts=None, hidden=7168, inter=384):
    r = Runner([[16, 7]])
    r.hidden_size, r.intermediate_size, r.num_local_experts, r.top_k = hidden, inter, local, top_k
    r.num_fused_shared_experts = nfse
    if num_experts is not None:
        r.num_experts = num_experts
    return r


def test_fused_shared_table_matches_both_fusion_forms_and_nothing_else():
    fs = {"shape": FS_SHAPE}
    plain = {"shape": dict(SHAPE, global_experts=384)}
    sglang_fused = _runner(385, 7, 0, 385)          # v0.5.20: shared weights appended as expert 384, top-k 7
    flashinfer_fused = _runner(384, 6, 1, 384)      # FlashInfer num_fused_shared_experts=1: raw counts
    unfused = _runner(384, 6, 0, 384)
    assert nmt.shape_matches(sglang_fused, fs) and nmt.shape_matches(flashinfer_fused, fs)
    assert not nmt.shape_matches(unfused, fs)
    assert nmt.shape_matches(unfused, plain)
    assert not nmt.shape_matches(sglang_fused, plain) and not nmt.shape_matches(flashinfer_fused, plain)


def test_old_tables_without_fused_or_global_keys_still_match_plain_runners():
    r = _runner(384, 6, 0)                          # a runner without num_experts (older FlashInfer)
    assert nmt.shape_matches(r, {"shape": SHAPE})
    assert nmt.shape_matches(_runner(384, 6, 0, 384), {"shape": SHAPE})


def test_global_expert_count_must_agree_when_both_record_it():
    t = {"shape": dict(SHAPE, global_experts=384)}
    assert not nmt.shape_matches(_runner(384, 6, 0, 512), t)


def test_runner_missing_shape_attributes_never_matches():
    class Bare:
        pass
    assert nmt.shape_matches(Bare(), {"shape": SHAPE}) is False


def test_four_gpu_glm_layouts_load_and_select_by_shape(tmp_path, monkeypatch):
    layouts = {"glm_nvfp4": (256, 256), "glm_nvfp4_ep8": (32, 2048), "glm_nvfp4_tp4": (256, 512),
               "glm_nvfp4_ep4": (64, 2048)}
    for name, (experts, inter) in layouts.items():
        (tmp_path / f"{name}_b300.json").write_text(json.dumps({
            "file": name, "shape": {"experts": experts, "global_experts": 256, "hidden": 6144,
                                    "intermediate_per_rank": inter, "top_k": 8, "fused_shared": 0},
            "table": [{"tokens": 256, "tactic": [8, 1], "speedup": 1.2}]}))
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_DIR", str(tmp_path))
    nmt._TABLES.clear()
    tables = nmt.load_tables("b300")
    assert sorted(t["file"] for t in tables) == sorted(layouts)
    for name, (experts, inter) in layouts.items():
        r = _runner(experts, 8, 0, 256, hidden=6144, inter=inter)
        assert [t["file"] for t in tables if nmt.shape_matches(r, t)] == [name]


def test_fused_table_is_chosen_for_fused_runner_among_all_pro_tables(tmp_path, monkeypatch):
    shapes = {"pro_mxfp4": dict(SHAPE, global_experts=384), "pro_mxfp4_ep8": dict(SHAPE, experts=48,
              intermediate_per_rank=3072, global_experts=384), "pro_mxfp4_fs": FS_SHAPE}
    for name, shape in shapes.items():
        (tmp_path / f"{name}_b200.json").write_text(json.dumps(
            {"file": name, "shape": shape, "table": [{"tokens": 768, "tactic": [16, 7], "speedup": 1.3}]}))
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_DIR", str(tmp_path))
    nmt._TABLES.clear()
    tables = nmt.load_tables("b200")
    for runner, want in ((_runner(385, 7, 0, 385), "pro_mxfp4_fs"), (_runner(384, 6, 1, 384), "pro_mxfp4_fs"),
                         (_runner(384, 6, 0, 384), "pro_mxfp4"), (_runner(48, 6, 0, 384, inter=3072), "pro_mxfp4_ep8")):
        assert [t["file"] for t in tables if nmt.shape_matches(runner, t)] == [want]
        assert nmt.choose(runner, [], next(t for t in tables if t["file"] == want), 700) == [16, 7]


def test_install_serves_matching_calls_and_defers_otherwise(tmp_path, monkeypatch):
    import sys
    import types
    import torch
    (tmp_path / "pro_mxfp4_b200.json").write_text(json.dumps({"shape": SHAPE, "table": [
        {"tokens": 1024, "tactic": [32, 2], "speedup": 1.5}]}))
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_DIR", str(tmp_path))
    nmt._TABLES.clear()

    class AutoTuner:
        is_tuning_mode = False

        def choose_one(self, custom_op, runners, tuning_config, inputs, **kw):
            return "flashinfer", None

    fake = types.ModuleType("flashinfer.autotuner")
    fake.AutoTuner = AutoTuner
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: calls.append(1) or (10, 0))
    monkeypatch.setattr(nmt, "_INSTALLED", False)
    monkeypatch.setattr(nmt, "_rows", lambda inputs: inputs[0])
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_SCOPE", "all")
    assert nmt.install() and not calls                      # no CUDA call at install
    tuner, r = AutoTuner(), Runner([[32, 2]])
    assert tuner.choose_one(nmt.OP, [r], None, [768]) == (r, [32, 2])
    assert calls == [1]                                     # resolved once, on the first MoE call
    assert tuner.choose_one("flashinfer::other_op", [r], None, [768]) == ("flashinfer", None)
    other = Runner([[32, 2]])
    other.hidden_size = 6144
    assert tuner.choose_one(nmt.OP, [other], None, [768]) == ("flashinfer", None)
    tuner.is_tuning_mode = True
    assert tuner.choose_one(nmt.OP, [r], None, [768]) == ("flashinfer", None)


def test_capture_only_mode_saves_routing_and_leaves_the_choice_to_flashinfer(tmp_path, monkeypatch):
    import sys
    import types
    import torch

    class AutoTuner:
        is_tuning_mode = False

        def choose_one(self, custom_op, runners, tuning_config, inputs, **kw):
            return "flashinfer", None

    fake = types.ModuleType("flashinfer.autotuner")
    fake.AutoTuner = AutoTuner
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake)
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS", "0")
    monkeypatch.setenv("NUMINOUS_MOE_CAPTURE_DIR", str(tmp_path / "cap"))
    monkeypatch.setenv("NUMINOUS_MOE_CAPTURE_PER_BUCKET", "2")
    monkeypatch.setattr(nmt, "_INSTALLED", False)
    nmt._CAPTURED.clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ids = torch.arange(40, dtype=torch.int32).view(10, 4)
    inputs = [None, None, ids, None, torch.zeros(10, 8)]
    assert nmt.install()
    for _ in range(3):
        assert AutoTuner().choose_one(nmt.OP, [Runner([])], None, inputs) == ("flashinfer", None)
    saved = sorted((tmp_path / "cap").glob("topk_r0_n10_*.pt"))
    assert len(saved) == 2 and torch.equal(torch.load(saved[0]), ids)


def test_graph_scope_serves_only_captured_calls(monkeypatch):
    import torch
    monkeypatch.delenv("NUMINOUS_MOE_TACTICS_SCOPE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert nmt.in_scope() is False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert nmt.in_scope() is True
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_SCOPE", "all")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert nmt.in_scope() is True


# ------------------------------------------------------------------------------------------ committed tables (CPU)
import pathlib  # noqa: E402

TABLE_DIR = pathlib.Path(nmt.__file__).with_name("moe_tactics")
# file stem -> (hidden, intermediate per rank, local experts incl. fused shared, top-k incl. shared, global, shared)
EXPECTED = {
    "pro_mxfp4": (7168, 384, 384, 6, 384, 0), "pro_mxfp4_ep8": (7168, 3072, 48, 6, 384, 0),
    "pro_mxfp4_fs": (7168, 384, 385, 7, 385, 1),
    "glm_nvfp4": (6144, 256, 256, 8, 256, 0), "glm_nvfp4_ep8": (6144, 2048, 32, 8, 256, 0),
    "glm_nvfp4_tp4": (6144, 512, 256, 8, 256, 0), "glm_nvfp4_ep4": (6144, 2048, 64, 8, 256, 0),
}


@pytest.mark.parametrize("gpu", ["b200", "b300"])
@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_committed_table_is_graph_timed_and_well_formed(stem, gpu):
    f = TABLE_DIR / f"{stem}_{gpu}.json"
    assert f.exists(), f"missing {f.name}"
    t = json.loads(f.read_text())
    hidden, inter, experts, top_k, glob, shared = EXPECTED[stem]
    s = t["shape"]
    assert (s["hidden"], s["intermediate_per_rank"], s["experts"], s["top_k"], s["global_experts"],
            s.get("fused_shared", 0)) == (hidden, inter, experts, top_k, glob, shared)
    assert t["file"] == stem and t["gpu"].lower() == gpu and t["sm"] == {"b200": "sm100", "b300": "sm103"}[gpu]
    assert t["timing"].startswith("cuda_graph") and not t.get("partial") and t["flashinfer"].startswith("0.6.18")
    toks = [r["tokens"] for r in t["table"]]
    assert toks == sorted(set(toks)) and len(toks) >= 30
    for r in t["table"]:
        assert len(r["tactic"]) == 2 and all(isinstance(v, int) for v in r["tactic"])
        assert r["us"] > 0 and abs(r["fallback_us"] / r["us"] - r["speedup"]) < 0.01


@pytest.mark.parametrize("gpu", ["b200", "b300"])
def test_each_runner_shape_selects_exactly_its_committed_table(gpu, monkeypatch):
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_DIR", str(TABLE_DIR))
    nmt._TABLES.clear()
    tables = nmt.load_tables(gpu)
    for stem, (hidden, inter, experts, top_k, glob, shared) in EXPECTED.items():
        for nfse in {0, shared}:                    # fused tables: SGLang-level and FlashInfer-level fusion
            r = _runner(experts - nfse, top_k - nfse, nfse, glob - nfse, hidden=hidden, inter=inter)
            assert [t["file"] for t in tables if nmt.shape_matches(r, t)] == [stem], (stem, nfse)
    nmt._TABLES.clear()


# ------------------------------------------------------------------------------------------ GPU (dev box)
def _gpu_ready() -> bool:
    try:
        import torch
        if not torch.cuda.is_available() or nmt.gpu_key(torch.cuda.get_device_capability()) == "":
            return False
        import flashinfer  # noqa: F401  (ptxas path as in the engine)
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


gpu = pytest.mark.skipif(not _gpu_ready(), reason="needs a B200/B300 with FlashInfer TRT-LLM MoE")

# name: (nvfp4, routed experts, routed top-k, fused shared, local experts incl. shared, intermediate, hidden)
GPU_CASES = {"pro_fs": (False, 384, 6, 1, 385, 384, 7168), "pro_tp8": (False, 384, 6, 0, 384, 384, 7168),
             "glm_tp4": (True, 256, 8, 0, 256, 512, 6144), "glm_ep4": (True, 256, 8, 0, 64, 2048, 6144)}


def _moe_case(name, g):
    """Real FlashInfer routed FP4 MoE call at a served per-GPU shape, called the way SGLang v0.5.20 calls it."""
    import torch
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType
    nv, E, topk, sh, L, inter, H = GPU_CASES[name]
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(1)
    blk = 16 if nv else 32

    def scales(*shape):
        if nv:
            return (torch.rand(*shape, device=dev, generator=gen) * 0.02 + 0.01).to(torch.float8_e4m3fn)
        return torch.randint(112, 120, shape, dtype=torch.uint8, device=dev, generator=gen).view(torch.float8_e4m3fn)

    w13 = torch.randint(0, 256, (L, 2 * inter, H // 2), dtype=torch.uint8, device=dev, generator=gen)
    w2 = torch.randint(0, 256, (L, H, inter // 2), dtype=torch.uint8, device=dev, generator=gen)
    w13_s, w2_s = scales(L, 2 * inter, H // blk), scales(L, H, inter // blk)
    if nv:
        x = torch.randint(0, 256, (g, H // 2), dtype=torch.uint8, device=dev, generator=gen)
        xs = scales(g, H // 16)
    else:
        x = (torch.randn(g, H, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
        xs = torch.full((g, H // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    ids = torch.argsort(torch.rand(g, E, device=dev, generator=gen), 1)[:, :topk].to(torch.int32)
    if sh:
        ids = torch.cat([ids, torch.full((g, sh), E, dtype=torch.int32, device=dev)], 1)
    ids = ids.contiguous()
    wts = torch.softmax(torch.randn(g, topk + sh, device=dev, generator=gen), -1).to(torch.bfloat16)
    ones = torch.ones(L, dtype=torch.float32, device=dev)
    clamp = None if nv else torch.full((L,), 10.0, dtype=torch.float32, device=dev)
    out = torch.empty(g, H, dtype=torch.bfloat16, device=dev)

    def call():
        extra = {"activation_type": ActivationType.Swiglu.value} if nv else {}
        trtllm_fp4_block_scale_routed_moe(
            topk_ids=(ids, wts), routing_bias=None, hidden_states=x, hidden_states_scale=xs,
            gemm1_weights=w13, gemm1_weights_scale=w13_s, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
            gemm1_clamp_limit=clamp, gemm2_weights=w2, gemm2_weights_scale=w2_s, gemm2_bias=None,
            output1_scale_scalar=ones, output1_scale_gate_scalar=ones, output2_scale_scalar=ones,
            num_experts=E + sh, top_k=topk + sh, n_group=0 if nv else 1, topk_group=0 if nv else 1,
            intermediate_size=inter, local_expert_offset=0, local_num_experts=L,
            routed_scaling_factor=None if nv else 1.0,
            routing_method_type=1 if nv else int(RoutingMethodType.TopK.value), do_finalize=True,
            tune_max_num_tokens=1 << max(0, g - 1).bit_length(), output=out, enable_pdl=True, **extra)
        return out
    shape = {"experts": L, "global_experts": E + sh, "hidden": H, "intermediate_per_rank": inter,
             "top_k": topk + sh, "fused_shared": sh}
    return call, shape, out


@gpu
@pytest.mark.parametrize("name", sorted(GPU_CASES))
def test_gpu_hook_serves_the_table_tactic_inside_graph_capture_only(name, tmp_path, monkeypatch):
    """The installed hook on the real FlashInfer op: the runner exposes the attributes shape_matches reads; a
    CUDA-graph-captured call runs the table's tactic (and gives the fallback's result), an eager call keeps
    FlashInfer's fallback, and a table for another shape (the plain 384/6 table vs the fused call) is never used."""
    import torch
    from flashinfer.autotuner import AutoTuner
    g = 768
    call, shape, out = _moe_case(name, g)
    monkeypatch.setattr(AutoTuner, "choose_one", AutoTuner.choose_one)   # restored after the test
    orig = AutoTuner.choose_one
    seen: dict = {}

    def probe(self, custom_op, runners, tuning_config, inputs, **kw):
        res = orig(self, custom_op, runners, tuning_config, inputs, **kw)
        if custom_op == nmt.OP:
            seen["runner"], seen["inputs"], seen["tactic"] = runners[0], inputs, res[1]
        return res

    AutoTuner.choose_one = probe
    ref = call().clone()
    torch.cuda.synchronize()
    r = seen["runner"]
    assert nmt.effective_shape(r) == {k: shape[k] for k in ("hidden", "intermediate_per_rank", "experts", "top_k",
                                                              "global_experts")}
    assert seen["tactic"] == -1                                      # autotune off: FlashInfer's fallback
    valid = [list(v) for v in r.get_valid_tactics(seen["inputs"], None)]
    want = valid[len(valid) // 2]
    decoy = valid[0] if valid[0] != want else valid[-1]
    key = nmt.gpu_key(torch.cuda.get_device_capability())
    (tmp_path / f"t_{key}.json").write_text(json.dumps({"shape": shape, "table": [
        {"tokens": g, "tactic": want, "speedup": 1.5}]}))
    other = dict(shape, experts=shape["experts"] - shape["fused_shared"], top_k=shape["top_k"] - shape["fused_shared"],
                 global_experts=shape["global_experts"] - shape["fused_shared"], fused_shared=0) \
        if shape["fused_shared"] else dict(shape, intermediate_per_rank=shape["intermediate_per_rank"] * 2)
    (tmp_path / f"s_{key}.json").write_text(json.dumps({"shape": other, "table": [
        {"tokens": g, "tactic": decoy, "speedup": 1.5}]}))
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS_DIR", str(tmp_path))
    monkeypatch.delenv("NUMINOUS_MOE_TACTICS_SCOPE", raising=False)
    monkeypatch.delenv("NUMINOUS_MOE_CAPTURE_DIR", raising=False)
    monkeypatch.setenv("NUMINOUS_MOE_TACTICS", "1")
    monkeypatch.setattr(nmt, "_INSTALLED", False)
    nmt._TABLES.clear()
    nmt._SERVED.clear()
    assert nmt.install()
    hooked = AutoTuner.choose_one
    served = []

    def spy(self, custom_op, runners, tuning_config, inputs, **kw):
        res = hooked(self, custom_op, runners, tuning_config, inputs, **kw)
        if custom_op == nmt.OP:
            served.append((torch.cuda.is_current_stream_capturing(), res[1]))
        return res

    AutoTuner.choose_one = spy
    eager = call().clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    graph_out = out.clone()
    assert served[0] == (False, -1) and served[1] == (False, -1)     # eager calls: FlashInfer's choice
    assert served[2] == (True, want)                                  # captured: the table's tactic
    torch.testing.assert_close(eager, ref)
    finite = torch.isfinite(ref.float())
    assert finite.float().mean() > 0.99
    err = (graph_out.float() - ref.float())[finite].norm() / ref.float()[finite].norm()
    assert err < 2e-2, err
    nmt._TABLES.clear()

