"""The tuned-tile W4A8 hook applies to SGLang v0.5.20 and its rules parse (CPU)."""
import os
import shutil
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glm_w4a8_tiles_hooks as th  # noqa: E402
import glm_w4a8_marlin_hooks as mh  # noqa: E402

SRC = os.path.expanduser("~/.cache/numinous/sgl-v0520/python/sglang")


def test_rules():
    r = th.parse_rules("1024x6144:576=9,1000000=5; 6144x512:1000000=5")
    assert r == {(1024, 6144): [(576, 9), (1000000, 5)], (6144, 512): [(1000000, 5)]}
    assert th.pick(r, 1024, 6144, 144) == 9 and th.pick(r, 1024, 6144, 16384) == 5
    assert th.pick(r, 512, 6144, 144) is None
    assert th.parse_align("1024x6144=1;6144x512=1") == {(1024, 6144): 1, (6144, 512): 1}


@pytest.mark.skipif(not os.path.isdir(SRC), reason="needs operations/hopper/restore_cpu_preflight.sh")
def test_hook_applies_with_the_marlin_hook_and_is_idempotent(tmp_path):
    root = tmp_path / "sglang"
    for sub in ("srt/layers/quantization", "srt/layers/moe"):
        shutil.copytree(os.path.join(SRC, sub), root / sub)
    for hook in (mh, th):                                    # both hooks in one image (P: decode Marlin, prefill tiles)
        assert any("applied" in x for x in hook.apply(str(root)))
        assert all("already applied" in x for x in hook.apply(str(root)) if not x.startswith("installed"))
    w4 = (root / "srt/layers/quantization/w4afp8.py").read_text()
    cm = (root / "srt/layers/moe/cutlass_w4a8_moe.py").read_text()
    # align, align2, selftest (boot self-test trigger after the w2 scales are set)
    assert w4.count("numinous-hook:glm_w4a8_tiles") == 3 and w4.count("numinous-hook:glm_w4a8_marlin") == 2
    assert "numinous_w4a8dg_selftest import selftest_once" in w4
    assert (root / "srt/layers/moe/numinous_w4a8dg_selftest.py").exists()
    assert "torch.ops.k3amd_w4a8.mm_cfg" in cm and "_nm_stock_w4a8_mm = cutlass_w4a8_moe_mm" in cm
    compile(w4, "w4afp8.py", "exec")
    compile(cm, "cutlass_w4a8_moe.py", "exec")


def test_measured_rules_are_consistent():
    """Every rule's configs share the shape's scale alignment (TileK / 128), and -1 (stock) appears only where the stock
    tile is TileK 128 (TP8 w2, K 256)."""
    sys.path.insert(0, os.path.join(HERE, "..", "kernels", "w4a8_tiles"))
    import configs
    for layout, (rules, align) in th.RULES.items():
        r, a = th.parse_rules(rules), th.parse_align(align)
        assert set(r) == set(a), layout
        for shape, ranges in r.items():
            for _, cfg in ranges:
                if cfg < 0:
                    assert shape == (6144, 256), (layout, shape)
                else:
                    assert configs.CFGS[cfg][3] // 128 == a[shape], (layout, shape, cfg)
    assert th.pick(th.parse_rules(th.RULES["tp8"][0]), 6144, 256, 16384) is None
    assert th.pick(th.parse_rules(th.RULES["ep4"][0]), 4096, 6144, 16384) == 5
    assert th.env_for("ep4")["NUMINOUS_W4A8_TILES_SO"].endswith("k3amd_w4a8.so")


def test_dispatcher_routes_by_shape_and_m(monkeypatch, tmp_path):
    """The DISPATCH text the hook inserts, executed with a fake stock op and a fake k3amd_w4a8.mm_cfg: rules pick the
    configured tile per (n_out, k, m), -1 and unruled shapes fall through to stock."""
    import types
    calls = []
    fake_torch = types.SimpleNamespace(ops=types.SimpleNamespace(
        load_library=lambda p: calls.append(("load", p)),
        k3amd_w4a8=types.SimpleNamespace(mm_cfg=lambda *a: calls.append(("cfg", a[-1])))))
    ns = {"torch": fake_torch, "cutlass_w4a8_moe_mm": lambda *a: calls.append(("stock",))}
    rules, align = th.RULES["tp8"]
    lib = tmp_path / "k3amd_w4a8.so"
    lib.write_bytes(b"not a real library: the fake load_library only records the staged path")
    monkeypatch.setenv("NUMINOUS_W4A8_TILES_SO", str(lib))
    monkeypatch.delenv("NUMINOUS_W4A8_TILES_SHA256", raising=False)
    monkeypatch.setenv("NUMINOUS_W4A8_TILES", rules)
    exec(th.DISPATCH.replace("{marker}", "t"), ns)
    T = lambda r, c: types.SimpleNamespace(shape=(r, c))  # noqa: E731
    mm = ns["cutlass_w4a8_moe_mm"]
    mm(T(1, 512), T(144 * 8, 6144), None, None, None, None, None, None, None, None, None, 128, 8)     # w13 decode
    mm(T(1, 6144), T(16384 * 8, 256), None, None, None, None, None, None, None, None, None, 128, 8)   # w2 prefill: -1
    mm(T(1, 777), T(8, 999), None, None, None, None, None, None, None, None, None, 128, 8)            # no rule
    mm(T(1, 512), T(16384 * 8, 6144), None, None, None, None, None, None, None, None, None, 128, 8)   # w13 prefill
    assert calls[0][0] == "load" and calls[0][1] != str(lib) and calls[0][1].endswith("k3amd_w4a8.so")
    assert "numinous_w4a8_tiles_" in calls[0][1]                     # loaded from the private local copy
    assert calls[1:] == [("cfg", 9), ("stock",), ("stock",), ("cfg", 5)]


def test_loaded_library_never_holds_a_volume_file_open(tmp_path, monkeypatch):
    """Arm P6 died at WEIGHTS.reload() ("there are open files preventing the operation") because the tile library
    was loaded straight from /cache. The dispatcher must stage it to local disk first: after a real dlopen of the
    library, no fd and no memory mapping of this process may point into the volume directory."""
    import ctypes
    import hashlib
    import types
    vol = tmp_path / "cache" / "k3amd_ext"
    vol.mkdir(parents=True)
    src_c = tmp_path / "lib.c"
    src_c.write_text("int k3amd_probe(void) { return 7; }\n")
    so = vol / "k3amd_w4a8.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(so), str(src_c)], check=True)
    sha = hashlib.sha256(so.read_bytes()).hexdigest()
    loaded = []

    def load_library(path):                                  # what torch.ops.load_library does: dlopen and keep it
        loaded.append(ctypes.CDLL(path))
        assert loaded[-1].k3amd_probe() == 7
    ns = {"torch": types.SimpleNamespace(ops=types.SimpleNamespace(load_library=load_library)),
          "cutlass_w4a8_moe_mm": lambda *a: None}
    monkeypatch.setenv("NUMINOUS_W4A8_TILES_SO", str(so))
    monkeypatch.setenv("NUMINOUS_W4A8_TILES_SHA256", sha)
    monkeypatch.setenv("NUMINOUS_W4A8_TILES", "512x6144:288=9")
    exec(th.DISPATCH.replace("{marker}", "t"), ns)
    ns["_nm_load_rules"]()
    assert loaded, "the library was not loaded"
    fds = [os.readlink(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")
           if os.path.exists(f"/proc/self/fd/{fd}")]
    maps = open("/proc/self/maps").read()
    assert not any(str(vol) in f for f in fds), fds
    assert str(vol) not in maps                              # mapped from the private copy, not the volume
    assert "numinous_w4a8_tiles_" in maps
    # re-reading the rules never loads a second copy (a second TORCH_LIBRARY registration aborts the process)
    n_loaded = len(loaded)
    ns["_nm_rules"] = None
    ns["_nm_load_rules"]()
    assert len(loaded) == n_loaded
    # a wrong pin refuses loudly instead of loading an unexpected library (a fresh process: nothing loaded yet)
    monkeypatch.setenv("NUMINOUS_W4A8_TILES_SHA256", "0" * 64)
    ns["_nm_rules"] = None
    ns["_nm_lib"] = None
    with pytest.raises(RuntimeError, match="sha256"):
        ns["_nm_load_rules"]()


def test_no_hook_loads_a_library_straight_from_a_path_it_did_not_stage():
    """Static guard over every hook (operations/patches): a torch.ops.load_library / ctypes.CDLL call must take a
    path staged to local disk (_nm_stage_so) - a library mapped from the weights-cache volume keeps a volume file open
    and breaks every later Volume.reload() of the process (P6, 2026-10-05)."""
    import glob
    import re
    bad = []
    for f in glob.glob(os.path.join(HERE, "*.py")):
        if os.path.basename(f).startswith("test_"):
            continue
        for i, line in enumerate(open(f, errors="ignore"), 1):
            if re.search(r"(torch\.ops\.load_library|ctypes\.CDLL)\(", line) and "_nm_stage_so(" not in line:
                bad.append(f"{os.path.basename(f)}:{i}: {line.strip()[:120]}")
    assert not bad, bad


def test_triton_prefill_path_selection(monkeypatch):
    """NUMINOUS_W4A8_TRITON routes calls with >= min_m tokens, a listed (N, K) and alignment-1 scales to
    glm_w4a8_triton_mm with the [E+1] offsets recovered from the expert_offsets[:-1] view; everything else stays on
    the cutlass dispatch."""
    import torch
    calls = []
    fake = types.ModuleType("sglang.srt.layers.moe.glm_w4a8_triton_mm")
    fake.w4a8_grouped_mm = lambda d, a, b, s, sc, offs, **kw: calls.append(("triton", offs.tolist(), kw))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.glm_w4a8_triton_mm", fake)
    stock = []
    ns = {"torch": torch, "cutlass_w4a8_moe_mm": lambda *a: stock.append("stock")}
    monkeypatch.delenv("NUMINOUS_W4A8_TILES_SO", raising=False)
    monkeypatch.setenv("NUMINOUS_W4A8_TRITON", "min_m=1024;512x6144=128,256,8,3")
    exec(th.DISPATCH.replace("{marker}", "t"), ns)
    e, n, k, topk = 4, 512, 6144, 8
    full = torch.tensor([0, 3000, 3000, 7000, 9000], dtype=torch.int32)

    def call(m, scales_shape):
        d = torch.empty(m * topk, n)
        a = torch.empty(m * topk, k)
        b = torch.empty(e, n, k // 2, dtype=torch.int8)
        ns["cutlass_w4a8_moe_mm"](d, a, b, torch.ones(1), torch.empty(scales_shape), full[:-1], None, None, None,
                                  None, None, 128, topk)
    call(2048, (e, k // 128, n))
    assert calls == [("triton", full.tolist(), {"block_n": 128, "block_t": 256, "num_warps": 8, "num_stages": 3})]
    call(512, (e, k // 128, n))                     # below min_m
    call(2048, (e, k // 128 // 4, n * 4))           # alignment-4 scales: not the kernel's layout
    assert len(calls) == 1 and stock == ["stock", "stock"]


def test_triton_preflight():
    ctx = types.SimpleNamespace(env={"NUMINOUS_W4A8_TRITON": "512x6144=128,256,8,3"},
                                args={"tp_size": 8, "ep_size": 1, "quantization": "w4afp8"},
                                hf_config=types.SimpleNamespace(moe_intermediate_size=2048, hidden_size=6144))
    probs = th.numinous_preflight(ctx)
    assert any("min_m" in p for p in probs) and any("ALIGN=1" in p for p in probs)
    ctx.env = {"NUMINOUS_W4A8_TRITON": "min_m=2048;512x6144=128,256,8,3", "NUMINOUS_W4A8_TILES_ALIGN": "512x6144=1"}
    assert th.numinous_preflight(ctx) == []


def test_w4a8dg_path_selection(monkeypatch, tmp_path):
    """NUMINOUS_W4A8_DG routes calls with >= min_m tokens, a listed (N, K) and alignment-1 scales to
    torch.ops.k3amd_w4a8dg.mm (staged, sha-pinned library; [E+1] offsets from the expert_offsets[:-1] view; the
    per-tensor activation scale as a [1] fp32 tensor); everything else stays on the cutlass dispatch."""
    import hashlib
    import torch
    lib = tmp_path / "k3amd_w4a8dg.so"
    lib.write_bytes(b"fake library: load_library is stubbed")
    sha = hashlib.sha256(lib.read_bytes()).hexdigest()
    calls, loads, stock = [], [], []

    class Ops:
        k3amd_w4a8dg = types.SimpleNamespace(mm=lambda d, a, b, s, offs, sc, cfg: calls.append((offs.tolist(),
                                                                                                 tuple(sc.shape), cfg)))

        @staticmethod
        def load_library(p):
            loads.append(p)
    fake_torch = types.SimpleNamespace(ops=Ops)
    ns = {"torch": fake_torch, "cutlass_w4a8_moe_mm": lambda *a: stock.append("stock")}
    monkeypatch.delenv("NUMINOUS_W4A8_TILES_SO", raising=False)
    monkeypatch.delenv("NUMINOUS_W4A8_TRITON", raising=False)
    monkeypatch.setenv("NUMINOUS_W4A8_DG", f"so={lib};sha={sha};min_m=2048;4096x6144=40;6144x2048=41")
    exec(th.DISPATCH.replace("{marker}", "t"), ns)
    e, topk = 4, 8
    full = torch.tensor([0, 3000, 3000, 7000, 9000], dtype=torch.int32)

    def call(m, n, k, scales_shape):
        d = torch.empty(m * topk, n)
        a = torch.empty(m * topk, k)
        b = torch.empty(e, n, k // 2, dtype=torch.int8)
        ns["cutlass_w4a8_moe_mm"](d, a, b, torch.ones(1), torch.empty(scales_shape), full[:-1], None, None, None,
                                  None, None, 128, topk)
    call(4096, 4096, 6144, (e, 48, 4096))
    call(4096, 6144, 2048, (e, 16, 6144))
    assert calls == [(full.tolist(), (1,), 40), (full.tolist(), (1,), 41)]
    assert len(loads) == 1 and loads[0] != str(lib) and "numinous_w4a8_tiles_" in loads[0]
    call(1024, 4096, 6144, (e, 48, 4096))              # below min_m
    call(4096, 4096, 6144, (e, 12, 4096 * 4))          # alignment-4 scales
    call(4096, 512, 6144, (e, 48, 512))                # unlisted shape
    assert len(calls) == 2 and stock == ["stock"] * 3
    monkeypatch.setenv("NUMINOUS_W4A8_DG", f"so={lib};sha={'0' * 64};min_m=1;4096x6144=40")
    ns2 = {"torch": fake_torch, "cutlass_w4a8_moe_mm": lambda *a: None}
    exec(th.DISPATCH.replace("{marker}", "t"), ns2)
    with pytest.raises(RuntimeError, match="sha256"):
        d = torch.empty(8, 4096)
        ns2["cutlass_w4a8_moe_mm"](d, torch.empty(8, 6144), torch.empty(e, 4096, 3072, dtype=torch.int8), torch.ones(1),
                                   torch.empty(e, 48, 4096), full[:-1], None, None, None, None, None, 128, 8)


def test_w4a8dg_preflight():
    hf = types.SimpleNamespace(moe_intermediate_size=2048, hidden_size=6144)
    ep8 = {"tp_size": 8, "ep_size": 8, "quantization": "w4afp8"}
    sm90 = {"sku": "H200", "sm_major": 9, "sm_minor": 0, "is_hip": False, "gpus": 8}
    env = {**th.env_for("ep8"), **th.dg_env("ep8", 40, 2048)}
    ctx = types.SimpleNamespace(env=env, args=ep8, hf_config=hf, device=sm90)
    assert th.numinous_preflight(ctx) == []
    assert th.parse_dg(env["NUMINOUS_W4A8_DG"])[1] == {(4096, 6144): 40, (6144, 2048): 40}
    bad = dict(env, NUMINOUS_W4A8_TILES_ALIGN="")
    probs = th.numinous_preflight(types.SimpleNamespace(env=bad, args=ep8, hf_config=hf, device=sm90))
    assert any("ALIGN=1" in p for p in probs)
    probs = th.numinous_preflight(types.SimpleNamespace(env=env, args={**ep8, "ep_size": 1}, hf_config=hf, device=sm90))
    assert any("inert" in p for p in probs)
    probs = th.numinous_preflight(types.SimpleNamespace(env=env, args=ep8, hf_config=hf,
                                                        device={**sm90, "sku": "B200", "sm_major": 10}))
    assert any("sm_90a only" in p for p in probs)
    nosha = dict(env, NUMINOUS_W4A8_DG=env["NUMINOUS_W4A8_DG"].replace(f"sha={th.DG_SHA256}", "sha=" + "1" * 64))
    probs = th.numinous_preflight(types.SimpleNamespace(env=nosha, args=ep8, hf_config=hf, device=sm90))
    assert any("build record" in p for p in probs)


@pytest.mark.skipif(not os.path.isdir(SRC), reason="needs operations/hopper/restore_cpu_preflight.sh")
def test_silu_hook_routes_gemm1(tmp_path, monkeypatch):
    """The patched cutlass_w4a8_moe (real v0.5.20 text + the hooks) executed on CPU with recording fakes: with
    NUMINOUS_W4A8_DG silu=<cfg> and a static a2_scale, GEMM1 + silu become ONE torch.ops.k3amd_w4a8dg.mm_silu call on
    the full [E+1] offsets and c1 is never allocated; without a2_scale (dynamic quant) or without silu= the stock
    GEMM1 + silu kernels run; GEMM2 and the reorders run in every case."""
    import hashlib
    import re
    import torch
    root = tmp_path / "sglang"
    shutil.copytree(os.path.join(SRC, "srt/layers/moe"), root / "srt/layers/moe")
    shutil.copytree(os.path.join(SRC, "srt/layers/quantization"), root / "srt/layers/quantization")
    th.apply(str(root))
    src = (root / "srt/layers/moe/cutlass_w4a8_moe.py").read_text()
    assert src.count("numinous-hook:glm_w4a8_tiles (silu)") == 1
    start = src.index("# numinous-hook:glm_w4a8_tiles (dispatch)")
    body = src[start:src.index("\ndef cutlass_w4a8_moe_deepep_normal(")]
    body = re.sub(r"^(\s*)import os as _nm_os", r"\1import os as _nm_os", body, flags=re.M)
    lib = tmp_path / "k3amd_w4a8dg.so"
    lib.write_bytes(b"fake")
    sha = hashlib.sha256(b"fake").hexdigest()
    log = []

    class Ops:
        k3amd_w4a8dg = types.SimpleNamespace(
            mm_silu=lambda q, a, w, s, offs, s1, s2, cfg: log.append(("mm_silu", offs.numel(), tuple(s1.shape),
                                                                       tuple(s2.shape), cfg)),
            mm=lambda *a: log.append(("dg_mm",)))

        @staticmethod
        def load_library(p):
            log.append(("load",))
    fake_torch = types.SimpleNamespace(**{k: getattr(torch, k) for k in dir(torch) if not k.startswith("_")
                                          and k != "ops"})
    fake_torch.ops = Ops
    fake_torch.Tensor = torch.Tensor
    rec = lambda name: (lambda *a, **k: log.append((name,)))  # noqa: E731
    import typing
    ns = {"torch": fake_torch, "Optional": typing.Optional, "cutlass_w4a8_moe_mm": rec("stock_mm"),
          "get_parallel": lambda: types.SimpleNamespace(moe_ep_size=1),
          "cutlass_w4_run_moe_ep_preproess": lambda ids: torch.zeros(ids.numel(), dtype=torch.int32),
          "pre_reorder_for_cutlass_moe": rec("pre"), "get_cutlass_w4a8_moe_mm_data": rec("mm_data"),
          "per_tensor_absmax_fp8": rec("absmax"), "post_reorder_for_cutlass_moe": rec("post"),
          "silu_mul_static_tensorwise_quant_for_cutlass_moe": rec("silu_static"),
          "silu_mul_dynamic_tensorwise_quant_for_cutlass_moe": rec("silu_dynamic")}
    exec(compile(body, "patched", "exec"), ns)
    e, m, topk, hid, inter = 4, 64, 2, 256, 128
    a = torch.zeros(m, hid, dtype=torch.bfloat16)
    w1 = torch.zeros(e, 2 * inter, hid // 2, dtype=torch.int8)
    w2 = torch.zeros(e, hid, inter // 2, dtype=torch.int8)
    s1 = torch.zeros(e, hid // 128, 2 * inter)            # alignment-1 [E, K/128, N]
    s2 = torch.zeros(e, inter // 128, hid)
    ids = torch.zeros(m, topk, dtype=torch.int32)
    tw = torch.zeros(m, topk)
    offs = torch.zeros(e + 1, dtype=torch.int32)
    z = torch.zeros(e, 3, dtype=torch.int64)

    def run(a2):
        log.clear()
        ns["cutlass_w4a8_moe"](a, w1, w2, s1, s2, tw, ids, z, z, z, z, z, z, z, z, offs, z, z,
                               torch.ones(1), a2, routed_scaling_factor=1.0)
        return [x[0] for x in log if x[0] != "load"]
    monkeypatch.setenv("NUMINOUS_W4A8_DG", f"so={lib};sha={sha};min_m=1;silu=50;{2 * inter}x{hid}=40;{hid}x{inter}=40")
    ns["_nm_dg"] = None
    assert run(torch.ones(1)) == ["pre", "mm_data", "mm_silu", "dg_mm", "post"]
    assert [x for x in log if x[0] == "mm_silu"] == [("mm_silu", e + 1, (1,), (1,), 50)]
    assert run(None) == ["pre", "mm_data", "dg_mm", "silu_dynamic", "dg_mm", "post"]   # dynamic a2: stock GEMM1 path
    monkeypatch.setenv("NUMINOUS_W4A8_DG", f"so={lib};sha={sha};min_m=1;{2 * inter}x{hid}=40;{hid}x{inter}=40")
    ns["_nm_dg"] = None
    assert run(torch.ones(1)) == ["pre", "mm_data", "dg_mm", "silu_static", "dg_mm", "post"]
    monkeypatch.delenv("NUMINOUS_W4A8_DG")
    ns["_nm_dg"] = None
    assert run(torch.ones(1)) == ["pre", "mm_data", "stock_mm", "silu_static", "stock_mm", "post"]
