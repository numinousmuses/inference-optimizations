"""CPU tests of the w4a8dg boot self-test guard (numinous_w4a8dg_selftest.py) and its wiring in glm_w4a8_tiles_hooks:
the decision rule, the parent side (subprocess on the rank's GPU, timeout, never raises, once per process), the
dispatcher adopting the decision, and the hooked w4afp8.py calling it after the scales are set.

    python3 -m pytest -q operations/kernels/w4a8dg/test_w4a8dg_selftest.py
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "patches"))
import numinous_w4a8dg_selftest as st  # noqa: E402
import glm_w4a8_tiles_hooks as th  # noqa: E402

SRC = pathlib.Path(os.path.expanduser("~/.cache/numinous/sgl-v0520/python/sglang"))


def us(rows):
    return {m: dict(zip(("tiles", "dg", "dg_silu"), v)) for m, v in rows.items()}


def test_decide_crossover_and_silu():
    # dg wins from 4K up, silu wins from 4K and beats dg there -> silu from 4096
    d = st.decide(us({2048: (300, 320, 310), 4096: (500, 450, 420), 8192: (900, 800, 760), 16384: (1700, 1400, 1350)}),
                  True, 5e-4, True)
    assert d["ok"] and d["min_m"] == 4096 and d["silu"] and d["min_m_dg"] == 4096
    # a win at 2K that is lost at 4K does not count: the crossover must hold at every larger m
    d = st.decide(us({2048: (300, 280, None), 4096: (500, 520, None), 8192: (900, 800, None)}), True, None, False)
    assert d["ok"] and d["min_m"] == 8192 and not d["silu"]
    # tiles everywhere -> off
    d = st.decide(us({2048: (300, 320, 330), 16384: (1700, 1800, 1750)}), True, 1e-4, True)
    assert not d["ok"] and d["min_m"] is None
    # silu off when its error is above the tolerance, even if faster
    d = st.decide(us({4096: (500, 450, 400), 16384: (1700, 1400, 1300)}), True, 5e-3, True)
    assert d["ok"] and d["min_m"] == 4096 and not d["silu"]
    # silu slower than dg alone -> dg without silu
    d = st.decide(us({4096: (500, 450, 470), 16384: (1700, 1400, 1450)}), True, 1e-4, True)
    assert d["ok"] and not d["silu"]
    # not bit-identical -> off whatever the speed
    assert not st.decide(us({4096: (500, 100, 90)}), False, 0.0, True)["ok"]


def test_min_m_is_converted_to_the_dispatchers_tokens():
    # EP8 top-8: rows = tokens; EP4 top-8: rows = 2 x tokens; TP (ep 1) top-8: rows = 8 x tokens
    assert st.tokens_for(4096, 8, 8) == 4096 and st.tokens_for(4096, 4, 8) == 2048
    assert st.tokens_for(16384, 1, 8) == 2048 and st.tokens_for(None, 8, 8) is None


def test_child_spec_strips_selftest():
    assert "silu" not in th.dg_env("ep8")["NUMINOUS_W4A8_DG"]          # measured slower: off by default
    spec = th.dg_env("ep8", silu=50)["NUMINOUS_W4A8_DG"]
    assert "selftest=1" in spec and "min_m=1" in spec and "silu=50" in spec
    kv, shapes = st.parse_spec(st.child_spec(spec))
    assert "selftest" not in kv and kv["min_m"] == "1" and kv["silu"] == "50"
    assert shapes == {(4096, 6144): 40, (6144, 2048): 40}


def test_summary_line():
    line = st.summary_line({"ok": True, "min_m": 4096, "silu": True, "exact": True, "silu_rel": 3e-4,
                            "us": {"2048": {"tiles": 300, "dg": 320, "dg_silu": 310},
                                   "16384": {"tiles": 1700, "dg": 1400, "dg_silu": 1350}}, "elapsed_s": 31.0}, "7")
    assert line.startswith("gpu=7 ok=True min_m=4096 silu=True") and "m2048=300/320/310 m16384=1700/1400/1350" in line


def test_child_paths():
    spec = st.child_spec(th.dg_env("ep8", silu=50)["NUMINOUS_W4A8_DG"])
    assert st.path_spec(spec, False, False) is None
    kv, shapes = st.parse_spec(st.path_spec(spec, True, False))
    assert "silu" not in kv and kv["min_m"] == "1" and len(shapes) == 2
    kv, _ = st.parse_spec(st.path_spec(spec, True, True))
    assert kv["silu"] == "50" and "selftest" not in kv


class FakeT:
    def __init__(self, shape, index=3):
        self.shape = shape
        self.device = types.SimpleNamespace(index=index)


def fake_layer():
    return types.SimpleNamespace(w13_weight=FakeT((32, 4096, 3072)), w2_weight=FakeT((32, 6144, 1024)), top_k=8,
                                 moe_ep_size=8)


def test_parent_runs_child_once_on_the_rank_gpu(monkeypatch):
    monkeypatch.setenv(st.ENV, th.dg_env("ep8", silu=50)["NUMINOUS_W4A8_DG"])
    monkeypatch.delenv(st.DECISION, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    seen = []

    def run(cmd, env=None, capture_output=None, text=None, timeout=None):
        seen.append((cmd, env, timeout))
        return types.SimpleNamespace(returncode=0, stdout="noise\n" + json.dumps({"ok": True, "min_m": 4096,
                                                                                  "silu": True}) + "\n", stderr="")
    monkeypatch.setattr(st.subprocess, "run", run)
    dec = st.selftest_once(fake_layer())
    assert dec["ok"] and dec["min_m"] == 4096
    cmd, env, timeout = seen[0]
    assert cmd[1:3] == ["-m", "sglang.srt.layers.moe.numinous_w4a8dg_selftest"]
    args = dict(zip(cmd[4::2], cmd[5::2]))
    assert args == {"--experts": "32", "--n13": "4096", "--k": "6144", "--n2": "2048", "--topk": "8", "--ep": "8",
                    "--silu": "1"}
    assert env["CUDA_VISIBLE_DEVICES"] == "7"                   # device index 3 of 4,5,6,7
    assert "selftest" not in env[st.ENV] and timeout == 240
    assert json.loads(os.environ[st.DECISION])["min_m"] == 4096
    assert st.selftest_once(fake_layer()) is None and len(seen) == 1   # once per process


@pytest.mark.parametrize("mode", ["timeout", "crash", "garbage"])
def test_parent_failure_leaves_a_not_ok_decision(monkeypatch, mode):
    monkeypatch.setenv(st.ENV, th.dg_env("ep8")["NUMINOUS_W4A8_DG"])
    monkeypatch.delenv(st.DECISION, raising=False)

    def run(*a, **k):
        if mode == "timeout":
            raise subprocess.TimeoutExpired("x", 1)
        if mode == "crash":
            raise OSError("no python")
        return types.SimpleNamespace(returncode=139, stdout="Segmentation fault\n", stderr="illegal address")
    monkeypatch.setattr(st.subprocess, "run", run)
    dec = st.selftest_once(fake_layer())
    assert dec["ok"] is False and json.loads(os.environ[st.DECISION])["ok"] is False


def test_parent_inert_without_selftest(monkeypatch):
    monkeypatch.setenv(st.ENV, th.dg_env("ep8", min_m=2048)["NUMINOUS_W4A8_DG"])
    monkeypatch.delenv(st.DECISION, raising=False)
    monkeypatch.setattr(st.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    assert st.selftest_once(fake_layer()) is None and st.DECISION not in os.environ


def dispatcher(monkeypatch, tmp_path, spec, decision):
    import hashlib
    lib = tmp_path / "k3amd_w4a8dg.so"
    lib.write_bytes(b"fake")
    spec = spec.replace(f"so={th.DG_SO}", f"so={lib}").replace(f"sha={th.DG_SHA256}",
                                                                 f"sha={hashlib.sha256(b'fake').hexdigest()}")
    monkeypatch.setenv("NUMINOUS_W4A8_DG", spec)
    if decision is None:
        monkeypatch.delenv(st.DECISION, raising=False)
    else:
        monkeypatch.setenv(st.DECISION, json.dumps(decision))
    for k in ("NUMINOUS_W4A8_TILES_SO", "NUMINOUS_W4A8_TRITON"):
        monkeypatch.delenv(k, raising=False)
    ops = types.SimpleNamespace(load_library=lambda p: None, k3amd_w4a8dg=types.SimpleNamespace())
    ns = {"torch": types.SimpleNamespace(ops=ops), "cutlass_w4a8_moe_mm": lambda *a: None}
    exec(th.DISPATCH.replace("{marker}", "t"), ns)
    return ns["_nm_load_dg"]()


def test_dispatcher_adopts_the_decision(monkeypatch, tmp_path):
    spec = th.dg_env("ep8", silu=50)["NUMINOUS_W4A8_DG"]
    assert dispatcher(monkeypatch, tmp_path, spec, {"ok": True, "min_m": 8192, "silu": True})[0::2] == (8192, 50)
    assert dispatcher(monkeypatch, tmp_path, spec, {"ok": True, "min_m": 4096, "silu": False})[0::2] == (4096, None)
    for dec in ({"ok": False, "reason": "x"}, None, {"ok": True, "min_m": None}):
        m, shapes, silu = dispatcher(monkeypatch, tmp_path, spec, dec)
        assert shapes == {}                                    # no decision / not ok -> the tiles path
    fixed = th.dg_env("ep8", min_m=2048, silu=None)["NUMINOUS_W4A8_DG"]
    assert dispatcher(monkeypatch, tmp_path, fixed, None)[0] == 2048   # explicit min_m: no self-test involved


@pytest.mark.skipif(not SRC.is_dir(), reason="needs operations/hopper/restore_cpu_preflight.sh")
def test_hooked_w4afp8_calls_the_selftest_after_the_scales(tmp_path):
    root = tmp_path / "sglang"
    for sub in ("srt/layers/quantization", "srt/layers/moe"):
        shutil.copytree(SRC / sub, root / sub)
    th.apply(str(root))
    w4 = (root / "srt/layers/quantization/w4afp8.py").read_text()
    fn = w4[w4.index("def process_weights_after_loading"):]
    i_os, i_w2, i_st = fn.index("import os as _nm_os"), fn.index("layer.w2_weight_scale_inv = Parameter"), \
        fn.index("selftest_once as _nm_selftest")
    assert i_os < i_w2 < i_st                      # _nm_os is bound before the trigger, the scales are final
    assert (root / "srt/layers/moe/numinous_w4a8dg_selftest.py").read_text() == (HERE / "numinous_w4a8dg_selftest.py"
                                                                                  ).read_text()
    compile(w4, "w4afp8.py", "exec")


def test_build_finds_a_bundled_cutlass(tmp_path):
    """build_so falls back to a CUTLASS tree inside the image when the box has no internet."""
    import build_so
    base = tmp_path / "site/flashinfer/data/cutlass"
    (base / "include/cute/arch").mkdir(parents=True)
    (base / "include/cute/arch/mma_sm90_gmma.hpp").write_text("")
    (base / "include/cutlass/arch").mkdir(parents=True)
    (base / "include/cutlass/arch/barrier.h").write_text("")
    assert build_so.bundled_cutlass([tmp_path / "site"]) == base
    assert build_so.bundled_cutlass([tmp_path / "nothing"]) is None
