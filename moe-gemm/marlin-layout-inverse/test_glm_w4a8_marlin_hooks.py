"""The Marlin decode hook applies cleanly to SGLang v0.5.20 (anchors exist once, idempotent, patched file compiles)."""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/.cache/numinous/sgl-v0520/python/sglang")


@pytest.mark.skipif(not os.path.isdir(SRC), reason="needs operations/hopper/restore_cpu_preflight.sh")
def test_hook_applies_to_v0520_and_is_idempotent(tmp_path):
    root = tmp_path / "sglang"
    shutil.copytree(os.path.join(SRC, "srt/layers/quantization"), root / "srt/layers/quantization")
    run = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, "glm_w4a8_marlin_hooks.py"), "--root",  # noqa: E731
                                     str(root), *a], capture_output=True, text=True, check=True).stdout
    assert "anchor ok" in run("--check")
    first = run()
    assert "applied to" in first and "installed" in first
    assert "already applied" in run()
    src = (root / "srt/layers/quantization/w4afp8.py").read_text()
    assert src.count("numinous-hook:glm_w4a8_marlin") == 2 and "fused_marlin_moe(" in src
    compile(src, "w4afp8.py", "exec")
    assert (root / "srt/layers/quantization/glm_w4a8_marlin.py").exists()


@pytest.mark.skipif(not os.path.isdir(SRC), reason="needs operations/hopper/restore_cpu_preflight.sh")
def test_dispatch_branch_is_in_the_patched_apply_and_installs_the_inverse(tmp_path):
    root = tmp_path / "sglang"
    shutil.copytree(os.path.join(SRC, "srt/layers/quantization"), root / "srt/layers/quantization")
    out = subprocess.run([sys.executable, os.path.join(HERE, "glm_w4a8_marlin_hooks.py"), "--root", str(root)],
                         capture_output=True, text=True, check=True).stdout
    assert "installed srt/layers/quantization/glm_marlin_inverse.py" in out
    src = (root / "srt/layers/quantization/w4afp8.py").read_text()
    compile(src, "w4afp8.py", "exec")
    # the size test sits before the Marlin branch, the cutlass call reads the scratch, never the Marlin tensors
    i_disp, i_marlin = src.index("x.shape[0] > _nm_d.threshold"), src.index("fused_marlin_moe(\n")
    assert i_disp < i_marlin
    call = src[i_disp:i_marlin]
    assert "_nm_d.cutlass_weights(layer)" in call and "x, _nm_w13, _nm_w2, _nm_s13, _nm_s2" in call
    assert "layer.w13_weight," not in call
    # dispatch keeps the stock input-scale processing the early return would skip
    assert 'if _nm_mode == "dispatch":' in src and "layer.w13_input_scale = Parameter(torch.tensor(" in src


def test_dispatch_scratch_rebuilds_the_cutlass_tensors(monkeypatch):
    """CPU (TRITON_INTERPRET): a layer holding Marlin tensors (the transcribed forward repack + marlin_permute_scales)
    gets back exactly the cutlass_w4a8 bytes and the scales interleave_scales would make."""
    os.environ.setdefault("TRITON_INTERPRET", "1")
    import types

    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "kernels"))
    import glm_marlin_inverse as gmi
    from glm_w4a8_marlin import pack_int4_pairs, scales_to_marlin_input
    monkeypatch.setenv("NUMINOUS_W4A8_DISPATCH_M", "512")
    monkeypatch.setenv("NUMINOUS_W4A8_TILES_ALIGN", "128x256=1")          # w13 shape tuned, w2 on the stock rule
    e, n13, k13, n2, k2 = 2, 128, 256, 256, 64 * 2
    rng = np.random.default_rng(5)
    q13 = rng.integers(-8, 8, size=(e, n13, k13))
    q2 = rng.integers(-8, 8, size=(e, n2, k2))

    def marlin_w(q):
        w = np.stack([gmi.marlin_forward_reference(q[x].T) for x in range(q.shape[0])])
        return torch.from_numpy(w.view(np.int32).reshape(q.shape[0], q.shape[2] // 16, 2 * q.shape[1]).copy())

    def marlin_s(raw):
        mi = scales_to_marlin_input(raw)
        return torch.stack([mi[x].reshape(-1, 64)[:, gmi.scale_perm()].reshape(mi.shape[1], mi.shape[2])
                            for x in range(raw.shape[0])])
    s13 = torch.randn(e, n13, k13 // 128).to(torch.bfloat16)
    s2 = torch.randn(e, n2, k2 // 128).to(torch.bfloat16)
    layer = types.SimpleNamespace(w13_weight=marlin_w(q13), w2_weight=marlin_w(q2), w13_weight_scale_inv=marlin_s(s13),
                                  w2_weight_scale_inv=marlin_s(s2))
    gmi._SCRATCH.clear()
    sc = gmi.dispatch_scratch(e, n13, k13, n2, k2, "cpu")
    assert sc.threshold == 512 and (sc.a13, sc.a2) == (1, 1)
    assert gmi.dispatch_scratch(e, n13, k13, n2, k2, "cpu") is sc               # one scratch per shape and device
    w13c, w2c, s13c, s2c = sc.cutlass_weights(layer)
    assert torch.equal(w13c, pack_int4_pairs(torch.from_numpy(q13).to(torch.int8)))
    assert torch.equal(w2c, pack_int4_pairs(torch.from_numpy(q2).to(torch.int8)))
    assert torch.equal(s13c, s13.permute(0, 2, 1))                               # alignment 1 (tuned)
    assert torch.equal(s2c, gmi.cutlass_scales(scales_to_marlin_input(s2), gmi.stock_alignment(k2 // 128)))


def test_dispatch_preflight():
    sys.path.insert(0, HERE)
    import types

    import glm_w4a8_marlin_hooks as h
    ctx = lambda env, **a: types.SimpleNamespace(env=env, args={"quantization": "w4afp8", **a}, checkpoint_quant="")  # noqa: E731
    assert h.numinous_preflight(ctx({"NUMINOUS_W4A8_MARLIN": "dispatch", "NUMINOUS_W4A8_DISPATCH_M": "2048"})) == []
    assert any("NUMINOUS_W4A8_DISPATCH_M" in p for p in h.numinous_preflight(ctx({"NUMINOUS_W4A8_MARLIN": "dispatch"})))
    assert any("expected 1" in p for p in h.numinous_preflight(ctx({"NUMINOUS_W4A8_MARLIN": "yes"})))
    assert h.numinous_preflight(ctx({"NUMINOUS_W4A8_MARLIN": "1"}, ep_size=8)) == []        # EP is correct (moe-r5)
    assert any("moe_a2a_backend" in p for p in h.numinous_preflight(ctx({"NUMINOUS_W4A8_MARLIN": "1"},
                                                                        moe_a2a_backend="deepep")))
