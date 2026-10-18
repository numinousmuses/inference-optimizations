"""moe_shared_post_reorder: the fused finalize kernel vs a torch reference (Triton interpreter), the stash protocol, and
the hooks on the v0.5.20 tree (apply once, idempotent, compile, compose with moe_src2dst_fused_hooks)."""
from __future__ import annotations

import os
import pathlib
import shutil
import sys

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "operations/kernels"))
import moe_post_reorder_shared as K  # noqa: E402
import moe_shared_post_reorder_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python/sglang"))


def _case(tokens=37, topk=4, experts=8, hidden=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    topk_ids = torch.randint(0, experts + 1, (tokens, topk), generator=g, dtype=torch.int32)   # == experts: skipped
    weights = torch.rand(tokens, topk, generator=g)
    src2dst = torch.randperm(tokens * topk, generator=g).to(torch.int32).view(tokens, topk)
    down = (torch.randn(tokens * topk, hidden, generator=g) * 0.5).to(torch.bfloat16)
    shared = (torch.randn(tokens, hidden, generator=g) * 0.5).to(torch.bfloat16)
    return topk_ids, weights, src2dst, down, shared


def _reference(topk_ids, weights, src2dst, down, shared, experts, scale):
    tokens, topk = topk_ids.shape
    out = torch.zeros(tokens, down.shape[1])
    for t in range(tokens):
        for j in range(topk):
            if int(topk_ids[t, j]) != experts:
                out[t] += down[int(src2dst[t, j])].float() * float(weights[t, j])
    return (out * scale + shared.float()).to(torch.bfloat16)


@pytest.mark.parametrize("hidden", [96, 1100])          # one and several hidden blocks per token
def test_fused_finalize_matches_reference(hidden):
    experts, scale = 8, 2.5
    topk_ids, weights, src2dst, down, shared = _case(hidden=hidden)
    out = torch.empty_like(shared)
    K.post_reorder_shared(down, out, src2dst, topk_ids, weights, experts, topk_ids.shape[1], topk_ids.shape[0], hidden,
                          scale, shared, sm_count=4)
    ref = _reference(topk_ids, weights, src2dst, down, shared, experts, scale)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), (out.float() - ref.float()).abs().max()


def test_launch_config_matches_stock_shape_rules():
    (gy, gx), bd = K.launch_config(None, 16384, 6144, sm_count=132)
    assert bd in (512, 1024) and gx == -(-6144 // bd) and 1 <= gy <= 16384
    (gy, gx), bd = K.launch_config(None, 1, 6144, sm_count=132)
    assert gy == 1


def test_stash_protocol():
    K.clear()
    assert K.take() is None and not K.consumed()
    t = torch.ones(2, 2)
    K.stash(t)
    assert K.take() is t and K.take() is None
    K.mark_consumed()
    assert K.consumed() and not K.consumed()               # read-once


def _copy(tmp):
    rels = ("srt/models/deepseek_v2.py", "srt/layers/moe/cutlass_w4a8_moe.py", "kernels/ops/moe/ep_moe_kernels.py")
    if not all((SRC / r).exists() for r in rels):
        pytest.skip("SGLang v0.5.20 source absent")
    root = tmp / "sglang"
    for r in rels:
        (root / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / r, root / r)
    return root


def test_hooks_apply_once_compile_and_place(tmp_path):
    root = _copy(tmp_path)
    assert H.apply(str(root), check=True) == ["model: anchor ok in srt/models/deepseek_v2.py",
                                              "runner: anchor ok in srt/layers/moe/cutlass_w4a8_moe.py"]
    out = H.apply(str(root))
    assert out[:2] == ["model: applied to srt/models/deepseek_v2.py",
                       "runner: applied to srt/layers/moe/cutlass_w4a8_moe.py"]
    assert (root / H.MODULE[1]).exists()
    assert H.apply(str(root), install_module=False) == ["model: already applied", "runner: already applied"]
    m = (root / "srt/models/deepseek_v2.py").read_text()
    fn = m[m.index("    def forward_normal("):]
    fn = fn[: fn.index("\n    def ", 10)]
    i_stash, i_call = fn.index("_npr.stash(shared_output)"), fn.index("final_hidden_states = self.experts(")
    i_cons, i_add = fn.index("if _npr.consumed():"), fn.index("maybe_fuse_routed_scale_and_shared_add(")
    assert i_stash < i_call < i_cons < i_add
    r = (root / "srt/layers/moe/cutlass_w4a8_moe.py").read_text()
    assert r.count("_npr.post_reorder_shared(") == 1 and r.count("post_reorder_for_cutlass_moe(\n") >= 1


def test_composes_with_moe_src2dst_fused(tmp_path):
    import moe_src2dst_fused_hooks as S
    root = _copy(tmp_path)
    S.apply(str(root)) if hasattr(S, "apply") else S.main(["--root", str(root)])
    H.apply(str(root))
    for r in ("srt/models/deepseek_v2.py", "srt/layers/moe/cutlass_w4a8_moe.py"):
        compile((root / r).read_text(), r, "exec")
    text = (root / "srt/layers/moe/cutlass_w4a8_moe.py").read_text()
    assert S.MARK in text and H.MARKER in text
