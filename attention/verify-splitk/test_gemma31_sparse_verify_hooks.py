"""CPU tests of rtxsys D2 (gemma31_sparse_verify hook).

- The hook applies to SGLang v0.5.20 eagle_worker_v2.py: anchors once, compiles, idempotent.
- The host-side max-top-k rule.
- The compact normalisation the hook routes verify sampling into (uno_utils._normalize_sparse_topk_probs, cut from the
  v0.5.20 source) equals the dense branch's distribution: softmax(l / T), then top_k_renorm, then top_p_renorm.
- The preflight flags inert setups.
"""
from __future__ import annotations

import ast
import os
import shutil
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gemma31_sparse_verify_hooks as sv  # noqa: E402

SRC = os.path.expanduser("~/.cache/numinous/sgl-v0520/python/sglang")
need_tree = pytest.mark.skipif(not os.path.isdir(SRC), reason="needs operations/hopper/restore_cpu_preflight.sh")


@need_tree
def test_hook_applies_and_is_idempotent(tmp_path):
    root = tmp_path / "sglang"
    shutil.copytree(os.path.join(SRC, "srt/speculative"), root / "srt/speculative",
                    ignore=shutil.ignore_patterns("__pycache__"))
    out = sv.apply(str(root))
    assert out == [f"helper: applied to {sv.PATH}", f"verify: applied to {sv.PATH}"]
    assert sv.apply(str(root)) == ["helper: already applied", "verify: already applied"]
    src = (root / sv.PATH).read_text()
    assert src.count(sv.MARKER) == 2
    compile(src, "eagle_worker_v2.py", "exec")
    # the verify call now passes the helper's value through run_eagle_verify's existing parameter
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "run_eagle_verify"]
    assert any(any(k.arg == "uno_target_max_top_k" for k in c.keywords) for c in calls)


def _helper(on: bool):
    ns: dict = {}
    text = sv.HELPER.replace("{marker}", sv.MARKER).replace("class EAGLEWorkerV2(", "")
    os.environ[sv.ENV] = "1" if on else "0"
    try:
        exec(compile(text, "helper", "exec"), ns)
    finally:
        os.environ.pop(sv.ENV, None)
    return ns["_nsv_max_top_k"]


def _batch(top_ks):
    return types.SimpleNamespace(reqs=[types.SimpleNamespace(sampling_params=types.SimpleNamespace(top_k=k))
                                       for k in top_ks])


def test_max_top_k_rule():
    f = _helper(True)
    assert f(_batch([64, 64, 1, 40])) == 64           # model defaults; a greedy request (top_k 1) mixes in
    assert f(_batch([64, 1 << 30])) is None            # one request without top-k (TOP_K_ALL) -> dense branch
    assert f(_batch([129])) is None                    # beyond uno_utils._SPARSE_TOP_K_LIMIT
    assert f(_batch([128])) == 128
    assert f(_batch([])) is None and f(types.SimpleNamespace()) is None
    assert _helper(False)(_batch([64])) is None        # env off: stock call


def _sparse_normalize():
    """uno_utils._normalize_sparse_topk_probs from the v0.5.20 source (the module imports flashinfer)."""
    src = open(os.path.join(SRC, "srt/speculative/uno_utils.py")).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_normalize_sparse_topk_probs")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "uno_utils", "exec"), ns)
    return ns["_normalize_sparse_topk_probs"]


def _dense_reference(logits, temps, top_ks, top_ps):
    """eagle_utils.py:880-898: softmax(l / T) -> top_k_renorm_prob -> top_p_renorm_prob (sorting form)."""
    p = torch.softmax(logits / temps, dim=-1)
    rows, v = p.shape
    out = torch.zeros_like(p)
    for i in range(rows):
        vals, idx = p[i].sort(descending=True)
        k = int(top_ks[i])
        vals = vals[:k] / vals[:k].sum()                       # top-k renorm
        excl = torch.cumsum(vals, 0) - vals
        vals = torch.where(excl > top_ps[i], torch.zeros_like(vals), vals)   # top-p over the survivors
        out[i, idx[:k]] = vals / vals.sum()
    return out


@need_tree
def test_compact_distribution_equals_dense_branch():
    norm = _sparse_normalize()
    g = torch.Generator().manual_seed(0)
    rows, vocab, kmax = 12, 4096, 64
    logits = 30.0 * torch.tanh(torch.randn(rows, vocab, generator=g) * 3 / 30.0)   # softcap-30-shaped logits
    temps = torch.full((rows, 1), 0.6)
    temps[3] = 1.0
    top_ks = torch.full((rows,), 64, dtype=torch.int64)
    top_ks[5] = 7
    top_ks[6] = 1                                                    # a greedy row
    top_ps = torch.full((rows,), 0.95)
    top_ps[7] = 0.5
    top_ps[8] = 1.0
    dense = _dense_reference(logits, temps, top_ks, top_ps)
    vals, ids = torch.topk(logits, kmax, dim=-1, sorted=True)       # flashinfer top_k(sorted=True) stand-in
    ranks = torch.arange(kmax)[None, :]
    sparse = norm(vals, temps, ranks < top_ks[:, None], top_ps[:, None])
    scattered = torch.zeros_like(dense).scatter_(1, ids, sparse)
    assert torch.allclose(scattered, dense, atol=1e-6, rtol=1e-5)
    assert int((scattered[6] > 0).sum()) == 1 and int(scattered[6].argmax()) == int(logits[6].argmax())


def test_preflight_flags():
    ctx = types.SimpleNamespace(env={sv.ENV: "1"}, args={"speculative_use_rejection_sampling": True,
                                                        "sampling_defaults": "openai"},
                                device={"is_hip": False})
    out = sv.numinous_preflight(ctx)
    assert any("without speculative decoding" in x for x in out)
    assert any("rejection" in x for x in out) and any("sampling-defaults" in x for x in out)
    ctx.env = {}
    assert sv.numinous_preflight(ctx) == []
