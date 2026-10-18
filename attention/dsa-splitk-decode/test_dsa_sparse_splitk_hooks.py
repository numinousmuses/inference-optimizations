"""CPU tests for the split-top-k FlashMLA sparse wrapper (operations/kernels/dsa_sparse_splitk.py) and its hook.

The real kernel is CUDA-only, so the split/merge math runs against a torch stand-in that follows FlashMLA's documented
reference for flash_mla_sparse_fwd (README, sparse prefill: P = QK^T * sm_scale * log2(e); lse = log2sumexp2(P);
out = exp2(P - lse) @ KV[:, :d_v]); the Triton merge kernel runs under the Triton interpreter.
"""
from __future__ import annotations

import math
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kernels"))
os.environ.setdefault("TRITON_INTERPRET", "1")
import dsa_sparse_splitk as sk  # noqa: E402

torch = pytest.importorskip("torch")
HOOK = HERE / "dsa_sparse_splitk_hooks.py"
KV4_HOOK = HERE / "glm_kv4_sm90_hooks.py"
SRC = pathlib.Path(os.environ.get("SGLANG_SRC", os.path.expanduser("~/.cache/numinous/sgl-v0520/python")))


def ref_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None):
    """flash_mla_sparse_fwd's contract as the binding returns it: normalized bf16 output, lse = BASE-2 log-sum-exp of
    the scaled logits (the kernel's natural log x LOG_2_E, vendor python_api.cpp:212-215), +inf (with zero output) on
    a row with no valid key (phase1.cuh:446-449)."""
    s_q, h, d = q.shape
    kv2 = kv.squeeze(1).float()
    idx = indices.squeeze(1).long()
    valid = (idx >= 0) & (idx < kv2.shape[0])
    if topk_length is not None:
        valid &= torch.arange(idx.shape[1]).view(1, -1) < topk_length.view(-1, 1).long()
    sel = kv2[idx.clamp(0, kv2.shape[0] - 1)]                       # [s_q, topk, d]
    p = torch.einsum("shd,skd->shk", q.float(), sel) * sm_scale * math.log2(math.e)
    p = p.masked_fill(~valid.unsqueeze(1), float("-inf"))
    lse2 = torch.logsumexp(p * math.log(2), dim=-1) / math.log(2)  # base-2 internally, as the kernel's rM
    w = torch.exp2(p - lse2.unsqueeze(-1))
    w = torch.nan_to_num(w, nan=0.0)
    out = torch.einsum("shk,skd->shd", w, sel[..., :d_v])
    lse = torch.where(torch.isfinite(lse2), lse2, torch.full_like(lse2, float("inf")))  # base 2, +inf when empty
    return out.to(torch.bfloat16), p.amax(-1), lse.float()


def _data(t=6, h=64, d=576, k=2048, skv=5000, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(t, h, d, generator=g) * 0.05).to(torch.bfloat16)
    kv = (torch.randn(skv, 1, d, generator=g) * 0.5).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(skv, generator=g)[:k] for _ in range(t)]).to(torch.int32).unsqueeze(1)
    return q, kv, idx


@pytest.mark.parametrize("tokens,heads,topk,sms,want", [
    (36, 64, 2048, 132, 2),   # verify, 6 req x 6 tokens per rank: 72 CTAs, one wave
    (6, 64, 2048, 132, 16),   # draft decode: 96 CTAs at the 16 cap
    (48, 64, 2048, 132, 2),
    (72, 64, 2048, 132, 1),   # 72 x 2 = 144 > 132: two waves, no gain
    (132, 64, 2048, 132, 1),  # the grid already fills the device
    (1, 64, 2048, 132, 16),
    (36, 128, 2048, 132, 1),  # two head blocks: 72 CTAs; a split doubles the waves
    (4096, 64, 2048, 132, 1), # prefill
])
def test_split_factor(tokens, heads, topk, sms, want):
    assert sk.split_factor(tokens, heads, topk, sms, max_split=16) == want


@pytest.mark.parametrize("s", [2, 4, 16])
def test_split_merge_matches_one_call(s):
    q, kv, idx = _data(t=5, k=1024)
    one, _, _ = ref_fwd(q, kv, idx, 0.07)
    many = sk.sparse_fwd_splitk(ref_fwd, q, kv, idx, 0.07, 512, None, s, merge_fn=sk.merge_reference)
    err = (many.float() - one.float()).abs().max().item()
    assert err < 2e-2 * one.float().abs().max().item() + 1e-3       # bf16 partials, fp32 merge


def test_invalid_indices_topk_length_and_empty_rows():
    q, kv, idx = _data(t=4, k=512)
    idx[0, 0, 100:] = -1                       # tail padding
    idx[1, 0, :] = -1                          # a row with no valid key
    tl_len = torch.tensor([100, 512, 37, 300], dtype=torch.int32)
    one, _, _ = ref_fwd(q, kv, idx, 0.07, topk_length=tl_len)
    many = sk.sparse_fwd_splitk(ref_fwd, q, kv, idx, 0.07, 512, tl_len, 4, merge_fn=sk.merge_reference)
    assert torch.equal(many[1], torch.zeros_like(many[1]))          # empty row -> 0
    for r in (0, 2, 3):
        assert (many[r].float() - one[r].float()).abs().max().item() < 2e-2


def test_split_inputs_layout():
    q, kv, idx = _data(t=3, k=256)
    tl_len = torch.tensor([256, 70, 0], dtype=torch.int32)
    q_s, idx_s, tl_s = sk.split_inputs(q, idx, tl_len, 4)
    assert q_s.shape == (12, 64, 576) and idx_s.shape == (12, 1, 64)
    assert torch.equal(q_s[5], q[1]) and torch.equal(idx_s[5, 0], idx[1, 0, 64:128])
    assert tl_s.tolist() == [64, 64, 64, 64, 64, 6, 0, 0, 0, 0, 0, 0]


def test_triton_merge_equals_reference():
    g = torch.Generator().manual_seed(1)
    t, s, h, dv = 3, 4, 8, 512
    o_s = torch.randn(t * s, h, dv, generator=g).to(torch.bfloat16)
    lse = torch.randn(t * s, h, generator=g) * 3
    lse[1, :] = -1e30                          # an empty split
    lse[8:12, 2] = -1e30                       # a fully empty (token 2, head 2)
    want = sk.merge_reference(o_s, lse, t, s)
    got = sk.merge(o_s, lse, t, s)
    # both round an fp32 merge to bf16; allow one bf16 ulp (2^-7 relative)
    assert torch.allclose(got.float(), want.float(), rtol=2 ** -7, atol=1e-3)
    assert torch.equal(got[2, 2], torch.zeros(dv, dtype=torch.bfloat16))


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(sk.ENV, raising=False)
    assert sk.maybe_split(36, 64, 2048, torch.device("cpu"), False) == 1
    monkeypatch.setenv(sk.ENV, "1")
    assert sk.maybe_split(36, 64, 2048, torch.device("cpu"), False) == 1   # cuda only
    assert sk.maybe_split(36, 64, 2048, torch.device("cpu"), True) == 1    # sinks are not split


def test_hook_applies_and_composes_with_kv4(tmp_path):
    if not (SRC / "sglang").is_dir():
        pytest.skip("no v0.5.20 tree")
    # only the files the two hook sets touch (a full-tree copy is slow on a loaded box). The KV4 hook's own target
    # list decides (it grew memory_pool.py for Flash's hybrid pool, fe838901), so the copy never goes stale again.
    sys.path.insert(0, str(HERE))
    import glm_kv4_sm90_hooks as kv4h
    targets = {"srt/layers/attention/dsa_backend.py"} | {hk.path for hk in kv4h.HOOKS}
    dst = tmp_path / "python"
    for rel in sorted("sglang/" + t for t in targets):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / rel, dst / rel)
    shutil.copytree(SRC / "sglang/srt/layers/attention/dsa", dst / "sglang/srt/layers/attention/dsa",
                    ignore=shutil.ignore_patterns("__pycache__"))
    root = str(dst / "sglang")
    for hook in (KV4_HOOK, HOOK):
        r = subprocess.run([sys.executable, str(hook), "--root", root], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, str(HOOK), "--root", root], capture_output=True, text=True)
    assert "already applied" in r.stdout
    src = (dst / "sglang/srt/layers/attention/dsa_backend.py").read_text()
    assert src.count("numinous-hook:dsa_sparse_splitk") == 1 and "numinous-hook:kv4h_" in src
    assert (dst / "sglang/srt/layers/attention/dsa/numinous_sparse_splitk.py").exists()


EMPTY_LSE_VALUES = (float("-inf"), float("inf"), float("nan"))   # rtxsys saw +INF from the real kernel


def nan_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, empty_lse=float("inf")):
    """A fully masked row: out = NaN, lse = +inf / -inf / NaN (the real SM90 kernel returned +inf, rtxsys)."""
    out, ml, lse = ref_fwd(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)
    idx = indices.squeeze(1).long()
    valid = (idx >= 0) & (idx < kv.shape[0])
    if topk_length is not None:
        valid &= torch.arange(idx.shape[1]).view(1, -1) < topk_length.view(-1, 1).long()
    empty = ~valid.any(dim=1)
    out[empty] = float("nan")
    lse[empty] = empty_lse
    return out, ml, lse


@pytest.mark.parametrize("empty_lse,s,use_len", [(float("inf"), 2, True), (float("inf"), 8, False),
                                                 (float("-inf"), 4, True), (float("nan"), 4, False)])
def test_partially_invalid_rows_never_poison_the_merge(s, use_len, empty_lse):
    """P5 sanity garbage (2026-10-05): short prompts give rows whose later splits hold only -1 / beyond topk_length."""
    q, kv, idx = _data(t=5, h=16, k=256, skv=1200, seed=4)   # 16 heads: the Triton merge runs interpreted
    valid = [30, 70, 150, 256, 1]                        # short prefixes: whole splits empty at S >= 2
    for r, v in enumerate(valid):
        idx[r, 0, v:] = -1
    tl_len = torch.tensor(valid, dtype=torch.int32) if use_len else None
    one, _, _ = ref_fwd(q, kv, idx, 0.07, topk_length=tl_len)
    # the Triton merge (interpreter, slow on CPU) runs once per lse value; the torch merge for every case
    merges = (sk.merge_reference, sk.merge) if (s == 4 and use_len) else (sk.merge_reference,)
    for merge in merges:
        fwd = lambda **kw: nan_fwd(**kw, empty_lse=empty_lse)  # noqa: E731
        got = sk.sparse_fwd_splitk(fwd, q, kv, idx, 0.07, 512, tl_len, s, merge_fn=merge)
        assert torch.isfinite(got.float()).all()
        assert (got.float() - one.float()).abs().max().item() < 2e-2 * one.float().abs().max().item() + 2e-3


@pytest.mark.parametrize("empty_lse", [float("inf"), float("nan")])
def test_all_splits_empty_gives_zeros(empty_lse):
    q, kv, idx = _data(t=2, h=16, k=256, seed=5)
    idx[1, 0, :] = -1
    fwd = lambda **kw: nan_fwd(**kw, empty_lse=empty_lse)  # noqa: E731
    got = sk.sparse_fwd_splitk(fwd, q, kv, idx, 0.07, 512, None, 4, merge_fn=sk.merge)
    assert torch.equal(got[1], torch.zeros_like(got[1])) and torch.isfinite(got[0].float()).all()


def _peaky(t=4, h=16, k=512, skv=3000, seed=7, boost=4.0, where="last"):
    """Rows whose attention mass sits in one split: a few selected keys aligned with each head's query."""
    q, kv, idx = _data(t=t, h=h, k=k, skv=skv, seed=seed)
    kvf = kv.float().clone()
    for r in range(t):
        pos = {"last": range(k - 4, k), "first": range(0, 4), "mid": range(k // 2, k // 2 + 4)}[where]
        for j, p_ in enumerate(pos):
            kvf[int(idx[r, 0, p_]), 0] = q[r, j % h].float() * boost / (q[r, j % h].float().norm() ** 2 / 576)
    return q, kvf.to(torch.bfloat16), idx


@pytest.mark.parametrize("where", ["last", "first", "mid"])
@pytest.mark.parametrize("s", [2, 8])
def test_peaky_rows_merge_in_base_two(where, s):
    """The binding's lse is base 2 (python_api.cpp:212-215). A natural-log merge (exp) skews peaky rows: 6-33%
    measured on H200, splitk_fix_bench_h200.json run 2."""
    q, kv, idx = _peaky(where=where)
    one, _, _ = ref_fwd(q, kv, idx, 0.0625)
    got = sk.sparse_fwd_splitk(ref_fwd, q, kv, idx, 0.0625, 512, None, s, merge_fn=sk.merge_reference)
    rel = ((got.float() - one.float()).norm() / one.float().norm()).item()
    assert rel < 1e-2, rel
    # the natural-log merge (the wrong base) is visibly wrong on the same rows: the test would catch a regression
    o_s, _, lse_s = ref_fwd(*sk.split_inputs(q, idx.clone(), None, s)[:1], kv, sk.split_inputs(q, idx, None, s)[1],
                            0.0625)
    lse = lse_s.view(q.shape[0], s, -1)
    w2 = torch.exp(lse - lse.amax(1, keepdim=True))
    w2 = w2 / w2.sum(1, keepdim=True)
    bad = (o_s.view(q.shape[0], s, q.shape[1], 512).float() * w2.unsqueeze(-1)).sum(1)
    assert ((bad - one.float()).norm() / one.float().norm()).item() > 3 * rel + 1e-3


def test_triton_merge_base_two_on_peaky_rows():
    q, kv, idx = _peaky(t=2, h=8, k=256)
    one, _, _ = ref_fwd(q, kv, idx, 0.0625)
    got = sk.sparse_fwd_splitk(ref_fwd, q, kv, idx, 0.0625, 512, None, 4, merge_fn=sk.merge)
    assert ((got.float() - one.float()).norm() / one.float().norm()).item() < 1e-2

