"""Tests for pro_fp8_lm_head (FP8 128x128 UE8M0 LM head through DeepGEMM).

GPU only (the path is DeepGEMM). Uses the real DeepSeek V4 Pro 0813 `head.weight` from /cache when present (else a
random N(0, 0.02) head), synthetic post-norm hidden states (unit RMS times the checkpoint's final `norm.weight` when
present). Reports logits relative error vs an fp32 reference, greedy-argmax agreement and top-5 recall for both the
bf16 engine path and the FP8 path, then CUDA-graph timing of bf16 torch.matmul vs FP8 (activation quant included).
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    import flashinfer  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_fp8_lm_head as LM  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="needs a GPU (DeepGEMM)")
CKPT = "/cache/deepseek-ai__DeepSeek-V4-Pro-0813"
V, K = 129280, 7168


def _load(name):
    idx = os.path.join(CKPT, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return None
    from safetensors import safe_open

    f = json.load(open(idx))["weight_map"].get(name)
    if f is None:
        return None
    with safe_open(os.path.join(CKPT, f), framework="pt", device="cuda") as h:
        return h.get_tensor(name)


@pytest.fixture(scope="module")
def head():
    w = _load("head.weight")
    src = "checkpoint"
    if w is None:
        g = torch.Generator(device="cuda").manual_seed(0)
        w = (torch.randn(V, K, device="cuda", generator=g) * 0.02)
        src = "random"
    w = w.to(torch.bfloat16).contiguous()
    nw = _load("norm.weight")

    class Head(torch.nn.Module):
        pass

    m = Head()
    m.weight = torch.nn.Parameter(w, requires_grad=False)
    assert LM.quantize_lm_head(m)
    print(f"head source: {src}, final norm weight: {'checkpoint' if nw is not None else 'ones'}")
    return m, (nw.float() if nw is not None else torch.ones(K, device="cuda"))


def _hidden(m, nw, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(m, K, device="cuda", generator=g) * nw).bfloat16()


def test_layout(head):
    m, _ = head
    wq, ws = m._numinous_fp8
    assert wq.shape == (V, K) and wq.dtype == torch.float8_e4m3fn
    assert ws.dtype == torch.int32 and ws.shape[0] == V  # MN-major packed UE8M0, one int32 per 4 K-blocks


@pytest.mark.parametrize("rows", [1, 96, 1024])
def test_accuracy(head, rows):
    m, nw = head
    h = _hidden(rows, nw, rows)
    ref = h.float() @ m.weight.float().T
    bf = torch.matmul(h, m.weight.T).float()
    f8 = LM.fp8_logits(h, m, out_fp32=True)
    rel = lambda a: ((a - ref).norm() / ref.norm()).item()  # noqa: E731
    top1 = ref.argmax(-1)
    agree = lambda a: (a.argmax(-1) == top1).float().mean().item()  # noqa: E731
    top5 = ref.topk(5, dim=-1).indices

    def recall5(a):
        t = a.topk(5, dim=-1).indices
        return (t[:, :, None] == top5[:, None, :]).any(-1).float().mean().item()

    print(f"LMHEAD rows={rows}: rel err bf16 {rel(bf):.2e} fp8 {rel(f8):.2e}; argmax agreement bf16 {agree(bf):.4f} "
          f"fp8 {agree(f8):.4f} (fp8 vs bf16 {(f8.argmax(-1) == bf.argmax(-1)).float().mean().item():.4f}); "
          f"top5 recall bf16 {recall5(bf):.4f} fp8 {recall5(f8):.4f}")
    assert rel(f8) < 0.05
    assert agree(f8) >= 0.9


@pytest.mark.parametrize("margin", [0.5, 1.0])
def test_argmax_on_peaked_logits(head, margin):
    """Random hidden states give near-flat logits (every argmax a near tie). Decode hidden states usually point at
    a token: h = noise + margin * sqrt(K) * unit(W[t]). Agreement on these is the number that matters for greedy."""
    m, nw = head
    rows = 1024
    g = torch.Generator(device="cuda").manual_seed(1)
    t = torch.randint(0, V, (rows,), device="cuda", generator=g)
    wt = m.weight[t].float()
    h = (torch.randn(rows, K, device="cuda", generator=g) * nw
         + margin * K ** 0.5 * wt / wt.norm(dim=-1, keepdim=True)).bfloat16()
    ref = (h.float() @ m.weight.float().T).argmax(-1)
    bf = torch.matmul(h, m.weight.T).argmax(-1)
    f8 = LM.fp8_logits(h, m).argmax(-1)
    a_bf, a_f8 = (bf == ref).float().mean().item(), (f8 == ref).float().mean().item()
    print(f"LMHEAD peaked margin={margin}: argmax agreement bf16 {a_bf:.4f} fp8 {a_f8:.4f}")
    assert a_f8 >= 0.98


def test_empty_batch(head):
    m, _ = head
    out = LM.fp8_logits(torch.empty(0, K, device="cuda", dtype=torch.bfloat16), m)
    assert out.shape == (0, V)


@pytest.mark.parametrize("rows", [1, 16, 96, 128, 512])
def test_timing(head, rows):
    from pro_mhc_timing import graph_us

    m, nw = head
    hs = [_hidden(rows, nw, s) for s in range(4)]  # the 1.85 / 0.93 GB weight already defeats L2
    t_bf = graph_us([lambda h=h: torch.matmul(h, m.weight.T) for h in hs], rounds=5)
    t_f8 = graph_us([lambda h=h: LM.fp8_logits(h, m) for h in hs], rounds=5)
    floor_bf, floor_f8 = V * K * 2 / 8e12 * 1e6, V * K / 8e12 * 1e6
    print(f"TIMING lm_head rows={rows}: bf16 {t_bf:.1f} us (floor {floor_bf:.0f}), fp8 {t_f8:.1f} us "
          f"(floor {floor_f8:.0f}), speedup {t_bf / t_f8:.2f}x")
