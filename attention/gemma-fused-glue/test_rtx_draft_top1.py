"""CPU tests for the D2-b fused softmax top-1 (rtx_draft_top1.py) under the Triton interpreter, against the stock
draft path: fast_topk(torch.softmax(logits, -1), 1, -1) = torch.max(probs, -1, keepdim=True)."""
from __future__ import annotations

import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import rtx_draft_top1 as T  # noqa: E402


def _stock(logits):
    return torch.max(torch.softmax(logits.float(), dim=-1), dim=-1, keepdim=True)


def _p64(logits):
    """The exact max probability (fp64). torch's CPU fp32 softmax accumulates 262K terms per lane sequentially and is
    off by ~2.5e-5 relative on these rows; the kernel's tree sums are within 5e-8. On GPU torch reduces per block."""
    return torch.max(torch.softmax(logits.double(), dim=-1), dim=-1, keepdim=True)[0]


def test_matches_stock_softmax_max_on_draft_shape():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 262144, generator=g) * 4.0
    logits[1, 200000] = 40.0                      # a peaked row (p_max near 1)
    logits[2, :1000] = -float("inf")              # masked entries
    p, i = T.softmax_top1(logits)
    sp, si = _stock(logits)
    assert p.shape == (3, 1) and i.shape == (3, 1) and i.dtype == torch.int64
    assert torch.equal(i, si)
    assert torch.allclose(p.double(), _p64(logits), rtol=1e-6, atol=0)
    assert torch.allclose(p, sp, rtol=1e-4, atol=0)


def test_ties_take_the_smallest_index_and_odd_vocab():
    logits = torch.zeros(2, 10007)
    logits[0, 9000] = 5.0
    logits[0, 17] = 5.0                           # exact tie across two splits
    logits[1, 4095] = 3.0
    logits[1, 4096] = 3.0                         # tie straddling the split boundary
    p, i = T.softmax_top1(logits)
    assert i[:, 0].tolist() == [17, 4095]
    assert torch.allclose(p.double(), _p64(logits), rtol=1e-6)


def test_bf16_logits_and_empty_batch():
    g = torch.Generator().manual_seed(1)
    logits = (torch.randn(2, 8192, generator=g) * 3).to(torch.bfloat16)
    p, i = T.softmax_top1(logits)
    sp, si = _stock(logits)
    assert torch.equal(i, si) and torch.allclose(p.double(), _p64(logits.float()), rtol=1e-6)
    p0, i0 = T.softmax_top1(torch.zeros(0, 4096))
    assert p0.shape == (0, 1) and i0.shape == (0, 1)


def test_hook_body_declines_off_cuda_and_topk_gt_1(monkeypatch):
    monkeypatch.setattr(T, "ON", True)
    x = torch.zeros(2, 4096)
    assert T.draft_top1(x, 1) is None             # CPU tensor: stock path
    monkeypatch.setattr(T, "ON", False)
    assert T.draft_top1(x, 1) is None


# ------------------------------------------------------------------------------------------- D2-c fused draft head
def _stock_head(h, w, vocab):
    logits = (h.float() @ w[:vocab].float().t()).to(torch.bfloat16).float()   # the stock bf16 head output
    return torch.max(torch.softmax(logits, dim=-1), dim=-1, keepdim=True), logits


def test_head_top1_matches_stock_head_softmax_max():
    g = torch.Generator().manual_seed(4)
    bs, k, vocab = 5, 256, 1000                       # vocab not a multiple of the 128-wide tile
    h = (torch.randn(bs, k, generator=g)).to(torch.bfloat16)
    w = (torch.randn(vocab + 24, k, generator=g) * 0.2).to(torch.bfloat16)   # padded rows beyond vocab
    (sp, si), logits = _stock_head(h, w, vocab)
    p, i = T.head_top1(h, w, vocab)
    assert torch.equal(i, si)
    # the kernel's fp32 dot order differs from torch's; a bf16 logit may round the other way (1 ulp)
    assert torch.allclose(p, sp, rtol=2e-3)


def test_head_top1_fp8_copy_matches_dequantized_reference():
    g = torch.Generator().manual_seed(5)
    bs, k, vocab = 3, 256, 640
    h = torch.randn(bs, k, generator=g).to(torch.bfloat16)
    w = (torch.randn(vocab, k, generator=g) * 0.2).to(torch.bfloat16)
    q, s = T.quantize_head_fp8(w)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (vocab,)
    wd = (q.float() * s[:, None])
    ref_logits = ((h.float() @ q.float().t()) * s[None, :]).to(torch.bfloat16).float()
    rp, ri = torch.max(torch.softmax(ref_logits, dim=-1), dim=-1, keepdim=True)
    p, i = T.head_top1(h, q, vocab, s)
    assert torch.equal(i, ri) and torch.allclose(p, rp, rtol=2e-3)
    assert float((wd - w.float()).abs().max()) <= float(w.float().abs().max()) / 16   # e4m3 per-row rounding


def test_placeholder_result_is_consumed_by_the_worker_hook(monkeypatch):
    monkeypatch.setattr(T, "ON", True)
    ph = torch.empty((1, 1)).expand(4, 1000)
    ph.numinous_top1 = ("p", "i")
    assert T.draft_top1(ph, 1) == ("p", "i")
    assert T.draft_top1(ph, 2) is None                 # topk > 1: stock path (the head never engages there)


def test_draft_head_swaps_get_logits_only_for_the_call(monkeypatch):
    import types as _t
    monkeypatch.setattr(T, "HEAD_ON", True)
    monkeypatch.setattr(T, "ON", True)
    monkeypatch.setattr(T, "_spec_topk", lambda: 1)
    monkeypatch.setattr(T, "_on_device", lambda t: True)

    class LP:
        do_tensor_parallel_all_gather = False
        do_tensor_parallel_all_gather_dp_attn = False
        logit_scale = None
        final_logit_softcapping = None

        def _get_logits(self, hs, lm_head, md, embedding_bias=None, use_logits_buffer=True):
            raise AssertionError("stock head must not run")

        def __call__(self, input_ids, hidden_states, lm_head, fb, hidden_states_before_norm=None):
            logits = self._get_logits(hidden_states, lm_head, None)
            return _t.SimpleNamespace(next_token_logits=logits, hidden_states=hidden_states_before_norm)

    g = torch.Generator().manual_seed(6)
    w = (torch.randn(300, 128, generator=g) * 0.2).to(torch.bfloat16)
    model = _t.SimpleNamespace(logits_processor=LP(), lm_head=_t.SimpleNamespace(weight=w), vocab_size=300,
                               use_ordered_embeddings=False)
    fb = _t.SimpleNamespace(forward_mode=_t.SimpleNamespace(is_decode=lambda: True), return_logprob=False)
    h = torch.randn(2, 128, generator=g).to(torch.bfloat16)
    proj = torch.randn(2, 64, generator=g)
    out = T.draft_head(model, None, h, proj, fb)
    assert out is not None and out.hidden_states is proj
    p, i = out.next_token_logits.numinous_top1
    (sp, si), _ = _stock_head(h, w, 300)
    assert torch.equal(i, si) and torch.allclose(p, sp, rtol=2e-3)
    assert "_get_logits" not in vars(model.logits_processor)          # restored to the class method
    fb.forward_mode = _t.SimpleNamespace(is_decode=lambda: False)
    assert T.draft_head(model, None, h, proj, fb) is None              # extend / prefill: stock
