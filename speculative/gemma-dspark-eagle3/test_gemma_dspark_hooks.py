"""gemma_dspark_hooks: CPU test on a synthetic copy of the three anchored blocks (Gemma-style head without
org_vocab_size/quant_method; scaled vs unscaled embedding), plus the installed-SGLang application on a box."""
import importlib.util
import os
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gemma_dspark_hooks as h  # noqa: E402

SYNTH = '''
import torch
def tensor_model_parallel_all_gather(x, dim=-1):
    return x
def should_apply_lm_head_quant_method(lm_head, qm):
    return True

def gather_and_crop_vocab(local_logits, lm_head):
    full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
    return full_logits[..., : int(lm_head.org_vocab_size)]


def project_through_lm_head(hidden, lm_head):
    quant_method = lm_head.quant_method
    if should_apply_lm_head_quant_method(lm_head, quant_method):
        return quant_method.apply(lm_head, hidden, None)
    weight = lm_head.weight
    return torch.matmul(hidden.to(weight.dtype), weight.T)


class Draft:
    is_nemotron_35_draft = False

    def __init__(self, emb):
        self.embed_tokens = emb

    def attach_shared_modules(self, *, embed_tokens, lm_head):
        if not self.is_nemotron_35_draft:
            self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def forward_embed(self, input_ids):
        # model exposes forward_embed).
        return self.embed_tokens(input_ids)
'''


class ScaledEmb(torch.nn.Embedding):  # Gemma3TextScaledWordEmbedding stand-in
    def __init__(self, n, d, scale):
        super().__init__(n, d)
        self.embed_scale = scale

    def forward(self, ids):
        return super().forward(ids) * self.embed_scale


def _ns():
    ns = {}
    exec(compile(h.apply(SYNTH), "synth", "exec"), ns)
    return ns


def test_crop_and_project_work_on_a_gemma_head():
    ns = _ns()
    head = ScaledEmb(10, 4, 2.0)
    logits = torch.randn(3, 10)
    assert ns["gather_and_crop_vocab"](logits, head).shape == (3, 10)
    hid = torch.randn(3, 4)
    assert torch.allclose(ns["project_through_lm_head"](hid, head), hid @ head.weight.T)


def test_attach_gives_gemma_head_org_vocab_size():
    ns = _ns()
    head = ScaledEmb(10, 4, 2.0)
    d = ns["Draft"](head)
    d.attach_shared_modules(embed_tokens=head, lm_head=head)
    assert d.lm_head.org_vocab_size == 10


def test_embed_native_default_and_unscaled_switch(monkeypatch):
    ns = _ns()
    emb = ScaledEmb(10, 4, 53.0)
    ids = torch.tensor([1, 2])
    d = ns["Draft"](emb)
    monkeypatch.delenv("NUMINOUS_DSPARK_EMBED", raising=False)
    assert torch.allclose(d.forward_embed(ids), emb.weight[ids] * 53.0)      # stock: target's scaled module
    monkeypatch.setenv("NUMINOUS_DSPARK_EMBED", "unscaled")
    assert torch.allclose(d.forward_embed(ids), emb.weight[ids])             # raw table


def test_idempotent_and_anchor_failure():
    once = h.apply(SYNTH)
    assert h.apply(once) == once
    with pytest.raises(SystemExit):
        h.apply("x = 1\n")


@pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="needs the installed SGLang (run on a box)")
def test_applies_to_installed_sglang(tmp_path):
    dst = tmp_path / "srt/models"
    dst.mkdir(parents=True)
    (dst / "dspark.py").write_text(open(os.path.join(h.sglang_root(), h.TARGET)).read())
    assert h.main(["--root", str(tmp_path)]) == 0
    assert h.MARK in (dst / "dspark.py").read_text()
