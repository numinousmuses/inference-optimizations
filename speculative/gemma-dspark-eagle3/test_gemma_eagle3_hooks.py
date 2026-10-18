"""gemma_eagle3_hooks: applies to the real llama_eagle3.py when SGLang is installed (box), and on CPU to a synthetic
copy of the anchored block, checking both norm_before_residual branches compute what vLLM/speculators compute."""
import importlib.util
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gemma_eagle3_hooks as h  # noqa: E402

SYNTH = '''
class Layer:
    def __init__(self, config, RMSNorm):
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states):
        if True:
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)
        return hidden_states, residual
'''


def _layer(flag):
    ns = {}
    exec(compile(h.apply(SYNTH), "synth", "exec"), ns)
    cfg = type("C", (), {"hidden_size": 4, "rms_norm_eps": 1e-6, **({"norm_before_residual": flag} if flag is not None else {})})
    return ns["Layer"](cfg, lambda *a, **k: (lambda x: x * 2))


def test_default_is_stock_order():
    out, res = _layer(None).forward(3)
    assert (out, res) == (6, 3)          # residual taken before the norm (stock)


def test_norm_before_residual_takes_normed_residual():
    out, res = _layer(True).forward(3)
    assert (out, res) == (6, 6)          # residual taken after the norm (speculators/vLLM)


def test_idempotent_and_anchor_failure():
    once = h.apply(SYNTH)
    assert h.apply(once) == once
    with pytest.raises(SystemExit):
        h.apply("class X: pass\n")


@pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="needs the installed SGLang (run on a box)")
def test_applies_to_installed_sglang(tmp_path):
    root = h.sglang_root()
    dst = tmp_path / "srt/models"
    dst.mkdir(parents=True)
    (dst / "llama_eagle3.py").write_text(open(os.path.join(root, h.TARGET)).read())
    assert h.main(["--root", str(tmp_path)]) == 0
    assert h.MARK in (dst / "llama_eagle3.py").read_text()
