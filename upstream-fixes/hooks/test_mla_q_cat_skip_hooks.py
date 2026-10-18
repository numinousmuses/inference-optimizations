"""mla_q_cat_skip: anchors/compile/placement on the v0.5.20 tree, and the core identity on CPU (torch.bmm into the
strided 576-wide view + the rope copy equals bmm + concat bit for bit)."""
from __future__ import annotations

import os
import pathlib
import shutil
import sys

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mla_q_cat_skip_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python/sglang"))


def test_strided_bmm_out_equals_bmm_then_concat():
    g = torch.Generator().manual_seed(0)
    t, h, nope, lora, rope = 37, 8, 24, 64, 16
    q_nope = torch.randn(t, h, nope, generator=g).bfloat16()
    w_kc = torch.randn(h, nope, lora, generator=g).bfloat16()
    q_pe = torch.randn(t, h, rope, generator=g).bfloat16()
    want = torch.cat([torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1), q_pe], dim=-1)
    cat = q_nope.new_empty(t, h, lora + rope)
    view = cat[:, :, :lora].transpose(0, 1)
    torch.bmm(q_nope.transpose(0, 1), w_kc, out=view)
    assert view.data_ptr() == cat.data_ptr()                      # written in place, no temp + copy
    cat[:, :, lora:].copy_(q_pe)
    assert torch.equal(cat, want)
    assert cat.is_contiguous()                                    # what dsa_backend's q_rope=None path views


def test_hooks_apply_once_compile_and_place(tmp_path):
    if not (SRC / H.REL).exists():
        pytest.skip("SGLang v0.5.20 source absent")
    root = tmp_path / "sglang"
    (root / H.REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / H.REL, root / H.REL)
    assert H.apply(str(root), check=True) == [f"bmm: anchor ok in {H.REL}", f"core: anchor ok in {H.REL}"]
    assert H.apply(str(root)) == [f"bmm: applied to {H.REL}", f"core: applied to {H.REL}"]
    assert H.apply(str(root)) == ["bmm: already applied", "core: already applied"]
    text = (root / H.REL).read_text()
    compile(text, H.REL, "exec")
    prep = text[text.index("    def forward_absorb_prepare("):text.index("    def forward_absorb_core(")]
    assert "torch.bmm(q_nope.transpose(0, 1), self.w_kc, out=_nm_view)" in prep
    core = text[text.index("    def forward_absorb_core("):]
    i_skip = core.index("numinous-hook:mla_q_cat_skip (core)")
    i_stock = core.index("q_rope=q_pe,", i_skip)
    assert core.index("elif is_dcp_mla_decode_phase(forward_batch):") < i_skip < i_stock
    assert "q_rope=None," in core[i_skip:i_stock]
