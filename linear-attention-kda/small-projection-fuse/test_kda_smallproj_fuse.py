"""CPU tests for kda_smallproj_fuse_hooks. Tree: $SGLANG_SRC or ~/.cache/numinous/sgl-v0520/python."""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import types

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kda_smallproj_fuse_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", pathlib.Path.home() / ".cache/numinous/sgl-v0520/python")) / "sglang"


def _patched(tmp_path):
    if not (SRC / H.PATH).is_file():
        pytest.skip(f"no SGLang tree at {SRC}")
    root = tmp_path / "sglang"
    (root / H.PATH).parent.mkdir(parents=True)
    shutil.copy(SRC / H.PATH, root / H.PATH)
    assert "anchors ok" in H.apply(str(root), check=True)[0]
    assert "applied" in H.apply(str(root))[0]
    assert H.apply(str(root)) == [f"{H.NAME}: already applied"]
    return (root / H.PATH).read_text()


def _helpers(text: str, env_on: bool, capturing: bool = False):
    """Exec the inserted helper block with a stub torch.cuda (CPU) and the env decided at import."""
    i = text.index(H.TAG)
    j = text.index("class Glm5NextLinearAttention(nn.Module):")
    os.environ["NUMINOUS_KDA_SMALLPROJ_FUSE"] = "1" if env_on else "0"
    cuda = types.SimpleNamespace(is_available=lambda: True, is_current_stream_capturing=lambda: capturing)
    fake_torch = types.SimpleNamespace(**{k: getattr(torch, k) for k in ("cat", "stack", "bmm", "bfloat16", "float16",
                                                                        "nn")}, cuda=cuda)
    ns = {"torch": fake_torch}
    exec(compile(text[i:j], "helpers", "exec"), ns)
    return ns


class UnquantizedLinearMethod:
    pass


class Fp8LinearMethod:
    pass


def _lin(out_f, in_f, qm=UnquantizedLinearMethod, dtype=torch.bfloat16):
    m = types.SimpleNamespace(weight=torch.nn.Parameter((torch.randn(out_f, in_f) * 0.05).to(dtype),
                                                        requires_grad=False),
                              bias=None, quant_method=qm())
    return m


def _attn(qm=UnquantizedLinearMethod, H_=256, nb=8, d=32, P=128):
    torch.manual_seed(0)
    return types.SimpleNamespace(b_proj=_lin(nb, H_, qm), f_a_proj=_lin(d, H_, qm), g_a_proj=_lin(d, H_, qm),
                                 f_b_proj=_lin(P, d, qm), g_b_proj=_lin(P, d, qm))


def _stock(a, x):
    F = torch.nn.functional
    return (F.linear(x, a.b_proj.weight), F.linear(F.linear(x, a.f_a_proj.weight), a.f_b_proj.weight),
            F.linear(F.linear(x, a.g_a_proj.weight), a.g_b_proj.weight))


def test_apply_inserts_both_hooks_and_compiles(tmp_path):
    text = _patched(tmp_path)
    fwd = text[text.index("    def forward_qkvbfg("):text.index("    def forward_qkvbfg_fused(")]
    assert "_nf = _numinous_kda_smallproj(self, hidden_states)" in fwd
    assert fwd.index("qkv, _ = self.qkv_proj(hidden_states)") < fwd.index("_numinous_kda_smallproj") < fwd.index(
        "beta = self.b_proj(hidden_states)[0]")
    assert text.index("def _numinous_kda_smallproj(") < text.index("class Glm5NextLinearAttention(nn.Module):")


def test_fused_equals_stock_and_aliases_storage(tmp_path):
    ns = _helpers(_patched(tmp_path), env_on=True)
    a = _attn()
    x = torch.randn(37, 256).to(torch.bfloat16)
    ref = _stock(a, x)
    old_ptr = a.f_b_proj.weight.data_ptr()
    out = ns["_numinous_kda_smallproj"](a, x)
    assert out is not None and len(out) == 3
    for r, o in zip(ref, out):
        assert o.shape == r.shape
        assert float((o.float() - r.float()).abs().max()) <= 2e-2          # bf16 accumulation-order ulps
    w_a, w_b, nb, d = a._numinous_smallproj
    assert a.b_proj.weight.data_ptr() == w_a.data_ptr()                    # re-pointed, memory neutral
    assert a.f_b_proj.weight.data_ptr() == w_b.data_ptr() and a.f_b_proj.weight.data_ptr() != old_ptr
    again = _stock(a, x)                                                   # separate params still correct
    for r, o in zip(ref, again):
        assert torch.equal(r, o)
    assert ns["_numinous_kda_smallproj"](a, x[:0])[1].shape == (0, 128)    # empty extend


def test_guards_keep_stock(tmp_path):
    text = _patched(tmp_path)
    off = _helpers(text, env_on=False)
    assert off["_numinous_kda_smallproj"](_attn(), torch.randn(4, 256).to(torch.bfloat16)) is None
    on = _helpers(text, env_on=True)
    q = _attn(qm=Fp8LinearMethod)
    assert on["_numinous_kda_smallproj"](q, torch.randn(4, 256).to(torch.bfloat16)) is None and \
        q._numinous_smallproj is False
    s = _attn()
    s.f_a_proj.weight_scale = torch.ones(1)
    assert on["_numinous_kda_smallproj"](s, torch.randn(4, 256).to(torch.bfloat16)) is None
    cap = _helpers(text, env_on=True, capturing=True)
    c = _attn()
    ptr = c.b_proj.weight.data_ptr()
    assert cap["_numinous_kda_smallproj"](c, torch.randn(4, 256).to(torch.bfloat16)) is None
    assert c._numinous_smallproj is False and c.b_proj.weight.data_ptr() == ptr   # never re-pointed under capture
