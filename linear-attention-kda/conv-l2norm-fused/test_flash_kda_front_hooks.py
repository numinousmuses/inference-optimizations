"""CPU tests for flash_kda_front_hooks.py: the three anchors resolve once on SGLang v0.5.20, apply once, compile, and
stay idempotent; the installed module's usable() gate follows the env and the shape preconditions."""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flash_kda_front_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python")) / "sglang"
needs_src = pytest.mark.skipif(not (SRC / H.BACKEND).exists(), reason="SGLang v0.5.20 source absent")


def _copy(root: pathlib.Path) -> pathlib.Path:
    pkg = root / "sglang"
    for rel in (H.BACKEND, H.TRITON):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / rel, pkg / rel)
    return pkg


@needs_src
def test_apply_once_compile_idempotent(tmp_path):
    pkg = _copy(tmp_path)
    assert H.apply(str(pkg), check=True) == ["front: anchor ok", "dispatch: anchor ok", "no_renorm: anchor ok"]
    out = H.apply(str(pkg))
    assert out[:3] == [f"front: applied to {H.BACKEND}", f"dispatch: applied to {H.BACKEND}",
                       f"no_renorm: applied to {H.TRITON}"]
    assert out[3] == f"{H.NAME}: installed {H.MODULE_DST}"
    assert H.apply(str(pkg), install_module=False) == ["front: already applied", "dispatch: already applied",
                                                       "no_renorm: already applied"]
    for rel in (H.BACKEND, H.TRITON, H.MODULE_DST):
        compile((pkg / rel).read_text(), rel, "exec")
    be = (pkg / H.BACKEND).read_text()
    assert "numinous_qk_normed=_nkf_on" in be and "causal_conv1d_fn(" in be      # the stock path stays as the fallback
    assert 'use_qk_l2norm_in_kernel=not kwargs.get("numinous_qk_normed", False)' in (pkg / H.TRITON).read_text()


@needs_src
def test_missing_anchor_fails_loudly(tmp_path):
    pkg = _copy(tmp_path)
    p = pkg / H.TRITON
    p.write_text(p.read_text().replace("use_qk_l2norm_in_kernel=True,\n            cu_seqlens=query_start_loc,", "X"))
    with pytest.raises(SystemExit, match="no_renorm"):
        H.apply(str(pkg))


def test_usable_gate(monkeypatch):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(HERE.parent / "kernels"))
    import flash_kda_conv_l2norm as fk
    if fk.triton is None:
        pytest.skip("triton absent")
    H_, D_ = 4, 16
    layer = types.SimpleNamespace(q_dim=H_ * D_, k_dim=H_ * D_, v_dim=H_ * D_, head_q_dim=D_, head_k_dim=D_,
                                  head_v_dim=D_, conv_weights=torch.zeros(3 * H_ * D_, 4, dtype=torch.bfloat16))
    x = torch.zeros(10, 3 * H_ * D_, dtype=torch.bfloat16)
    raw = torch.zeros(5, 3, 3 * H_ * D_, dtype=torch.bfloat16)
    monkeypatch.delenv("NUMINOUS_KDA_FUSED_FRONT", raising=False)
    assert fk.usable(layer, x, raw) is False                     # env off: stock
    monkeypatch.setenv("NUMINOUS_KDA_FUSED_FRONT", "1")
    assert fk.usable(layer, x, raw) is True
    assert fk.usable(layer, x.float(), raw) is False             # dtype
    assert fk.usable(layer, x, raw.transpose(1, 2)) is False     # conv-state channel stride != 1
    bad = types.SimpleNamespace(**{**vars(layer), "v_dim": 2 * H_ * D_})
    assert fk.usable(bad, x, raw) is False                       # unequal q/k/v widths
