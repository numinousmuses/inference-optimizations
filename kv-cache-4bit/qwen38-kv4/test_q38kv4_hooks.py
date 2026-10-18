"""q38kv4_hooks (unquantized KV cache ignores checkpoint FP8 KV scales): the anchor applies once to the stock v0.5.20
model_runner.py, lands inside configure_kv_cache_dtype after the runner's KV dtype is resolved (which runs after
load_model), is idempotent and compiles; the installed helper resets exactly the modules that carry non-unit scales.

    python3 -m pytest -q operations/patches/test_q38kv4_hooks.py
"""
import importlib.util
import pathlib
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q38kv4_hooks as h  # noqa: E402

FORK = pathlib.Path("/home/ubuntu/sglang-fork")


def _stock() -> str:
    if FORK.exists():
        return subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{h.TARGET}"],
                              capture_output=True, text=True, check=True).stdout
    if importlib.util.find_spec("sglang") is not None:
        return (pathlib.Path(h.sglang_root()) / h.TARGET).read_text()
    pytest.skip("no SGLang v0.5.20 source")


def test_anchor_applies_inside_configure_kv_cache_dtype_after_load():
    src = _stock()
    new = h.apply(src)
    assert new.count(h.MARK) == 1 and h.apply(new) == new
    compile(new, h.TARGET, "exec")
    body = new[new.index("    def configure_kv_cache_dtype(self):"):new.index("    def _get_attention_backend(")]
    assert h.MARK in body and body.index("self.kv_cache_dtype_str = (") < body.index(h.MARK)
    # ordering the fix relies on: weights are loaded before the KV dtype is configured
    assert new.index("        self.load_model()\n") < new.index("        self.configure_kv_cache_dtype()\n")


def _helper():
    mod = types.ModuleType("q38kv4_scale_reset")
    exec(compile(h.MODULE, h.MODULE_DST, "exec"), mod.__dict__)
    return mod


def test_helper_resets_only_scaled_layers():
    class Attn(torch.nn.Module):
        def __init__(self, ks, vs):
            super().__init__()
            self.k_scale = torch.nn.Parameter(torch.tensor(ks), requires_grad=False)
            self.v_scale = torch.nn.Parameter(torch.tensor(vs), requires_grad=False)
            self.k_scale_float, self.v_scale_float = ks, vs

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = Attn(0.02747, 0.02454)       # Qwen3.8 layers.3 (checkpoint FP8 KV scales)
            self.b = Attn(1.0, 1.0)               # no checkpoint scales (MTP layer)
            self.c = torch.nn.Linear(2, 2)

    m = Model()
    assert _helper().reset_unquantized_kv_scales(m) == 1
    assert m.a.k_scale_float == 1.0 and m.a.v_scale_float == 1.0
    assert float(m.a.k_scale) == 1.0 and float(m.a.v_scale) == 1.0
    assert _helper().reset_unquantized_kv_scales(m) == 0


def test_main_patches_a_copy(tmp_path):
    root = tmp_path / "sglang"
    (root / h.TARGET).parent.mkdir(parents=True)
    (root / h.MODULE_DST).parent.mkdir(parents=True, exist_ok=True)
    (root / h.TARGET).write_text(_stock())
    assert h.main(["--root", str(root)]) == 0
    assert h.MARK in (root / h.TARGET).read_text() and (root / h.MODULE_DST).exists()
    assert h.main(["--root", str(root)]) == 0      # idempotent
    assert (root / h.TARGET).read_text().count(h.MARK) == 1
