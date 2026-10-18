"""CPU tests for frozen_mtp_nosync_hooks.py on the real v0.5.20 frozen_kv_mtp_worker_v2.py (amdpro 2026-10-06).
The patched seq_lens_sum block is executed on a fake batch whose GPU tensor refuses .item(), so a D2H on that path
would fail the test."""
from __future__ import annotations

import ast
import os
import pathlib
import shutil
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_mtp_nosync_hooks as h  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", str(pathlib.Path.home() / ".cache/numinous/sgl-v0520/python"))) / "sglang"
if not (SRC / h.WORKER).exists():
    pytest.skip("v0.5.20 tree not present", allow_module_level=True)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("sgl") / "sglang"
    (root / h.WORKER).parent.mkdir(parents=True)
    shutil.copy(SRC / h.WORKER, root / h.WORKER)
    assert h.apply(str(root)) == [f"frozen_mtp_nosync: applied to {h.WORKER}"]
    assert h.apply(str(root)) == ["frozen_mtp_nosync: already applied"]
    return root


def _block(tree):
    """The patched `if NUMINOUS_FROZEN_MTP_NOSYNC ...: ... else: ...` statement inside FrozenKVMTPDraftWorker.draft."""
    mod = ast.parse((tree / h.WORKER).read_text())
    for cls in (n for n in mod.body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "draft"):
            for st in ast.walk(fn):
                if isinstance(st, ast.If) and "NUMINOUS_FROZEN_MTP_NOSYNC" in ast.unparse(st.test):
                    return compile(ast.Module(body=[st], type_ignores=[]), "draft_block", "exec")
    raise AssertionError("patched block not found")


class _NoD2H:
    """A GPU tensor stand-in: shape works, any reduction-to-host fails."""

    def __init__(self, n):
        self.shape = (n,)

    def sum(self):
        raise AssertionError("D2H on batch.seq_lens")


def _self(max_ctx=262144):
    backend = types.SimpleNamespace(max_context_len=max_ctx)
    runner = types.SimpleNamespace(attn_backend=backend, model_config=types.SimpleNamespace(context_len=max_ctx))
    return types.SimpleNamespace(target_worker=types.SimpleNamespace(model_runner=runner),
                                 speculative_num_draft_tokens=6)


def test_no_d2h_and_an_int_upper_bound(tree, monkeypatch):
    code = _block(tree)
    monkeypatch.setenv(h.ENV, "1")
    batch = types.SimpleNamespace(seq_lens=_NoD2H(32), seq_lens_cpu=None, seq_lens_sum=None)
    exec(code, {"torch": None, "__import__": __import__}, {"self": _self(), "batch": batch})
    assert batch.seq_lens_sum == 32 * 262144 and isinstance(batch.seq_lens_sum, int)


def test_context_len_fallback(tree, monkeypatch):
    code = _block(tree)
    monkeypatch.setenv(h.ENV, "1")
    s = _self()
    s.target_worker.model_runner.attn_backend = types.SimpleNamespace()        # no max_context_len attribute
    s.target_worker.model_runner.model_config.context_len = 131072
    batch = types.SimpleNamespace(seq_lens=_NoD2H(4), seq_lens_cpu=None, seq_lens_sum=None)
    exec(code, {"torch": None}, {"self": s, "batch": batch})
    assert batch.seq_lens_sum == 4 * 131072


def test_env_off_keeps_the_stock_sum(tree, monkeypatch):
    code = _block(tree)
    monkeypatch.delenv(h.ENV, raising=False)

    class _Torch:
        @staticmethod
        def sum(x):
            return types.SimpleNamespace(item=lambda: 1234)
    batch = types.SimpleNamespace(seq_lens=_NoD2H(2), seq_lens_cpu=None, seq_lens_sum=None)
    exec(code, {"torch": _Torch}, {"self": _self(), "batch": batch})
    assert batch.seq_lens_sum == 1234


def test_upstream_contract_still_holds():
    """EAGLE v2 dropped the same sync (the comment this hook relies on) and the replay needs an int."""
    common = (SRC / "srt/speculative/eagle_worker_common.py").read_text()
    assert "Skip per-iter .sum().item() D2H via UB" in common
    runner = (SRC / "srt/speculative/frozen_kv_mtp_cuda_graph_runner.py").read_text()
    assert "forward_batch.seq_lens_sum\n            + (expanded_bs - raw_expanded_bs) * self.seq_len_fill_value" in runner
    tb = (SRC / "srt/layers/attention/triton_backend.py").read_text()
    assert "over-allocate is safe" in tb


def test_preflight():
    ctx = types.SimpleNamespace(env={h.ENV: "1"}, args={"speculative_algorithm": "NEXTN"})
    assert h.numinous_preflight(ctx) == []
    ctx.args = {"speculative_algorithm": "EAGLE"}
    assert h.numinous_preflight(ctx)
    ctx.env = {}
    assert h.numinous_preflight(ctx) == []
