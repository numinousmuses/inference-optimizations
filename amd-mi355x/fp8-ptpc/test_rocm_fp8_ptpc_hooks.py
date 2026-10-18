"""rocm_fp8_ptpc_hooks: anchors on 1330e437e1 apply / compile / idempotent; block -> rowwise requant math (k3amd)."""
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TREE = pathlib.Path("/tmp/sgl-1002/python/sglang")
sys.path.insert(0, str(ROOT / "operations/patches"))
sys.path.insert(0, str(ROOT / "operations/kernels"))
import rocm_fp8_ptpc_hooks as hk  # noqa: E402
import numinous_fp8_ptpc as np8  # noqa: E402


def test_apply_idempotent_compiles(tmp_path):
    if not TREE.exists():
        pytest.skip("1330e437e1 tree not unpacked")
    dst = tmp_path / "sglang"
    (dst / "srt/layers/quantization").mkdir(parents=True)
    for f in (hk.F8, hk.FU):
        shutil.copy(TREE / f, dst / f)
    assert all("anchor ok" in x for x in hk.apply(str(dst), check=True))
    out = hk.apply(str(dst))
    assert sum("applied to" in x for x in out) == 2
    assert all("already applied" in x for x in hk.apply(str(dst)) if not x.startswith("module"))
    for f in (hk.F8, hk.FU):
        compile((dst / f).read_text(), f, "exec")
    src = (dst / hk.F8).read_text()
    assert src.index("ptpc_attach") < src.index("t = shuffle_weight(layer.weight, (16, 16))")


def test_rowwise_from_block_math():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    n, k = 384, 512
    w = torch.randn(n, k) * torch.linspace(0.1, 3.0, n)[:, None]
    # block-quantize like a checkpoint: one fp32 scale per 128x128 block
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    wb = w.view(n // 128, 128, k // 128, 128)
    scale = wb.abs().amax(dim=(1, 3)) / fp8_max
    q_block = (wb / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
    deq_block = (q_block.float().view(n // 128, 128, k // 128, 128) * scale[:, None, :, None]).view(n, k)
    q_row, row = np8.rowwise_from_block(q_block, scale)
    assert q_row.dtype == torch.float8_e4m3fn and row.shape == (n, 1)
    deq_row = q_row.float() * row
    rel = (deq_row - deq_block).norm() / deq_block.norm()
    assert rel < 0.05                      # second rounding of an already-FP8 weight: a few percent at most
    assert (row.squeeze() > 0).all()


def test_env(monkeypatch):
    monkeypatch.delenv("NUMINOUS_PTPC_MAX_M", raising=False)
    assert np8.max_m(128) == 128
    monkeypatch.setenv("NUMINOUS_PTPC_MAX_M", "1048576")
    assert np8.max_m(128) == 1048576
    monkeypatch.delenv("NUMINOUS_FP8_BLOCK_TO_PTPC", raising=False)
    assert not np8.enabled()
