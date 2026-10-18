"""CPU tests for flash_sparse_prefill.py (TRITON_INTERPRET for the dequant kernel) and its hook."""
import os
import pathlib
import shutil
import sys
import types

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "patches"))
import flash_sparse_prefill as fsp  # noqa: E402
import flash_sparse_prefill_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python")) / "sglang"


def _pool(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, 4, 128, generator=g)
    scale = x.abs().amax(-1).clamp(min=1e-8) / 448.0
    q8 = (x / scale[..., None]).to(torch.float8_e4m3fn).view(rows, 512)
    out = torch.empty(rows, 528, dtype=torch.uint8)
    out[:, :512] = q8.view(torch.uint8)
    out[:, 512:] = scale.contiguous().view(torch.uint8).view(rows, 16)
    return out


def test_dequant_ref_layout():
    pool = _pool(10)
    d = fsp.dequant_528_ref(pool, torch.tensor([3, 3, 7]))
    assert d.shape == (3, 1, 512) and d.dtype == torch.bfloat16
    assert torch.equal(d[0], d[1])


@pytest.mark.skipif(fsp.triton is None, reason="triton absent")
def test_dequant_kernel_matches_reference():
    pool = _pool(37)
    locs = torch.tensor([5, 0, 36, 5, 17, 2, 9, 30, 31, 1, 12], dtype=torch.int64)
    ref = fsp.dequant_528_ref(pool, locs)
    got = fsp.dequant_528_rows(pool, locs, block_r=4)
    # the interpreter emulates the e4m3 decode in numpy (measured ~0.5% off); on the GPU the bench asserts equality
    assert torch.allclose(got.float(), ref.float(), rtol=1e-2, atol=1e-2)
    assert torch.equal(got[0], got[3])                                # duplicate locations, identical rows


def test_remap_keeps_minus_one_and_pads_to_128():
    locs = torch.tensor([10, 4, 7, 4], dtype=torch.int64)            # duplicate 4: last compact row wins
    idx = torch.tensor([[[4, 10, -1, 7]]], dtype=torch.int32)
    idx_c, inv = fsp.remap_indices(idx, locs, pool_rows=12)
    assert idx_c.shape == (1, 1, 128) and idx_c.dtype == torch.int32
    assert idx_c[0, 0, :4].tolist() == [3, 0, -1, 2] and (idx_c[0, 0, 4:] == -1).all()
    assert inv[4].item() == 3 and inv[5].item() == -1


def test_batch_locations_live_rows_only():
    pt = torch.tensor([[3, 9, 99, 77], [4, 5, 6, 88]], dtype=torch.int32)       # 99 / 77 / 88: stale entries
    md = types.SimpleNamespace(page_table_1=pt, seq_lens_sum=5, cu_seqlens_k=torch.tensor([0, 2, 5], dtype=torch.int32))
    assert fsp.batch_locations(md, 64, pool_rows=50, q_rows=300).tolist() == [3, 9, 4, 5, 6]
    md2 = types.SimpleNamespace(page_table_1=None, real_page_table=torch.tensor([[1], [0]], dtype=torch.int32),
                                seq_lens_sum=5, cu_seqlens_k=torch.tensor([0, 3, 5], dtype=torch.int32))
    assert fsp.batch_locations(md2, 4, pool_rows=100, q_rows=300).tolist() == [4, 5, 6, 0, 1]


def test_batch_locations_refuses_expanded_and_oversize(monkeypatch):
    pt = torch.zeros(300, 8, dtype=torch.int32)                                  # one row per query token
    md = types.SimpleNamespace(page_table_1=pt, seq_lens_sum=10, cu_seqlens_k=torch.zeros(301, dtype=torch.int32))
    assert fsp.batch_locations(md, 64, pool_rows=50, q_rows=300) is None
    md_none = types.SimpleNamespace(page_table_1=pt[:2], seq_lens_sum=None, cu_seqlens_k=None)
    assert fsp.batch_locations(md_none, 64, pool_rows=50, q_rows=300) is None
    monkeypatch.setattr(fsp, "MAX_ROWS", 4)
    md_big = types.SimpleNamespace(page_table_1=pt[:2], seq_lens_sum=5, cu_seqlens_k=torch.tensor([0, 2, 5]))
    assert fsp.batch_locations(md_big, 64, pool_rows=50, q_rows=300) is None


def test_usable_gate(monkeypatch):
    be = types.SimpleNamespace(kv_cache_dim=528, dsa_kv_cache_store_fp8=True)
    q = torch.zeros(300, 64, 512)
    monkeypatch.delenv(fsp.ENV, raising=False)
    assert fsp.usable(be, q, None) is False
    monkeypatch.setenv(fsp.ENV, "1")
    assert fsp.usable(be, q, None) is (fsp.triton is not None)
    assert fsp.usable(be, torch.zeros(100, 64, 512), None) is False          # decode-sized batch
    assert fsp.usable(types.SimpleNamespace(kv_cache_dim=656), q, None) is False   # RoPE row: stock path


@pytest.mark.skipif(not (SRC / H.BACKEND).exists(), reason="SGLang v0.5.20 source absent")
def test_hook_apply_once_and_compile(tmp_path):
    pkg = tmp_path / "sglang"
    (pkg / H.BACKEND).parent.mkdir(parents=True)
    shutil.copy(SRC / H.BACKEND, pkg / H.BACKEND)
    assert H.apply(str(pkg), check=True) == ["prefill_kernel: anchor ok"]
    assert H.apply(str(pkg))[:2] == [f"prefill_kernel: applied to {H.BACKEND}", f"{H.NAME}: installed {H.MODULE_DST}"]
    assert H.apply(str(pkg), install_module=False) == ["prefill_kernel: already applied"]
    compile((pkg / H.BACKEND).read_text(), H.BACKEND, "exec")
    compile((pkg / H.MODULE_DST).read_text(), H.MODULE_DST, "exec")
