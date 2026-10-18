"""CPU (Triton interpreter) checks of the Flash split-K sparse decode kernel (rocm_flash_dsa_decode.py)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

import rocm_flash_dsa_decode as M  # noqa: E402


def _case(t=3, h=16, k=256, slots=3000, masked=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(t, h, 512, generator=g) * 0.3).to(torch.bfloat16)
    kv = (torch.randn(slots, 1, 512, generator=g) * 0.5).to(torch.float8_e4m3fn)
    idx = torch.randint(1, slots, (t, 1, k), generator=g, dtype=torch.int32)
    idx[torch.rand(t, 1, k, generator=g) < masked] = -1
    return q, kv, idx


@pytest.mark.parametrize("split,hs", [(1, 1), (2, 1), (4, 2)])
def test_matches_reference(split, hs):
    q, kv, idx = _case(h=32)
    sm = 512 ** -0.5
    o = M.flash_sparse_decode_fp8(q, kv, idx, sm, split=split, head_split=hs)[0].float()
    r = M.flash_sparse_decode_ref(q, kv, idx, sm)[0].float()
    assert (o - r).abs().max() / r.abs().max() < 0.08


def test_fully_masked_row_is_zero_and_slot0_never_read():
    q, kv, idx = _case(t=2, h=16, k=128, masked=0.3)
    idx[1, 0, :] = -1
    kv.view(torch.uint8)[0].fill_(0x7F)                      # junk in slot 0
    o = M.flash_sparse_decode_fp8(q, kv, idx, 512 ** -0.5, split=2)[0].float()
    assert not torch.isnan(o).any() and o[1].abs().max() == 0
    r = M.flash_sparse_decode_ref(q, kv, idx, 512 ** -0.5)[0].float()
    assert (o[0] - r[0]).abs().max() / r[0].abs().max() < 0.08


def test_split_count_does_not_change_the_answer_much():
    q, kv, idx = _case(h=16, k=512)
    sm = 512 ** -0.5
    a = M.flash_sparse_decode_fp8(q, kv, idx, sm, split=1)[0].float()
    b = M.flash_sparse_decode_fp8(q, kv, idx, sm, split=8)[0].float()
    assert (a - b).abs().max() / a.abs().max() < 0.06     # FP8 P is quantized against each split's own max


def test_split_table(monkeypatch):
    assert [M.split_for_rows(r) for r in (8, 64, 128, 256, 384)] == [4, 4, 2, 1, 2]
    monkeypatch.setenv("NUMINOUS_FLASH_DSA_DECODE_SPLIT_TABLE", "100:8,1000000:1")
    assert M.split_for_rows(64) == 8 and M.split_for_rows(500) == 1
