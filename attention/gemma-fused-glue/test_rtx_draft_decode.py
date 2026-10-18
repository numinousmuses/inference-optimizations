"""CPU tests for the draft decode kernel (rtx_draft_decode.py) under the Triton interpreter, against an fp32 torch
reference of the stock decode semantics (softmax(q K^T * sm_scale * k_descale) V * v_descale over kv_indices)."""
from __future__ import annotations

import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

import rtx_draft_decode as DD  # noqa: E402


def _case(bs, hq, hkv, d, lens, slots, fp8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(bs, hq, d, generator=g).to(torch.bfloat16)
    kb = (torch.randn(slots, hkv, d, generator=g) * 2)
    vb = torch.randn(slots, hkv, d, generator=g)
    if fp8:
        kb, vb = kb.to(torch.float8_e4m3fn), vb.to(torch.float8_e4m3fn)
    else:
        kb, vb = kb.to(torch.bfloat16), vb.to(torch.bfloat16)
    indptr = torch.zeros(bs + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    indices = torch.randperm(slots, generator=g)[: int(indptr[-1])].to(torch.int64)
    return q, kb, vb, indptr, indices


@pytest.mark.parametrize("hq,hkv,d,fp8,splits", [(8, 4, 256, True, 1), (8, 4, 256, True, 3), (8, 1, 512, False, 2),
                                                 (16, 2, 256, False, 4)])
def test_matches_reference(hq, hkv, d, fp8, splits):
    lens = [70, 1, 200, 0]                           # ragged, a 1-token and an empty request
    q, kb, vb, indptr, indices = _case(len(lens), hq, hkv, d, lens, 512, fp8)
    o = torch.empty(len(lens), hq, d, dtype=torch.bfloat16)
    DD.decode(q, kb, vb, o, indptr, indices, sm_scale=0.0625, k_scale=1.0, v_scale=0.5, n_splits=splits)
    ref = DD.decode_ref(q, kb, vb, indptr, indices, 0.0625, 1.0, 0.5)
    assert torch.allclose(o.float(), ref, atol=2e-2, rtol=2e-2)
    assert torch.equal(o[3].float(), torch.zeros_like(o[3].float()))   # empty request -> zeros


def test_choose_splits_fills_the_gpu_and_respects_the_window():
    # run5 lesson: 1 split at bs 32 x 16 kv heads (512 programs, 1 per SM) lost to stock's 8 splits
    assert DD.choose_splits(32, 16, 1024) == 3          # ~8 programs per SM
    assert DD.choose_splits(32, 4, 0) >= 8              # the full layer: 128 programs -> split more
    assert DD.choose_splits(1, 4, 1024) == 8            # a window caps useful splits at span / 128


def test_cfg_override_env(monkeypatch):
    monkeypatch.setenv("NUMINOUS_DRAFT_DECODE_CFG", "4,64,8,2")
    assert DD._cfg_override() == [4, 64, 8, 2]
    lens = [70, 3]
    q, kb, vb, indptr, indices = _case(2, 8, 4, 256, lens, 256, True, seed=2)
    o = torch.empty(2, 8, 256, dtype=torch.bfloat16)
    DD.decode(q, kb, vb, o, indptr, indices, 0.0625)
    ref = DD.decode_ref(q, kb, vb, indptr, indices, 0.0625)
    assert torch.allclose(o.float(), ref, atol=2e-2, rtol=2e-2)


def test_hook_declines_out_of_scope(monkeypatch):
    import types
    monkeypatch.setattr(DD, "ON", True)
    layer = types.SimpleNamespace(qk_head_dim=256, v_head_dim=256, xai_temperature_len=-1, layer_id=0,
                                  tp_q_head_num=8, sliding_window_size=1024, scaling=1.0)
    be = types.SimpleNamespace(page_size=1, use_mla=False, dcp_size=1)
    q = torch.zeros(2, 8 * 256, dtype=torch.bfloat16)
    assert DD.maybe_decode(be, q, q, layer, None, None, None, 1.0, 1.0, 0.0, None, None) is False   # CPU tensor
    be.page_size = 16
    assert DD.maybe_decode(be, q, q, layer, None, None, None, 1.0, 1.0, 0.0, None, None) is False
