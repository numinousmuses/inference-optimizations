"""CPU tests for the fused KDA prefill front (flash_kda_conv_l2norm.py): the Triton kernel under TRITON_INTERPRET=1
against the pure-torch reference of the stock chain (causal_conv1d_fn -> split -> l2norm), on varlen batches with
warm and cold sequences, sequences shorter than the conv window, and the conv-state update."""
import os
import pathlib
import sys

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import flash_kda_conv_l2norm as fk  # noqa: E402

needs_triton = pytest.mark.skipif(fk.triton is None, reason="triton absent")


def _inputs(lens, warm, H=2, D=16, W=4, seed=0, bias=False):
    g = torch.Generator().manual_seed(seed)
    n, dim = sum(lens), 3 * H * D
    x = (torch.randn(n, dim, generator=g) * 0.7).to(torch.bfloat16)
    w = (torch.randn(dim, W, generator=g) * 0.4).to(torch.bfloat16)
    b = (torch.randn(dim, generator=g) * 0.1).to(torch.bfloat16) if bias else None
    cs = (torch.randn(6, W - 1, dim, generator=g) * 0.5).to(torch.bfloat16)
    ci = torch.tensor([5, 2, 4, 1][:len(lens)], dtype=torch.int32)
    his = torch.tensor(warm, dtype=torch.bool)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    return x, w, b, cs, ci, his, cu


def test_block_plan_never_crosses_sequences():
    seqs, starts = fk.block_plan([5, 3, 1], 4)
    assert seqs == [0, 0, 1, 2] and starts == [0, 4, 5, 8]


def test_reference_matches_a_direct_conv_formula():
    x, w, b, cs, ci, his, cu = _inputs([6], [True], H=1, D=4)
    cs0 = cs.clone()
    y = fk.conv_ref(x, w, b, cs, ci, his, cu)
    # token 0 sees the 3 state rows (oldest first) then itself
    xe = torch.cat([cs0[5].float(), x.float()], 0)
    acc = sum(xe[j] * w[:, j].float() for j in range(4))
    assert torch.allclose(y[0].float(), (acc * torch.sigmoid(acc)).to(torch.bfloat16).float())
    assert torch.equal(cs[5], x[-3:])                       # new state = last 3 inputs


@needs_triton
@pytest.mark.parametrize("lens,warm,bias", [
    ([7], [True], False),
    ([5, 3, 1], [True, False, True], False),
    ([2, 9], [True, True], True),                           # a sequence shorter than W-1 shifts the old state
    ([4, 4, 4, 4], [False, True, False, True], False),
])
def test_kernel_matches_stock_chain(lens, warm, bias):
    x, w, b, cs, ci, his, cu = _inputs(lens, warm, bias=bias)
    cs_ref, cs_k = cs.clone(), cs.clone()
    q0, k0, v0 = fk.conv_l2norm_ref(x, w, b, cs_ref, ci, his, cu, 2, 16)
    q1, k1, v1 = fk.conv_l2norm_qkv(x, w, b, cs_k, ci, his, cu, 2, 16, seq_lens=lens, bt=4, rowloop_seg=0)
    for a_, b_ in ((q1, q0), (k1, k0), (v1, v0)):
        assert a_.shape == b_.shape and a_.is_contiguous()
        assert (a_.float() - b_.float()).abs().max().item() <= 1e-2   # one bf16 ulp at these magnitudes
    assert torch.equal(cs_k, cs_ref)                         # state update byte-exact
    # q/k rows are unit-norm per head
    assert torch.allclose(q1.float().norm(dim=-1), torch.ones(q1.shape[:-1]), atol=2e-2)


@needs_triton
@pytest.mark.parametrize("lens,warm", [([7], [True]), ([5, 3, 1], [True, False, True]), ([2, 9], [True, True]),
                                       ([4, 4, 4, 4], [False, True, False, True])])
def test_rowloop_variant_matches_stock_chain(lens, warm):
    x, w, b, cs, ci, his, cu = _inputs(lens, warm, bias=True)
    cs_ref, cs_k = cs.clone(), cs.clone()
    q0, k0, v0 = fk.conv_l2norm_ref(x, w, b, cs_ref, ci, his, cu, 2, 16)
    q1, k1, v1 = fk.conv_l2norm_qkv(x, w, b, cs_k, ci, his, cu, 2, 16, seq_lens=lens, rowloop_seg=4)
    for a_, b_ in ((q1, q0), (k1, k0), (v1, v0)):
        assert (a_.float() - b_.float()).abs().max().item() <= 1e-2
    assert torch.equal(cs_k, cs_ref)
