"""rocm_onehot_verify: kernel == PyTorch reference, and the rule is exact (output distribution == target p).

CPU (Triton interpreter):  TRITON_INTERPRET=1 python3 -m pytest -q operations/kernels/test_rocm_onehot_verify.py
MI355X:                    HIP_VISIBLE_DEVICES=4 python3 -m pytest -q operations/kernels/test_rocm_onehot_verify.py
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from rocm_onehot_verify import chain_speculative_sampling_onehot, reference  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)


def _case(bs, n, vocab, seed, peaky=False, greedy_rows=()):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(bs, n, vocab, generator=g) * (4.0 if peaky else 1.0)
    p = torch.softmax(logits, -1)
    for b in greedy_rows:                       # top_k=1 renorm: one-hot target rows
        p[b] = torch.nn.functional.one_hot(p[b].argmax(-1), vocab).float()
    # drafts: mostly the target argmax of the previous row (high acceptance), sometimes random
    cand = torch.zeros(bs, n, dtype=torch.int64)
    cand[:, 0] = torch.randint(0, vocab, (bs,), generator=g)
    for s in range(1, n):
        am = p[:, s - 1].argmax(-1)
        rnd = torch.randint(0, vocab, (bs,), generator=g)
        cand[:, s] = torch.where(torch.rand(bs, generator=g) < 0.7, am, rnd)
    ridx = torch.arange(bs * n, dtype=torch.int64).view(bs, n)
    coins = torch.rand(bs, n, generator=g)
    final = torch.rand(bs, generator=g)
    return cand, ridx, coins, final, p


def _run(cand, ridx, coins, final, p):
    bs, n = cand.shape
    predicts = torch.zeros(bs * n, dtype=torch.int32, device=DEV)
    acc_idx = torch.full((bs, n), -1, dtype=torch.int32, device=DEV)
    acc_num = torch.empty(bs, dtype=torch.int32, device=DEV)
    chain_speculative_sampling_onehot(
        predicts=predicts, accept_index=acc_idx, accept_token_num=acc_num, candidates=cand.to(DEV),
        retrive_index=ridx.to(DEV), retrive_next_token=None, retrive_next_sibling=None,
        uniform_samples=coins.to(DEV), uniform_samples_for_final_sampling=final.to(DEV),
        target_probs=p.to(DEV), draft_probs=p.to(DEV), threshold_single=1.0, threshold_acc=1.0, deterministic=True)
    return predicts.cpu(), acc_idx.cpu(), acc_num.cpu()


@pytest.mark.parametrize("bs,n,vocab,peaky", [(7, 4, 1000, False), (16, 4, 5000, True), (3, 6, 4097, True),
                                              (5, 2, 154880 if DEV != "cpu" else 9000, True)])
def test_kernel_matches_reference(bs, n, vocab, peaky):
    cand, ridx, coins, final, p = _case(bs, n, vocab, seed=bs * 31 + n, peaky=peaky, greedy_rows=(0,))
    got = _run(cand, ridx, coins, final, p)
    ref = reference(cand, ridx, coins, final, p)
    assert torch.equal(got[2], ref[2]), (got[2], ref[2])
    assert torch.equal(got[1], ref[1])
    # predicts only at committed positions (the rest stay 0 in both)
    assert torch.equal(got[0], ref[0])


def test_greedy_row_commits_argmax():
    cand, ridx, coins, final, p = _case(4, 4, 512, seed=3, peaky=True, greedy_rows=(0, 1, 2, 3))
    pred, acc_idx, acc_num = _run(cand, ridx, coins, final, p)
    for b in range(4):
        k = int(acc_num[b])
        # every accepted draft equals the argmax of its row, the committed bonus is the argmax of row k
        for s in range(1, k + 1):
            assert int(cand[b, s]) == int(p[b, s - 1].argmax())
        last = int(acc_idx[b, k])
        assert int(pred[last]) == int(p[b, k].argmax())


def test_first_token_distribution_is_exact():
    """Monte Carlo: the first committed token is distributed as p_0 whatever the (deterministic) draft is."""
    vocab, trials = 8, (40000 if DEV != "cpu" else 1500)
    torch.manual_seed(0)
    p0 = torch.tensor([0.05, 0.4, 0.15, 0.1, 0.1, 0.05, 0.1, 0.05])
    p = torch.zeros(trials, 2, vocab)
    p[:, 0] = p0
    p[:, 1] = torch.full((vocab,), 1.0 / vocab)
    for draft_tok in (1, 0):                      # argmax draft and a poor draft
        cand = torch.zeros(trials, 2, dtype=torch.int64)
        cand[:, 1] = draft_tok
        ridx = torch.arange(trials * 2, dtype=torch.int64).view(trials, 2)
        coins = torch.rand(trials, 2)
        final = torch.rand(trials)
        pred, acc_idx, acc_num = _run(cand, ridx, coins, final, p)
        first = pred[ridx[:, 0]]                  # the token committed at the root slot
        freq = torch.bincount(first.long(), minlength=vocab).float() / trials
        tol = 4 * torch.sqrt(p0 * (1 - p0) / trials) + 1e-3
        assert torch.all((freq - p0).abs() < tol), (draft_tok, freq, p0)
        acc_rate = acc_num.float().mean().item()
        assert abs(acc_rate - float(p0[draft_tok])) < 4 * (0.25 / trials) ** 0.5 + 1e-3


def test_rejects_thresholds():
    cand, ridx, coins, final, p = _case(2, 3, 64, seed=1)
    with pytest.raises(ValueError):
        chain_speculative_sampling_onehot(torch.zeros(6, dtype=torch.int32), torch.zeros(2, 3, dtype=torch.int32),
                                          torch.zeros(2, dtype=torch.int32), cand, ridx, None, None, coins, final, p,
                                          None, 0.9, 1.0, True)
