import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
import torch  # noqa: E402
from sglang.srt.layers.attention.dsa.dcp_topk import (  # noqa: E402
    compact_owned, global_to_local_owned, local_to_global, merge_topk, reference_topk_global, select_dcp,
)


def test_owner_rule_roundtrip():
    S = 8
    for r in range(S):
        loc = torch.arange(0, 50)
        g = local_to_global(loc, S, r)
        assert torch.all(g % S == r)
        back = global_to_local_owned(g, S, r)
        assert torch.equal(back, loc)
        other = global_to_local_owned(g, S, (r + 1) % S)
        assert torch.all(other == -1)
    assert local_to_global(torch.tensor([-1, 3]), S, 2).tolist() == [-1, 26]


def test_merge_topk_ignores_padding_and_sorts():
    s = torch.tensor([[0.1, 0.9, 0.5, 0.7]])
    p = torch.tensor([[10, 11, -1, 13]])
    ts, tp = merge_topk(s, p, 3)
    assert tp.tolist() == [[11, 13, 10]]
    ts, tp = merge_topk(s, p, 5)
    assert tp.tolist() == [[11, 13, 10, -1, -1]] and ts.shape[-1] == 5


def test_compact_owned_moves_valid_first():
    x = torch.tensor([[-1, 4, -1, 2], [7, -1, -1, -1]])
    assert compact_owned(x).tolist() == [[4, 2, -1, -1], [7, -1, -1, -1]]


def test_sharded_selection_matches_single_rank_topk():
    torch.manual_seed(3)
    B, T, k, S = 4, 96_000, 2048, 8
    scores = torch.randn(B, T)
    owned = select_dcp(scores, k, S)
    ref = reference_topk_global(scores, k)
    for b in range(B):
        union = set()
        for r in range(S):
            loc = owned[r][b]
            loc = loc[loc >= 0]
            union |= set(local_to_global(loc, S, r).tolist())
        assert union == set(ref[b].tolist()), f"row {b}: sharded selection differs from the single-rank top-k"
        assert sum(int((owned[r][b] >= 0).sum()) for r in range(S)) == k


def test_short_context_fewer_than_k_tokens():
    torch.manual_seed(1)
    B, T, k, S = 2, 500, 2048, 8
    scores = torch.randn(B, T)
    owned = select_dcp(scores, k, S)
    total = sum(int((owned[r] >= 0).sum()) for r in range(S))
    assert total == B * T                     # every token selected, none duplicated
    for r in range(S):
        assert owned[r].shape == (B, k)       # kernels see a fixed width with -1 padding


def test_owned_local_slots_compacts_and_counts():
    from sglang.srt.layers.attention.dsa.dcp_topk import owned_local_slots
    slots = torch.tensor([[16, 9, -1, 24, 3, 8], [1, 2, 3, 4, 5, 6]], dtype=torch.int32)
    local, counts = owned_local_slots(slots, 8, 0)
    assert local.tolist()[0] == [2, 3, 1, -1, -1, -1] and counts.tolist() == [3, 0]
    assert local.dtype == torch.int32 and counts.dtype == torch.int32
