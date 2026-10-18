"""kvmix_core on CPU: pool split math, the two-pool allocator against the stock paged allocator, per-pool free /
eviction accounting, cross-pool fallback, and promotion against a real SGLang v0.5.20 RadixCache.

Runs on the CPU host (SGLang v0.5.20 sources: installed package or the local fork's v0.5.20 tree extracted to
/tmp/sgl0520/python) and on the boxes. The allocator's CPU path uses paged.alloc_extend_naive (the torch mirror of
the Triton kernel); on CUDA the same tests exercise the Triton kernels.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import types

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
FORK = pathlib.Path("/home/ubuntu/sglang-fork")
EXTRACT = pathlib.Path("/tmp/sgl0520/python")


def _ensure_sglang():
    if importlib.util.find_spec("sglang") is not None:
        return
    if not (EXTRACT / "sglang").exists():
        if not (FORK / ".git").exists():
            pytest.skip("sglang not importable and no local fork")
        EXTRACT.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"git -C {FORK} archive v0.5.20 python/sglang | tar -x -C {EXTRACT.parent}", shell=True,
                       check=True, timeout=300)
    sys.path.insert(0, str(EXTRACT))


_ensure_sglang()
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(HERE))
import kvmix_core as core  # noqa: E402
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PS = 64


class FakeKV:
    """Stands in for the pool: records promote calls."""

    def __init__(self):
        self.promotions = []

    def kvmix_promote_pages(self, src, dst):
        self.promotions.append((src.clone().cpu(), dst.clone().cpu()))


def make_alloc(n8=64 * 40, n4=64 * 20, **env):
    for k, v in env.items():
        os.environ[k] = str(v)
    try:
        return core.KVMixAllocator(n8, n4, PS, torch.float8_e4m3fn, DEV, FakeKV(), need_sort=False)
    finally:
        for k in env:
            os.environ.pop(k, None)


# ------------------------------------------------------------------------------------------------ sizing


def test_blended_and_split_fit_the_budget():
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        b = core.blended_row_bytes(576, 352, frac)
        assert 352 <= b <= 576
        for total in (64, 6400, 1_900_032):
            n8, n4 = core.split_tokens(total, 576, 352, b, frac, PS)
            assert n8 + n4 == total and n8 % PS == 0 and n4 % PS == 0
            assert n8 * 576 + n4 * 352 <= total * b
            # within one page of the requested byte fraction (unless clamped by the ceil of the blended row)
            want4 = frac * total * b / 352
            assert abs(n4 - want4) <= PS + total * (b - 1.0 / ((1 - frac) / 576 + frac / 352)) / 352 + 1


def test_blended_half_matches_derived_numbers():
    b = core.blended_row_bytes(576, 352, 0.5)
    assert b == 437  # harmonic mix of 576 and 352 = 436.97 -> 437 (derived)
    n8, n4 = core.split_tokens(1_900_032, 576, 352, b, 0.5, PS)
    assert n8 < n4 and abs(n8 * 576 - n4 * 352) < 2 * PS * 576


# ------------------------------------------------------------------------------------------------ allocator


def _stock_extend(size, prefix, seq, last, ext):
    a = PagedTokenToKVPoolAllocator(size, PS, torch.float8_e4m3fn, DEV, None, need_sort=False)
    if DEV == "cpu":
        from sglang.srt.mem_cache.allocator.paged import alloc_extend_naive

        out = torch.empty(ext, dtype=torch.int64)
        alloc_extend_naive(prefix, seq, last, a.free_pages, out, PS, "cpu")
        return out
    return a.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, ext)


def test_single_group_matches_stock_kernel():
    a = make_alloc()
    prefix = torch.tensor([0, 100, 130], dtype=torch.int64, device=DEV)
    seq = torch.tensor([70, 200, 131], dtype=torch.int64, device=DEV)
    last = torch.tensor([-1, 2 * PS + 35, 4 * PS + 1], dtype=torch.int64, device=DEV)
    ext = int((seq - prefix).sum())
    a.pending = [False, False, False]
    out = a.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, ext)
    ref = _stock_extend(a.n8, prefix, seq, last, ext)
    assert torch.equal(out.cpu(), ref.cpu())
    assert bool((out < a.boundary).all())
    pages_used = 2 + 2 + 0  # req0: 70 tokens -> 2 pages; req1: 100->200 crosses pages 2..3 -> 2 new; req2: none
    assert a.fp8.available_size() == a.n8 - pages_used * PS and a.fp4.available_size() == a.n4


def test_mixed_groups_scatter_in_request_order():
    a = make_alloc()
    prefix = torch.tensor([0, 0, 64, 0], dtype=torch.int64, device=DEV)
    seq = torch.tensor([10, 130, 100, 64], dtype=torch.int64, device=DEV)
    last = torch.tensor([-1, -1, 63 + 5 * PS, -1], dtype=torch.int64, device=DEV)
    flags = [False, True, True, False]
    ext = int((seq - prefix).sum())
    a.pending = list(flags)
    out = a.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, ext)
    lens = (seq - prefix).tolist()
    pos = 0
    for i, n in enumerate(lens):
        seg = out[pos:pos + n]
        if flags[i]:
            assert bool((seg >= a.boundary).all()), i
        else:
            assert bool((seg < a.boundary).all()), i
        # contiguous inside pages, pages page-aligned
        assert bool(((seg[1:] - seg[:-1] == 1) | (seg[1:] % PS == 0)).all())
        pos += n
    assert len(set(out.tolist())) == out.numel()
    # request 2 continues its FP8 last page (prefix 64 -> 100 needs a new page: prefix ended exactly on a page)
    assert a.fp4.available_size() == a.n4 - (3 + 1) * PS  # req1: 130 -> 3 pages; req2: 36 tokens -> 1 page
    assert a.fp8.available_size() == a.n8 - 2 * PS  # req0 1 page, req3 1 page


def test_alloc_decode_mixed_and_new_pages():
    a = make_alloc()
    seq = torch.tensor([65, 66, 129, 2], dtype=torch.int64, device=DEV)  # after +1: new page when seq % 64 == 1
    last = torch.tensor([7 * PS + 63, a.boundary + 3 * PS + 0, a.boundary + 5 * PS + 63, 9 * PS], dtype=torch.int64,
                        device=DEV)
    a.pending = [False, True, True, False]
    out = a.alloc_decode(seq, seq.cpu(), last)
    assert out[1].item() == last[1].item() + 1 and out[3].item() == last[3].item() + 1
    assert out[0].item() % PS == 0 and out[0].item() < a.boundary
    assert out[2].item() % PS == 0 and out[2].item() >= a.boundary
    assert a.fp8.available_size() == a.n8 - PS and a.fp4.available_size() == a.n4 - PS


def test_flags_inferred_from_last_loc_when_missing():
    a = make_alloc()
    seq = torch.tensor([65, 65], dtype=torch.int64, device=DEV)
    last = torch.tensor([PS + 63, a.boundary + PS + 63], dtype=torch.int64, device=DEV)
    out = a.alloc_decode(seq, seq.cpu(), last)
    assert out[0] < a.boundary <= out[1] and a.stats["inferred_flags"] == 1


def test_free_routes_by_slot_and_page():
    a = make_alloc()
    a.pending = [True, False]
    prefix = torch.tensor([0, 0], dtype=torch.int64, device=DEV)
    seq = torch.tensor([PS * 3, PS * 2], dtype=torch.int64, device=DEV)
    last = torch.tensor([-1, -1], dtype=torch.int64, device=DEV)
    out = a.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, PS * 5)
    assert a.available_size() == a.n8 + a.n4 - 5 * PS
    a.free(out)
    assert a.fp8.available_size() == a.n8 and a.fp4.available_size() == a.n4
    a.pending = [True, False]
    out = a.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, PS * 5)
    a.free_segment(out[: 3 * PS], start_pos=0)  # the FP4 request's three pages
    assert a.fp4.available_size() == a.n4 and a.fp8.available_size() == a.n8 - 2 * PS
    a.free_page_ids(out[3 * PS:: PS] // PS)
    assert a.fp8.available_size() == a.n8
    a.free_group_begin()
    a.pending = [False]
    o2 = a.alloc_extend(prefix[:1], prefix[:1].cpu(), seq[:1], seq[:1].cpu(), last[:1], 3 * PS)
    a.free(o2)
    assert a.fp8.available_size() == a.n8 - 3 * PS  # held until the group ends
    a.free_group_end()
    assert a.fp8.available_size() == a.n8


def test_fallback_when_a_pool_is_exhausted():
    a = make_alloc(n8=64 * 4, n4=64 * 2)
    prefix = torch.tensor([0], dtype=torch.int64, device=DEV)
    seq = torch.tensor([PS * 3], dtype=torch.int64, device=DEV)
    last = torch.tensor([-1], dtype=torch.int64, device=DEV)
    a.pending = [True]  # FP4 wants 3 pages, the FP4 pool has 2
    out = a.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, PS * 3)
    assert out is not None and int((out >= a.boundary).sum()) == 2 * PS and int((out < a.boundary).sum()) == PS
    assert a.stats["fallback_4to8_pages"] == 1 and a.fp4.available_size() == 0
    a.pending = [False]
    seq2 = torch.tensor([PS * 4], dtype=torch.int64, device=DEV)
    out2 = a.alloc_extend(prefix, prefix.cpu(), seq2, seq2.cpu(), last, PS * 4)
    assert out2 is None  # 3 FP8 pages left, FP4 empty: total short -> None (stock raises on this)
    os.environ["NUMINOUS_KVMIX_FALLBACK"] = "0"
    try:
        b = make_alloc(n8=64 * 4, n4=64 * 2)
    finally:
        del os.environ["NUMINOUS_KVMIX_FALLBACK"]
    b.pending = [True]
    assert b.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last, PS * 3) is None
    assert b.fp4.available_size() == b.n4  # nothing consumed on failure


def test_alloc_fp4_slots_and_capacity_checks():
    a = make_alloc(n8=64 * 4, n4=64 * 3)
    s = a.alloc_fp4_slots(2)
    assert s.numel() == 2 * PS and bool((s >= a.boundary).all()) and a.fp4.available_size() == PS
    assert a.alloc_fp4_slots(2) is None
    assert a.check_decode_capacity(num_tokens=PS, tree_cache=None, requests=None)
    assert a.available_size() == 5 * PS


# ------------------------------------------------------------------------------------------------ promotion


def _radix(alloc):
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache

    r2t = ReqToTokenPool(8, 4096, DEV, False)
    tree = RadixCache(CacheInitParams(disable=False, req_to_token_pool=r2t, token_to_kv_pool_allocator=alloc,
                                      page_size=PS))
    return tree, r2t


def _insert(tree, tokens, slots):
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    key = RadixKey(list(tokens)).page_aligned(PS)
    return tree.insert(InsertParams(key=key, value=slots[: len(key)].clone()))


def _match(tree, tokens):
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    return tree.match_prefix(MatchPrefixParams(key=RadixKey(list(tokens))))


def _req(rid, tokens, prefix_indices, last_node, protected, req_pool_idx=None, allocated=0):
    kv = types.SimpleNamespace(req_pool_idx=req_pool_idx, cache_protected_len=protected, kv_committed_len=allocated,
                               kv_allocated_len=allocated, holds_kv=req_pool_idx is not None)
    return types.SimpleNamespace(rid=rid, origin_input_ids=list(tokens), output_ids=[], prefix_indices=prefix_indices,
                                 last_node=last_node, kv=kv, extend_range=types.SimpleNamespace(end=len(tokens),
                                                                                                    length=0))


def test_promote_exclusive_nodes_and_own_pages():
    a = make_alloc(n8=64 * 40, n4=64 * 30, NUMINOUS_KVMIX_MIN_PROMOTE_TOKENS=64, NUMINOUS_KVMIX_THRESHOLD=100)
    tree, r2t = _radix(a)
    # a cached conversation of 4 pages (FP8) inserted by a finished request
    slots = a.alloc(4 * PS)
    tokens = list(range(1000, 1000 + 4 * PS))
    _insert(tree, tokens, slots)
    assert a.fp8.available_size() == a.n8 - 4 * PS
    # the next turn: matches the 4 pages, then holds 2 own pages (one partial) in its req_to_token row
    m = _match(tree, tokens + [7] * 90)
    assert m.device_indices.numel() == 4 * PS
    tree.inc_lock_ref(m.last_device_node)
    own = a.alloc(2 * PS)
    row = torch.cat([m.device_indices.to(DEV), own])
    L = 4 * PS + 90
    r2t.write((3, slice(0, 6 * PS)), row.to(torch.int32))
    req = _req("r1", tokens + [7] * 90, m.device_indices, m.last_device_node, protected=4 * PS, req_pool_idx=3,
               allocated=L)
    before8, before4 = a.fp8.available_size(), a.fp4.available_size()
    ok = core.promote_request(a, tree, req, r2t, "decode")
    assert ok
    kvc = a.get_kvcache()
    assert len(kvc.promotions) == 1
    src, dst = kvc.promotions[0]
    assert src.numel() == 6 * PS and dst.numel() == 6 * PS  # 4 node pages + 2 own pages (the partial one whole)
    assert torch.equal(src, row.cpu()[: 6 * PS]) and bool((src < a.boundary).all())
    assert bool((dst >= a.boundary).all())
    # the FP8 pages went back, 6 FP4 pages were taken
    assert a.fp8.available_size() == before8 + 6 * PS and a.fp4.available_size() == before4 - 6 * PS
    # node value and req_to_token point at the FP4 copies, position structure preserved
    new_row = r2t.req_to_token[3, :L].to(torch.int64)
    assert bool((new_row >= a.boundary).all())
    assert torch.equal(new_row % PS, torch.arange(L, device=DEV) % PS)
    node_vals = torch.cat([n.value for n in core._path_nodes(tree, req)])
    assert torch.equal(node_vals.cpu(), new_row[: 4 * PS].cpu())
    # promoting again finds nothing to do
    assert core.promote_request(a, tree, req, r2t, "decode") and len(kvc.promotions) == 1
    # a later request matching the cached turn now reads FP4 rows from the tree
    m2 = _match(tree, tokens)
    assert bool((m2.device_indices >= a.boundary).all())
    # eviction returns the FP4 pages to the FP4 pool
    tree.dec_lock_ref(m.last_device_node)
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    tree.evict(EvictParams(num_tokens=4 * PS))
    assert a.fp4.available_size() == before4 - 2 * PS


def test_shared_and_small_nodes_stay_fp8():
    a = make_alloc(n8=64 * 40, n4=64 * 30, NUMINOUS_KVMIX_MIN_PROMOTE_TOKENS=128, NUMINOUS_KVMIX_THRESHOLD=100)
    tree, r2t = _radix(a)
    sys_tokens = list(range(500, 500 + PS))  # one-page "system prompt" node (below the 128-token floor)
    conv_tokens = list(range(2000, 2000 + 3 * PS))
    s_all = a.alloc(4 * PS)
    _insert(tree, sys_tokens + conv_tokens, s_all)
    m_short = _match(tree, sys_tokens)  # another running request shares the system prompt node
    tree.inc_lock_ref(m_short.last_device_node)
    m = _match(tree, sys_tokens + conv_tokens)
    tree.inc_lock_ref(m.last_device_node)
    req = _req("r2", sys_tokens + conv_tokens, m.device_indices, m.last_device_node, protected=4 * PS)
    assert core.promote_request(a, tree, req, r2t, "extend")
    src, dst = a.get_kvcache().promotions[0]
    assert src.numel() == 3 * PS  # only the exclusive conversation node moved
    assert bool((req.prefix_indices[:PS] < a.boundary).all()) and bool((req.prefix_indices[PS:] >= a.boundary).all())
    # the shared node kept its FP8 slots
    assert bool((m_short.last_device_node.value < a.boundary).all())


def test_unified_like_cache_promotes_own_pages_only():
    """A cache that is not the plain RadixCache: the tree-owned prefix stays where it is, own pages move."""
    a = make_alloc(n8=64 * 40, n4=64 * 30, NUMINOUS_KVMIX_MIN_PROMOTE_TOKENS=64, NUMINOUS_KVMIX_THRESHOLD=100)
    tree, r2t = _radix(a)

    class UnifiedLike:
        root_node = None
        disable = False

        def is_chunk_cache(self):
            return False

    row = a.alloc(5 * PS)
    L = 4 * PS + 30
    r2t.write((2, slice(0, 5 * PS)), row.to(torch.int32))
    req = _req("u1", [3] * L, None, None, protected=3 * PS, req_pool_idx=2, allocated=L)
    assert core.promote_request(a, UnifiedLike(), req, r2t, "decode")
    src, dst = a.get_kvcache().promotions[0]
    assert src.numel() == 2 * PS and torch.equal(src, row.cpu()[3 * PS: 5 * PS])  # pages 3 and 4 only
    new_row = r2t.req_to_token[2, :L].to(torch.int64)
    assert torch.equal(new_row[: 3 * PS], row[: 3 * PS]) and bool((new_row[3 * PS:] >= a.boundary).all())
    assert a.fp8.available_size() == a.n8 - 3 * PS and a.fp4.available_size() == a.n4 - 2 * PS


def test_promotion_waits_for_fp4_room():
    a = make_alloc(n8=64 * 40, n4=64 * 2, NUMINOUS_KVMIX_MIN_PROMOTE_TOKENS=64, NUMINOUS_KVMIX_THRESHOLD=100)
    tree, r2t = _radix(a)
    slots = a.alloc(4 * PS)
    tokens = list(range(3000, 3000 + 4 * PS))
    _insert(tree, tokens, slots)
    m = _match(tree, tokens)
    tree.inc_lock_ref(m.last_device_node)
    req = _req("r3", tokens, m.device_indices, m.last_device_node, protected=4 * PS)
    assert core.promote_request(a, tree, req, r2t, "extend") is True  # done: nothing moved, no retry storm
    assert not a.get_kvcache().promotions and a.stats["promotion_no_fp4_pages"] == 1
    assert bool((req.prefix_indices < a.boundary).all()) and a.fp4.available_size() == a.n4


def test_before_alloc_sets_flags_and_promotes_once():
    a = make_alloc(n8=64 * 40, n4=64 * 30, NUMINOUS_KVMIX_MIN_PROMOTE_TOKENS=64, NUMINOUS_KVMIX_THRESHOLD=200)
    tree, r2t = _radix(a)
    short = _req("s", [1] * 50, torch.empty(0, dtype=torch.int64, device=DEV), tree.root_node, protected=0)
    long_ = _req("l", [2] * 300, torch.empty(0, dtype=torch.int64, device=DEV), tree.root_node, protected=0)
    long_.extend_range = types.SimpleNamespace(end=300, length=300)
    short.extend_range = types.SimpleNamespace(end=50, length=50)
    core.before_alloc(a, tree, [short, long_], r2t, "extend")
    assert a.pending == [False, True] and long_.kvmix_fp4 and not getattr(short, "kvmix_fp4", False)
    assert long_.kvmix_promoted is True  # nothing to promote (no pages yet) counts as done
    assert a.pending_need == (50 + PS, 300 + PS)
    # decode: a short request that just reached the threshold flips
    short.kv.kv_allocated_len = 199
    core.before_alloc(a, tree, [short], r2t, "decode")
    assert a.pending == [True] and short.kvmix_fp4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
