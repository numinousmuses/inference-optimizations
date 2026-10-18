#!/usr/bin/env python3
"""CPU tests for the MoE decode grouped GEMM: routing, work list, roofline, fp4 packing, and the bf16
reference. The kernels themselves need a GPU and are exercised by operations/modal/moe_kernel_bench.py."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_decode_gemm as mg


def test_experts_touched_matches_hand_numbers():
    assert abs(mg.experts_touched_fraction(1) - 10 / 512) < 1e-9
    assert 0.46 < mg.experts_touched_fraction(32) < 0.48
    assert 0.91 < mg.experts_touched_fraction(128) < 0.93
    assert mg.experts_touched_fraction(2048) > 0.999
    print("ok expert-touch fraction: 2% at one token, 92% at 128, ~all past 512")


def test_roofline_is_the_number_in_the_note():
    r = mg.roofline_ms(128)
    assert 130 < r["expert_gb_per_gpu"] < 150, r  # ~147 GB per GPU at 92% touched
    assert 16 < r["ms_at_hbm"] < 19, r
    r1 = mg.roofline_ms(1)
    assert r1["expert_gb_per_gpu"] < 4, r1
    print(f"ok roofline: c=128-tokens step reads {r['expert_gb_per_gpu']} GB/GPU = {r['ms_at_hbm']} ms; one token {r1['expert_gb_per_gpu']} GB")


def test_grouping_is_a_permutation_with_correct_offsets():
    torch.manual_seed(0)
    M, E, topk = 64, 512, 10
    ids = torch.stack([torch.randperm(E)[:topk] for _ in range(M)])
    sorted_tok, offsets, counts = mg.group_by_expert(ids, E)
    assert sorted_tok.numel() == M * topk and int(offsets[-1]) == M * topk
    assert torch.equal(offsets[1:] - offsets[:-1], counts)
    # every (token, expert) pair appears exactly once in its expert's segment
    for e in range(E):
        a, b = int(offsets[e]), int(offsets[e + 1])
        toks = sorted_tok[a:b].tolist()
        assert all(e in ids[t].tolist() for t in toks), e
        assert len(toks) == int((ids == e).sum()), e
    print("ok grouping: offsets partition M*topk rows exactly, each row in its expert's segment")


def test_work_list_skips_untouched_experts_and_tiles_m():
    counts = torch.tensor([0, 3, 0, 0, 1, 0], dtype=torch.int32)
    e, t, m = mg.work_list(counts, block_n=128, n=512, block_m=16)
    assert e.tolist() == [1, 1, 1, 1, 4, 4, 4, 4] and t.tolist() == [0, 1, 2, 3] * 2 and m.tolist() == [0] * 8
    # an expert with 30 rows at BLOCK_M=16 gets two M tiles per N tile; at 128 it gets one
    counts = torch.tensor([30, 0, 5], dtype=torch.int32)
    e, t, m = mg.work_list(counts, block_n=256, n=512, block_m=16)
    assert e.tolist() == [0, 0, 0, 0, 2, 2] and m.tolist() == [0, 0, 1, 1, 0, 0] and t.tolist() == [0, 1, 0, 1, 0, 1]
    e, t, m = mg.work_list(counts, block_n=256, n=512, block_m=128)
    assert m.tolist() == [0, 0, 0, 0]
    print("ok work list: touched experts only, M tiled on the host (30 rows -> 2 tiles at 16, 1 at 128)")


def test_e2m1_pack_roundtrip_is_bounded():
    torch.manual_seed(1)
    w = (torch.randn(4, 64, 256) * 0.05).to(torch.bfloat16)
    packed, scales, deq = mg.pack_e2m1(w)
    assert packed.shape == (4, 64, 128) and scales.shape == (4, 64, 8) and packed.dtype == torch.uint8
    rel = ((deq.float() - w.float()).norm() / w.float().norm()).item()
    assert rel < 0.25, rel  # 4-bit with block-32 power-of-two scales, expect ~15-20% rms error
    print(f"ok e2m1 pack: shapes right, relative rms error {rel:.3f}")


def test_reference_matches_dense_matmul():
    torch.manual_seed(2)
    M, E, topk, K, N = 8, 16, 2, 64, 32
    x = torch.randn(M, K).to(torch.bfloat16)
    w = (torch.randn(E, N, K) * 0.1).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E)[:topk] for _ in range(M)])
    st, off, cnt = mg.group_by_expert(ids, E)
    out = mg.reference_bf16(x, w, st, off)
    for r in range(out.shape[0]):
        e = int(torch.searchsorted(off, torch.tensor(r), right=True) - 1)
        want = (x[st[r].long()].float() @ w[e].float().T).to(torch.bfloat16)
        assert torch.allclose(out[r].float(), want.float(), atol=1e-2), r
    print("ok reference oracle agrees with dense per-row matmul")


if __name__ == "__main__":
    test_experts_touched_matches_hand_numbers()
    test_roofline_is_the_number_in_the_note()
    test_grouping_is_a_permutation_with_correct_offsets()
    test_work_list_skips_untouched_experts_and_tiles_m()
    test_e2m1_pack_roundtrip_is_bounded()
    test_reference_matches_dense_matmul()


def test_pack_e2m1_handles_activations_too():
    torch.manual_seed(3)
    x = torch.randn(8, 256).to(torch.bfloat16)
    xp, xs, xd = mg.pack_e2m1(x)
    assert xp.shape == (8, 128) and xs.shape == (8, 8) and xd.shape == (8, 256)
    rel = ((xd.float() - x.float()).norm() / x.float().norm()).item()
    assert rel < 0.25, rel
    print(f"ok e2m1 pack works on [M, K] activations (rel rms {rel:.3f})")


test_pack_e2m1_handles_activations_too()


def test_tile_layout_is_a_permutation_of_the_bytes():
    torch.manual_seed(4)
    E, N, K = 3, 256, 512
    w = (torch.randn(E, N, K) * 0.05).to(torch.bfloat16)
    wp, ws, _ = mg.pack_e2m1(w)
    wt, st, nt, kt = mg.tile_layout(wp, ws, block_n=128, block_k=256)
    assert (nt, kt) == (2, 2) and wt.shape == (E, 2, 2, 128, 128) and st.shape == (E, 2, 2, 128, 8)
    # tile (e, n_tile, k_tile) must equal the corresponding slice of the row-major packed weight
    for e in range(E):
        for a in range(nt):
            for b in range(kt):
                assert torch.equal(wt[e, a, b], wp[e, a * 128:(a + 1) * 128, b * 128:(b + 1) * 128])
                assert torch.equal(st[e, a, b], ws[e, a * 128:(a + 1) * 128, b * 8:(b + 1) * 8])
    assert wt.is_contiguous() and st.is_contiguous()
    print("ok tile layout: every tile is the right slice, contiguous")


test_tile_layout_is_a_permutation_of_the_bytes()


def test_transposed_tile_layout():
    torch.manual_seed(5)
    w = (torch.randn(2, 256, 512) * 0.05).to(torch.bfloat16)
    wp, ws, _ = mg.pack_e2m1(w)
    wt, st, nt, kt = mg.tile_layout(wp, ws, block_n=128, block_k=256, transposed=True)
    assert wt.shape == (2, 2, 2, 128, 128) and wt.is_contiguous()
    assert torch.equal(wt[1, 0, 1], wp[1, 0:128, 128:256].T)
    print("ok transposed tile layout: tile is the [KB, N] transpose of the row-major slice")


test_transposed_tile_layout()
