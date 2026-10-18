"""extend_attention_gqa against the float32 reference, on CPU through Triton's interpreter (TRITON_INTERPRET=1):
several requests with different prefix and new-token lengths, a shuffled page table, full and sliding windows,
DV split and unsplit, BLOCK_T spanning partial tiles."""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

import extend_attention_gqa as eg  # noqa: E402


def _case(H, KVH, D, prefixes, news, seed=0):
    torch.manual_seed(seed)
    B = len(prefixes)
    P_tot, T_tot = sum(prefixes), sum(news)
    k_buf = torch.randn(P_tot + 8, KVH, D) * 0.5
    v_buf = torch.randn(P_tot + 8, KVH, D) * 0.5
    q = torch.randn(T_tot, H, D).to(torch.bfloat16)
    k_ext = (torch.randn(T_tot, KVH, D) * 0.5).to(torch.bfloat16)
    v_ext = (torch.randn(T_tot, KVH, D) * 0.5).to(torch.bfloat16)
    qo = torch.tensor([0] + list(torch.cumsum(torch.tensor(news), 0)), dtype=torch.int32)
    kvp = torch.tensor([0] + list(torch.cumsum(torch.tensor(prefixes), 0)), dtype=torch.int32)
    idx = torch.randperm(P_tot + 8)[:P_tot].to(torch.int64)
    return q, k_ext, v_ext, k_buf.to(torch.bfloat16), v_buf.to(torch.bfloat16), qo, kvp, idx, max(news)


@pytest.mark.parametrize("window", [-1, 24])
@pytest.mark.parametrize("dv", [None, 32])
def test_matches_reference(window, dv):
    H, KVH, D = 8, 2, 64
    q, ke, ve, kb, vb, qo, kvp, idx, mx = _case(H, KVH, D, prefixes=[70, 5, 0], news=[19, 33, 7])
    o = torch.empty_like(q)
    eg.extend_attention_gqa(q, ke, ve, o, kb, vb, qo, kvp, idx, mx, D ** -0.5, sliding_window_size=window,
                            block_t=4, block_n=16, dk=32, dv=dv, num_stages=1)
    ref = eg.reference(q, ke, ve, kb, vb, qo, kvp, idx, D ** -0.5, sliding_window_size=window)
    assert (o.float() - ref).abs().max() < 0.03


def test_kv_scale_is_applied_to_the_cached_prefix_only():
    H, KVH, D = 4, 1, 32
    q, ke, ve, kb, vb, qo, kvp, idx, mx = _case(H, KVH, D, prefixes=[40], news=[9], seed=3)
    o = torch.empty_like(q)
    eg.extend_attention_gqa(q, ke, ve, o, kb, vb, qo, kvp, idx, mx, D ** -0.5, k_scale=0.5, v_scale=2.0,
                            block_t=8, block_n=16, dk=32, num_stages=1)
    ref = eg.reference(q, ke, ve, kb, vb, qo, kvp, idx, D ** -0.5, k_scale=0.5, v_scale=2.0)
    assert (o.float() - ref).abs().max() < 0.03


def test_non_power_of_two_groups():
    # Qwen 3.8: 24 query heads on 4 KV heads, 6 per group
    H, KVH, D = 12, 2, 32
    q, ke, ve, kb, vb, qo, kvp, idx, mx = _case(H, KVH, D, prefixes=[33, 0], news=[11, 20], seed=5)
    o = torch.empty_like(q)
    eg.extend_attention_gqa(q, ke, ve, o, kb, vb, qo, kvp, idx, mx, D ** -0.5, block_t=4, block_n=16, dk=32,
                            num_stages=1)
    ref = eg.reference(q, ke, ve, kb, vb, qo, kvp, idx, D ** -0.5)
    assert (o.float() - ref).abs().max() < 0.03


def test_query_as_a_view_into_a_fused_projection():
    # in the engine q is sliced out of the fused qkv output, so its token stride is wider than heads x dim
    H, KVH, D = 8, 2, 32
    q, ke, ve, kb, vb, qo, kvp, idx, mx = _case(H, KVH, D, prefixes=[25], news=[13], seed=9)
    fused = torch.zeros(q.shape[0], H * D + 2 * KVH * D, dtype=q.dtype)
    fused[:, :H * D] = q.reshape(q.shape[0], -1)
    qv = fused[:, :H * D].view(q.shape[0], H, D)
    assert not qv.is_contiguous()
    o = torch.empty(q.shape, dtype=q.dtype)
    eg.extend_attention_gqa(qv, ke, ve, o, kb, vb, qo, kvp, idx, mx, D ** -0.5, block_t=4, block_n=16, dk=32,
                            num_stages=1)
    ref = eg.reference(q, ke, ve, kb, vb, qo, kvp, idx, D ** -0.5)
    assert (o.float() - ref).abs().max() < 0.03


def test_sliding_window_rule_is_q_minus_kv_le_w():
    # Stock SGLang extend rule (q_id <= kv_id + W): a query sees W + 1 keys including itself (Gemma 4: W = 1023 ->
    # 1024 keys, the HF window). Independent of eg.reference: one prefix key sits at distance exactly W from the
    # first new token and carries a dominant value, so it must show up in that token's output (and the key at
    # distance W + 1 must not).
    H, KVH, D, W = 2, 1, 32, 8
    P = 12
    torch.manual_seed(11)
    q = torch.zeros(3, H, D).to(torch.bfloat16)
    q[:, :, 0] = 1.0
    k_buf = torch.zeros(P + 1, KVH, D)
    v_buf = torch.zeros(P + 1, KVH, D)
    # new token 0 is at position P; the key at position P - W (distance W) is visible, P - W - 1 is not
    k_buf[P - W, 0, 0] = 8.0
    v_buf[P - W, 0, :] = 5.0
    k_buf[P - W - 1, 0, 0] = 8.0
    v_buf[P - W - 1, 0, :] = -50.0
    ke = torch.zeros(3, KVH, D).to(torch.bfloat16)
    ve = torch.zeros(3, KVH, D).to(torch.bfloat16)
    qo = torch.tensor([0, 3], dtype=torch.int32)
    kvp = torch.tensor([0, P], dtype=torch.int32)
    idx = torch.arange(P, dtype=torch.int64)
    o = torch.empty_like(q)
    eg.extend_attention_gqa(q, ke, ve, o, k_buf.to(torch.bfloat16), v_buf.to(torch.bfloat16), qo, kvp, idx, 3, 1.0,
                            sliding_window_size=W, block_t=4, block_n=16, dk=32, num_stages=1)
    # token 0: distance-W key visible (large positive weight on +5), distance-(W+1) key masked (no -50 pull)
    assert o[0].float().mean() > 4.0
    # token 1 (position P + 1): the +5 key is now at distance W + 1 -> masked
    assert o[1].float().abs().max() < 1e-3
    ref = eg.reference(q, ke, ve, k_buf.to(torch.bfloat16), v_buf.to(torch.bfloat16), qo, kvp, idx, 1.0,
                       sliding_window_size=W)
    assert (o.float() - ref).abs().max() < 0.03
