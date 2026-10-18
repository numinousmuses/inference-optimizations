"""The fused DP-MoE phases against SGLang's current path, eight simulated ranks, CPU (Triton interpreter).

Current path (reference): gather every rank's bf16 tokens, run router / shared expert / MXFP8 quantization / the
rank's TP slice of the experts on the whole gathered buffer on every rank, all-reduce the partial outputs, scale,
take this rank's rows, add the shared output.

Fused path: route, quantize and run the shared expert on local rows only, push MXFP8 + scales + top-k to every
peer, run the experts on the gathered quantized rows into a per-rank partial buffer, pull and reduce own rows.
"""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

import numinous_dp_moe as dm  # noqa: E402

WORLD, H, TOPK, E, INTER = 8, 256, 6, 16, 64
SCALE = 2.5


def mxfp8(x: torch.Tensor):
    """Per-32-element e8m0 scale (power of two >= amax/448) and e4m3 payload, as FlashInfer's MXFP8 quantizer."""
    g = x.float().reshape(x.shape[0], x.shape[1] // dm.MXFP8_GROUP, dm.MXFP8_GROUP)
    amax = g.abs().amax(-1).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (g / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn).reshape(x.shape)
    return q, (e + 127).to(torch.uint8)


def dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    g = q.float().reshape(q.shape[0], q.shape[1] // dm.MXFP8_GROUP, dm.MXFP8_GROUP)
    return (g * torch.exp2(s.float() - 127)[..., None]).reshape(q.shape)


class Model:
    """A small DeepSeek-style MoE: router, top-k with weights, E experts whose intermediate dim is split over ranks
    (each rank computes a TP partial), a replicated shared expert."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.wg = torch.randn(H, E, generator=g) / H ** 0.5
        self.w1 = torch.randn(E, H, INTER, generator=g) / H ** 0.5
        self.w2 = torch.randn(E, INTER, H, generator=g) / INTER ** 0.5
        self.ws1 = torch.randn(H, INTER, generator=g) / H ** 0.5
        self.ws2 = torch.randn(INTER, H, generator=g) / INTER ** 0.5

    def route(self, x):
        logits = x.float() @ self.wg
        w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
        return ids.to(torch.int32), (w / w.sum(-1, keepdim=True)).float()

    def shared(self, x):
        return (torch.relu(x.float() @ self.ws1) @ self.ws2).to(torch.bfloat16)

    def partial(self, rank, xq, xs, ids, wts, out):
        """Rank's TP slice of every expert on every row (unscaled), written into out (bf16)."""
        x = dequant(xq, xs)
        sl = slice(rank * INTER // WORLD, (rank + 1) * INTER // WORLD)
        acc = torch.zeros(x.shape[0], H)
        for k in range(TOPK):
            e = ids[:, k].long()
            h = torch.relu(torch.einsum("th,thi->ti", x, self.w1[e][:, :, sl]))
            acc += wts[:, k:k + 1] * torch.einsum("ti,tih->th", h, self.w2[e][:, sl, :])
        out.copy_(acc.to(torch.bfloat16))


def reference(model, xs_local):
    """SGLang's current path."""
    gathered = torch.cat(xs_local)
    ids, wts = model.route(gathered)
    q, s = mxfp8(gathered)
    total = torch.zeros(gathered.shape[0], H)
    for r in range(WORLD):
        part = torch.empty(gathered.shape[0], H, dtype=torch.bfloat16)
        model.partial(r, q, s, ids, wts, part)
        total += part.float()                               # the all-reduce
    full = (total * SCALE).to(torch.bfloat16).float()       # routed.mul_(scale), bf16 like the live buffer
    shared = model.shared(gathered).float()
    outs, off = [], 0
    for x in xs_local:
        n = x.shape[0]
        outs.append(full[off:off + n] + shared[off:off + n])
        off += n
    return outs


def fused(model, xs_local, rows_cap=64):
    counts = [x.shape[0] for x in xs_local]
    offs = [sum(counts[:r]) for r in range(WORLD)]
    rows = sum(counts)
    peers = lambda shape, dtype: [torch.zeros(shape, dtype=dtype) for _ in range(WORLD)]  # noqa: E731
    xq, xs, ids, wts, part = (peers((rows_cap, H), torch.uint8), peers((rows_cap, H // dm.MXFP8_GROUP), torch.uint8),
                              peers((rows_cap, TOPK), torch.int32), peers((rows_cap, TOPK), torch.float32),
                              peers((rows_cap, H), torch.bfloat16))
    bufs = [dm.DpMoeBuffers(WORLD, r, xq, xs, ids, wts, part, barrier=lambda: None) for r in range(WORLD)]
    preps = []
    for r, x in enumerate(xs_local):                        # phase 1 on every rank: local work, push
        i, w = model.route(x)
        q, s = mxfp8(x)
        preps.append(dm.LocalPrep(q, s, i, w, model.shared(x)))
        dm.phase_push(preps[r], bufs[r], offs[r])
    for r in range(WORLD):                                  # phase 2: experts on the gathered quantized rows
        dm.phase_experts(bufs[r], rows, lambda q, s, i, w, o, r=r: model.partial(r, q, s, i, w, o))
    outs = []
    for r in range(WORLD):                                  # phase 3: pull own rows, scale, add shared
        out = torch.empty(counts[r], H, dtype=torch.bfloat16)
        dm.phase_pull(bufs[r], offs[r], counts[r], out, SCALE, preps[r].shared)
        outs.append(out.float())
    return outs


@pytest.mark.parametrize("counts", [[3, 1, 4, 1, 5, 9, 2, 6], [0, 12, 0, 0, 1, 0, 0, 3], [4] * 8])
def test_fused_matches_the_current_path(counts):
    torch.manual_seed(sum(counts))
    model = Model()
    xs_local = [torch.randn(c, H).to(torch.bfloat16) for c in counts]
    for got, want in zip(fused(model, xs_local), reference(model, xs_local)):
        assert got.shape == want.shape
        if got.numel():
            torch.testing.assert_close(got, want, rtol=2e-2, atol=3e-2)


def test_local_routing_equals_gathered_routing():
    """Routing is per token, so routing local rows then gathering equals routing the gathered buffer."""
    model = Model(3)
    xs_local = [torch.randn(c, H).to(torch.bfloat16) for c in [2, 5, 1, 0, 3, 3, 1, 4]]
    ids_g, w_g = model.route(torch.cat(xs_local))
    loc = [model.route(x) for x in xs_local]
    assert torch.equal(torch.cat([i for i, _ in loc]), ids_g)
    torch.testing.assert_close(torch.cat([w for _, w in loc]), w_g)


def test_wire_bytes_per_token_are_about_half():
    per_token = 7168 + 7168 // dm.MXFP8_GROUP + 6 * 4 + 6 * 4
    assert per_token == 7440
    assert per_token / (7168 * 2) < 0.52


def test_declined_path_logs_once_and_returns_false(caplog):
    import logging
    import types
    import numinous_dp_moe as ndm
    ndm._LOGGED.clear()
    fb = types.SimpleNamespace(dp_padding_mode=object())
    with caplog.at_level(logging.WARNING, logger="sglang.numinous"):
        assert ndm.covers(types.SimpleNamespace(), fb) is False
        assert ndm.covers(types.SimpleNamespace(), fb) is False
    assert sum("numinous_dp_moe declined" in r.message for r in caplog.records) == 1


def test_routed_scale_is_applied_once():
    import types
    import numinous_dp_moe as ndm
    mx = types.SimpleNamespace(routed_scaling_factor=2.5, experts=types.SimpleNamespace(
        should_fuse_routed_scaling_factor_in_topk=False))
    nv = types.SimpleNamespace(routed_scaling_factor=2.5, experts=types.SimpleNamespace(
        should_fuse_routed_scaling_factor_in_topk=True))
    assert ndm.routed_scale(mx) == 2.5          # MXFP4 TRT-LLM: scaled after the experts
    assert ndm.routed_scale(nv) == 1.0          # NVFP4: the top-k weights already carry it


# ---------------------------------------------------------------------------------------------------------------
# Verify mode (NUMINOUS_DP_MOE_VERIFY): the comparison plumbing, CPU.
# ---------------------------------------------------------------------------------------------------------------
def test_diff_stats_reads_scale_and_part_projection():
    torch.manual_seed(0)
    ref = torch.randn(64, 32)
    st = dm.diff_stats(ref.clone(), ref)
    assert st["rows"] == 64 and st["rel_l2"] == 0.0 and st["err_absmax"] == 0.0
    assert abs(st["alpha"] - 1.0) < 1e-6
    st = dm.diff_stats(2.5 * ref, ref)                       # routed scale applied twice
    assert abs(st["alpha"] - 2.5) < 1e-6 and abs(st["rel_l2"] - 1.5) < 1e-6
    shared = torch.randn(64, 32)
    routed = torch.randn(64, 32)
    routed -= (routed * shared).sum() / (shared * shared).sum() * shared     # make the parts orthogonal
    ref = routed + shared
    st = dm.diff_stats(ref + shared, ref, {"shared": shared, "routed": routed})
    assert abs(st["beta_shared"] - 1.0) < 1e-6               # the error IS one extra shared add
    assert abs(st["beta_routed"]) < 1e-6
    st = dm.diff_stats(ref - shared, ref, {"shared": shared, "routed": routed})
    assert abs(st["beta_shared"] + 1.0) < 1e-6               # one missing shared add


def test_pack_gathered_matches_layout_offsets():
    counts = [3, 0, 5, 2]
    m = max(counts)
    parts = [torch.full((m, 4), float(r)) for r in range(4)]
    packed = dm.pack_gathered(parts, counts)
    assert packed.shape == (sum(counts), 4)
    for r, c in enumerate(counts):
        off = sum(counts[:r])
        assert torch.all(packed[off:off + c] == float(r)), r


class _VerifyMlp:
    """A stand-in layer whose stock forward is y = 3x; no shared expert."""
    layer_id = 7
    is_hash = False
    shared_experts = None
    num_fused_shared_experts = 0

    def forward_normal(self, x, **kw):
        return 3.0 * x


def _verify_env(monkeypatch, layers="4", steps="2"):
    monkeypatch.setenv("NUMINOUS_DP_MOE_VERIFY", layers)
    monkeypatch.setenv("NUMINOUS_DP_MOE_VERIFY_STEPS", steps)
    dm._VERIFY_SEEN.clear()
    dm._VERIFY_DONE.clear()


def _fb_sum_len(counts):
    import types
    return types.SimpleNamespace(dp_padding_mode=types.SimpleNamespace(is_max_len=lambda: False),
                                 global_num_tokens_cpu=counts)


def test_maybe_verify_logs_stats_and_respects_limits(monkeypatch, caplog):
    import logging
    _verify_env(monkeypatch)
    # simulate two identical ranks: the gathered buffer is this rank's rows twice (rank 0 at row 0)
    monkeypatch.setattr(dm, "_verify_gather", lambda x, n, fb, rows: torch.cat([x[:n], x[:n]]))
    mlp = _VerifyMlp()
    x = torch.randn(4, 16)
    fb = _fb_sum_len([4, 4])
    with caplog.at_level(logging.WARNING, logger="sglang.numinous"):
        dm.maybe_verify(mlp, x, 3.0 * x, fb, 0, 4, 8, 0)                    # exact
        dm.maybe_verify(mlp, x, 2.5 * 3.0 * x, fb, 0, 4, 8, 0)              # scaled 2.5x
        dm.maybe_verify(mlp, x, 3.0 * x, fb, 0, 4, 8, 0)                    # step 3 > NUMINOUS_DP_MOE_VERIFY_STEPS
    recs = [r.message for r in caplog.records if "verify layer 7" in r.message]
    assert len(recs) == 2, recs
    assert "step 0" in recs[0] and "rel_l2 0.000e+00" in recs[0] and "alpha 1.000000" in recs[0]
    assert "step 1" in recs[1] and "alpha 2.500000" in recs[1]


def test_maybe_verify_layer_cap_and_off(monkeypatch, caplog):
    import logging
    _verify_env(monkeypatch, layers="1", steps="8")
    monkeypatch.setattr(dm, "_verify_gather", lambda x, n, fb, rows: x[:n].clone())
    x = torch.randn(2, 8)
    fb = _fb_sum_len([2])
    first, second = _VerifyMlp(), _VerifyMlp()
    with caplog.at_level(logging.WARNING, logger="sglang.numinous"):
        dm.maybe_verify(first, x, 3.0 * x, fb, 0, 2, 2, 0)
        dm.maybe_verify(second, x, 3.0 * x, fb, 0, 2, 2, 0)                 # beyond the layer cap
    assert sum("verify layer" in r.message for r in caplog.records) == 1
    caplog.clear()
    monkeypatch.setenv("NUMINOUS_DP_MOE_VERIFY", "0")
    with caplog.at_level(logging.WARNING, logger="sglang.numinous"):
        dm.maybe_verify(first, x, 0.0 * x, fb, 0, 2, 2, 0)                  # off: no gather, no log
    assert not any("verify" in r.message for r in caplog.records)


def test_maybe_verify_shared_part_projection(monkeypatch, caplog):
    """With an unfused shared expert the log carries beta_shared / beta_routed; a double shared add reads as
    beta_shared 1."""
    import logging
    import types
    _verify_env(monkeypatch)
    monkeypatch.setattr(dm, "_verify_gather", lambda x, n, fb, rows: x[:n].clone())
    shared_w = torch.randn(8, 8)

    class Mlp(_VerifyMlp):
        shared_experts = object()

        def _forward_shared_experts(self, x):
            return x @ shared_w

        def forward_normal(self, x, **kw):
            return 3.0 * x + x @ shared_w

    x = torch.randn(4, 8)
    fb = _fb_sum_len([4])
    mlp = Mlp()
    with caplog.at_level(logging.WARNING, logger="sglang.numinous"):
        dm.maybe_verify(mlp, x, mlp.forward_normal(x) + x @ shared_w, fb, 0, 4, 4, 0)
    rec = [r.message for r in caplog.records if "verify layer" in r.message]
    assert len(rec) == 1 and "beta_shared 1.0000" in rec[0], rec


def test_maybe_verify_max_len_uses_real_rows(monkeypatch, caplog):
    """MAX_LEN: only this rank's real rows (global_num_tokens_cpu) are compared, pads are ignored."""
    import logging
    import types
    _verify_env(monkeypatch)
    monkeypatch.setattr(dm, "_verify_gather", lambda x, n, fb, rows: x[:n].clone())
    fb = types.SimpleNamespace(dp_padding_mode=types.SimpleNamespace(is_max_len=lambda: True),
                               global_num_tokens_cpu=[3])
    x = torch.randn(4, 8)                                    # 4-row block, 3 real rows
    out = 3.0 * x
    out[3:] = 123.0                                          # junk in the pad row must not count
    mlp = _VerifyMlp()
    with caplog.at_level(logging.WARNING, logger="sglang.numinous"):
        dm.maybe_verify(mlp, x, out, fb, 0, 4, 4, 0)
    rec = [r.message for r in caplog.records if "verify layer" in r.message]
    assert len(rec) == 1 and "rows 3" in rec[0] and "rel_l2 0.000e+00" in rec[0], rec
