"""CPU tests for the pure parts of numinous_dp_moe_nvfp4 (Triton interpreter; no SGLang, no GPU).

  - the symmetric allocation's layout (NVFP4 widths, 2 MiB-aligned parts, 16-byte payload rows for vector pushes)
  - the routing mode default (logits unless the shared expert is fused or NUMINOUS_DP_MOE_ROUTE=ids)
  - wire bytes per token vs the bf16 gather
  - eight (and four: one TP4/DP4 replica) simulated ranks through push + the reused phases with NVFP4-shaped
    payloads (uint8 packed rows, E4M3 scale rows, and either top-9 ids / weights or fp32 router logits):
    MAX_LEN blocks with padding and an idle rank, uneven SUM_LEN with an idle rank; the gathered buffer each
    rank's experts read equals the stock gather byte for byte, and the pulled output equals the stock all-reduce
    + own-row copy (fixed peer order, fp32 sum) to one bf16 ulp
  - a rank with no rows produces empty routing of the configured width without calling the router
  - coverage declines with one log line and caches the static reason
"""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import logging  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import numinous_dp_moe as dm  # noqa: E402
import numinous_dp_moe_nvfp4 as n4  # noqa: E402

WORLD, H, K = 8, 256, 9                                  # H small for the interpreter; GLM is 6144


def test_buffer_specs_widths_and_alignment():
    specs = n4.buffer_specs(40960, 6144, n4.Route("ids", 9))
    by = {nm: (shape, dtype, off) for nm, shape, dtype, off in specs}
    assert by["xq"][0] == (40960, 3072) and by["xq"][1] == torch.uint8
    assert by["xs"][0] == (40960, 384) and by["xs"][1] == torch.uint8
    assert by["ids"][0] == (40960, 9) and by["ids"][1] == torch.int32
    assert by["wts"][0] == (40960, 9) and by["wts"][1] == torch.float32
    assert by["part"][0] == (40960, 6144) and by["part"][1] == torch.bfloat16
    offs = [off for _, _, _, off in specs]
    assert offs == sorted(offs) and all(o % n4.ALIGN == 0 for o in offs)
    for (nm, shape, dtype, off), nxt in zip(specs[:-2], offs[1:]):
        assert off + shape[0] * shape[1] * torch.empty(0, dtype=dtype).element_size() <= nxt, nm
    # payload rows are 16-byte multiples (vector / multicast push); ids and weights take the 32-bit path
    assert 3072 % 16 == 0 and 384 % 16 == 0
    assert by["total"][2] < 700 << 20                    # ~0.65 GB per GPU at the default 40,960 rows
    lg = {nm: (shape, dtype) for nm, shape, dtype, _ in n4.buffer_specs(40960, 6144, n4.Route("logits", 256))}
    assert lg["rt"] == ((40960, 256), torch.float32) and "ids" not in lg and "wts" not in lg
    assert (256 * 4) % 16 == 0                           # logits rows take the vector / multicast push too


def test_wire_bytes_per_token():
    assert n4.wire_bytes_per_token(6144, n4.Route("ids", 9)) == 3072 + 384 + 36 + 36 == 3528
    assert n4.wire_bytes_per_token(6144, n4.Route("ids", 8)) == 3520
    assert n4.wire_bytes_per_token(6144, n4.Route("logits", 256)) == 3072 + 384 + 1024 == 4480
    assert n4.wire_bytes_per_token(6144, n4.Route("logits", 256)) / (6144 * 2) < 0.37


def test_route_default_follows_shared_fusion(monkeypatch):
    monkeypatch.delenv("NUMINOUS_DP_MOE_ROUTE", raising=False)
    mk = lambda nfse: types.SimpleNamespace(num_fused_shared_experts=nfse, config=types.SimpleNamespace(  # noqa: E731
        n_routed_experts=256), topk=types.SimpleNamespace(topk_config=types.SimpleNamespace(top_k=8 + nfse)))
    assert n4.route_of(mk(0)) == n4.Route("logits", 256)
    assert n4.route_of(mk(1)) == n4.Route("ids", 9)
    monkeypatch.setenv("NUMINOUS_DP_MOE_ROUTE", "ids")
    assert n4.route_of(mk(0)) == n4.Route("ids", 8)


def _payload(n, seed, route):
    g = torch.Generator().manual_seed(seed)
    xq = torch.randint(0, 256, (n, H // 2), dtype=torch.uint8, generator=g)
    xs = torch.randint(0, 256, (n, H // n4.NVFP4_GROUP), dtype=torch.uint8, generator=g)
    if route.mode == "logits":
        return [("xq", xq), ("xs", xs), ("rt", torch.randn(n, route.width, generator=g))]
    ids = torch.randint(0, 257, (n, route.width), dtype=torch.int32, generator=g)
    wts = torch.rand(n, route.width, generator=g).to(torch.bfloat16).float()
    return [("xq", xq), ("xs", xs), ("ids", ids), ("wts", wts)]


def _partial(rank, xq, xs, rt, wts, out):
    """A deterministic stand-in for a rank's TP-partial expert output: any function of the gathered row bytes."""
    f = xq.float()[:, :1] * 0.01 + xs.float()[:, :1] * 0.001 + (rt.float() * wts.float()).sum(-1, keepdim=True)
    col = torch.arange(H, dtype=torch.float32)[None, :]
    out.copy_((f * torch.cos(col * (rank + 1) * 0.01)).to(torch.bfloat16))


def _simulate(counts, block, route):
    WORLD = len(counts)  # noqa: N806
    rows = WORLD * block if block else sum(counts)
    offs = [r * block for r in range(WORLD)] if block else [sum(counts[:r]) for r in range(WORLD)]
    cap = max(rows, 16)
    views = {nm: [torch.zeros(shape, dtype=dtype) for _ in range(WORLD)]
             for nm, shape, dtype, _ in n4.buffer_specs(cap, H, route)[:-1]}
    bufs = [n4.make_buffers(WORLD, r, views, route, barrier=lambda: None) for r in range(WORLD)]
    local = [_payload(block if block else c, 100 + r, route) for r, c in enumerate(counts)]
    for r in range(WORLD):
        n4.push(bufs[r], local[r], offs[r], multicast=block is None)
    stock = {nm: torch.cat([dict(p)[nm] for p in local]) for nm, _ in local[0]}   # the stock gather, same rows
    for r in range(WORLD):
        for nm, t in stock.items():
            assert torch.equal(views[nm][r][:rows], t), f"rank {r} {nm}"
        dm.phase_experts(bufs[r], rows, lambda q, s, i, w, o, r=r: _partial(r, q.view(torch.uint8), s, i, w, o))
    args = [stock["xq"], stock["xs"]] + ([stock["rt"], stock["rt"]] if route.mode == "logits"
                                         else [stock["ids"], stock["wts"]])
    total = torch.zeros(rows, H)
    for r in range(WORLD):
        p = torch.empty(rows, H, dtype=torch.bfloat16)
        _partial(r, *args, p)
        total += p.float()
    for r in range(WORLD):
        n = block if block else counts[r]
        valid = torch.tensor([counts[r]], dtype=torch.int64) if block else None
        out = torch.full((n, H), 7.0, dtype=torch.bfloat16)
        if n:
            dm.phase_pull(bufs[r], offs[r], n, out, 1.0, None, valid)
        want = total[offs[r]:offs[r] + counts[r]]
        # the same fp32 sum in the same peer order; the interpreter's fp32 -> bf16 cast may round toward zero where
        # the GPU rounds to nearest (bit-exactness is checked on 8xB200 by fused_path_bench_nvfp4), so: 1 bf16 ulp
        err = (out[:counts[r]].float() - want).abs()
        assert bool((err <= want.abs() * 2.0 ** -7 + 1e-6).all()), f"rank {r}: max err {float(err.max())}"
        if block:
            assert (out[counts[r]:] == 0).all(), f"rank {r} padding"


@pytest.mark.parametrize("route", [n4.Route("ids", K), n4.Route("logits", 32)], ids=["ids9", "logits32"])
@pytest.mark.parametrize("counts,block", [([12, 7, 12, 1, 12, 0, 12, 5], 12), ([3, 0, 9, 1, 0, 4, 2, 6], None),
                                          ([4] * WORLD, None),
                                          ([12, 0, 5, 12], 12), ([0, 9, 1, 4], None)],   # a 4-GPU replica (TP4/DP4)
                         ids=["w8_maxlen", "w8_sumlen_idle", "w8_even", "w4_maxlen", "w4_sumlen_idle"])
def test_nvfp4_payload_through_the_phases_matches_the_stock_gather_and_reduce(counts, block, route):
    _simulate(counts, block, route)


def test_idle_rank_routing_has_the_configured_width_and_skips_the_router():
    def boom(_):
        raise AssertionError("router called for an idle rank")
    mlp = types.SimpleNamespace(gate=boom, topk=types.SimpleNamespace(topk_config=types.SimpleNamespace(top_k=9)))
    ids, wts = n4.local_topk(mlp, torch.empty(0, H, dtype=torch.bfloat16), 0)
    assert ids.shape == (0, 9) and ids.dtype == torch.int32 and wts.shape == (0, 9) and wts.dtype == torch.float32
    ids, _ = n4.local_topk(mlp, torch.empty(0, H, dtype=torch.bfloat16), 0, width=8)
    assert ids.shape == (0, 8)


def test_decline_logs_once_and_caches_the_static_reason(caplog, monkeypatch):
    dm._LOGGED.clear()
    calls = []

    def reason(layer):
        calls.append(layer)
        return "experts use UnquantizedFusedMoEMethod, not ModelOptNvFp4FusedMoEMethod"
    monkeypatch.setattr(n4, "_layer_static_reason", reason)
    layer = types.SimpleNamespace()
    fb = types.SimpleNamespace(dp_padding_mode=None)
    with caplog.at_level(logging.WARNING, logger="sglang.numinous"):
        assert n4.covers(layer, fb) is False
        assert n4.covers(layer, fb) is False
    assert len(calls) == 1
    assert sum("numinous_dp_moe_nvfp4 declined" in r.message for r in caplog.records) == 1


def test_enabled_follows_the_shared_switch(monkeypatch):
    monkeypatch.setenv("NUMINOUS_DP_MOE", "1")
    assert n4.enabled()
    monkeypatch.setenv("NUMINOUS_DP_MOE", "0")
    assert not n4.enabled()
