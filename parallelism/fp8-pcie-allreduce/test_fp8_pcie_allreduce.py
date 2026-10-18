"""CPU tests (Triton interpreter) for the quantized PCIe all-reduce (agent rtxv41).

    python3 -m pytest -q operations/kernels/test_fp8_pcie_allreduce.py

* every rank's output is bitwise identical (MoE routing is recomputed per rank from the reduced state);
* the Triton kernels are bitwise equal to the torch model of the arithmetic (`reference_all_reduce`), for W in
  {2, 4, 8}, all four payload modes, V4.1 shapes (rows x 5120, shards that are not multiples of the 1024 block);
* error bounds against the exact fp32 sum, zero groups and outlier groups;
* persistent buffers reused across consecutive calls (the two-barrier protocol's ordering) give the same results;
* the integer e4m3 encoder/decoder used under the interpreter is bit-exact with torch's cast on every code, every
  midpoint and the subnormal range; int8 quantization matches torch.round (half-even).
"""
from __future__ import annotations

import os
import sys

os.environ["TRITON_INTERPRET"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8_pcie_allreduce as ar  # noqa: E402

BOUND = {"q8": 0.015, "q8_rs": 0.010, "fp8": 0.05, "fp8_rs": 0.035, "q8o": 0.015, "q8o_rs": 0.010}  # relative L2 vs exact sum, Gaussian inputs


def _inputs(W, rows, hidden=5120, seed=0, scale_rows=None):
    g = torch.Generator().manual_seed(seed)
    xs = []
    for _ in range(W):
        x = torch.randn(rows, hidden, generator=g)
        if scale_rows is not None:
            x = x * scale_rows
        xs.append(x.bfloat16())
    return xs


@pytest.mark.parametrize("mode", ar.MODES)
@pytest.mark.parametrize("W,rows", [(2, 1), (4, 3), (8, 2)])
def test_bitwise_reference_and_rank_identity(mode, W, rows):
    xs = _inputs(W, rows, seed=W * 10 + rows)
    outs, _ = ar.simulate_all_reduce(xs, mode)
    ref = ar.reference_all_reduce(xs, mode)
    for o in outs:
        assert torch.equal(o, outs[0])
    assert torch.equal(outs[0], ref)
    exact = sum(x.float() for x in xs)
    rel = ((outs[0].float() - exact).norm() / exact.norm()).item()
    assert rel < BOUND[mode], rel


def test_zero_and_outlier_groups():
    W = 4
    xs = _inputs(W, 2, seed=7)
    for x in xs:
        x.view(-1)[:128] = 0  # an all-zero group on every rank
        x.view(-1)[300] = 5000.0  # an outlier inside one group
    for mode in ("q8", "fp8", "q8o"):
        outs, _ = ar.simulate_all_reduce(xs, mode)
        assert torch.equal(outs[0], ar.reference_all_reduce(xs, mode))
        assert torch.all(outs[0].view(-1)[:128] == 0)
        assert torch.isfinite(outs[0].float()).all()
        assert abs(outs[0].view(-1)[300].float().item() - 20000.0) / 20000.0 < 0.01


def test_persistent_buffers_across_calls():
    W, rows = 4, 2
    g = ar.SimGroup(W, rows * 5120 * 2, "q8")  # capacity larger than the call: S < S_max
    a = _inputs(W, rows, seed=1)
    b = _inputs(W, rows, seed=2)
    oa = g.all_reduce(a)
    ob = g.all_reduce(b)
    oa2 = g.all_reduce(a)
    assert torch.equal(oa[0], ar.reference_all_reduce(a, "q8"))
    assert torch.equal(ob[0], ar.reference_all_reduce(b, "q8"))
    assert torch.equal(oa2[0], oa[0])


def test_layout_and_supports():
    lay = ar.buffer_layout(8, 384 * 5120, "q8")
    assert lay["S_max"] == 384 * 640
    for k in ("RS", "GQ", "GS", "bytes"):
        assert lay[k] % 256 == 0
    # per-rank bytes: 2 x (payload + scales) of the full tensor ~ 2.06 B/elem of max_numel
    assert abs(lay["bytes"] / (384 * 5120) - 2 * (1 + 4 / 128)) < 0.01
    assert ar.buffer_layout(8, 384 * 5120, "q8_rs")["bytes"] > lay["bytes"]
    r = ar.Fp8AllReduceRank(0, 8, torch.zeros(8, dtype=torch.int64), 384 * 5120, "q8")
    assert r.supports(torch.zeros(3, 5120, dtype=torch.bfloat16))
    assert not r.supports(torch.zeros(3, 5120, dtype=torch.float32))
    assert not r.supports(torch.zeros(385, 5120, dtype=torch.bfloat16))  # beyond capacity -> caller's fallback
    assert not r.supports(torch.zeros(1, 1000, dtype=torch.bfloat16))  # not a multiple of W x 128
    with pytest.raises(AssertionError):
        ar.buffer_layout(8, 1000)


@triton.jit
def _enc_kernel(y_ptr, q_ptr, d_ptr, N: tl.constexpr):
    y = tl.load(y_ptr + tl.arange(0, N))
    q = ar._e4m3_encode_manual(y)
    tl.store(q_ptr + tl.arange(0, N), q)
    tl.store(d_ptr + tl.arange(0, N), ar._e4m3_decode_manual(q))


def test_e4m3_integer_codec_bit_exact_with_torch():
    vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    vals = vals[~torch.isnan(vals)]
    pos = torch.sort(vals[vals >= 0]).values.unique()
    mids = (pos[:-1] + pos[1:]) / 2
    g = torch.Generator().manual_seed(0)
    y = torch.cat([vals, mids, -mids, torch.randn(3000, generator=g) * 60, torch.rand(1000, generator=g) * 0.03,
                   torch.tensor([448.0, -448.0, 0.0, 1e-9, 0.0009765625, 0.0029296875])]).clamp(-448, 448)
    n = 1 << (y.numel() - 1).bit_length()
    y = torch.cat([y, torch.zeros(n - y.numel())])
    q = torch.empty(n, dtype=torch.uint8)
    d = torch.empty(n)
    _enc_kernel[(1,)](y, q, d, n)
    assert torch.equal(q, y.to(torch.float8_e4m3fn).view(torch.uint8))
    assert torch.equal(d, q.view(torch.float8_e4m3fn).float())


def test_int8_quant_half_even_matches_torch():
    # values that land exactly on .5 after scaling: 127 / a with a = 127 -> inv = 1
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, 126.5, 127.0] + [0.0] * 121)
    q, s = ar._quant_ref(x, True)
    assert q[:7].tolist() == [0, 2, 2, 0, -2, 126, 127] and s.item() == 1.0
    xs = [x.bfloat16().view(1, 128).repeat(1, 2) for _ in range(2)]  # W=2, S=128
    outs, _ = ar.simulate_all_reduce(xs, "q8_rs")
    assert torch.equal(outs[0], ar.reference_all_reduce(xs, "q8_rs"))


def test_enabled_mode(monkeypatch):
    monkeypatch.delenv("SGLANG_RTX_QUANT_ALLREDUCE", raising=False)
    assert ar.enabled_mode() is None
    monkeypatch.setenv("SGLANG_RTX_QUANT_ALLREDUCE", "q8")
    assert ar.enabled_mode() == "q8"
    monkeypatch.setenv("SGLANG_RTX_QUANT_ALLREDUCE", "bogus")
    assert ar.enabled_mode() is None


def test_standalone_comm_covers_pcie_ipc_interface():
    """Every public method of upstream PcieIpcCommunicator (the slot StandaloneComm fills) exists on StandaloneComm.
    Regression: the 8 x RTX re-gate boot died in BaseRunner._pre_initialize_pcie_ipc_workspace on a missing prepare()."""
    import ast
    import inspect

    path = os.path.join(os.environ.get("SGLANG_SRC", "/tmp/sgl-cae69be5/python"),
                        "sglang/srt/distributed/device_communicators/pcie_ipc_ar.py")
    if not os.path.exists(path):
        pytest.skip("no sglang checkout")
    cls = next(n for n in ast.parse(open(path).read()).body
               if isinstance(n, ast.ClassDef) and n.name == "PcieIpcCommunicator")
    public = [f.name for f in cls.body if isinstance(f, ast.FunctionDef) and not f.name.startswith("_")]
    assert {"prepare", "destroy", "should_pcie_ipc_ar", "pcie_ipc_all_reduce"} <= set(public)
    for name in public:
        assert callable(getattr(ar.StandaloneComm, name, None)), name
    up = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "prepare")
    assert [a.arg for a in up.args.kwonlyargs] == list(inspect.signature(ar.StandaloneComm.prepare).parameters)[1:]
    c = ar.StandaloneComm(group=None, device="cpu", mode="q8")
    assert c.disabled is False
    c.destroy()
    assert c.disabled is True
    c.prepare(hidden=5120, max_rows=8)  # no-op once destroyed


def test_outlier_mode_protects_outlier_channel_groups():
    """Outlier channels (DeepSeek hidden states): q8o keeps the rest of an outlier's group accurate, q8 does not."""
    W, rows = 4, 2
    xs = _inputs(W, rows, seed=11)
    ch = [17, 900, 2048, 4000]
    for x in xs:
        x[:, ch] = (x[:, ch].float() * 200.0).bfloat16()
    exact = sum(x.float() for x in xs)
    keep = torch.ones(5120, dtype=torch.bool)
    keep[ch] = False
    err = {}
    for mode in ("q8", "q8o", "q8o_rs"):
        outs, _ = ar.simulate_all_reduce(xs, mode)
        assert all(torch.equal(o, outs[0]) for o in outs)
        assert torch.equal(outs[0], ar.reference_all_reduce(xs, mode))
        d = (outs[0].float() - exact)[:, keep]
        err[mode] = (d.norm() / exact[:, keep].norm()).item()
    assert err["q8"] > 0.05, err  # the failure the 8x accuracy gate saw
    assert err["q8o"] < 0.015 and err["q8o_rs"] < 0.012, err


def test_outlier_layout_bytes():
    a, b = ar.buffer_layout(8, 8 * 128 * 40, "q8"), ar.buffer_layout(8, 8 * 128 * 40, "q8o")
    assert b["bytes"] > a["bytes"] and b["GQ"] - b["RS"] == 3 * (a["GQ"] - a["RS"])


@pytest.mark.parametrize("mode", ["q8", "q8o", "q8o_rs", "fp8"])
@pytest.mark.parametrize("W,rows", [(2, 1), (4, 3), (8, 2)])
def test_pull_variant_bitwise_equal_to_push(mode, W, rows):
    """P2P-read (pull) phases: same bytes, same reduction order -> bitwise-identical to push and to the reference."""
    xs = _inputs(W, rows, seed=100 + W * 10 + rows)
    push, _ = ar.simulate_all_reduce(xs, mode, pull=False)
    pull, _ = ar.simulate_all_reduce(xs, mode, pull=True)
    ref = ar.reference_all_reduce(xs, mode)
    for o in pull:
        assert torch.equal(o, pull[0])
    assert torch.equal(pull[0], push[0])
    assert torch.equal(pull[0], ref)


def test_pull_persistent_buffers_across_calls():
    W = 4
    g = ar.SimGroup(W, 4 * 5120, "q8o", pull=True)
    for seed in range(3):
        xs = _inputs(W, 4, seed=200 + seed)
        outs = g.all_reduce(xs)
        assert torch.equal(outs[0], ar.reference_all_reduce(xs, "q8o"))
