"""CPU tests for GLM-5.3-Flash FP8 KV on Hopper (SM90): the stock rope-free 528-byte FP8 DSA row read by our Triton
sparse-MLA kernel through the engine (agents/findings/hopper_fp8kv.md).

    TRITON_INTERPRET=1 python3 -m pytest -q operations/kernels/test_glmflash_hopper_fp8kv.py

The writer test runs SGLang v0.5.20's OWN quantize_k_cache_separate (the stock FP8 DSA writer) under the Triton
interpreter when the tree is present (SGLANG_SRC, default /tmp/sgl-v0520-wt/python), so the layout our kernel reads is
the layout the stock pool writes, byte for byte.
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest
import torch

import glmflash_sm120_attn as A

pytestmark = pytest.mark.skipif(not A.HAVE_TRITON, reason="triton not installed")
SGLANG_SRC = os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python")


def _table(t: int, k: int, n: int, n_valid, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = torch.full((t, k), -1, dtype=torch.int32)
    for i in range(t):
        v = int(n_valid[i])
        idx[i, :v] = (torch.randperm(n - 1, generator=g)[:v] + 1).to(torch.int32)
    return idx


def _rows528(lat: torch.Tensor) -> torch.Tensor:
    return A.nsa656_quantize_ref(lat)[:, : A.STOCK_FP8_ROW].contiguous()


def test_row_constants():
    assert A.ROW_BYTES[A.FMT_FP8_528] == 528 == 512 + 512 // 128 * 4 + 0  # kv_cache_configurator, rope 0
    assert A.FMT_FP8_528 in A.FP8_FMTS and A.FMT_FP8_656 in A.FP8_FMTS


@pytest.mark.parametrize("splits", [1, 3])
def test_sparse_mla_528_matches_reference(splits):
    n, t, h, k = 300, 3, 16, 70
    lat = torch.randn(n, 512, generator=torch.Generator().manual_seed(0)) * 0.7
    rows = _rows528(lat)
    q = (torch.randn(t, h, 512) * 0.2).to(torch.bfloat16)
    idx = _table(t, k, n, [70, 37, 5])
    out, lse = A.sparse_mla(q, rows.reshape(n, 1, 528), idx, 1 / 16, A.FMT_FP8_528, splits=splits, block_n=16)
    ro, rl = A.sparse_mla_ref(q.float(), A.nsa656_dequant_ref(rows), idx, 1 / 16)
    assert (out.float() - ro).norm() / ro.norm() < 1e-2
    assert torch.allclose(lse, rl, atol=1e-3, rtol=1e-4)


def test_528_equals_656_on_the_same_values():
    """The 528 row is the 656 row minus its zero rope: identical attention."""
    n, t, h, k = 200, 2, 16, 64
    lat = torch.randn(n, 512, generator=torch.Generator().manual_seed(4)) * 0.5
    r656 = A.nsa656_quantize_ref(lat)
    q = (torch.randn(t, h, 512) * 0.2).to(torch.bfloat16)
    idx = _table(t, k, n, [64, 40], seed=2)
    o1, l1 = A.sparse_mla(q, r656.reshape(n, 1, 656), idx, 0.07, A.FMT_FP8_656, block_n=16)
    o2, l2 = A.sparse_mla(q, r656[:, :528].contiguous().reshape(n, 1, 528), idx, 0.07, A.FMT_FP8_528, block_n=16)
    assert torch.equal(o1, o2) and torch.equal(l1, l2)


def test_528_kpool_width_2051_and_tail():
    n, t, h, k = 2200, 1, 16, 2051
    lat = torch.randn(n, 512, generator=torch.Generator().manual_seed(3)) * 0.1
    rows = _rows528(lat)
    q = (torch.randn(t, h, 512) * 0.1).to(torch.bfloat16)
    idx = _table(t, k, n, [2051])
    out, _ = A.sparse_mla(q, rows.reshape(n, 1, 528), idx, 0.0625, A.FMT_FP8_528, block_n=32)
    ref_rows = A.nsa656_dequant_ref(rows)
    ro, _ = A.sparse_mla_ref(q.float(), ref_rows, idx, 0.0625)
    assert (out.float() - ro).norm() / ro.norm() < 1e-2
    ro2, _ = A.sparse_mla_ref(q.float(), ref_rows, idx[:, :2048], 0.0625)
    assert (ro2 - ro).abs().max() > 0  # the 3 kpool tail columns are attended


def test_stock_writer_layout_is_what_the_kernel_reads():
    """SGLang's own quantize_k_cache_separate (stock FP8 DSA writer, rope 0) -> 528-byte rows -> our kernel."""
    if not pathlib.Path(SGLANG_SRC, "sglang/kernels/ops/attention/dsa/quant_k_cache.py").exists():
        pytest.skip("no SGLang v0.5.20 tree")
    sys.path.insert(0, SGLANG_SRC)
    try:
        from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"stock writer import failed on CPU: {type(e).__name__}: {e}")
    n, t, h, k = 96, 2, 16, 48
    lat = (torch.randn(n, 512, generator=torch.Generator().manual_seed(7)) * 0.6).to(torch.bfloat16)
    try:
        nope, rope = quantize_k_cache_separate(lat, None)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"stock writer did not run under the interpreter: {type(e).__name__}: {e}")
    assert rope.shape[-1] == 0 and nope.shape[-1] == 528
    rows = nope.reshape(n, 528).view(torch.uint8).contiguous()
    # layout: [512, 528) holds 4 fp32 scales = amax(128-block) / 448, the stock convention our kernel multiplies by
    scales = rows[:, 512:528].contiguous().view(torch.float32)
    amax = lat.float().reshape(n, 4, 128).abs().amax(-1)
    assert torch.allclose(scales, amax / 448.0, rtol=1e-6, atol=0), (scales[0], amax[0] / 448)
    # codes decode to the latent within e4m3 rounding. Loose bound: under TRITON_INTERPRET the writer's fp32->e4m3
    # cast is not IEEE (measured 2026-09-30: lost mantissa carries), measured 5.4% here; the H200 GPU test is exact.
    deq = A.nsa656_dequant_ref(rows)
    rel = (deq - lat.float()).norm() / lat.float().norm()
    assert rel < 0.1, rel
    q = (torch.randn(t, h, 512) * 0.2).to(torch.bfloat16)
    idx = _table(t, k, n, [48, 30], seed=9)
    out, _ = A.sparse_mla(q, rows.reshape(n, 1, 528), idx, 1 / 16, A.FMT_FP8_528, block_n=16)
    ro, _ = A.sparse_mla_ref(q.float(), deq, idx, 1 / 16)
    assert (out.float() - ro).norm() / ro.norm() < 1e-2
    # and against the unquantized bf16 latent: the FP8 KV error itself (what the accuracy gate measures end to end)
    rb, _ = A.sparse_mla_ref(q.float(), lat.float(), idx, 1 / 16)
    assert (out.float() - rb).norm() / rb.norm() < 0.1


# ------------------------------------------------------------------------------------------------ engine on SM90


@pytest.fixture()
def eng(monkeypatch):
    monkeypatch.setenv("NUMINOUS_RTXFLASH_ATTN", "1")
    monkeypatch.delenv("NUMINOUS_RTXFLASH_KV", raising=False)
    monkeypatch.delenv("NUMINOUS_RTXFLASH_FP8_KERNEL", raising=False)
    import glmflash_sm120_engine as E
    return E


def test_fmt_by_arch(eng, monkeypatch):
    assert eng.fmt_for("fp8_e4m3", 9) == A.FMT_FP8_528
    assert eng.fmt_for("fp8_e4m3", 12) == A.FMT_FP8_656
    assert eng.fmt_for("bfloat16", 9) == A.FMT_BF16
    monkeypatch.setenv("NUMINOUS_RTXFLASH_KV", "fp4")
    assert eng.fmt_for("fp8_e4m3", 9) == A.FMT_FP4_288


def test_validate_sm90(eng, monkeypatch):
    impl = eng.IMPL
    assert eng.validate("Glm5NextForConditionalGeneration", 9, torch.float8_e4m3fn, impl, impl) is True
    assert eng.validate("GlmMoeDsaForCausalLM", 9, torch.float8_e4m3fn, impl, impl) is False   # GLM-5.3: stock path
    with pytest.raises(ValueError):
        eng.validate("Glm5NextForConditionalGeneration", 8, torch.float8_e4m3fn, impl, impl)
    with pytest.raises(ValueError):
        eng.validate("Glm5NextForConditionalGeneration", 9, torch.float8_e4m3fn, "fa3", "fa3")
    monkeypatch.setenv("NUMINOUS_RTXFLASH_FP8_KERNEL", "fi_jit")
    with pytest.raises(ValueError):
        eng.validate("Glm5NextForConditionalGeneration", 9, torch.float8_e4m3fn, impl, impl)


def test_kv_dim_and_pool_on_sm90(eng, monkeypatch):
    monkeypatch.setenv("NUMINOUS_RTXFLASH_SM", "9")

    class HF:
        architectures = ["Glm5NextForConditionalGeneration"]

    class MC:
        hf_config = HF()
        kv_lora_rank = 512
        qk_rope_head_dim = 0

    assert eng.kv_cache_dim(MC(), "fp8_e4m3") is None          # stock 528 row
    assert eng.kv_cache_dim(MC(), "bfloat16") is None
    sentinel = object()
    assert eng.pool_class(sentinel, 528) is sentinel           # stock pool + stock writer
    monkeypatch.setenv("NUMINOUS_RTXFLASH_SM", "12")
    assert eng.kv_cache_dim(MC(), "fp8_e4m3") == 656           # SM120 unchanged


def test_engine_forward_528_through_triton(eng):
    n, t, h = 400, 4, 16
    lat = torch.randn(n, 512, generator=torch.Generator().manual_seed(11)) * 0.4
    rows = _rows528(lat).reshape(n, 1, 528)
    e = eng.Engine(A.FMT_FP8_528, h, "triton")
    e.tiles = {"block_n": 16, "num_warps": 4, "num_stages": 1}
    q576 = torch.zeros(t, h, 512, dtype=torch.bfloat16)
    q576[:] = (torch.randn(t, h, 512) * 0.2).to(torch.bfloat16)
    idx = _table(t, 64, n, [64, 64, 10, 0], seed=5)
    out = e.forward(q576, rows, idx, 0.05)
    ro, _ = A.sparse_mla_ref(q576.float(), A.nsa656_dequant_ref(rows.reshape(n, 528)), idx, 0.05)
    assert (out[:3].float() - ro[:3]).norm() / ro[:3].norm() < 1e-2
    assert torch.all(out[3] == 0)


def test_capacity_derived():
    """Flash DSA bytes per token (11 layers + the 580 B indexer cache measured on B300): bf16 vs stock FP8 vs 656."""
    bf16 = A.bytes_per_token(A.FMT_BF16, indexer_bytes=580)
    f528 = A.bytes_per_token(A.FMT_FP8_528, indexer_bytes=580)
    f656 = A.bytes_per_token(A.FMT_FP8_656, indexer_bytes=580)
    assert (bf16, f528, f656) == (11 * 1024 + 580, 11 * 528 + 580, 11 * 656 + 580)
    assert 1.85 < bf16 / f528 < 1.86 and 1.51 < bf16 / f656 < 1.53


def test_block_h_32_matches_reference_and_env(eng, monkeypatch):
    """BLOCK_H 32 (one head block for Flash's 32 heads/rank on SM90) is exact; the env reaches the tile dict."""
    n, t, h, k = 300, 2, 32, 48
    lat = torch.randn(n, 512, generator=torch.Generator().manual_seed(21)) * 0.5
    rows = _rows528(lat)
    q = (torch.randn(t, h, 512) * 0.2).to(torch.bfloat16)
    idx = _table(t, k, n, [48, 20], seed=3)
    out, lse = A.sparse_mla(q, rows.reshape(n, 1, 528), idx, 0.06, A.FMT_FP8_528, block_n=16, block_h=32)
    ro, rl = A.sparse_mla_ref(q.float(), A.nsa656_dequant_ref(rows), idx, 0.06)
    assert (out.float() - ro).norm() / ro.norm() < 1e-2
    monkeypatch.setenv("NUMINOUS_RTXFLASH_BLOCK_H", "32")
    assert eng.tiles(A.FMT_FP8_528, 64)["block_h"] == 32
    for k_ in ("NUMINOUS_RTXFLASH_BLOCK_N", "NUMINOUS_RTXFLASH_WARPS", "NUMINOUS_RTXFLASH_STAGES"):
        monkeypatch.setenv(k_, {"NUMINOUS_RTXFLASH_BLOCK_N": "32", "NUMINOUS_RTXFLASH_WARPS": "8",
                                "NUMINOUS_RTXFLASH_STAGES": "2"}[k_])
    assert eng.tiles(A.FMT_FP8_528, 64) == {"block_n": 32, "num_warps": 8, "num_stages": 2, "block_h": 32}


# ------------------------------------------------------------------------------------------------ SM90 prefill via FA3


def _ref_attn(q, ws, table, lens, sm_scale):
    """torch stand-in for the FA3 q_v call: per-row softmax over the remapped workspace rows."""
    out = torch.zeros(q.shape[0], q.shape[1], 512)
    for i in range(q.shape[0]):
        sel = table[i, : int(lens[i])].long()
        if sel.numel() == 0:
            continue
        k = ws[sel].float()
        s = (q[i].float() @ k.T) * sm_scale
        out[i] = torch.softmax(s, dim=-1) @ k
    return out.to(torch.bfloat16)


def test_compact_slots_remap(eng):
    idx = torch.tensor([[5, 9, 5, -1], [9, 2, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32)
    slots, table = eng.compact_slots(idx, 12)
    assert slots.tolist() == [2, 5, 9]
    assert table.tolist() == [[1, 2, 1, -1], [2, 0, -1, -1], [-1, -1, -1, -1]]


@pytest.mark.parametrize("fmt", [A.FMT_FP8_528, A.FMT_FP4_288])
def test_prefill_fa3_path_equals_reference(eng, fmt):
    n, t, h, k = 500, 6, 32, 40
    lat = torch.randn(n, 512, generator=torch.Generator().manual_seed(31)) * 0.5
    if fmt == A.FMT_FP8_528:
        rows = _rows528(lat)
        ref_rows = A.nsa656_dequant_ref(rows)
    else:
        import glmflash_kv4
        rows = glmflash_kv4.quantize_rows_ref(lat)
        ref_rows = A.fp4_288_dequant_ref(rows)
    buf = rows.reshape(n, 1, rows.shape[1])
    q = (torch.randn(t, h, 512) * 0.2).to(torch.bfloat16)
    idx = _table(t, k, n, [40, 33, 7, 0, 40, 1], seed=8)
    out = eng.prefill_fa3(q, buf, idx, 0.05, fmt, attn=_ref_attn)
    ro, _ = A.sparse_mla_ref(q.float(), ref_rows, idx, 0.05)
    keep = torch.tensor([0, 1, 2, 4, 5])
    assert (out[keep].float() - ro[keep]).norm() / ro[keep].norm() < 1e-2
    assert torch.all(out[3] == 0)   # a row with no valid entry


def test_engine_routes_prefill_to_fa3_on_sm90(eng, monkeypatch):
    monkeypatch.setenv("NUMINOUS_RTXFLASH_SM", "9")
    monkeypatch.setenv("NUMINOUS_RTXFLASH_PREFILL_FA3", "1")
    monkeypatch.setenv("NUMINOUS_RTXFLASH_PREFILL_MIN_ROWS", "4")
    calls = []
    monkeypatch.setattr(eng, "_fa3_attn", lambda *a: (calls.append(a[0].shape[0]), _ref_attn(*a))[1])
    e = eng.Engine(A.FMT_FP8_528, 16, "triton")
    e.tiles = {"block_n": 16, "num_warps": 4, "num_stages": 1}
    assert e.prefill_fa3
    n = 200
    rows = _rows528(torch.randn(n, 512, generator=torch.Generator().manual_seed(2)) * 0.3).reshape(n, 1, 528)
    q8 = (torch.randn(8, 16, 512) * 0.2).to(torch.bfloat16)
    e.forward(q8, rows, _table(8, 24, n, [24] * 8, seed=4), 0.05)          # 8 rows >= 4 -> FA3 path
    q2 = (torch.randn(2, 16, 512) * 0.2).to(torch.bfloat16)
    e.forward(q2, rows, _table(2, 24, n, [24, 24], seed=5), 0.05)          # 2 rows -> Triton path
    assert calls == [8]
    monkeypatch.setenv("NUMINOUS_RTXFLASH_SM", "12")
    assert not eng.Engine(A.FMT_FP8_656, 16, "triton").prefill_fa3        # SM120 unchanged
