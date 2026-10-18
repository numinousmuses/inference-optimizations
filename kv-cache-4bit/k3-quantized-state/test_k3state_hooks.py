"""Tests for operations/patches/k3state_hooks.py and operations/kernels/k3state_engine.py.

CPU (local, needs the SGLang fork at ~/sglang-fork): every anchor resolves exactly once in the v0.5.20 sources, patching
is idempotent, patched files compile, and the engine's sizing matches the stock byte counts for Kimi K3 shapes.
Box (installed SGLang v0.5.20 + CUDA): anchors on the installed package; patch a copy of the package and import the
patched modules; build a real MambaPool with K3 per-rank shapes (inert and fp8 / fp4) and check the allocations;
drive the real fused Kimi-K3/DSpARK CuTe verify kernel through the engine's proxy against the fp32 stock run on the
same state, then the commit, and check verify outputs, working state, pool mirror and ring rows.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "kernels"))
import k3state_hooks as H  # noqa: E402

FORK = os.path.expanduser("~/sglang-fork")


def _fork_src(rel):
    return subprocess.run(["git", "-C", FORK, "show", f"v0.5.20:python/sglang/{rel}"], capture_output=True,
                          text=True, check=True).stdout


fork = pytest.mark.skipif(not os.path.exists(os.path.join(FORK, ".git")), reason="needs ~/sglang-fork")


def _installed():
    try:
        return H.sglang_root()
    except SystemExit:
        return None


box = pytest.mark.skipif(_installed() is None, reason="needs the installed SGLang v0.5.20")


@fork
def test_anchors_resolve_once_and_compile_on_v0520():
    for rel in sorted({h.path for h in H.HOOKS}):
        src = _fork_src(rel)
        hooks = [h for h in H.HOOKS if h.path == rel]
        new, lines, problems = H.patch_source(src, hooks)
        assert not problems, problems
        assert all("applied" in ln for ln in lines), lines
        compile(new, rel, "exec")
        again, lines2, problems2 = H.patch_source(new, hooks)
        assert again == new and not problems2 and all("already" in ln for ln in lines2)


@fork
def test_missing_anchor_fails_loudly(tmp_path):
    for rel in {h.path for h in H.HOOKS}:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# empty\n")
    with pytest.raises(SystemExit):
        H.apply(str(tmp_path), check=True)


def test_engine_sizing_k3(monkeypatch):
    """Per-rank K3 (TP8): 69 KDA layers, 12 heads, 128x128, conv [3, 3 x 1536] bf16."""
    import types

    import torch
    import k3state_engine as E
    import k3state_quant as kq

    cp = types.SimpleNamespace(
        shape=types.SimpleNamespace(temporal=(12, 128, 128), conv=[(3, 4608)]),
        dtype=types.SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32), layers=list(range(69)))
    conv = 3 * 4608 * 2
    monkeypatch.setenv("NUMINOUS_K3STATE", "fp8")
    assert E.slot_bytes(cp) == 69 * (conv + 12 * 128 * 132)
    assert E.request_bytes(cp) == 69 * (12 * 128 * 128 * 4 + conv)
    monkeypatch.setenv("NUMINOUS_K3STATE", "fp4")
    assert E.slot_bytes(cp) == 69 * (conv + 12 * 128 * 76)
    monkeypatch.setenv("NUMINOUS_K3STATE", "off")
    assert E.mode() is None
    monkeypatch.setenv("NUMINOUS_K3STATE", "int3")
    with pytest.raises(ValueError):
        E.mode()
    stock = 69 * (conv + 12 * 128 * 128 * 4)
    assert stock / (69 * (conv + 12 * 128 * kq.row_bytes("fp8"))) > 3.5   # 56.2 MB -> 15.9 MB per slot


# ------------------------------------------------------------------------------------------------ box
@box
def test_anchors_on_installed_package():
    lines = H.apply(_installed(), check=True)
    assert len(lines) == len(H.HOOKS), lines


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    if _installed() is None:
        pytest.skip("needs the installed SGLang v0.5.20")
    root = tmp_path_factory.mktemp("sgl")
    dst = root / "sglang"
    shutil.copytree(_installed(), dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    H.apply(str(dst))
    for m in [m for m in sys.modules if m == "sglang" or m.startswith("sglang.")]:
        del sys.modules[m]
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    import sglang
    assert str(dst) in sglang.__file__
    yield dst
    sys.path.remove(str(root))


@box
def test_patched_modules_import(patched):
    for m in ("sglang.srt.mem_cache.memory_pool", "sglang.srt.mem_cache.kv_cache_configurator",
              "sglang.srt.layers.attention.linear.kda_backend",
              "sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode",
              "sglang.srt.layers.attention.numinous_k3state"):
        importlib.import_module(m)


def _k3_cache_params():
    from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape, Mamba2StateDType
    import torch
    shape = KimiLinearStateShape.create(tp_world_size=8, num_heads=96, head_dim=128, num_k_heads=96, head_k_dim=128,
                                        conv_kernel_size=4, num_spec=0)
    return KimiLinearCacheParams(shape=shape, layers=[1, 2, 3],
                                 dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32))


@box
@pytest.mark.parametrize("fmt", ["off", "fp8", "fp4", "bf16"])
def test_mamba_pool_allocations(patched, monkeypatch, fmt):
    import torch
    from sglang.srt.mem_cache.memory_pool import MambaPool
    import sglang.srt.layers.attention.numinous_k3state_quant as kq

    monkeypatch.setenv("NUMINOUS_K3STATE", fmt)
    cp = _k3_cache_params()
    pool = MambaPool(size=10, spec_state_size=4, cache_params=cp, mamba_layer_ids=[1, 2, 3], device="cuda",
                     speculative_num_draft_tokens=8, speculative_eagle_topk=1, enable_linear_replayssm_spec=True,
                     linear_replayssm_cache_len=16)
    t = pool.mamba_cache.temporal
    ring = pool.mamba_cache.replayssm_rawv
    if fmt == "off":
        assert t.dtype == torch.float32 and t.shape == (3, 11, 12, 128, 128)
        assert ring.shape[1] == 11 and not hasattr(pool, "k3s_w")
    else:
        assert t.dtype == torch.uint8 and t.shape == (3, 11, 12, 128, kq.row_bytes(fmt))
        assert ring.shape[1] == 6 and pool.k3s_w.shape == (3, 6, 12, 128, 128)
        assert pool.k3s_cw[0].shape == (3, 6) + tuple(pool.mamba_cache.conv[0].shape[2:])
        # engine slot ops stay byte-exact on the quantized pool
        t[:, 3] = 7
        pool.copy_from(torch.tensor([3], device="cuda"), torch.tensor([5], device="cuda"))
        assert torch.equal(t[:, 5], t[:, 3])
        pool.clear_slots(torch.tensor([5], device="cuda"))
        assert not t[:, 5].any()


# ------------------------------------------------------------------------------------------------ engine paths (box)
HV, KD, NTOK = 12, 128, 8           # K3 per rank: 12 heads of 128; DSpark block 7 -> 8 verify tokens per request
DIM = 3 * HV * KD


def _rows_match(got, ref, fmt):
    from test_k3state_quant import _assert_rows_match   # operations/kernels is on sys.path
    _assert_rows_match(got, ref, fmt)


def _close_fp32(got, ref):
    """fp32 working state vs the stock fp32 state: the same op order, equal up to a few fp32 ULPs of the state scale
    (the fp4 fold compiles its multiply-adds next to the quantizing store and may contract one differently)."""
    tol = 4 * 2 ** -23 * max(ref.abs().max().item(), 1.0)
    assert (got - ref).abs().max().item() <= tol, (got - ref).abs().max().item()


def _layer(layer_id=1):
    import types

    import torch
    g = torch.Generator(device="cuda").manual_seed(layer_id)
    return types.SimpleNamespace(
        layer_id=layer_id, num_q_heads=HV, num_k_heads=HV, num_v_heads=HV, head_q_dim=KD, head_k_dim=KD,
        head_v_dim=KD, q_dim=HV * KD, k_dim=HV * KD, v_dim=HV * KD, bias=None, lower_bound=-5.0,
        conv_weights=torch.randn(DIM, 4, generator=g, device="cuda") * 0.3,
        A_log=torch.randn(HV, generator=g, device="cuda").reshape(1, 1, HV, 1) * 0.5,
        dt_bias=torch.randn(HV * KD, generator=g, device="cuda") * 0.5)


def _pool(fmt, monkeypatch, size=10, spec=6):
    from sglang.srt.mem_cache.memory_pool import MambaPool
    monkeypatch.setenv("NUMINOUS_K3STATE", fmt)
    return MambaPool(size=size, spec_state_size=spec, cache_params=_k3_cache_params(), mamba_layer_ids=[1, 2, 3],
                     device="cuda", speculative_num_draft_tokens=NTOK, speculative_eagle_topk=1,
                     enable_linear_replayssm_spec=True, linear_replayssm_cache_len=16)


class _ReqPool:
    def __init__(self, pool):
        self.mamba_pool = pool
        self.mamba_map = {1: 0, 2: 1, 3: 2}
        self.size = 6

    def mamba2_layer_index(self, layer_id):
        return self.mamba_map[layer_id]

    def mamba2_layer_cache(self, layer_id):
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])


def _backend(pool, slots, qsl):
    import types

    import torch
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend, KDAKernelDispatcher
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend as LB
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    be = object.__new__(KDAAttnBackend)
    be.req_to_token_pool = _ReqPool(pool)
    be.kernel_dispatcher = KDAKernelDispatcher(LB.TRITON, LB.TRITON, LB.NV_CUTEDSL)
    be._fused_chain_verify_fn = None
    be.accept_lens_pool = None
    be.verify_intermediate_state_indices = torch.arange(8, dtype=torch.int32, device="cuda")
    be.forward_metadata = ForwardMetadata(query_start_loc=qsl, mamba_cache_indices=slots)
    be.forward_metadata.has_mamba_track_mask = False
    return be


def _fill_state(pool_fp32, pool_q, slots, rows_seed=0):
    """Same committed state in the fp32 pool (stock) and, quantization-free, in the two-tier W + its pool mirror."""
    import torch
    g = torch.Generator(device="cuda").manual_seed(rows_seed)
    t = pool_fp32.mamba_cache.temporal
    for s in slots.tolist():
        if s >= 0:
            t[:, s] = torch.randn(t[:, s].shape, generator=g, device="cuda") * 0.05
    conv = pool_fp32.mamba_cache.conv[0]
    conv.copy_(torch.randn(conv.shape, generator=g, device="cuda").to(conv.dtype))
    pool_q.mamba_cache.conv[0].copy_(conv)


@box
@pytest.mark.parametrize("fmt", ["fp8", "fp4", "bf16"])
def test_verify_and_commit_match_stock(patched, monkeypatch, fmt):
    """The real fused CuTe verify through the engine proxy (W + conv copy, request rows) equals the stock run on the
    fp32 slot state; the commit leaves W bit-identical to the stock fp32 slot state and mirrors it into the pool."""
    import types

    import torch
    from sglang.kernels.ops.attention.fla import kda_replayssm_spec_decode as R
    import sglang.srt.layers.attention.numinous_k3state as E
    import sglang.srt.layers.attention.numinous_k3state_quant as kq

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the Kimi-K3/DSpARK CuTe verify kernel is SM100-family only")
    bs = 3
    slots = torch.tensor([7, 2, 9], dtype=torch.int32, device="cuda")
    req_rows = torch.tensor([4, 1, 5], dtype=torch.int64, device="cuda")
    qsl = torch.arange(0, (bs + 1) * NTOK, NTOK, dtype=torch.int32, device="cuda")
    ref_pool = _pool("off", monkeypatch)
    q_pool = _pool(fmt, monkeypatch)
    _fill_state(ref_pool, q_pool, slots)
    # the two-tier working rows hold the same exact state (as after a prefill); mirror into the quantized pool
    for li in range(3):
        kq.quant_rows(ref_pool.mamba_cache.temporal[li], slots, q_pool.mamba_cache.temporal[li], slots, fmt)
        q_pool.k3s_w[li][req_rows] = ref_pool.mamba_cache.temporal[li][slots.long()]
    g = torch.Generator(device="cuda").manual_seed(3)
    fb = types.SimpleNamespace(spec_info=types.SimpleNamespace(draft_token_num=NTOK, ragged_verify_layout=None),
                               req_pool_indices=req_rows,
                               forward_mode=types.SimpleNamespace(is_target_verify=lambda: True,
                                                                  is_draft_extend_v2=lambda: False))
    outs = {}
    for name, pool in (("stock", ref_pool), ("k3s", q_pool)):
        monkeypatch.setenv("NUMINOUS_K3STATE", "off" if name == "stock" else fmt)
        be = _backend(pool, slots, qsl)
        outs[name] = []
        for lid in (1, 2, 3):
            layer = _layer(lid)
            gg = torch.Generator(device="cuda").manual_seed(100 + lid)
            mixed = (torch.randn(bs * NTOK, DIM, generator=gg, device="cuda")).to(torch.bfloat16)
            a = torch.randn(1, bs * NTOK, HV, KD, generator=gg, device="cuda").to(torch.bfloat16)
            b = torch.randn(1, bs * NTOK, HV, generator=gg, device="cuda").to(torch.bfloat16)
            outs[name].append(be.forward_extend(layer, fb, mixed, a, b))
        # commit: request 0 accepts 3, request 1 accepts 1, request 2 accepts 8
        accept = torch.tensor([3, 1, 8], dtype=torch.int32, device="cuda")
        R.commit_kda_replayssm_after_verify(
            spec_state=pool.mamba_cache, state_batch_indices=slots, accept_lens=accept,
            last_correct_step_indices=accept - 1, null_block_id=-1)
    for o_ref, o in zip(outs["stock"], outs["k3s"]):
        assert torch.equal(o_ref, o), (o_ref.float() - o.float()).abs().max()
    for li in range(3):
        ref_state = ref_pool.mamba_cache.temporal[li][slots.long()]
        w = q_pool.k3s_w[li][req_rows]
        _close_fp32(w, ref_state)
        _rows_match(q_pool.mamba_cache.temporal[li][slots.long()].cpu(), kq.ref_quant(w.cpu(), fmt), fmt)
    assert torch.equal(ref_pool.mamba_cache.conv[0], q_pool.mamba_cache.conv[0])     # conv rollback identical
    assert E.mode() == fmt


@box
@pytest.mark.parametrize("fmt", ["fp8", "fp4", "bf16"])
def test_extend_matches_stock(patched, monkeypatch, fmt):
    """Prefill through the engine: fresh requests (zero state) give bit-identical outputs and final states; a request
    resuming from the pool equals stock run on the dequantized initial state."""
    import types

    import torch
    import sglang.srt.layers.attention.numinous_k3state_quant as kq

    slots = torch.tensor([3, 8], dtype=torch.int32, device="cuda")
    req_rows = torch.tensor([2, 5], dtype=torch.int64, device="cuda")
    lens = [37, 70]
    qsl = torch.tensor([0, 37, 107], dtype=torch.int32, device="cuda")
    for resume in (False, True):
        ref_pool = _pool("off", monkeypatch)
        q_pool = _pool(fmt, monkeypatch)
        if resume:
            g = torch.Generator(device="cuda").manual_seed(9)
            init = torch.randn(3, 2, HV, KD, KD, generator=g, device="cuda") * 0.05
            for li in range(3):
                kq.quant_rows(init[li], None, q_pool.mamba_cache.temporal[li], slots, fmt)
                ref_pool.mamba_cache.temporal[li][slots.long()] = kq.ref_dequant(
                    q_pool.mamba_cache.temporal[li][slots.long()].cpu(), fmt).cuda()
        fb = types.SimpleNamespace(
            req_pool_indices=req_rows, extend_prefix_lens=torch.tensor([5 if resume else 0] * 2, device="cuda"),
            extend_seq_lens_cpu=lens,
            forward_mode=types.SimpleNamespace(is_target_verify=lambda: False, is_draft_extend_v2=lambda: False))
        res = {}
        for name, pool in (("stock", ref_pool), ("k3s", q_pool)):
            monkeypatch.setenv("NUMINOUS_K3STATE", "off" if name == "stock" else fmt)
            be = _backend(pool, slots, qsl)
            res[name] = []
            for lid in (1, 2, 3):
                layer = _layer(lid)
                gg = torch.Generator(device="cuda").manual_seed(200 + lid)
                mixed = torch.randn(sum(lens), DIM, generator=gg, device="cuda").to(torch.bfloat16)
                a = torch.randn(1, sum(lens), HV * KD, generator=gg, device="cuda").to(torch.bfloat16)
                b = torch.randn(1, sum(lens), HV, generator=gg, device="cuda").to(torch.bfloat16)
                res[name].append(be.forward_extend(layer, fb, mixed, a, b))
        for o_ref, o in zip(res["stock"], res["k3s"]):
            assert torch.equal(o_ref, o), (resume, (o_ref.float() - o.float()).abs().max())
        for li in range(3):
            ref_state = ref_pool.mamba_cache.temporal[li][slots.long()]
            w = q_pool.k3s_w[li][req_rows]
            _close_fp32(w, ref_state)
            _rows_match(q_pool.mamba_cache.temporal[li][slots.long()].cpu(), kq.ref_quant(w.cpu(), fmt), fmt)
        assert torch.equal(ref_pool.mamba_cache.conv[0], q_pool.mamba_cache.conv[0])


@box
@pytest.mark.parametrize("fmt", ["fp8", "fp4", "bf16"])
def test_decode_matches_stock(patched, monkeypatch, fmt):
    """Non-speculative decode through the engine proxy (conv copy + W by request row, then conv and the quantized
    state written back to the pool slots) equals the stock decode on the fp32 pool."""
    import types

    import torch
    import sglang.srt.layers.attention.numinous_k3state_quant as kq

    bs = 3
    slots = torch.tensor([6, 1, 9], dtype=torch.int32, device="cuda")
    req_rows = torch.tensor([2, 5, 3], dtype=torch.int64, device="cuda")
    qsl = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
    ref_pool = _pool("off", monkeypatch)
    q_pool = _pool(fmt, monkeypatch)
    _fill_state(ref_pool, q_pool, slots)
    for li in range(3):
        kq.quant_rows(ref_pool.mamba_cache.temporal[li], slots, q_pool.mamba_cache.temporal[li], slots, fmt)
        # the working rows hold the exact state (as after the prefill / last commit)
        q_pool.k3s_w[li][req_rows] = ref_pool.mamba_cache.temporal[li][slots.long()]
    fb = types.SimpleNamespace(req_pool_indices=req_rows, mamba_track_mask=None,
                               forward_mode=types.SimpleNamespace(is_target_verify=lambda: False,
                                                                  is_draft_extend_v2=lambda: False))
    outs = {}
    for name, pool in (("stock", ref_pool), ("k3s", q_pool)):
        monkeypatch.setenv("NUMINOUS_K3STATE", "off" if name == "stock" else fmt)
        be = _backend(pool, slots, qsl)
        outs[name] = []
        for lid in (1, 2, 3):
            layer = _layer(lid)
            gg = torch.Generator(device="cuda").manual_seed(300 + lid)
            mixed = torch.randn(bs, DIM, generator=gg, device="cuda").to(torch.bfloat16)
            a = torch.randn(1, bs, HV, KD, generator=gg, device="cuda").to(torch.bfloat16)
            b = torch.randn(1, bs, HV, generator=gg, device="cuda").to(torch.bfloat16)
            outs[name].append(be.forward_decode(layer, fb, mixed, a, b))
    for o_ref, o in zip(outs["stock"], outs["k3s"]):
        assert torch.equal(o_ref, o), (o_ref.float() - o.float()).abs().max()
    for li in range(3):
        ref_state = ref_pool.mamba_cache.temporal[li][slots.long()]
        w = q_pool.k3s_w[li][req_rows]
        _close_fp32(w, ref_state)
        _rows_match(q_pool.mamba_cache.temporal[li][slots.long()].cpu(), kq.ref_quant(w.cpu(), fmt), fmt)
    assert torch.equal(ref_pool.mamba_cache.conv[0], q_pool.mamba_cache.conv[0])
