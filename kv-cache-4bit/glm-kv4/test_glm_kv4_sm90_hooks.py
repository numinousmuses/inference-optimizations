"""glm_kv4_sm90_hooks + glm_kv4_sm90 (GLM-5.3 4-bit KV on Hopper), on CPU.

1. The eight anchors resolve once each on the v0.5.20 tree (SGLANG_SRC, default ~/.cache/numinous/sgl-v0520/python).
   Applying them to a copy compiles and is idempotent. Each hook lands where it must:
   - the 352-B dim before the 656-B scaled-FP8 return;
   - the route before flashmla_kv's metadata assert;
   - the batch notes before the decode/extend dispatch.
2. The read path's glue, with the Triton/FlashMLA kernels replaced by torch references (the kernels themselves are
   checked bit-exact on an H200 by operations/kernels/hopper_kv4_check.py). forward_flashmla_kv over an FP4 pool
   equals the fp32 sparse MLA reference over the same pool, for:
   - plain decode;
   - verify with dedup;
   - chunked eager calls;
   - the prefill sequence path (slot -> row remap);
   - -1 rows and an empty call.
3. attach(): refuses every backend that would read the 352-B row as its own layout. It turns MHA one-shot off.

    python3 -m pytest -q operations/patches/test_glm_kv4_sm90_hooks.py
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import glm_kv4_sm90_hooks as hooks  # noqa: E402

SRC = os.path.join(os.environ.get("SGLANG_SRC", os.path.expanduser("~/.cache/numinous/sgl-v0520/python")), "sglang")
have_tree = pytest.mark.skipif(not os.path.isdir(SRC), reason=f"no v0.5.20 tree at {SRC}")


# ------------------------------------------------------------------------------------------------ 1. anchors


@have_tree
def test_anchors_resolve_once_on_v0520():
    lines = hooks.apply(SRC, check=True)
    assert len(lines) == len(hooks.HOOKS), lines
    assert all("anchor ok in" in ln for ln in lines), lines


@pytest.fixture
def patched(tmp_path):
    root = tmp_path / "sglang"
    for rel in (hooks.CONFIGURATOR, hooks.BACKEND, hooks.MEMORY_POOL):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(SRC, rel), root / rel)
    return root


@have_tree
def test_apply_compiles_idempotent_and_places_hooks(patched):
    first = hooks.apply(str(patched), install_modules=False)
    assert sum("applied to" in ln for ln in first) == len(hooks.HOOKS), first
    second = hooks.apply(str(patched), install_modules=False)
    assert all(ln.endswith("already applied") for ln in second), second
    for rel in (hooks.CONFIGURATOR, hooks.BACKEND, hooks.MEMORY_POOL):
        compile((patched / rel).read_text(), rel, "exec")
    mp = (patched / hooks.MEMORY_POOL).read_text()
    hy = mp[mp.index("class HybridLinearKVPool("):]
    # inside the use_dsa branch, right after the DSATokenToKVPool construction, before the plain-MLA branch
    assert hy.index("self.full_kv_pool = DSATokenToKVPool(") < hy.index("numinous-hook:kv4h_hybrid_pool")
    assert hy.index("numinous-hook:kv4h_hybrid_pool") < hy.index("TokenToKVPoolClass = MLATokenToKVPool")
    conf = (patched / hooks.CONFIGURATOR).read_text()
    fn = conf[conf.index("def calculate_mla_kv_cache_dim("):]
    # after the trtllm return (SM90 never takes it), before the 656-byte scaled-FP8 dim
    assert fn.index("if uses_trtllm_kv_layout:") < fn.index("numinous-hook:kv4h_dim")
    assert fn.index("numinous-hook:kv4h_dim") < fn.index("quant_block_size = DSATokenToKVPool.quant_block_size")
    be = (patched / hooks.BACKEND).read_text()
    fk = be[be.index("    def _forward_flashmla_kv("):]
    assert fk.index("numinous-hook:kv4h_route") < fk.index("assert metadata.flashmla_metadata is not None")
    fd = be[be.index("    def forward_decode("):]
    assert fd.index("numinous-hook:kv4h_fb_dec") < fd.index('elif dsa_impl == "flashmla_kv":')
    fe = be[be.index("    def forward_extend("):]
    assert fe.index("numinous-hook:kv4h_fb_ext") < fe.index('if dsa_impl == "trtllm" and not self.use_mha:')
    init = be[be.index("self.kv_cache_dtype = model_runner.kv_cache_dtype"):]
    assert init.index("numinous-hook:kv4h_init") < init.index("def ")
    # attach() reads attributes __init__ sets before its anchor
    pre = be[: be.index("numinous-hook:kv4h_init")]
    for attr in ("self.dsa_prefill_impl", "self.dsa_decode_impl", "self.device_sm_major =",
                 "self.hisparse_coordinator =", "self.dsa_index_kpool =", "self.supports_mha_one_shot"):
        assert attr in pre, attr


def test_hook_set_matches_module_list():
    names = [h.name for h in hooks.HOOKS]
    assert len(names) == len(set(names)) == 8
    assert set(hooks.MODULES) == {"operations/kernels/glm_kv4.py", "operations/kernels/glm_kv4_engine.py",
                                  "operations/kernels/glm_kv4_sm90.py"}
    for local in hooks.MODULES:
        assert os.path.exists(os.path.join(REPO, local)), local


# ------------------------------------------------------------------------------------------------ 2. read-path glue


def _load_modules():
    """glm_kv4 (real), glm_kv4_engine and glm_kv4_sm90 under the package names the hooks install them as, with a
    stub DSATokenToKVPool (the engine only subclasses it)."""
    for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.attention",
                 "sglang.srt.layers.attention.dsa", "sglang.srt.mem_cache"):
        sys.modules.setdefault(name, types.ModuleType(name))
    mp = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    mp.DSATokenToKVPool = type("DSATokenToKVPool", (), {})
    sys.modules["sglang.srt.mem_cache.memory_pool"] = mp
    pkg = sys.modules["sglang.srt.layers.attention.dsa"]
    mods = {}
    for short in ("glm_kv4", "glm_kv4_engine", "glm_kv4_sm90"):
        full = f"sglang.srt.layers.attention.dsa.{short}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(REPO, "operations/kernels", short + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        setattr(pkg, short, m)
        spec.loader.exec_module(m)
        mods[short] = m
    return mods["glm_kv4"], mods["glm_kv4_engine"], mods["glm_kv4_sm90"]


@pytest.fixture
def mods(monkeypatch):
    kv4, eng, h = _load_modules()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    calls = {"rows": 0, "dedup": 0, "seq": 0}

    def deq(kv_buffer, slots):
        rows = kv4.pool_rows(kv_buffer)
        return kv4.dequantize_rows_ref(rows[slots.long()], torch.bfloat16)

    def dequant_rows(kv_buffer, idx, dst, **_):
        calls["rows"] += 1
        idx = idx.reshape(-1)
        ok = idx >= 0
        assert dst.dtype == torch.bfloat16 and dst.shape[0] >= idx.numel()
        dst[: idx.numel()][ok] = deq(kv_buffer, idx[ok])

    def identity_table(table, out):
        n = table.numel()
        flat = torch.arange(n, dtype=torch.int32).view(table.shape)
        out.copy_(torch.where(table >= 0, flat, torch.full_like(flat, -1)))

    def dequant_rows_dedup(kv_buffer, table, stamp, dst, out, work=None, **_):
        calls["dedup"] += 1
        flat = table.reshape(-1)
        first = {}
        for i, s in enumerate(flat.tolist()):
            if s >= 0 and s not in first:
                first[s] = len(first)
        assert len(first) <= dst.shape[0]
        if first:
            slots = torch.tensor(list(first), dtype=torch.int64)
            dst[: len(first)] = deq(kv_buffer, slots)
        out.view(-1).copy_(torch.tensor([first.get(s, -1) if s >= 0 else -1 for s in flat.tolist()],
                                        dtype=torch.int32))
        return torch.tensor([len(first)], dtype=torch.int32)

    def dequant_seq(kv_buffer, page_table, seq_lens, bases, max_len, dst, slot_to_pos, **_):
        calls["seq"] += 1
        assert dst.dtype == torch.bfloat16
        for b in range(page_table.shape[0]):
            n = min(int(seq_lens[b]), max_len)
            for p in range(n):
                s = int(page_table[b, p])
                if s < 0:
                    continue
                r = int(bases[b]) + p
                dst[r] = deq(kv_buffer, torch.tensor([s]))[0]
                slot_to_pos[s] = r

    def remap_table(table, slot_to_pos, n_pool, out):
        ok = (table >= 0) & (table < n_pool)
        out.copy_(torch.where(ok, slot_to_pos[table.clamp(min=0).long()], torch.full_like(table, -1)))

    for name, fn in (("dequant_rows", dequant_rows), ("identity_table", identity_table),
                     ("dequant_rows_dedup", dequant_rows_dedup), ("dequant_seq", dequant_seq),
                     ("remap_table", remap_table)):
        monkeypatch.setattr(kv4, name, fn)
    return kv4, eng, h, calls


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_extend_or_draft_extend_or_mixed(self):
        return self.name in ("EXTEND", "MIXED", "DRAFT_EXTEND", "DRAFT_EXTEND_V2")

    def is_mixed(self):
        return self.name == "MIXED"

    def is_target_verify(self):
        return self.name == "TARGET_VERIFY"

    def is_decode_or_idle(self):
        return self.name in ("DECODE", "IDLE")


def _sparse_bf16_ref(q_all, kv_cache, v_head_dim, page_table_1, sm_scale, topk_length=None):
    """What flash_mla_sparse_fwd computes: softmax(q k^T * scale) v over the indexed scratch rows, fp32 math."""
    kv = kv_cache.reshape(kv_cache.shape[0], -1).float()
    t, h, _ = q_all.shape
    out = torch.zeros(t, h, v_head_dim)
    for i in range(t):
        idx = page_table_1[i].long()
        if topk_length is not None:
            idx = idx[: int(topk_length[i])]
        idx = idx[idx >= 0]
        if idx.numel() == 0:
            continue
        k = kv[idx]
        p = torch.softmax(q_all[i].float() @ k.T * sm_scale, dim=-1)
        out[i] = p @ k[:, :v_head_dim]
    return out.to(torch.bfloat16)


def _setup(kv4, eng, h, n_slots=600, heads=4, sel_tokens=256, dedup=True, monkeypatch=None):
    torch.manual_seed(0)
    monkeypatch.setenv("NUMINOUS_GLM_KV4_DEDUP", "1" if dedup else "0")
    monkeypatch.setenv("NUMINOUS_GLM_KV4_SEL_ROWS", str(sel_tokens))
    monkeypatch.setenv("NUMINOUS_GLM_KV4_ATTN", "scratch")
    nope = (torch.randn(n_slots, 512) * 0.5).bfloat16()
    rope = (torch.randn(n_slots, 64) * 0.5).bfloat16()
    rows = kv4.quantize_rows_ref(nope, rope)  # [n, 352]
    pool = types.SimpleNamespace(kv_buffer=[rows.view(n_slots, 1, kv4.ROW_BYTES)], start_layer=0)
    be = types.SimpleNamespace(token_to_kv_pool=pool, device="cpu", dsa_index_topk=64, num_q_heads=heads)
    be._forward_flashmla_sparse = _sparse_bf16_ref
    layer = types.SimpleNamespace(layer_id=0, tp_q_head_num=heads, head_dim=576)
    return rows, be, layer


@pytest.mark.parametrize("t,k,sel,dedup,mode", [
    (5, 64, 256, False, "DECODE"),        # plain decode: one call, gather in table order
    (9, 64, 2, False, "DECODE"),          # eager chunking: 2 tokens per scratch fill
    (96, 64, 256, True, "TARGET_VERIFY"),  # verify: dedup of rows shared by draft tokens
    (70, 64, 16, True, "DRAFT_EXTEND"),   # no batch tables -> selected rows, dedup, chunked
])
def test_selected_rows_equal_reference(mods, monkeypatch, t, k, sel, dedup, mode):
    kv4, eng, h, calls = mods
    rows, be, layer = _setup(kv4, eng, h, sel_tokens=sel, dedup=dedup, monkeypatch=monkeypatch)
    table = torch.randint(0, rows.shape[0], (t, k), dtype=torch.int32)
    if dedup:
        table[1::2] = table[0::2][: table[1::2].shape[0]]
    table[0, 10:] = -1
    if t > 3:
        table[3] = -1  # a row with no valid slot -> zeros
    q = (torch.randn(t, layer.tp_q_head_num, 576) * 0.3).bfloat16()
    be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode(mode), seq_lens_cpu=None, batch_size=t)
    meta = types.SimpleNamespace(dsa_cache_seqlens_int32=None, page_table_1=None, cache_seqlens_int32=None)
    out = h.forward_flashmla_kv(be, q, table, 576 ** -0.5, 512, layer, meta)
    ref = kv4.sparse_mla_ref(q, rows, table, 576 ** -0.5)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2), (out.float() - ref).abs().max()
    if t > 3:
        assert out[3].abs().max() == 0
    assert (calls["dedup"] > 0) == (dedup and t >= h.DEDUP_MIN_ROWS), calls
    assert calls["seq"] == 0


def test_sequence_prefill_remaps_through_slot_map(mods, monkeypatch):
    kv4, eng, h, calls = mods
    rows, be, layer = _setup(kv4, eng, h, monkeypatch=monkeypatch)
    # two requests with scattered pool slots; every token's top-k picks positions of its own request
    lens = [40, 25]
    perm = torch.randperm(rows.shape[0])
    pt = torch.full((2, 48), -1, dtype=torch.int32)
    pt[0, :40] = perm[:40].int()
    pt[1, :25] = perm[100:125].int()
    t, k = 12, 64
    table = torch.full((t, k), -1, dtype=torch.int32)
    for i in range(t):
        b = 0 if i < 7 else 1
        pos = torch.randperm(lens[b])[: min(k, lens[b])]
        table[i, : pos.numel()] = pt[b, pos]
    q = (torch.randn(t, layer.tp_q_head_num, 576) * 0.3).bfloat16()
    be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode("EXTEND"), seq_lens_cpu=torch.tensor(lens),
                                                batch_size=2)
    meta = types.SimpleNamespace(dsa_cache_seqlens_int32=None, page_table_1=pt,
                                 cache_seqlens_int32=torch.tensor(lens, dtype=torch.int32))
    out = h.forward_flashmla_kv(be, q, table, 576 ** -0.5, 512, layer, meta)
    ref = kv4.sparse_mla_ref(q, rows, table, 576 ** -0.5)
    assert calls["seq"] == 1 and calls["rows"] == 0 and calls["dedup"] == 0, calls
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2), (out.float() - ref).abs().max()


def test_sequence_prefill_falls_back_without_tables(mods, monkeypatch):
    kv4, eng, h, calls = mods
    rows, be, layer = _setup(kv4, eng, h, monkeypatch=monkeypatch, dedup=False)
    t, k = 4, 64
    table = torch.randint(0, rows.shape[0], (t, k), dtype=torch.int32)
    q = (torch.randn(t, layer.tp_q_head_num, 576) * 0.3).bfloat16()
    be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode("EXTEND"), seq_lens_cpu=None, batch_size=1)
    meta = types.SimpleNamespace(dsa_cache_seqlens_int32=None, page_table_1=None, cache_seqlens_int32=None)
    out = h.forward_flashmla_kv(be, q, table, 576 ** -0.5, 512, layer, meta)
    assert calls["seq"] == 0 and calls["rows"] == 1
    assert torch.allclose(out.float(), kv4.sparse_mla_ref(q, rows, table, 576 ** -0.5), atol=2e-2, rtol=2e-2)


def test_empty_call_and_topk_length_slicing(mods, monkeypatch):
    kv4, eng, h, calls = mods
    rows, be, layer = _setup(kv4, eng, h, sel_tokens=2, monkeypatch=monkeypatch, dedup=False)
    be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode("DECODE"), seq_lens_cpu=None, batch_size=0)
    meta = types.SimpleNamespace(dsa_cache_seqlens_int32=None, page_table_1=None, cache_seqlens_int32=None)
    out = h.forward_flashmla_kv(be, torch.empty(0, 4, 576, dtype=torch.bfloat16), torch.empty(0, 64, dtype=torch.int32),
                                576 ** -0.5, 512, layer, meta)
    assert out.shape == (0, 4, 512)
    # topk_length (valid prefix per row) is sliced with the token chunks
    t, k = 5, 64
    table = torch.randint(0, rows.shape[0], (t, k), dtype=torch.int32)
    lengths = torch.tensor([64, 10, 64, 1, 30], dtype=torch.int32)
    for i, n in enumerate(lengths.tolist()):
        table[i, n:] = -1
    q = (torch.randn(t, 4, 576) * 0.3).bfloat16()
    meta.dsa_cache_seqlens_int32 = lengths
    out = h.forward_flashmla_kv(be, q, table, 576 ** -0.5, 512, layer, meta)
    assert torch.allclose(out.float(), kv4.sparse_mla_ref(q, rows, table, 576 ** -0.5), atol=2e-2, rtol=2e-2)


# ------------------------------------------------------------------------------------------------ 3. attach


def _attach_backend(h, **kw):
    pool = types.SimpleNamespace(numinous_kv4=True, kv_buffer=[torch.zeros(8, 1, 352, dtype=torch.uint8)])
    be = types.SimpleNamespace(device_sm_major=9, dsa_prefill_impl="flashmla_kv", dsa_decode_impl="flashmla_kv",
                               kv_cache_dtype=torch.float8_e4m3fn, hisparse_coordinator=None, dsa_index_kpool=1,
                               supports_mha_one_shot=True, token_to_kv_pool=pool, device="cpu", num_q_heads=64,
                               dsa_index_topk=2048)
    for k_, v in kw.items():
        setattr(be, k_, v)
    return be, types.SimpleNamespace(token_to_kv_pool=pool)


def test_attach_accepts_sm90_flashmla_kv_and_turns_mha_off(mods, monkeypatch):
    kv4, eng, h, _ = mods
    monkeypatch.setattr(torch.version, "hip", None)
    be, mr = _attach_backend(h)
    assert h.attach(be, mr) is True
    assert be.supports_mha_one_shot is False
    assert isinstance(be._numinous_kv4_state, h._State90)


@pytest.mark.parametrize("kw,needle", [
    ({"device_sm_major": 10}, "need SM90"),
    ({"dsa_prefill_impl": "flashmla_sparse"}, "need flashmla_kv/flashmla_kv"),
    ({"dsa_decode_impl": "fa3"}, "need flashmla_kv/flashmla_kv"),
    ({"kv_cache_dtype": torch.bfloat16}, "need fp8_e4m3"),
    ({"hisparse_coordinator": object()}, "hisparse"),
    ({"dsa_index_kpool": 4}, "kpool"),
])
def test_attach_refuses_backends_that_misread_the_row(mods, monkeypatch, kw, needle):
    kv4, eng, h, _ = mods
    monkeypatch.setattr(torch.version, "hip", None)
    be, mr = _attach_backend(h, **kw)
    with pytest.raises(RuntimeError, match=needle):
        h.attach(be, mr)


def test_attach_without_fp4_pool_is_inert(mods, monkeypatch):
    kv4, eng, h, _ = mods
    be, mr = _attach_backend(h)
    mr.token_to_kv_pool = types.SimpleNamespace(kv_buffer=[torch.zeros(8, 1, 656, dtype=torch.uint8)])
    assert h.attach(be, mr) is False
    assert be.supports_mha_one_shot is True


def test_bf16_scratch_state(mods, monkeypatch):
    kv4, eng, h, _ = mods
    monkeypatch.setenv("NUMINOUS_GLM_KV4_ATTN", "scratch")
    be, mr = _attach_backend(h)
    st = h._state(be)
    st.reserve_graph(3, 64)
    assert st.graph_scratch.dtype == torch.bfloat16 and st.graph_scratch.shape[1] == 576
    assert st.graph_scratch.shape[0] >= 3 * 64 and st.graph_scratch.shape[0] % 64 == 0
    sc, tab = st.eager(100, 2, 64)
    assert sc.dtype == torch.bfloat16 and sc.shape[0] >= 100 and tab.shape == (2, 64)


# ------------------------------------------------------------------------------------------------ 4. v32 decode scratch


def _v32_rows_ref(kv4, rows4: torch.Tensor) -> torch.Tensor:
    """FP4 pool rows -> FlashMLA V32 rows (656 B) with quant_k_cache's convention: per 128, scale = amax / 448,
    q = clamp(x * (1 / scale)) -> E4M3; four fp32 scales; the E4M3 rope as bf16."""
    v = kv4.dequantize_rows_ref(rows4, torch.float32)  # exact FP4 x E4M3 values + rope
    n = v.shape[0]
    tiles = v[:, :512].view(n, 4, 128)
    s = tiles.abs().amax(dim=-1) / 448.0
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    q = (tiles * inv[:, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(n, 512)
    out = torch.empty(n, 656, dtype=torch.uint8)
    out[:, :512] = q.view(torch.uint8)
    out[:, 512:528] = s.contiguous().view(torch.uint8).view(n, 16)
    out[:, 528:] = v[:, 512:].to(torch.bfloat16).contiguous().view(torch.uint8).view(n, 128)
    return out


def _v32_decode_ref(scratch: torch.Tensor, q: torch.Tensor, table: torch.Tensor, sm_scale: float) -> torch.Tensor:
    n = scratch.shape[0]
    q8 = scratch[:, :512].contiguous().view(torch.float8_e4m3fn).float().view(n, 4, 128)
    s = scratch[:, 512:528].contiguous().view(torch.float32).view(n, 4)
    nope = (q8 * s[:, :, None]).view(n, 512)
    rope = scratch[:, 528:].contiguous().view(torch.bfloat16).float()
    kv = torch.cat([nope, rope], dim=-1)
    out = torch.zeros(q.shape[0], q.shape[1], 512)
    for i in range(q.shape[0]):
        idx = table[i].long()
        idx = idx[idx >= 0]
        if idx.numel() == 0:
            continue
        k = kv[idx]
        p = torch.softmax(q[i].float() @ k.T * sm_scale, dim=-1)
        out[i] = p @ k[:, :512]
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("t,sel", [(6, 256), (9, 2)])
def test_v32_decode_route_matches_fp4_reference(mods, monkeypatch, t, sel):
    kv4, eng, h, calls = mods
    monkeypatch.setenv("NUMINOUS_GLM_KV4_SM90_DECODE", "v32")
    rows, be, layer = _setup(kv4, eng, h, sel_tokens=sel, dedup=False, monkeypatch=monkeypatch)
    seen = {"meta": [], "conv": 0}

    def kv4_to_v32(kv_buffer, idx, dst, **_):
        seen["conv"] += 1
        idx = idx.reshape(-1)
        ok = idx >= 0
        dst[: idx.numel()][ok] = _v32_rows_ref(kv4, kv4.pool_rows(kv_buffer)[idx[ok].long()])

    def fp8_decode(backend, q_rows, scratch, table, sm_scale, v_head_dim, cache_seqlens, meta):
        seen["meta"].append(meta)
        assert scratch.dtype == torch.uint8 and scratch.shape[1] == 656 and cache_seqlens.shape[0] == q_rows.shape[0]
        return _v32_decode_ref(scratch, q_rows, table, sm_scale)

    monkeypatch.setattr(h, "kv4_to_v32", kv4_to_v32)
    monkeypatch.setattr(h, "_flashmla_fp8_decode", fp8_decode)
    table = torch.randint(0, rows.shape[0], (t, 64), dtype=torch.int32)
    table[1, 20:] = -1
    q = (torch.randn(t, layer.tp_q_head_num, 576) * 0.3).bfloat16()
    be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode("TARGET_VERIFY"), seq_lens_cpu=None, batch_size=t)
    batch_meta = object()
    meta = types.SimpleNamespace(dsa_cache_seqlens_int32=torch.full((t,), 64, dtype=torch.int32), page_table_1=None,
                                 cache_seqlens_int32=None, flashmla_metadata=batch_meta)
    out = h.forward_flashmla_kv(be, q, table, 576 ** -0.5, 512, layer, meta)
    ref = kv4.sparse_mla_ref(q, rows, table, 576 ** -0.5)
    assert be._numinous_kv4_state.calls["v32"] == 1 and calls["rows"] == 0 and calls["dedup"] == 0
    # the per-128 E4M3 re-quantization costs FP8-level error on top of the exact FP4 values
    err = float((out.float() - ref).norm() / ref.norm())
    assert err < 0.03, err
    chunks = -(-t // sel)
    assert seen["conv"] == chunks
    # the batch's own FlashMLA tile metadata only when one call covers the whole batch
    assert seen["meta"] == ([batch_meta] if chunks == 1 else [None] * chunks)


def test_v32_ref_matches_stock_quantizer_convention():
    """Our V32 reference against an independent restatement of quant_k_cache._quantize_k_cache_ref (scale = amax/448
    stored fp32, q = (x / scale) -> e4m3, rope copied as bf16): identical bytes on exact FP4 values."""
    kv4, _, _ = _load_modules()
    torch.manual_seed(1)
    nope = (torch.randn(32, 512) * 0.7).bfloat16()
    rope = (torch.randn(32, 64) * 0.7).bfloat16()
    rows = kv4.quantize_rows_ref(nope, rope)
    ours = _v32_rows_ref(kv4, rows)
    x = kv4.dequantize_rows_ref(rows, torch.bfloat16)  # exact values, as the bf16 input the stock quantizer takes
    stock = torch.empty(32, 656, dtype=torch.uint8)
    for tile in range(4):
        part = x[:, tile * 128:(tile + 1) * 128].float()
        sinv = part.abs().max(dim=-1).values / 448.0
        stock[:, 512 + 4 * tile: 516 + 4 * tile] = sinv.contiguous().view(torch.uint8).view(32, 4)
        stock[:, tile * 128:(tile + 1) * 128] = (part / sinv[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    stock[:, 528:] = x[:, 512:].contiguous().view(torch.uint8).view(32, 128)
    # x * (1/s) vs x / s can differ in the last fp32 bit, which can flip an e4m3 rounding: allow a handful of bytes
    diff = (ours != stock).sum().item()
    assert diff <= 32 * 656 // 1000, diff
    assert torch.equal(ours[:, 512:], stock[:, 512:])


# ------------------------------------------------------------------------------------------------ 5. q8 prefill
# NUMINOUS_GLM_KV4_SM90_PREFILL=q8 (rtxattn, M6 lever): the sequence prefill through v0.5.20's native FP8 SM90 sparse
# prefill kernel. The kernel is emulated with its source semantics: scores x q_scale x kv_scale x sm_scale
# (kernel.cuh:198-200), output x kv_scale / rowsum (:415-417), and -1 slots clamped onto zero pad rows at
# s_kv - topk + slot but masked to -inf (:803-822, :328-329).

def _install_fake_q8(monkeypatch, record):
    def fwd(q, kv, indices, sm_scale, q_scale, kv_scale, d_v, attn_sink=None, topk_length=None, out=None,
            max_logits=None, lse=None):
        assert q.dtype == torch.float8_e4m3fn and kv.dtype == torch.float8_e4m3fn
        assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
        assert q.dim() == 3 and kv.dim() == 3 and kv.shape[1] == 1 and indices.dim() == 3 and indices.shape[1] == 1
        assert q.shape[1] % 64 == 0 and indices.dtype == torch.int32 and attn_sink is None and topk_length is None
        s_q, _, d = q.shape
        s_kv, topk = kv.shape[0], indices.shape[2]
        pad_base = s_kv - topk
        record.append({"s_kv": s_kv, "topk": topk, "pad": kv[pad_base:].float().abs().max().item(),
                       "q_scale": float(q_scale), "kv_scale": float(kv_scale), "heads": q.shape[1],
                       "q_pad": q[:, record_heads[0]:].float().abs().max().item() if q.shape[1] > record_heads[0]
                       else 0.0})
        qf = q.float() * float(q_scale)
        kf = kv.view(s_kv, d).float() * float(kv_scale)
        o = torch.zeros(s_q, q.shape[1], d_v)
        for i in range(s_q):
            idx = indices[i, 0].long()
            slot = torch.arange(topk)
            pad = idx < 0
            idx = torch.where(pad, pad_base + slot, idx)
            assert int(idx.max()) < s_kv and int(idx.min()) >= 0
            k = kf[idx]
            sc = qf[i] @ k.T * sm_scale
            sc[:, pad] = float("-inf")
            if bool(pad.all()):
                continue
            o[i] = torch.softmax(sc, dim=-1) @ k[:, :d_v]
        out.copy_(o.to(torch.bfloat16))
        return out, max_logits, lse

    record_heads = [0]
    mod = types.ModuleType("sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90")
    mod.sparse_mla_q8kv8_prefill_fwd = fwd
    for name in ("sglang.kernels", "sglang.kernels.ops", "sglang.kernels.ops.attention"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return record_heads


def _prefill_case(kv4, eng, h, monkeypatch, lens=(40, 25), t=12, k=64):
    rows, be, layer = _setup(kv4, eng, h, monkeypatch=monkeypatch)
    perm = torch.randperm(rows.shape[0])
    pt = torch.full((len(lens), 48), -1, dtype=torch.int32)
    pt[0, : lens[0]] = perm[: lens[0]].int()
    pt[1, : lens[1]] = perm[100: 100 + lens[1]].int()
    table = torch.full((t, k), -1, dtype=torch.int32)
    for i in range(t):
        b = 0 if i < 7 else 1
        pos = torch.randperm(lens[b])[: min(k, lens[b])]
        table[i, : pos.numel()] = pt[b, pos]
    table[5] = -1                                          # a token with no valid slot -> zeros
    q = (torch.randn(t, layer.tp_q_head_num, 576) * 0.3).bfloat16()
    be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode("EXTEND"), seq_lens_cpu=torch.tensor(lens),
                                                batch_size=len(lens))
    meta = types.SimpleNamespace(dsa_cache_seqlens_int32=None, page_table_1=pt,
                                 cache_seqlens_int32=torch.tensor(lens, dtype=torch.int32))
    return rows, be, layer, q, table, meta


@pytest.mark.parametrize("scale", ["identity", "amax"])
def test_q8_prefill_matches_reference(mods, monkeypatch, scale):
    kv4, eng, h, calls = mods
    monkeypatch.setenv(h.PREFILL_ENV, "q8")
    monkeypatch.setenv(h.Q8_SCALE_ENV, scale)
    record = []
    heads_box = _install_fake_q8(monkeypatch, record)
    rows, be, layer, q, table, meta = _prefill_case(kv4, eng, h, monkeypatch)
    heads_box[0] = layer.tp_q_head_num
    out = h.forward_flashmla_kv(be, q, table, 576 ** -0.5, 512, layer, meta)
    ref = kv4.sparse_mla_ref(q, rows, table, 576 ** -0.5)
    st = h._state(be)
    assert calls["seq"] == 1 and st.calls["q8"] == 1 and len(record) == 1, (calls, st.calls)
    r = record[0]
    assert r["s_kv"] == 65 + 64 and r["topk"] == 64 and r["pad"] == 0.0     # sum(lens) rows + topk zero rows
    assert r["heads"] == 64 and r["q_pad"] == 0.0                         # 4 heads zero-padded to 64
    if scale == "identity":
        assert r["q_scale"] == 1.0 and r["kv_scale"] == 1.0
    else:
        assert r["q_scale"] == pytest.approx(q.float().abs().max().item() / 448.0, rel=1e-6)
        assert 0 < r["kv_scale"] < 1.0
    assert out.shape == ref.shape
    assert out[5].abs().max() == 0
    # E4M3 rounding of q and kv: a few percent of the output scale, far from the bf16 path's 2e-2 bound only by FP8
    err = (out.float() - ref).abs().max().item()
    assert err < 6e-2 * ref.abs().max().item() + 1e-2, err


def test_q8_prefill_zeroes_pad_rows_left_by_a_larger_call(mods, monkeypatch):
    kv4, eng, h, calls = mods
    monkeypatch.setenv(h.PREFILL_ENV, "q8")
    record = []
    heads_box = _install_fake_q8(monkeypatch, record)
    rows, be, layer, q, table, meta = _prefill_case(kv4, eng, h, monkeypatch, lens=(40, 25))
    heads_box[0] = layer.tp_q_head_num
    h.forward_flashmla_kv(be, q, table, 576 ** -0.5, 512, layer, meta)
    st = h._state(be)
    st.q8_kv.fill_(1.0)                                    # stale data everywhere, incl. the next call's pad band
    rows2, be2, layer2, q2, table2, meta2 = _prefill_case(kv4, eng, h, monkeypatch, lens=(20, 10), t=6)
    be2._numinous_kv4_state = st                           # same state object, smaller call
    be2.token_to_kv_pool = be.token_to_kv_pool
    out = h.forward_flashmla_kv(be2, q2, table2, 576 ** -0.5, 512, layer2, meta2)
    assert record[-1]["s_kv"] == 30 + 64 and record[-1]["pad"] == 0.0
    ref = kv4.sparse_mla_ref(q2, rows, table2, 576 ** -0.5)
    assert (out.float() - ref).abs().max().item() < 6e-2 * ref.abs().max().item() + 1e-2


def test_q8_env_validation_and_default_bf16(mods, monkeypatch):
    kv4, eng, h, calls = mods
    monkeypatch.setenv(h.PREFILL_ENV, "fp8")
    with pytest.raises(ValueError, match="expected bf16 or q8"):
        h.prefill_mode()
    monkeypatch.setenv(h.PREFILL_ENV, "q8")
    monkeypatch.setenv(h.Q8_SCALE_ENV, "rowwise")
    with pytest.raises(ValueError, match="expected identity or amax"):
        h.q8_scale_mode()
    monkeypatch.delenv(h.PREFILL_ENV)
    monkeypatch.delenv(h.Q8_SCALE_ENV)
    record = []
    _install_fake_q8(monkeypatch, record)
    rows, be, layer, q, table, meta = _prefill_case(kv4, eng, h, monkeypatch)
    out = h.forward_flashmla_kv(be, q, table, 576 ** -0.5, 512, layer, meta)
    assert record == [] and h._state(be).calls["q8"] == 0 and calls["seq"] == 1   # default: the bf16 kernel
    assert torch.allclose(out.float(), kv4.sparse_mla_ref(q, rows, table, 576 ** -0.5), atol=2e-2, rtol=2e-2)


# ------------------------------------------------------------------------------------------------ 6. GLM-5.3-Flash
# (rtxattn 2026-10-05, coordinator: the Flash PD decode pool must hold ~170 resident sessions). Flash builds its DSA
# pool inside HybridLinearKVPool and its DSA layers are zero-rope (config qk_rope_head_dim 0).


def test_flash_arch_enables_the_fp4_pool(mods, monkeypatch):
    kv4, eng, h, _ = mods
    monkeypatch.setenv(eng.ENV, "1")
    flash = types.SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"])
    glm = types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
    other = types.SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])
    assert eng.enabled_for(flash, torch.float8_e4m3fn) and eng.enabled_for(glm, torch.float8_e4m3fn)
    assert not eng.enabled_for(other, torch.float8_e4m3fn)
    monkeypatch.delenv(eng.ENV)
    assert not eng.enabled_for(flash, torch.float8_e4m3fn)


class _HybridWrapper:
    """The two HybridLinearKVPool members the read path uses (memory_pool.py:3870-3916, :4010-4015)."""

    def __init__(self, inner, full_ids):
        self.full_kv_pool = inner
        self.full_attention_layer_id_mapping = {lid: i for i, lid in enumerate(full_ids)}

    def _transfer_full_attention_id(self, layer_id):
        return self.full_attention_layer_id_mapping[layer_id]


def test_flash_zero_rope_read_through_the_hybrid_wrapper(mods, monkeypatch):
    """forward_flashmla_kv on a Flash-shaped layer: global layer 7 is dense index 1 of the wrapper's full pool, q is
    512-wide (zero rope), the rows were written with zero rope. Equals the fp32 reference with q zero-padded to 576."""
    kv4, eng, h, calls = mods
    rows0, be, layer = _setup(kv4, eng, h, monkeypatch=monkeypatch, dedup=False)
    n = rows0.shape[0]
    nope = (torch.randn(n, 512) * 0.5).bfloat16()
    rows = kv4.quantize_rows_ref(nope, torch.zeros(n, 64).bfloat16())          # zero rope bytes
    other = kv4.quantize_rows_ref((torch.randn(n, 512) * 0.5).bfloat16(), torch.zeros(n, 64).bfloat16())
    inner = types.SimpleNamespace(kv_buffer=[other.view(n, 1, kv4.ROW_BYTES), rows.view(n, 1, kv4.ROW_BYTES)],
                                  start_layer=0, numinous_kv4=True)
    be.token_to_kv_pool = _HybridWrapper(inner, full_ids=[3, 7])
    flash_layer = types.SimpleNamespace(layer_id=7, tp_q_head_num=layer.tp_q_head_num, head_dim=512)
    t, k = 6, 64
    table = torch.randint(0, n, (t, k), dtype=torch.int32)
    table[2, 20:] = -1
    q = (torch.randn(t, layer.tp_q_head_num, 512) * 0.3).bfloat16()
    be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode("DECODE"), seq_lens_cpu=None, batch_size=t)
    meta = types.SimpleNamespace(dsa_cache_seqlens_int32=None, page_table_1=None, cache_seqlens_int32=None)
    out = h.forward_flashmla_kv(be, q, table, 512 ** -0.5, 512, flash_layer, meta)
    ref = kv4.sparse_mla_ref(torch.nn.functional.pad(q, (0, 64)), rows, table, 512 ** -0.5)
    assert out.shape == (t, layer.tp_q_head_num, 512)
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2), (out.float() - ref).abs().max()
    st = h._state(be)
    assert st.n_pool == n                                                      # sized from the wrapped pool


def test_flash_attach_unwraps_the_hybrid_pool(mods, monkeypatch):
    kv4, eng, h, _ = mods
    be, mr = _attach_backend(h)
    inner = mr.token_to_kv_pool
    mr.token_to_kv_pool = _HybridWrapper(inner, full_ids=[3, 7])
    be.token_to_kv_pool = mr.token_to_kv_pool
    assert h.attach(be, mr) is True and be.supports_mha_one_shot is False


@have_tree
def test_flash_hybrid_pool_adopts_fp4_rows_and_round_trips_zero_rope(tmp_path):
    """The real classes on a hooked copy of the v0.5.20 tree, run in a subprocess (TRITON_INTERPRET for quant_write /
    dequant_rows on CPU).
    - A Flash-shaped HybridLinearKVPool (use_dsa, zero rope, kv_cache_dim 352 from kv4h_dim) holds a
      GLMKV4DSATokenToKVPool after kv4h_hybrid_pool: uint8 352-B rows, zero_rope set, no FP8 store path.
    - set_mla_kv_buffer through the wrapper (k_rope empty, as Flash passes it) followed by get_mla_kv_buffer gives
      back the latent within FP4 error and rope None."""
    import subprocess
    import textwrap
    root = tmp_path / "tree"
    shutil.copytree(SRC, root / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
    hooks.apply(str(root / "sglang"))
    script = textwrap.dedent(f"""
        import os, sys, torch
        os.environ["NUMINOUS_GLM_KV4"] = "1"
        sys.path.insert(0, {str(root)!r})
        from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=64))
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
        from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsa import glm_kv4_engine as eng
        lin = [0, 1, 2]
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(tp_world_size=1, intermediate_size=64, n_groups=1, num_heads=2,
                                            head_dim=16, state_size=16, conv_kernel=4)
            params = Mamba2CacheParams(shape=shape, layers=lin)
        req = HybridReqToTokenPool(size=4, mamba_size=8, mamba_spec_state_size=4, max_context_len=256,
                                   device="cpu", enable_memory_saver=False, cache_params=params,
                                   mamba_layer_ids=lin, enable_mamba_extra_buffer=False,
                                   speculative_num_draft_tokens=3)
        pool = HybridLinearKVPool(size=64 * 8, dtype=torch.float8_e4m3fn, page_size=64, head_num=1, head_dim=512,
                                  full_attention_layer_ids=[3, 7], device="cpu", mamba_pool=req.mamba_pool,
                                  use_mla=True, kv_lora_rank=512, qk_rope_head_dim=0, use_dsa=True,
                                  index_head_dim=128, kv_cache_dim=eng.ROW_BYTES)
        inner = pool.full_kv_pool
        assert type(inner).__name__ == "GLMKV4DSATokenToKVPool", type(inner)
        assert inner.kv_cache_dim == 352 and inner.numinous_zero_rope and not inner.dsa_kv_cache_store_fp8
        assert inner.kv_buffer[0].dtype == torch.uint8 and inner.kv_buffer[0].shape[-1] == 352
        class L: pass
        layer = L(); layer.layer_id = 7
        torch.manual_seed(0)
        loc = torch.tensor([65, 66, 130, 131], dtype=torch.int64)
        nope = (torch.randn(4, 1, 512) * 0.5).bfloat16()
        pool.set_mla_kv_buffer(layer, loc, nope, nope.new_empty(4, 1, 0))
        got_nope, got_rope = pool.get_mla_kv_buffer(layer, loc)
        assert got_rope is None and layer.layer_id == 7
        rel = ((got_nope.float() - nope.float()).norm() / nope.float().norm()).item()
        assert rel < 0.2, rel                                   # E2M1 x E4M3-per-16 latent
        raw = inner.kv_buffer[1][loc].view(4, 352)
        assert raw[:, 288:].eq(0).all()                         # the rope bytes stay zero
        print("OK", rel)
    """)
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout[-2000:], r.stderr[-4000:])
