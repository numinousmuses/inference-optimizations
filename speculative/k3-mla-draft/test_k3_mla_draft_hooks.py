"""k3_mla_draft_hooks: the three v0.5.20 edits apply once and compile, and the MLA draft module's expanded-MHA
attention equals latent MLA attention exactly (CPU, random weights, stubbed SGLang layers): non-causal (Inferact
block 7, DFlash2) and causal (Block5) block attention over injected context KV, YaRN softmax scale, the shared
rotated k_pe, the 192/128 -> 256 padding and the Inferact checkpoint renames."""
from __future__ import annotations

import enum
import math
import pathlib
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k3_mla_draft_hooks as H  # noqa: E402

FORK = pathlib.Path("/home/ubuntu/sglang-fork")


def _show(rel):
    if not (FORK / ".git").exists():
        pytest.skip("fork missing")
    r = subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{rel}"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip("v0.5.20 missing")
    return r.stdout


@pytest.mark.parametrize("path", sorted({e[0] for e in H.EDITS}))
def test_edits_apply_once_and_compile(path):
    src = _show(path)
    new, line = H.patch_source(path, src)
    want = {e[0]: e[4] + (1 if e[3] else 0) for e in H.EDITS}[path]   # call sites + helper
    assert "applied" in line and new.count(H.MARKER) == want
    compile(new, path, "exec")
    again, line2 = H.patch_source(path, new)
    assert again == new and "already" in line2


def test_model_config_branch_precedes_dsv4_and_sizes_256():
    new, _ = H.patch_source(H.CFG, _show(H.CFG))
    i, j = new.index("_numinous_is_mla_draft(self.hf_config)"), new.index('"DeepseekV4ForCausalLM" in self.hf_config')
    assert i < j and "_numinous_mla_draft_dims(self)" in new[i:j] and "AttentionArch.MHA" in new[i:j]
    ns: dict = {"os": __import__("os")}
    exec(H.CFG_TAIL, ns)
    f = ns["_numinous_is_mla_draft"]
    assert f(types.SimpleNamespace(architectures=["K3DSparkModel"]))
    assert f(types.SimpleNamespace(architectures=["DFlash2DraftModel"], dflash_config={"attention_mode": "mla"}))
    assert not f(types.SimpleNamespace(architectures=["DSparkDraftModel"], dflash_config={"block_size": 7}))
    assert not f(types.SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]))


def test_module_source_compiles():
    compile(H.MODULE_SRC.read_text(), H.MODULE_DST, "exec")


# ---------------------------------------------------------------- stubbed SGLang, exact-math check
class _Lin(nn.Module):
    def __init__(self, i, o, bias=False, prefix="", **_):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i, dtype=torch.float64) / math.sqrt(i))

    def forward(self, x):
        return x @ self.weight.T, None


class _RMS(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(d, dtype=torch.float64))
        self.variance_epsilon = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon) * self.weight


def _rope_tables(d, pos):
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    f = pos.to(torch.float64)[:, None] * inv[None]
    return f.cos(), f.sin()


def _rope_interleaved(x, pos):
    """[T, H, d] interleaved (non-neox) rotation."""
    cos, sin = _rope_tables(x.shape[-1], pos)
    cos, sin = cos[:, None, :], sin[:, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


class _Rope:
    def __init__(self, head, rotary_dim=None, max_position=None, base=None, rope_scaling=None, is_neox_style=True):
        assert is_neox_style is False   # K3 MLA rope is interleaved
        self.rope_scaling = rope_scaling

    def __call__(self, positions, q, k):
        return _rope_interleaved(q, positions), _rope_interleaved(k, positions)


class _AT(enum.Enum):
    DECODER = 0
    ENCODER_ONLY = 1


class _Ctx:
    """The draft KV pool + the block forward batch: context K/V written through the injection API."""
    def __init__(self):
        self.k = {}
        self.v = {}


class _Radix(nn.Module):
    """RadixAttention + backend semantics: MHA/GQA/MQA over the pool's context K/V and the block's own K/V
    (q [T, H*d], k [T, KVH*d], v [T, KVH*dv]; output [T, H*dv])."""
    def __init__(self, num_heads, head_dim, scaling, num_kv_heads, layer_id, v_head_dim=-1, sliding_window_size=-1,
                 attn_type=_AT.DECODER, **_):
        super().__init__()
        self.h, self.d, self.scaling, self.layer_id, self.attn_type = num_heads, head_dim, scaling, layer_id, attn_type
        self.kvh, self.dv = num_kv_heads, (v_head_dim if v_head_dim > 0 else head_dim)
        self.k_scale = self.v_scale = None

    def forward(self, q, k, v, fb):
        t = q.shape[0]
        q, k, v = q.view(t, self.h, self.d), k.view(t, self.kvh, self.d), v.view(t, self.kvh, self.dv)
        kc, vc = fb.k[self.layer_id], fb.v[self.layer_id]
        kk, vv = torch.cat([kc, k]), torch.cat([vc, v])
        g = self.h // self.kvh
        kk, vv = kk.repeat_interleave(g, dim=1), vv.repeat_interleave(g, dim=1)
        s = torch.einsum("thd,shd->hts", q, kk) * self.scaling
        n_ctx = kc.shape[0]
        if self.attn_type == _AT.DECODER:   # causal inside the block
            mask = torch.ones(t, n_ctx + t, dtype=torch.bool)
            mask[:, n_ctx:] = torch.tril(torch.ones(t, t, dtype=torch.bool))
            s = s.masked_fill(~mask[None], float("-inf"))
        return torch.einsum("hts,shd->thd", s.softmax(-1), vv).reshape(t, -1)


def _stub_modules(monkeypatch):
    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    src = _show("srt/configs/model_config.py")
    start = src.index("def yarn_get_mscale")
    end = src.index("\ndef ", src.index("def compute_mla_mscale_scaling") + 10)
    ns = {"logger": types.SimpleNamespace(warning=lambda *a, **k: None, warning_once=lambda *a, **k: None),
          "math": math}
    exec("import math\n" + src[start:end], ns)
    for pkg in ("sglang", "sglang.srt", "sglang.srt.configs", "sglang.srt.layers", "sglang.srt.models",
                "sglang.srt.model_executor", "sglang.srt.utils"):
        mod(pkg)
    mod("sglang.srt.configs.model_config", compute_mla_mscale_scaling=ns["compute_mla_mscale_scaling"])
    mod("sglang.srt.layers.layernorm", RMSNorm=_RMS)
    mod("sglang.srt.layers.linear", ColumnParallelLinear=_Lin, ReplicatedLinear=_Lin, RowParallelLinear=_Lin)
    mod("sglang.srt.layers.radix_attention", AttentionType=_AT, RadixAttention=_Radix)
    mod("sglang.srt.layers.rotary_embedding", get_rope=_Rope)
    mod("sglang.srt.model_executor.forward_batch_info", ForwardBatch=object)

    class _DecoderLayer(nn.Module):
        attention_cls = None

    class _DSpark(nn.Module):
        def __init__(self, config, quant_config=None, prefix=""):
            super().__init__()
            self.config = config

        def load_weights(self, weights):
            self.got = [n for n, _ in weights]

    mod("sglang.srt.models.dflash", DFlashDecoderLayer=_DecoderLayer,
        _get_dflash_layer_attention_params=lambda cfg, i: (-1, _AT.ENCODER_ONLY))
    mod("sglang.srt.models.dspark", DSparkDraftModel=_DSpark)
    mod("sglang.srt.runtime_context", get_parallel=lambda: types.SimpleNamespace(tp_size=1))
    mod("sglang.srt.utils.hf_transformers_utils",
        get_rope_config=lambda c: (c.rope_parameters["rope_theta"], dict(c.rope_parameters)))
    spec = __import__("importlib.util").util.spec_from_file_location("k3_mla_draft_under_test", H.MODULE_SRC)
    m = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cfg(causal=None, mscale_all_dim=1.0, extra=None):
    c = types.SimpleNamespace(hidden_size=48, num_attention_heads=4, qk_nope_head_dim=16, qk_rope_head_dim=8,
                              v_head_dim=16, q_lora_rank=24, kv_lora_rank=20, rms_norm_eps=1e-5,
                              max_position_embeddings=4096, architectures=["K3DSparkModel"],
                              rope_parameters={"rope_type": "yarn", "factor": 32.0, "rope_theta": 50000.0,
                                               "original_max_position_embeddings": 32768, "beta_fast": 32,
                                               "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": mscale_all_dim})
    if causal is not None:
        c.dflash_config = {"causal": causal}
    for k, v in (extra or {}).items():
        setattr(c, k, v)
    return c


def _reference_latent_mla(attn, x_ctx, pos_ctx, x_blk, pos_blk, causal):
    """Absorbed/latent MLA (the vLLM kernels' math): scores = q_nope . W_uk c + q_pe . k_pe over the latent cache."""
    H, dn, dr, dv = attn.num_heads, attn.d_nope, attn.d_rope, attn.d_v
    x_all = torch.cat([x_ctx, x_blk])
    pos_all = torch.cat([pos_ctx, pos_blk])
    kv_a = x_all @ attn.kv_a_proj_with_mqa.weight.T
    c, k_pe = kv_a[:, :attn.kv_lora_rank], kv_a[:, attn.kv_lora_rank:]
    c = attn.kv_a_layernorm(c)
    k_pe = _rope_interleaved(k_pe[:, None, :], pos_all)[:, 0, :]
    wkv = attn.kv_b_proj.weight.view(H, dn + dv, attn.kv_lora_rank)
    w_uk, w_uv = wkv[:, :dn, :], wkv[:, dn:, :]
    q = (attn.q_a_layernorm(x_blk @ attn.q_a_proj.weight.T) @ attn.q_b_proj.weight.T).view(-1, H, dn + dr)
    q_nope, q_pe = q[..., :dn], _rope_interleaved(q[..., dn:].contiguous(), pos_blk)
    q_lat = torch.einsum("thd,hdl->thl", q_nope, w_uk)              # absorb W_uk into q
    s = (torch.einsum("thl,sl->hts", q_lat, c) + torch.einsum("thd,sd->hts", q_pe, k_pe)) * attn.scaling
    t, n_ctx = x_blk.shape[0], x_ctx.shape[0]
    if causal:
        mask = torch.ones(t, n_ctx + t, dtype=torch.bool)
        mask[:, n_ctx:] = torch.tril(torch.ones(t, t, dtype=torch.bool))
        s = s.masked_fill(~mask[None], float("-inf"))
    o_lat = torch.einsum("hts,sl->thl", s.softmax(-1), c)
    o = torch.einsum("thl,hvl->thv", o_lat, w_uv).reshape(t, H * dv)  # absorb W_uv after
    return o @ attn.o_proj.weight.T


@pytest.mark.parametrize("causal", [None, False, True])
def test_expanded_attention_equals_latent_mla(monkeypatch, causal):
    m = _stub_modules(monkeypatch)
    torch.manual_seed(0)
    attn = m.MLAExpandedAttention(_cfg(causal), layer_id=0)
    assert attn.head_dim == m.PAD_HEAD_DIM == 256 and attn.attn.d == 256 and attn.attn.kvh == attn.num_heads
    assert attn.attn.attn_type == (_AT.DECODER if causal else _AT.ENCODER_ONLY)
    x_ctx, x_blk = torch.randn(11, 48, dtype=torch.float64), torch.randn(4, 48, dtype=torch.float64)
    pos_ctx, pos_blk = torch.arange(11) + 100, torch.arange(4) + 111
    # context KV through the DSPARK / DFLASH injection API (kv_proj_only -> apply_k_norm -> apply_k_rope)
    k, v = attn.kv_proj_only(x_ctx)
    k = attn.apply_k_rope(pos_ctx, attn.apply_k_norm(k))
    fb = _Ctx()
    fb.k[0], fb.v[0] = k.view(11, attn.num_heads, 256), v.view(11, attn.num_heads, 256)
    got = attn(pos_blk, x_blk, fb)
    ref = _reference_latent_mla(attn, x_ctx, pos_ctx, x_blk, pos_blk, bool(causal))
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9)
    # padding columns carry nothing
    assert torch.count_nonzero(fb.v[0][..., attn.d_v:]) == 0 and torch.count_nonzero(fb.k[0][..., attn.d_qk:]) == 0


def test_yarn_softmax_scale_matches_vllm(monkeypatch):
    m = _stub_modules(monkeypatch)
    # vLLM kimi_k3/nvidia/mla.py: scale = d_qk^-0.5 * (0.1 * mscale_all_dim * ln(factor) + 1)^2
    for mad in (1.0, 0.0):
        _, rs, scale = m.mla_rope_and_scale(_cfg(mscale_all_dim=mad))
        mscale = 0.1 * mad * math.log(32.0) + 1.0
        assert scale == pytest.approx(24 ** -0.5 * mscale * mscale, rel=1e-12)
        assert rs["rope_type"] == "deepseek_yarn"


def test_inferact_renames_and_target_layer_count(monkeypatch):
    m = _stub_modules(monkeypatch)
    cfg = _cfg(extra={"target_layer_ids": [2, 23, 47, 71, 89], "num_target_layers": 5,
                      "target_num_hidden_layers": 93})
    model = m.K3DSparkModel(cfg)
    assert cfg.num_target_layers == 93     # SGLang reads it as the target layer count
    assert model.decoder_layer_cls is m.MLALatentDecoderLayer   # default layout
    model.check = None
    m.check_attention_loaded = lambda *a: None
    model.load_weights(iter([("context_proj.weight", 0), ("context_norm.weight", 0), ("final_norm.weight", 0),
                             ("layers.0.self_attn.q_a_proj.weight", 0), ("markov_head.markov_w1.weight", 0)]))
    assert model.got == ["fc.weight", "hidden_norm.weight", "norm.weight", "layers.0.self_attn.q_a_proj.weight",
                         "markov_head.markov_w1.weight"]
    assert model._build_fused_kv_write_bundle(None) is None and model._stacked_ctx_kv_params() is None
    assert m.K3DSparkModel.supports_fused_context_kv is False


def test_missing_attention_weights_raise(monkeypatch):
    m = _stub_modules(monkeypatch)
    lay = nn.Module()
    lay.self_attn = m.MLAExpandedAttention(_cfg(), layer_id=0)
    root = nn.Module()
    root.layers = nn.ModuleList([lay])
    with pytest.raises(ValueError, match="not loaded"):
        m.check_attention_loaded(root, {"layers.0.self_attn.q_a_proj.weight"})
    names = {n for n, _ in root.named_parameters() if ".self_attn." in n}
    m.check_attention_loaded(root, names)


# ---------------------------------------------------------------- latent layout (rtxattn, 2026-10-01)
def _inject(attn, x_ctx, pos_ctx, layer_id=0):
    """Context KV through the DSPARK / DFLASH injection API exactly as dspark.write_target_hidden_kv (patched) does:
    kv_proj_only -> apply_k_norm -> apply_k_rope -> view(-1, num_kv_heads, head_dim / v_head_dim)."""
    k, v = attn.kv_proj_only(x_ctx)
    k = attn.apply_k_rope(pos_ctx, attn.apply_k_norm(k))
    fb = _Ctx()
    fb.k[layer_id] = k.view(-1, attn.num_kv_heads, attn.head_dim)
    fb.v[layer_id] = v.view(-1, attn.num_kv_heads, getattr(attn, "v_head_dim", attn.head_dim))
    return fb


@pytest.mark.parametrize("causal", [None, False, True])
def test_latent_attention_equals_reference_mla(monkeypatch, causal):
    m = _stub_modules(monkeypatch)
    torch.manual_seed(1)
    attn = m.MLALatentAttention(_cfg(causal), layer_id=0)
    L, dr = attn.kv_lora_rank, attn.d_rope
    assert (attn.num_kv_heads, attn.head_dim, attn.v_head_dim) == (1, L + dr, L)
    assert (attn.attn.kvh, attn.attn.d, attn.attn.dv) == (1, L + dr, L)
    assert attn.attn.attn_type == (_AT.DECODER if causal else _AT.ENCODER_ONLY)
    x_ctx, x_blk = torch.randn(13, 48, dtype=torch.float64), torch.randn(5, 48, dtype=torch.float64)
    pos_ctx, pos_blk = torch.arange(13) + 200, torch.arange(5) + 213
    fb = _inject(attn, x_ctx, pos_ctx)
    got = attn(pos_blk, x_blk, fb)
    ref = _reference_latent_mla(attn, x_ctx, pos_ctx, x_blk, pos_blk, bool(causal))
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9)
    # the cache holds the latent: v == k[:, :L] (normalized c), k[:, L:] == rope(k_pe) shared by every head
    torch.testing.assert_close(fb.v[0], fb.k[0][..., :L])


def test_latent_equals_expanded_same_weights(monkeypatch):
    m = _stub_modules(monkeypatch)
    torch.manual_seed(2)
    cfg = _cfg(False)
    lat = m.MLALatentAttention(cfg, layer_id=0)
    exp = m.MLAExpandedAttention(cfg, layer_id=0)
    exp.load_state_dict(lat.state_dict())
    x_ctx, x_blk = torch.randn(9, 48, dtype=torch.float64), torch.randn(3, 48, dtype=torch.float64)
    pos_ctx, pos_blk = torch.arange(9) + 7, torch.arange(3) + 16
    a = lat(pos_blk, x_blk, _inject(lat, x_ctx, pos_ctx))
    b = exp(pos_blk, x_blk, _inject(exp, x_ctx, pos_ctx))
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


def test_latent_empty_batch_and_quantized_weight_guard(monkeypatch):
    m = _stub_modules(monkeypatch)
    attn = m.MLALatentAttention(_cfg(), layer_id=0)
    assert attn(torch.zeros(0, dtype=torch.long), torch.zeros(0, 48, dtype=torch.float64), _Ctx()).shape == (0, 48)
    attn.kv_b_proj.weight = nn.Parameter(torch.zeros(3, 7, dtype=torch.float64))   # e.g. a packed FP8/FP4 weight
    with pytest.raises(RuntimeError, match="unquant"):
        attn._w_uk_uv()


def test_layout_env(monkeypatch):
    m = _stub_modules(monkeypatch)
    monkeypatch.delenv("NUMINOUS_K3_MLA_DRAFT_LAYOUT", raising=False)
    assert m.mla_decoder_layer_cls() is m.MLALatentDecoderLayer
    assert m.mla_pool_dims(_cfg()) == (20 + 8, 20, 1)
    monkeypatch.setenv("NUMINOUS_K3_MLA_DRAFT_LAYOUT", "expanded")
    assert m.mla_decoder_layer_cls() is m.MLAExpandedDecoderLayer and m.mla_pool_dims(_cfg()) == (256, 256, 0)
    monkeypatch.setenv("NUMINOUS_K3_MLA_DRAFT_LAYOUT", "bogus")
    with pytest.raises(ValueError):
        m.mla_draft_layout()


# ---------------------------------------------------------------- the real checkpoints (safetensors headers)
DATA = HERE / "testdata" / "k3_mla_draft"
CKPTS = ["Inferact__Kimi-K3-DSpark", "Inferact__Kimi-K3-DSpark-Block5", "lightseekorg__kimi-k3-dflash2"]


def _ckpt(name):
    import json
    return json.loads((DATA / f"{name}.json").read_text())


@pytest.mark.parametrize("name", CKPTS)
def test_checkpoint_attention_tensors_match_module(monkeypatch, name):
    """Every self_attn tensor in the pinned checkpoint header maps (after the Inferact renames) onto a parameter of
    the same shape in both layouts at TP=1, and every module attention parameter is present in the checkpoint."""
    m = _stub_modules(monkeypatch)
    d = _ckpt(name)
    cfg = types.SimpleNamespace(**d["config"])
    tensors = {}
    for n, t in d["tensors"].items():
        for a, b in m.INFERACT_RENAMES:
            if n.startswith(a):
                n = b + n[len(a):]
        tensors[n] = tuple(t["shape"])
    for cls in (m.MLALatentAttention, m.MLAExpandedAttention):
        with torch.device("meta"):
            attn = cls(cfg, layer_id=0)
        params = {f"layers.0.self_attn.{k}": tuple(v.shape) for k, v in attn.named_parameters()}
        ck = {k: v for k, v in tensors.items() if k.startswith("layers.0.self_attn.")}
        assert params == ck, (cls.__name__, set(params) ^ set(ck))
    for n in ("fc.weight", "hidden_norm.weight", "norm.weight"):
        assert n in tensors   # DSpark/DFlash names after the renames
    assert sum(1 for k in tensors if k.endswith("self_attn.kv_b_proj.weight")) == cfg.num_hidden_layers


def _draft_kv_bytes(cfg, layout, tp=8, dtype_bytes=1):
    if layout == "latent":
        per_layer = (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) + cfg["kv_lora_rank"]
    else:
        per_layer = (cfg["num_attention_heads"] // tp) * 2 * 256
    return cfg["num_hidden_layers"] * per_layer * dtype_bytes


def test_draft_kv_bytes_per_token_rank():
    """derived: fp8 draft KV bytes per token per rank at TP8 (what the pool configurator charges; the DSPARK ring
    multiplies it by rows x R, DFLASH by the DCP-replicated span)."""
    got = {n: (_draft_kv_bytes(_ckpt(n)["config"], "latent"), _draft_kv_bytes(_ckpt(n)["config"], "expanded"))
           for n in CKPTS}
    assert got == {"Inferact__Kimi-K3-DSpark": (5440, 20480), "Inferact__Kimi-K3-DSpark-Block5": (3264, 18432),
                   "lightseekorg__kimi-k3-dflash2": (5440, 20480)}
    # the pool sizing hook gives the same geometry SGLang's dflash_draft_cell_size_per_token multiplies out
    ns: dict = {"os": __import__("os")}
    exec(H.CFG_TAIL, ns)
    for n in CKPTS:
        c = _ckpt(n)["config"]
        mc = types.SimpleNamespace(hf_text_config=types.SimpleNamespace(**c))
        ns["_numinous_mla_draft_dims"](mc)
        assert mc.hf_text_config.num_key_value_heads == 1
        cell = max(1, mc.hf_text_config.num_key_value_heads // 8) * (mc.head_dim + mc.v_head_dim) * c["num_hidden_layers"]
        assert cell == got[n][0]


def test_model_config_dims_env(monkeypatch):
    ns: dict = {"os": __import__("os")}
    exec(H.CFG_TAIL, ns)
    c = _ckpt(CKPTS[0])["config"]
    monkeypatch.setenv("NUMINOUS_K3_MLA_DRAFT_LAYOUT", "expanded")
    mc = types.SimpleNamespace(hf_text_config=types.SimpleNamespace(**c))
    ns["_numinous_mla_draft_dims"](mc)
    assert (mc.head_dim, mc.v_head_dim, mc.hf_text_config.num_key_value_heads) == (256, 256, 64)
    monkeypatch.setenv("NUMINOUS_K3_MLA_DRAFT_LAYOUT", "latent")
    mc = types.SimpleNamespace(hf_text_config=types.SimpleNamespace(**c))
    ns["_numinous_mla_draft_dims"](mc)
    assert (mc.head_dim, mc.v_head_dim, mc.swa_head_dim, mc.swa_v_head_dim) == (576, 512, 576, 512)


def test_block5_rope_keeps_attn_factor(monkeypatch):
    """Block5's YaRN: mscale 1 / mscale_all_dim 0 / attn_factor 0.7426 -> cos/sin x 1.0 (attn_factor reaches the
    deepseek_yarn rope through rope_scaling) and softmax scale d_qk^-0.5 (mscale_all_dim 0)."""
    m = _stub_modules(monkeypatch)
    cfg = types.SimpleNamespace(**_ckpt("Inferact__Kimi-K3-DSpark-Block5")["config"])
    theta, rs, scale = m.mla_rope_and_scale(cfg)
    assert theta == 1e6 and rs["rope_type"] == "deepseek_yarn" and rs["attn_factor"] == pytest.approx(0.7426255848)
    assert scale == pytest.approx(192 ** -0.5, rel=1e-12)
    cos_mult = (0.1 * 1.0 * math.log(32.0) + 1.0) / 1.0 * rs["attn_factor"]
    assert cos_mult == pytest.approx(1.0, abs=1e-6)
    assert m.draft_is_causal(cfg) is True
    cfg7 = types.SimpleNamespace(**_ckpt("Inferact__Kimi-K3-DSpark")["config"])
    _, rs7, scale7 = m.mla_rope_and_scale(cfg7)
    assert scale7 == pytest.approx(192 ** -0.5 * (0.1 * math.log(32.0) + 1.0) ** 2, rel=1e-12)
    assert m.draft_is_causal(cfg7) is None   # non-causal block (ENCODER_ONLY)
    assert m.draft_is_causal(types.SimpleNamespace(**_ckpt("lightseekorg__kimi-k3-dflash2")["config"])) is False


# ---------------------------------------------------------------- engine edits behave (executed, not just compiled)
def _patched(path):
    new, _ = H.patch_source(path, _show(path))
    return new


def test_prefix_valid_asymmetric_rows_write_k_and_v():
    """The patched MHATokenToKVPool.set_kv_buffer_prefix_valid on CUDA with k 576 / v 512 rows: rows with
    col < commit_len land in both buffers at their own widths, the rest stay untouched."""
    import ast
    src = _patched(H.MPL)
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "set_kv_buffer_prefix_valid":
            seg = ast.get_source_segment(src, node)
            if H.MARKER in seg:
                fn = seg
    assert fn is not None
    calls = []

    def impl(k, v, kc, vc, loc_2d, commit_lens, row_dim, store_dtype):   # the tiled kernel's semantics
        calls.append(row_dim)
        assert k.shape[-1] * k.shape[-2] == row_dim if k.dim() == 3 else k.shape[-1] == row_dim
        rows = loc_2d.shape[1]
        valid = (torch.arange(rows)[None, :] < commit_lens.long()[:, None]).reshape(-1)
        loc = loc_2d.reshape(-1)[valid]
        kc[loc] = k[valid]
        vc[loc] = v[valid]

    ns = {"torch": torch, "Optional": __import__("typing").Optional, "RadixAttention": object, "_is_cuda": True,
          "_is_hip": False, "_set_kv_buffer_prefix_valid_impl": impl}
    exec("import textwrap\n" + "def _f():\n    pass\n", ns)
    exec(__import__("textwrap").dedent(fn), ns)
    pool = types.SimpleNamespace(dtype=torch.float32, store_dtype=torch.float32, start_layer=0, head_dim=576,
                                 v_head_dim=512, row_dim=576, v_row_dim=512,
                                 k_buffer=[torch.zeros(32, 1, 576)], v_buffer=[torch.zeros(32, 1, 512)])
    k = torch.randn(2 * 3, 1, 576)
    v = torch.randn(2 * 3, 1, 512)
    loc_2d = torch.tensor([[4, 5, 6], [10, 11, 12]])
    commit = torch.tensor([2, 3], dtype=torch.int32)
    ns["set_kv_buffer_prefix_valid"](pool, types.SimpleNamespace(layer_id=0), loc_2d, commit, k, v)
    assert calls == [576, 512]
    for i, loc in enumerate([4, 5, 6, 10, 11, 12]):
        live = loc != 6
        assert torch.equal(pool.k_buffer[0][loc], k[i] if live else torch.zeros(1, 576))
        assert torch.equal(pool.v_buffer[0][loc], v[i] if live else torch.zeros(1, 512))


def test_triton_draft_dcp_helper():
    ns: dict = {}
    exec(H.TRI_TAIL, ns)
    f = ns["_numinous_mla_draft_runner"]

    def runner(draft, archs, dcfg=None):
        hf = types.SimpleNamespace(architectures=archs, dflash_config=dcfg)
        return types.SimpleNamespace(is_draft_worker=draft, model_config=types.SimpleNamespace(hf_config=hf))
    assert f(runner(True, ["K3DSparkModel"]))
    assert f(runner(True, ["DFlash2DraftModel"], {"attention_mode": "mla"}))
    assert not f(runner(False, ["K3DSparkModel"]))          # never the target
    assert not f(runner(True, ["DSparkDraftModel"]))        # stock MHA drafts keep stock behaviour
    src = _patched(H.TRI)
    i = src.index("_numinous_mla_draft_runner(model_runner)")
    assert "self.dcp_size = 1" in src[i:i + 200]


def test_v_view_uses_v_head_dim():
    for path in (H.DSP, H.DFW):
        src = _patched(path)
        assert H.V_VIEW not in src.replace(H.V_VIEW_NEW, "")
    attn = types.SimpleNamespace(num_kv_heads=1, head_dim=576, v_head_dim=512)
    v = torch.zeros(4, 512)
    assert eval(H.V_VIEW_NEW.split("=", 1)[1].split("#")[0]).shape == (4, 1, 512)
    attn = types.SimpleNamespace(num_kv_heads=2, head_dim=128)   # stock draft: unchanged
    v = torch.zeros(4, 256)
    assert eval(H.V_VIEW_NEW.split("=", 1)[1].split("#")[0]).shape == (4, 2, 128)
