# Installed as sglang/srt/models/k3_mla_draft.py by operations/patches/k3_mla_draft_hooks.py (agent k3b300).
"""MLA-attention speculative drafts for Kimi-K3 on SGLang v0.5.20: Inferact/Kimi-K3-DSpark (K3DSparkModel, 5 layers,
block 7, non-causal), Inferact/Kimi-K3-DSpark-Block5 (K3DSparkModel, 3 layers, causal) and
lightseekorg/kimi-k3-dflash2 (DFlash2DraftModel with dflash_config.attention_mode "mla").

v0.5.20's DSpark/DFlash drafts are MHA only (q/k/v projections, an MHA draft KV pool, kv injection through
kv_proj_only / apply_k_norm / apply_k_rope). These drafts use DeepSeek-style MLA: q_a -> q_a_layernorm -> q_b,
kv_a_proj_with_mqa -> (latent c, shared k_pe), kv_a_layernorm(c) -> kv_b -> per-head (k_nope, v). Attention over the
latent cache equals MHA over the expanded per-head keys/values:
    q_h = [q_nope_h, rope(q_pe_h)],  k_h = [W_uk_h c, rope(k_pe)],  v_h = W_uv_h c,  scale = (d_nope + d_rope)^-0.5 * m^2
so this module runs the expanded form on the existing MHA machinery (draft KV pool, attention backends, ENCODER_ONLY
non-causal block attention, DSPARK/DFLASH kv injection) without a new kernel. q/k (192) and v (128) are zero-padded to
one head size of 256 (a size every MHA backend supports); the padding adds nothing to q.k and the output keeps the first
v_head_dim columns, so the result is exact. Cost: the draft KV is 2 x 256 per head per token (vs 576 latent), paid
only for the draft's own layers (sized by the draft window ring for DSPARK).
vLLM reference read for the math (vllm/models/kimi_k3/nvidia/{mla,dspark_mla}.py, image vllm-openai:kimi-k3): RoPE is
interleaved (is_neox_style False), DeepSeek YaRN with softmax scale (d_qk)^-0.5 * mscale(factor, mscale_all_dim)^2,
fp32 cos/sin; the confidence head is training-only there (skipped), the Markov head is the vanilla low-rank head.
Checkpoint naming: Inferact uses context_proj / context_norm / final_norm for SGLang's fc / hidden_norm / norm.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.model_config import compute_mla_mscale_scaling
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dflash import DFlashDecoderLayer, _get_dflash_layer_attention_params
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)

PAD_HEAD_DIM = 256
INFERACT_RENAMES = (("context_proj.", "fc."), ("context_norm.", "hidden_norm."), ("final_norm.", "norm."))


def _cfg(config, key, default=None):
    v = getattr(config, key, None)
    return default if v is None else v


def _dflash_cfg(config) -> dict:
    d = getattr(config, "dflash_config", None) or {}
    return d if isinstance(d, dict) else dict(vars(d))


def is_mla_draft_config(config) -> bool:
    """The draft configs this module serves (also used by the model_config hook to size the MHA pool)."""
    archs = list(getattr(config, "architectures", None) or [])
    return "K3DSparkModel" in archs or _dflash_cfg(config).get("attention_mode") == "mla"


def draft_is_causal(config) -> Optional[bool]:
    """Block-internal causality: Block5 says dflash_config.causal / full_attention_causal true; the DFlash2 draft says
    is_causal false; None = the DFlash default (non-causal block, ENCODER_ONLY)."""
    d = _dflash_cfg(config)
    for v in (d.get("causal"), getattr(config, "full_attention_causal", None), getattr(config, "is_causal", None)):
        if v is not None:
            return bool(v)
    return None


def mla_rope_and_scale(config) -> Tuple[float, Optional[dict], float]:
    d_qk = int(config.qk_nope_head_dim) + int(config.qk_rope_head_dim)
    theta, scaling = get_rope_config(config)
    scale = d_qk ** -0.5
    rs = None
    if scaling and (scaling.get("rope_type") or scaling.get("type")) not in (None, "default"):
        rs = dict(scaling)
        rs.pop("type", None)
        rs["rope_type"] = "deepseek_yarn"
        if rs.get("apply_yarn_scaling", True):
            scale = compute_mla_mscale_scaling(rs, scale)
    return float(theta), rs, float(scale)


class MLAExpandedAttention(nn.Module):
    """DeepSeek MLA (q-LoRA) run as exact expanded MHA with heads padded to PAD_HEAD_DIM."""

    def __init__(self, config, layer_id: int, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        tp = int(get_parallel().tp_size)
        hidden = int(config.hidden_size)
        heads = int(config.num_attention_heads)
        assert heads % tp == 0, f"MLA draft heads {heads} not divisible by tp {tp}"
        self.num_heads = heads // tp
        self.num_kv_heads = self.num_heads
        self.d_nope = int(config.qk_nope_head_dim)
        self.d_rope = int(config.qk_rope_head_dim)
        self.d_v = int(config.v_head_dim)
        self.d_qk = self.d_nope + self.d_rope
        assert self.d_qk <= PAD_HEAD_DIM and self.d_v <= PAD_HEAD_DIM
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        eps = float(_cfg(config, "rms_norm_eps", 1e-6))
        self.head_dim = PAD_HEAD_DIM                      # what the MHA pool / backends see
        self.q_size = self.num_heads * PAD_HEAD_DIM
        self.kv_size = self.num_kv_heads * PAD_HEAD_DIM
        self.v_scale = None
        self.attention_sink_bias = None
        p = (prefix + ".") if prefix else ""
        self.q_a_proj = ReplicatedLinear(hidden, self.q_lora_rank, bias=False, prefix=p + "q_a_proj")
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=eps)
        self.q_b_proj = ColumnParallelLinear(self.q_lora_rank, heads * self.d_qk, bias=False, prefix=p + "q_b_proj")
        self.kv_a_proj_with_mqa = ReplicatedLinear(hidden, self.kv_lora_rank + self.d_rope, bias=False,
                                                   prefix=p + "kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=eps)
        self.kv_b_proj = ColumnParallelLinear(self.kv_lora_rank, heads * (self.d_nope + self.d_v), bias=False,
                                              prefix=p + "kv_b_proj")
        self.o_proj = RowParallelLinear(heads * self.d_v, hidden, bias=False, prefix=p + "o_proj")
        theta, rs, self.scaling = mla_rope_and_scale(config)
        self.rotary_emb = get_rope(self.d_rope, rotary_dim=self.d_rope,
                                   max_position=int(_cfg(config, "max_position_embeddings", 32768)), base=theta,
                                   rope_scaling=rs, is_neox_style=False)
        self.sliding_window_size, attn_type = _get_dflash_layer_attention_params(config, layer_id)
        causal = draft_is_causal(config)
        if causal is not None:
            attn_type = AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY
        self.attn_type = attn_type
        self._init_attn(layer_id)

    def _init_attn(self, layer_id: int) -> None:
        self.attn = RadixAttention(num_heads=self.num_heads, head_dim=PAD_HEAD_DIM, scaling=self.scaling,
                                   num_kv_heads=self.num_kv_heads, layer_id=layer_id, v_head_dim=PAD_HEAD_DIM,
                                   sliding_window_size=self.sliding_window_size, attn_type=self.attn_type)

    # -- expanded projections (pure torch; shared by forward and the kv-injection API) --
    def _latent(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kv_a, _ = self.kv_a_proj_with_mqa(hidden_states)
        c, k_pe = kv_a.split([self.kv_lora_rank, self.d_rope], dim=-1)
        return self.kv_a_layernorm(c.contiguous()), k_pe

    def _expand_kv(self, c: torch.Tensor, k_pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[T, Hl, PAD] k = [k_nope | k_pe (unrotated) | 0], v = [v | 0]."""
        t = c.shape[0]
        kv, _ = self.kv_b_proj(c)
        kv = kv.view(t, self.num_heads, self.d_nope + self.d_v)
        k = c.new_zeros(t, self.num_heads, PAD_HEAD_DIM)
        v = c.new_zeros(t, self.num_heads, PAD_HEAD_DIM)
        k[..., :self.d_nope] = kv[..., :self.d_nope]
        k[..., self.d_nope:self.d_qk] = k_pe.view(t, 1, self.d_rope)
        v[..., :self.d_v] = kv[..., self.d_nope:]
        return k, v

    def _rope_k(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Rotate the shared k_pe once and broadcast it to every head (in place on the padded layout)."""
        k_pe = k[:, 0:1, self.d_nope:self.d_qk].contiguous()
        dummy_q = torch.zeros_like(k_pe)
        _, k_pe = self.rotary_emb(positions, dummy_q, k_pe)
        k[..., self.d_nope:self.d_qk] = k_pe.view(-1, 1, self.d_rope)
        return k

    def _query(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        t = hidden_states.shape[0]
        q_c, _ = self.q_a_proj(hidden_states)
        q, _ = self.q_b_proj(self.q_a_layernorm(q_c))
        q = q.view(t, self.num_heads, self.d_qk)
        q_pe = q[..., self.d_nope:].contiguous()
        q_pe, _ = self.rotary_emb(positions, q_pe, torch.zeros_like(q_pe[:, :1, :]))
        out = q.new_zeros(t, self.num_heads, PAD_HEAD_DIM)
        out[..., :self.d_nope] = q[..., :self.d_nope]
        out[..., self.d_nope:self.d_qk] = q_pe.view(t, self.num_heads, self.d_rope)
        return out

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        t = hidden_states.shape[0]
        if t == 0:
            return hidden_states.new_zeros(0, hidden_states.shape[-1])
        q = self._query(positions, hidden_states)
        c, k_pe = self._latent(hidden_states)
        k, v = self._expand_kv(c, k_pe)
        k = self._rope_k(positions, k)
        o = self.attn(q.view(t, -1), k.view(t, -1), v.view(t, -1), forward_batch)
        o = o.view(t, self.num_heads, PAD_HEAD_DIM)[..., :self.d_v].reshape(t, self.num_heads * self.d_v)
        out, _ = self.o_proj(o)
        return out

    # -- the DSPARK / DFLASH context-KV API (dspark.write_target_hidden_kv, dflash_worker_v2) --
    def kv_proj_only(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c, k_pe = self._latent(hidden_states)
        k, v = self._expand_kv(c, k_pe)
        t = hidden_states.shape[0]
        return k.view(t, -1), v.view(t, -1)

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        return k   # MLA normalizes the latent (kv_a_layernorm) inside kv_proj_only; there is no per-head k norm

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        t = k.shape[0]
        return self._rope_k(positions, k.view(t, self.num_heads, PAD_HEAD_DIM)).view(t, -1)


class MLALatentAttention(MLAExpandedAttention):
    """DeepSeek MLA run in absorbed (latent) form as MQA on the existing MHA draft pool and backends.

    One KV head per rank: k = [kv_a_layernorm(c) | rope(k_pe)] (kv_lora + d_rope = 576), v = c (kv_lora = 512); every
    query head is absorbed through W_uk (q_lat = W_uk_h^T q_nope_h, 512) and keeps its rotated q_pe (64), so
    q_lat . c + q_pe . k_pe = q_nope . k_nope + q_pe . k_pe exactly; the attention output (512 per head) goes through
    W_uv_h. Draft KV = (576 + 512) per token per layer per rank, independent of the head count (expanded: 2 x 256 per
    local head = 4,096 at 8 heads). Head dims 576/512 are the Triton backend's MLA tiles (BLOCK_DMODEL 512 + DPE 64);
    FlashInfer FA2 / trtllm_mha stop at 256, so the latent layout requires --speculative-draft-attention-backend triton.
    """

    def _init_attn(self, layer_id: int) -> None:
        self.num_kv_heads = 1
        self.head_dim = self.kv_lora_rank + self.d_rope       # 576: what the pool / backend see for k (and q)
        self.v_head_dim = self.kv_lora_rank                    # 512
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.head_dim
        self.attn = RadixAttention(num_heads=self.num_heads, head_dim=self.head_dim, scaling=self.scaling,
                                   num_kv_heads=1, layer_id=layer_id, v_head_dim=self.v_head_dim,
                                   sliding_window_size=self.sliding_window_size, attn_type=self.attn_type)

    def _w_uk_uv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Views of the local kv_b weight [Hl * (d_nope + d_v), kv_lora]: W_uk [Hl, d_nope, L], W_uv [Hl, d_v, L]."""
        w = self.kv_b_proj.weight
        if w.dim() != 2 or w.shape[-1] != self.kv_lora_rank:
            raise RuntimeError(f"MLA latent draft needs an unquantized kv_b_proj weight [H*(dn+dv), {self.kv_lora_rank}]"
                               f", got {tuple(w.shape)} {w.dtype}; run the draft with "
                               "--speculative-draft-model-quantization unquant")
        w = w.view(self.num_heads, self.d_nope + self.d_v, self.kv_lora_rank)
        return w[:, :self.d_nope, :], w[:, self.d_nope:, :]

    def _latent_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[T, 576] = [c | k_pe (unrotated)]."""
        c, k_pe = self._latent(hidden_states)
        return torch.cat([c, k_pe.to(c.dtype)], dim=-1)

    def _rope_latent_k(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        k = k.view(-1, 1, self.head_dim)
        k_pe = k[..., self.kv_lora_rank:].contiguous()
        _, k_pe = self.rotary_emb(positions, torch.zeros_like(k_pe), k_pe)
        out = k.clone()
        out[..., self.kv_lora_rank:] = k_pe.view(-1, 1, self.d_rope)
        return out.view(-1, self.head_dim)

    def _query_latent(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        t = hidden_states.shape[0]
        q_c, _ = self.q_a_proj(hidden_states)
        q, _ = self.q_b_proj(self.q_a_layernorm(q_c))
        q = q.view(t, self.num_heads, self.d_qk)
        q_pe = q[..., self.d_nope:].contiguous()
        q_pe, _ = self.rotary_emb(positions, q_pe, torch.zeros_like(q_pe[:, :1, :]))
        w_uk, _ = self._w_uk_uv()
        q_lat = torch.bmm(q[..., :self.d_nope].transpose(0, 1).to(w_uk.dtype), w_uk)   # [Hl, T, L]
        out = q.new_empty(t, self.num_heads, self.head_dim)
        out[..., :self.kv_lora_rank] = q_lat.transpose(0, 1)
        out[..., self.kv_lora_rank:] = q_pe.view(t, self.num_heads, self.d_rope)
        return out

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        t = hidden_states.shape[0]
        if t == 0:
            return hidden_states.new_zeros(0, hidden_states.shape[-1])
        q = self._query_latent(positions, hidden_states)
        k = self._rope_latent_k(positions, self._latent_kv(hidden_states))
        v = k[:, :self.kv_lora_rank].contiguous()
        o = self.attn(q.view(t, -1), k, v, forward_batch)                      # [T, Hl * 512]
        _, w_uv = self._w_uk_uv()
        o = torch.bmm(o.view(t, self.num_heads, self.kv_lora_rank).transpose(0, 1).to(w_uv.dtype),
                      w_uv.transpose(1, 2))                                     # [Hl, T, d_v]
        out, _ = self.o_proj(o.transpose(0, 1).reshape(t, self.num_heads * self.d_v))
        return out

    # -- the DSPARK / DFLASH context-KV API: k [T, 576] (rotated by apply_k_rope), v [T, 512] (the latent c) --
    def kv_proj_only(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self._latent_kv(hidden_states)
        return k, k[:, :self.kv_lora_rank].contiguous()

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return self._rope_latent_k(positions, k)


class MLAExpandedDecoderLayer(DFlashDecoderLayer):
    attention_cls = MLAExpandedAttention


class MLALatentDecoderLayer(DFlashDecoderLayer):
    attention_cls = MLALatentAttention


def mla_draft_layout() -> str:
    """NUMINOUS_K3_MLA_DRAFT_LAYOUT: "latent" (default; MQA over the 576/512 latent, Triton draft backend) or
    "expanded" (per-head 256-padded MHA, any backend with head_dim 256; 3.8-5.6x the draft KV)."""
    v = os.environ.get("NUMINOUS_K3_MLA_DRAFT_LAYOUT", "latent").strip().lower() or "latent"
    if v not in ("latent", "expanded"):
        raise ValueError(f"NUMINOUS_K3_MLA_DRAFT_LAYOUT={v!r}: expected latent or expanded")
    return v


def mla_decoder_layer_cls():
    return MLALatentDecoderLayer if mla_draft_layout() == "latent" else MLAExpandedDecoderLayer


def mla_pool_dims(config) -> Tuple[int, int, int]:
    """(head_dim, v_head_dim, kv_heads or 0 = keep the config's) the draft MHA pool must be sized with."""
    if mla_draft_layout() == "latent":
        return int(config.kv_lora_rank) + int(config.qk_rope_head_dim), int(config.kv_lora_rank), 1
    return PAD_HEAD_DIM, PAD_HEAD_DIM, 0


def check_attention_loaded(model: nn.Module, loaded: set) -> None:
    """DFlash's loader ignores unknown names; an MLA weight that fails to map would silently stay at init."""
    need = {n for n, _ in model.named_parameters() if ".self_attn." in n and "rotary" not in n}
    missing = sorted(need - set(loaded))
    if missing:
        raise ValueError(f"MLA draft: attention weights not loaded from the checkpoint: {missing[:6]} "
                         f"(+{max(0, len(missing) - 6)} more)")


class K3DSparkModel(DSparkDraftModel):
    """Inferact Kimi-K3 DSpark drafts (vLLM-trained, MLA) on SGLang's DSPARK worker."""

    decoder_layer_cls = MLALatentDecoderLayer   # rebound per instance from NUMINOUS_K3_MLA_DRAFT_LAYOUT
    supports_fused_context_kv = False
    supports_pre_gather_target_hidden_projection = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        # vLLM/speculators write num_target_layers = the number of captured feature layers (5 / 3); SGLang reads it as
        # the target's layer count and range-checks target_layer_ids against it (dflash_utils.resolve_target_layer_ids)
        ids = list(getattr(config, "target_layer_ids", None) or _dflash_cfg(config).get("target_layer_ids") or [])
        ntl = getattr(config, "num_target_layers", None)
        if ids and ntl is not None and int(ntl) <= max(ids):
            config.num_target_layers = int(getattr(config, "target_num_hidden_layers", None) or max(ids) + 1)
        self.decoder_layer_cls = mla_decoder_layer_cls()
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

    def _build_fused_kv_write_bundle(self, pool):
        return None   # the fused writer assumes Qwen-style q/k/v + k_norm; the per-layer path handles MLA

    def _stacked_ctx_kv_params(self):
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        def renamed():
            for name, w in weights:
                n = name.removeprefix("model.")
                for a, b in INFERACT_RENAMES:
                    if n.startswith(a):
                        n = b + n[len(a):]
                        break
                yield n, w
        tracked = []

        def tap():
            for n, w in renamed():
                tracked.append(n)
                yield n, w
        out = super().load_weights(tap())
        check_attention_loaded(self, {n for n in tracked})
        return out


EntryClass = [K3DSparkModel]
