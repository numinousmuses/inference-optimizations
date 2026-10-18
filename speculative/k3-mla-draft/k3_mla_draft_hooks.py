"""MLA speculative drafts for Kimi-K3 on SGLang v0.5.20 (agent k3b300, coordinator 17:58): Inferact/Kimi-K3-DSpark,
Inferact/Kimi-K3-DSpark-Block5 (K3DSparkModel) and lightseekorg/kimi-k3-dflash2 (DFlash2DraftModel, attention_mode mla).

Anchored edits, all inert for every other draft:
  1. srt/models/k3_mla_draft.py (new; source operations/patches/k3_mla_draft/k3_mla_draft.py): the MLA attention in two
     layouts (NUMINOUS_K3_MLA_DRAFT_LAYOUT): "latent" (default; absorbed MLA as MQA on the MHA draft pool, one KV head,
     k = [c | rope k_pe] 576, v = c 512, Triton draft backend) or "expanded" (per-head 256-padded MHA), plus the
     K3DSparkModel draft class (EntryClass) with the Inferact weight renames.
  2. srt/configs/model_config.py: MLA drafts get the layout's pool dims (latent: head_dim 576, v_head_dim 512,
     num_key_value_heads 1; expanded: 256/256) and AttentionArch.MHA; without it the pool is sized from hidden/heads.
  3. srt/models/dflash.py: DFlashDraftModel builds MLA decoder layers when dflash_config.attention_mode == "mla" and
     then turns off the Qwen-style fused context-KV writer (it slices a q/k/v weight MLA does not have).
  4. srt/utils/hf_transformers/config.py: registers model_type "k3_dspark" (Inferact's configs) with transformers'
     AutoConfig as a plain PretrainedConfig; AutoConfig rejects unknown model types.
  5. srt/models/dspark.py + srt/speculative/dflash_worker_v2.py: the context-KV injection views v with the layer's
     v_head_dim (latent v is 512 wide, k 576); identical for every draft without a v_head_dim attribute.
  6. srt/mem_cache/memory_pool.py: MHA set_kv_buffer_prefix_valid (the DSPARK/DFLASH commit injection) writes
     asymmetric K/V rows with one tiled launch per tensor at its own width (stock raises NotImplementedError).
  7. srt/layers/attention/triton_backend.py: an MLA draft's Triton backend ignores DCP, as the HIP DSpark draft's
     already does (drafts never join the DCP group; their pool is replicated and indexed by raw virtual locs).
Same contract as the other patchers: exact anchors (once each), marker -> re-application is a no-op, every patched
file must compile.   python3 k3_mla_draft_hooks.py [--root <sglang pkg>] [--check]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
MARKER = "# numinous-hook:k3_mla_draft"
MODULE_SRC = HERE / "k3_mla_draft" / "k3_mla_draft.py"
MODULE_DST = "srt/models/k3_mla_draft.py"

CFG = "srt/configs/model_config.py"
CFG_ANCHOR = '        elif (\n            "DeepseekV4ForCausalLM" in self.hf_config.architectures\n'
CFG_TEXT = ('        elif _numinous_is_mla_draft(self.hf_config):  ' + MARKER + '\n'
            '            # MLA drafts run on the MHA draft pool (models/k3_mla_draft.py): latent MQA or padded expanded heads\n'
            '            _numinous_mla_draft_dims(self)\n'
            '            self.attention_arch = AttentionArch.MHA\n' + CFG_ANCHOR)
CFG_TAIL = f'''


{MARKER}
def _numinous_is_mla_draft(hf_config) -> bool:
    archs = list(getattr(hf_config, "architectures", None) or [])
    d = getattr(hf_config, "dflash_config", None) or {{}}
    if not isinstance(d, dict):
        d = dict(vars(d))
    return "K3DSparkModel" in archs or d.get("attention_mode") == "mla"


def _numinous_mla_draft_dims(mc) -> None:
    """Pool geometry of the NUMINOUS_K3_MLA_DRAFT_LAYOUT (mirrors k3_mla_draft.mla_pool_dims; kept import-free)."""
    hf = mc.hf_text_config
    layout = os.environ.get("NUMINOUS_K3_MLA_DRAFT_LAYOUT", "latent").strip().lower() or "latent"
    if layout == "latent":
        hd, vd = int(hf.kv_lora_rank) + int(hf.qk_rope_head_dim), int(hf.kv_lora_rank)
        hf.num_key_value_heads = 1     # one latent KV head per rank (MQA); read below into num_key_value_heads
    elif layout == "expanded":
        hd = vd = 256
    else:
        raise ValueError(f"NUMINOUS_K3_MLA_DRAFT_LAYOUT={{layout!r}}: expected latent or expanded")
    mc.head_dim, mc.v_head_dim, mc.swa_head_dim, mc.swa_v_head_dim = hd, vd, hd, vd
'''

DFL = "srt/models/dflash.py"
DFL_ANCHOR = '                self.decoder_layer_cls(\n                    config=config,\n                    layer_id=i,\n'
DFL_TEXT = ('                _numinous_decoder_layer_cls(self, config)(  ' + MARKER + '\n'
            '                    config=config,\n                    layer_id=i,\n')
DFL_TAIL = f'''


{MARKER}
def _numinous_decoder_layer_cls(model, config):
    """DFlash drafts with dflash_config.attention_mode == "mla" (lightseekorg/kimi-k3-dflash2) use the MLA layer of
    NUMINOUS_K3_MLA_DRAFT_LAYOUT; their fused context-KV writer is off (it slices a Qwen q/k/v weight)."""
    d = getattr(config, "dflash_config", None) or {{}}
    if not isinstance(d, dict):
        d = dict(vars(d))
    if d.get("attention_mode") == "mla":
        from sglang.srt.models.k3_mla_draft import mla_decoder_layer_cls
        model.supports_fused_context_kv = False
        return mla_decoder_layer_cls()
    return model.decoder_layer_cls
'''

HFC = "srt/utils/hf_transformers/config.py"
HFC_ANCHOR = '        if config is None:\n            config = _try_load_raw_mamba_config(model, revision, **kwargs)\n'
HFC_TEXT = HFC_ANCHOR + '        _numinous_register_k3_dspark_config()  ' + MARKER + '\n'
HFC_TAIL = f'''


{MARKER}
def _numinous_register_k3_dspark_config() -> None:
    """Inferact's Kimi-K3 DSpark configs say model_type "k3_dspark", which transformers does not know."""
    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "k3_dspark" in CONFIG_MAPPING:
        return

    class K3DSparkConfig(PretrainedConfig):
        model_type = "k3_dspark"

    AutoConfig.register("k3_dspark", K3DSparkConfig)
'''

V_VIEW = "v = v.view(-1, attn.num_kv_heads, attn.head_dim)\n"
V_VIEW_NEW = ('v = v.view(-1, attn.num_kv_heads, getattr(attn, "v_head_dim", attn.head_dim))  ' + MARKER + '\n')
DSP = "srt/models/dspark.py"
DFW = "srt/speculative/dflash_worker_v2.py"

MPL = "srt/mem_cache/memory_pool.py"
MPL_ANCHOR = ("        if self.v_row_dim != self.row_dim:\n            raise NotImplementedError(\n"
              '                "prefix-valid commit requires equal-width K/V rows, got "\n')
MPL_TEXT = ("        if self.v_row_dim != self.row_dim:  " + MARKER + "\n"
            "            # asymmetric rows (MLA latent draft: k 576, v 512): one tiled launch per tensor at its own width\n"
            "            for _src, _dst, _w in ((cache_k, self.k_buffer[layer_id - self.start_layer], self.row_dim),\n"
            "                                   (cache_v, self.v_buffer[layer_id - self.start_layer], self.v_row_dim)):\n"
            "                _set_kv_buffer_prefix_valid_impl(_src, _src, _dst, _dst, loc_2d, commit_lens, row_dim=_w,\n"
            "                                                 store_dtype=self.store_dtype)\n"
            "            return\n" + MPL_ANCHOR)

TRI = "srt/layers/attention/triton_backend.py"
TRI_ANCHOR = "        if self.is_hip_dspark_draft:\n            # Drafts never join the dcp group so we ignore it\n"
TRI_TEXT = ("        if self.is_hip_dspark_draft or _numinous_mla_draft_runner(model_runner):  " + MARKER + "\n"
            "            # Drafts never join the dcp group so we ignore it\n")
TRI_TAIL = f'''


{MARKER}
def _numinous_mla_draft_runner(model_runner) -> bool:
    """MLA drafts (K3DSparkModel, DFlash attention_mode mla) run on Triton with a replicated, raw-loc draft pool."""
    if not getattr(model_runner, "is_draft_worker", False):
        return False
    hf = model_runner.model_config.hf_config
    archs = list(getattr(hf, "architectures", None) or [])
    d = getattr(hf, "dflash_config", None) or {{}}
    if not isinstance(d, dict):
        d = dict(vars(d))
    return "K3DSparkModel" in archs or d.get("attention_mode") == "mla"
'''

# (path, anchor, replacement, tail appended once, expected anchor count)
EDITS = ((CFG, CFG_ANCHOR, CFG_TEXT, CFG_TAIL, 1), (DFL, DFL_ANCHOR, DFL_TEXT, DFL_TAIL, 1),
         (HFC, HFC_ANCHOR, HFC_TEXT, HFC_TAIL, 1), (DSP, V_VIEW, V_VIEW_NEW, "", 1), (DFW, V_VIEW, V_VIEW_NEW, "", 2),
         (MPL, MPL_ANCHOR, MPL_TEXT, "", 1), (TRI, TRI_ANCHOR, TRI_TEXT, TRI_TAIL, 1))


def patch_source(path: str, src: str) -> tuple[str, str]:
    for p, anchor, text, tail, count in EDITS:
        if p != path:
            continue
        if MARKER in src:
            return src, f"k3_mla_draft: {p} already applied"
        n = src.count(anchor)
        if n != count:
            raise SystemExit(f"k3_mla_draft_hooks: anchor found {n} times in {p} (expected {count})")
        new = src.replace(anchor, text)
        if tail:
            new = new.rstrip("\n") + "\n" + tail
        return new, f"k3_mla_draft: applied to {p}"
    raise KeyError(path)


def sglang_root() -> str:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def _write(full: str, new: str) -> None:
    tmp = f"{full}.numinous.{os.getpid()}"
    open(tmp, "w").write(new)
    os.replace(tmp, full)


def apply(root: str, check: bool = False) -> list[str]:
    lines = []
    for p, *_ in EDITS:
        full = os.path.join(root, p)
        src = open(full).read()
        new, line = patch_source(p, src)
        if check:
            lines.append(line.replace("applied to", "anchor ok in"))
            continue
        compile(new, full, "exec")
        if new != src:
            _write(full, new)
        lines.append(line)
    mod = MODULE_SRC.read_text()
    compile(mod, MODULE_DST, "exec")
    dst = os.path.join(root, MODULE_DST)
    if check:
        lines.append(f"k3_mla_draft: module {'present' if os.path.exists(dst) else 'to install'} at {MODULE_DST}")
    else:
        if not os.path.exists(dst) or open(dst).read() != mod:
            _write(dst, mod)
        lines.append(f"k3_mla_draft: module installed at {MODULE_DST}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
