"""Anchored hooks for serving AMD Quark mixed checkpoints (MXFP4 routed experts, per-block FP8 attention; e.g.
amd/GLM-5.3-Quark-MXFP4-AttnFP8) on an installed SGLang. Same contract as sglang_numinous_hooks.py: each hook names
its file, an exact anchor and the lines inserted after it; a missing anchor fails; hooks are idempotent (marker).

    python3 rocm_quark_fused_hooks.py            # patch the installed sglang package
    python3 rocm_quark_fused_hooks.py --check    # report which anchors resolve, change nothing

What breaks without them (sglang 0.5.20.dev20260923, ROCm image, measured 2026-09-28):

  quark_scale_rename  The checkpoint names every FP8 block scale `<module>.weight_scale`. QuarkConfig routes those
                      layers to Fp8LinearMethod(block) which registers `<module>.weight_scale_inv`. The DeepSeek
                      weight loader has no rename, so kv_b_proj/o_proj/q_b_proj/indexer.wk/indexer.wq_b scales are
                      logged "not found in params_dict" and skipped (FP8 weights served with no scale -> garbage),
                      and the fused q_a/kv_a branch raises KeyError on `fused_qkv_a_proj_with_mqa.weight_scale`.
  quark_block_size    QuarkConfig.weight_block_size stays None (the FP8 block size lives per layer in
                      layer_quant_config). post_load_weights reads the model-level value to choose the block
                      dequant path for kv_b_proj; None sends a [out/128, in/128] scale into
                      channel_quant_to_tensor_quant, which expects a 1-D per-channel scale.
  mxfp4_even_k        (measured 2026-09-29, agent amdopt) AITER's Triton gemm_afp4wfp4 on gfx950 returns NaN rows
                      when its config leaves K not a multiple of BLOCK_SIZE_K (the masked, non-EVEN_K path) and other
                      kernels share the CUs: GLM-5.3's TP8 dense down_proj (K=1536) gets BLOCK_SIZE_K=1024 at M=12..32
                      -> NaN logits at 16/32 concurrent decodes on the DSA sparse path -> "!!!!" output and the
                      HSA 0x1016 fault under sampling. Stress (rocm_mxfp4_even_k_check.py): default config 600/600 calls
                      NaN under a concurrent side-stream GEMM, EVEN_K config 0/600. The hook passes an EVEN_K config
                      (NUM_KSPLIT 1, largest power-of-two BLOCK_SIZE_K dividing K) whenever the default would not be.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Hook:
    name: str
    group: str
    path: str  # relative to the sglang package directory
    anchor: str  # exact text; the insertion goes after it
    text: str

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


SCALE_RENAME = '''
                # numinous-hook:quark_scale_rename
                # Quark FP8 block checkpoints name the scale `.weight_scale`; Fp8LinearMethod(block) registers
                # `.weight_scale_inv`. Rename only when the `_inv` target exists: directly, through the fused
                # q_a/kv_a projection, or through the fused bf16 indexer wk (which dequantizes from weight+scale).
                # MXFP4 layers (QuarkLinearMethod, stacked dense gate_up_proj, routed experts) keep `.weight_scale`.
                if name.endswith(".weight_scale") and name not in params_dict:
                    _inv = name[: -len(".weight_scale")] + ".weight_scale_inv"
                    # one conditional replace, as the fused branch below does: "fused_qkv_a_proj_with_mqa" itself
                    # contains "kv_a_proj_with_mqa", so chaining both replaces mangles the q_a_proj name
                    if "q_a_proj" in name:
                        _fused_qkv = _inv.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                    elif "kv_a_proj_with_mqa" in name:
                        _fused_qkv = _inv.replace("kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa")
                    else:
                        _fused_qkv = None
                    _fused_wk = (
                        name.rsplit(".indexer.", 1)[0] + ".indexer.wk_weights_proj.weight"
                        if ".indexer.wk." in name or ".indexer.weights_proj." in name
                        else None
                    )
                    if _inv in params_dict or _fused_qkv in params_dict or _fused_wk in params_dict:
                        name = _inv
'''

BLOCK_SIZE = '''
        # numinous-hook:quark_block_size
        # Mixed Quark checkpoints carry the FP8 block size only per layer (layer_quant_config). post_load_weights
        # reads the model-level weight_block_size to pick the block-dequant path for kv_b_proj, so lift the single
        # layer-wide FP8 block size up to the config.
        if getattr(self, "weight_block_size", None) is None:
            _fp8_block_sizes = set()
            for _lc in (self.quant_config.get("layer_quant_config") or {}).values():
                _w = (_lc or {}).get("weight") or {}
                if _w.get("qscheme") == "per_block" and _w.get("dtype") in {"fp8_e4m3", "fp8_e4m3fn"}:
                    _bs = _w.get("block_size")
                    if isinstance(_bs, list) and len(_bs) == 2:
                        _fp8_block_sizes.add(tuple(_bs))
            if len(_fp8_block_sizes) == 1:
                self.weight_block_size = list(_fp8_block_sizes.pop())
'''

EVEN_K_HELPER = '''
    # numinous-hook:mxfp4_even_k_helper
    import functools as _nm_functools

    @_nm_functools.lru_cache(maxsize=None)
    def _numinous_even_k_config(M: int, N: int, K: int):
        """None when AITER's own config for (M, N, K packed) already takes the EVEN_K path, else a config that does:
        the masked non-EVEN_K path of the gfx950 Triton kernel returns NaN rows under CU sharing (amdopt 2026-09-29)."""
        import triton as _nm_triton
        from aiter.ops.triton.gemm.basic import gemm_afp4wfp4 as _nm_g
        cfg = dict(_nm_g._get_config(M, N, K)[0])
        bk, ks = cfg["BLOCK_SIZE_K"], cfg["NUM_KSPLIT"]
        if ks > 1:
            sbs, bk, ks = _nm_g.get_splitk(K, bk, ks)
        else:
            sbs = 2 * K
        if bk >= 2 * K:
            bk, sbs = _nm_triton.next_power_of_2(2 * K), 2 * K
        if K % (bk // 2) == 0 and sbs % bk == 0 and K % (sbs // 2) == 0:
            return None
        b = cfg["BLOCK_SIZE_K"]
        while b > 64 and K % (b // 2):
            b //= 2
        if K % (b // 2):
            return None
        cfg.update(NUM_KSPLIT=1, BLOCK_SIZE_K=b)
        return cfg
'''

EVEN_K_CALL = '''        # numinous-hook:mxfp4_even_k_call
        _nm_cfg = _numinous_even_k_config(x.shape[0], w.shape[0], x.shape[1])
        if _nm_cfg is not None:
            _gemm_afp4wfp4_orig(x, w, x_scales, w_scales, y.dtype, y, config=dict(_nm_cfg))
            return
'''

HOOKS = [
    Hook("quark_scale_rename", "quark", "srt/models/deepseek_common/deepseek_weight_loader.py",
         '                if "rotary_emb.inv_freq" in name:\n'
         "                    continue\n",
         SCALE_RENAME.lstrip("\n")),
    Hook("quark_block_size", "quark", "srt/layers/quantization/quark/quark.py",
         "        if isinstance(self.dequantization_config, Fp8Config):\n"
         "            self.weight_block_size = self.dequantization_config.weight_block_size\n",
         BLOCK_SIZE.lstrip("\n")),
    Hook("mxfp4_even_k_helper", "mxfp4", "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py",
         "    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4 as _gemm_afp4wfp4_orig\n",
         EVEN_K_HELPER.lstrip("\n")),
    Hook("mxfp4_even_k_call", "mxfp4", "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py",
         "    def _aiter_gemm_afp4wfp4(\n"
         "        x: torch.Tensor,\n"
         "        w: torch.Tensor,\n"
         "        x_scales: torch.Tensor,\n"
         "        w_scales: torch.Tensor,\n"
         "        y: torch.Tensor,\n"
         "    ) -> None:\n",
         EVEN_K_CALL),
]


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, groups: list[str], check: bool = False) -> list[str]:
    """Returns one status line per selected hook; raises SystemExit if an anchor is missing."""
    out, missing = [], []
    for h in HOOKS:
        if groups and h.group not in groups and h.name not in groups:
            continue
        path = os.path.join(root, h.path)
        src = open(path).read()
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        new = src.replace(h.anchor, h.anchor + h.text, 1)
        if check:
            out.append(f"{h.name}: anchor ok")
            continue
        compile(new, path, "exec")  # never leave a file that does not parse
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("rocm quark hooks: " + "; ".join(missing))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", help="hook groups or names (default: all)")
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.groups, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
