"""Anchored patcher: native NVFP4 KV cache on SGLang v0.5.20's Triton attention backend (`--kv-cache-dtype nvfp4
--attention-backend triton`). Style follows operations/patches/sglang_numinous_hooks.py / gemma_extend_hooks.py.

Why (measured on RTX PRO 6000, operations/kernels/kv4tri_probe.py): v0.5.20's validator accepts nvfp4 + triton, but
the Triton backend reads the packed FP4 pool as bf16/fp8 (Triton rejects the float4 tensor: KeyError
'float4_e2m1fn_x2'), SWA pools (Gemma 4) are built without the NVFP4 method (allocation crashes: fill_cuda not
implemented for Float4_e2m1fn_x2), hybrid pools report half the value head dim, and the hybrid-SWA pool sizer
counts FP4 as one byte per element (no capacity gain even if it booted).

Hooks (each: file, exact anchor, inserted text; a missing or duplicated anchor fails loudly):
  triton_decode      TritonAttnBackend.forward_decode  -> numinous_kv4tri.forward_decode for NVFP4 pools
  triton_extend      TritonAttnBackend.forward_extend  -> numinous_kv4tri.forward_extend for NVFP4 pools
  triton_vhead       TritonAttnBackend.__init__ (plain MHA pool branch): value head dim from the pool, not the
                     packed buffer width
  hybrid_vhead       HybridLinearKVPool.get_v_head_dim: same (Qwen 3.8 / 3.6 full-attention pool)
  swa_vhead          SWAKVPool.get_v_head_dim: same (Gemma 4)
  quant_adapt        KVCacheConfigurator._build_fp4_quant_method: NVFP4 without the FlashInfer FP8 dequant
                     workspace when Triton serves both phases
  swa_quant          KVCacheConfigurator._build_hybrid_swa_kv_pool: NVFP4 methods for both SWA sub-pools
  swa_pool           SWAKVPool.__init__: hand those methods to the sub-pool constructors
  cell_workspace     DefaultPoolConfigurator._compute_cell_size: drop the workspace bytes the Triton path never
                     allocates (more KV tokens per GB)
  swa_sizing         HybridSWAPoolConfigurator.__init__: FP4 bytes per token (half a byte per element + one E4M3
                     scale per 16) for both pools
Modules installed:
  operations/kernels/kv4tri_nvfp4_attention.py -> sglang/kernels/ops/attention/numinous_kv4tri_attention.py
  operations/kernels/kv4tri_backend.py         -> sglang/srt/layers/attention/numinous_kv4tri.py
  operations/kernels/extend_attention_gqa.py   -> sglang/kernels/ops/attention/numinous_kv4tri_gqa.py (long prefill
                                                  chunks: gather-dequant workspace + GQA-packed FP8 kernel)

NUMINOUS_KV4TRI=0 turns every hook into a no-op at runtime.

Usage:
    python3 operations/patches/kv4tri_hooks.py                  # patch installed sglang
    python3 operations/patches/kv4tri_hooks.py --check          # check anchors, change nothing
    python3 operations/patches/kv4tri_hooks.py --root /tmp/sgl  # patch a copy (the sglang package directory)
Test: operations/patches/test_kv4tri_hooks.py (box: applies to a full copy of the installed package and imports).
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.join(_HERE, "..", "kernels")

MODULES = {
    os.path.join(_KERNELS, "kv4tri_nvfp4_attention.py"): "kernels/ops/attention/numinous_kv4tri_attention.py",
    os.path.join(_KERNELS, "kv4tri_backend.py"): "srt/layers/attention/numinous_kv4tri.py",
    os.path.join(_KERNELS, "extend_attention_gqa.py"): "kernels/ops/attention/numinous_kv4tri_gqa.py",
}

TRITON = "srt/layers/attention/triton_backend.py"
POOL = "srt/mem_cache/memory_pool.py"
SWA = "srt/mem_cache/swa_memory_pool.py"
CONFIGURATOR = "srt/mem_cache/kv_cache_configurator.py"
SIZER = "srt/model_executor/pool_configurator.py"

HOOKS = [
    {
        "name": "triton_decode",
        "path": TRITON,
        "anchor": ("        # During torch.compile, there is a bug in rotary_emb that causes the\n"
                   "        # output value to have a 3D tensor shape. This reshapes the output correctly.\n"
                   "        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)\n"),
        "text": ("        # numinous-hook:triton_decode\n"
                 "        if getattr(self, \"_numinous_kv4tri\", None) is not False:\n"
                 "            from sglang.srt.layers.attention import numinous_kv4tri as _nk4\n"
                 "\n"
                 "            _nk4_o = _nk4.forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache,\n"
                 "                                         sinks, score_mod, aux_tensors)\n"
                 "            if _nk4_o is not None:\n"
                 "                return _nk4_o\n"),
    },
    {
        # anchor = forward_extend's signature tail + the head of its first statement (the signature tail alone is
        # not unique); the text goes between them ("at" = offset into the anchor)
        "name": "triton_extend",
        "path": TRITON,
        "anchor": ("        aux_tensors=None,\n    ):\n        if (\n            k is not None\n"
                   "            and v is not None\n            and sinks is None\n"),
        "at": len("        aux_tensors=None,\n    ):\n"),
        "text": ("        # numinous-hook:triton_extend\n"
                 "        if getattr(self, \"_numinous_kv4tri\", None) is not False:\n"
                 "            from sglang.srt.layers.attention import numinous_kv4tri as _nk4\n"
                 "\n"
                 "            _nk4_o = _nk4.forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache,\n"
                 "                                         sinks, score_mod, aux_tensors)\n"
                 "            if _nk4_o is not None:\n"
                 "                return _nk4_o\n"),
    },
    {
        "name": "triton_vhead",
        "path": TRITON,
        "anchor": ("            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(\n"
                   "                model_runner.token_to_kv_pool.start_layer\n"
                   "            ).shape[-1]\n"),
        "text": ("            # numinous-hook:triton_vhead\n"
                 "            if getattr(model_runner.token_to_kv_pool, \"is_quantized_kv_cache\", False):\n"
                 "                self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()\n"),
    },
    {
        "name": "hybrid_vhead",
        "path": POOL,
        "anchor": ("                mamba_cpu, self._mamba_translate(mamba_indices)\n            )\n\n"
                   "    def get_v_head_dim(self):\n"),
        "text": ("        # numinous-hook:hybrid_vhead\n"
                 "        if getattr(self.full_kv_pool, \"is_quantized_kv_cache\", False):\n"
                 "            return self.full_kv_pool.get_v_head_dim()\n"),
    },
    {
        "name": "swa_vhead",
        "path": SWA,
        "anchor": "        # and because layer 0 need not be a full-attention layer.\n",
        "text": ("        # numinous-hook:swa_vhead\n"
                 "        if getattr(self.full_kv_pool, \"is_quantized_kv_cache\", False):\n"
                 "            return self.full_kv_pool.get_v_head_dim()\n"),
    },
    {
        "name": "quant_adapt",
        "path": CONFIGURATOR,
        "anchor": "        quant_method.load_scales_from_model(self.model)\n",
        "text": ("        # numinous-hook:quant_adapt\n"
                 "        from sglang.srt.layers.attention import numinous_kv4tri as _nk4\n"
                 "\n"
                 "        quant_method = _nk4.adapt_quant_method(quant_method, self.server_args)\n"),
    },
    {
        "name": "swa_quant",
        "path": CONFIGURATOR,
        "anchor": ("        size_swa = (\n            full_max_total_num_tokens\n"
                   "            if self.draft_swa_full_capacity\n            else swa_max_total_num_tokens\n"
                   "        )\n"),
        "text": ("        # numinous-hook:swa_quant\n"
                 "        if is_float4_e2m1fn_x2(self.kv_cache_dtype) and swa_pool_class is mha_pool_class:\n"
                 "            from sglang.srt.layers.attention import numinous_kv4tri as _nk4\n"
                 "\n"
                 "            if _nk4.enabled():\n"
                 "                kwargs.update(_nk4.swa_quant_kwargs(self, swa_attention_layer_ids,\n"
                 "                                                    full_attention_layer_ids))\n"),
    },
    {
        "name": "swa_pool",
        "path": SWA,
        "anchor": ('                "allocation_label": "SWA",\n                **kwargs,\n            }\n'),
        "text": ("        # numinous-hook:swa_pool\n"
                 "        _nk4_qms = kwargs.get(\"numinous_quant_methods\")\n"
                 "        full_kv_pool_kwargs = {k: v for k, v in full_kv_pool_kwargs.items()\n"
                 "                               if k != \"numinous_quant_methods\"}\n"
                 "        swa_kv_pool_kwargs = {k: v for k, v in swa_kv_pool_kwargs.items()\n"
                 "                              if k != \"numinous_quant_methods\"}\n"
                 "        if _nk4_qms is not None:\n"
                 "            full_kv_pool_kwargs[\"quant_method\"], swa_kv_pool_kwargs[\"quant_method\"] = _nk4_qms\n"),
    },
    {
        "name": "cell_workspace",
        "path": SIZER,
        "anchor": ("                # FP4 prefill uses one shared FP8 dequant workspace across layers.\n"
                   "                cell_size += n * k * 2 * kv_size\n"),
        "text": ("                # numinous-hook:cell_workspace\n"
                 "                from sglang.srt.layers.attention import numinous_kv4tri as _nk4\n"
                 "\n"
                 "                cell_size -= _nk4.workspace_bytes_per_token(kvc.server_args, n, k, kv_size)\n"),
    },
    {
        "name": "swa_sizing",
        "path": SIZER,
        "anchor": ("            self._swa_per_token += (\n                model_config.get_swa_num_kv_heads(tp_size)\n"
                   "                * (model_config.swa_head_dim + model_config.swa_v_head_dim)\n"
                   "            ) // scale_block_size\n"),
        "text": ("        # numinous-hook:swa_sizing\n"
                 "        if is_float4_e2m1fn_x2(kv_cache_dtype) and model_config.attention_arch != AttentionArch.MLA:\n"
                 "            # per element: half a byte of E2M1 + 1/16 byte of E4M3 block scale (kv_size is 1 for FP4)\n"
                 "            self._full_per_token = self._full_per_token // 2 + self._full_per_token // 16\n"
                 "            self._swa_per_token = self._swa_per_token // 2 + self._swa_per_token // 16\n"),
    },
]

def sglang_root() -> str:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang not importable")
    return list(spec.submodule_search_locations)[0]


def install_modules(root: str, check: bool) -> list[str]:
    out = []
    for src, rel in MODULES.items():
        dst = os.path.join(root, rel)
        if check:
            out.append(f"module {rel}: src exists={os.path.isfile(src)}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "rb") as f:
            content = f.read()
        with open(dst, "wb") as f:
            f.write(content)
        out.append(f"module {rel}: installed")
    return out


def _new_source(src: str, h: dict) -> str:
    anchor = h["anchor"]
    at = h.get("at", len(anchor))
    return src.replace(anchor, anchor[:at] + h["text"] + anchor[at:], 1)


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook; SystemExit if an anchor is missing or not unique."""
    out, missing = [], []
    by_path: dict[str, str] = {}
    for h in HOOKS:
        path = os.path.join(root, h["path"])
        src = by_path.get(path)
        if src is None:
            with open(path) as f:
                src = f.read()
        if f"# numinous-hook:{h['name']}\n" in src:
            out.append(f"{h['name']}: already applied")
            by_path[path] = src
            continue
        n = src.count(h["anchor"])
        if n != 1:
            missing.append(f"{h['name']}: anchor found {n} times in {h['path']}")
            continue
        if check:
            out.append(f"{h['name']}: anchor ok")
            continue
        by_path[path] = _new_source(src, h)
        out.append(f"{h['name']}: applied to {h['path']}")
    if missing:
        raise SystemExit("kv4tri_hooks: " + "; ".join(missing))
    if not check:
        for path, new in by_path.items():
            compile(new, path, "exec")
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="", help="sglang package root (default: installed)")
    ap.add_argument("--check", action="store_true", help="check anchors only, change nothing")
    a = ap.parse_args(argv)
    root = a.root or sglang_root()
    for line in install_modules(root, a.check):
        print(line)
    for line in apply(root, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
