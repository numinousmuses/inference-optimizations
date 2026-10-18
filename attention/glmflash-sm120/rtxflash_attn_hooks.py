"""GLM-5.3-Flash DSA attention on SM120 (RTX PRO 6000): anchored hooks into SGLang v0.5.20 plus the modules they import.

Same contract as sglang_numinous_hooks.py / glmflash_kv4_hooks.py: each hook names a file, an exact anchor (must occur
once), and text that goes right after it (or, for `replace`, in its place); a marker line makes re-application a
no-op; a patched file must compile before it replaces the original. `apply` installs:

    operations/kernels/glmflash_sm120_attn.py   -> sglang/srt/layers/attention/dsa/glmflash_sm120_attn.py (Triton kernels)
    operations/kernels/glmflash_sm120_fi.py     -> .../dsa/glmflash_sm120_fi.py      (FlashInfer SM120 glue, top-k 2112 JIT)
    operations/kernels/glmflash_sm120_engine.py -> .../dsa/glmflash_sm120_engine.py  (validate, dims, pools, routing)
    operations/kernels/glm_kv4.py, glmflash_kv4.py -> .../dsa/ (the FP4 row writer, shared with glmflash_kv4)

    python3 rtxflash_attn_hooks.py                  # patch the installed sglang
    python3 rtxflash_attn_hooks.py --root /tmp/sgl --check

Everything is inert without NUMINOUS_RTXFLASH_ATTN=1. Do not combine with glmflash_kv4_hooks.py (same pool seam;
apply() refuses when its marker is present).
Hooks:
  rtxattn_validate   kernels/ops/attention/flash_mla_sm120._validate_flashinfer_sparse_mla_backend: accept Glm5Next
                     (+NextN) on SM12x with bf16 or fp8 KV (stock: GLM-5.3 archs + fp8 only)
  rtxattn_dim        mem_cache/kv_cache_configurator.calculate_mla_kv_cache_dim: 656 (fp8) / 288 (fp4) byte rows
                     (stock: 528 = 512 + 16 for a rope-free fp8 row, which no SM120 kernel reads)
  rtxattn_dsa_pool   kv_cache_configurator._build_dsa_kv_pool: pool class swap (the NextN draft's plain DSA pool)
  rtxattn_hybrid_pool mem_cache/memory_pool.HybridLinearKVPool: pool class swap (Flash's target pool)
  rtxattn_init       layers/attention/dsa_backend.__init__: attach the engine (builds the FlashInfer JIT module here,
                     before CUDA-graph capture)
  rtxattn_route      dsa_backend._forward_flashinfer_sparse_mla: engine.forward (decode, verify, prefill)
  rtxattn_kpool      layers/attention/dsa/dsa_backend_kpool._check_kpool_tail_backend: the engine handles the 2051-wide
                     kpool table (stock: only fa3 / tilelang / trtllm may see it)
  rtxattn_kpool_unfused_a..d  layers/attention/dsa/dsa_indexer_kpool: honour DSAIndexerMetadata.force_unfused_topk (PD
                     prefill draft-extend under EAGLE IndexShare): raw request-relative positions, as the PD seed
                     contract needs (stock kpool checks only SGLANG_DSA_FUSE_TOPK -> page-mapped slots)
  rtxattn_transform_prefill   kernels/ops/attention/dsa/transform_index: accept width 2051 (the kernel is width-generic)
  rtxattn_transform_decode    same file: widths != 2048 -> glmflash_sm120_engine.transform_decode_any (gather + where)
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = "srt/layers/attention/dsa_backend.py"
KPOOL = "srt/layers/attention/dsa/dsa_backend_kpool.py"
CONFIGURATOR = "srt/mem_cache/kv_cache_configurator.py"
MEMORY_POOL = "srt/mem_cache/memory_pool.py"
FLASH_MLA_SM120 = "kernels/ops/attention/flash_mla_sm120.py"
KPOOL_INDEXER = "srt/layers/attention/dsa/dsa_indexer_kpool.py"
TRANSFORM = "kernels/ops/attention/dsa/transform_index.py"
FILES = (FLASH_MLA_SM120, CONFIGURATOR, MEMORY_POOL, BACKEND, KPOOL, KPOOL_INDEXER, TRANSFORM)
MODULES = {
    "operations/kernels/glmflash_sm120_attn.py": "srt/layers/attention/dsa/glmflash_sm120_attn.py",
    "operations/kernels/glmflash_sm120_fi.py": "srt/layers/attention/dsa/glmflash_sm120_fi.py",
    "operations/kernels/glmflash_sm120_engine.py": "srt/layers/attention/dsa/glmflash_sm120_engine.py",
    "operations/kernels/glm_kv4.py": "srt/layers/attention/dsa/glm_kv4.py",
    "operations/kernels/glmflash_kv4.py": "srt/layers/attention/dsa/glmflash_kv4.py",
}
ENGINE = "sglang.srt.layers.attention.dsa.glmflash_sm120_engine"
CONFLICT_MARKER = "# numinous-hook:glmflash_kv4_hybrid_pool"


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str
    replace: bool = False

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


VALIDATE_ANCHOR = ('    selected = {prefill_impl, decode_impl}\n'
                   '    uses_flashinfer_sparse_mla = "flashinfer_sparse_mla" in selected\n')
VALIDATE = '''    # numinous-hook:rtxattn_validate
    from sglang.srt.layers.attention.dsa import glmflash_sm120_engine as _nrtx

    if _nrtx.validate(model_arch, device_sm_major, kv_cache_dtype, prefill_impl, decode_impl):
        return True
'''

DIM_ANCHOR = "    quant_block_size = DSATokenToKVPool.quant_block_size\n"
DIM = '''    # numinous-hook:rtxattn_dim
    from sglang.srt.layers.attention.dsa import glmflash_sm120_engine as _nrtx

    _nrtx_dim = _nrtx.kv_cache_dim(model_config, kv_cache_dtype)
    if _nrtx_dim is not None:
        return _nrtx_dim
'''

DSA_POOL_ANCHOR = "            PoolCls = DSATokenToKVPool\n"
DSA_POOL = '''            # numinous-hook:rtxattn_dsa_pool
            from sglang.srt.layers.attention.dsa import glmflash_sm120_engine as _nrtx

            PoolCls = _nrtx.pool_class(
                PoolCls,
                calculate_mla_kv_cache_dim(
                    model_config=self.model_config, kv_cache_dtype=self.kv_cache_dtype
                ),
            )
'''

HYBRID_ANCHOR = "            self.full_kv_pool = DSATokenToKVPool(\n"
HYBRID = (f'            self.full_kv_pool = __import__("{ENGINE}", fromlist=["pool_class"])'
          '.pool_class(DSATokenToKVPool, kv_cache_dim)(  # numinous-hook:rtxattn_hybrid_pool\n')

INIT_ANCHOR = "        self.kv_cache_dtype = model_runner.kv_cache_dtype\n"
INIT = '''        # numinous-hook:rtxattn_init
        from sglang.srt.layers.attention.dsa import glmflash_sm120_engine as _nrtx

        self._numinous_rtx = _nrtx.attach(self, model_runner)
'''

ROUTE_ANCHOR = ("        from sglang.kernels.ops.attention.flash_mla_sm120 import (\n"
                "            flashinfer_sparse_mla_forward,\n"
                "        )\n")
ROUTE = '''        # numinous-hook:rtxattn_route
        if getattr(self, "_numinous_rtx", None) is not None:
            return self._numinous_rtx.forward(
                q_all, kv_cache, page_table_1, sm_scale, self.workspace_buffer
            )
'''

KPOOL_ANCHOR = '            or dsa_impl in ("fa3", "tilelang", "trtllm")\n'
KPOOL_TEXT = ('            or dsa_impl in ("fa3", "tilelang", "trtllm")\n'
              '            or (dsa_impl == "flashinfer_sparse_mla" and getattr(self, "_numinous_rtx", None) is not None)'
              '  # numinous-hook:rtxattn_kpool\n')

# ---- PD (disaggregation) coverage, 2026-10-01 (rtxsys G3: PD prefill asserted topk width 2048) ----
_ENG = f'__import__("{ENGINE}", fromlist=["kpool_unfused"])'
KU_A_ANCHOR = "        if not envs.SGLANG_DSA_FUSE_TOPK.get():\n            return None, None, None\n"
KU_A = (f"        if not envs.SGLANG_DSA_FUSE_TOPK.get() or {_ENG}.kpool_unfused(metadata):"
        "  # numinous-hook:rtxattn_kpool_unfused_a\n            return None, None, None\n")
KU_B_ANCHOR = ("        if envs.SGLANG_DSA_FUSE_TOPK.get():\n            if topk_method == TopkTransformMethod.PAGED:\n"
               "                page_table_all = plan.ragged_paged_page_table\n")
KU_B = (f"        if envs.SGLANG_DSA_FUSE_TOPK.get() and not {_ENG}.kpool_unfused(metadata):"
        "  # numinous-hook:rtxattn_kpool_unfused_b\n            if topk_method == TopkTransformMethod.PAGED:\n"
        "                page_table_all = plan.ragged_paged_page_table\n")
KU_C_ANCHOR = ("            if (\n                envs.SGLANG_DSA_FUSE_TOPK.get()\n"
               "                and topk_method == TopkTransformMethod.PAGED\n            ):\n")
KU_C = ("            if (\n                envs.SGLANG_DSA_FUSE_TOPK.get()\n"
        f"                and not {_ENG}.kpool_unfused(metadata)  # numinous-hook:rtxattn_kpool_unfused_c\n"
        "                and topk_method == TopkTransformMethod.PAGED\n            ):\n")
KU_D_ANCHOR = ("            elif (\n                envs.SGLANG_DSA_FUSE_TOPK.get()\n"
               "                and topk_method == TopkTransformMethod.RAGGED\n")
KU_D = ("            elif (\n                envs.SGLANG_DSA_FUSE_TOPK.get()\n"
        f"                and not {_ENG}.kpool_unfused(metadata)  # numinous-hook:rtxattn_kpool_unfused_d\n"
        "                and topk_method == TopkTransformMethod.RAGGED\n")
TP_ANCHOR = ("    assert page_size == 1\n    assert topk_indices.shape[1] == 2048\n"
             "    real_num_tokens = sum(extend_lens_cpu)\n")
TP = ("    assert page_size == 1\n    assert topk_indices.shape[1] == 2048 or "
      f'__import__("{ENGINE}", fromlist=["wide_topk_ok"]).wide_topk_ok(topk_indices.shape[1])'
      "  # numinous-hook:rtxattn_transform_prefill\n    real_num_tokens = sum(extend_lens_cpu)\n")
TD_ANCHOR = ("    assert page_table.shape[0] == topk_indices.shape[0]\n    assert topk_indices.shape[1] == 2048\n"
             "    qo_len = topk_indices.shape[0]\n")
TD = ("    assert page_table.shape[0] == topk_indices.shape[0]\n"
      "    if topk_indices.shape[1] != 2048:  # numinous-hook:rtxattn_transform_decode\n"
      f'        return __import__("{ENGINE}", fromlist=["transform_decode_any"]).transform_decode_any(\n'
      "            page_table, topk_indices, result\n        )\n    qo_len = topk_indices.shape[0]\n")

HOOKS = [
    Hook("rtxattn_validate", FLASH_MLA_SM120, VALIDATE_ANCHOR, VALIDATE),
    Hook("rtxattn_dim", CONFIGURATOR, DIM_ANCHOR, DIM),
    Hook("rtxattn_dsa_pool", CONFIGURATOR, DSA_POOL_ANCHOR, DSA_POOL),
    Hook("rtxattn_hybrid_pool", MEMORY_POOL, HYBRID_ANCHOR, HYBRID, replace=True),
    Hook("rtxattn_init", BACKEND, INIT_ANCHOR, INIT),
    Hook("rtxattn_route", BACKEND, ROUTE_ANCHOR, ROUTE),
    Hook("rtxattn_kpool", KPOOL, KPOOL_ANCHOR, KPOOL_TEXT, replace=True),
    Hook("rtxattn_kpool_unfused_a", KPOOL_INDEXER, KU_A_ANCHOR, KU_A, replace=True),
    Hook("rtxattn_kpool_unfused_b", KPOOL_INDEXER, KU_B_ANCHOR, KU_B, replace=True),
    Hook("rtxattn_kpool_unfused_c", KPOOL_INDEXER, KU_C_ANCHOR, KU_C, replace=True),
    Hook("rtxattn_kpool_unfused_d", KPOOL_INDEXER, KU_D_ANCHOR, KU_D, replace=True),
    Hook("rtxattn_transform_prefill", TRANSFORM, TP_ANCHOR, TP, replace=True),
    Hook("rtxattn_transform_decode", TRANSFORM, TD_ANCHOR, TD, replace=True),
]


def sglang_root() -> str:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def patch_source(src: str, path: str) -> tuple[str, list[str]]:
    """Apply every hook for one file. Returns (new text, status lines); raises SystemExit on a bad anchor."""
    out, missing = [], []
    if path == MEMORY_POOL and CONFLICT_MARKER in src:
        raise SystemExit("rtxflash_attn hooks: glmflash_kv4 hooks already applied to this tree (same pool seam)")
    for h in HOOKS:
        if h.path != path:
            continue
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        src = src.replace(h.anchor, h.text if h.replace else h.anchor + h.text, 1)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("rtxflash_attn hooks: " + "; ".join(missing))
    return src, out


def apply(root: str, check: bool = False, install_modules: bool = True) -> list[str]:
    lines = []
    for path in FILES:
        full = os.path.join(root, path)
        src = open(full).read()
        new, ls = patch_source(src, path)
        if check:
            lines += [ln.replace("applied to", "anchor ok in") for ln in ls]
            continue
        compile(new, full, "exec")
        if new != src:
            tmp = f"{full}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, full)
        lines += ls
    if install_modules and not check:
        for local, rel in MODULES.items():
            dst = os.path.join(root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(REPO / local, dst)
            lines.append(f"installed {rel}")
    return lines


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
