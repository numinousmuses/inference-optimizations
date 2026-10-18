"""GLM-5.3 4-bit KV cache on Hopper (SM90): anchored hooks into SGLang v0.5.20 (lmsysorg/sglang IMG_V0520) plus the
three modules they import. Same contract as glm_kv4_hooks.py (SM100) and glm_kv4_rocm_hooks.py (ROCm): each hook names
a file, an exact anchor that must occur once and the text that goes right after it; a marker per hook makes
re-application a no-op; a patched file must compile before it replaces the original. Inert without NUMINOUS_GLM_KV4=1.

    operations/kernels/glm_kv4.py        -> sglang/srt/layers/attention/dsa/glm_kv4.py         (kernels)
    operations/kernels/glm_kv4_engine.py -> sglang/srt/layers/attention/dsa/glm_kv4_engine.py  (pool, scratch state)
    operations/kernels/glm_kv4_sm90.py   -> sglang/srt/layers/attention/dsa/glm_kv4_sm90.py    (flashmla_kv read path)

    python3 glm_kv4_sm90_hooks.py [--root <sglang pkg dir>] [--check]

Hooks (why and the read path: operations/kernels/glm_kv4_sm90.py):
  kv4h_dim     kv_cache_configurator.calculate_mla_kv_cache_dim: 352 B rows for the FP4 pool, ahead of the 656-byte
               scaled-FP8 dim the SM90 flashmla_kv roles would get (the SM100 file anchors inside the trtllm branch,
               which SM90 never takes; same anchor as the ROCm file)
  kv4h_pool    kv_cache_configurator._build_dsa_kv_pool: GLMKV4DSATokenToKVPool
  kv4h_init    dsa_backend.__init__: glm_kv4_sm90.attach (flashmla_kv roles, FP8 KV flag, SM90; loud otherwise;
               turns MHA one-shot prefill off for this backend)
  kv4h_graph   dsa_backend.init_cuda_graph_state: reserve the bf16 decode/verify scratch before capture
  kv4h_fb_dec  dsa_backend.forward_decode: note the batch (mode, seq_lens_cpu) for the read path
  kv4h_fb_ext  dsa_backend.forward_extend: the same
  kv4h_route   dsa_backend._forward_flashmla_kv: FP4 pool -> glm_kv4_sm90.forward_flashmla_kv (exact bf16 dequant
               scratch + the stock SM90 bf16 sparse MLA kernel)
  kv4h_hybrid_pool  memory_pool.HybridLinearKVPool (use_dsa): GLM-5.3-Flash builds its DSA pool here; adopt it as the
               FP4 pool when kv4h_dim sized it to 352 B (glm_kv4_engine.adopt_pool). Flash's DSA layers are zero-rope:
               the row's rope bytes stay zero and q is padded to 576 (rtxattn 2026-10-05)
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
CONFIGURATOR = "srt/mem_cache/kv_cache_configurator.py"
MEMORY_POOL = "srt/mem_cache/memory_pool.py"
MODULES = {
    "operations/kernels/glm_kv4.py": "srt/layers/attention/dsa/glm_kv4.py",
    "operations/kernels/glm_kv4_engine.py": "srt/layers/attention/dsa/glm_kv4_engine.py",
    "operations/kernels/glm_kv4_sm90.py": "srt/layers/attention/dsa/glm_kv4_sm90.py",
}


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


DIM = '''    # numinous-hook:kv4h_dim
    # Numinous (NUMINOUS_GLM_KV4=1): GLM-5.3 FP4 latent rows, 352 B (operations/kernels/glm_kv4.py), in place of the
    # 656-byte scaled FP8 row the SM90 flashmla_kv roles would get below
    from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

    if _nkv4.enabled_for(model_config.hf_config, kv_cache_dtype):
        return _nkv4.ROW_BYTES
'''

POOL = '''            # numinous-hook:kv4h_pool
            from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

            if _nkv4.enabled_for(self.model_config.hf_config, self.kv_cache_dtype):
                PoolCls = _nkv4.GLMKV4DSATokenToKVPool
'''

INIT = '''        # numinous-hook:kv4h_init
        # Numinous (NUMINOUS_GLM_KV4=1): FP4 DSA pool -> _forward_flashmla_kv reads it through glm_kv4_sm90
        from sglang.srt.layers.attention.dsa import glm_kv4_sm90 as _nkv4h

        self._numinous_kv4 = _nkv4h.attach(self, model_runner)
'''

GRAPH = '''        # numinous-hook:kv4h_graph
        if getattr(self, "_numinous_kv4", False):
            from sglang.srt.layers.attention.dsa import glm_kv4_sm90 as _nkv4h

            _nkv4h.reserve_graph_scratch(self, max_num_tokens)
'''

FB_DEC = '''        # numinous-hook:kv4h_fb_dec
        if getattr(self, "_numinous_kv4", False):
            self._numinous_kv4_fb = forward_batch
'''

FB_EXT = '''        # numinous-hook:kv4h_fb_ext
        if getattr(self, "_numinous_kv4", False):
            self._numinous_kv4_fb = forward_batch
'''

ROUTE = '''        # numinous-hook:kv4h_route
        if getattr(self, "_numinous_kv4", False):
            from sglang.srt.layers.attention.dsa import glm_kv4_sm90 as _nkv4h

            return _nkv4h.forward_flashmla_kv(self, q_all, page_table_1, sm_scale, v_head_dim, layer, metadata)
'''

HYBRID_POOL = '''            # numinous-hook:kv4h_hybrid_pool
            # Numinous (NUMINOUS_GLM_KV4=1): GLM-5.3-Flash (hybrid KDA + DSA) builds its DSA pool here, not in
            # _build_dsa_kv_pool (kv4h_pool). kv4h_dim already sized the rows to 352 B, so adopt the instance as the
            # FP4 pool (class + state, no second allocation)
            from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

            if _nkv4.enabled() and kv_cache_dim == _nkv4.ROW_BYTES:
                _nkv4.adopt_pool(self.full_kv_pool)
'''

HOOKS = [
    Hook("kv4h_dim", CONFIGURATOR, "    # nope(512 fp8) + rope(64 fp8), without extra per-block scales.\n", DIM),
    Hook("kv4h_pool", CONFIGURATOR, "            PoolCls = DSATokenToKVPool\n", POOL),
    Hook("kv4h_init", BACKEND, "        self.kv_cache_dtype = model_runner.kv_cache_dtype\n", INIT),
    Hook("kv4h_graph", BACKEND,
         "        # Whether we can skip the wide [max_num_tokens, max_ctx_len] page_size=1\n", GRAPH),
    Hook("kv4h_fb_dec", BACKEND,
         '        self._check_kpool_tail_backend(topk_indices, dsa_impl, "decode")\n', FB_DEC),
    Hook("kv4h_fb_ext", BACKEND, "        self._check_kpool_tail_backend(topk_indices, dsa_impl, phase)\n", FB_EXT),
    Hook("kv4h_route", BACKEND, "        from sgl_kernel.flash_mla import flash_mla_with_kvcache\n", ROUTE),
    Hook("kv4h_hybrid_pool", MEMORY_POOL, "                skip_topk_layers=skip_topk_layers,\n            )\n",
         HYBRID_POOL),
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
    for h in HOOKS:
        if h.path != path:
            continue
        if h.marker + "\n" in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        src = src.replace(h.anchor, h.anchor + h.text, 1)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("glm_kv4_sm90 hooks: " + "; ".join(missing))
    return src, out


def apply(root: str, check: bool = False, install_modules: bool = True) -> list[str]:
    lines = []
    for path in (CONFIGURATOR, BACKEND, MEMORY_POOL):
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
