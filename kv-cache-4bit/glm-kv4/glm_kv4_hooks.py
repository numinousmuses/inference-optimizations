"""GLM-5.3 4-bit KV cache: anchored hooks into SGLang v0.5.20 plus the two modules they import.

Same contract as sglang_numinous_hooks.py: each hook names a file, an exact anchor (must occur once), and text that
goes right after it (or, for `replace`, in its place); a marker line makes re-application a no-op and a patched file
must compile before it replaces the original. `apply` installs:

    operations/kernels/glm_kv4.py        -> sglang/srt/layers/attention/dsa/glm_kv4.py         (kernels, reference)
    operations/kernels/glm_kv4_engine.py -> sglang/srt/layers/attention/dsa/glm_kv4_engine.py  (pool, routing)

    python3 glm_kv4_hooks.py                   # patch the installed sglang
    python3 glm_kv4_hooks.py --root /tmp/sgl --check

Everything is inert without NUMINOUS_GLM_KV4=1 (one env test at pool build, one attribute test per attention call).
Hooks:
  glm_kv4_dim     mem_cache/kv_cache_configurator.calculate_mla_kv_cache_dim: 352 B rows for the FP4 pool (prices the
                  pool and sizes the buffers; the pool configurator and the pool builder both call it)
  glm_kv4_pool    kv_cache_configurator._build_dsa_kv_pool: GLMKV4DSATokenToKVPool instead of DSATokenToKVPool
  glm_kv4_init    dsa_backend.__init__: attach (verifies trtllm roles, SM100, FP8 q path; loud otherwise)
  glm_kv4_graph   dsa_backend.init_cuda_graph_state: reserve the fixed decode/verify scratch before capture
  glm_kv4_stash   dsa_backend._forward_trtllm: keep the bf16 latent before the FP8 quantize + RoPE kernel
  glm_kv4_write   dsa_backend._forward_trtllm: the pool write takes the bf16 latent (replace)
  glm_kv4_route   dsa_backend._forward_trtllm: after the top-k page table is padded, run engine_forward
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
MODULES = {
    "operations/kernels/glm_kv4.py": "srt/layers/attention/dsa/glm_kv4.py",
    "operations/kernels/glm_kv4_engine.py": "srt/layers/attention/dsa/glm_kv4_engine.py",
}


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


DIM = '''        # numinous-hook:glm_kv4_dim
        # Numinous (NUMINOUS_GLM_KV4=1): GLM-5.3 FP4 latent rows, 352 B (operations/kernels/glm_kv4.py)
        from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

        if _nkv4.enabled_for(model_config.hf_config, kv_cache_dtype):
            return _nkv4.ROW_BYTES
'''

POOL = '''            # numinous-hook:glm_kv4_pool
            from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

            if _nkv4.enabled_for(self.model_config.hf_config, self.kv_cache_dtype):
                PoolCls = _nkv4.GLMKV4DSATokenToKVPool
'''

INIT = '''        # numinous-hook:glm_kv4_init
        # Numinous (NUMINOUS_GLM_KV4=1): FP4 DSA pool -> attention reads it through glm_kv4_engine.engine_forward
        from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

        self._numinous_kv4 = _nkv4.attach(self, model_runner)
'''

GRAPH = '''        # numinous-hook:glm_kv4_graph
        if getattr(self, "_numinous_kv4", False):
            from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

            _nkv4.reserve_graph_scratch(self, max_num_tokens)
'''

STASH = '''        # numinous-hook:glm_kv4_stash
        _numinous_kv4_k_bf16 = k if getattr(self, "_numinous_kv4", False) else None
'''

WRITE_ANCHOR = "            self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)\n"
WRITE = '''            # numinous-hook:glm_kv4_write
            if getattr(self, "_numinous_kv4", False):
                from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

                _nkv4.write_bf16_latent(self, _numinous_kv4_k_bf16, k, k_rope, layer, cache_loc)
            else:
                self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)
'''

ROUTE = '''        # numinous-hook:glm_kv4_route
        if getattr(self, "_numinous_kv4", False):
            from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

            return _nkv4.engine_forward(
                self, q_all, page_table_1, sparse_mla_top_k, layer, forward_batch, metadata, seq_lens, is_prefill
            )
'''

HOOKS = [
    Hook("glm_kv4_dim", CONFIGURATOR, "    if uses_trtllm_kv_layout:\n", DIM),
    Hook("glm_kv4_pool", CONFIGURATOR, "            PoolCls = DSATokenToKVPool\n", POOL),
    Hook("glm_kv4_init", BACKEND, "        self.kv_cache_dtype = model_runner.kv_cache_dtype\n", INIT),
    Hook("glm_kv4_graph", BACKEND,
         "        # Whether we can skip the wide [max_num_tokens, max_ctx_len] page_size=1\n", GRAPH),
    Hook("glm_kv4_stash", BACKEND, "        merge_query = q_rope is not None and self.qk_rope_head_dim > 0\n", STASH),
    Hook("glm_kv4_write", BACKEND, WRITE_ANCHOR, WRITE, replace=True),
    Hook("glm_kv4_route", BACKEND,
         "        page_table_1, sparse_mla_top_k = self._pad_trtllm_sparse_page_table(\n"
         "            page_table_1\n"
         "        )\n", ROUTE),
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
        src = src.replace(h.anchor, h.text if h.replace else h.anchor + h.text, 1)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("glm_kv4 hooks: " + "; ".join(missing))
    return src, out


def apply(root: str, check: bool = False, install_modules: bool = True) -> list[str]:
    lines = []
    for path in (CONFIGURATOR, BACKEND):
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
