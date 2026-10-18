"""GLM-5.3 mixed FP8 + FP4 KV cache (kvmix): anchored hooks into SGLang v0.5.20 plus the modules they import.

Same contract as sglang_numinous_hooks.py / glm_kv4_hooks.py: each hook names a file, an exact anchor (must occur
once), and text that goes right after it (or, for `replace`, in its place); a marker line makes re-application a
no-op and a patched file must compile before it replaces the original. `apply` installs:

    operations/kernels/glm_kv4.py           -> sglang/srt/layers/attention/dsa/glm_kv4.py           (FP4 row format)
    operations/kernels/kvmix_glm.py         -> sglang/srt/layers/attention/dsa/kvmix_glm.py         (mixed kernels)
    operations/kernels/kvmix_glm_engine.py  -> sglang/srt/layers/attention/dsa/kvmix_glm_engine.py  (pool, routing)
    operations/kernels/kvmix_core.py        -> sglang/srt/mem_cache/kvmix_core.py                   (allocator, policy)

    python3 kvmix_glm_hooks.py                   # patch the installed sglang
    python3 kvmix_glm_hooks.py --root /tmp/sgl --check

Everything is inert without NUMINOUS_KVMIX=1 (env tests at pool / allocator build, one attribute test per call).
Hooks:
  kvmix_dim      mem_cache/kv_cache_configurator.calculate_mla_kv_cache_dim: blended row bytes (prices the split)
  kvmix_pool     kv_cache_configurator._build_dsa_kv_pool: KVMixDSATokenToKVPool instead of DSATokenToKVPool
  kvmix_alloc    kv_cache_configurator._build_token_to_kv_pool_allocator: KVMixAllocator over the mixed pool
  kvmix_evict    mem_cache/common.evict_from_tree_cache: per-pool eviction through the allocator
  kvmix_extend   mem_cache/allocation.alloc_for_extend: precision flags + promotion before the prefix rows are read
  kvmix_decode   mem_cache/allocation.alloc_for_decode: same before last_loc is read
  kvmix_spec     mem_cache/allocation.alloc_for_spec_decode: same (EAGLE / spec v2 decode allocation)
  kvmix_init     dsa_backend.__init__: attach (verifies trtllm roles, SM100, FP8 q path; loud otherwise)
  kvmix_graph    dsa_backend.init_cuda_graph_state: reserve the redirect table, check the scratch capacity
  kvmix_stash    dsa_backend._forward_trtllm: keep the bf16 latent before the FP8 quantize + RoPE kernel
  kvmix_write    dsa_backend._forward_trtllm: the pool write takes the bf16 latent (replace)
  kvmix_route    dsa_backend._forward_trtllm: after the top-k page table is padded, run engine_forward
kvmix_write's anchor includes the line before the stock write, so applying it on top of glm_kv4_hooks (which
replaces that write) fails loudly: the two pools are exclusive.
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
COMMON = "srt/mem_cache/common.py"
ALLOCATION = "srt/mem_cache/allocation.py"
FILES = (CONFIGURATOR, COMMON, ALLOCATION, BACKEND)
MODULES = {
    "operations/kernels/glm_kv4.py": "srt/layers/attention/dsa/glm_kv4.py",
    "operations/kernels/kvmix_glm.py": "srt/layers/attention/dsa/kvmix_glm.py",
    "operations/kernels/kvmix_glm_engine.py": "srt/layers/attention/dsa/kvmix_glm_engine.py",
    "operations/kernels/kvmix_core.py": "srt/mem_cache/kvmix_core.py",
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


DIM = '''        # numinous-hook:kvmix_dim
        # Numinous (NUMINOUS_KVMIX=1): GLM-5.3 mixed FP8 + FP4 pools priced as one blended row
        from sglang.srt.layers.attention.dsa import kvmix_glm_engine as _nkvmix

        if _nkvmix.enabled_for(model_config.hf_config, kv_cache_dtype):
            return _nkvmix.blended_dim()
'''

POOL = '''            # numinous-hook:kvmix_pool
            from sglang.srt.layers.attention.dsa import kvmix_glm_engine as _nkvmix

            if _nkvmix.enabled_for(self.model_config.hf_config, self.kv_cache_dtype):
                PoolCls = _nkvmix.KVMixDSATokenToKVPool
'''

ALLOC_ANCHOR = "            if get_memory().enable_hisparse and is_dsv4_model:\n"
ALLOC = '''            # numinous-hook:kvmix_alloc
            if getattr(token_to_kv_pool, "numinous_kvmix", False):
                from sglang.srt.layers.attention.dsa import kvmix_glm_engine as _nkvmix

                token_to_kv_pool_allocator = _nkvmix.build_allocator(
                    token_to_kv_pool,
                    get_schedule().page_size,
                    self.kv_cache_dtype,
                    self.device,
                    need_sort,
                    self.forward_stream,
                )
''' + ALLOC_ANCHOR

EVICT_ANCHOR = ("    allocator = tree_cache.token_to_kv_pool_allocator\n\n"
                "    if isinstance(allocator, SWATokenToKVPoolAllocator):\n")
EVICT = '''    allocator = tree_cache.token_to_kv_pool_allocator
    # numinous-hook:kvmix_evict
    if getattr(allocator, "kvmix", False):
        allocator.evict_to_free_tokens(tree_cache, num_tokens)
        return

    if isinstance(allocator, SWATokenToKVPoolAllocator):
'''

EXTEND_ANCHOR = "    prefix_tensors = [r.prefix_indices for r in batch.reqs]\n"
EXTEND = '''    # numinous-hook:kvmix_extend
    if getattr(batch.tree_cache.token_to_kv_pool_allocator, "kvmix", False):
        from sglang.srt.mem_cache.kvmix_core import before_alloc as _kvmix_before_alloc

        _kvmix_before_alloc(
            batch.tree_cache.token_to_kv_pool_allocator,
            batch.tree_cache,
            batch.reqs,
            batch.req_to_token_pool,
            "extend",
        )
''' + EXTEND_ANCHOR

DECODE_ANCHOR = "        last_loc = batch.req_to_token_pool.req_to_token[\n"
DECODE = '''        # numinous-hook:kvmix_decode
        if getattr(batch.tree_cache.token_to_kv_pool_allocator, "kvmix", False):
            from sglang.srt.mem_cache.kvmix_core import before_alloc as _kvmix_before_alloc

            _kvmix_before_alloc(
                batch.tree_cache.token_to_kv_pool_allocator,
                batch.tree_cache,
                batch.reqs,
                batch.req_to_token_pool,
                "decode",
            )
''' + DECODE_ANCHOR

SPEC_ANCHOR = "            last_loc = get_last_loc(\n"
SPEC = '''            # numinous-hook:kvmix_spec
            if getattr(tree_cache.token_to_kv_pool_allocator, "kvmix", False):
                from sglang.srt.mem_cache.kvmix_core import before_alloc as _kvmix_before_alloc

                _kvmix_before_alloc(
                    tree_cache.token_to_kv_pool_allocator, tree_cache, reqs, req_to_token_pool, "decode"
                )
''' + SPEC_ANCHOR

INIT = '''        # numinous-hook:kvmix_init
        # Numinous (NUMINOUS_KVMIX=1): mixed FP8/FP4 pool -> attention reads it through kvmix_glm_engine
        from sglang.srt.layers.attention.dsa import kvmix_glm_engine as _nkvmix

        self._numinous_kvmix = _nkvmix.attach(self, model_runner)
'''

GRAPH = '''        # numinous-hook:kvmix_graph
        if getattr(self, "_numinous_kvmix", False):
            from sglang.srt.layers.attention.dsa import kvmix_glm_engine as _nkvmix

            _nkvmix.reserve_graph_scratch(self, max_num_tokens)
'''

STASH = '''        # numinous-hook:kvmix_stash
        _numinous_kvmix_k_bf16 = k if getattr(self, "_numinous_kvmix", False) else None
'''

WRITE_ANCHOR = ("            )\n"
                "            self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)\n")
WRITE = '''            )
            # numinous-hook:kvmix_write
            if getattr(self, "_numinous_kvmix", False):
                from sglang.srt.layers.attention.dsa import kvmix_glm_engine as _nkvmix

                _nkvmix.write_door(self, _numinous_kvmix_k_bf16, k, k_rope, layer, cache_loc)
            else:
                self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)
'''

ROUTE = '''        # numinous-hook:kvmix_route
        if getattr(self, "_numinous_kvmix", False):
            from sglang.srt.layers.attention.dsa import kvmix_glm_engine as _nkvmix

            return _nkvmix.engine_forward(
                self, q_all, page_table_1, sparse_mla_top_k, layer, forward_batch, metadata, seq_lens, is_prefill
            )
'''

HOOKS = [
    Hook("kvmix_dim", CONFIGURATOR, "    if uses_trtllm_kv_layout:\n", DIM),
    Hook("kvmix_pool", CONFIGURATOR, "            PoolCls = DSATokenToKVPool\n", POOL),
    Hook("kvmix_alloc", CONFIGURATOR, ALLOC_ANCHOR, ALLOC, replace=True),
    Hook("kvmix_evict", COMMON, EVICT_ANCHOR, EVICT, replace=True),
    Hook("kvmix_extend", ALLOCATION, EXTEND_ANCHOR, EXTEND, replace=True),
    Hook("kvmix_decode", ALLOCATION, DECODE_ANCHOR, DECODE, replace=True),
    Hook("kvmix_spec", ALLOCATION, SPEC_ANCHOR, SPEC, replace=True),
    Hook("kvmix_init", BACKEND, "        self.kv_cache_dtype = model_runner.kv_cache_dtype\n", INIT),
    Hook("kvmix_graph", BACKEND,
         "        # Whether we can skip the wide [max_num_tokens, max_ctx_len] page_size=1\n", GRAPH),
    Hook("kvmix_stash", BACKEND, "        merge_query = q_rope is not None and self.qk_rope_head_dim > 0\n", STASH),
    Hook("kvmix_write", BACKEND, WRITE_ANCHOR, WRITE, replace=True),
    Hook("kvmix_route", BACKEND,
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
        raise SystemExit("kvmix hooks: " + "; ".join(missing))
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
