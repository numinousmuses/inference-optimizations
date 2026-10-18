"""Anchored patcher: 4-bit KV cache for DeepSeek V4 Pro on SGLang v0.5.20 (NUMINOUS_KV4PRO=c4|all; inert when unset).

Installs the kernel module (operations/kernels/kv4pro_fp4.py -> srt/layers/attention/numinous_kv4pro.py) and the engine
glue (operations/kernels/kv4pro_engine.py -> srt/layers/attention/numinous_kv4pro_engine.py), then hooks:

  kv4pro_pool_select   mem_cache/deepseek_v4_memory_pool.py DeepSeekV4TokenToKVPool.__init__: asks the engine module which
                       sub-pools take the 4-bit cell (validates: no unified KV, no trtllm DSV4 attention, no HiSparse,
                       not SM120) and logs the choice
  kv4pro_pool_swa      the SWA window pool is built with the 4-bit class in mode all
  kv4pro_pool_c4       the c4 compressed pool is built with the 4-bit class (c128 stays FP8)
  kv4pro_swa_store     set_swa_key_buffer_radix_fused_norm_rope: norm + RoPE (Triton) then the pool's quantizing
                       store, exactly the uniform-FP8 route (the fused FlashMLA-layout kernel cannot write 4-bit cells)
  kv4pro_compressor    compressor_v2.forward_unified: a 4-bit c4 pool takes the unfused compressor store
                       (compress -> norm + RoPE -> set_extra_key_buffer_fused), the uniform-FP8 route
  kv4pro_forward       deepseek_v4_backend.DeepseekV4AttnBackend.forward: layers reading a 4-bit pool go to
                       numinous_kv4pro_engine.forward_kv4 (sparse prefill with exact dequant; decode / verify / draft
                       through an FP8 workspace + the stock FlashMLA kernel); all-FP8 layers fall through
  kv4pro_price_cells   model_executor/pool_configurator.py DSV4PoolConfigurator._get_bytes_per_full_token: prices the
  kv4pro_price_expr    SWA / c4 / c128 terms with the cell each pool really allocates (320 or 384 B for 4-bit pools)
  kv4pro_reserve_ws    calculate_pool_sizes reserves the 4-bit decode workspaces (numinous_kv4pro.workspace_bytes,
  kv4pro_reserve_ws_fn ~0.2 GB) before sizing tokens, so the non-KV headroom stays what stock leaves (measured: without
                       it the final stack at 83k x chunk 8192 hit a 2 GiB prefill allocation OOM)
Group fp4idx (opt-in, NUMINOUS_DSV4_FP4_INDEXER_BYTES!=0 at run time):
  kv4pro_fp4_indexer_bytes  with --enable-deepseek-v4-fp4-indexer on NVIDIA the indexer pool allocates the FP4 cell
                       (DeepSeekV4IndexerPool.get_bytes_per_token: 68 B) but the configurator prices 132 B (HIP-only
                       condition); price what is allocated.

    python3 operations/patches/kv4pro_hooks.py [groups...] [--root DIR] [--check]     # default groups: core fp4idx
Test: operations/patches/test_kv4pro_hooks.py. Missing or repeated anchors fail loudly; files are compile-checked.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = {
    os.path.join(HERE, "..", "kernels", "kv4pro_fp4.py"): "srt/layers/attention/numinous_kv4pro.py",
    os.path.join(HERE, "..", "kernels", "kv4pro_engine.py"): "srt/layers/attention/numinous_kv4pro_engine.py",
}
POOL = "srt/mem_cache/deepseek_v4_memory_pool.py"
COMP = "srt/layers/attention/dsv4/compressor_v2.py"
BACKEND = "srt/layers/attention/deepseek_v4_backend.py"
CFG = "srt/model_executor/pool_configurator.py"


@dataclass(frozen=True)
class Hook:
    name: str
    group: str
    path: str
    anchor: str
    text: str
    replace: bool = False  # False: insert text after the anchor; True: the anchor becomes text

    @property
    def marker(self) -> str:
        return f"numinous-hook:{self.name}"


HOOKS = [
    Hook("kv4pro_pool_select", "core", POOL,
         "        kv_pool_cls: type = DeepSeekV4SingleKVPool\n",
         "        # numinous-hook:kv4pro_pool_select\n"
         "        from sglang.srt.layers.attention.numinous_kv4pro_engine import kv4_pool_classes as _nkv4_classes\n"
         "\n"
         "        self.kv4_swa_cls, self.kv4_c4_cls = _nkv4_classes(self)\n"
         "        if self.kv4_swa_cls is not None or self.kv4_c4_cls is not None:\n"
         "            logger.info(\n"
         "                \"numinous kv4pro: 4-bit KV cells for \"\n"
         "                f\"swa={self.kv4_swa_cls is not None} c4={self.kv4_c4_cls is not None} c128=False\"\n"
         "            )\n"),
    Hook("kv4pro_pool_swa", "core", POOL,
         "                global_page_size=swa_page_size,\n                cls=kv_pool_cls,\n",
         "                global_page_size=swa_page_size,\n"
         "                cls=(self.kv4_swa_cls or kv_pool_cls),  # numinous-hook:kv4pro_pool_swa\n",
         replace=True),
    Hook("kv4pro_pool_c4", "core", POOL,
         "                pool_cls = kv_pool_cls\n",
         "                if ratio == 4 and getattr(self, \"kv4_c4_cls\", None) is not None:  # numinous-hook:kv4pro_pool_c4\n"
         "                    pool_cls = self.kv4_c4_cls\n"),
    Hook("kv4pro_swa_store", "core", POOL,
         "        if self.uniform_fp8:\n            # Uniform-FP8 (trtllm-gen) layout: norm + RoPE",
         "        if self.uniform_fp8 or getattr(self.swa_kv_pool, \"numinous_kv4\", False):  # numinous-hook:kv4pro_swa_store\n"
         "            # Uniform-FP8 (trtllm-gen) layout: norm + RoPE",
         replace=True),
    Hook("kv4pro_compressor", "core", COMP,
         "        if token_to_kv_pool.uniform_fp8 and not compressor.is_in_indexer:\n",
         "        if (  # numinous-hook:kv4pro_compressor\n"
         "            token_to_kv_pool.uniform_fp8\n"
         "            or getattr(token_to_kv_pool.layer_mapping[layer_id].compress_kv_pool, \"numinous_kv4\", False)\n"
         "        ) and not compressor.is_in_indexer:\n",
         replace=True),
    Hook("kv4pro_forward", "core", BACKEND,
         "            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)\n",
         "            # numinous-hook:kv4pro_forward\n"
         "            if (\n"
         "                getattr(token_to_kv_pool, \"kv4_swa_cls\", None) is not None\n"
         "                or getattr(token_to_kv_pool, \"kv4_c4_cls\", None) is not None\n"
         "            ):\n"
         "                from sglang.srt.layers.attention.numinous_kv4pro_engine import forward_kv4 as _nkv4_forward\n"
         "\n"
         "                _nkv4_o = _nkv4_forward(\n"
         "                    self,\n"
         "                    q=q,\n"
         "                    layer=layer,\n"
         "                    forward_batch=forward_batch,\n"
         "                    compress_ratio=compress_ratio,\n"
         "                    core_attn_metadata=core_attn_metadata,\n"
         "                    attn_sink=attn_sink,\n"
         "                )\n"
         "                if _nkv4_o is not None:\n"
         "                    return _nkv4_o\n"),
    Hook("kv4pro_price_cells", "core", CFG,
         "        c4_frac = 1 / (4 * self.c4_shrink_factor)\n",
         "        # numinous-hook:kv4pro_price_cells: per-pool cell bytes (4-bit pools under NUMINOUS_KV4PRO)\n"
         "        _nkv4_swa = _nkv4_c4 = _nkv4_c128 = kv_bytes\n"
         "        if not self._unified:\n"
         "            from sglang.srt.layers.attention.numinous_kv4pro import priced_cell_bytes as _nkv4_price\n"
         "\n"
         "            _nkv4_swa, _nkv4_c4, _nkv4_c128 = _nkv4_price(kv_bytes)\n"
         "            if (_nkv4_swa, _nkv4_c4) != (kv_bytes, kv_bytes):\n"
         "                logger.info(\n"
         "                    f\"numinous kv4pro: priced cells swa={_nkv4_swa} c4={_nkv4_c4} c128={_nkv4_c128} B\"\n"
         "                )\n"),
    Hook("kv4pro_price_expr", "core", CFG,
         "                else self.swa_ratio * kv_bytes * self.num_layers_total\n"
         "            )\n"
         "            + c4_frac * kv_bytes * self.num_layers_ca4\n"
         "            + 1 / 128 * kv_bytes * self.num_layers_ca128\n",
         "                else self.swa_ratio * _nkv4_swa * self.num_layers_total  # numinous-hook:kv4pro_price_expr\n"
         "            )\n"
         "            + c4_frac * _nkv4_c4 * self.num_layers_ca4\n"
         "            + 1 / 128 * _nkv4_c128 * self.num_layers_ca128\n",
         replace=True),
    Hook("kv4pro_reserve_ws", "core", CFG,
         "        available_bytes_for_tokens = max(\n"
         "            available_bytes\n"
         "            - c128_state_fixed_bytes\n",
         "        available_bytes_for_tokens = max(\n"
         "            available_bytes\n"
         "            - _numinous_kv4pro_ws_bytes()  # numinous-hook:kv4pro_reserve_ws\n"
         "            - c128_state_fixed_bytes\n",
         replace=True),
    Hook("kv4pro_reserve_ws_fn", "core", CFG,
         "class DSV4PoolConfigurator(MemoryPoolConfigurator):\n",
         "def _numinous_kv4pro_ws_bytes() -> int:  # numinous-hook:kv4pro_reserve_ws_fn\n"
         "    \"\"\"FP8 decode workspaces of the 4-bit KV path (0 when NUMINOUS_KV4PRO is off).\"\"\"\n"
         "    from sglang.srt.layers.attention.numinous_kv4pro import workspace_bytes\n"
         "\n"
         "    return workspace_bytes()\n"
         "\n"
         "\n"
         "class DSV4PoolConfigurator(MemoryPoolConfigurator):\n",
         replace=True),
    Hook("kv4pro_fp4_indexer_bytes", "fp4idx", CFG,
         "            _is_hip and get_exec().kernel.enable_deepseek_v4_fp4_indexer,\n        )\n",
         "        # numinous-hook:kv4pro_fp4_indexer_bytes: the NVIDIA FP4 indexer pool allocates the FP4 cell too\n"
         "        import os as _nkv4_os\n"
         "\n"
         "        if _nkv4_os.environ.get(\"NUMINOUS_DSV4_FP4_INDEXER_BYTES\", \"1\") != \"0\":\n"
         "            self.indexer_bytes_per_token = get_dsv4_indexer_bytes_per_token(\n"
         "                self.indexer_head_dim, get_exec().kernel.enable_deepseek_v4_fp4_indexer\n"
         "            )\n"),
]


def sglang_root() -> str:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def patch_source(src: str, hooks) -> tuple[str, list[str], list[str]]:
    """Apply hooks (all for one file) to src. Returns (new_src, status lines, problems)."""
    out, problems = [], []
    for h in hooks:
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            problems.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        src = src.replace(h.anchor, h.text if h.replace else h.anchor + h.text, 1)
        out.append(f"{h.name}: applied to {h.path}")
    return src, out, problems


def apply(root: str, groups: list[str], check: bool = False) -> list[str]:
    sel = [h for h in HOOKS if not groups or h.group in groups or h.name in groups]
    out, problems = [], []
    by_path: dict[str, list[Hook]] = {}
    for h in sel:
        by_path.setdefault(h.path, []).append(h)
    new_files = {}
    for rel, hooks in by_path.items():
        path = os.path.join(root, rel)
        src = open(path).read()
        new, lines, probs = patch_source(src, hooks)
        problems += probs
        out += [ln.replace("applied to", "anchor ok in") if check else ln for ln in lines]
        if new != src:
            compile(new, path, "exec")  # never leave a file that does not parse
            new_files[path] = new
    if problems:
        raise SystemExit("numinous kv4pro hooks: " + "; ".join(problems))
    if check:
        return out
    for src, rel in MODULES.items():
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.normpath(src), dst)
        out.append(f"module: {rel}")
    for path, new in new_files.items():
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", help="hook groups (core, fp4idx) or names; default all")
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.groups, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
