"""GLM-5.3 4-bit KV cache on ROCm: anchored hooks into the ROCm SGLang tree (0.5.20.dev20260923, MI355X image)
plus the three modules they import. Same contract as glm_kv4_hooks.py (the SM100 version): exact anchors that must
occur once, a marker per hook, compile before replace. Everything is inert without NUMINOUS_GLM_KV4=1.

    operations/kernels/glm_kv4.py        -> sglang/srt/layers/attention/dsa/glm_kv4.py         (kernels)
    operations/kernels/glm_kv4_engine.py -> sglang/srt/layers/attention/dsa/glm_kv4_engine.py  (pool, scratch state)
    operations/kernels/glm_kv4_rocm.py   -> sglang/srt/layers/attention/dsa/glm_kv4_rocm.py    (TileLang read path)

    python3 glm_kv4_rocm_hooks.py [--root <sglang pkg dir>] [--check]

The box is shared: apply this to a private overlay copy (amd_gate recipes with "overlay_patchers"), not to the
installed tree. Hooks:
  kv4r_dim     kv_cache_configurator.calculate_mla_kv_cache_dim: 352 B rows for the FP4 pool (ahead of the HIP
               raw-layout return; the SM100 file anchors in the trtllm branch, which HIP never takes)
  kv4r_pool    kv_cache_configurator._build_dsa_kv_pool: GLMKV4DSATokenToKVPool
  kv4r_init    dsa_backend.__init__: glm_kv4_rocm.attach (tilelang roles, FP8 KV, ROCm; loud otherwise)
  kv4r_graph   dsa_backend.init_cuda_graph_state: reserve the decode/verify scratch before capture
  kv4r_route   dsa_backend._forward_tilelang: FP4 pool -> glm_kv4_rocm.forward_tilelang (dequant scratch + TileLang)
  kv4r_nofuse  forward_mla_rocm._skip_rope_for_dsa_tilelang_fused: False for the FP4 pool, so the AITER fused
               rope + concat + cache kernel (which writes FP8 rows into the pool) is not used; the backend writes
               the bf16 latent through set_mla_kv_buffer instead
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
FWD_ROCM = "srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py"
MODULES = {
    "operations/kernels/glm_kv4.py": "srt/layers/attention/dsa/glm_kv4.py",
    "operations/kernels/glm_kv4_engine.py": "srt/layers/attention/dsa/glm_kv4_engine.py",
    "operations/kernels/glm_kv4_rocm.py": "srt/layers/attention/dsa/glm_kv4_rocm.py",
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


DIM = '''    # numinous-hook:kv4r_dim
    # Numinous (NUMINOUS_GLM_KV4=1): GLM-5.3 FP4 latent rows, 352 B (operations/kernels/glm_kv4.py), in place of the
    # HIP raw 576-byte FP8 layout below
    from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

    if _nkv4.enabled_for(model_config.hf_config, kv_cache_dtype):
        return _nkv4.ROW_BYTES
'''

POOL = '''            # numinous-hook:kv4r_pool
            from sglang.srt.layers.attention.dsa import glm_kv4_engine as _nkv4

            if _nkv4.enabled_for(self.model_config.hf_config, self.kv_cache_dtype):
                PoolCls = _nkv4.GLMKV4DSATokenToKVPool
'''

INIT = '''        # numinous-hook:kv4r_init
        # Numinous (NUMINOUS_GLM_KV4=1): FP4 DSA pool -> _forward_tilelang reads it through glm_kv4_rocm
        from sglang.srt.layers.attention.dsa import glm_kv4_rocm as _nkv4r

        self._numinous_kv4 = _nkv4r.attach(self, model_runner)
'''

GRAPH = '''        # numinous-hook:kv4r_graph
        if getattr(self, "_numinous_kv4", False):
            from sglang.srt.layers.attention.dsa import glm_kv4_rocm as _nkv4r

            _nkv4r.reserve_graph_scratch(self, max_num_tokens)
'''

ROUTE = '''        # numinous-hook:kv4r_route
        if getattr(self, "_numinous_kv4", False):
            from sglang.srt.layers.attention.dsa import glm_kv4_rocm as _nkv4r

            return _nkv4r.forward_tilelang(self, q_all, kv_cache, page_table_1, sm_scale, v_head_dim)
'''

NOFUSE = '''        # numinous-hook:kv4r_nofuse
        # the fused AITER rope+cat+cache kernel writes FP8 rows into the pool; the FP4 pool takes the backend write
        if getattr(get_token_to_kv_pool(), "numinous_kv4", False):
            return False
'''

HOOKS = [
    Hook("kv4r_dim", CONFIGURATOR, "    # nope(512 fp8) + rope(64 fp8), without extra per-block scales.\n", DIM),
    Hook("kv4r_pool", CONFIGURATOR, "            PoolCls = DSATokenToKVPool\n", POOL),
    Hook("kv4r_init", BACKEND, "        self.kv_cache_dtype = model_runner.kv_cache_dtype\n", INIT),
    Hook("kv4r_graph", BACKEND,
         "        # Whether we can skip the wide [max_num_tokens, max_ctx_len] page_size=1\n", GRAPH),
    Hook("kv4r_route", BACKEND,
         "        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd\n", ROUTE),
    Hook("kv4r_nofuse", FWD_ROCM,
         "        Check if we should skip rope and use fused rope+cache path for TileLang/Triton DSA on gfx95.\n"
         '        """\n', NOFUSE),
]


def sglang_root() -> str:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def patch_source(src: str, path: str) -> tuple[str, list[str]]:
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
        raise SystemExit("glm_kv4 rocm hooks: " + "; ".join(missing))
    return src, out


def apply(root: str, check: bool = False, install_modules: bool = True) -> list[str]:
    lines = []
    for path in (CONFIGURATOR, BACKEND, FWD_ROCM):
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
