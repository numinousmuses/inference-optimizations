"""Kimi K2.7 Code 4-bit MLA KV cache: anchored hooks into SGLang v0.5.20 plus the modules they import.

Same contract as sglang_numinous_hooks.py / glm_kv4_hooks.py: each hook names a file, an exact anchor (must occur
once) and text inserted after it (or in its place for `replace`); a marker line makes re-application a no-op; a
patched file must compile before it replaces the original. `apply` installs:

    operations/kernels/glm_kv4.py          -> sglang/srt/layers/attention/dsa/glm_kv4.py       (row format, kernels)
    operations/kernels/kv4mla/{kv4mla.py, csrc/} -> sglang/srt/layers/attention/dsa/kv4mla/    (native FP4 kernel)
    operations/kernels/k27_kv4_engine.py   -> sglang/srt/layers/attention/dsa/k27_kv4_engine.py (pool, routing)

and compiles the kv4mla CUDA extension (no GPU needed; --no-build skips), so engine processes load it from the cache.

    python3 k27_kv4_hooks.py                    # patch the installed sglang + build
    python3 k27_kv4_hooks.py --root /tmp/sgl --check

Everything is inert without NUMINOUS_K27_KV4=1 (one env test at pool build, one attribute test per attention call).
Hooks:
  k27_kv4_dim      mem_cache/kv_cache_configurator.calculate_mla_kv_cache_dim: 352 B rows for dense MLA models
  k27_kv4_pool     kv_cache_configurator._build_mla_kv_pool: K27KV4MLATokenToKVPool instead of MLATokenToKVPool
  k27_kv4_init     trtllm_mla_backend.__init__: attach (layout checks, dense decoder, fused raw writes off)
  k27_kv4_graph    trtllm_mla_backend.init_cuda_graph_state: allocate the q tiles / limits / schedules before capture
  k27_kv4_decode   trtllm_mla_backend.forward_decode: metadata first, then route to k27_kv4_engine.decode
  k27_kv4_verify_a trtllm_mla_backend.forward_extend (verify / draft-extend branch): skip the raw pool view
  k27_kv4_verify_b trtllm_mla_backend.forward_extend: run k27_kv4_engine.verify in place of the FP8 kernel
  k27_kv4_prefill  tokenspeed_mla_backend.prepare_prefill_qkv: the pool write takes the bf16 latent
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIGURATOR = "srt/mem_cache/kv_cache_configurator.py"
TRT = "srt/layers/attention/trtllm_mla_backend.py"
TS = "srt/layers/attention/tokenspeed_mla_backend.py"
KDIR = REPO / "operations/kernels/kv4mla"
PKG_DIR = "srt/layers/attention/dsa/kv4mla"
MODULES = {
    "operations/kernels/glm_kv4.py": "srt/layers/attention/dsa/glm_kv4.py",
    "operations/kernels/k27_kv4_engine.py": "srt/layers/attention/dsa/k27_kv4_engine.py",
}
IMPORT = "from sglang.srt.layers.attention.dsa import k27_kv4_engine as _nk27"


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


DIM = f'''    # numinous-hook:k27_kv4_dim
    # Numinous (NUMINOUS_K27_KV4=1): dense MLA FP4 latent rows, 352 B (operations/kernels/k27_kv4_engine.py)
    if not is_dsa_model:
        {IMPORT}

        if _nk27.enabled_for(model_config.hf_config, kv_cache_dtype):
            return _nk27.ROW_BYTES
'''
POOL_ANCHOR = '''    def _build_mla_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        token_to_kv_pool = MLATokenToKVPool(
'''
POOL = f'''    def _build_mla_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        # numinous-hook:k27_kv4_pool
        {IMPORT}

        PoolCls = (
            _nk27.K27KV4MLATokenToKVPool
            if _nk27.enabled_for(self.model_config.hf_config, self.kv_cache_dtype)
            else MLATokenToKVPool
        )
        token_to_kv_pool = PoolCls(
'''
INIT_ANCHOR = '''            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
            and can_use_set_mla_kv_concat_q_fp8()
        )
'''
INIT = f'''        # numinous-hook:k27_kv4_init
        if getattr(model_runner.token_to_kv_pool, "numinous_kv4", False):
            {IMPORT}

            _nk27.attach(self, model_runner)
'''
GRAPH_ANCHOR = '''        """Initialize CUDA graph state for TRTLLM MLA."""

        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
'''
GRAPH = f'''        # numinous-hook:k27_kv4_graph
        if getattr(self, "_numinous_k27_kv4", None) is not None:
            {IMPORT}

            _nk27.prepare_graph(self, max_bs)
'''
DECODE_ANCHOR = '''        # Prepare KV cache inline
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        # Get metadata
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )

        # Backstop: metadata was built pre-pad (marked) and DP padding then
        # grew the batch. The marker path deliberately does not re-plan
        # post-pad (DSA can't rebuild on a padded batch, see #27091), so this
        # local re-plan catches the size mismatch.
        batch_size = getattr(metadata, "batch_size", None)
        if batch_size is not None and batch_size < forward_batch.batch_size:
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata
'''
DECODE = '''        # Get metadata
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )

        # Backstop: metadata was built pre-pad (marked) and DP padding then
        # grew the batch. The marker path deliberately does not re-plan
        # post-pad (DSA can't rebuild on a padded batch, see #27091), so this
        # local re-plan catches the size mismatch.
        batch_size = getattr(metadata, "batch_size", None)
        if batch_size is not None and batch_size < forward_batch.batch_size:
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        # numinous-hook:k27_kv4_decode
        if getattr(self, "_numinous_k27_kv4", None) is not None:
            from sglang.srt.layers.attention.dsa import k27_kv4_engine as _nk27

            return _nk27.decode(self, query, layer, forward_batch, metadata)

        # Prepare KV cache inline
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

'''
VERIFY_A_ANCHOR = '''            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

            q = q.to(self.data_type)
'''
VERIFY_A = '''            q = q.to(self.data_type)

            # numinous-hook:k27_kv4_verify_a
            _nk27_on = getattr(self, "_numinous_k27_kv4", None) is not None
            kv_cache = None
            if not _nk27_on:
                k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)
'''
VERIFY_B_ANCHOR = '''            assert kv_cache.dtype == self.data_type

            if (
                forward_batch.forward_mode.is_target_verify()
                and get_parallel().dcp_enabled
            ):
'''
VERIFY_B = f'''            # numinous-hook:k27_kv4_verify_b
            if _nk27_on:
                {IMPORT}

                raw_out = _nk27.verify(
                    self, q, layer, forward_batch, metadata, q.shape[1],
                    unpad_seq_lens_q if needs_unpad else None,
                )
                if needs_unpad:
                    output = self.unpad_draft_extend_output(
                        raw_out,
                        unpad_cu_seqlens_q,
                        unpad_seq_lens_q,
                        unpad_sum_seq_lens_q,
                        zero_uncovered=unpad_zero_uncovered,
                    )
                    return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
                return raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            assert kv_cache.dtype == self.data_type

            if (
                forward_batch.forward_mode.is_target_verify()
                and get_parallel().dcp_enabled
            ):
'''
PREFILL_ANCHOR = '''        self.token_to_kv_pool.set_mla_kv_buffer(
            layer.attn_mha,
            forward_batch.out_cache_loc,
            kv_a_fp8.unsqueeze(1),
            k_pe_fp8,
        )
'''
PREFILL = '''        # numinous-hook:k27_kv4_prefill
        # the FP4 pool quantizes from the bf16 latent (one rounding less than from its FP8 copy)
        self.token_to_kv_pool.set_mla_kv_buffer(
            layer.attn_mha,
            forward_batch.out_cache_loc,
            (kv_a if getattr(self, "_numinous_k27_kv4", None) is not None else kv_a_fp8).unsqueeze(1),
            k_pe_fp8,
        )
'''

HOOKS = [
    Hook("k27_kv4_dim", CONFIGURATOR, "    # For non-DSA models, MLA kv cache dim is simply kv_lora_rank + qk_rope_head_dim\n", DIM),
    Hook("k27_kv4_pool", CONFIGURATOR, POOL_ANCHOR, POOL, replace=True),
    Hook("k27_kv4_init", TRT, INIT_ANCHOR, INIT),
    Hook("k27_kv4_graph", TRT, GRAPH_ANCHOR, GRAPH),
    Hook("k27_kv4_decode", TRT, DECODE_ANCHOR, DECODE, replace=True),
    Hook("k27_kv4_verify_a", TRT, VERIFY_A_ANCHOR, VERIFY_A, replace=True),
    Hook("k27_kv4_verify_b", TRT, VERIFY_B_ANCHOR, VERIFY_B, replace=True),
    Hook("k27_kv4_prefill", TS, PREFILL_ANCHOR, PREFILL, replace=True),
]
FILES = (CONFIGURATOR, TRT, TS)


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
        raise SystemExit("k27_kv4 hooks: " + "; ".join(missing))
    return src, out


def install_modules(root: str) -> list[str]:
    lines = []
    for local, rel in MODULES.items():
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(REPO / local, dst)
        lines.append(f"installed {rel}")
    dst = pathlib.Path(root) / PKG_DIR
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(KDIR / "csrc", dst / "csrc", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(KDIR / "kv4mla.py", dst / "kv4mla.py")
    (dst / "__init__.py").write_text("")
    lines.append(f"installed {PKG_DIR}/ (kv4mla.py, csrc)")
    return lines


def apply(root: str, check: bool = False, build: bool = True) -> list[str]:
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
    if check:
        return lines
    lines += install_modules(root)
    if build:
        mod_path = pathlib.Path(root) / PKG_DIR / "kv4mla.py"
        code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('kv4mla_build', %r); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print('built', m.load().__file__)"
                % str(mod_path))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("k27_kv4: kv4mla extension build failed:\n" + r.stdout[-3000:] + r.stderr[-6000:])
        lines.append(r.stdout.strip().splitlines()[-1])
    return lines


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check, build=not a.no_build):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
