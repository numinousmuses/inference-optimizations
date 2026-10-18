"""Anchored hook: GLM-5.3-Flash prefill-sized sparse MLA on FlashMLA's sparse PREFILL kernel over the FP8 no-RoPE pool
(amdflash, Hopper kernel program 2026-10-05). Tree: SGLang v0.5.20 (IMG_V0520).

    python3 flash_sparse_prefill_hooks.py [--root <sglang pkg dir>] [--check]

The module is operations/kernels/flash_sparse_prefill.py, installed as srt/layers/attention/numinous_sparse_prefill.py.

Profile rows attacked (F4 prefill TP0, per 4,096-token step), 37.8 ms in all:
- FlashMLA sparse decode kernel 32.3;
- get_mla_metadata 3.39;
- slice cat 1.59;
- combine 0.48.
Measured on 1x H200 (flash_sparse_mla_bench_h200.json): the sparse prefill kernel is 0.65x / 0.63x of the hooked
decode path at 4,096 / 8,192 query tokens, i.e. -13.2 / -28.3 ms per step, with rel err vs a bf16 reference 2.2e-3
(decode path 3.3e-3).

One anchor, inert unless NUMINOUS_FLASH_SPARSE_PREFILL=1:
- prefill_kernel (srt/layers/attention/dsa_backend.py, _forward_flashmla_kv, right after the flash_mla import): it calls
  numinous_sparse_prefill.maybe_forward and returns its output when that is not None.
  - maybe_forward takes the path only outside CUDA-graph capture, for >= 256 query rows, with the 528-byte row and
    512-dim q.
  - Everything else (decode graphs, small eager batches) keeps rtxsys's hopper_fp8kv decode-kernel path.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
NAME = "flash_sparse_prefill"
ENV = "NUMINOUS_FLASH_SPARSE_PREFILL"
BACKEND = "srt/layers/attention/dsa_backend.py"
MODULE_SRC = "operations/kernels/flash_sparse_prefill.py"
MODULE_DST = "srt/layers/attention/numinous_sparse_prefill.py"
ANCHOR = (
    "        from sgl_kernel.flash_mla import flash_mla_with_kvcache\n"
    "\n"
    "        cache_seqlens = metadata.dsa_cache_seqlens_int32\n"
)
TEXT = (
    "        from sgl_kernel.flash_mla import flash_mla_with_kvcache\n"
    "\n"
    "        # numinous-hook:flash_sparse_prefill (amdflash): prefill-sized batches run FlashMLA's sparse PREFILL kernel\n"
    "        # on a compact bf16 copy of the batch's FP8 rows (NUMINOUS_FLASH_SPARSE_PREFILL=1); None = this stock path\n"
    "        _nsp_o = __import__(\n"
    "            \"sglang.srt.layers.attention.numinous_sparse_prefill\", fromlist=[\"maybe_forward\"]\n"
    "        ).maybe_forward(self, q_all, kv_cache, v_head_dim, sm_scale, layer, metadata, page_table_1)\n"
    "        if _nsp_o is not None:\n"
    "            return _nsp_o\n"
    "\n"
    "        cache_seqlens = metadata.dsa_cache_seqlens_int32\n"
)


def apply(root: str, check: bool = False, install_module: bool = True) -> list[str]:
    path = os.path.join(root, BACKEND)
    src = open(path).read()
    out = []
    if TEXT in src:
        out.append("prefill_kernel: already applied")
    else:
        n = src.count(ANCHOR)
        if n != 1:
            raise SystemExit(f"{NAME}.prefill_kernel: anchor found {n} times in {BACKEND}")
        if check:
            return ["prefill_kernel: anchor ok"]
        new = src.replace(ANCHOR, TEXT, 1)
        compile(new, path, "exec")
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"prefill_kernel: applied to {BACKEND}")
    if install_module and not check:
        dst = os.path.join(root, MODULE_DST)
        shutil.copy(REPO / MODULE_SRC, dst)
        py_compile.compile(dst, doraise=True)
        out.append(f"{NAME}: installed {MODULE_DST}")
    return out



def numinous_preflight(ctx) -> list:
    """Attach-time check: the sparse PREFILL kernel path lives in _forward_flashmla_kv and reads the 528-byte no-RoPE
    FP8 row. It needs FP8 KV, the flashmla_kv prefill backend, rtxsys's hopper_fp8kv hooks (kpool rows), rope 0 /
    kv_lora 512, and SM90; GLM KV4 (352-byte rows, kv4h_route on the same anchor) clashes."""
    def cfg_get(o, n, d=None):
        return d if o is None else (o.get(n, d) if isinstance(o, dict) else getattr(o, n, d))
    hf = getattr(ctx, "hf_config", None)
    t = getattr(hf, "text_config", None) or hf
    env_ = getattr(ctx, "env", None) or {}
    args_ = getattr(ctx, "args", None) or {}

    def env_on(name):
        return str(env_.get(name, "0")) == "1"

    def arg(name):
        return args_.get(name)
    prefill_like = getattr(ctx, "role", None) in (None, "prefill")
    sm = int((getattr(ctx, "device", None) or {}).get("sm_major", 0) or 0)
    if not env_on(ENV):
        return []
    probs = []
    if arg("kv_cache_dtype") != "fp8_e4m3":
        probs.append(f"{NAME}: {ENV}=1 needs --kv-cache-dtype fp8_e4m3 (got {arg('kv_cache_dtype')}): inert")
    if arg("dsa_prefill_backend") != "flashmla_kv":
        probs.append(f"{NAME}: {ENV}=1 needs dsa_prefill_backend flashmla_kv (got {arg('dsa_prefill_backend')})")
    if not env_on("NUMINOUS_HOPPER_FP8KV"):
        probs.append(f"{NAME}: {ENV}=1 needs NUMINOUS_HOPPER_FP8KV=1 (flashmla_kv refuses the kpool rows without it)")
    if cfg_get(t, "qk_rope_head_dim") not in (0, None) or cfg_get(t, "kv_lora_rank") not in (512, None):
        probs.append(f"{NAME}: {ENV}=1 needs qk_rope_head_dim 0 and kv_lora_rank 512 (the 528-byte row): inert")
    if sm and sm != 9:
        probs.append(f"{NAME}: {ENV}=1 is the SM90 FlashMLA path (sm {sm})")
    if env_on("NUMINOUS_GLM_KV4"):
        probs.append(f"{NAME}: clashes with NUMINOUS_GLM_KV4=1 (352-byte rows; kv4h_route anchors on the same import)")
    return probs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = a.root
    if root is None:
        import importlib.util
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("sglang is not importable")
        root = list(spec.submodule_search_locations)[0]
    for line in apply(root, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
