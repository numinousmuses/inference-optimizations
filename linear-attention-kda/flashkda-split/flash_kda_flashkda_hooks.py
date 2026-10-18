"""Anchored hooks: GLM-5.3-Flash KDA prefill on FlashKDA past SGLang's caps, tracked batches included (amdflash, Hopper
kernel program L4, 2026-10-05). Tree: SGLang v0.5.20 (IMG_V0520).

    python3 flash_kda_flashkda_hooks.py [--root <sglang pkg dir>] [--check]

Module: operations/kernels/flash_kda_prefill.py, installed as srt/layers/attention/linear/numinous_flashkda.py. The
FlashKDA package is staged on the volume by operations/modal/flashkda_stage.py (NUMINOUS_FLASHKDA_DIR).

Profile row attacked (F4 prefill TP0, ms per 4,096-token step, 34 KDA layers), 29.1 ms in all:
- intra 5.27, inter_solve 5.24, recompute_w_u 4.57, delta_h 5.41, gla_o 6.28, gate cumsum 2.34.
Measured on 1x H200 at the served shapes:
- FlashKDA is 0.51-0.75x of Triton (flash_kda_flashkda_bench_h200.json);
- outputs agree at 5.9e-3 (flash_kda_flashkda_debug_h200.json).

Two anchors, inert unless NUMINOUS_FLASHKDA=1 (the cell also sets --linear-attn-prefill-backend flashkda on the
prefill role):
- extend (srt/layers/attention/linear/kernels/kda_flashkda.py, FlashKDAKernel.extend, top of the body): calls
  numinous_flashkda.extend. That routes every batch, tracked ones included, to FlashKDA.
  - Unaligned track points are split; the < 64-token tails go through a second small FlashKDA call.
  - It falls back to TritonKDAKernel.extend when it cannot (see the module doc).
- stash (srt/layers/attention/hybrid_linear_attn_backend.py, _init_track_ssm_indices' return): it records the CPU
  track_chunk_idx next to its device copy, so the split plan needs no GPU sync.
Package:
- The non-check run also builds FlashKDA 7afb9f45 (sm90a) into /opt/numinous/flash_kda-7afb9f45 once (ensure_package).
- So L4 does not depend on the Modal volume copy.
- A failed build never fails the series; the runtime then falls back to Triton.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
NAME = "flash_kda_flashkda"
ENV = "NUMINOUS_FLASHKDA"
MODULE_SRC = "operations/kernels/flash_kda_prefill.py"
MODULE_DST = "srt/layers/attention/linear/numinous_flashkda.py"
MOD = "sglang.srt.layers.attention.linear.numinous_flashkda"
HOOKS = [
    ("extend", "srt/layers/attention/linear/kernels/kda_flashkda.py",
     "        # The fused kernel cannot expose per-chunk states (h), which the mamba\n"
     "        # radix extra_buffer track path needs; route tracked batches through\n",
     "        # numinous-hook:flash_kda_flashkda (amdflash): FlashKDA for every prefill batch, tracked ones included\n"
     "        # (unaligned track points split, < 64-token tails in a second FlashKDA call); NUMINOUS_FLASHKDA=1\n"
     f"        _nfk = __import__(\"{MOD}\", fromlist=[\"extend\"])\n"
     "        if _nfk.enabled():\n"
     "            from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel as _NfkTriton\n"
     "            if not hasattr(self, \"_numinous_triton\"):\n"
     "                self._numinous_triton = _NfkTriton()\n"
     "            return _nfk.extend(\n"
     "                q, k, v, g, beta, ssm_states=ssm_states, cache_indices=cache_indices,\n"
     "                query_start_loc=query_start_loc, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,\n"
     "                extend_seq_lens_cpu=extend_seq_lens_cpu, is_spec_decode=is_spec_decode, beta_is_raw=beta_is_raw,\n"
     "                return_intermediate_states=return_intermediate_states,\n"
     "                triton_extend=self._numinous_triton.extend, **kwargs)\n"
     "        # The fused kernel cannot expose per-chunk states (h), which the mamba\n"
     "        # radix extra_buffer track path needs; route tracked batches through\n"),
    ("stash", "srt/layers/attention/hybrid_linear_attn_backend.py",
     "        return (\n"
     "            to_device(track_chunk_idx),\n",
     "        # numinous-hook:flash_kda_flashkda (amdflash): CPU track_chunk_idx for the FlashKDA split plan (no sync)\n"
     "        _nfk_dev = to_device(track_chunk_idx)\n"
     f"        __import__(\"{MOD}\", fromlist=[\"stash_track_idx\"]).stash_track_idx(_nfk_dev, track_chunk_idx)\n"
     "        return (\n"
     "            _nfk_dev,\n"),
]


REV = "7afb9f454f160a6c4bbc0999beca0a8c40a38934"
BAKED_DIR = f"/opt/numinous/flash_kda-{REV[:8]}"
REPO_URL = "https://github.com/MoonshotAI/FlashKDA.git"


def ensure_package(dst: str = BAKED_DIR, rev: str = REV, run=None) -> str:
    """Build FlashKDA (sm90a) into dst once, so L4 works off the Modal volume too.
    - It runs wherever the hook series runs: the Modal image build (verify_registry.hook_commands) and a node's
      serving.ts hookInstallSteps.
    - No GPU is needed: FLASH_KDA_CUDA_ARCHS=90a is explicit.
    - Measured on the h200 kbench image: 88-95 s.
    - Never fails the series. On any error it reports, and numinous_flashkda falls back to Triton at runtime."""
    import fcntl
    import glob
    import hashlib
    import json
    import subprocess
    import tempfile
    real = run is None
    run = run or (lambda c, **k: subprocess.run(c, capture_output=True, text=True, **k))
    if os.path.exists(os.path.join(dst, "manifest.json")):
        return f"{NAME}: flash_kda present at {dst}"
    nvcc = shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")
    if real and (os.environ.get("NUMINOUS_FLASHKDA_NO_BUILD") == "1" or not nvcc):
        return f"{NAME}: flash_kda build skipped (no CUDA toolkit here, e.g. a CPU preflight)"
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if os.path.exists(os.path.join(dst, "manifest.json")):
                return f"{NAME}: flash_kda present at {dst}"
            work = tempfile.mkdtemp(prefix="fkda_")
            src, tgt = os.path.join(work, "src"), os.path.join(work, "site")
            env = {**os.environ, "FLASH_KDA_CUDA_ARCHS": "90a", "MAX_JOBS": str(min(32, os.cpu_count() or 8))}
            steps = [["git", "clone", "--recursive", REPO_URL, src], ["git", "-C", src, "checkout", rev],
                     ["git", "-C", src, "submodule", "update", "--init", "--recursive"],
                     [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "--target", tgt,
                      src]]
            for c in steps:
                r = run(c, env=env)
                if r.returncode:
                    return (f"{NAME}: flash_kda build skipped ({' '.join(c[:3])} rc={r.returncode}); L4 falls back to "
                            f"Triton: {(r.stderr or '')[-300:]}")
            os.makedirs(dst, exist_ok=True)
            man = {}
            for f in sorted(glob.glob(os.path.join(tgt, "flash_kda*")) + glob.glob(os.path.join(tgt, "flash_kda", "**"),
                                                                                      recursive=True)):
                rel = os.path.relpath(f, tgt)
                if os.path.isdir(f) or "__pycache__" in rel or ".dist-info" in rel:
                    continue
                os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
                shutil.copy2(f, os.path.join(dst, rel))
                man[rel] = hashlib.sha256(open(f, "rb").read()).hexdigest()
            if not any(k.startswith("flash_kda_C") for k in man) or "flash_kda/__init__.py" not in man:
                return f"{NAME}: flash_kda build produced no extension; L4 falls back to Triton"
            json.dump({"rev": rev, "arch": "90a", "files": man}, open(os.path.join(dst, "manifest.json"), "w"))
            return f"{NAME}: flash_kda {rev[:8]} built into {dst} ({len(man)} files)"
    except Exception as e:  # noqa: BLE001
        return f"{NAME}: flash_kda build skipped ({type(e).__name__}: {e}); L4 falls back to Triton"


def apply(root: str, check: bool = False, install_module: bool = True) -> list[str]:
    out = []
    for name, rel, anchor, text in HOOKS:
        path = os.path.join(root, rel)
        src = open(path).read()
        if text in src:
            out.append(f"{name}: already applied")
            continue
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"{NAME}.{name}: anchor found {n} times in {rel}")
        if check:
            out.append(f"{name}: anchor ok")
            continue
        new = src.replace(anchor, text, 1)
        compile(new, path, "exec")
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"{name}: applied to {rel}")
    if install_module and not check:
        dst = os.path.join(root, MODULE_DST)
        shutil.copy(REPO / MODULE_SRC, dst)
        py_compile.compile(dst, doraise=True)
        out.append(f"{NAME}: installed {MODULE_DST}")
    return out



def numinous_preflight(ctx) -> list:
    """Attach-time check (prefill / monolithic role): L4 needs --linear-attn-prefill-backend flashkda (else the hook is
    never reached), the safe gate (gate_lower_bound set; FlashKDA rejects the unbounded gate), K = V = 128, a bf16 or
    fp32 KDA state, and SM90+."""
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
    if not env_on(ENV) or not prefill_like:
        return []
    la = cfg_get(t, "linear_attn_config") or {}
    probs = []
    if arg("linear_attn_prefill_backend") != "flashkda":
        probs.append(f"{NAME}: {ENV}=1 but linear_attn_prefill_backend={arg('linear_attn_prefill_backend')} "
                     "(needs flashkda): inert")
    if cfg_get(la, "gate_lower_bound") is None:
        probs.append(f"{NAME}: {ENV}=1 but the model has no gate_lower_bound (FlashKDA needs the safe gate): inert")
    if cfg_get(la, "head_dim") != 128:
        probs.append(f"{NAME}: {ENV}=1 needs KDA head_dim 128 (got {cfg_get(la, 'head_dim')})")
    if arg("mamba_ssm_dtype") not in (None, "bfloat16", "float32"):
        probs.append(f"{NAME}: {ENV}=1 needs a bf16 / fp32 KDA state (mamba_ssm_dtype={arg('mamba_ssm_dtype')})")
    if sm and sm < 9:
        probs.append(f"{NAME}: FlashKDA needs SM90+ (sm {sm})")
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
    if not a.check:
        print(ensure_package())
    return 0


if __name__ == "__main__":
    sys.exit(main())
