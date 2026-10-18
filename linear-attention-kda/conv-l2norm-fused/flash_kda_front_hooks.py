"""Anchored hooks: the fused KDA prefill front for GLM-5.3-Flash (amdflash, Hopper kernel program 2026-10-05).
Tree: SGLang v0.5.20 (lmsysorg/sglang IMG_V0520).

    python3 flash_kda_front_hooks.py [--root <sglang pkg dir>] [--check]

The kernel: operations/kernels/flash_kda_conv_l2norm.py, installed as srt/layers/attention/linear/numinous_kda_front.py.
It does causal conv1d + SiLU + per-head q/k l2norm + token-major q/k/v store + conv-state update in one Triton launch.

Profile row attacked (F4 prefill profile TP0, ms per 4,096-token step over 34 KDA layers): causal conv1d 5.46,
direct_copy 8.32 (the q/k/v .contiguous() copies), l2norm 2.19.
Measured on 1x H200 at the served shapes (flash_kda_prefill_bench_h200.json):
- stock 0.462 ms per layer at 4,096 tokens -> fused row-loop 0.178 ms: -9.67 ms per step (tile variant 0.229 ms);
- stock 0.897 ms at 8,192 tokens -> fused row-loop 0.340 ms: -18.93 ms per step (tile variant 0.447 ms);
- q/k rel err 3.2e-3 (bf16 ulp) vs the stock kernels; conv state byte-exact.

Two anchors, both inert unless NUMINOUS_KDA_FUSED_FRONT=1:
- front (srt/layers/attention/linear/kda_backend.py, KDABackend.forward_extend:843-857): the stock conv + split +
  unflatten is replaced by numinous_kda_front.conv_l2norm_qkv when its preconditions hold:
  - equal q/k/v head counts and dims (power of 2), conv width 4, bf16 token-major input;
  - a conv-state pool with unit channel stride.
  Otherwise the stock code runs. The call passes numinous_qk_normed=True to the kernel dispatcher.
- no_renorm (srt/layers/attention/linear/kernels/kda_triton.py, TritonKDAKernel.extend's chunk_kda call):
  use_qk_l2norm_in_kernel follows "not numinous_qk_normed", so q/k are not normalized twice. Re-normalizing is
  numerically harmless; skipping it saves the 2 l2norm kernels and the .contiguous() calls.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
NAME = "flash_kda_front"
ENV = "NUMINOUS_KDA_FUSED_FRONT"
MODULE_SRC = "operations/kernels/flash_kda_conv_l2norm.py"
MODULE_DST = "srt/layers/attention/linear/numinous_kda_front.py"
BACKEND = "srt/layers/attention/linear/kda_backend.py"
TRITON = "srt/layers/attention/linear/kernels/kda_triton.py"

FRONT_ANCHOR = (
    "        # Depthwise conv is channel-independent, so one packed call over the\n"
    "        # full qkv width matches the decode path and saves two kernel launches.\n"
    "        qkv = causal_conv1d_fn(\n"
    "            mixed_qkv.transpose(0, 1),\n"
    "            layer.conv_weights,\n"
    "            layer.bias,\n"
    "            activation=\"silu\",\n"
    "            conv_states=conv_states,\n"
    "            has_initial_state=has_initial_state,\n"
    "            cache_indices=cache_indices,\n"
    "            query_start_loc=query_start_loc,\n"
    "            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,\n"
    "        ).transpose(0, 1)\n"
    "        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)\n"
    "\n"
    "        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d\n"
    "        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d\n"
    "        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d\n"
)
FRONT_TEXT = (
    "        # numinous-hook:flash_kda_front (amdflash): one fused Triton launch for conv1d + SiLU + q/k l2norm +\n"
    "        # token-major q/k/v store + conv-state update (NUMINOUS_KDA_FUSED_FRONT=1); stock otherwise\n"
    "        _nkf = __import__(\"sglang.srt.layers.attention.linear.numinous_kda_front\", fromlist=[\"usable\"])\n"
    "        _nkf_raw = mamba_cache_params.conv[0]\n"
    "        _nkf_on = _nkf.usable(layer, mixed_qkv, _nkf_raw)\n"
    "        if _nkf_on:\n"
    "            q, k, v = _nkf.conv_l2norm_qkv(\n"
    "                mixed_qkv,\n"
    "                layer.conv_weights,\n"
    "                layer.bias,\n"
    "                _nkf_raw,\n"
    "                cache_indices,\n"
    "                has_initial_state,\n"
    "                query_start_loc,\n"
    "                layer.q_dim // layer.head_q_dim,\n"
    "                layer.head_q_dim,\n"
    "                seq_lens=forward_batch.extend_seq_lens_cpu,\n"
    "            )\n"
    "        else:\n"
    + "".join("    " + ln + "\n" if ln.strip() else "\n" for ln in FRONT_ANCHOR.splitlines())
)
DISPATCH_ANCHOR = (
    "            track_state=h_track_buf,\n"
    "            track_chunk_idx=(track_chunk_idx if h_track_buf is not None else None),\n"
    "        )\n"
)
DISPATCH_TEXT = (
    "            track_state=h_track_buf,\n"
    "            track_chunk_idx=(track_chunk_idx if h_track_buf is not None else None),\n"
    "            # numinous-hook:flash_kda_front (amdflash): q/k already unit-norm from the fused front\n"
    "            numinous_qk_normed=_nkf_on,\n"
    "        )\n"
)
NORENORM_ANCHOR = (
    "            initial_state_indices=cache_indices,\n"
    "            use_qk_l2norm_in_kernel=True,\n"
    "            cu_seqlens=query_start_loc,\n"
)
NORENORM_TEXT = (
    "            initial_state_indices=cache_indices,\n"
    "            # numinous-hook:flash_kda_front (amdflash): skip the in-kernel l2norm when the fused front did it\n"
    "            use_qk_l2norm_in_kernel=not kwargs.get(\"numinous_qk_normed\", False),\n"
    "            cu_seqlens=query_start_loc,\n"
)
HOOKS = [  # (name, file, anchor, text)
    ("front", BACKEND, FRONT_ANCHOR, FRONT_TEXT),
    ("dispatch", BACKEND, DISPATCH_ANCHOR, DISPATCH_TEXT),
    ("no_renorm", TRITON, NORENORM_ANCHOR, NORENORM_TEXT),
]


def apply(root: str, check: bool = False, install_module: bool = True) -> list[str]:
    """One status line per hook. A hook is applied when its text is present; otherwise its anchor must occur exactly
    once (SystemExit names the hook). Each file is written once, after a compile check."""
    files, out, todo = {}, [], []
    for name, rel, anchor, text in HOOKS:
        path = os.path.join(root, rel)
        src = files.setdefault(path, open(path).read())
        if text in src:
            out.append(f"{name}: already applied")
            continue
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"{NAME}.{name}: anchor found {n} times in {rel}")
        todo.append((name, rel, path, anchor, text))
    if check:
        return out + [f"{name}: anchor ok" for name, *_ in todo]
    for name, rel, path, anchor, text in todo:
        files[path] = files[path].replace(anchor, text, 1)
        out.append(f"{name}: applied to {rel}")
    for path, new in files.items():
        if new != open(path).read():
            compile(new, path, "exec")
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
    if install_module and not check:
        dst = os.path.join(root, MODULE_DST)
        shutil.copy(REPO / MODULE_SRC, dst)
        py_compile.compile(dst, doraise=True)
        out.append(f"{NAME}: installed {MODULE_DST}")
    return out



def numinous_preflight(ctx) -> list:
    """Attach-time check: the fused front runs only for conv width 4 and power-of-two head dims (its usable() gate);
    otherwise NUMINOUS_KDA_FUSED_FRONT=1 is silently inert."""
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
    hd, kw = cfg_get(la, "head_dim"), cfg_get(la, "short_conv_kernel_size")
    probs = []
    if kw != 4:
        probs.append(f"{NAME}: {ENV}=1 but short_conv_kernel_size={kw} (the fused front needs 4): inert")
    if not (isinstance(hd, int) and hd > 0 and hd & (hd - 1) == 0):
        probs.append(f"{NAME}: {ENV}=1 but head_dim={hd} is not a power of two: inert")
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
