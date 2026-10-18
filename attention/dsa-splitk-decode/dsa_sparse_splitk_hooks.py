"""Split-top-k for FlashMLA's SM90 sparse kernel on small token counts, in SGLang v0.5.20's DSA backend (amdpro).

The why, the math and the sizing are in operations/kernels/dsa_sparse_splitk.py. This file installs that module as
sglang/srt/layers/attention/dsa/numinous_sparse_splitk.py and routes
DeepseekSparseAttnBackend._forward_flashmla_sparse (srt/layers/attention/dsa_backend.py, the flash_mla_sparse_fwd
call) through it when NUMINOUS_DSA_SPARSE_SPLITK=1. The routing applies only when the grid would leave SMs idle
(split_factor > 1) and no attention sink is passed. Unset, or S = 1: the stock call, unchanged.
It composes with glm_kv4_sm90_hooks.py: that path's _attend calls _forward_flashmla_sparse.

    python3 dsa_sparse_splitk_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
MODULE_SRC = HERE.parent / "kernels" / "dsa_sparse_splitk.py"
MODULE_DST = "srt/layers/attention/dsa/numinous_sparse_splitk.py"
BACKEND = "srt/layers/attention/dsa_backend.py"
MARK = "numinous-hook:dsa_sparse_splitk"

ANCHOR = '''        o, _, _ = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv_cache,
            indices=indices_input,
            sm_scale=sm_scale,
            d_v=v_head_dim,
            attn_sink=sink_input,
            topk_length=topk_length,
        )
'''
TEXT = '''        # numinous-hook:dsa_sparse_splitk (amdpro): the sparse prefill kernel launches (h_q/64) x s_q CTAs, so a
        # decode/verify call of a few dozen tokens leaves most SMs idle; split each top-k into S chunks and merge
        from sglang.srt.layers.attention.dsa import numinous_sparse_splitk as _nsk

        _nsk_s = _nsk.maybe_split(
            num_tokens, q_input.shape[1], indices_input.shape[-1], q_input.device, sink_input is not None
        )
        if _nsk_s > 1:
            o = _nsk.sparse_fwd_splitk(
                flash_mla_sparse_fwd, q_input, kv_cache, indices_input, sm_scale, v_head_dim, topk_length, _nsk_s
            )
        else:
            o, _, _ = flash_mla_sparse_fwd(
                q=q_input,
                kv=kv_cache,
                indices=indices_input,
                sm_scale=sm_scale,
                d_v=v_head_dim,
                attn_sink=sink_input,
                topk_length=topk_length,
            )
'''


def _root(arg: str) -> str:
    if arg:
        return arg
    import sglang

    return os.path.dirname(sglang.__file__)


def apply(root: str, check: bool = False) -> list[str]:
    path = os.path.join(root, BACKEND)
    src = open(path).read()
    out = []
    if MARK in src:
        out.append("route: already applied")
    else:
        n = src.count(ANCHOR)
        if n != 1:
            raise SystemExit(f"route: anchor found {n} times in {BACKEND}")
        out.append("route: anchor ok" if check else f"route: applied to {BACKEND}")
        if not check:
            with open(path, "w") as f:
                f.write(src.replace(ANCHOR, TEXT, 1))
            py_compile.compile(path, doraise=True)
    dst = os.path.join(root, MODULE_DST)
    if check:
        out.append(f"module: {'present' if os.path.exists(dst) else 'to install'} ({MODULE_DST})")
    else:
        shutil.copyfile(MODULE_SRC, dst)
        py_compile.compile(dst, doraise=True)
        out.append(f"module: installed {MODULE_DST}")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(_root(a.root), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
