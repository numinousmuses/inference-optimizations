"""rtxsys hook (Gemma 4 31B on H100, kernel program 2026-10-06): GQA split-K decode attention for the frozen-KV
MTP draft.

rtxv41's run1 composed profile: "Draft attention 3.3 ms per 32-session verify step (floor ~1)", owner rtxsys.
- The draft's 4 layers x 5 steps go through TritonAttnBackend.forward_decode -> decode_attention_fwd
  (_fwd_grouped_kernel_stage1: BLOCK_N 32, 2 stages, heads padded to 16; ~45% of HBM on the stock profile).
- With NUMINOUS_DRAFT_DECODE=1, operations/kernels/gemma31/rtx_draft_decode.py (installed as
  srt/layers/attention/numinous_draft_decode.py) serves those calls over the same kv_indptr / kv_indices.
- Derived: ~-1.5 ms per verify step (45% -> ~80% of HBM on ~1 GB per draft step x 5).

Hook (anchored, idempotent, the patched file must compile): triton_backend.py forward_decode, right before the stock
non-DCP decode call. The anchor is the call itself, which rtxattn's gemma31_kv1_hooks also prefixes. Both blocks
keep the call, so they compose in either order. KV1 layers (K aliased to V) are declined by the kernel's own
aliasing check.

    python3 gemma31_draft_decode_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sglang_numinous_hooks import sglang_root  # noqa: E402

MARKER = "numinous-hook:gemma31_draft_decode"
ENV = "NUMINOUS_DRAFT_DECODE"
PATH = "srt/layers/attention/triton_backend.py"
INSTALL = [("../kernels/gemma31/rtx_draft_decode.py", "srt/layers/attention/numinous_draft_decode.py")]
CELL_INSTALL = [("operations/kernels/gemma31/rtx_draft_decode.py", "srt/layers/attention/numinous_draft_decode.py")]

ANCHOR = """        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
"""
BLOCK = """        if __import__("os").environ.get("NUMINOUS_DRAFT_DECODE") == "1":  # {marker} (decode)
            from sglang.srt.layers.attention import numinous_draft_decode as _ndd

            if _ndd.maybe_decode(
                self, q, o, layer, forward_batch, kv_indptr, kv_indices, k_descale, v_descale,
                logits_soft_cap, sinks, score_mod,
            ):
                return o
"""


def numinous_preflight(ctx) -> list:
    """Where NUMINOUS_DRAFT_DECODE=1 never engages."""
    if (ctx.env or {}).get(ENV) != "1":
        return []
    out, a = [], ctx.args or {}
    if str(a.get("attention_backend") or "") not in ("triton", ""):
        out.append(f"{ENV}=1 with --attention-backend {a.get('attention_backend')}: the hook patches the triton "
                   "backend's decode, inert")
    if int(a.get("page_size") or 1) != 1:
        out.append(f"{ENV}=1 with --page-size {a.get('page_size')}: page size 1 only, inert")
    dev = getattr(ctx, "device", None) or {}
    if dev and dev.get("is_hip"):
        out.append(f"{ENV}=1 on ROCm: e4m3fn KV only (fnuz pools differ), inert by design")
    return out


def apply(root: str, check: bool = False) -> list:
    out = []
    if not check:
        for rel, dst in INSTALL:
            shutil.copy(os.path.normpath(os.path.join(HERE, rel)), os.path.join(root, dst))
            out.append(f"installed {dst}")
    path = os.path.join(root, PATH)
    src = open(path).read()
    tag = f"{MARKER} (decode)"
    if tag in src:
        out.append("decode: already applied")
        return out
    n = src.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"gemma31_draft_decode hooks: decode anchor found {n} times in {PATH}")
    new = src.replace(ANCHOR, BLOCK.replace("{marker}", MARKER) + ANCHOR, 1)
    out.append(f"decode: {'anchor ok' if check else 'applied to ' + PATH}")
    if not check:
        compile(new, path, "exec")
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
