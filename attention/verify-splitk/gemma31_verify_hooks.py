"""Gemma 4 SM90 target verify through the GQA-packed split-K kernel (rtxattn, kernel program A3 + A1).
Kernel + glue: operations/kernels/gemma31_verify.py (installed as srt/layers/attention/numinous_gemma_verify.py).

One anchor; the block is entered when NUMINOUS_GEMMA_VERIFY_GQA=1. NUMINOUS_GEMMA_SWA_EXTEND=1 additionally routes
sliding-window PREFILL / extend through extend_gqa_window (measured 1xH100: 8K chunk 5,381 -> 2,419 us per SWA layer).
  verify  layers/attention/triton_backend.py forward_extend. It sits after the per-layer kv_indptr / window /
          descale selection and before the stock split-KV verify block (gfx95-only). A topk-1 TARGET_VERIFY with a
          constant draft length, page size 1, no sinks / softcap / score_mod goes to verify_gqa, for SWA (window)
          and full layers alike. Anything else falls through to the stock kernels.
Composes with gemma31_kv1_hooks: KV1 full layers return earlier (to verify_kv1), and the rest reach this anchor.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import shutil
import sys

MARKER = "numinous-hook:gemma31_verify"
MODULE_SRC = pathlib.Path(__file__).resolve().parents[1] / "kernels" / "gemma31_verify.py"
MODULE_DST = "srt/layers/attention/numinous_gemma_verify.py"
REL = "srt/layers/attention/triton_backend.py"
OLD = """        # Split-KV EAGLE-verify fast path (ROCm/Triton). On target-verify
"""
NEW = """        if __import__("os").environ.get("NUMINOUS_GEMMA_VERIFY_GQA") == "1":  # {marker}
            from sglang.srt.layers.attention import numinous_gemma_verify as _ngv

            if _ngv.maybe_verify(
                self, q, k, v, o, layer, forward_batch, kv_indptr, kv_indices, sliding_window_size,
                k_descale, v_descale, logits_soft_cap, sinks, score_mod,
            ):
                return o
            if _ngv.maybe_extend_window(
                self, q, k, v, o, layer, forward_batch, kv_indptr, kv_indices, sliding_window_size,
                k_descale, v_descale, logits_soft_cap, sinks, score_mod,
            ):
                return o
""" + OLD


def apply(root: str, check: bool = False, install_module: bool = True) -> list[str]:
    path = pathlib.Path(root) / REL
    text = path.read_text()
    tag = f"{MARKER} (verify)"
    out = []
    if tag in text:
        out.append("verify: already applied")
    else:
        n = text.count(OLD)
        if n != 1:
            raise RuntimeError(f"verify: anchor found {n} times in {REL} (need exactly 1)")
        if check:
            return ["verify: anchor ok in " + REL]
        new = text.replace(OLD, NEW.replace("{marker}", tag))
        tmp = path.with_suffix(path.suffix + f".numinous.{os.getpid()}")
        tmp.write_text(new)
        py_compile.compile(str(tmp), doraise=True)
        os.replace(tmp, path)
        out.append("verify: applied to " + REL)
    if install_module and not check:
        dst = pathlib.Path(root) / MODULE_DST
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MODULE_SRC, dst)
        out.append(f"module: installed {MODULE_DST}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
