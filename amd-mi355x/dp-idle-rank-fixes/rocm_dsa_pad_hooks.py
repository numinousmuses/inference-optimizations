#!/usr/bin/env python3
"""DSA eager metadata on ROCm: size the page table / indexer logits from the device sequence lengths. Overlay patcher,
inert unless NUMINOUS_DSA_HIP_UNPAD=1.

Bug (measured 2026-10-03, amdglm runs F and G on mi355x8b, 20261002 image, GLM-5.3 Quark, DP attention + EAGLE, eager):
'Memory access fault by GPU' under load. With HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3 the faulting launch is
  aiter deepgemm_fp8_paged_mqa_logits <- sglang aiter_paged_mqa_logits <- dsa_indexer._get_topk_paged
  <- deepseek_nextn (MTP draft) <- eagle_worker_v2._draft_extend_for_decode (eager DRAFT_EXTEND_V2).
Source: eager init_forward_metadata (dsa_backend.py) sizes the page table, and through it the indexer's logits width
(max_seq_len = page-table width), from the host mirror max_seqlen_k = seq_lens_cpu.max() + ..., which lags the device
seq_lens in the spec-v2 overlap flow (draft extend after verify: device lengths already include the accepted tokens).
The AITER gluon kernel stores logits for columns up to each row's device context length with only a >= 0 mask, so a
context longer than the page-table width reads page ids past the row and writes logits past the buffer. CUDA-graph
replay uses the static full-width tables, so only eager forwards (DP idle-rank steps, --disable-cuda-graph, batches
beyond the captured sizes) hit it.

Hook dsapad_maxk (dsa_backend.py init_forward_metadata): max_seqlen_k = max(host value, device max + draft tokens),
clamped to the req_to_token width; one host sync per eager forward, skipped under stream capture.

Rejected hypothesis (kept unapplied below as the record): DP padding rows in q on the HIP AITER path (slice q to
q_offset, pad the top-k back). Run G (slice only) still faulted in the same launch, and run G2 showed the indexer
metadata already has one row per q row there: slicing to q_offset left fewer logits rows than seqlens rows ("lengths
must have one entry per score row" in fast_topk_transform_fused).

    python3 rocm_dsa_pad_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import sys

ENV = "NUMINOUS_DSA_HIP_UNPAD"
TARGET = "srt/layers/attention/dsa/dsa_indexer.py"
_ON = f'__import__("os").environ.get("{ENV}") == "1"'

AITER_ANCHOR = ("        if self.paged_mqa_logits_backend.is_aiter():\n"
                "            logits = aiter_paged_mqa_logits(\n"
                "                q_fp8,\n"
                "                kv_cache_fp8,\n"
                "                weights,\n"
                "                seqlens_32,\n")
AITER_REPL = ("        if self.paged_mqa_logits_backend.is_aiter():\n"
              f"            _np_unpad = {_ON}  # numinous-hook:dsapad_aiter\n"
              "            logits = aiter_paged_mqa_logits(\n"
              "                q_fp8[:q_offset] if _np_unpad else q_fp8,\n"
              "                kv_cache_fp8,\n"
              "                weights[:q_offset] if _np_unpad else weights,\n"
              "                seqlens_32,\n")
RESTORE_ANCHOR = "        if not _is_hip and q_offset < q_fp8.shape[0]:\n"
RESTORE_REPL = (f"        if (not _is_hip or {_ON}) and q_offset < q_fp8.shape[0]:"
                "  # numinous-hook:dsapad_restore\n")

MAXK_TARGET = "srt/layers/attention/dsa_backend.py"
MAXK_ANCHOR = "        # [b, max_seqlen_k]\n"
MAXK_REPL = (f"        if {_ON} and forward_batch.seq_lens.numel() > 0 and not torch.cuda.is_current_stream_capturing():"
             "  # numinous-hook:dsapad_maxk\n"
             "            max_seqlen_k = min(max(max_seqlen_k, int(forward_batch.seq_lens.max().item()) + draft_token_num),\n"
             "                               self.req_to_token_pool.req_to_token.shape[1])\n"
             + MAXK_ANCHOR)

# dsapad_aiter / dsapad_restore (slice q to q_offset on HIP) are NOT applied: run G2 (2026-10-03 22:17) showed that on
# this path the indexer metadata already has one row per (padded) q row, so slicing to q_offset left fewer logits rows
# than seqlens rows ("lengths must have one entry per score row" in fast_topk_transform_fused; the preceding
# _mask_init_and_local_tokens(logits, seqlens_32) can write past the shortened logits -- the likely value4b fault).
# Kept here, unapplied, as the record of the rejected hypothesis.
HOOKS = [(MAXK_TARGET, "dsapad_maxk", MAXK_ANCHOR, MAXK_REPL)]


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, check: bool = False) -> list[str]:
    out, missing, srcs = [], [], {}
    for rel, name, anchor, repl in HOOKS:
        path = os.path.join(root, rel)
        src = srcs.get(path) or open(path).read()
        if f"numinous-hook:{name}" in src:
            out.append(f"{name}: already applied")
            continue
        n = src.count(anchor)
        if n != 1:
            missing.append(f"{name}: anchor found {n} times in {rel}")
            continue
        srcs[path] = src.replace(anchor, repl, 1)
        out.append(f"{name}: {'anchor ok' if check else 'applied'}")
    if missing:
        raise SystemExit("rocm dsa pad hooks: " + "; ".join(missing))
    if not check:
        for path, src in srcs.items():
            compile(src, path, "exec")
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(src)
            os.replace(tmp, path)
    return out


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
