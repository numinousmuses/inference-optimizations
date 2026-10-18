"""Allow the PD decode-side radix cache together with speculative decoding in SGLang v0.5.20: a port of upstream PR
sgl-project/sglang#32170 ("Fix decode-side release crashes and allow decode radix cache with speculative decoding",
open, validated by its author on a production EAGLE PD deployment on v0.5.15), in the anchored-replace style of
glm_tp1_shared_hooks.py: each hook names its file, an exact unique anchor and the replacement; a missing or repeated
anchor fails loudly; hooks are idempotent (marker checked first); a patched file must compile.

Why (measured, research/gpu-evidence/verify/glm-5.3_standard@8xB300-PD_20261002-214636.json, engine logs in
weights-cache:/verify_logs/glm-5.3_standard@8xB300-PD_20261002-214633): GLM-5.3 standard in-box PD (4 prefill +
4 decode B300s) on OR-shaped agent sessions holds 115-128 per user, but its TTFT p90 climbs from 1.6 s at 8 sessions
to 9-21 s at 24-48 because, without a decode radix cache, EVERY turn ships the session's whole ~93K-token prefix
(target + MTP draft KV, ~5.7 GB) over mooncake; the decode ranks show #transfer-req up to 11 of 12 running. With the
decode radix cache the prefill side starts sending at `decode_prefix_len` (prefill.py: req.start_send_idx), so a warm
turn ships only its ~6K uncached tokens. The radix tree is already EAGLE-aware (bigram keys, spec-tail handling in
cache_finished_req / cache_unfinished_req); stock v0.5.20 just refuses the flag combination.

The two hooks:
  1. srt/arg_groups/pd_disaggregation_hook.py: the ValueError for decode radix + --speculative-algorithm becomes a
     warning (hisparse and the fake backend stay hard errors).
  2. srt/disaggregation/decode_kvcache_offload_manager.py: with the decode radix cache on, a finished request is
     released through mem_cache.common.release_kv_cache (tree insert, frees only duplicates / unaligned tail /
     spec over-allocation, balances the prealloc lock) instead of the raw row free that double-frees tree-owned
     prefix slots and the raw protected_size_ decrement. v0.5.20's release is the req.kv API version of the code the
     PR edits; the branch returns before both raw operations, so the chunk-cache path stays stock. That path runs
     only with --disaggregation-decode-enable-offload-kvcache; it is ported so the combination is complete.

    python3 pd_decode_radix_spec_hooks.py                           # patch the installed sglang package
    python3 pd_decode_radix_spec_hooks.py --root /tmp/sgl --check   # report which anchors resolve, change nothing
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sglang_numinous_hooks import sglang_root  # noqa: E402

MARKER = "numinous-hook:pd_decode_radix_spec"


@dataclass(frozen=True)
class Replace:
    name: str
    path: str
    anchor: str
    text: str


HOOKS = [
    Replace(
        "allow_spec", "srt/arg_groups/pd_disaggregation_hook.py",
        "            if cfg.speculative_algorithm is not None:\n"
        "                raise ValueError(\n"
        "                    \"--disaggregation-decode-enable-radix-cache is incompatible \"\n"
        "                    \"with speculative decoding \"\n"
        "                    f\"(--speculative-algorithm {cfg.speculative_algorithm})\"\n"
        "                )\n",
        "            if cfg.speculative_algorithm is not None:\n"
        "                # {marker}: upstream sgl-project/sglang#32170\n"
        "                logger.warning(\n"
        "                    \"EXPERIMENTAL: decode radix cache with speculative decoding \"\n"
        "                    f\"(--speculative-algorithm {cfg.speculative_algorithm}); the radix \"\n"
        "                    \"tree uses EAGLE bigram keys and the eagle-aware release paths.\"\n"
        "                )\n",
    ),
    Replace(
        "offload_release", "srt/disaggregation/decode_kvcache_offload_manager.py",
        "        if req.kv.req_pool_idx is None or req.kv.req_pool_idx == -1:\n"
        "            return\n"
        "\n"
        "        # Released only at request finish; a mid-decode free races with\n",
        "        if req.kv.req_pool_idx is None or req.kv.req_pool_idx == -1:\n"
        "            return\n"
        "\n"
        "        if not self.tree_cache.disable:\n"
        "            # {marker}: upstream sgl-project/sglang#32170. Decode radix\n"
        "            # cache on: the raw row free below would double-free tree-owned\n"
        "            # prefix slots and the raw protected_size_ change would leak the\n"
        "            # prealloc prefix lock; release through the radix-aware path.\n"
        "            from sglang.srt.mem_cache.common import release_kv_cache\n"
        "\n"
        "            release_kv_cache(req, self.tree_cache, is_insert=True)\n"
        "            self.offloaded_state.pop(req, None)\n"
        "            return\n"
        "\n"
        "        # Released only at request finish; a mid-decode free races with\n",
    ),
]


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook; SystemExit if an anchor is missing or repeated. All hooks of a file are applied to
    its text together and the file is written once (never half-patched)."""
    out, missing, files = [], [], {}
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = files.get(path)
        if src is None:
            src = files[path] = open(path).read()
        tag = f"{MARKER} ({h.name})"
        if tag in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        files[path] = src.replace(h.anchor, h.text.replace("{marker}", tag), 1)
        out.append(f"{h.name}: {'anchor ok' if check else 'applied to ' + h.path}")
    if missing:
        raise SystemExit("pd_decode_radix_spec hooks: " + "; ".join(missing))
    if not check:
        for path, new in files.items():
            if new == open(path).read():
                continue
            compile(new, path, "exec")  # never leave a file that does not parse
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
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
