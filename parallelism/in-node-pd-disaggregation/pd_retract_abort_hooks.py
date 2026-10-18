"""Anchored hook: a PD decode retraction whose KV pool has no CPU copy aborts that request instead of the scheduler.
Tree: SGLang v0.5.20 (lmsysorg/sglang IMG_V0520), srt/mem_cache/common.py.

    python3 pd_retract_abort_hooks.py [--root <sglang pkg dir>] [--check]

Bug (measured, rtxattn run 3 on 8xB300, gate_arms pro_pd_mf comment): when the decode pool fills, retract_decode
backs the retracted request's KV up to host memory first. For DeepSeek V4 the backend resolves to "cpu_tensor"
(kv_cache_builder.resolve_decode_retraction_backup: the DSV4 pool is neither MHA nor SWA), and
DeepSeekV4TokenToKVPool has no get_cpu_copy, so req.offload_kv_cache raises NotImplementedError inside the scheduler.
The decode server dies and every session on the node with it. pro_pd_mf avoids retraction at the gate's shape
(--num-reserved-decode-tokens 1024 covers 896-token turns), but production turns can be longer than the reservation
and contexts longer than the gate's, so the pool can still fill.

Fix: in retraction_backup's cpu_tensor branch, a NotImplementedError from the copy returns False. That is SGLang's own
"backup could not be held" result: retract_decode then finishes that request with FINISH_ABORT (HTTP 500, "Retraction
host KV pool exhausted") and the server keeps serving the rest. One request fails instead of the node.
NUMINOUS_PD_RETRACT_ABORT=0 restores the stock behaviour.
"""
from __future__ import annotations

import argparse
import os
import sys

PATH = "srt/mem_cache/common.py"
NAME = "pd_retract_abort"
ANCHOR = ('    if backend == "cpu_tensor":\n'
          "        req.offload_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)\n"
          "        return True\n")
TEXT = ('    if backend == "cpu_tensor":\n'
        "        # numinous-hook:pd_retract_abort (rtxv41): a pool without a CPU copy (DeepSeekV4TokenToKVPool) cannot back\n"
        "        # the request up; report it like an exhausted host pool so retract_decode aborts this request only\n"
        "        try:\n"
        "            req.offload_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)\n"
        "        except NotImplementedError:\n"
        '            if __import__("os").environ.get("NUMINOUS_PD_RETRACT_ABORT", "1") == "0":\n'
        "                raise\n"
        "            req.kv.retraction_backup = None\n"
        '            __import__("logging").getLogger(__name__).warning(\n'
        '                "retraction_backup: the KV pool has no CPU copy; aborting request %s instead of the scheduler",\n'
        '                getattr(req, "rid", "?"))\n'
        "            return False\n"
        "        return True\n")


def apply(root: str, check: bool = False) -> list[str]:
    f = os.path.join(root, PATH)
    src = open(f).read()
    if f"numinous-hook:{NAME}" in src:
        return [f"{NAME}: already applied"]
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"{NAME}: anchor found {src.count(ANCHOR)} times in {PATH}")
    if check:
        return [f"{NAME}: anchor ok in {PATH}"]
    open(f, "w").write(src.replace(ANCHOR, TEXT))
    return [f"{NAME}: applied to {PATH}"]


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
    for ln in apply(root, a.check):
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
