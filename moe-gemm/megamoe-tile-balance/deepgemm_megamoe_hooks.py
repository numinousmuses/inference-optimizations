"""Anchored hooks for SGLang v0.5.20's DeepGEMM MegaMoE path (`--moe-a2a-backend megamoe`), style of
sglang_numinous_hooks.py: each hook names a file, an exact anchor and the inserted text; a missing anchor fails
loudly; hooks are idempotent (marker line).

    python3 deepgemm_megamoe_hooks.py                 # patch the installed sglang
    python3 deepgemm_megamoe_hooks.py --root /tmp/sgl --check

Hook `megamoe_balance` (runtime switch NUMINOUS_MEGAMOE_BALANCE=1, default off):
DeepGEMM 0.2.0 picks the kernel's BLOCK_M on each rank from that rank's own token count
(`get_block_config_for_mega_moe`: expected tokens per expert = num_tokens * ranks * topk / experts). Under DP
attention the ranks of one eager (extend / mixed) step hold different counts, e.g. one rank prefilling a 4,096-token
chunk while the others hold 0-96 decode tokens. Every expert then receives ~64 of the prefill rank's tokens, but
the small ranks size their tiles for their own 0-2 tokens per expert (BLOCK_M 16), so each of their experts runs
4+ M blocks and re-reads its weights per block. Measured on 8xB300 (operations/kernels/megamoe_bench.py
--tokens-by-rank): 4096,0,...,0 costs 956 us per layer call on every rank (vs 744 us for 4096,96,...,96).
The hook pads the kernel's row count on each rank to the global per-rank mean of the DP token counts (never below
its own count) with no-route rows (top-k -1: never dispatched; the combine writes zeros, sliced off), so every rank
sizes its tiles for the load its experts actually receive. It acts only on eager calls where the host knows every
rank's count (`get_dp_global_num_tokens`) and the DP group is the EP group; CUDA-graph calls (decode, where DP
padding already equalizes counts) are untouched. One log line on first activation.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Hook:
    name: str
    path: str  # relative to the sglang package directory
    anchor: str  # exact text; the insertion goes after it ("" = end of file)
    text: str

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


BALANCE_CALL = '''    # numinous-hook:megamoe_balance
    # NUMINOUS_MEGAMOE_BALANCE=1: size DeepGEMM's tiles for the global per-expert load (operations/patches/
    # deepgemm_megamoe_hooks.py); padded rows carry top-k -1 and are sliced off below.
    _numinous_rows = _numinous_megamoe_balanced_rows(num_tokens, num_max_tokens_per_rank, ep_group)
    if _numinous_rows > num_tokens:
        buf.topk_idx[num_tokens:_numinous_rows].fill_(-1)
        if _numinous_rows > y.shape[0]:
            y = torch.empty((_numinous_rows, hidden_size), dtype=torch.bfloat16, device=hidden_states.device)
'''

BALANCE_HELPER = '''

# numinous-hook:megamoe_balance_helper
_NUMINOUS_MEGAMOE_BALANCE = None


def _numinous_megamoe_balanced_rows(num_tokens: int, cap: int, ep_group) -> int:
    """Rows to run the MegaMoE kernel with on this rank: max(own, ceil(mean over DP ranks)), or own if off/unknown."""
    global _NUMINOUS_MEGAMOE_BALANCE
    if _NUMINOUS_MEGAMOE_BALANCE is None:
        import os as _os

        _NUMINOUS_MEGAMOE_BALANCE = _os.environ.get("NUMINOUS_MEGAMOE_BALANCE", "0") == "1"
    if not _NUMINOUS_MEGAMOE_BALANCE or get_is_capture_mode() or is_dsa_enable_prefill_cp():
        return num_tokens
    counts = get_dp_global_num_tokens()
    if not counts or len(counts) != ep_group.size():
        return num_tokens
    rows = min(cap, max(num_tokens, -(-sum(counts) // len(counts))))
    if rows > num_tokens and not getattr(_numinous_megamoe_balanced_rows, "_logged", False):
        import logging as _logging

        _numinous_megamoe_balanced_rows._logged = True
        _logging.getLogger(__name__).info(
            "numinous_megamoe_balance active: DP token counts %s, rank rows %d -> %d", list(counts), num_tokens, rows)
    return rows
'''

HOOKS = [
    Hook("megamoe_balance", "srt/layers/moe/mega_moe.py",
         "    y = torch.empty(\n"
         "        (max(num_tokens, 1), hidden_size),\n"
         "        dtype=torch.bfloat16,\n"
         "        device=hidden_states.device,\n"
         "    )\n", BALANCE_CALL),
    Hook("megamoe_balance_helper", "srt/layers/moe/mega_moe.py", "", BALANCE_HELPER),
]


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, names: list[str] | None = None, check: bool = False) -> list[str]:
    """One status line per selected hook; SystemExit if an anchor does not resolve exactly once."""
    out, missing = [], []
    for h in HOOKS:
        if names and h.name not in names:
            continue
        path = os.path.join(root, h.path)
        src = open(path).read()
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        if h.anchor:
            n = src.count(h.anchor)
            if n != 1:
                missing.append(f"{h.name}: anchor found {n} times in {h.path}")
                continue
            new = src.replace(h.anchor, h.anchor + h.text, 1)
        else:
            new = src.rstrip("\n") + "\n" + h.text
        if check:
            out.append(f"{h.name}: anchor ok")
            continue
        compile(new, path, "exec")
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("deepgemm megamoe hooks: " + "; ".join(missing))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="hook names (default: all)")
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.names, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
