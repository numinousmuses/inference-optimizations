"""Frozen-KV MTP draft without the per-step D2H sync (SGLang v0.5.20), amdpro 2026-10-06, Gemma 4 31B H100.

measured (rtxv41, run2b trace, 1xH100, 32 sessions, operations/gemma31_h100/trace_gaps.py):
  - 3.71 ms of GPU idle per verify step (44.5 ms) in gaps >= 20 us;
  - the host is preparing graph inputs (DtoD copies into graph buffers 1.56 ms, kv-split counts, kv indices, a pinned
    DtoH) and calling cudaGraphLaunch (1.74 ms in flight).
  - The draft -> verify hand-off is not overlapped.
source_code (v0.5.20):
  - FrozenKVMTPDraftWorker.draft runs `batch.seq_lens_sum = torch.sum(batch.seq_lens).item()` on every decode step
    (speculative/frozen_kv_mtp_worker_v2.py:467). That D2H waits for the previous iteration's verify and draft-extend
    to finish, so all host glue for this step runs while the GPU waits.
  - EAGLE v2 removed exactly this sync (speculative/eagle_worker_common.py:356-364: "build_tree_kernel uses
    seq_lens_sum only to size the (non-preallocated) FULL_MASK tree mask; over-size is safe. Skip per-iter
    .sum().item() D2H via UB").
  - The other consumers already accept None or an over-size: the triton backend's draft / verify metadata
    (layers/attention/triton_backend.py:802-807, 908-913, 2387-2391: "seq_lens_sum may be None; over-allocate is
    safe").
What the hook changes (NUMINOUS_FROZEN_MTP_NOSYNC=1; unset = stock):
  - draft() sets batch.seq_lens_sum without a D2H, always to EAGLE v2's upper bound bs x max_context_len. It is an
    int because the draft graph replay adds padding to it (frozen_kv_mtp_cuda_graph_runner.py:445-449).
  - The CPU mirror is not used: its staleness under the overlap loop is not a contract, and an under-size would
    corrupt.
  - build_tree_kernel_efficient, the draft metadata and the verify input all receive that value.
  - Over-allocation only; nothing is under-sized.
Cost: transient over-allocation per step (Gemma, 32 sessions, context 262,144): FULL_MASK tree mask 32 x 262,144 x 6
B = 50 MB memset (~17 us), draft / verify kv_indices ~34 MB each, written ragged. All of it comes from the caching
allocator.
derived: the host can run one step ahead again, hiding up to the 3.71 ms idle of a 44.5 ms verify step (~8%). The
graph-launch time itself (~1.7 ms) stays, now off the critical path.

    python3 frozen_mtp_nosync_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import sys

ENV = "NUMINOUS_FROZEN_MTP_NOSYNC"
WORKER = "srt/speculative/frozen_kv_mtp_worker_v2.py"

A_SUM = '''        spec_info.positions = self._position_for_batch(batch)
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
'''
T_SUM = '''        spec_info.positions = self._position_for_batch(batch)
        # numinous-hook:frozen_mtp_nosync (amdpro): no per-step D2H here (EAGLE v2 dropped the same sync,
        # eagle_worker_common.py:356-364); a CPU-mirror upper bound or None (every consumer over-allocates safely)
        if __import__("os").environ.get("NUMINOUS_FROZEN_MTP_NOSYNC", "0") == "1":
            # always an int (the graph replay adds padding to it, frozen_kv_mtp_cuda_graph_runner.py:445-449), always
            # EAGLE v2's upper bound bs x max_context_len: the CPU mirror's staleness under the overlap loop is not a
            # contract, and an over-size is safe for every consumer while an under-size is not
            batch.seq_lens_sum = int(batch.seq_lens.shape[0]) * int(
                getattr(self.target_worker.model_runner.attn_backend, "max_context_len", 0)
                or self.target_worker.model_runner.model_config.context_len
            )
        else:
            batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
'''
HOOKS = [("frozen_mtp_nosync", WORKER, A_SUM, T_SUM)]


def numinous_preflight(ctx) -> list:
    """CPU preflight (operations/hopper/hook_preflight.py): where NUMINOUS_FROZEN_MTP_NOSYNC=1 is inert."""
    env = ctx.env
    if env.get(ENV, "0") != "1":
        return []
    a = ctx.args
    algo = str(a.get("speculative_algorithm") or "").upper()
    if algo not in ("NEXTN", "FROZEN_KV_MTP"):
        return [f"{ENV}=1 with speculative_algorithm {algo or 'none'}: only the frozen-KV MTP draft is patched, inert"]
    return []


def _root(arg: str) -> str:
    if arg:
        return arg
    import sglang

    return os.path.dirname(sglang.__file__)


def apply(root: str, check: bool = False) -> list[str]:
    out = []
    path = os.path.join(root, WORKER)
    src = open(path).read()
    new = src
    for name, _rel, anchor, text in HOOKS:
        if text in new:
            out.append(f"{name}: already applied")
            continue
        n = new.count(anchor)
        if n != 1:
            raise SystemExit(f"{name}: anchor found {n} times in {WORKER}")
        out.append(f"{name}: anchor ok" if check else f"{name}: applied to {WORKER}")
        new = new.replace(anchor, text, 1)
    if not check and new != src:
        compile(new, path, "exec")
        with open(path, "w") as f:
            f.write(new)
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
