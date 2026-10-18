"""Prefill/decode disaggregation on one MI355X node without RDMA NICs: anchored hook for the overlay copy.

    python3 rocm_pd_hooks.py [--root <sglang pkg dir>] [--check]

  pd_draft_probs  speculative/eagle_worker_v2.py draft_forward: on ROCm EAGLE auto-enables rejection sampling (there is no
             HIP sampling-verify kernel), which stacks the first draft step's full-vocab distribution from spec_info.
             In PD the prefill node hands over topk_p / topk_index / hidden states but not that distribution, so the
             decode side crashed (torch.stack got None). With topk 1 the first proposal is deterministic, and the exact
             speculative rejection rule for a deterministic proposal is q = one-hot(proposed token): accept with p(x),
             resample from p without x otherwise. The hook builds that one-hot, sized like the later steps' probs.
  mori_stats  disaggregation/mori/conn.py: NUMINOUS_MORI_STATS=1 logs KV transfer throughput per prefill rank every
             20 s (bytes, wall and busy GB/s, submit / wait ms, item size); module rocm_mori_xfer_stats.py.
  mori_xgmi  disaggregation/mori/conn.py _init_engine: NUMINOUS_MORI_XGMI=1 creates MORI-IO's XGMI backend instead of
             the hard-coded RDMA one. The Daytona box has no /dev/infiniband; XGMI moves KV GPU->GPU inside the node
             with the same register_memory / batch_write / wait_all calls (operations/kernels/rocm_mori_xgmi_check.py:
             byte-exact, 28.8 GB/s at 2 MB items, 1.3 GB/s at 36 KB items, warm).
"""
from __future__ import annotations

import argparse
import os
import sys

CONN = "srt/disaggregation/mori/conn.py"
EAGLE = "srt/speculative/eagle_worker_v2.py"
STATS_MODULE = ("operations/patches/rocm_mori_xfer_stats.py", "srt/disaggregation/mori/numinous_xfer_stats.py")
ST_PAIRS = [
    ("        statuses = self._submit_kv_transfer(\n            room,\n",
     "        from sglang.srt.disaggregation.mori import numinous_xfer_stats as _nm_xs  # numinous-hook:mori_stats\n"
     "        if _nm_xs.ENABLED:\n            _nm_xs.chunk_begin()\n"
     "        statuses = self._submit_kv_transfer(\n            room,\n"),
    ("        failure_reason = self._wait_transfer_completion(statuses)\n",
     "        if _nm_xs.ENABLED:\n            _nm_xs.chunk_submitted()\n"
     "        failure_reason = self._wait_transfer_completion(statuses)\n"
     "        if _nm_xs.ENABLED:\n            _nm_xs.chunk_end()\n"),
    ("        if plan.empty():\n            return []\n\n        transfer_uid = self.engine.allocate_transfer_uid()\n",
     "        if plan.empty():\n            return []\n"
     "        from sglang.srt.disaggregation.mori import numinous_xfer_stats as _nm_xs2\n"
     "        if _nm_xs2.ENABLED:\n            _nm_xs2.planned(plan.sizes)\n"
     "\n        transfer_uid = self.engine.allocate_transfer_uid()\n"),
]
DP_ANCHOR1 = "            draft_probs_list: List[torch.Tensor] = [spec_info.draft_probs]\n"
DP_TEXT1 = ("            draft_probs_list: List[torch.Tensor] = [spec_info.draft_probs]\n"
            "            _nm_first_idx = topk_index  # numinous-hook:pd_draft_probs (PD: first-step draft probs missing)\n")
DP_ANCHOR2 = ("        draft_probs = (\n"
              "            torch.stack(draft_probs_list, dim=1)\n")
DP_TEXT2 = ("        if (get_spec().speculative_use_rejection_sampling and draft_probs_list[0] is None\n"
            "                and len(draft_probs_list) > 1):  # numinous-hook:pd_draft_probs_fill\n"
            "            _nm_ref = draft_probs_list[1]\n"
            "            _nm_q = torch.zeros_like(_nm_ref)\n"
            "            _nm_q.scatter_(1, _nm_first_idx[:, :1].long(), 1.0)\n"
            "            draft_probs_list[0] = _nm_q\n"
            "        draft_probs = (\n"
            "            torch.stack(draft_probs_list, dim=1)\n")
ANCHOR = ("        engine.create_backend(BackendType.RDMA, rdma_cfg)\n"
          "        actual_port = engine.get_engine_desc().port\n"
          "        assert actual_port > 0, f\"Failed to bind port for engine {engine_key}\"\n")
TEXT = '''        # numinous-hook:mori_xgmi
        if os.environ.get("NUMINOUS_MORI_XGMI") == "1":
            # XGMI moves bytes through IPC handles and opens no TCP port (engine desc port stays 0)
            from mori.io import XgmiBackendConfig as _NmXgmi

            engine.create_backend(BackendType.XGMI, _NmXgmi())
            actual_port = engine.get_engine_desc().port
        else:
            engine.create_backend(BackendType.RDMA, rdma_cfg)
            actual_port = engine.get_engine_desc().port
            assert actual_port > 0, f"Failed to bind port for engine {engine_key}"
'''


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    return list(spec.submodule_search_locations)[0]


def _patch(root: str, rel: str, pairs, marker: str, check: bool) -> str:
    full = os.path.join(root, rel)
    src = open(full).read()
    if marker in src:
        return f"{marker}: already applied"
    for anchor, _ in pairs:
        if src.count(anchor) != 1:
            raise SystemExit(f"rocm pd hooks: anchor found {src.count(anchor)} times in {rel}")
    if check:
        return f"{marker}: anchor ok"
    for anchor, text in pairs:
        src = src.replace(anchor, text, 1)
    compile(src, full, "exec")
    tmp = f"{full}.numinous.{os.getpid()}"
    open(tmp, "w").write(src)
    os.replace(tmp, full)
    return f"{marker}: applied to {rel}"


def apply(root: str, check: bool = False) -> list[str]:
    out = [_patch(root, CONN, [(ANCHOR, TEXT)], "numinous-hook:mori_xgmi", check),
           _patch(root, CONN, ST_PAIRS, "numinous-hook:mori_stats", check),
           _patch(root, EAGLE, [(DP_ANCHOR1, DP_TEXT1), (DP_ANCHOR2, DP_TEXT2)], "numinous-hook:pd_draft_probs", check)]
    if not check:
        import pathlib
        import shutil
        repo = pathlib.Path(__file__).resolve().parents[2]
        shutil.copy(repo / STATS_MODULE[0], os.path.join(root, STATS_MODULE[1]))
        out.append(f"installed {STATS_MODULE[1]}")
    return [l.replace("numinous-hook:", "") for l in out]


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
