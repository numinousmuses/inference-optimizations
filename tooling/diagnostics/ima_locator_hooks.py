"""Anchored patcher: sync-free IMA locator for SGLang v0.5.20 (94602c9c2b, the rtxsys image tree) and main 172b1b4825.
Agent k3amd for rtxsys, 2026-10-01. Inert unless NUMINOUS_IMA_LOCATOR=1 (mechanism: operations/kernels/ima_locator.py,
installed as sglang/srt/numinous_ima_locator.py).

Every index tensor that a kernel in the hybrid mamba/KDA + DSA serving path uses to address a pool is range-checked
with device ops (no host sync) right before its consumer; the first out-of-range value per site, with its position,
bounds and forward step, lands in pinned host memory ahead of the consumer in stream order, and a daemon thread dumps
it (stderr line "numinous ima_locator HIT {...}" + $NUMINOUS_IMA_LOCATOR_DIR/ima_locator_<pid>.json).

  ima_fwd      model_executor/model_runner.py _forward_raw, before graph replay / eager dispatch:
               req_pool_indices vs req_to_token rows, out_cache_loc vs KV rows (target + draft runners).
  ima_cow      model_runner._maybe_execute_deferred_mamba_cow_and_clear: clear / COW dst vs mamba pool, COW src vs
               mamba pool or the int8 checkpoint pool.
  ima_eager    hybrid_linear_attn_backend._forward_metadata: eager state indices (-1 = skipped row) and track
               destinations of tracked rows.
  ima_replay   hybrid_linear_attn_backend._replay_metadata: the static state-index buffer a decode/verify graph is
               about to replay with.
  ima_verify   speculative/spec_utils.commit_mamba_states_after_verify (generic commit): state slots, accepted step,
               track slots of crossing rows.
  ima_kdaverify  the same for the KDA ReplaySSM fold commit.
  ima_dsa      layers/attention/dsa_backend._forward_flashinfer_sparse_mla: page_table_1 vs KV rows (also guards the
               rtxattn route, which returns from inside this method).
  ima_moe      layers/moe/topk.select_experts after _post_process_topk_ids: topk_ids vs logical / physical experts
               (-1 = masked token) and NaN routing weights (a NaN router is the other classic road to an MoE IMA).

    python3 operations/patches/ima_locator_hooks.py [--root DIR] [--check]
Test: operations/patches/test_ima_locator.py. Missing or repeated anchors fail loudly; files are compile-checked.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from k3state_hooks import Hook, patch_source, sglang_root  # noqa: E402

MODULES = {os.path.join(HERE, "..", "kernels", "ima_locator.py"): "srt/numinous_ima_locator.py"}
RUNNER = "srt/model_executor/model_runner.py"
HLA = "srt/layers/attention/hybrid_linear_attn_backend.py"
SPEC = "srt/speculative/spec_utils.py"
DSA = "srt/layers/attention/dsa_backend.py"
TOPK = "srt/layers/moe/topk.py"

_IMP = "from sglang.srt import numinous_ima_locator as _nil\n"

FWD_ANCHOR = "            # Replay cuda graph if applicable\n"
FWD = ("            # numinous-hook:ima_fwd\n"
       "            " + _IMP +
       "            if _nil.ENABLED and not forward_batch.forward_mode.is_idle():\n"
       "                _nil.tick(self.device)\n"
       "                _nil.check(\"fwd_req_pool_indices\", forward_batch.req_pool_indices, 0,\n"
       "                           self.req_to_token_pool.req_to_token.shape[0])\n"
       "                _nil.check(\"fwd_out_cache_loc\", forward_batch.out_cache_loc, 0,\n"
       "                           _nil.kv_rows(self.token_to_kv_pool))\n"
       + FWD_ANCHOR)

COW_ANCHOR = ("        if (\n"
              "            forward_batch.mamba_clear_indices is not None\n"
              "            and len(forward_batch.mamba_clear_indices) > 0\n"
              "        ):\n")
COW = ("        # numinous-hook:ima_cow\n"
       "        " + _IMP +
       "        if _nil.ENABLED:\n"
       "            _nil_hi = pool.mamba_pool.size + 1\n"
       "            _nil.check(\"mamba_clear\", forward_batch.mamba_clear_indices, 0, _nil_hi)\n"
       "            _nil.check(\"mamba_cow_dst\", forward_batch.mamba_cow_dst_indices, 0, _nil_hi)\n"
       "            _nil.check(\"mamba_cow_src\", forward_batch.mamba_cow_src_indices, 0,\n"
       "                       _nil.ckpt_rows(pool.mamba_ckpt_pool) or _nil_hi)\n"
       + COW_ANCHOR)

EAGER_ANCHOR = "            mamba_cache_indices[_real_bs:] = -1\n"
EAGER = ("        # numinous-hook:ima_eager\n"
         "        " + _IMP +
         "        if _nil.ENABLED:\n"
         "            _nil_hi = self.req_to_token_pool.mamba_pool.size + 1\n"
         "            _nil.check(\"mamba_eager_state\", mamba_cache_indices, 0, _nil_hi, allow_neg1=True)\n"
         "            if forward_batch.mamba_track_mask is not None:\n"
         "                _nil.check(\"mamba_eager_track\", forward_batch.mamba_track_indices, 0, _nil_hi,\n"
         "                           mask=forward_batch.mamba_track_mask)\n")

REPLAY_ANCHOR = "        # Refresh the static track-dest buffer in-place (translated); the captured\n"
REPLAY = ("        # numinous-hook:ima_replay\n"
          "        if not in_capture:\n"
          "            " + _IMP +
          "            if _nil.ENABLED:\n"
          "                _nil.check(\"mamba_replay_state\", self.state_indices_list[bs - 1][:bs], 0,\n"
          "                           self.req_to_token_pool.mamba_pool.size + 1, allow_neg1=True)\n"
          + REPLAY_ANCHOR)

VERIFY_ANCHOR = "        if hasattr(attn_backend, \"update_mamba_state_after_mtp_verify\"):\n"
VERIFY = ("        # numinous-hook:ima_verify\n"
          "        " + _IMP +
          "        if _nil.ENABLED:\n"
          "            _nil_pool = model_runner.req_to_token_pool\n"
          "            _nil_hi = _nil_pool.mamba_pool.size + 1\n"
          "            _nil.check(\"verify_state\", _nil_pool.get_mamba_indices(batch.req_pool_indices[:bs]), 0,\n"
          "                       _nil_hi, allow_neg1=True)\n"
          "            _nil.check(\"verify_step\", last_correct_step_indices, 0, draft_token_num)\n"
          "            if mamba_steps_to_track is not None:\n"
          "                _nil.check(\"verify_track\", batch.mamba_track_indices, 0, _nil_hi,\n"
          "                           mask=mamba_steps_to_track >= 0)\n"
          + VERIFY_ANCHOR)

VERIFY_KDA_ANCHOR = "        commit_kda_replayssm_after_verify(\n"
VERIFY_KDA = ("        # numinous-hook:ima_kdaverify\n"
              "        " + _IMP +
              "        if _nil.ENABLED:\n"
              "            _nil_hi = req_pool.mamba_pool.size + 1\n"
              "            _nil.check(\"verify_state\", state_batch_indices, 0, _nil_hi, allow_neg1=True)\n"
              "            _nil.check(\"verify_step\", last_correct_step_indices, 0, draft_token_num)\n"
              "            if mamba_steps_to_track is not None:\n"
              "                _nil.check(\"verify_track\", mamba_track_indices, 0, _nil_hi,\n"
              "                           mask=mamba_steps_to_track >= 0)\n"
              + VERIFY_KDA_ANCHOR)

DSA_ANCHOR = ("        page_table_1: torch.Tensor,\n"
              "        seq_lens: torch.Tensor,\n"
              "        sm_scale: float,\n"
              "        skip_softmax_threshold_scale_factor: float | None,\n"
              "    ) -> torch.Tensor:\n"
              "        from sglang.kernels.ops.attention.flash_mla_sm120 import (\n")
DSA_HOOK = (DSA_ANCHOR.split("        from sglang.kernels")[0] +
            "        # numinous-hook:ima_dsa\n"
            "        " + _IMP +
            "        if _nil.ENABLED:\n"
            "            _nil.check(\"dsa_page_table\", page_table_1, 0, _nil.token_rows(kv_cache), allow_neg1=True)\n"
            "        from sglang.kernels.ops.attention.flash_mla_sm120 import (\n")

MOE_ANCHOR = ("    get_global_expert_distribution_recorder().on_select_experts(\n"
              "        topk_ids=recorder_topk_ids\n"
              "    )\n")
MOE = ("    # numinous-hook:ima_moe\n"
       "    " + _IMP +
       "    if _nil.ENABLED:\n"
       "        _nil.check(\"moe_topk_ids\", topk_ids, 0,\n"
       "                   router_logits.shape[-1] if expert_location_dispatch_info is None\n"
       "                   else expert_location_dispatch_info.num_physical_experts, allow_neg1=True)\n"
       "        _nil.check(\"moe_topk_weights_nan\", topk_weights.isnan().to(torch.int64), 0, 1)\n"
       + MOE_ANCHOR)

HOOKS = [
    Hook("ima_fwd", RUNNER, FWD_ANCHOR, FWD, replace=True),
    Hook("ima_cow", RUNNER, COW_ANCHOR, COW, replace=True),
    Hook("ima_eager", HLA, EAGER_ANCHOR, EAGER),
    Hook("ima_replay", HLA, REPLAY_ANCHOR, REPLAY, replace=True),
    Hook("ima_verify", SPEC, VERIFY_ANCHOR, VERIFY, replace=True),
    Hook("ima_kdaverify", SPEC, VERIFY_KDA_ANCHOR, VERIFY_KDA, replace=True),
    Hook("ima_dsa", DSA, DSA_ANCHOR, DSA_HOOK, replace=True),
    Hook("ima_moe", TOPK, MOE_ANCHOR, MOE, replace=True),
]


def apply(root: str, check: bool = False) -> list[str]:
    out, problems, new_files = [], [], {}
    by_path: dict[str, list[Hook]] = {}
    for h in HOOKS:
        by_path.setdefault(h.path, []).append(h)
    for rel, hooks in by_path.items():
        path = os.path.join(root, rel)
        src = open(path).read()
        new, lines, probs = patch_source(src, hooks)
        problems += probs
        out += [ln.replace("applied to", "anchor ok in") if check else ln for ln in lines]
        if new != src:
            compile(new, path, "exec")
            new_files[path] = new
    if problems:
        raise SystemExit("numinous ima_locator hooks: " + "; ".join(problems))
    if check:
        return out
    for path, new in new_files.items():
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
    for src, rel in MODULES.items():
        shutil.copyfile(os.path.normpath(src), os.path.join(root, rel))
        out.append(f"module: {rel}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
