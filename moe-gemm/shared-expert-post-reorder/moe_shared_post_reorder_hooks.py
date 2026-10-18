"""Fold the shared-expert add into the cutlass W4A8 MoE finalize (post_reorder) for GLM-5.3 (DeepseekV2 MoE, shared
expert not TP1). rtxattn, Hopper kernel program 2026-10-05. Kernel + rationale: operations/kernels/moe_post_reorder_shared.py.

Hooks (anchored replace, idempotent by marker; a missing or repeated anchor fails loudly; a patched file must compile).
They are inert unless NUMINOUS_MOE_SHARED_IN_POST_REORDER=1:
  model   srt/models/deepseek_v2.py forward_normal
          - when enabled (shared expert not TP1, not fused into SBO, not skipped, non-empty batch), compute the shared
            output before the experts (it does not depend on them) and stash it;
          - after the experts, drop the stock `routed += shared` (deepseek_v2.py:1164-1169) only if the runner
            consumed it. Otherwise clear the stash and keep the stock add.
  runner  srt/layers/moe/cutlass_w4a8_moe.py (the post_reorder call that ends cutlass_w4a8_moe)
          - a stashed shared tensor of the output's shape and dtype goes through post_reorder_shared; the call is
            marked consumed. Every other case runs the stock post_reorder_for_cutlass_moe.
  module  operations/kernels/moe_post_reorder_shared.py -> srt/layers/moe/numinous_post_reorder_shared.py
It composes with moe_src2dst_fused_hooks, whose anchor is the preprocess call in the same runner file.

    python3 moe_shared_post_reorder_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import shutil
import sys
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parents[2]
MARKER = "numinous-hook:moe_shared_post_reorder"
ENV = "NUMINOUS_MOE_SHARED_IN_POST_REORDER"
MODULE = ("operations/kernels/moe_post_reorder_shared.py", "srt/layers/moe/numinous_post_reorder_shared.py")


@dataclass(frozen=True)
class Replace:
    name: str
    rel: str
    old: str
    new: str


EXPERTS_CALL = '''        if pre_quant_input is not None:
            final_hidden_states = self.experts(
                hidden_states,
                topk_output,
                pre_quant_input=pre_quant_input,
            )
        else:
            final_hidden_states = self.experts(
                hidden_states,
                topk_output,
            )
'''

MODEL_NEW = '''        # {marker}: compute the shared expert first and let the cutlass W4A8 post_reorder add it
        from sglang.srt.layers.moe import numinous_post_reorder_shared as _npr

        _nm_shared_stashed = False
        if (
            _npr.enabled()
            and not getattr(self, "_shared_expert_tp1", False)
            and not self._fuse_shared_experts_inside_sbo
            and not skip_shared_experts
            and hidden_states.shape[0] > 0
            and getattr(self, "shared_experts", None) is not None
        ):
            if shared_output is None and defer_shared:
                shared_output = self._forward_shared_experts(
                    hidden_states,
                    gemm_output_zero_allocator,
                    pre_quant_input=pre_quant_input,
                )
                defer_shared = False
            if shared_output is not None:
                _npr.stash(shared_output)
                _nm_shared_stashed = True
''' + EXPERTS_CALL + '''        if _nm_shared_stashed:
            if _npr.consumed():
                shared_output = None
            else:
                _npr.clear()
'''

RUNNER_OLD = '''    output = torch.empty_like(a)

    post_reorder_for_cutlass_moe(
        c2,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        num_local_experts,
        topk,
        m,
        k,
        routed_scaling_factor,
    )
    return output
'''

RUNNER_NEW = '''    output = torch.empty_like(a)

    # {marker}: a stashed shared-expert output is added inside the finalize (one pass instead of two)
    from sglang.srt.layers.moe import numinous_post_reorder_shared as _npr

    _nshared = _npr.take() if _npr.enabled() else None
    if _nshared is not None and _nshared.shape == output.shape and _nshared.dtype == output.dtype:
        _npr.post_reorder_shared(
            c2,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            num_local_experts,
            topk,
            m,
            k,
            routed_scaling_factor,
            _nshared.contiguous(),
        )
        _npr.mark_consumed()
    else:
        post_reorder_for_cutlass_moe(
            c2,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            num_local_experts,
            topk,
            m,
            k,
            routed_scaling_factor,
        )
    return output
'''

HOOKS = [
    Replace("model", "srt/models/deepseek_v2.py", EXPERTS_CALL, MODEL_NEW),
    Replace("runner", "srt/layers/moe/cutlass_w4a8_moe.py", RUNNER_OLD, RUNNER_NEW),
]


def apply(root: str, check: bool = False, install_module: bool = True) -> list[str]:
    out = []
    base = pathlib.Path(root)
    for h in HOOKS:
        path = base / h.rel
        text = path.read_text()
        tag = f"{MARKER} ({h.name})"
        if tag in text:
            out.append(f"{h.name}: already applied")
            continue
        n = text.count(h.old)
        if n != 1:
            raise RuntimeError(f"{h.name}: anchor found {n} times in {h.rel} (need exactly 1)")
        if check:
            out.append(f"{h.name}: anchor ok in {h.rel}")
            continue
        new = text.replace(h.old, h.new.replace("{marker}", tag))
        tmp = path.with_suffix(path.suffix + f".numinous.{os.getpid()}")
        tmp.write_text(new)
        py_compile.compile(str(tmp), doraise=True)
        os.replace(tmp, path)
        out.append(f"{h.name}: applied to {h.rel}")
    if install_module and not check:
        dst = base / MODULE[1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / MODULE[0], dst)
        out.append(f"installed {MODULE[1]}")
    return out


def _default_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang not importable; pass --root")
    return list(spec.submodule_search_locations)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or _default_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
