"""Route before the DP gather (GLM M prefill): each rank runs the MoE router on its own rows. The topk rides on the FP8
gather's scale gatherv, and forward_normal uses it instead of routing all 16,384 gathered rows on every rank.
rtxattn, Hopper kernel program 2026-10-05, lever "GLM fusions".
Kernel/module: operations/kernels/dp_router_piggyback.py (installed as srt/layers/numinous_dp_router_piggyback.py).

Profile row (measured, M5-spec314 cold-wave prefill, per 2,048-tok/rank forward): router GEMM ~4.8 ms + topk ~2.2 ms
on the gathered buffer, of which 7/8 is repeated on every rank.

Hooks (anchored replace, idempotent by marker, fail loudly on missing/repeated anchors, compile-checked). They are
inert unless NUMINOUS_DP_ROUTER_BEFORE_GATHER=1, and they need SGLANG_ENABLE_DP_GATHER_FP8 (the FP8 gatherv path).
  gather   dp_attention._dp_gather_via_all_gatherv_fp8: delegate to gather_fp8_with_router, which widens the scale
           gatherv by 16 fp32 columns. The width is rank-uniform.
  pending  DeepseekV2DecoderLayer.forward: around prepare_mlp, mark this layer's MoE module as the pending router;
           clear the mark afterwards.
  consume  DeepseekV2MoE.forward_normal: when EPLB is off and the stash matches this module and row count, use the
           gathered topk and skip gate + TopK. Otherwise run the stock router.
Lazy dequant (NUMINOUS_DP_LAZY_DEQUANT=1, also needs NUMINOUS_DP_FP8_PASSTHROUGH=1). The gather skips its bf16 dequant
(5.86 ms per M5 prefill pass; 77.3 us/layer measured at 16,384 rows), and every reader takes the FP8 form or
materializes first:
  lazy_moe_entry     DeepseekV2MoE.forward: any dispatch other than forward_normal materializes.
  lazy_normal_check  forward_normal entry: materializes unless the router stash, the passthrough linear and no
                     SBO-shared all hold.
  lazy_runner        cutlass_w4a8_moe: pre_reorder_fp8 reads the FP8 payload (out = fp8(q*s/a1)); dynamic a1 (absmax)
                     materializes.
  A deferred buffer that reaches the next layer untaken raises (set_pending): never served silently.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import shutil
import sys
from dataclasses import dataclass

MARKER = "numinous-hook:dp_router_before_gather"
ENV = "NUMINOUS_DP_ROUTER_BEFORE_GATHER"
MODULE_SRC = pathlib.Path(__file__).resolve().parents[1] / "kernels" / "dp_router_piggyback.py"
MODULE_DST = "srt/layers/numinous_dp_router_piggyback.py"
IMPORT = "from sglang.srt.layers import numinous_dp_router_piggyback as _nm_rbg"
ON = f'__import__("os").environ.get("{ENV}") == "1"'


@dataclass(frozen=True)
class Replace:
    name: str
    rel: str
    old: str
    new: str


GATHER_OLD = '''    The combine leg (reduce_scatterv) stays bf16: NCCL SUM cannot run on fp8."""
'''
GATHER_NEW = GATHER_OLD + f'''    if {ON}:  # {{marker}}
        {IMPORT}

        return _nm_rbg.gather_fp8_with_router(
            global_tokens, local_real, sizes, _DP_GATHER_FP8_GROUP
        )
'''

PENDING_OLD = '''        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        fuse_mlp_allreduce = (
'''
PENDING_NEW = f'''        _nm_rbg_on = {ON}  # {{marker}}
        if _nm_rbg_on:
            {IMPORT}

            _nm_rbg.set_pending(self.mlp if isinstance(self.mlp, DeepseekV2MoE) else None)
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        if _nm_rbg_on:
            _nm_rbg.clear_pending()

        fuse_mlp_allreduce = (
'''

CONSUME_OLD = '''            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
        else:
            pre_quant_input = None
'''
CONSUME_NEW = f'''            # {{marker}}: topk computed per rank before the DP gather, gathered with the FP8 scales
            _nm_pre = None
            if dispatch_info is None and {ON}:
                {IMPORT}

                _nm_pre = _nm_rbg.take(self, hidden_states.shape[0])
            if _nm_pre is not None:
                topk_output = _nm_pre
            else:
                # router_logits: (num_tokens, n_experts)
                router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
                topk_kwargs = (
                    {{"input_ids": input_ids_global}}
                    if getattr(self, "is_hash", False)
                    else {{}}
                )
                topk_output = self.topk(
                    hidden_states,
                    router_logits,
                    expert_location_dispatch_info=dispatch_info,
                    **topk_kwargs,
                )
        else:
            pre_quant_input = None
'''

LAZY_ON = '__import__("os").environ.get("NUMINOUS_DP_LAZY_DEQUANT") == "1"'

MOE_ENTRY_OLD = """        if should_use_mega_moe(self, hidden_states):
            return forward_mega_moe(
"""
MOE_ENTRY_NEW = f"""        if {LAZY_ON}:  # {{marker}}: any dispatch but forward_normal reads bf16
            {IMPORT}

            if _nm_rbg.is_lazy(hidden_states) and (
                should_use_mega_moe(self, hidden_states)
                or self._enable_a2a_moe
                or self._can_dual_stream_graph(hidden_states)
                or (
                    self.alt_stream is not None
                    and self.num_fused_shared_experts == 0
                    and hidden_states.shape[0] > 0
                    and get_is_capture_mode()
                )
            ):
                _nm_rbg.materialize(hidden_states)
""" + MOE_ENTRY_OLD

NORMAL_OLD = """        if hidden_states.shape[0] > 0:
            # Quantize-once (SGLANG_OPT_MOE_QUANT_ONCE): only worthwhile when
"""
NORMAL_NEW = f"""        if hidden_states.shape[0] > 0:
            if {LAZY_ON}:  # {{marker}}: keep the buffer lazy only if every reader takes FP8
                {IMPORT}

                _nm_rbg.lazy_check_normal(self, hidden_states, skip_shared_experts)
            # Quantize-once (SGLANG_OPT_MOE_QUANT_ONCE): only worthwhile when
"""

RUNNER_OLD = """    if a1_scale is None:
        a1_scale = torch.zeros(1, dtype=torch.float32, device=device)
        per_tensor_absmax_fp8(a, a1_scale)

    pre_reorder_for_cutlass_moe(
        a,
        gateup_input,
        src2dst,
        topk_ids,
        a1_scale,
        num_local_experts,
        topk,
        m,
        k,
    )
"""
RUNNER_NEW = f"""    _nm_lazy = None
    if {LAZY_ON}:  # {{marker}}: pre_reorder from the gathered FP8 payload
        {IMPORT}

        if a1_scale is None:
            _nm_rbg.materialize(a)
        else:
            _nm_lazy = _nm_rbg.lazy_take(a)
    if a1_scale is None:
        a1_scale = torch.zeros(1, dtype=torch.float32, device=device)
        per_tensor_absmax_fp8(a, a1_scale)

    if _nm_lazy is not None:
        _nm_rbg.pre_reorder_fp8(
            _nm_lazy[0], _nm_lazy[1], _nm_lazy[2], _nm_lazy[3], gateup_input, src2dst, topk_ids, a1_scale,
            num_local_experts, topk, m, k,
        )
    else:
        pre_reorder_for_cutlass_moe(
            a,
            gateup_input,
            src2dst,
            topk_ids,
            a1_scale,
            num_local_experts,
            topk,
            m,
            k,
        )
"""

DP = "srt/layers/dp_attention.py"
DSV2 = "srt/models/deepseek_v2.py"
W4A8 = "srt/layers/moe/cutlass_w4a8_moe.py"
HOOKS = [Replace("gather", DP, GATHER_OLD, GATHER_NEW), Replace("pending", DSV2, PENDING_OLD, PENDING_NEW),
         Replace("consume", DSV2, CONSUME_OLD, CONSUME_NEW), Replace("lazy_moe_entry", DSV2, MOE_ENTRY_OLD, MOE_ENTRY_NEW),
         Replace("lazy_normal_check", DSV2, NORMAL_OLD, NORMAL_NEW), Replace("lazy_runner", W4A8, RUNNER_OLD, RUNNER_NEW)]


def apply(root: str, check: bool = False, install_module: bool = True) -> list[str]:
    base = pathlib.Path(root)
    out, texts = [], {}
    for h in HOOKS:
        text = texts.setdefault(h.rel, (base / h.rel).read_text())
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
        texts[h.rel] = text.replace(h.old, h.new.replace("{marker}", tag))
        out.append(f"{h.name}: applied to {h.rel}")
    if check:
        return out
    for rel, text in texts.items():
        path = base / rel
        if path.read_text() == text:
            continue
        tmp = path.with_suffix(path.suffix + f".numinous.{os.getpid()}")
        tmp.write_text(text)
        py_compile.compile(str(tmp), doraise=True)
        os.replace(tmp, path)
    if install_module:
        dst = base / MODULE_DST
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MODULE_SRC, dst)
        out.append(f"module: installed {MODULE_DST}")
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
