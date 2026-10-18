"""Anchored hooks: pdmux split prefill for the DeepSeek-V2 family (GLM-5.3 DSA) on SGLang v0.5.20 (amdglm,
DP-attention lockstep engine fix, agents/findings/amdglm.md section 10, S5).

    python3 glm_pdmux_split_hooks.py [--root <sglang pkg dir>] [--check]

Why: pdmux (multiplex/multiplexing_mixin.py) runs a prefill as SPLIT_PREFILL, a few decoder layers per scheduler
iteration on one green-context SM partition, while decode steps run on the other. model_runner.forward_split_prefill
(model_runner.py:1606-1626) calls model.forward_split_prefill(input_ids, positions, forward_batch, (lo, hi)), and the
ScheduleBatch keeps the ForwardBatch across splits (tp_worker.py:705-731). v0.5.20 implements forward_split_prefill
for glm4 / qwen / gemma-style models only (glm4.py:505-544). GLM-5.3 (GlmMoeDsaForCausalLM, glm4_moe.py:1445)
inherits DeepseekV2ForCausalLM, which has none.

What the hook adds (models/deepseek_v2.py, after DeepseekV2ForCausalLM.forward):
  forward_split_prefill(input_ids, positions, forward_batch, split_interval, input_embeds=None)
  - split 0 embeds (DeepseekV2Model.forward :2735-2741);
  - decoder layers [lo, hi) with the stock per-layer call (:2814-2831): the DSA top-k sharing state
    (IndexTopKShareState, index_topk_share.py:14-60; GLM-5.3 index_topk_freq 4) is carried between splits in
    forward_batch.model_specific_states["dsa_topk"], hidden / residual in forward_batch.hidden_states / .residual
    (forward_batch_info.py:582-587, the fields pdmux defines for this); a per-split BumpAllocator sized to the split;
  - the last split runs the final norm (:2866-2871) and the logits processor (:3052-3064), whose capture_hidden_mode
    hands the hidden states to the EAGLE draft-extend hand-off (S4);
  - the attention-TP input scatter context (get_attn_tp_context().maybe_input_scattered) wraps every split, as the
    stock forward does;
  - refused shapes (raise at the call): pipeline parallelism, two-batch overlap (the TBO tail at :2834-2846 is a
    different loop), aux-hidden capture (EAGLE3 / DFlash layers_to_capture: their packer would have to persist across
    splits), llama-4 scaling (Mistral-Large-3 only).
Inert unless the server runs pdmux (only the pdmux scheduler issues SPLIT_PREFILL batches).
"""
from __future__ import annotations

import argparse
import os
import sys

NAME = "glm_pdmux_split"
MARK = "numinous-hook:glm_pdmux_split"
DSV2 = "srt/models/deepseek_v2.py"

SPLIT_ANCHOR = '''        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer
'''
SPLIT_TEXT = '''        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(  # ''' + MARK + ''' (amdglm): pdmux split prefill
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based decoder layers
        input_embeds: torch.Tensor = None,
    ):
        model = self.model
        start, end = split_interval
        if not (model.pp_group.is_first_rank and model.pp_group.is_last_rank):
            raise NotImplementedError("pdmux split prefill needs pipeline parallel size 1")
        if forward_batch.can_run_tbo:
            raise NotImplementedError("pdmux split prefill does not run two-batch overlap")
        if self.capture_aux_hidden_states or len(model.layers_to_capture) > 0:
            raise NotImplementedError("pdmux split prefill does not carry aux hidden states (EAGLE3 / DFlash)")
        if model.llama_4_scaling_config is not None:
            raise NotImplementedError("pdmux split prefill does not carry llama-4 scaling")
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            if start == 0:
                forward_batch.hidden_states = (
                    model.embed_tokens(input_ids) if input_embeds is None else input_embeds
                )
                forward_batch.residual = None
                forward_batch.model_specific_states = {"dsa_topk": None}
            states = forward_batch.model_specific_states or {"dsa_topk": None}
            hidden_states, residual = forward_batch.hidden_states, forward_batch.residual
            index_topk_share = IndexTopKShareState(forward_batch, states.get("dsa_topk"))
            zero_allocator = BumpAllocator(
                buffer_size=max(1, end - start) * 2, dtype=torch.float32, device=hidden_states.device
            )
            gemm_output_zero_allocator = (
                BumpAllocator(
                    buffer_size=model.gemm_output_zero_allocator_size,
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                if getattr(model, "gemm_output_zero_allocator_size", 0) > 0
                else None
            )
            for i in range(start, end):
                with get_global_expert_distribution_recorder().with_current_layer(i):
                    hidden_states, residual, topk_indices = model.layers[i](
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        zero_allocator,
                        gemm_output_zero_allocator,
                        None,
                        prev_topk_indices=index_topk_share.topk_indices,
                        captured_last_layer_outputs=None,
                        next_full_attention_layer_id=model.next_full_attention_layer_id.get(i),
                    )
                    index_topk_share.update(topk_indices)
            forward_batch.hidden_states, forward_batch.residual = hidden_states, residual
            states["dsa_topk"] = index_topk_share.topk_indices
            forward_batch.model_specific_states = states
            if end < self.config.num_hidden_layers:
                return None
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = model.norm(hidden_states)
                else:
                    hidden_states, _ = model.norm(hidden_states, residual)
            forward_batch.hidden_states = hidden_states
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch, None)

    @property
    def start_layer(self):
        return self.model.start_layer
'''

HOOKS = [("split_prefill", DSV2, SPLIT_ANCHOR, SPLIT_TEXT)]


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook (applied when its text is present; else its anchor must occur exactly once)."""
    files, out, todo = {}, [], []
    for name, rel, anchor, text in HOOKS:
        path = os.path.join(root, rel)
        src = files.setdefault(path, open(path).read())
        if text in src:
            out.append(f"{name}: already applied")
            continue
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"{NAME}.{name}: anchor found {n} times in {rel}")
        todo.append((name, rel, path, anchor, text))
    if check:
        return out + [f"{name}: anchor ok" for name, *_ in todo]
    for name, rel, path, anchor, text in todo:
        files[path] = files[path].replace(anchor, text, 1)
        out.append(f"{name}: applied to {rel}")
    for path, new in files.items():
        if new != open(path).read():
            compile(new, path, "exec")
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
    return out


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
    for line in apply(root, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
