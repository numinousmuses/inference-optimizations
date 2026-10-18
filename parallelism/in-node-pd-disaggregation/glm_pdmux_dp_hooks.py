"""Anchored hooks: pdmux under DP attention, scheduler + worker side (amdglm; DP-attention lockstep engine fix,
agents/findings/amdglm.md section 10, items S1-S4; research/pdmux-lane/README.md items 1-9).

    python3 glm_pdmux_dp_hooks.py [--root <sglang pkg dir>] [--check]

Pairs with rtxv41's glm_pdmux_lane_hooks.py (collective side: per-lane group twins, per-lane DP state, per-lane W4A8
metadata) and amdglm's glm_pdmux_split_hooks.py (S5: DeepseekV2ForCausalLM.forward_split_prefill). Inert unless the
server runs --enable-pdmux with NUMINOUS_PDMUX_DP=1.

Edits (v0.5.20):
  1. multiplex/numinous_pdmux_dp.py (installed): the DP-synchronized pdmux loop, S1-S4.
  2. multiplex/multiplexing_mixin.py event_loop_pdmux (:100): delegates to it when NUMINOUS_PDMUX_DP=1 and the
     scheduler needs the MLP sync (DP attention).
  3. arg_groups/validation_hook.py:144-146: chunked prefill is allowed under pdmux when NUMINOUS_PDMUX_CHUNKED=1 (the
     M chunk is 16,384 = 2,048 per rank; the S3 hand-off keeps an unfinished chunk out of the decode batch). Shared
     with k3b300's K3 port; a tree that already carries it is left as is.
  4. managers/tp_worker.py forward_batch_split_prefill (:705-731):
     - the split ForwardBatch captures the full hidden states when speculative decoding is on (the EAGLE draft-extend
       reads logits_output.hidden_states);
     - an IDLE split is marked on its ForwardBatch (numinous_idle_split) and is not sampled.
  5. managers/tp_worker.py forward_batch_generation (:595-610): returns a stashed split result once
     (batch._numinous_split_result), so the speculative worker's prefill path consumes the finished target instead
     of re-running it (the K3 port's trick).
  6. model_executor/model_runner.py:
     - _forward_raw (:1791-1795, :1829): an IDLE split never replays the decode CUDA graph and goes to
       forward_split_prefill, so it runs the same layer range as the real splits (0 rows);
     - forward_split_prefill (:1612-1613): split 0 gets the eager extend's metadata preparation
       (eager_runner.py:318-331: prepare_forward_batch, init_forward_metadata, the prefill shared-read snapshot). An
       IDLE split drops stale metadata as the eager idle path does (eager_runner.py:454-459).
  7. model_executor/runner/eager_runner.py _execute_extend (:273-298): under pdmux an eager target-verify runs on the
     per-stream decode backend (as eager decode does, :231-250). Stock re-plans the main backend, which the in-flight
     prefill split reads. Same rule as the K3 port's item 7.
  8. multiplex/pdmux_context.py initialize_stream_groups (:137-140), with NUMINOUS_PDMUX_DECODE_STREAM=normal: a split
     group's DECODE stream is a primary-context high-priority stream; its prefill stream stays the green context's.
     - The EAGLE draft graph runners capture once, outside pdmux (eagle_draft_cuda_graph_runner.py:141,
       eagle_draft_extend_cuda_graph_runner.py:136 set enable_pdmux = False), and replay on the D lane's current
       stream. Only a primary-context stream is known to accept those graphs; the target decode graphs are captured
       per group on sg[1] (decode_cuda_graph_runner.py:1041-1050), which is then this stream as well.
     - Decode is HBM-bound, and high priority lets its CTAs go first on every SM. The prefill is the work fenced to
       its green SMs (sgl_kernel greenctx_stream.cu: cuGreenCtxStreamCreate).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

NAME = "glm_pdmux_dp"
MARK = "numinous-hook:glm_pdmux_dp"
ENV = "NUMINOUS_PDMUX_DP"
HERE = pathlib.Path(__file__).resolve().parent
MODULE_SRC = HERE / "numinous_pdmux_dp.py"
MODULE_DST = "srt/multiplex/numinous_pdmux_dp.py"

MIX = "srt/multiplex/multiplexing_mixin.py"
VAL = "srt/arg_groups/validation_hook.py"
TW = "srt/managers/tp_worker.py"
MR = "srt/model_executor/model_runner.py"
ER = "srt/model_executor/runner/eager_runner.py"
CTX = "srt/multiplex/pdmux_context.py"

MIX_ANCHOR = '''    @torch.inference_mode()
    def event_loop_pdmux(self: Scheduler):
        """A scheduler loop for pd multiplexing."""
'''
MIX_TEXT = MIX_ANCHOR + f'''        from sglang.srt.multiplex import numinous_pdmux_dp as _nm_pdmux_dp  # {MARK}

        if _nm_pdmux_dp.enabled() and self.require_mlp_sync:
            return _nm_pdmux_dp.event_loop_pdmux_dp(self)
'''

VAL_ANCHOR = ('        assert cfg.chunked_prefill_size == -1, (\n'
              '            "PD-Multiplexing is not compatible with chunked prefill."\n        )\n')
VAL_TEXT = ('        assert cfg.chunked_prefill_size == -1 or __import__("os").environ.get(  # ' + MARK + '\n'
            '            "NUMINOUS_PDMUX_CHUNKED") == "1", (\n'
            '            "PD-Multiplexing is not compatible with chunked prefill."\n        )\n')
VAL_ALREADY = '"NUMINOUS_PDMUX_CHUNKED") == "1"'      # k3b300's K3 port relaxes the same assert

TW_SPLIT_ANCHOR = '''    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        if batch.split_index == 0:
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_runner,
                return_hidden_states_before_norm=False,
            )
            batch.split_forward_batch = forward_batch

        out = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if logits_output:
'''
TW_SPLIT_TEXT = f'''    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        nm_idle_split = bool(getattr(batch, "_numinous_idle_split", False))  # {MARK}
        if batch.split_index == 0:
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_runner,
                capture_hidden_mode=(
                    None if batch.spec_algorithm.is_none() else CaptureHiddenMode.FULL
                ),
                return_hidden_states_before_norm=False,
            )
            forward_batch.numinous_idle_split = nm_idle_split
            batch.split_forward_batch = forward_batch

        out = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if logits_output and nm_idle_split:
            next_token_ids = torch.empty(0, dtype=torch.int64, device=self.model_runner.device)
        elif logits_output:
'''

TW_GEN_ANCHOR = ("    ) -> GenerationBatchResult:\n        # Get forward batch from schedule batch\n"
                 "        if batch is not None:\n")
TW_GEN_TEXT = ("    ) -> GenerationBatchResult:\n"
               f"        if batch is not None and getattr(batch, \"_numinous_split_result\", None) is not None:  # {MARK}\n"
               "            # pdmux: the last layer split already ran the target; hand its result to the spec worker\n"
               "            stashed, batch._numinous_split_result = batch._numinous_split_result, None\n"
               "            return stashed\n"
               "        # Get forward batch from schedule batch\n"
               "        if batch is not None:\n")
TW_GEN_ALREADY = "stashed, batch._numinous_split_result = batch._numinous_split_result, None"   # the K3 port

MR_GRAPH_ANCHOR = '''            can_run_graph = bool(
                mode_check()
                and self.decode_cuda_graph_runner
                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
            )
'''
MR_GRAPH_TEXT = f'''            can_run_graph = bool(
                mode_check()
                and self.decode_cuda_graph_runner
                and not getattr(forward_batch, "numinous_idle_split", False)  # {MARK}
                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
            )
'''
MR_DISPATCH_ANCHOR = "            if forward_batch.forward_mode.is_split_prefill():\n"
MR_DISPATCH_TEXT = (f"            if forward_batch.forward_mode.is_split_prefill() or getattr(  # {MARK}\n"
                    "                forward_batch, \"numinous_idle_split\", False\n            ):\n")
MR_SPLIT0_ANCHOR = '''        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
'''
MR_SPLIT0_TEXT = f'''        if forward_batch.split_index == 0 or reinit_attn_backend:
            if getattr(forward_batch, "numinous_idle_split", False):  # {MARK}
                # an IDLE split of 0 rows: drop stale metadata, as the eager idle path does
                self.attn_backend.forward_metadata = None
            else:
                # the eager extend's metadata block (eager_runner.py:318-331), not only the backend re-init
                from sglang.srt.model_executor.runner.eager_runner import maybe_publish_prefill_shared_read_done

                if hasattr(self.model, "prepare_forward_batch"):
                    self.model.prepare_forward_batch(forward_batch)
                self.attn_backend.init_forward_metadata(forward_batch)
                self.attn_backend.prepare_prefill_shared_read_snapshot(
                    forward_batch, num_qo_tokens=len(forward_batch.input_ids)
                )
                maybe_publish_prefill_shared_read_done(
                    self, forward_batch, torch.get_device_module(self.device)
                )
'''

ER_ANCHOR = "class EagerRunner(BaseRunner):\n"
ER_TAIL = f'''


# {MARK}
_numinous_dp_stock_execute_extend = EagerRunner._execute_extend


def _numinous_dp_execute_extend(self, forward_batch, pp_proxy_tensors=None):
    """Under pdmux an eager target-verify runs on the per-stream decode backend (as eager decode does), leaving the
    main backend's metadata to the in-flight prefill split."""
    if not (self.enable_pdmux and forward_batch.forward_mode.is_target_verify()):
        return _numinous_dp_stock_execute_extend(self, forward_batch, pp_proxy_tensors)
    model_runner = self.model_runner
    backend, ctx = self._resolve_decode_pdmux()
    saved = model_runner.attn_backend
    model_runner.attn_backend = backend
    try:
        with ctx:
            return _numinous_dp_stock_execute_extend(self, forward_batch, pp_proxy_tensors)
    finally:
        model_runner.attn_backend = saved


EagerRunner._execute_extend = _numinous_dp_execute_extend
'''
ER_ALREADY = "EagerRunner._execute_extend = _numinous_execute_extend"   # the K3 port's identical rule

# v0.5.20's Scheduler keeps gpu_id / tp_size on its ParallelState (self.ps; scheduler.py:777, :1009), but the stock
# pdmux mixin still reads self.gpu_id / self.tp_size: init_pdmux raised AttributeError at boot in the final run (GX0,
# 2026-10-06 04:01) and the stock loop's done-flag check would raise at the first finished split.
MIX_INIT_ANCHOR = "        initialize_stream_groups(self.gpu_id, self.pdmux_config)\n"
MIX_INIT_TEXT = f"        initialize_stream_groups(self.ps.gpu_id, self.pdmux_config)  # {MARK}\n"
MIX_TP_ANCHOR = "                    if flags.item() == self.tp_size:\n"
MIX_TP_TEXT = f"                    if flags.item() == self.ps.tp_size:  # {MARK}\n"

CTX_ANCHOR = '    for prefill_sm, decode_sm in divisions:\n        STREAM_GROUPS.append(\n            (spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id))\n        )\n'
CTX_TEXT = ("    for prefill_sm, decode_sm in divisions:\n"
            f"        _nm_group = spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id)  # {MARK}\n"
            "        if __import__(\"os\").environ.get(\"NUMINOUS_PDMUX_DECODE_STREAM\", \"\") == \"normal\":\n"
            "            # decode on a primary-context high-priority stream (EAGLE draft graphs replay there)\n"
            "            _nm_group = (_nm_group[0], torch.cuda.Stream(gpu_id, priority=-1))\n"
            "        STREAM_GROUPS.append(_nm_group)\n")

HOOKS = [
    ("mixin_delegate", MIX, MIX_ANCHOR, MIX_TEXT, None),
    ("mixin_init_gpu_id", MIX, MIX_INIT_ANCHOR, MIX_INIT_TEXT, None),
    ("mixin_tp_size", MIX, MIX_TP_ANCHOR, MIX_TP_TEXT, None),
    ("ctx_decode_stream", CTX, CTX_ANCHOR, CTX_TEXT, None),
    ("validation_chunked", VAL, VAL_ANCHOR, VAL_TEXT, VAL_ALREADY),
    ("tp_split_capture_idle", TW, TW_SPLIT_ANCHOR, TW_SPLIT_TEXT, None),
    ("tp_stashed_result", TW, TW_GEN_ANCHOR, TW_GEN_TEXT, TW_GEN_ALREADY),
    ("mr_idle_no_graph", MR, MR_GRAPH_ANCHOR, MR_GRAPH_TEXT, None),
    ("mr_idle_split_dispatch", MR, MR_DISPATCH_ANCHOR, MR_DISPATCH_TEXT, None),
    ("mr_split0_prepare", MR, MR_SPLIT0_ANCHOR, MR_SPLIT0_TEXT, None),
]


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook. A hook is applied when its text is present (or another hook in the tree already
    carries the identical rule); otherwise its anchor must occur exactly once."""
    files, out, todo = {}, [], []
    for name, rel, anchor, text, already in HOOKS:
        path = os.path.join(root, rel)
        src = files.setdefault(path, open(path).read())
        if text in src:
            out.append(f"{name}: already applied")
            continue
        if already and already in src:
            out.append(f"{name}: present (identical rule from another hook)")
            continue
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"{NAME}.{name}: anchor found {n} times in {rel}")
        todo.append((name, rel, path, anchor, text))
    er = os.path.join(root, ER)
    er_src = files.setdefault(er, open(er).read())
    er_state = ("already applied" if ER_TAIL.strip() in er_src else
                "present (identical rule from another hook)" if ER_ALREADY in er_src else None)
    if er_state is None and er_src.count(ER_ANCHOR) != 1:
        raise SystemExit(f"{NAME}.eager_verify_backend: EagerRunner class not found once in {ER}")
    mod_dst = os.path.join(root, MODULE_DST)
    mod_state = ("already installed" if os.path.exists(mod_dst) and open(mod_dst).read() == MODULE_SRC.read_text()
                 else None)
    if check:
        return (out + [f"{name}: anchor ok" for name, *_ in todo]
                + [f"eager_verify_backend: {er_state or 'anchor ok'}", f"module: {mod_state or 'install ok'}"])
    for name, rel, path, anchor, text in todo:
        files[path] = files[path].replace(anchor, text, 1)
        out.append(f"{name}: applied to {rel}")
    if er_state is None:
        files[er] = er_src.rstrip("\n") + "\n" + ER_TAIL
        out.append(f"eager_verify_backend: applied to {ER}")
    else:
        out.append(f"eager_verify_backend: {er_state}")
    for path, new in files.items():
        if new != open(path).read():
            compile(new, path, "exec")
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
    if mod_state is None:
        compile(MODULE_SRC.read_text(), str(MODULE_SRC), "exec")
        shutil.copy(MODULE_SRC, mod_dst)
        out.append(f"module: installed {MODULE_DST}")
    else:
        out.append(f"module: {mod_state}")
    return out


def numinous_preflight(ctx) -> list[str]:
    """Attach-time check on the resolved args (operations/hopper/hook_preflight.py contract). Problems only when the
    lever is on (NUMINOUS_PDMUX_DP=1)."""
    env = getattr(ctx, "env", {}) or {}
    if env.get(ENV) != "1":
        return []
    a = getattr(ctx, "args", None)
    if isinstance(a, dict):              # hook_preflight passes the resolved args as a dict (hook_preflight.py:239)
        g = lambda k, d=None: a.get(k, d)  # noqa: E731
    elif a is not None:
        g = lambda k, d=None: getattr(a, k, d)  # noqa: E731
    else:
        g = lambda k, d=None: d  # noqa: E731
    probs = []
    if not g("enable_pdmux", False):
        probs.append(f"{ENV}=1 without --enable-pdmux: the DP pdmux loop never runs")
    if not g("enable_dp_attention", False):
        probs.append(f"{ENV}=1 without DP attention: the stock pdmux loop already covers it")
    if not g("disable_overlap_schedule", False):
        probs.append("pdmux needs --disable-overlap-schedule (validation_hook.py:150-152)")
    if g("enable_mixed_chunk", False):
        probs.append("--enable-mixed-chunk under pdmux: the prefill plan would carry decode requests into the split")
    if g("enable_prefill_delayer", False) or env.get("NUMINOUS_DP_PREFILL_PACE") == "1":
        probs.append("prefill delayer / DP pacing under pdmux: decode no longer waits on prefill; both target the "
                     "lockstep this loop removes (and the delayer negotiates inside get_new_batch_prefill)")
    if (g("chunked_prefill_size", -1) or -1) > 0 and env.get("NUMINOUS_PDMUX_CHUNKED") != "1":
        probs.append("chunked prefill under pdmux needs NUMINOUS_PDMUX_CHUNKED=1")
    if g("enable_two_batch_overlap", False):
        probs.append("two-batch overlap under pdmux is not supported")
    if (g("pp_size", 1) or 1) != 1:
        probs.append("pdmux needs pipeline parallel size 1")
    if env.get("NUMINOUS_PDMUX_LANE_STATE", "1") == "0":
        probs.append("NUMINOUS_PDMUX_LANE_STATE=0: the lanes would share DP buffers (glm_pdmux_lane_hooks)")
    return probs


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
