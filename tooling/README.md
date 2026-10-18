# Tooling

## [hotpatch/](hotpatch/): edit an installed engine safely

Every feature in this repo reaches SGLang through this layer. It doesn't monkey-patch at runtime. It edits the installed source once at image build or boot, before the server starts.

- A hook is `(name, group, file, anchor, text)`. The anchor must match exactly once in the target file, or the patcher exits. Zero matches means the upstream version moved; two means the anchor is ambiguous.
- The edit is compiled with `compile()` before an atomic `os.replace`, so a broken patch never lands on disk.
- A `# numinous-hook:<name>` marker makes every hook idempotent.
- `--check` reports `anchor ok` or `already applied` without writing. The launcher runs every entry, then runs `--check`, and refuses to boot unless every hook reports `already applied`.
- [`git_series_hooks.py`](hotpatch/git_series_hooks.py) applies git-format patch series under the same contract, using `git apply --check -R` for idempotency.
- Features are off at runtime unless their `NUMINOUS_*` env var is set. So one patched image can serve stock behaviour for the control arm.

About 140 hooks with 1,023 tests use this pattern. The anchor text doubles as the version pin: an upstream rebase means running `--check` and fixing the anchors that moved.

## [serving-gate/](serving-gate/)

- [`cell_gate.py`](serving-gate/cell_gate.py) runs the A/B harness described in [docs/methodology.md](../docs/methodology.md). Arms are the shipped config plus one delta. It replays agent sessions with DP-rank pinning, a warm-up, a flush and a measured round, and takes repeats in alternating order. It also handles probes and an optional profiler.
- [`gate_local.py`](serving-gate/gate_local.py) runs the same gate on a bare box.
- [`dc_bench.py`](serving-gate/dc_bench.py) is a pure-asyncio session load generator: N distinct prefixes, `/generate` with `input_ids`, `routed_dp_rank`. Its summaries are CPU-tested.
- [`preflight.py`](serving-gate/preflight.py) resolves server args against the image's own SGLang code before any GPU time is spent. It catches unread env vars and flags that silently resolve to something else.

## [accuracy-gate/](accuracy-gate/)

[`accuracy_gate.py`](accuracy-gate/accuracy_gate.py) runs teacher-forced KL, top-1 agreement and p99 |dlogprob| over at least 200 prompts, plus GSM8K-200 with McNemar's test. Verdicts are calibrated against a stock-vs-stock noise floor measured on the same node. It also supports vLLM `prompt_logprobs`.

## [rooflines/](rooflines/)

Per-operation byte and FLOP floors at served shapes, computed from checkpoint configs, for DeepSeek V4 Pro, GLM-5.3, GLM-5.3-Flash, Kimi K3 and Gemma 4 31B. Each has tests.

## [diagnostics/](diagnostics/)

- [`sched_trace_hooks.py`](diagnostics/sched_trace_hooks.py): per-rank scheduler stall traces.
- [`ima_locator.py`](diagnostics/ima_locator.py): finds which kernel caused an illegal memory access without adding syncs.
- [`numinous_step_trace.py`](diagnostics/numinous_step_trace.py): per-step attribution.
- [`byte_model.py`](diagnostics/byte_model.py): bytes moved per token per op from a HF config.
