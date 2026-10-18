# How results were measured

There are three kinds of number in this repo. Each README says which kind it is quoting.

## 1. Served A/B (the headline numbers)

Tool: [`tooling/serving-gate/cell_gate.py`](../tooling/serving-gate/cell_gate.py) on cloud nodes, or [`gate_local.py`](../tooling/serving-gate/gate_local.py) on a bare box.

- **Arms.** An arm is the shipped launch config plus one change, which can be a flag, an env var or a patch. The baseline arm is stock `lmsysorg/sglang:v0.5.20` with the same flags.
- **One image, env toggles.** Wherever possible both arms run the same image with a feature turned off and on. That way the node, driver and weights cancel out.
- **Traffic.** Recorded multi-turn coding-agent sessions, 3 or 8 growing turns, 46k to 104k-token prefixes, replayed at a fixed concurrency. Requests are pinned to DP ranks so every rank sees the same load.
- **Protocol.** Warm-up, cache flush, then a measured round. Two repeats with arm order reversed on the second, and probes for retrieval correctness.
- **Reported as** steady aggregate output tok/s / warm per-user tok/s / TTFT p99 in seconds.
- **Records.** Every run writes a JSON with the config digest, the image digest, flags, env, patches, per-run timings, KV pool size and speculative acceptance. A subset is in [`evidence/gate/`](../evidence/gate/).

## 2. Accuracy gate

Tool: [`tooling/accuracy-gate/accuracy_gate.py`](../tooling/accuracy-gate/accuracy_gate.py)

- At least 200 prompts drawn from the same sessions, teacher-forced through both arms. The gate compares per-position KL, top-1 agreement and the p99 absolute log-prob difference.
- GSM8K-200 on both arms, compared with McNemar's test on the answers that changed.
- **Noise calibration.** Stock is run against stock on the same node across two boots, and that sets the floor. A change passes if every metric is within 3x of the floor. The floor varies a lot by model: GLM-5.3's DSA top-2048 selection flips on tiny numeric changes, so its floor KL is 0.18 against 0.013 for DeepSeek V4 Pro. A fixed threshold would fail GLM every time, or wave bad DeepSeek changes through.

## 3. Kernel microbenchmarks

- CUDA-graph-timed per-layer calls at served shapes, compared against the op the kernel replaces on the same GPU.
- Correctness first: every kernel is checked against an fp32 or torch reference, with tolerances and SQNR or bit-exact checks listed in its tests.
- Rooflines in [`tooling/rooflines/`](../tooling/rooflines/) give the bytes and FLOPs floor per op at served shapes, so a speedup can be read as a fraction of the floor.

## Rules I held to

- A kernel win isn't a serving win until it shows up in the served A/B. Several kernels that were 2 to 3x faster on their own changed nothing in the engine, or made it slower. Those are listed as such.
- Kernel tables come from runs on the target GPU, never extrapolated from another GPU.
- Results that didn't reproduce were withdrawn, not averaged in.
