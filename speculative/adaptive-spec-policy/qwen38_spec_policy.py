"""Qwen 3.8-27B speculation candidates on RTX PRO 6000: MTP depth ladder, the three real drafts on the weights
volume, and a load-adaptive DSpark row policy reusing `operations/kernels/numinous_spec_policy.py` (agent spec's;
not edited here, only imported).

Measured accept lengths (not re-run here; cited from pre-existing, committed evidence, real 60k-token agent
sessions, `research/gpu-evidence/dc_arch/qwen38_drafts_real_*_run_20260926-035913.json`, `unsloth/Qwen3.8-27B-
NVFP4` on one RTX PRO 6000):

| draft | sessions=1 | sessions=2 | sessions=4 |
|---|---|---|---|
| control (EAGLE, built-in MTP head, gamma 3) | 3.63 | 3.38 | 3.67 |
| DSpark NVFP4 (`gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`, block 7) | 4.80 | 4.08 | 3.51 |
| DSpark bf16 (`rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70`, block 7) | 4.40 | 3.95 | 3.76 |
| DFlash2 FP8 (`tcclaviger/Qwen3.8-27B-DFlash2-FP8`, block 8, needs a nightly on that run; DFLASH ships in
  v0.5.20 now, see `preflight` note below) | 2.97 | 3.06 | 3.19 |

DSpark NVFP4 wins at low concurrency (1.32x control at c=1) and loses it back by c=4 (0.94x); DFlash2 does the
opposite (0.82x at c=1, 1.16x at its own control by c=4, though its control was measured on a different image).
Both drafts had to drop `--enable-linear-replayssm-spec` (DeepSeek/Kimi-only ring path; the engine refuses to
start otherwise) and shrink `--max-mamba-cache-size` from 340 to 64 (see the cited evidence file's comment): a
block drafter's full per-slot recurrent-state snapshots do not fit the replay ring's memory budget the way the
built-in MTP head's smaller verify window does.

`preflight.py --cell qwen3.8-27b/standard --arms v20_stock` resolution notes (v0.5.20, this box), all clean:
  - DFLASH is a v0.5.20 builtin (`--speculative-algorithm DFLASH`); the dc_arch evidence's "needs a nightly" note
    is stale for this engine version.
  - DSpark and DFlash both force `speculative_num_steps` to 1 regardless of what is passed (an informational log
    line, not a rejection); DSpark additionally requires `speculative_num_draft_tokens == dspark_block_size + 1`
    (rejects otherwise). The recipes below set both correctly so preflight reports zero warnings.

Why a load-adaptive DSpark row policy is a WORSE fit here than on DeepSeek V4 Pro (source_code + the measured
accept-length table above, `derived` reasoning): `numinous_spec_policy.py` caps verify rows per request as
concurrency rises because Pro's step cost scales steeply with total verify tokens (a fine-grained MoE: more
verify tokens route to more of 384 experts, `research/qwen38-moe-kernel.md`-style effect). Qwen 3.8-27B is
**dense** (no MoE): a decode step's dominant cost is one pass over ~22-29 GB of weight, which does not grow with a
few more verify tokens per request until the step becomes compute- rather than memory-bound. The accept-length
table itself is the evidence: DSpark's accept length falls from 4.80 to 3.51 over c=1 to c=4 with the SAME fixed
gamma (no adaptive cap was active in that run), which is the ordinary concurrency-dilutes-acceptance curve, not a
step-cost cliff -- there is no measured point in this table where fewer rows raises rank throughput. This module
still exposes `build_qwen38_policy` (reusing the shared math) as a documented, testable candidate, with a `_roofline_cost_table`
built from the checkpoint's own disk bytes and a measured HBM bandwidth probe (labelled `derived`, not `measured`,
since no real step-cost profile exists without booting a server), so the coordinator can decide with the actual
curve once one is profiled (`SGLANG_DSPARK_ENABLE_SPS_RECORD=1`, the `sps_profile` arm in `gate_arms.py`) instead
of by construction.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from operations.kernels.numinous_spec_policy import (
    AcceptanceModel, DSparkLoadPolicy, StepCostTable, rank_tokens_per_second,
)

# measured (dc_arch/qwen38_drafts_real_*_run_20260926-035913.json, cited above, not re-run by this module)
ACCEPT_LENGTHS = {
    "control_eagle_gamma3": {1: 3.63, 2: 3.38, 4: 3.67},
    "dspark_nvfp4_gamma7": {1: 4.80, 2: 4.08, 4: 3.51},
    "dspark_bf16_gamma7": {1: 4.40, 2: 3.95, 4: 3.76},
    "dflash2_fp8_gamma8": {1: 2.97, 2: 3.06, 4: 3.19},
}

CHECKPOINT_GB = 23.44  # measured, du -sb /cache/unsloth__Qwen3.8-27B-NVFP4 (research/gpu-evidence/rtx/qwen38_quant_report.json)


def measure_hbm_bandwidth_gbps(size_mb: int = 512, iters: int = 20) -> float:
    """A simple device-to-device copy bandwidth probe (measured, this box); used to derive, not assume, the
    dense-model step-cost roofline below."""
    import torch

    n = size_mb * 1024 * 1024 // 4
    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    bytes_moved = size_mb * 1024 * 1024 * 2  # read + write
    return (bytes_moved / (ms / 1000)) / 1e9


def roofline_cost_table(gamma: int, max_per_rank: int, bandwidth_gbps: float,
                        per_token_overhead_s: float = 2e-5) -> StepCostTable:
    """derived: a dense decode step's floor is one weight read (roughly constant with verify-token count, since
    this model has no MoE routing to make more tokens touch more weight) plus a small assumed per-token overhead
    (attention/GDN compute, launch gaps) that DOES grow with total verify tokens. `per_token_overhead_s` is
    `assumed` (not measured: it needs a real SPS profile, see module doc) and picked small enough that it never
    dominates until batch x rows is in the hundreds, which is the point being made: on this architecture, a few
    more verify rows should be nearly free relative to the weight-read floor, unlike Pro's MoE."""
    weight_read_s = CHECKPOINT_GB / bandwidth_gbps  # both in GB / GB per second
    max_tokens = max_per_rank * (gamma + 1)
    tokens = tuple(sorted({1, gamma + 1, max_tokens // 4 or 1, max_tokens // 2 or 1, max_tokens}))
    seconds = tuple(weight_read_s + per_token_overhead_s * t for t in tokens)
    return StepCostTable(tokens, seconds)


def build_qwen38_policy(gamma: int, max_per_rank: int, bandwidth_gbps: float, accept_key: str) -> DSparkLoadPolicy:
    accept_len = ACCEPT_LENGTHS[accept_key][1]  # the c=1 (least diluted) point as the model's per-position rate
    accept = AcceptanceModel.from_accept_length(accept_len, gamma, positions=gamma)
    cost = roofline_cost_table(gamma, max_per_rank, bandwidth_gbps)
    return DSparkLoadPolicy.build(cost, accept, gamma, max_per_rank,
                                  source={"cost": "derived (roofline, see module doc)", "accept_key": accept_key,
                                         "accept_length_c1": accept_len, "bandwidth_gbps": bandwidth_gbps})


def report(max_per_rank: int = 32) -> dict:
    import torch

    bw = measure_hbm_bandwidth_gbps() if torch.cuda.is_available() else 1400.0
    pol = build_qwen38_policy(gamma=7, max_per_rank=max_per_rank, bandwidth_gbps=bw, accept_key="dspark_nvfp4_gamma7")
    return {"bandwidth_gbps": bw, "checkpoint_gb": CHECKPOINT_GB, "policy": pol.to_json(),
           "rank_tok_s_by_load": [
               {"n": n, "rows": pol.rows(n),
                "tok_s": rank_tokens_per_second(n, pol.rows(n), roofline_cost_table(7, max_per_rank, bw),
                                                AcceptanceModel.from_accept_length(4.80, 7, positions=7))}
               for n in (1, 2, 4, 8, 16, 32) if n <= max_per_rank]}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rep = report()
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
