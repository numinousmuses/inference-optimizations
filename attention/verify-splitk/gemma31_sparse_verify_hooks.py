"""rtxsys D2 hook (Gemma 4 31B on H100, kernel program 2026-10-06): sparse target sampling for spec-verify.

Measured (rtxv41, findings rtxv41_gemma31.md section 9, 1xH100, 32 sessions x 6 verify tokens): "Sampling: softmax
+ flashinfer top-p/top-k over 262K" = 2.4 ms per verify step (3.0 ms at 48), floor ~0.2 ms.

Where it goes (source, v0.5.20):
- The served requests sample: the replay sends temperature 0.6, and --sampling-defaults model (the default,
  arg_groups/fields/serving.py:223-229) adds generation_config's top_k 64 / top_p 0.95 (RedHatAI FP8-block
  generation_config.json).
- So eagle_sample (eagle_utils.py:706) takes the dense branch (:843-937) for every verify step:
  - logits / temperature and F.softmax over [B*6, 262144] fp32;
  - top_k_renorm_prob, then top_p_renorm_prob, each a multi-pass kernel over the same 201 MB at 32 sessions;
  - draft_probs = torch.zeros_like(target_probs), another 201 MB memset;
  - tree_speculative_sampling_target_only.
- FrozenKVMTPWorkerV2 inherits EAGLEWorkerV2.verify (eagle_worker_v2.py:1666-1680), which calls run_eagle_verify
  without uno_target_max_top_k.
- v0.5.20 already has the compact path that branch skips: _can_use_sparse_uno_tree_target_sampling (eagle_utils.py:
  688-703) -> sample_uno_tree_target_tokens (uno_utils.py:486-502).
  - One flashinfer radix top_k over the logits; softmax over <= 128 survivors with top-k-first top-p
    (_normalize_sparse_topk_probs, uno_utils.py:21-33);
  - one Gumbel-max draw per node (fast_sample); then verify_tree_greedy_func.

What this hook does: EAGLEWorkerV2.verify passes uno_target_max_top_k = max over the batch's requests of top_k,
computed on the host from batch.reqs with no device sync. It passes it only when every request has 1 <= top_k <= 128.
Otherwise it passes None, and the stock dense branch runs unchanged. The sparse branch then engages under its own
stock conditions: CUDA, no sampling seed, no min_p, no rejection sampling.

Why the output distribution is the same:
- Stock target-only sampling (speculative_sampling.cuh:72-97, 103-134) accepts draft t at a node with probability
  p(t) (coin <= p(t), thresholds 1.0). On reject it zeroes t (draft_probs[t] = p[t]) and samples from
  relu(p - draft): p without t, renormalized. The bonus node samples from p.
- The sparse path draws y ~ p at every node and accepts t iff y == t. The probability is p(t), and on reject y is
  distributed as p without t, renormalized. That is the same law.
- p is the same distribution in both: temperature, then top-k renorm, then top-p renorm over the top-k survivors,
  exactly the dense branch's order. test_gemma31_sparse_verify_hooks.py checks it against a dense reference.
- Accept length is unchanged in expectation. Greedy batches are unaffected (is_all_greedy goes to argmax first,
  eagle_utils.py:784).

Derived gain: the dense row's measured 2.4 / 3.0 ms becomes one radix top-k pass over the logits (201 MB fp32 read,
~0.06 ms at 3.35 TB/s, a few passes) plus ~10 tiny [rows, 64] ops: ~0.25 ms. That is about -2.1 ms per verify step
at 32 sessions and -2.7 ms at 48 (1.9% of the 111.5 ms step).

Env NUMINOUS_SPARSE_VERIFY=1. Hook (anchored replace, v0.5.20, idempotent, the patched file must compile):
  eagle_worker_v2.py: a module helper _nsv_max_top_k before class EAGLEWorkerV2; verify passes its result.

    python3 gemma31_sparse_verify_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sglang_numinous_hooks import sglang_root  # noqa: E402

MARKER = "numinous-hook:gemma31_sparse_verify"
ENV = "NUMINOUS_SPARSE_VERIFY"
PATH = "srt/speculative/eagle_worker_v2.py"
SPARSE_LIMIT = 128   # uno_utils._SPARSE_TOP_K_LIMIT

HELPER_ANCHOR = "class EAGLEWorkerV2("
HELPER = '''# {marker} (helper): rtxsys D2, host-side max top_k for the compact verify sampler (None = stock dense branch)
_NSV_ON = __import__("os").environ.get("NUMINOUS_SPARSE_VERIFY", "0") == "1"


def _nsv_max_top_k(batch):
    if not _NSV_ON:
        return None
    reqs = getattr(batch, "reqs", None)
    if not reqs:
        return None
    m = 0
    for r in reqs:
        k = r.sampling_params.top_k
        if k is None or k < 1 or k > 128:
            return None
        if k > m:
            m = k
    return m


class EAGLEWorkerV2('''

VERIFY_ANCHOR = """            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
"""
VERIFY = """            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
            uno_target_max_top_k=_nsv_max_top_k(batch),  # {marker} (verify)
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
"""
HOOKS = [("helper", HELPER_ANCHOR, HELPER), ("verify", VERIFY_ANCHOR, VERIFY)]


def numinous_preflight(ctx) -> list:
    """Where NUMINOUS_SPARSE_VERIFY=1 never engages."""
    if (ctx.env or {}).get(ENV) != "1":
        return []
    out, a = [], ctx.args or {}
    algo = str(a.get("speculative_algorithm") or "")
    if not algo:
        out.append(f"{ENV}=1 without speculative decoding: inert")
    if a.get("speculative_use_rejection_sampling"):
        out.append(f"{ENV}=1 with --speculative-use-rejection-sampling: the compact branch refuses it, inert")
    dev = getattr(ctx, "device", None) or {}
    if dev and dev.get("is_hip"):
        out.append(f"{ENV}=1 on ROCm: the compact branch is CUDA-only, inert")
    if str(a.get("sampling_defaults") or "model") != "model":
        out.append(f"{ENV}=1 with --sampling-defaults {a.get('sampling_defaults')}: requests then carry top_k -1 "
                   "unless the client sets one, so the dense branch runs")
    return out


def apply(root: str, check: bool = False) -> list:
    out = []
    path = os.path.join(root, PATH)
    src = open(path).read()
    new = src
    for name, anchor, text in HOOKS:
        tag = f"{MARKER} ({name})"
        if tag in new:
            out.append(f"{name}: already applied")
            continue
        n = new.count(anchor)
        if n != 1:
            raise SystemExit(f"gemma31_sparse_verify hooks: {name} anchor found {n} times in {PATH}")
        new = new.replace(anchor, text.replace("{marker}", MARKER), 1)
        out.append(f"{name}: {'anchor ok' if check else 'applied to ' + PATH}")
    if not check and new != src:
        compile(new, path, "exec")
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
