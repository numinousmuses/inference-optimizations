#!/bin/bash
# One-node (1 x RTX PRO 6000) gate for rtxattn's pow2 656-B rows -> FlashInfer DSV3_2 sparse-MLA kernels.
# Run inside the v0.5.20 image from a repo copy at /root/work (any free GPU of the node; ~6 min). Preregistered gates
# (research/rtx-flash/attention.md section 12):
#   P1 correctness: test_glmflash_sm120_gpu.py -k "pow2 or writer" all pass (DSV3_2 vs fp32 < 3e-2, writer bit-exact)
#   P2 prefill: FlashInfer jit pow2 (DSV3_2) 8192-token chunk, one-request tables, <= 0.80 x the arbitrary (GLM_NSA)
#      time at H64 and at H16 (modeled 0.70-0.75x); H16 target <= 5 ms/layer (the section-8 gate)
#   P3 decode/verify: pow2 <= 1.05 x arbitrary (noise band) at every decode/verify shape (no regression)
set -e
cd /root/work
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT=${OUT:-/root/rtxattn_pow2}
mkdir -p $OUT
python3 operations/patches/rtxflash_attn_hooks.py > $OUT/hooks.log
python3 -m pytest -q -p no:cacheprovider operations/kernels/test_glmflash_sm120_gpu.py -k "pow2 or writer" 2>&1 | tee $OUT/p1_tests.log | tail -3
python3 operations/kernels/glmflash_sm120_bench.py --prefill-ctx 31000 --heads 16,64 --out $OUT/prefill.json > $OUT/prefill.log 2>&1
python3 operations/kernels/glmflash_sm120_bench.py --heads 16,64 --out $OUT/bench.json > $OUT/bench.log 2>&1
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
def rows(f):
    return {(r["kernel"], r["phase"], r["T"], r["H"]): r["us"] for r in json.load(open(f"{out}/{f}"))["rows"]}
pre, dec = rows("prefill.json"), rows("bench.json")
A, P = "flashinfer_jit_fp8_656", "flashinfer_jit_fp8_656_pow2_dsv3_2"
res = {}
for h in (16, 64):
    a, p = pre[(A, "prefill", 8192, h)], pre[(P, "prefill", 8192, h)]
    res[f"P2_prefill8192_H{h}"] = {"arbitrary_us": a, "pow2_us": p, "ratio": round(p / a, 3), "pass": p <= 0.80 * a}
res["P2_H16_abs_le_5ms"] = {"pow2_us": pre[(P, "prefill", 8192, 16)], "pass": pre[(P, "prefill", 8192, 16)] <= 5000}
worst = max((dec[(P, ph, t, h)] / dec[(A, ph, t, h)], ph, t, h) for (k, ph, t, h) in dec
            if k == A and ph in ("decode", "verify") and dec.get((P, ph, t, h)))
res["P3_decode_verify_worst_ratio"] = {"ratio": round(worst[0], 3), "at": worst[1:], "pass": worst[0] <= 1.05}
json.dump(res, open(f"{out}/gates.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print("POW2_GATE_PASS" if all(v["pass"] for v in res.values()) else "POW2_GATE_FAIL")
PY
