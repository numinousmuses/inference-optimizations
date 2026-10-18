"""Anchored hook: run FlashInfer's SM90 Humming W4A8 weight preparation over experts in chunks (SGLang v0.5.20).

Why (measured 2026-10-04, Modal H200:8, DeepSeek V4 Pro 0813, DP8, `--flashinfer-mxfp4-moe-precision fp8`):
the boot died in `flashinfer/fused_moe/prepare.py:_process_humming_mxfp4_w4a8_payload` with
"CUDA out of memory. Tried to allocate 15.75 GiB ... 12.04 GiB is free". That function expands the packed FP4
payload into int64 per-element tensors: `delta_scale_offsets.repeat_interleave(32).to(torch.long)` and
`fp4_codes.to(torch.long)` are each 16 bytes per packed byte. For one Pro layer's w13 at TP8 (384 experts x 768 x
3584 B = 1.06 GB packed) that is ~34 GB of temporaries, more than any H200 has free after the weights load.
(evidence: research/gpu-evidence/hopper/pro-h200_20261004-215151_dp8tbo_boot_oom.txt)

Every step of the preparation is per expert (FlashInfer 0.6.18 `fused_moe/prepare.py`):
`_preprocess_humming_e8m0_weight_scale` takes max/min over each expert's own scales, the payload rewrite is
elementwise, the weight interleave maps [E, n, k/2] to [E, n, k/2], the scale interleave keeps E as the leading dim,
and the residual is one FP32 per expert. So slicing along dim 0, preparing each slice and concatenating is exact,
and the temporaries shrink by E / chunk (16 of 384 experts: ~1.4 GB).

    python3 operations/patches/humming_chunk_hooks.py            # patch the installed sglang package
    python3 operations/patches/humming_chunk_hooks.py --check    # report, change nothing
    python3 operations/patches/humming_chunk_hooks.py --root /tmp/sgl/python/sglang --check

NUMINOUS_HUMMING_CHUNK (default 16) experts per call; 0 disables the chunking (the original single call).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

PATH = "srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py"
MARK = "# numinous-humming-chunk"


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str
    marker: str


WRAPPER = '''

def _numinous_humming_chunked(fn):  # numinous-humming-chunk
    """FlashInfer's SM90 Humming W4A8 weight preparation, run over experts in chunks (operations/patches/
    humming_chunk_hooks.py): its payload rewrite builds int64 per-element temporaries (~34 GB for one DeepSeek V4 Pro
    layer at TP8), which OOM'd the H200 boot. Each step is per expert, so slice-and-concatenate on dim 0 is exact."""
    chunk = int(os.environ.get("NUMINOUS_HUMMING_CHUNK", "16"))

    def wrapped(weight, raw_scale, *args, **kwargs):
        experts = weight.shape[0]
        if chunk <= 0 or experts <= chunk:
            return fn(weight, raw_scale, *args, **kwargs)
        ws, ss, rs = [], [], []
        for s in range(0, experts, chunk):
            w_i, s_i, r_i = fn(weight[s:s + chunk].contiguous(), raw_scale[s:s + chunk].contiguous(), *args, **kwargs)
            ws.append(w_i)
            ss.append(s_i)
            rs.append(r_i)
        out = (torch.cat(ws, 0), torch.cat(ss, 0), torch.cat(rs, 0))
        del ws, ss, rs
        torch.cuda.empty_cache()
        return out

    return wrapped
'''

HOOKS = [
    Hook("humming_chunk_wrapper", PATH, "logger = logging.getLogger(__name__)\n", WRAPPER,
         "def _numinous_humming_chunked(fn):  " + MARK),
    Hook("humming_chunk_bind", PATH,
         "                from flashinfer.fused_moe import (\n"
         "                    preprocess_moe_weights_for_sm90_mixed_gemm_humming,\n"
         "                )\n",
         "                preprocess_moe_weights_for_sm90_mixed_gemm_humming = _numinous_humming_chunked(  "
         + MARK + "\n"
         "                    preprocess_moe_weights_for_sm90_mixed_gemm_humming\n"
         "                )\n",
         "preprocess_moe_weights_for_sm90_mixed_gemm_humming = _numinous_humming_chunked(  " + MARK),
]


def sglang_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook; SystemExit if an anchor is missing or repeated."""
    out, missing = [], []
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = open(path).read()
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        if check:
            out.append(f"{h.name}: anchor ok")
            continue
        new = src.replace(h.anchor, h.anchor + h.text, 1)
        compile(new, path, "exec")  # never leave a file that does not parse
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("numinous humming chunk hooks: " + "; ".join(missing))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
