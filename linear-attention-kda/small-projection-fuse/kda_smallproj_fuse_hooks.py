"""Anchored hook: fuse GLM-5.3-Flash's five small bf16 KDA projections into one GEMM + one batched GEMM when the
checkpoint is quantized (rtxsys, kernel program 2026-10-05). Tree: SGLang v0.5.20 (IMG_V0520),
srt/models/glm5_next.py.

    python3 kda_smallproj_fuse_hooks.py [--root <sglang pkg dir>] [--check]

Why (measured):
- With a quant_config, `do_fuse_qkvbfg` is False (glm5_next.py:340). Each KDA layer then runs b_proj, f_a_proj,
  g_a_proj, f_b_proj and g_b_proj as five separate bf16 GEMMs (forward_qkvbfg :491-503); the FP8 checkpoints leave
  them unquantized.
- 1xH200, graph replay (research/gpu-evidence/rtxsys/flash_kda_smallproj_h200.json), separate -> fused per KDA layer:

  | M (tokens) | separate us | fused us | per step / cycle (34 KDA layers) |
  |---|---|---|---|
  | 4,096 (prefill) | 96.69 | 70.06 | -0.9 ms |
  | 120 (decode verify, b20) | 26.65 | 11.97 | -0.5 ms |
  | 24-144 (decode verify, b4-b24) | 24.8-26.7 | 10.1-13.0 | -0.44..-0.51 ms |

- Outputs identical: max abs diff 0.0, except one bf16 ulp (0.0156) at M=144.

The fused path is the unquantized model's own layout (forward_qkvbfg_fused :505-521):
- one GEMM H -> nb + 2d (b | f_a | g_a);
- one bmm [2, M, d] x [2, d, P] (f_b, g_b).

When the weights are built:
- Lazily, at the layer's first call. Graph warmup runs eager before capture, so the captured graphs use the fused
  path. Building at post_load would see the overlapped loader's sentinel weights (loader.py:825-845).
- If that first call is already under capture, the layer stays stock forever. The build re-points the separate params
  at the fused storage (memory unchanged), which would free addresses a captured graph still reads.
- If any of the five is quantized, has a bias, or the shapes disagree, the build returns False and the stock path
  stays.

Env NUMINOUS_KDA_SMALLPROJ_FUSE=1 (both roles). Unset: stock, byte for byte.
"""
from __future__ import annotations

import argparse
import os
import py_compile
import sys

PATH = "srt/models/glm5_next.py"
NAME = "kda_smallproj_fuse"
TAG = f"# numinous-hook:{NAME}"

HELPERS = f'''{TAG} (rtxsys): one GEMM + one bmm for the five small bf16 KDA projections (see the hook file)
_NUMINOUS_KDA_SMALLPROJ = __import__("os").environ.get("NUMINOUS_KDA_SMALLPROJ_FUSE", "0") == "1"


def _numinous_kda_smallproj_build(attn):
    mods = [attn.b_proj, attn.f_a_proj, attn.g_a_proj, attn.f_b_proj, attn.g_b_proj]
    for m in mods:
        w = getattr(m, "weight", None)
        if (w is None or w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16)
                or getattr(m, "bias", None) is not None
                or "Unquantized" not in type(getattr(m, "quant_method", None)).__name__
                or any(hasattr(m, a) for a in ("weight_scale", "weight_scale_inv", "input_scale"))):
            return False
    wb, wfa, wga = attn.b_proj.weight, attn.f_a_proj.weight, attn.g_a_proj.weight
    wfb, wgb = attn.f_b_proj.weight, attn.g_b_proj.weight
    nb, d = wb.shape[0], wfa.shape[0]
    if (wga.shape != wfa.shape or wb.shape[1] != wfa.shape[1] or wfb.shape != wgb.shape or wfb.shape[1] != d
            or len({{w.dtype for w in (wb, wfa, wga, wfb, wgb)}}) != 1):
        return False
    w_a = torch.cat([wb.data, wfa.data, wga.data], 0).contiguous()
    w_b = torch.stack([wfb.data.t(), wgb.data.t()], 0).contiguous()
    attn.b_proj.weight.data = w_a[:nb]
    attn.f_a_proj.weight.data = w_a[nb:nb + d]
    attn.g_a_proj.weight.data = w_a[nb + d:]
    attn.f_b_proj.weight.data = w_b[0].t()
    attn.g_b_proj.weight.data = w_b[1].t()
    return (w_a, w_b, nb, d)


def _numinous_kda_smallproj(attn, x):
    """(beta, forget_gate, g_proj_states) from one GEMM + one bmm, or None for the stock path."""
    if not _NUMINOUS_KDA_SMALLPROJ:
        return None
    st = getattr(attn, "_numinous_smallproj", None)
    if st is None:
        # Decide once, at the layer's first call. Under capture the decision is "stock forever": a captured graph
        # holds the separate weights' addresses, and the build re-points (frees) them.
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            st = False
        else:
            st = _numinous_kda_smallproj_build(attn)
        attn._numinous_smallproj = st
    if st is False or x.dim() != 2:
        return None
    w_a, w_b, nb, d = st
    y = torch.nn.functional.linear(x, w_a)
    beta, fg = y.split([nb, 2 * d], dim=-1)
    fgb = torch.bmm(fg.reshape(-1, 2, d).transpose(0, 1), w_b)
    return beta, fgb[0], fgb[1]


'''

HOOKS = [
    ("helpers", "class Glm5NextLinearAttention(nn.Module):\n",
     HELPERS + "class Glm5NextLinearAttention(nn.Module):\n"),
    ("forward", ("    def forward_qkvbfg(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):\n"
                 "        qkv, _ = self.qkv_proj(hidden_states)\n\n"
                 "        beta = self.b_proj(hidden_states)[0]\n"),
     ("    def forward_qkvbfg(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):\n"
      "        qkv, _ = self.qkv_proj(hidden_states)\n\n"
      f"        _nf = _numinous_kda_smallproj(self, hidden_states)  {TAG}\n"
      "        if _nf is not None:\n"
      "            return (qkv, *_nf)\n\n"
      "        beta = self.b_proj(hidden_states)[0]\n")),
]


def apply(root: str, check: bool = False) -> list[str]:
    f = os.path.join(root, PATH)
    src = open(f).read()
    if TAG in src:
        return [f"{NAME}: already applied"]
    for name, anchor, _ in HOOKS:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"{NAME}.{name}: anchor found {n} times in {PATH}")
    if check:
        return [f"{NAME}: {len(HOOKS)} anchors ok in {PATH}"]
    for _, anchor, text in HOOKS:
        src = src.replace(anchor, text)
    open(f, "w").write(src)
    py_compile.compile(f, doraise=True)
    return [f"{NAME}: applied {len(HOOKS)} hooks to {PATH}"]


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
    for ln in apply(root, a.check):
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
