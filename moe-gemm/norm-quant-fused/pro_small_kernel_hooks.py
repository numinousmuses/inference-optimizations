"""Anchored hooks that switch on the DeepSeek V4 Pro small-kernel fusions in an installed SGLang (v0.5.20), each
behind its own env flag, each logging one activation line (logger `numinous_pro`, WARNING) on first use:

  NUMINOUS_MHC_FUSED=1    mHC post + next pre in one Triton kernel     (pro_mhc_fused; NUMINOUS_MHC_TWO_PASS=1: 2 launches)
  NUMINOUS_NORM_QUANT=1   q_norm + FP8 UE8M0 quant once, consumed by wq_b and the indexer wq_b (pro_norm_quant)
  NUMINOUS_INDEXER_Q=1    indexer wq_b FP8 GEMM + RoPE + Hadamard + quant + weights (pro_indexer_q; needs NORM_QUANT)
  NUMINOUS_FP8_LM_HEAD=1  LM head quantized to FP8 128x128 at load, DeepGEMM logits (pro_fp8_lm_head)

Same contract as sglang_numinous_hooks.py: every hook names a file, an exact anchor and inserted text; a missing or
ambiguous anchor fails loudly; a marker makes it idempotent. `apply` also installs the four kernel modules as the
package `sglang.srt.layers.numinous_pro` (copied from operations/kernels), so the patch is self-contained.

    python3 pro_small_kernel_hooks.py                 # patch the installed sglang (all hooks)
    python3 pro_small_kernel_hooks.py --root /tmp/sgl --check
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(os.path.dirname(HERE), "kernels")
MODULES = ["pro_mhc_fused.py", "pro_norm_quant.py", "pro_indexer_q.py", "pro_fp8_lm_head.py"]
PACKAGE = "srt/layers/numinous_pro"


@dataclass(frozen=True)
class Hook:
    name: str
    path: str  # relative to the sglang package directory
    anchor: str  # exact text; the insertion goes right after it
    text: str

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


HOOKS = [
    Hook("mhc_fused", "srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py",
         "    fn_transpose: bool,\n"
         ") -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]:\n"
         "    # Try the aiter/Triton fused post+pre kernels first; if neither fires,\n"
         "    # fall back to the TileLang fused kernel, else return None so the caller\n"
         "    # runs the unfused hc_post + hc_pre sequence.\n",
         "    # numinous-hook:mhc_fused\n"
         "    from sglang.srt.layers.numinous_pro import pro_mhc_fused as _npm\n"
         "\n"
         "    if _npm.engine_enabled() and not _is_hip:\n"
         "        _r = _npm.engine_boundary(layer_input, residual, post, comb, hc_fn, hc_scale, hc_base, hc_mult,\n"
         "                                  rms_eps, hc_eps, hc_post_mult, sinkhorn_iters, norm_weight, norm_eps)\n"
         "        if _r is not None:\n"
         "            return _r\n"),
    Hook("norm_quant", "srt/models/deepseek_v4.py",
         "        if qkv_a is not None:\n"
         "            q = qkv_a[..., : self.q_lora_rank]\n"
         "        else:\n"
         "            q, _ = self.wq_a(x)\n",
         "        # numinous-hook:norm_quant\n"
         "        from sglang.srt.layers.numinous_pro import pro_norm_quant as _nq\n"
         "\n"
         "        if _nq.engine_enabled():\n"
         "            return _nq.engine_q_norm(self.q_norm, q)\n"),
    Hook("norm_quant_linear", "srt/layers/quantization/fp8.py",
         "                **extra_kwargs,\n"
         "            )\n"
         "\n"
         "        if self.block_quant:\n",
         "            # numinous-hook:norm_quant_linear\n"
         "            _pq = getattr(x, \"_numinous_fp8\", None)\n"
         "            if _pq is not None:\n"
         "                from sglang.srt.layers.numinous_pro import pro_norm_quant as _nq\n"
         "\n"
         "                _out = _nq.engine_linear(layer, x, _pq, bias)\n"
         "                if _out is not None:\n"
         "                    return _out\n"),
    Hook("indexer_q", "srt/layers/attention/dsv4/indexer.py",
         "        q_lora: torch.Tensor,\n"
         "        positions: torch.Tensor,\n"
         "        weight: torch.Tensor,\n"
         "    ) -> Tuple[IndexerQuery, torch.Tensor]:\n",
         "        # numinous-hook:indexer_q\n"
         "        from sglang.srt.layers.numinous_pro import pro_indexer_q as _iq\n"
         "\n"
         "        if _iq.engine_enabled() and not self.use_fp4_indexer:\n"
         "            _r = _iq.engine_compute_q(self, q_lora, positions, weight)\n"
         "            if _r is not None:\n"
         "                return _r\n"),
    Hook("fp8_lm_head", "srt/layers/logits_processor.py",
         "        embedding_bias: Optional[torch.Tensor] = None,\n"
         "    ) -> torch.Tensor:\n"
         "        quant_method = getattr(lm_head, \"quant_method\", None)\n",
         "        # numinous-hook:fp8_lm_head\n"
         "        if getattr(lm_head, \"_numinous_fp8\", None) is not None and embedding_bias is None:\n"
         "            from sglang.srt.layers.numinous_pro import pro_fp8_lm_head as _lm\n"
         "\n"
         "            return _lm.engine_logits(hidden_states, lm_head, self.use_fp32_lm_head)\n"),
    Hook("pro_post_load", "srt/models/deepseek_v4.py",
         "        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)\n",
         "        # numinous-hook:pro_post_load\n"
         "        from sglang.srt.layers.numinous_pro import pro_fp8_lm_head as _lm, pro_indexer_q as _iq\n"
         "\n"
         "        if _lm.engine_enabled():\n"
         "            _lm.engine_post_load(self)\n"
         "        if _iq.engine_enabled():\n"
         "            _n = sum(_iq.engine_prepare(m) for m in self.modules() if type(m).__name__ == \"C4Indexer\")\n"
         "            logger.warning(\"numinous indexer_q prepared %d C4 indexers\", _n)\n"),
]


def sglang_root() -> str:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def install_modules(root: str, kernels_dir: str = KERNELS) -> str:
    dst = os.path.join(root, PACKAGE)
    os.makedirs(dst, exist_ok=True)
    for m in MODULES:
        shutil.copy(os.path.join(kernels_dir, m), os.path.join(dst, m))
    with open(os.path.join(dst, "__init__.py"), "w") as f:
        f.write('"""Numinous DeepSeek V4 Pro small-kernel fusions (installed by pro_small_kernel_hooks.py)."""\n')
    return dst


def apply(root: str, names: list[str], check: bool = False, modules: bool = True) -> list[str]:
    """One status line per selected hook; raises SystemExit if an anchor is missing or ambiguous."""
    out, missing = [], []
    pending: dict[str, str] = {}
    for h in HOOKS:
        if names and h.name not in names:
            continue
        path = os.path.join(root, h.path)
        src = pending.get(path) or open(path).read()
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        pending[path] = src.replace(h.anchor, h.anchor + h.text, 1)
        out.append(f"{h.name}: anchor ok" if check else f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("pro small-kernel hooks: " + "; ".join(missing))
    if not check:
        for path, new in pending.items():
            compile(new, path, "exec")  # never leave a file that does not parse
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
        if modules:
            out.append(f"modules installed in {install_modules(root)}")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="hook names (default: all)")
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.names, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
