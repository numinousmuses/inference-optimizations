"""Anchored patcher: two-tier quantized KDA state for Kimi K3 (NUMINOUS_K3STATE=fp8|fp4|bf16; inert when unset or
"off"). Trees: SGLang v0.5.20 (94602c9, the CUDA B200/B300 image) and main 172b1b4825 (the ROCm MI355X image
lmsysorg/sglang-rocm v0.5.20-rocm720-mi35x-20260923). A hook whose anchor differs between trees carries the other
tree's (anchor, text) in `alts`; exactly one variant must match exactly once (k3amd, 2026-09-30).

Installs operations/kernels/k3state_quant.py -> srt/layers/attention/numinous_k3state_quant.py and
operations/kernels/k3state_engine.py -> srt/layers/attention/numinous_k3state.py, then hooks:

  k3s_pool_temporal   mem_cache/memory_pool.py MambaPool.__init__: the per-slot KDA state is allocated quantized
                      (uint8 rows with in-row scales) instead of fp32
  k3s_pool_ring       the KDA ReplaySSM ring is sized by request rows (spec_state_size + 2), not by mamba slots
  k3s_pool_attach     allocates the fp32 working state W and the conv working copy (request rows) and validates the
                      configuration (KDA, --enable-linear-replayssm-spec, no page-major envelope)
  k3s_cfg_slot        mem_cache/kv_cache_configurator.py _handle_max_mamba_cache: prices a mamba slot at its quantized
                      bytes, so the ratio-based solve hands out more slots
  k3s_cfg_ring        the ring + W + conv copy are priced per request (the GDN fixed-bytes branch), not per slot
  k3s_cfg_ring_fixed
  k3s_extend          layers/attention/linear/kda_backend.py forward_extend: prefill and target verify go through
                      numinous_k3state (dequant -> stock kernel on W -> quantized mirror; verify via a proxy cache)
  k3s_decode          forward_decode: non-speculative decode through the same proxy
  k3s_commit          kernels/ops/attention/fla/kda_replayssm_spec_decode.py commit_kda_replayssm_after_verify: the
                      two-tier fold (numinous_k3state_quant.commit_fold_all_layers) + stock conv rollback

    python3 operations/patches/k3state_hooks.py [--root DIR] [--check]
Test: operations/patches/test_k3state_hooks.py. Missing or repeated anchors fail loudly; files are compile-checked.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = {
    os.path.join(HERE, "..", "kernels", "k3state_quant.py"): "srt/layers/attention/numinous_k3state_quant.py",
    os.path.join(HERE, "..", "kernels", "k3state_engine.py"): "srt/layers/attention/numinous_k3state.py",
}
POOL = "srt/mem_cache/memory_pool.py"
CFG = "srt/mem_cache/kv_cache_configurator.py"
KDA = "srt/layers/attention/linear/kda_backend.py"
REPLAY = "kernels/ops/attention/fla/kda_replayssm_spec_decode.py"
IMPORT = "from sglang.srt.layers.attention import numinous_k3state as _nk3s"


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str
    replace: bool = False  # False: insert text after the anchor; True: the anchor becomes text
    alts: tuple = ()       # ((anchor, text), ...): the same hook on another SGLang tree

    def variants(self) -> tuple:
        return ((self.anchor, self.text),) + tuple(self.alts)

    @property
    def marker(self) -> str:
        return f"numinous-hook:{self.name}"


_TEMPORAL = ("                temporal_state = torch.zeros(\n"
             "                    size=(num_mamba_layers, size + 1) + temporal_state_shape,\n"
             "                    dtype=ssm_dtype,\n"
             "                    device=device,\n"
             "                )\n")
_RING = ("                num_slots = (\n"
         "                    spec_state_size + 1\n"
         "                    if enable_linear_replayssm_spec and not cache_params.is_kda\n"
         "                    else size + 1\n"
         "                )\n")
_VERIFY = ("        # MTP / speculative-decode verify is a multi-token-per-seq path with\n"
           "        # per-step state checkpointing + central rollback; handled separately.\n"
           "        if forward_batch.forward_mode.is_target_verify():\n")
_DECODE = ("        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)\n"
           "        conv_states = layer_cache.conv[0]\n"
           "        ssm_states = layer_cache.temporal\n")
_STAGE = ("        stage_per_req = int(\n"
          "            config.mamba2_cache_params.mamba_cache_per_req * pp_layer_scale\n"
          "        )\n")

HOOKS = [
    Hook("k3s_pool_temporal", POOL, _TEMPORAL,
         "                # numinous-hook:k3s_pool_temporal\n"
         f"                {IMPORT}\n"
         "\n"
         "                if _nk3s.mode():\n"
         "                    temporal_state = _nk3s.pool_temporal(\n"
         "                        num_mamba_layers, size, temporal_state_shape, device\n"
         "                    )\n"
         "                else:\n"
         "                    temporal_state = torch.zeros(\n"
         "                        size=(num_mamba_layers, size + 1) + temporal_state_shape,\n"
         "                        dtype=ssm_dtype,\n"
         "                        device=device,\n"
         "                    )\n",
         replace=True),
    Hook("k3s_pool_ring", POOL, _RING,
         "                # numinous-hook:k3s_pool_ring: ring rows follow the request rows\n"
         f"                {IMPORT}\n"
         "\n"
         "                if _nk3s.mode():\n"
         "                    num_slots = _nk3s.ring_rows(spec_state_size)\n"),
    Hook("k3s_pool_attach", POOL, "            self.num_mamba_layers = num_mamba_layers\n",
         "            # numinous-hook:k3s_pool_attach\n"
         f"            {IMPORT}\n"
         "\n"
         "            if _nk3s.mode():\n"
         "                _nk3s.attach(\n"
         "                    self,\n"
         "                    num_layers=num_mamba_layers,\n"
         "                    mamba_layer_ids=mamba_layer_ids,\n"
         "                    spec_state_size=spec_state_size,\n"
         "                    temporal_shape=cache_params.shape.temporal,\n"
         "                    device=device,\n"
         "                    replayssm_spec=enable_linear_replayssm_spec,\n"
         "                    envelope=envelope_layout,\n"
         "                    is_kda=cache_params.is_kda,\n"
         "                )\n"),
    Hook("k3s_cfg_slot", CFG, _STAGE,
         "        # numinous-hook:k3s_cfg_slot: a mamba slot holds the quantized state\n"
         f"        {IMPORT}\n"
         "\n"
         "        if _nk3s.mode():\n"
         "            if get_memory().enable_unified_memory:\n"
         "                raise RuntimeError(\"NUMINOUS_K3STATE does not support the unified memory pool\")\n"
         "            stage_per_req = int(\n"
         "                _nk3s.slot_bytes(config.mamba2_cache_params) * pp_layer_scale\n"
         "            )\n"),
    Hook("k3s_cfg_ring", CFG, "        replayssm_ring_per_req = int(replayssm_ring_per_req * pp_layer_scale)\n",
         "        # numinous-hook:k3s_cfg_ring: fp32 working state + conv copy per request row\n"
         "        if _nk3s.mode() and replayssm_active:\n"
         "            replayssm_ring_per_req += int(\n"
         "                _nk3s.request_bytes(config.mamba2_cache_params) * pp_layer_scale\n"
         "            )\n"),
    Hook("k3s_cfg_ring_fixed", CFG,
         "        if replayssm_active and kimi_linear_config(self.model_config) is None:\n",
         "        if replayssm_active and (\n"
         "            kimi_linear_config(self.model_config) is None or _nk3s.mode()\n"
         "        ):  # numinous-hook:k3s_cfg_ring_fixed\n",
         replace=True,
         alts=(("        if replayssm_active and self.hybrid_kda_config is None:\n",       # main 172b1b4825
                "        if replayssm_active and (\n"
                "            self.hybrid_kda_config is None or _nk3s.mode()\n"
                "        ):  # numinous-hook:k3s_cfg_ring_fixed\n"),)),
    Hook("k3s_extend", KDA, _VERIFY,
         "        # numinous-hook:k3s_extend\n"
         f"        {IMPORT}\n"
         "\n"
         "        if _nk3s.mode():\n"
         "            return _nk3s.forward_extend(\n"
         "                self, layer, forward_batch, mixed_qkv, a, b, **kwargs\n"
         "            )\n" + _VERIFY,
         replace=True),
    Hook("k3s_decode", KDA, _DECODE,
         "        # numinous-hook:k3s_decode\n"
         f"        {IMPORT}\n"
         "\n"
         "        if _nk3s.mode() and not getattr(self, \"_k3s_inside\", False):\n"
         "            return _nk3s.forward_decode(\n"
         "                self, type(self).forward_decode, layer, forward_batch, mixed_qkv, a, b, **kwargs\n"
         "            )\n" + _DECODE,
         replace=True),
    Hook("k3s_commit", REPLAY, "\n    L = spec_state.replayssm_rawv.shape[-2]\n",
         "\n    # numinous-hook:k3s_commit\n"
         f"    {IMPORT}\n"
         "\n"
         "    if _nk3s.mode():\n"
         "        return _nk3s.commit(\n"
         "            spec_state=spec_state,\n"
         "            state_batch_indices=state_batch_indices,\n"
         "            accept_lens=accept_lens,\n"
         "            last_correct_step_indices=last_correct_step_indices,\n"
         "            mamba_track_indices=mamba_track_indices,\n"
         "            mamba_steps_to_track=mamba_steps_to_track,\n"
         "            null_block_id=null_block_id,\n"
         "        )\n"
         "    L = spec_state.replayssm_rawv.shape[-2]\n",
         replace=True),
]


def sglang_root() -> str:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def patch_source(src: str, hooks) -> tuple[str, list[str], list[str]]:
    """Apply hooks (all for one file) to src. Returns (new_src, status lines, problems)."""
    out, problems = [], []
    for h in hooks:
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        counts = [src.count(anchor) for anchor, _ in h.variants()]
        if sorted(counts)[-1:] != [1] or sum(counts) != 1:
            problems.append(f"{h.name}: anchor variants found {counts} times in {h.path}")
            continue
        anchor, text = h.variants()[counts.index(1)]
        src = src.replace(anchor, text if h.replace else anchor + text, 1)
        out.append(f"{h.name}: applied to {h.path}")
    return src, out, problems


def apply(root: str, check: bool = False) -> list[str]:
    out, problems = [], []
    by_path: dict[str, list[Hook]] = {}
    for h in HOOKS:
        by_path.setdefault(h.path, []).append(h)
    new_files = {}
    for rel, hooks in by_path.items():
        path = os.path.join(root, rel)
        src = open(path).read()
        new, lines, probs = patch_source(src, hooks)
        problems += probs
        out += [ln.replace("applied to", "anchor ok in") if check else ln for ln in lines]
        if new != src:
            compile(new, path, "exec")  # never leave a file that does not parse
            new_files[path] = new
    if problems:
        raise SystemExit("numinous k3state hooks: " + "; ".join(problems))
    if check:
        return out
    for src, rel in MODULES.items():
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.normpath(src), dst)
        out.append(f"module: {rel}")
    for path, new in new_files.items():
        tmp = f"{path}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
