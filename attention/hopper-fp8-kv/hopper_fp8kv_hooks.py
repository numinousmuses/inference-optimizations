"""FP8 KV cache for GLM-5.3-Flash DSA on Hopper (SM90): anchored hooks into SGLang v0.5.20 (rtxsys, 2026-10-04;
agents/briefs/hopper-fp8kv.md, findings agents/findings/hopper_fp8kv.md).

Why Flash runs a bf16 KV on SM90 today (source_code):
  * GLM-5.3-Flash sets index_kpool 4 (config.json). The kpool indexer appends up to 3 not-yet-pooled tail tokens to every
    top-k row, so attention sees 2048 + 3 = 2051-wide index rows (layers/attention/dsa/kpool_fp8_index.py
    append_kpool_tail_to_topk).
  * dsa_backend_kpool._check_kpool_tail_backend lets only fa3 / tilelang / trtllm see those rows. On SM90 that leaves fa3
    and tilelang, which read a bf16 pool only, so the cell must run --kv-cache-dtype bfloat16.
  * The FP8 kernel exists: FlashMLA (sgl fork 3e18517, kernels/aot/cmake/flashmla.cmake) builds the SM90 sparse decode
    instantiation v32_no_rope_persistent_h64 / _h128 for sm_90a. Its format V32_NO_ROPE is 528 bytes per token: 512 e4m3
    + four fp32 scales, one per 128 dims (the descale travels in the row). That is exactly the row SGLang already
    allocates and writes for kv_lora 512 / rope 0 under FP8 off trtllm (kv_cache_configurator.calculate_mla_kv_cache_dim,
    memory_pool._write_mla_kv_buffer -> quant_k_cache.quantize_k_cache_separate).
  * The SGLang flashmla_kv path refuses 2051-wide rows: _forward_flashmla_kv asserts width == index_topk, and the kernel
    requires topk % 64 == 0 (FlashMLA csrc/kernels/sm90/decode/sparse/splitkv_mla.cuh KernelTemplate::run). V3.2-like
    formats take no topk_length / extra_kv there. Indices of -1 are masked by the kernel.

What the hooks change (everything inert unless NUMINOUS_HOPPER_FP8KV=1):
  hopper_fp8kv_kpool   dsa_backend_kpool._check_kpool_tail_backend: flashmla_kv may see kpool tails
  hopper_fp8kv_meta    dsa_backend._compute_flashmla_metadata: FlashMLA tile-scheduler topk = the padded row width
                       (index_topk + kpool - 1 rounded up to 64: 2112 for Flash) instead of index_topk
  hopper_fp8kv_pad     dsa_backend._forward_flashmla_kv: pad index rows with -1 to that width (masked by the kernel);
                       the stock equality assert stays for every other case
  hopper_fp8kv_slice   dsa_backend._forward_flashmla_kv: outside CUDA-graph capture, call the kernel on row slices of
                       NUMINOUS_HOPPER_FP8KV_SLICE (default 2048) rows. Prefill runs this decode kernel once per token
                       and its split-KV fp32 accumulators scale with rows x heads x 512: a 16K-token chunk asked 2.01 GiB
                       and OOMed on H200 (measured, flash-h200-fp8kv-flashmla 2026-10-04 23:21Z)

Serve Flash with: --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv and
env NUMINOUS_HOPPER_FP8KV=1. GLM-5.3 (no kpool) needs none of this: it already runs FP8 KV on SM90 with an explicit
--kv-cache-dtype fp8_e4m3 (the stock auto is bf16 on SM90, arg_groups/overrides.py _dsa_kv_cache_dtype_default).

    python3 operations/patches/hopper_fp8kv_hooks.py                 # patch the installed sglang
    python3 operations/patches/hopper_fp8kv_hooks.py --root /tmp/sgl0520/python/sglang --check
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

BACKEND = "srt/layers/attention/dsa_backend.py"
KPOOL = "srt/layers/attention/dsa/dsa_backend_kpool.py"
XFORM = "kernels/ops/attention/dsa/transform_index.py"
FILES = (KPOOL, BACKEND)

# hopper_fp8kv_xform (measured 2026-10-05, flash-h200-F in-box PD, prefill engine log): the PD PREFILL worker's EAGLE
# draft extend runs with fused TopK off (dsa/utils.should_use_dsa_fused_topk: "PD Prefill worker: Draft extend: fused
# TopK disabled"), so its kpool rows (index_topk + kpool - 1 = 2051 for Flash) reach the unfused PAGED transform,
# whose only width check is `assert topk_indices.shape[1] == 2048`; the kernel itself takes TOPK = shape[1] and masks
# offsets >= TOPK. The assert killed the prefill scheduler on the first long prompt and every request returned 0
# tokens. Monolithic and PD-decode servers keep fused TopK and never reach it.
XFORM_ANCHOR = ("    assert page_size == 1\n"
                "    assert topk_indices.shape[1] == 2048\n"
                "    real_num_tokens = sum(extend_lens_cpu)\n")
XFORM_TEXT = ("    assert page_size == 1\n"
              "    # numinous-hook:hopper_fp8kv_xform (rtxsys): kpool rows (2048 + kpool - 1) reach this unfused transform\n"
              "    # on the PD prefill draft extend; the kernel below is width-generic (TOPK = shape[1], masked)\n"
              "    assert topk_indices.shape[1] >= 2048\n"
              "    real_num_tokens = sum(extend_lens_cpu)\n")
ENV = "NUMINOUS_HOPPER_FP8KV"


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str
    replace: bool = False
    satisfied_by: str = ""   # another series' marker that already makes the same change (then this hook is skipped)

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


KPOOL_ANCHOR = '            or dsa_impl in ("fa3", "tilelang", "trtllm")\n'
KPOOL_TEXT = ('            or dsa_impl in ("fa3", "tilelang", "trtllm")\n'
              '            # numinous-hook:hopper_fp8kv_kpool\n'
              f'            or (dsa_impl == "flashmla_kv" and __import__("os").environ.get("{ENV}") == "1")\n')

META_ANCHOR = ('            is_fp8_kvcache=True,\n'
               '            topk=self.dsa_index_topk,\n'
               '        )\n'
               '\n'
               '        return DSAFlashMLAMetadata(\n')
META_TEXT = ('            is_fp8_kvcache=True,\n'
             '            # numinous-hook:hopper_fp8kv_meta\n'
             '            topk=(\n'
             '                (self.dsa_index_topk + self.dsa_index_kpool - 1 + 63) // 64 * 64\n'
             f'                if self.dsa_index_kpool > 1 and __import__("os").environ.get("{ENV}") == "1"\n'
             '                else self.dsa_index_topk\n'
             '            ),\n'
             '        )\n'
             '\n'
             '        return DSAFlashMLAMetadata(\n')

PAD_ANCHOR = ('        indices = page_table_1.unsqueeze(1)\n'
              '        assert (\n'
              '            indices.shape[-1] == self.dsa_index_topk\n'
              '        )  # requirement of FlashMLA decode kernel\n')
PAD_TEXT = ('        indices = page_table_1.unsqueeze(1)\n'
            '        # numinous-hook:hopper_fp8kv_pad\n'
            f'        if self.dsa_index_kpool > 1 and __import__("os").environ.get("{ENV}") == "1":\n'
            '            # kpool tails make rows index_topk + kpool - 1 wide; the SM90 FlashMLA sparse decode kernel wants a\n'
            '            # multiple of 64 and masks -1 entries, so pad with -1 to the width the tile scheduler was built for\n'
            '            _width = (self.dsa_index_topk + self.dsa_index_kpool - 1 + 63) // 64 * 64\n'
            '            assert indices.shape[-1] <= _width, (indices.shape, _width)\n'
            '            if indices.shape[-1] < _width:\n'
            '                indices = torch.nn.functional.pad(\n'
            '                    indices, (0, _width - indices.shape[-1]), value=-1\n'
            '                )\n'
            '        else:\n'
            '            assert (\n'
            '                indices.shape[-1] == self.dsa_index_topk\n'
            '            )  # requirement of FlashMLA decode kernel\n')

SLICE_ANCHOR = ('        o, _ = flash_mla_with_kvcache(\n'
                '            q=q_input,\n'
                '            k_cache=kv_cache,\n'
                '            cache_seqlens=cache_seqlens,\n'
                '            head_dim_v=v_head_dim,\n'
                '            tile_scheduler_metadata=metadata.flashmla_metadata.flashmla_metadata,\n'
                '            num_splits=metadata.flashmla_metadata.num_splits,\n'
                '            softmax_scale=sm_scale,\n'
                '            indices=indices,\n'
                '            # doc says it is not used, but if pass in None then error\n'
                '            block_table=torch.empty(\n'
                '                (q_all.shape[0], 0), dtype=torch.int32, device=q_all.device\n'
                '            ),\n'
                '            is_fp8_kvcache=True,\n'
                '        )\n')
SLICE_TEXT = ('        # numinous-hook:hopper_fp8kv_slice\n'
              f'        _slice = int(__import__("os").environ.get("{ENV}_SLICE", "2048"))\n'
              '        if (\n'
              f'            __import__("os").environ.get("{ENV}") == "1"\n'
              '            and _slice > 0\n'
              '            and q_input.shape[0] > _slice\n'
              '            and not torch.cuda.is_current_stream_capturing()\n'
              '        ):\n'
              '            # prefill runs this decode kernel once per token; its split-KV fp32 accumulators grow with\n'
              '            # rows x heads x 512, so a 16K-token chunk asks ~2 GiB at once (measured OOM, Flash on H200,\n'
              '            # 2026-10-04). Slice the rows; each slice gets its own tile-scheduler metadata.\n'
              '            from sgl_kernel.flash_mla import get_mla_metadata as _numinous_gmm\n'
              '\n'
              '            _outs = []\n'
              '            for _s0 in range(0, q_input.shape[0], _slice):\n'
              '                _s1 = min(q_input.shape[0], _s0 + _slice)\n'
              '                _ones = torch.ones(_s1 - _s0, dtype=torch.int32, device=q_all.device)\n'
              '                _meta, _splits = _numinous_gmm(\n'
              '                    cache_seqlens=_ones,\n'
              '                    num_q_tokens_per_head_k=q_input.shape[2],\n'
              '                    num_heads_k=1,\n'
              '                    num_heads_q=q_input.shape[2],\n'
              '                    is_fp8_kvcache=True,\n'
              '                    topk=indices.shape[-1],\n'
              '                )\n'
              '                _o, _ = flash_mla_with_kvcache(\n'
              '                    q=q_input[_s0:_s1],\n'
              '                    k_cache=kv_cache,\n'
              '                    cache_seqlens=_ones,\n'
              '                    head_dim_v=v_head_dim,\n'
              '                    tile_scheduler_metadata=_meta,\n'
              '                    num_splits=_splits,\n'
              '                    softmax_scale=sm_scale,\n'
              '                    indices=indices[_s0:_s1].contiguous(),\n'
              '                    block_table=torch.empty(\n'
              '                        (_s1 - _s0, 0), dtype=torch.int32, device=q_all.device\n'
              '                    ),\n'
              '                    is_fp8_kvcache=True,\n'
              '                )\n'
              '                _outs.append(_o)\n'
              '            o = torch.cat(_outs, dim=0)\n'
              '        else:\n'
              + "".join("    " + ln + "\n" if ln else "\n" for ln in SLICE_ANCHOR.splitlines()))

HOOKS = (
    Hook("hopper_fp8kv_kpool", KPOOL, KPOOL_ANCHOR, KPOOL_TEXT, replace=True),
    Hook("hopper_fp8kv_meta", BACKEND, META_ANCHOR, META_TEXT, replace=True),
    Hook("hopper_fp8kv_pad", BACKEND, PAD_ANCHOR, PAD_TEXT, replace=True),
    Hook("hopper_fp8kv_slice", BACKEND, SLICE_ANCHOR, SLICE_TEXT, replace=True),
)
# hopper_fp8kv_xform is WITHDRAWN (2026-10-05 05:2x, measured): with it, arm F relaunch 2 passed the assert, then hit
# CUDA illegal memory access + XID 31 FAULT_PDE VIRT_READ on all 4 PREFILL GPUs at the first long prompt
# (weights-cache:/verify_logs/glm-5.3-flash_standard@8xH200-F_20261005-050839). The 2048-only assert was the only guard
# against index rows that are not prefill positions; relaxing it alone is unsafe. rtxattn's RTX fix pairs the same
# relaxation with four rtxattn_kpool_unfused hooks that keep the kpool index space consistent. Without those, the PD
# prefill draft extend with MTP IndexShare must stay off (amdflash relaunch 3: index_share_for_mtp_iteration=false ->
# fused TopK everywhere, the path never reaches this transform). XFORM_ANCHOR / XFORM_TEXT are kept for the record.
WITHDRAWN = {"hopper_fp8kv_xform": "unsafe alone for kpool + PD prefill + MTP IndexShare (measured XID 31 2026-10-05)"}


def padded_width(index_topk: int, kpool: int) -> int:
    """The row width the hooks feed FlashMLA: index_topk + kpool - 1 rounded up to the kernel's 64-entry block."""
    return (index_topk + kpool - 1 + 63) // 64 * 64


def sglang_root() -> str:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def patch_source(src: str, path: str) -> tuple[str, list[str]]:
    """Apply every hook for one file. Returns (new text, status lines); raises SystemExit on a bad anchor."""
    out, missing = [], []
    for h in HOOKS:
        if h.path != path:
            continue
        if h.marker in src:
            out.append(f"{h.name}: already applied")
            continue
        if h.satisfied_by and h.satisfied_by in src:
            out.append(f"{h.name}: satisfied by {h.satisfied_by.split(':')[-1]}")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        src = src.replace(h.anchor, h.text if h.replace else h.anchor + h.text, 1)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("hopper_fp8kv hooks: " + "; ".join(missing))
    return src, out


def apply(root: str, check: bool = False) -> list[str]:
    lines = []
    for path in FILES:
        full = os.path.join(root, path)
        src = open(full).read()
        new, ls = patch_source(src, path)
        compile(new, full, "exec")
        if check:
            lines += [ln.replace("applied to", "anchor ok in") for ln in ls]
            continue
        if new != src:
            tmp = f"{full}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, full)
        lines += ls
    return lines


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
