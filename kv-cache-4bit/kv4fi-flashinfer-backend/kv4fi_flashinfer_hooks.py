"""Anchored patcher: `--kv-cache-dtype nvfp4` with `--attention-backend flashinfer` on SGLang v0.5.20 (kv4fi).

Stock v0.5.20 rejects the pair twice: the validator (srt/arg_groups/kv_cache_hook.py) lists MHA 4-bit backends
without flashinfer, and the KV access registry (srt/layers/quantization/fp4_kv_cache_quant_method.py) gives nvfp4
a flashinfer rule for prefill only (FP8 dequant workspace); decode is trtllm_mha-only, so FlashInferAttnBackend
raises at init. The only stock decode alternative inside the flashinfer backend would be the dead-code dequant
workspace path (re-dequantizes every token per layer per step through host loops; CUDA-graph unsafe).

This hook keeps FlashInfer for prefill (stock dequant-workspace path, unchanged) and routes every attention over the
FP4 cache that is not a plain prefill to a native NVFP4 Triton kernel (operations/kernels/kv4fi_nvfp4_decode.py,
installed as sglang/srt/layers/attention/kv4fi_nvfp4_decode.py):
  - decode (also EAGLE draft decode, which plans through the same decode wrappers),
  - target verify (EAGLE tree mask read from the planned packed custom mask; DFLASH/DSPARK causal),
  - draft extend when it runs the paged (non-ragged) path (the cuda-graph path),
driven by the exact kv_indptr / kv_indices / mask buffers the FlashInfer wrapper planned, so CUDA-graph replay and
speculative decoding stay correct (stock never prepares the workspace for verify: it would read stale FP8).

    python3 operations/patches/kv4fi_flashinfer_hooks.py            # patch the installed sglang
    python3 operations/patches/kv4fi_flashinfer_hooks.py --check    # verify anchors only
    python3 operations/patches/kv4fi_flashinfer_hooks.py --root DIR # patch a copy (DIR = sglang package dir)

Limits (raise loudly): sliding-window layers (Gemma) and cross attention are not routed; Gemma also fails the stock
single-wrapper assert in the nvfp4 prefill path. Test: operations/patches/test_kv4fi_flashinfer_hooks.py.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
KERNEL_SRC = HERE.parents[0] / "kernels" / "kv4fi_nvfp4_decode.py"
KERNEL_DST = "srt/layers/attention/kv4fi_nvfp4_decode.py"
MARK = "numinous-hook:kv4fi"

VALIDATOR = "srt/arg_groups/kv_cache_hook.py"
REGISTRY = "srt/layers/quantization/fp4_kv_cache_quant_method.py"
BACKEND = "srt/layers/attention/flashinfer_backend.py"

FORWARD_METHOD = f'''
    def _kv4fi_forward(self, q, layer, wrapper, ql, packed_mask, mask_indptr, causal=True, return_lse=False):
        # {MARK}: native NVFP4 attention over the FP4 pool with the indices the FlashInfer wrapper planned.
        from sglang.srt.layers.attention.kv4fi_nvfp4_decode import nvfp4_attention

        if layer.is_cross_attention:
            raise NotImplementedError("kv4fi: cross attention is not supported with the NVFP4 KV cache")
        if layer.sliding_window_size is not None and layer.sliding_window_size > 0:
            raise NotImplementedError("kv4fi: sliding-window layers are not supported with the NVFP4 KV cache")
        k_fp4, v_fp4, k_sf, v_sf = self.token_to_kv_pool.get_raw_kv_buffer(layer.layer_id)
        k_gs, v_gs = self.kv_cache_quant_method.get_bmm_scales(layer.layer_id)
        res = nvfp4_attention(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k_fp4,
            k_sf,
            v_fp4,
            v_sf,
            wrapper._paged_kv_indptr_buf,
            wrapper._paged_kv_indices_buf,
            k_gs,
            v_gs,
            layer.scaling,
            ql=ql,
            qo_indptr=getattr(wrapper, "_qo_indptr_buf", None),
            causal=causal,
            mask=packed_mask,
            mask_indptr=mask_indptr,
            logit_cap=layer.logit_cap,
            return_lse=return_lse,
        )
        if return_lse:
            return res[0], res[1]
        return res.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
'''

EDITS = {
    VALIDATOR: [
        (
            '                    KV4_ATTENTION_MHA_BACKEND_CHOICES = [\n'
            '                        "triton",\n'
            '                        "torch_native",\n'
            '                        "flex_attention",\n'
            '                        "trtllm_mha",\n'
            '                    ]\n',
            '                    KV4_ATTENTION_MHA_BACKEND_CHOICES = [\n'
            '                        "triton",\n'
            '                        "torch_native",\n'
            '                        "flex_attention",\n'
            '                        "trtllm_mha",\n'
            f'                        "flashinfer",  # {MARK}: native NVFP4 decode/verify kernel\n'
            '                    ]\n',
        ),
    ],
    REGISTRY: [
        (
            '_NVFP4_DECODE_BACKENDS = frozenset({"trtllm_mha"})\n',
            f'_NVFP4_DECODE_BACKENDS = frozenset({{"trtllm_mha", "flashinfer"}})  # {MARK}\n',
        ),
    ],
    BACKEND: [
        # flag: this backend instance serves decode over an nvfp4 pool natively
        (
            '        self.dq_page_table = None\n'
            '        self.dq_paged_kernel_lens = None\n'
            '        self.cpu_req_pool_indices = None\n'
            '        # FP4 fake-quant prefill/decode exposes an FP8 workspace to FlashInfer.\n',
            f'        # {MARK}: decode/verify over the packed FP4 pool with the native Triton kernel\n'
            '        self.kv4fi_native_decode = (\n'
            '            self.decode_kv_access is not None\n'
            '            and self.decode_kv_access.kind == KVCacheAttentionAccessKind.NATIVE_FP4\n'
            '            and self.decode_kv_access.scale_recipe == "nvfp4"\n'
            '            and decode_backend == "flashinfer"\n'
            '        )\n'
            '        self.dq_page_table = None\n'
            '        self.dq_paged_kernel_lens = None\n'
            '        self.cpu_req_pool_indices = None\n'
            '        # FP4 fake-quant prefill/decode exposes an FP8 workspace to FlashInfer.\n',
        ),
        # decode
        (
            '        if self.decode_uses_dequant_workspace:\n'
            '            kv_cache = (\n'
            '                self.token_to_kv_pool.get_flashinfer_decode_dequant_workspace_kv_buffer(\n',
            f'        if self.kv4fi_native_decode:  # {MARK}\n'
            '            return self._kv4fi_forward(q, layer, decode_wrapper, 1, None, None)\n'
            '        if self.decode_uses_dequant_workspace:\n'
            '            kv_cache = (\n'
            '                self.token_to_kv_pool.get_flashinfer_decode_dequant_workspace_kv_buffer(\n',
        ),
        # target verify and paged draft extend: write the new K/V, then attend natively over the planned indices
        (
            '        assert not (self.prefill_uses_dequant_workspace and layer.is_cross_attention), (\n'
            '            "FP4 dequant KV cache is not supported for cross-attention"\n'
            '        )\n',
            '        assert not (self.prefill_uses_dequant_workspace and layer.is_cross_attention), (\n'
            '            "FP4 dequant KV cache is not supported for cross-attention"\n'
            '        )\n'
            f'        if self.kv4fi_native_decode and (  # {MARK}\n'
            '            forward_batch.forward_mode.is_target_verify()\n'
            '            or (\n'
            '                forward_batch.forward_mode.is_draft_extend_v2()\n'
            '                and not self.forward_metadata.use_ragged\n'
            '                and forward_batch.spec_info is not None\n'
            '            )\n'
            '        ):\n'
            '            if k is not None and save_kv_cache:\n'
            '                assert v is not None\n'
            '                self.token_to_kv_pool.set_kv_buffer(\n'
            '                    layer,\n'
            '                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),\n'
            '                    k,\n'
            '                    v,\n'
            '                    *self._kv_write_scales(layer),\n'
            '                )\n'
            '            return self._kv4fi_forward(\n'
            '                q,\n'
            '                layer,\n'
            '                prefill_wrapper_paged,\n'
            '                int(forward_batch.spec_info.num_tokens_per_req),\n'
            '                getattr(prefill_wrapper_paged, "_custom_mask_buf", None),\n'
            '                getattr(prefill_wrapper_paged, "_mask_indptr_buf", None),\n'
            '            )\n',
        ),
        # stock bug (measured on unsloth/Qwen3.8-27B-NVFP4, which ships k_scale/v_scale ~0.0245): the paged prefill
        # over the FP4 dequant workspace still passes the checkpoint's FP8 per-tensor KV scales to FlashInfer, but the
        # workspace holds values in original units (dequant = code * block scale * global; current chunk = plain FP8
        # cast), so logits were multiplied by k_scale (near-uniform attention) and outputs by v_scale (41x too small).
        # FP8 pools pre-divide K/V by these scales at write time; the FP4 workspace does not.
        (
            '                logits_soft_cap=logits_soft_cap,\n'
            '                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.\n'
            '                k_scale=layer.k_scale_float,\n'
            '                v_scale=layer.v_scale_float,\n'
            '            )\n'
            '        else:\n',
            '                logits_soft_cap=logits_soft_cap,\n'
            '                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.\n'
            f'                # {MARK}: the FP4 dequant workspace is in original units; no FP8 per-tensor scales\n'
            '                k_scale=None if self.prefill_uses_dequant_workspace else layer.k_scale_float,\n'
            '                v_scale=None if self.prefill_uses_dequant_workspace else layer.v_scale_float,\n'
            '            )\n'
            '        else:\n',
        ),
        # eager draft extend (ragged current chunk + paged prefix): the paged prefix part over the FP4 pool
        (
            '                o2, s2 = prefill_wrapper_paged.forward_return_lse(\n'
            '                    q.view(-1, layer.tp_q_head_num, layer.head_dim),\n'
            '                    kv_cache,\n'
            '                    causal=False,\n'
            '                    sm_scale=layer.scaling,\n'
            '                    window_left=swa_window_left,\n'
            '                    logits_soft_cap=logits_soft_cap,\n'
            '                    # Must use _float to avoid device-to-host copy that breaks cuda graph capture.\n'
            '                    k_scale=layer.k_scale_float,\n'
            '                    v_scale=layer.v_scale_float,\n'
            '                )\n',
            f'                if self.kv4fi_native_decode and (  # {MARK}\n'
            '                    forward_batch.forward_mode.is_draft_extend_v2()\n'
            '                    and forward_batch.spec_info is not None\n'
            '                ):\n'
            '                    o2, s2 = self._kv4fi_forward(\n'
            '                        q,\n'
            '                        layer,\n'
            '                        prefill_wrapper_paged,\n'
            '                        int(forward_batch.spec_info.num_tokens_per_req),\n'
            '                        None,\n'
            '                        None,\n'
            '                        causal=False,\n'
            '                        return_lse=True,\n'
            '                    )\n'
            '                else:\n'
            '                    o2, s2 = prefill_wrapper_paged.forward_return_lse(\n'
            '                        q.view(-1, layer.tp_q_head_num, layer.head_dim),\n'
            '                        kv_cache,\n'
            '                        causal=False,\n'
            '                        sm_scale=layer.scaling,\n'
            '                        window_left=swa_window_left,\n'
            '                        logits_soft_cap=logits_soft_cap,\n'
            '                        k_scale=None if self.prefill_uses_dequant_workspace else layer.k_scale_float,\n'
            '                        v_scale=None if self.prefill_uses_dequant_workspace else layer.v_scale_float,\n'
            '                    )\n',
        ),
        (
            '\n    def _get_wrapper_idx(self, layer: RadixAttention):\n',
            FORWARD_METHOD,
        ),
    ],
}


def sglang_root() -> str:
    import importlib.util
    return list(importlib.util.find_spec("sglang").submodule_search_locations)[0]


def apply(src: str, target: str) -> str:
    if MARK in src:
        return src
    for anchor, repl in EDITS[target]:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"kv4fi_flashinfer_hooks: anchor found {n} times (need 1) in {target}:\n{anchor}")
        src = src.replace(anchor, repl)
    return src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = a.root or sglang_root()
    for target in (VALIDATOR, REGISTRY, BACKEND):
        path = os.path.join(root, target)
        src = open(path).read()
        new = apply(src, target)
        compile(new, path, "exec")
        if a.check:
            print(f"kv4fi_flashinfer_hooks: anchors ok in {path}" + (" (already patched)" if new == src else ""))
            continue
        if new != src:
            open(path, "w").write(new)
        print(f"kv4fi_flashinfer_hooks: patched {path}")
    if not a.check:
        dst = os.path.join(root, KERNEL_DST)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(KERNEL_SRC, dst)
        compile(open(dst).read(), dst, "exec")
        print(f"kv4fi_flashinfer_hooks: installed {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
