"""Anchored patcher: serve NVFP4 KV with native FP4 reads in TRT-LLM MHA for speculative decoding (SGLang v0.5.20:
srt/layers/attention/trtllm_mha_backend.py, srt/layers/attention/hybrid_attn_backend.py,
srt/speculative/draft_utils.py).

Stock v0.5.20 with `--kv-cache-dtype nvfp4 --prefill-attention-backend flashinfer --decode-attention-backend trtllm_mha`
and EAGLE/MTP cannot boot (measured on RTX PRO 6000, Qwen3.8 and Qwen3.6): target-verify goes to the prefill backend
(speculative_attention_mode=prefill), and FlashInfer's NVFP4 path dequantizes each request's whole prefix into an FP8
workspace from host-side prefix lengths (memory_pool._prepare_dequant_extend_workspace), which is None under CUDA-graph
capture ("TypeError: 'NoneType' object is not subscriptable"). Even eagerly it would re-dequantize ~100k tokens x
layers every verify step. With --speculative-attention-mode decode, verify and draft-extend go to trtllm_mha, whose
forward_extend refuses native FP4 outright ("supports decode only"); and mode=decode also sends the draft model's
post-prefill DRAFT_EXTEND (a real prompt prefill) to trtllm_mha (measured: same raise in the draft prefill-graph
capture), so that flag is not the route.

Verify (TARGET_VERIFY) and draft-extend (DRAFT_EXTEND_V2) are fixed-q-length decode calls over the paged cache (stock
already serves them with the decode kernel for FP8/BF16). This hook:
  0. __init__: prebuilds the XQA speculative mask for q_len = speculative_num_draft_tokens (outside graph capture);
  1. forward_extend: for native NVFP4 and a non-ragged TARGET_VERIFY / DRAFT_EXTEND_V2 batch, runs
     `_numinous_fp4_decode_mode_extend` (quantize-write the new K/V exactly as forward_decode does, then the decode
     kernel over FP4 pages + FP8 block scales with q_len_per_req) instead of raising; real prefills still raise;
  2. _run_fixed_q_len_decode takes an optional `mask` (sliced per request group) and forwards it;
  3. the stock fixed-q decode-mode call passes the XQA causal mask too: XQA (SM90/SM120) asserts "Mask is required
     for speculative decoding" when q_len_per_req > 1, so stock trtllm_mha verify in decode mode could not run on
     SM120 at all (FP8 or FP4). trtllm-gen (SM100) takes no mask: `_numinous_xqa_causal_mask` returns None there.
  5. hybrid_attn_backend.HybridAttnBackend: when the decode backend reads FP4 natively and the prefill backend does
     not, TARGET_VERIFY and DRAFT_EXTEND_V2 are routed to the decode backend (spec_attn_is_decode forced True for
     this pair; prompt prefills and DRAFT_EXTEND stay on the prefill backend);
  6. draft_utils.create_draft_extend_backend: with NVFP4 KV, prefill flashinfer and decode trtllm_mha, the draft's
     extend backend becomes HybridAttnBackend(flashinfer, trtllm_mha) instead of flashinfer alone, so the draft's
     DRAFT_EXTEND_V2 (graph-captured) also reads FP4 natively while its post-prefill DRAFT_EXTEND stays on flashinfer.
Without FP4 KV every branch is the stock code path (FP8/BF16 arms are unaffected apart from the XQA mask in 3/4).
Mask format (flashinfer/xqa.py): uint16 [bs, q_len, ceil(q_len/32)*2], bit i of row j = row j attends draft token i;
causal chain = bits 0..j (verified against a causal reference: operations/kernels/test_kv4trt_xqa_nvfp4.py).

    python3 operations/patches/kv4trt_hooks.py [--check] [--root DIR]
Test: operations/patches/test_kv4trt_hooks.py. Missing anchors fail loudly.
"""
from __future__ import annotations

import argparse
import os
import sys

TARGET = "srt/layers/attention/trtllm_mha_backend.py"
HYBRID = "srt/layers/attention/hybrid_attn_backend.py"
DRAFT = "srt/speculative/draft_utils.py"
REGISTRY = "srt/layers/attention/attention_registry.py"
FLASHINFER = "srt/layers/attention/flashinfer_backend.py"
MARK = "numinous-hook:kv4trt"

EDITS = [
    # 0. prebuild the XQA causal mask for the configured draft length (never allocate during graph capture)
    ("        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens\n",
     "        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens\n"
     f"        # {MARK}: XQA speculative masks, keyed by q_len, prebuilt outside CUDA-graph capture\n"
     "        self._numinous_masks = {}\n"
     "        self._numinous_mask_rows = max(4096, int(model_runner.req_to_token_pool.req_to_token.shape[0]) + 1)\n"),
    ("        self.is_xqa_impl = get_platform().is_sm90 or get_platform().is_sm120\n",
     "        self.is_xqa_impl = get_platform().is_sm90 or get_platform().is_sm120\n"
     f"        # {MARK}: prebuild the verify / draft-extend mask (q_len = draft tokens) before any graph capture\n"
     "        if (self.speculative_num_draft_tokens or 0) > 1:\n"
     "            self._numinous_xqa_causal_mask(int(self.speculative_num_draft_tokens), model_runner.device)\n"),
    # 1. native FP4 verify / draft-extend instead of the blanket raise
    ("        if self.decode_uses_native_fp4:\n"
     "            raise RuntimeError(\n"
     "                \"TRTLLM MHA with native FP4 KV cache supports decode only; \"\n",
     "        if self.decode_uses_native_fp4:\n"
     f"            # {MARK}: verify / draft-extend are fixed-q decode calls: read FP4 natively\n"
     "            _fm = forward_batch.forward_mode\n"
     "            if (\n"
     "                self.is_nvfp4_kvcache\n"
     "                and (_fm.is_target_verify() or _fm.is_draft_extend_v2())\n"
     "                and not self.forward_metadata.is_ragged_verify\n"
     "                and getattr(layer, \"attn_type\", None) != AttentionType.ENCODER_ONLY\n"
     "                and not is_cp_active(forward_batch)\n"
     "            ):\n"
     "                return self._numinous_fp4_decode_mode_extend(\n"
     "                    q, k, v, layer, forward_batch, save_kv_cache, **kwargs\n"
     "                )\n"
     "            raise RuntimeError(\n"
     "                \"TRTLLM MHA with native FP4 KV cache supports decode only; \"\n"),
    # 2. the helpers (before _get_nvfp4_decode_kv_cache)
    ("    def _get_nvfp4_decode_kv_cache(\n",
     f"    # {MARK}: helpers\n"
     "    def _numinous_xqa_causal_mask(self, q_len: int, device=None):\n"
     "        \"\"\"XQA draft mask (uint16 [rows, q_len, ceil(q_len/32)*2], causal chain) or None when the kernel\n"
     "        takes none (trtllm-gen) or q_len == 1.\"\"\"\n"
     "        if not self.is_xqa_impl or q_len <= 1:\n"
     "            return None\n"
     "        m = self._numinous_masks.get(q_len)\n"
     "        if m is None:\n"
     "            if torch.cuda.is_current_stream_capturing():\n"
     "                raise RuntimeError(\n"
     "                    f\"kv4trt: XQA mask for q_len {q_len} was not prebuilt before CUDA-graph capture\"\n"
     "                )\n"
     "            dev = device or self.device\n"
     "            words = ((q_len + 31) // 32) * 2\n"
     "            j = torch.arange(q_len, device=dev)[:, None]\n"
     "            i = torch.arange(words * 16, device=dev)[None, :]\n"
     "            bits = (i <= j).to(torch.int32).view(q_len, words, 16)\n"
     "            vals = (bits << torch.arange(16, device=dev, dtype=torch.int32)).sum(-1)\n"
     "            m = vals.to(torch.uint16)[None].expand(self._numinous_mask_rows, q_len, words).contiguous()\n"
     "            self._numinous_masks[q_len] = m\n"
     "        return m\n"
     "\n"
     "    def _numinous_fp4_decode_mode_extend(\n"
     "        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs\n"
     "    ):\n"
     "        \"\"\"TARGET_VERIFY / DRAFT_EXTEND_V2 over native NVFP4 KV: forward_decode's write + read with\n"
     "        q_len_per_req = the batch's fixed draft length.\"\"\"\n"
     "        if save_kv_cache and k is not None:\n"
     "            self.token_to_kv_pool.set_kv_buffer(\n"
     "                layer,\n"
     "                KVWriteLoc(forward_batch.out_cache_loc, self.forward_metadata.swa_out_cache_loc),\n"
     "                k,\n"
     "                v,\n"
     "                *self._kv_write_scales(layer),\n"
     "            )\n"
     "        if self.is_xqa_impl:\n"
     "            q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)\n"
     "        else:\n"
     "            q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)\n"
     "        kv_cache, kv_cache_block_scales = self._get_nvfp4_decode_kv_cache(layer)\n"
     "        k_scale, v_scale = self._get_nvfp4_bmm_scales(layer)\n"
     "        q_len = int(self.forward_metadata.max_seq_len_q)\n"
     "        o = self._run_fixed_q_len_decode(\n"
     "            q,\n"
     "            kv_cache,\n"
     "            self._get_layer_page_table(layer, forward_batch),\n"
     "            self.forward_metadata.cache_seqlens_int32,\n"
     "            bmm1_scale=k_scale * layer.scaling,\n"
     "            bmm2_scale=v_scale,\n"
     "            window_left=layer.sliding_window_size,\n"
     "            sinks=kwargs.get(\"sinks\", None),\n"
     "            q_len_per_req=q_len,\n"
     "            kv_cache_sf=kv_cache_block_scales,\n"
     "            mask=self._numinous_xqa_causal_mask(q_len),\n"
     "        )\n"
     "        if o.dtype != self.q_data_type:\n"
     "            o = o.to(self.q_data_type)\n"
     "        return o.view(-1, layer.tp_q_head_num * layer.head_dim)\n"
     "\n"
     "    def _get_nvfp4_decode_kv_cache(\n"),
    # 3. _run_fixed_q_len_decode forwards an optional mask
    ("        kv_cache_sf=None,\n"
     "    ) -> torch.Tensor:\n",
     "        kv_cache_sf=None,\n"
     f"        mask=None,  # {MARK}\n"
     "    ) -> torch.Tensor:\n"),
    ("                kwargs[\"q_len_per_req\"] = q_len_per_req\n",
     "                kwargs[\"q_len_per_req\"] = q_len_per_req\n"
     f"            if mask is not None:  # {MARK}: one identical causal block per request\n"
     "                kwargs[\"mask\"] = mask[: group_query.shape[0] // q_len_per_req]\n"),
    # 4. stock fixed-q verify / draft-extend (FP8/BF16 KV) gets the XQA mask too
    ("                    sinks=attention_sink,\n"
     "                    q_len_per_req=self.forward_metadata.max_seq_len_q,\n"
     "                )\n",
     "                    sinks=attention_sink,\n"
     "                    q_len_per_req=self.forward_metadata.max_seq_len_q,\n"
     f"                    mask=self._numinous_xqa_causal_mask(  # {MARK}\n"
     "                        int(self.forward_metadata.max_seq_len_q)\n"
     "                    ),\n"
     "                )\n"),
]


HYBRID_EDITS = [
    ("        self.spec_attn_is_prefill = get_spec().speculative_attention_mode == \"prefill\"\n",
     "        self.spec_attn_is_prefill = get_spec().speculative_attention_mode == \"prefill\"\n"
     f"        # {MARK}: a native-FP4 decode backend serves verify + draft-extend-v2 (the prefill backend's FP4\n"
     "        # path is a host-driven dequant workspace: not graph-capturable, and O(prefix) per step)\n"
     "        self._numinous_fp4_decode_modes = bool(\n"
     "            getattr(decode_backend, \"decode_uses_native_fp4\", False)\n"
     "        ) and not bool(getattr(prefill_backend, \"decode_uses_native_fp4\", False))\n"
     "        if self._numinous_fp4_decode_modes:\n"
     "            self.spec_attn_is_decode, self.spec_attn_is_prefill = True, False\n"),
    ("        if forward_mode.is_decode_or_idle():\n"
     "            return self.decode_backend\n"
     "        elif forward_mode.is_target_verify():\n",
     "        if forward_mode.is_decode_or_idle():\n"
     "            return self.decode_backend\n"
     f"        elif forward_mode.is_draft_extend_v2() and getattr(self, \"_numinous_fp4_decode_modes\", False):  # {MARK}\n"
     "            return self.decode_backend\n"
     "        elif forward_mode.is_target_verify():\n"),
]

DRAFT_EDITS = [
    ("        backend = self._create_backend(\n"
     "            backend_name,\n"
     "            backend_map,\n"
     "            \"EAGLE is not supported in attention backend {backend_type}\",\n"
     "        )\n",
     "        backend = self._create_backend(\n"
     "            backend_name,\n"
     "            backend_map,\n"
     "            \"EAGLE is not supported in attention backend {backend_type}\",\n"
     "        )\n"
     f"        # {MARK}: NVFP4 KV with flashinfer prefill + trtllm_mha decode -> the draft extend backend is the\n"
     "        # same split (DRAFT_EXTEND on flashinfer, graph-captured DRAFT_EXTEND_V2 on native-FP4 trtllm_mha)\n"
     "        _kv4_pf, _kv4_dc = attention_backends()\n"
     "        if (\n"
     "            backend is not None\n"
     "            and backend_name == \"prefill_attention_backend\"\n"
     "            and not self.draft_attn_backend\n"
     "            and _kv4_dc == \"trtllm_mha\"\n"
     "            and _kv4_pf != \"trtllm_mha\"\n"
     "            and not self.draft_model_runner.use_mla_backend\n"
     "        ):\n"
     "            from sglang.srt.utils import is_float4_e2m1fn_x2 as _kv4_is_fp4\n"
     "\n"
     "            if _kv4_is_fp4(self.draft_model_runner.kv_cache_dtype):\n"
     "                from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend\n"
     "\n"
     "                _kv4_stamp, _kv4_dec = self._create_trtllm_mha_prefill_backend()\n"
     "                if getattr(_kv4_dec, \"decode_uses_native_fp4\", False):\n"
     "                    _kv4_dec.prefill_attention_backend_str = _kv4_stamp\n"
     "                    _kv4_dec.decode_attention_backend_str = _kv4_stamp\n"
     "                    _kv4_h = HybridAttnBackend(self.draft_model_runner, backend, _kv4_dec)\n"
     "                    _kv4_h.prefill_attention_backend_str = backend.prefill_attention_backend_str\n"
     "                    _kv4_h.decode_attention_backend_str = _kv4_stamp\n"
     "                    backend = _kv4_h\n"),
]

# 7. SM100 (B200/B300): hybrid GDN models accept only triton / trtllm_mha / fa4, but NVFP4 prefill exists only on
#    flashinfer. Admit flashinfer as the PREFILL backend exactly for the NVFP4 split (decode trtllm_mha); every other
#    combination keeps the stock assert.
REGISTRY_EDITS = [
    ("                decode_be = runner.decode_attention_backend_str\n",
     "                decode_be = runner.decode_attention_backend_str\n"
     f"                # {MARK}: NVFP4 KV split on SM100: flashinfer prefill (dequant workspace) + trtllm_mha decode\n"
     "                if (\n"
     "                    prefill_be == \"flashinfer\"\n"
     "                    and decode_be == \"trtllm_mha\"\n"
     "                    and str(getattr(runner, \"kv_cache_dtype\", \"\")).endswith(\"float4_e2m1fn_x2\")\n"
     "                ):\n"
     "                    allowed = set(allowed) | {\"flashinfer\"}\n"),
]

# 8. Accuracy bug (measured: Qwen3.8 NVFP4 KL 2.39, top-1 0.42, GSM8K 0.88 -> 0.785; Qwen3.6, whose checkpoint has no
#    KV scales, was unaffected): FlashInfer's FP4 path dequantizes the cache into the FP8 workspace at REAL scale
#    (FP4 x block scale x global scale), then runs the FP8 wrapper with k_scale=layer.k_scale_float, i.e. multiplies
#    K and V a second time by the checkpoint's FP8 KV scale (0.02-0.04 for Qwen3.8). Workspace reads use scale 1.
def _fi_scale_edit(indent: int, flag: str):
    pad = " " * indent
    return (f"{pad}k_scale=layer.k_scale_float,\n{pad}v_scale=layer.v_scale_float,\n",
            f"{pad}# {MARK}: the FP4 dequant workspace already holds real-scale values\n"
            f"{pad}k_scale=None if self.{flag} else layer.k_scale_float,\n"
            f"{pad}v_scale=None if self.{flag} else layer.v_scale_float,\n")


FLASHINFER_EDITS = [_fi_scale_edit(16, "prefill_uses_dequant_workspace"),   # paged prefill
                    _fi_scale_edit(20, "prefill_uses_dequant_workspace"),   # ragged + paged-prefix merge
                    _fi_scale_edit(12, "decode_uses_dequant_workspace")]    # decode via workspace

TARGETS = {TARGET: EDITS, HYBRID: HYBRID_EDITS, DRAFT: DRAFT_EDITS, REGISTRY: REGISTRY_EDITS,
           FLASHINFER: FLASHINFER_EDITS}


def sglang_root() -> str:
    import importlib.util
    return list(importlib.util.find_spec("sglang").submodule_search_locations)[0]


def apply(src: str, target: str = TARGET) -> str:
    if MARK in src:
        return src
    for anchor, repl in TARGETS[target]:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"kv4trt_hooks: anchor found {n} times (need 1) in {target}:\n{anchor}")
        src = src.replace(anchor, repl)
    return src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    news = {}
    for target in TARGETS:                       # check every file before writing any
        path = os.path.join(a.root or sglang_root(), target)
        src = open(path).read()
        new = apply(src, target)
        compile(new, path, "exec")
        news[path] = (src, new)
    for path, (src, new) in news.items():
        if a.check:
            print(f"kv4trt_hooks: anchors ok in {path}" + (" (already patched)" if new == src else ""))
        elif new != src:
            open(path, "w").write(new)
            print(f"kv4trt_hooks: patched {path}")
        else:
            print(f"kv4trt_hooks: already patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
