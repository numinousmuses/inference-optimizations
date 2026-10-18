"""Regression test for the Qwen 3.8 NVFP4-KV accuracy failure (agent q38kv4; root cause found by kv4trt/kv4fi).

Failure (measured, RTX PRO 6000 gate, 2026-09-29): kv4trt_q38_nvfp4 vs qwen38_standard_m80 KL 2.39, top-1 0.425,
GSM8K 0.88 -> 0.785; Qwen 3.6 on the same route was clean. Root cause (source_code, SGLang v0.5.20
flashinfer_backend.py forward_extend): with an NVFP4 pool, prefill reads a FlashInfer FP8 *dequant workspace* that
already holds real-unit K/V (FP4 code x E4M3 block scale x the per-layer global scale; the current chunk is a plain
FP8 cast), but the call still passes k_scale=layer.k_scale_float / v_scale=layer.v_scale_float, the checkpoint's FP8
per-tensor KV scales. unsloth/Qwen3.8-27B-NVFP4 ships them (kv_cache_scheme FP8 static per-tensor, 0.02-0.04), so
logits are multiplied by ~0.03 (near-uniform attention) and outputs by ~0.025 in all 16 full-attention layers on
every prefill. nvidia/Qwen3.6-35B-A3B-NVFP4 has no KV scales (1.0) -> unaffected. The fix (kv4trt_hooks edit 8,
kv4fi_flashinfer_hooks prefill edits): no per-tensor scales when the read goes through the dequant workspace.
After the fix (measured, same gate): KL 0.230, top-1 0.937, GSM8K 0.87 (control 0.88).

What this file checks:
  CPU  both hook sets rewrite every FlashInfer prefill call that can read the workspace (anchored on the v0.5.20 file);
       the checkpoint facts behind the Qwen3.8-only break (when the checkpoint is on /cache).
  GPU  the exact numeric chain SGLang runs: NVFP4KVCacheMethod store (global scale = checkpoint k_scale) ->
       dequantize_prev_kv into the FP8 workspace (+ current chunk FP8 cast) -> FlashInfer paged prefill; with the
       fix the output matches a bf16 reference within FP4 error, with the stock scales it is off by ~100%.

    python3 -m pytest -q operations/patches/test_q38kv4_fp4_prefill_scale.py                                 # CPU
    python3 operations/devbox/devbox.py test rtx-qwen38 operations/patches/test_q38kv4_fp4_prefill_scale.py -q -s
"""
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kv4trt_hooks as trt  # noqa: E402

FORK = pathlib.Path("/home/ubuntu/sglang-fork")
FI_FILE = "srt/layers/attention/flashinfer_backend.py"
CKPT = "/cache/unsloth__Qwen3.8-27B-NVFP4"
HAVE_SGL = importlib.util.find_spec("sglang") is not None
need_gpu = pytest.mark.skipif(not (HAVE_SGL and torch.cuda.is_available()), reason="needs SGLang + a CUDA GPU")
# layers.3 of unsloth/Qwen3.8-27B-NVFP4 (measured from the safetensors by kv4tri; re-read below when on /cache)
K_SCALE, V_SCALE = 0.02747, 0.02454


def _stock_fi_src() -> str:
    """The stock v0.5.20 flashinfer_backend.py: git tag first (installed packages on shared boxes may carry hooks)."""
    if FORK.exists():
        return subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{FI_FILE}"],
                              capture_output=True, text=True, check=True).stdout
    if HAVE_SGL:
        root = list(importlib.util.find_spec("sglang").submodule_search_locations)[0]
        repo = pathlib.Path(root).parents[1]
        r = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:python/sglang/{FI_FILE}"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
    pytest.skip("no stock SGLang v0.5.20 source")


def _scale_sites(src: str) -> list[str]:
    """Each `k_scale=...` argument line in forward_extend (the prefill paths)."""
    body = src[src.index("    def forward_extend("):src.index("    def forward_decode(")]
    return [ln.strip() for ln in body.splitlines() if re.match(r"\s*k_scale=", ln)]


def test_stock_prefill_passes_checkpoint_scales_over_workspace():
    src = _stock_fi_src()
    assert "prefill_uses_dequant_workspace" in src
    sites = _scale_sites(src)
    assert sites and all(s == "k_scale=layer.k_scale_float," for s in sites), sites  # the bug, as shipped


def test_kv4trt_hook_drops_scales_on_every_workspace_prefill_site():
    patched = trt.apply(_stock_fi_src(), trt.FLASHINFER)
    sites = _scale_sites(patched)
    assert sites and all(s == "k_scale=None if self.prefill_uses_dequant_workspace else layer.k_scale_float,"
                         for s in sites), sites
    body = patched[patched.index("    def forward_decode("):]
    assert "k_scale=None if self.decode_uses_dequant_workspace else layer.k_scale_float," in body
    compile(patched, FI_FILE, "exec")


def test_kv4fi_hook_drops_scales_on_every_workspace_prefill_site():
    import kv4fi_flashinfer_hooks as fi
    patched = fi.apply(_stock_fi_src(), fi.BACKEND)
    sites = _scale_sites(patched)
    assert sites and all(s == "k_scale=None if self.prefill_uses_dequant_workspace else layer.k_scale_float,"
                         for s in sites), sites
    compile(patched, FI_FILE, "exec")


@pytest.mark.skipif(not os.path.exists(os.path.join(CKPT, "config.json")), reason="checkpoint not on /cache")
def test_checkpoint_facts_behind_the_qwen38_only_break():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    kv = cfg["quantization_config"]["kv_cache_scheme"]
    assert kv["type"] == "float" and kv["num_bits"] == 8 and kv["strategy"] == "tensor"  # FP8 convention amax/448
    wm = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    ks = sorted(k for k in wm if k.endswith("self_attn.k_scale"))
    assert len(ks) == 16 and not [k for k in ks if k.startswith("mtp.")]  # 16 full-attention layers; MTP has none
    from safetensors import safe_open
    key = "model.language_model.layers.3.self_attn.k_scale"
    with safe_open(os.path.join(CKPT, wm[key]), "pt") as f:
        k3 = float(f.get_tensor(key))
    assert abs(k3 - K_SCALE) < 1e-4


def _attn_ref(q, k, v, scale):
    """Causal attention, the C queries aligned to the end of the L keys (FlashInfer's causal convention). fp32."""
    C, H, D = q.shape
    L, KVH, _ = k.shape
    g = H // KVH
    kk = k.float().repeat_interleave(g, dim=1)
    vv = v.float().repeat_interleave(g, dim=1)
    s = torch.einsum("chd,lhd->hcl", q.float(), kk) * scale
    pos_q = torch.arange(L - C, L, device=q.device)[:, None]
    s = s.masked_fill(torch.arange(L, device=q.device)[None, :] > pos_q, float("-inf"))
    return torch.einsum("hcl,lhd->chd", s.softmax(-1), vv)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@need_gpu
@pytest.mark.parametrize("hq,hkv", [(24, 4)])            # Qwen 3.8 full attention: 24 q / 4 kv heads x 256
@pytest.mark.parametrize("prefix,chunk", [(3000, 1024), (12000, 512)])
def test_fp4_workspace_prefill_numerics(hq, hkv, prefix, chunk):
    import flashinfer  # noqa: F401  (engine ptxas + wrappers)
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import NVFP4KVCacheMethod

    dev, d = "cuda", 256
    L = prefix + chunk
    g = torch.Generator(device=dev).manual_seed(3)
    # K/V with the calibrated range: amax ~ 0.7 x (448 x checkpoint scale), a few outlier channels (as real K has)
    k = torch.randn(L, hkv, d, device=dev, generator=g) * 1.2
    k[:, :, :8] *= 5.0
    k = (k * (0.7 * 448 * K_SCALE / k.abs().max())).to(torch.bfloat16)
    v = torch.randn(L, hkv, d, device=dev, generator=g)
    v = (v * (0.7 * 448 * V_SCALE / v.abs().max())).to(torch.bfloat16)
    q = (torch.randn(chunk, hq, d, device=dev, generator=g) * 1.5).to(torch.bfloat16)
    sm = d ** -0.5
    ref = _attn_ref(q, k, v, sm)

    # SGLang store: NVFP4 with global scale = checkpoint scale (load_scales_from_model on SM120)
    qm = NVFP4KVCacheMethod(num_layers=1, device=dev)
    qm.k_scales_gpu.fill_(K_SCALE)
    qm.v_scales_gpu.fill_(V_SCALE)
    qm.k_scales_float, qm.v_scales_float = [K_SCALE], [V_SCALE]
    buf = qm.create_buffers(prefix + 16, hkv, d, 1, dev)
    loc = torch.arange(prefix, device=dev)
    qm.quantize_and_store(buf["k_buffer"][0], buf["v_buffer"][0], buf["k_scale_buffer"][0],
                          buf["v_scale_buffer"][0], loc, k[:prefix].contiguous(), v[:prefix].contiguous(),
                          qm.k_scales_gpu[:1], qm.v_scales_gpu[:1])
    kp, vp = qm.dequantize_prev_kv(buf["k_buffer"][0][loc], buf["k_scale_buffer"][0][loc].view(torch.float8_e4m3fn),
                                   buf["v_buffer"][0][loc], buf["v_scale_buffer"][0][loc].view(torch.float8_e4m3fn),
                                   0)
    # workspace = dequantized prefix + the current chunk cast to FP8 (memory_pool dequant-workspace build)
    kw = torch.cat([kp, k[prefix:].to(torch.float8_e4m3fn)])
    vw = torch.cat([vp, v[prefix:].to(torch.float8_e4m3fn)])
    # FP8 pool (the control's store): pre-divided by the checkpoint scale, FlashInfer re-applies it
    kf = (k.float() / K_SCALE).to(torch.float8_e4m3fn)
    vf = (v.float() / V_SCALE).to(torch.float8_e4m3fn)

    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    qo = torch.tensor([0, chunk], dtype=torch.int32, device=dev)
    kvp = torch.tensor([0, L], dtype=torch.int32, device=dev)
    idx = torch.arange(L, dtype=torch.int32, device=dev)
    last = torch.ones(1, dtype=torch.int32, device=dev)
    w.plan(qo, kvp, idx, last, hq, hkv, d, 1, causal=True, q_data_type=torch.bfloat16,
           kv_data_type=torch.float8_e4m3fn, sm_scale=sm)

    def run(kc, vc, ks, vs):
        return w.forward(q, (kc.unsqueeze(1), vc.unsqueeze(1)), causal=True, sm_scale=sm, k_scale=ks, v_scale=vs)

    err_fp8 = _rel(run(kf, vf, K_SCALE, V_SCALE), ref)   # control class
    err_fix = _rel(run(kw, vw, None, None), ref)         # fixed NVFP4 read
    err_bug = _rel(run(kw, vw, K_SCALE, V_SCALE), ref)   # stock NVFP4 read
    print(f"\n[q38kv4] prefix {prefix} chunk {chunk}: rel err vs bf16  fp8-pool {err_fp8:.4f}  "
          f"nvfp4-fixed {err_fix:.4f}  nvfp4-stock {err_bug:.4f}")
    assert err_fp8 < 0.06
    assert err_fix < 0.15            # FP4 KV error class (kv4trt random-data attention error ~0.11)
    assert err_bug > 0.5             # the shipped bug: output scaled by ~v_scale, softmax flattened by ~k_scale
