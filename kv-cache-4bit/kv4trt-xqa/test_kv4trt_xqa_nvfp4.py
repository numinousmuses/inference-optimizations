"""kv4trt: does TRT-LLM MHA decode read NVFP4 KV natively on this GPU, and how close is it to the reference?

SGLang v0.5.20 `--decode-attention-backend trtllm_mha --kv-cache-dtype nvfp4` calls
`flashinfer.decode.trtllm_batch_decode_with_kv_cache` with packed FP4 K/V (uint8, head_dim/2), FP8 E4M3 block scales
(one per 16 values) and bmm1 = k_global * sm_scale, bmm2 = v_global (trtllm_mha_backend.forward_decode). On SM120
(RTX PRO 6000) the `auto` backend is XQA (JIT-compiled), on SM100 it is trtllm-gen (cubins). This test mirrors that
exact call sequence (the pool's `NVFP4KVQuantizeUtil.quantize` writer, the backend's page reshape and scales) for
each RTX model's full-attention head geometry and checks:

  1. the kernel runs (no missing cubin / JIT failure) for NVFP4 and FP8 KV, bf16 q, as SGLang drives it;
  2. its output matches an fp32 reference computed from the *dequantized* KV (kernel correctness, tight tol);
  3. the attention-output error of FP4 KV vs the bf16 original stays bounded, and is reported next to FP8 KV's
     (quantization cost at the attention output; the served accuracy gate decides shipping).

Geometries (config.json on /cache, measured 2026-09-28): Qwen3.8-27B 24q/4kv x256; Qwen3.6-35B-A3B 16q/2kv x256;
Gemma-4-26B-A4B sliding 16q/8kv x256 window 1024 and global 16q/2kv x512.

Run on a Blackwell box (imports flashinfer first, as the engine does):
    python3 operations/devbox/devbox.py test rtx-qwen38 operations/kernels/test_kv4trt_xqa_nvfp4.py -q -s
"""
from __future__ import annotations

import math

import pytest
import torch

try:
    import flashinfer  # noqa: F401  (sets the toolkit ptxas path the engine uses)
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil
    HAVE = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="needs a CUDA GPU with flashinfer + sglang")

GEOMS = {
    "qwen38": dict(hq=24, hkv=4, d=256, window=-1),
    "qwen36": dict(hq=16, hkv=2, d=256, window=-1),
    "gemma_swa": dict(hq=16, hkv=8, d=256, window=1024),
    "gemma_global": dict(hq=16, hkv=2, d=512, window=-1),
}


def _sm() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


CHUNK = 1 << 20  # tokens per (de)quantize call


def build_case(geom: dict, seq_lens: list[int], page: int, kv_dtype: str, global_scale: str, seed: int = 0,
               deq: bool = True):
    """Pool-shaped KV (tokens, hkv, d) written the way SGLang's pool writes it, paged like the backend reads it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    hq, hkv, d = geom["hq"], geom["hkv"], geom["d"]
    dev = "cuda"
    pages_per = [math.ceil(s / page) for s in seq_lens]
    n_pages = sum(pages_per) + 1  # page 0 unused (SGLang reserves slot 0)
    tokens = n_pages * page
    # K with per-channel outliers (RoPE'd keys have a few large channels), V plainer
    k = torch.randn(tokens, hkv, d, device=dev, generator=g, dtype=torch.float32)
    k[..., : d // 16] *= 6.0
    v = torch.randn(tokens, hkv, d, device=dev, generator=g, dtype=torch.float32)
    k, v = k.bfloat16(), v.bfloat16()
    q = torch.randn(len(seq_lens), hq, d, device=dev, generator=g, dtype=torch.float32).bfloat16()
    # block table: request i owns a contiguous run of pages, in a shuffled order (real pools are fragmented)
    perm = torch.randperm(n_pages - 1, generator=torch.Generator().manual_seed(seed)) + 1
    width = max(pages_per)
    bt = torch.zeros(len(seq_lens), width, dtype=torch.int32)
    off = 0
    for i, p in enumerate(pages_per):
        bt[i, :p] = perm[off: off + p]
        off += p
    bt = bt.to(dev)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)

    out = dict(q=q, k=k, v=v, bt=bt, sl=sl, page=page, geom=geom)
    if kv_dtype == "nvfp4":
        if global_scale == "ckpt_fp8":   # the RTX checkpoints ship FP8 KV calibration: k_scale = amax / 448
            ks = (k.float().abs().amax() / 448.0).reshape(1)
            vs = (v.float().abs().amax() / 448.0).reshape(1)
        else:                            # no scale in the checkpoint -> SGLang default 1.0
            ks = torch.ones(1, device=dev)
            vs = torch.ones(1, device=dev)
        if _sm()[0] == 10:               # NVFP4KVCacheMethod.load_scales_from_model: SM100 multiplies by E2M1_MAX
            ks, vs = ks * 6.0, vs * 6.0
        ks, vs = ks.float().contiguous(), vs.float().contiguous()
        # chunked: flashinfer's nvfp4_kv_(de)quantize index with 32-bit offsets (a >2^31-element call faults,
        # measured); the engine only ever quantizes the new tokens of a step
        def q4(x, gs):
            parts = [NVFP4KVQuantizeUtil.quantize(x[i:i + CHUNK].contiguous(), gs) for i in range(0, len(x), CHUNK)]
            return (torch.cat([p[0].view(torch.uint8) for p in parts]),
                    torch.cat([p[1].view(torch.uint8) for p in parts]))

        def dq4(x4, sf, gs):
            return torch.cat([NVFP4KVQuantizeUtil.dequantize(x4[i:i + CHUNK], sf[i:i + CHUNK].view(
                torch.float8_e4m3fn), gs).bfloat16() for i in range(0, len(x4), CHUNK)])

        k4, ksf = q4(k, ks)
        v4, vsf = q4(v, vs)
        out.update(k_store=k4, v_store=v4, k_sf=ksf, v_sf=vsf, ks=float(ks), vs=float(vs))
        if deq:
            out.update(k_deq=dq4(k4, ksf, ks), v_deq=dq4(v4, vsf, vs))
    elif kv_dtype == "fp8":
        ks = float(k.float().abs().amax() / 448.0)
        vs = float(v.float().abs().amax() / 448.0)
        k8 = (k.float() / ks).to(torch.float8_e4m3fn)
        v8 = (v.float() / vs).to(torch.float8_e4m3fn)
        out.update(k_store=k8, v_store=v8, ks=ks, vs=vs)
        if deq:
            out.update(k_deq=(k8.float() * ks).bfloat16(), v_deq=(v8.float() * vs).bfloat16())
    else:
        out.update(k_store=k, v_store=v, ks=1.0, vs=1.0, k_deq=k, v_deq=v)
    out["kv_dtype"] = kv_dtype
    return out


def paged(t: torch.Tensor, page: int, heads: int, d: int) -> torch.Tensor:
    """trtllm_mha_backend._reshape_paged_kv_cache: NHD pool rows -> HND page view (no copy)."""
    x = t.view(-1, page, heads, d).permute(0, 2, 1, 3)
    if heads == 1:
        x = x.contiguous()
    return x


def run_trtllm_decode(c: dict, workspace: torch.Tensor, max_seq_len: int, backend: str = "auto") -> torch.Tensor:
    geom, page = c["geom"], c["page"]
    hkv, d = geom["hkv"], geom["d"]
    sm_scale = 1.0 / math.sqrt(d)
    q = c["q"].contiguous()
    kw = {}
    if c["kv_dtype"] == "nvfp4":
        kv = (paged(c["k_store"], page, hkv, d // 2), paged(c["v_store"], page, hkv, d // 2))
        sf = (paged(c["k_sf"], page, hkv, d // 16).view(torch.float8_e4m3fn),
              paged(c["v_sf"], page, hkv, d // 16).view(torch.float8_e4m3fn))
        kw["kv_cache_sf"] = sf
        bmm1, bmm2 = c["ks"] * sm_scale, c["vs"]
    else:
        kv = (paged(c["k_store"], page, hkv, d), paged(c["v_store"], page, hkv, d))
        if c["kv_dtype"] == "fp8" and _sm()[0] == 10:
            q = q.to(torch.float8_e4m3fn)  # trtllm-gen: q fp8 when KV fp8 (XQA keeps bf16 q)
        bmm1, bmm2 = c["ks"] * sm_scale, c["vs"]
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q, kv_cache=kv, workspace_buffer=workspace, block_tables=c["bt"], seq_lens=c["sl"],
        max_seq_len=max_seq_len, bmm1_scale=bmm1, bmm2_scale=bmm2, window_left=geom["window"],
        out_dtype=torch.bfloat16, backend=backend, **kw)


def reference(c: dict, which: str) -> torch.Tensor:
    """fp32 attention for each request over its paged tokens; `which` = 'deq' (what the kernel should compute) or
    'orig' (the bf16 KV before quantization)."""
    geom, page = c["geom"], c["page"]
    hq, hkv, d, w = geom["hq"], geom["hkv"], geom["d"], geom["window"]
    K = c["k_deq"] if which == "deq" else c["k"]
    V = c["v_deq"] if which == "deq" else c["v"]
    outs = []
    for i, L in enumerate(c["sl"].tolist()):
        pages = c["bt"][i, : math.ceil(L / page)].long()
        idx = (pages[:, None] * page + torch.arange(page, device="cuda")[None]).reshape(-1)[:L]
        k = K[idx].float()            # L, hkv, d
        v = V[idx].float()
        if w >= 0:                    # SGLang/flashinfer window_left: attend to the last window_left + 1 tokens
            k, v = k[-(w + 1):], v[-(w + 1):]
        rep = hq // hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        s = torch.einsum("hd,lhd->hl", c["q"][i].float(), k) / math.sqrt(d)
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hl,lhd->hd", p, v))
    return torch.stack(outs)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


SEQS = [1, 17, 1000, 4096, 12345]


@pytest.fixture(scope="module")
def workspace():
    return torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")


@pytest.mark.parametrize("geom", list(GEOMS))
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize("kv", ["fp8", "nvfp4"])
def test_decode_matches_dequant_reference(geom, page, kv, workspace):
    c = build_case(GEOMS[geom], SEQS, page, kv, "ckpt_fp8")
    try:
        o = run_trtllm_decode(c, workspace, max_seq_len=131072)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"trtllm decode failed on sm{_sm()} geom={geom} page={page} kv={kv}: "
                    f"{type(e).__name__}: {str(e)[:600]}")
    torch.cuda.synchronize()
    ref = reference(c, "deq")
    e_deq = rel_err(o, ref)
    e_orig = rel_err(o, reference(c, "orig"))
    print(f"\n[kv4trt] sm{_sm()} {geom:12s} page{page:3d} {kv:6s} rel_err vs dequant-ref {e_deq:.4f}  "
          f"vs bf16-orig {e_orig:.4f}")
    assert torch.isfinite(o).all()
    # kernel vs its own (dequantized) input: only bf16 P/accumulation rounding remains
    assert e_deq < 0.02, e_deq
    # quantization cost at the attention output (random data is the pessimistic case for 4-bit)
    assert e_orig < (0.25 if kv == "nvfp4" else 0.06), e_orig


@pytest.mark.parametrize("geom", ["qwen38", "gemma_global"])
def test_nvfp4_unit_global_scale(geom, workspace):
    """Checkpoint without KV scales (SGLang falls back to global scale 1.0): still correct vs its dequant ref."""
    c = build_case(GEOMS[geom], [33, 5000], 64, "nvfp4", "unit")
    o = run_trtllm_decode(c, workspace, max_seq_len=131072)
    e = rel_err(o, reference(c, "deq"))
    print(f"\n[kv4trt] sm{_sm()} {geom} unit-scale nvfp4 rel_err vs dequant-ref {e:.4f}")
    assert e < 0.02, e


def test_nvfp4_long_context_104k(workspace):
    """The sold ~104k context, Qwen3.8 geometry, batch 2 (one long, one short): kernel matches the dequant ref."""
    c = build_case(GEOMS["qwen38"], [104_000, 777], 64, "nvfp4", "ckpt_fp8", seed=3)
    o = run_trtllm_decode(c, workspace, max_seq_len=262144)
    e = rel_err(o, reference(c, "deq"))
    print(f"\n[kv4trt] sm{_sm()} qwen38 104k nvfp4 rel_err vs dequant-ref {e:.4f}")
    assert e < 0.02, e


def reference_spec(c: dict, qlen: int) -> torch.Tensor:
    """Speculative verify / draft-extend reference: request i has seq_len L (prefix + qlen new tokens already in the
    cache); its query row j attends causally to the first L - qlen + j + 1 tokens (dequantized KV)."""
    geom, page = c["geom"], c["page"]
    hq, hkv, d = geom["hq"], geom["hkv"], geom["d"]
    q = c["q"].view(-1, qlen, hq, d)
    outs = []
    for i, L in enumerate(c["sl"].tolist()):
        pages = c["bt"][i, : math.ceil(L / page)].long()
        idx = (pages[:, None] * page + torch.arange(page, device="cuda")[None]).reshape(-1)[:L]
        k = c["k_deq"][idx].float().repeat_interleave(hq // hkv, dim=1)
        v = c["v_deq"][idx].float().repeat_interleave(hq // hkv, dim=1)
        rows = []
        for j in range(qlen):
            n = L - qlen + j + 1
            s = torch.einsum("hd,lhd->hl", q[i, j].float(), k[:n]) / math.sqrt(d)
            rows.append(torch.einsum("hl,lhd->hd", torch.softmax(s, -1), v[:n]))
        outs.append(torch.stack(rows))
    return torch.stack(outs).reshape(-1, hq, d)


def xqa_causal_mask(bs: int, qlen: int, device) -> torch.Tensor:
    """XQA speculative-decoding draft mask (flashinfer/xqa.py): uint16 [bs, qlen, ceil(qlen/32)*2], bit-packed rows,
    bit i of row j set = draft row j may attend draft token i. Causal chain: bits 0..j. Same builder as the hook
    (operations/patches/kv4trt_hooks.py `_numinous_xqa_causal_mask`)."""
    words = ((qlen + 31) // 32) * 2
    j = torch.arange(qlen, device=device)[:, None]
    i = torch.arange(words * 16, device=device)[None, :]
    bits = (i <= j).to(torch.int32).view(qlen, words, 16)
    vals = (bits << torch.arange(16, device=device, dtype=torch.int32)).sum(-1)
    return vals.to(torch.uint16)[None].expand(bs, qlen, words).contiguous()


@pytest.mark.parametrize("geom,qlen", [("qwen38", 4), ("qwen36", 5), ("qwen36", 4)])
def test_nvfp4_spec_verify_q_len(geom, qlen, workspace):
    """The kv4trt hook's target-verify / draft-extend-v2 call: q_len_per_req > 1 over native NVFP4 KV (XQA on SM120),
    exactly as trtllm_mha_backend._run_fixed_q_len_decode issues it (bf16 q, [bs*qlen, hq, d])."""
    seqs = [9, 300, 4100, 20000]
    c = build_case(GEOMS[geom], seqs, 64, "nvfp4", "ckpt_fp8", seed=5)
    hq, d = GEOMS[geom]["hq"], GEOMS[geom]["d"]
    g = torch.Generator(device="cuda").manual_seed(11)
    c["q"] = torch.randn(len(seqs) * qlen, hq, d, device="cuda", generator=g).bfloat16()
    hkv = GEOMS[geom]["hkv"]
    kv = (paged(c["k_store"], 64, hkv, d // 2), paged(c["v_store"], 64, hkv, d // 2))
    sf = (paged(c["k_sf"], 64, hkv, d // 16).view(torch.float8_e4m3fn),
          paged(c["v_sf"], 64, hkv, d // 16).view(torch.float8_e4m3fn))
    o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=c["q"], kv_cache=kv, workspace_buffer=workspace, block_tables=c["bt"], seq_lens=c["sl"],
        max_seq_len=131072, bmm1_scale=c["ks"] / math.sqrt(d), bmm2_scale=c["vs"], window_left=-1,
        out_dtype=torch.bfloat16, kv_cache_sf=sf, q_len_per_req=qlen,
        mask=xqa_causal_mask(len(seqs), qlen, "cuda"))
    e = rel_err(o, reference_spec(c, qlen))
    print(f"\n[kv4trt] sm{_sm()} {geom} spec q_len {qlen} nvfp4 rel_err vs causal dequant-ref {e:.4f}")
    assert e < 0.02, e
