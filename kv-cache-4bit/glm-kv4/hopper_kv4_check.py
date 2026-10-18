"""Kernel-level check of the GLM-5.3 4-bit KV path on Hopper (rtxsys, 2026-10-05; operations/kernels/glm_kv4_sm90.py).

Runs on one H200 (or H100) with the v0.5.20 image. No model, no server. The three modules must be importable as
sglang.srt.layers.attention.dsa.{glm_kv4, glm_kv4_engine, glm_kv4_sm90}; the Modal wrapper
(operations/modal/hopper_kv4_check.py) copies them into the image's package.

Correctness:
  1. The kernels against the torch reference on GLM shapes (bit-exact):
     - quant_write from bf16;
     - dequant_rows to bf16;
     - dequant_seq to bf16, with its slot -> row map (NEW output dtype);
     - dequant_rows_dedup to bf16, compact rows plus the remapped table (NEW output dtype).
  2. glm_kv4_sm90.forward_flashmla_kv with the REAL stock SM90 bf16 sparse kernel (DeepseekSparseAttnBackend.
     _forward_flashmla_sparse -> flash_mla_sparse_fwd), against the fp32 sparse-MLA reference over the same FP4 pool:
     - decode (plain gather);
     - verify (dedup);
     - chunked eager;
     - the prefill sequence path;
     - -1 rows.
  3. The fused Triton kernel (glm_kv4.sparse_mla, NUMINOUS_GLM_KV4_ATTN=triton) on sm_90 against the same reference.
Timing (CUDA events, per layer call, GLM-5.3: 64 heads, top-2048, 576/512):
  - decode, 48 tokens (8 requests x 6 verify tokens):
    - stock FP8 flashmla_kv (flash_mla_with_kvcache on the 656-B V32 row);
    - stock bf16 flashmla_sparse on a bf16 pool;
    - KV4 scratch, plain and dedup;
    - KV4 Triton.
  - prefill: a 4,096-token chunk of 2 requests over ~47K contexts each:
    - stock FP8 flashmla_kv (the decode kernel per token, as v0.5.20 runs flashmla_kv prefill);
    - KV4 sequence path.

    python3 hopper_kv4_check.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import types

import torch

H, K, D, DV = 64, 2048, 576, 512
SCALE = 192 ** -0.5          # GLM-5.3: qk_nope 128 + rope 64


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def timeit(fn, reps: int = 30, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1e3  # us


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_extend_or_draft_extend_or_mixed(self):
        return self.name in ("EXTEND", "MIXED", "DRAFT_EXTEND", "DRAFT_EXTEND_V2")

    def is_mixed(self):
        return self.name == "MIXED"

    def is_target_verify(self):
        return self.name == "TARGET_VERIFY"

    def is_decode_or_idle(self):
        return self.name in ("DECODE", "IDLE")


def backend_shim(rows4: torch.Tensor, n_slots: int):
    """A DeepseekSparseAttnBackend stand-in with the REAL stock _forward_flashmla_sparse bound to it."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    pool = types.SimpleNamespace(kv_buffer=[rows4.view(n_slots, 1, 352)], start_layer=0)
    be = types.SimpleNamespace(token_to_kv_pool=pool, device="cuda", dsa_index_topk=K, num_q_heads=H,
                               device_sm_major=torch.cuda.get_device_capability()[0], _sink_pad_cache={})
    be._forward_flashmla_sparse = types.MethodType(DeepseekSparseAttnBackend._forward_flashmla_sparse, be)
    return be


def ref_mla(q, rows4, table, kv4):
    """fp32 reference over the FP4 pool, batched over tokens in chunks (sparse_mla_ref loops per token)."""
    kv = kv4.dequantize_rows_ref(rows4, torch.float32)
    out = torch.zeros(q.shape[0], q.shape[1], DV, device=q.device)
    for i in range(q.shape[0]):
        idx = table[i].long()
        idx = idx[idx >= 0]
        if idx.numel() == 0:
            continue
        k = kv[idx]
        p = torch.softmax(q[i].float() @ k.T * SCALE, dim=-1)
        out[i] = p @ k[:, :DV]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--profile", action="store_true", help="per-kernel GPU time + CUDA-graph replay per path")
    ap.add_argument("--splitk", action="store_true", help="dsa_sparse_splitk merge vs the real unsplit kernel")
    a = ap.parse_args()
    if a.profile:
        return profile_main(a.json)
    if a.splitk:
        return splitk_main(a.json)
    os.environ.setdefault("NUMINOUS_GLM_KV4", "1")
    from sglang.srt.layers.attention.dsa import glm_kv4 as kv4
    from sglang.srt.layers.attention.dsa import glm_kv4_sm90 as h

    dev = torch.device("cuda")
    torch.manual_seed(0)
    res: dict = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())}
    n_slots = 120_000
    chan = torch.exp(torch.randn(512, device=dev) * 0.75)            # heavy latent channels, like real MLA latents
    nope = (torch.randn(n_slots, 512, device=dev) * chan * 0.5).to(torch.bfloat16)
    rope = (torch.randn(n_slots, 64, device=dev) * 0.5).to(torch.bfloat16)

    # ---- 1. kernels, bit-exact
    rows4 = torch.zeros(n_slots, 352, dtype=torch.uint8, device=dev)
    kv4.quant_write(rows4, torch.arange(n_slots, device=dev), nope, rope)
    ref_rows = kv4.quantize_rows_ref(nope, rope)
    res["quant_write_bit_exact"] = bool(torch.equal(rows4, ref_rows))
    deq_ref = kv4.dequantize_rows_ref(rows4, torch.bfloat16)
    idx = torch.randint(-1, n_slots, (5000,), dtype=torch.int32, device=dev)
    dst = torch.zeros(5000, D, dtype=torch.bfloat16, device=dev)
    kv4.dequant_rows(rows4, idx, dst)
    ok = idx >= 0
    res["dequant_rows_bf16_exact"] = bool(torch.equal(dst[ok], deq_ref[idx[ok].long()]))
    # dequant_seq to bf16
    lens = [30_000, 17_000]
    pt = torch.full((2, 32_768), -1, dtype=torch.int32, device=dev)
    perm = torch.randperm(n_slots, device=dev)
    pt[0, :lens[0]] = perm[:lens[0]].int()
    pt[1, :lens[1]] = perm[50_000:50_000 + lens[1]].int()
    seq = torch.zeros(sum(lens), D, dtype=torch.bfloat16, device=dev)
    s2p = torch.full((n_slots,), -1, dtype=torch.int32, device=dev)
    bases = torch.tensor([0, lens[0]], dtype=torch.int32, device=dev)
    kv4.dequant_seq(rows4, pt, torch.tensor(lens, dtype=torch.int32, device=dev), bases, max(lens), seq, s2p)
    want = torch.cat([deq_ref[pt[0, :lens[0]].long()], deq_ref[pt[1, :lens[1]].long()]])
    res["dequant_seq_bf16_exact"] = bool(torch.equal(seq, want))
    res["dequant_seq_slot_map_ok"] = bool(torch.equal(s2p[pt[1, :lens[1]].long()],
                                                      torch.arange(lens[0], sum(lens), device=dev, dtype=torch.int32)))
    # dedup to bf16: verify-like table (8 requests x 6 tokens, a request's tokens share 90% of their picks)
    tab = torch.randint(0, n_slots, (48, K), dtype=torch.int32, device=dev)
    for r in range(8):
        base = tab[r * 6].clone()
        for j in range(1, 6):
            m = torch.rand(K, device=dev) < 0.9
            tab[r * 6 + j] = torch.where(m, base, tab[r * 6 + j])
    tab[0, 1500:] = -1
    comp = torch.zeros(48 * K, D, dtype=torch.bfloat16, device=dev)
    out_t = torch.empty_like(tab)
    stamp = torch.full((n_slots,), kv4.STAMP_CLEAN, dtype=torch.int32, device=dev)
    n_own = kv4.dequant_rows_dedup(rows4, tab, stamp, comp, out_t)
    v = tab >= 0
    res["dedup_bf16_rows_exact"] = bool(torch.equal(comp[out_t[v].long()], deq_ref[tab[v].long()]))
    res["dedup_distinct_rows"] = int(n_own.item())
    res["dedup_stamps_clean"] = bool((stamp == kv4.STAMP_CLEAN).all())
    # V32 decode scratch: FP4 rows -> FlashMLA's 656-B row, against the stock quantizer on the exact values
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
    vidx = torch.arange(0, 4096, dtype=torch.int32, device=dev)
    v32 = torch.zeros(4096, h.V32_ROW, dtype=torch.uint8, device=dev)
    h.kv4_to_v32(rows4, vidx, v32)
    sn, sr = quantize_k_cache_separate(deq_ref[:4096, :512].contiguous(), deq_ref[:4096, 512:].contiguous())
    stock = torch.cat([sn.view(4096, -1), sr.view(4096, -1)], dim=-1)
    res["v32_row_bytes_stock"] = int(stock.shape[-1])
    res["v32_bit_exact_vs_stock_quantizer"] = bool(torch.equal(v32, stock))
    res["v32_mismatch_bytes"] = int((v32 != stock).sum().item()) if stock.shape == v32.shape else -1

    # ---- 2. the SM90 read path with the stock bf16 sparse kernel
    be = backend_shim(rows4, n_slots)
    layer = types.SimpleNamespace(layer_id=0, tp_q_head_num=H, head_dim=D)
    meta = types.SimpleNamespace(dsa_cache_seqlens_int32=None, page_table_1=None, cache_seqlens_int32=None)
    cases = {}

    def run_case(name, q, table, mode, fb=None, md=meta, env=None):
        for k_, v_ in (env or {}).items():
            os.environ[k_] = v_
        be._numinous_kv4_state = None
        be._numinous_kv4_fb = fb or types.SimpleNamespace(forward_mode=_Mode(mode), seq_lens_cpu=None,
                                                          batch_size=q.shape[0])
        out = h.forward_flashmla_kv(be, q, table, SCALE, DV, layer, md)
        torch.cuda.synchronize()
        ref = ref_mla(q, rows4, table, kv4)
        cases[name] = {"rel_err": rel(out, ref), "max_abs": float((out.float() - ref).abs().max()),
                       "nan": bool(torch.isnan(out.float()).any()), "calls": dict(be._numinous_kv4_state.calls)}
        return out

    q48 = (torch.randn(48, H, D, device=dev) * 0.3).to(torch.bfloat16)
    os.environ["NUMINOUS_GLM_KV4_ATTN"] = "scratch"
    run_case("decode_plain_48", q48, tab, "DECODE", env={"NUMINOUS_GLM_KV4_DEDUP": "0", "NUMINOUS_GLM_KV4_SEL_ROWS": "256"})
    run_case("verify_dedup_48", q48, tab, "TARGET_VERIFY", env={"NUMINOUS_GLM_KV4_DEDUP": "1"})
    run_case("eager_chunked_48", q48, tab, "DECODE", env={"NUMINOUS_GLM_KV4_DEDUP": "0", "NUMINOUS_GLM_KV4_SEL_ROWS": "7"})
    os.environ["NUMINOUS_GLM_KV4_SEL_ROWS"] = "256"
    # prefill sequence path: 512 query tokens of the two requests, each picking 2048 positions of its own sequence
    tq = 512
    ptab = torch.full((tq, K), -1, dtype=torch.int32, device=dev)
    for i in range(tq):
        b = 0 if i < 300 else 1
        pos = torch.randperm(lens[b], device=dev)[:K]
        ptab[i] = pt[b, pos]
    qp = (torch.randn(tq, H, D, device=dev) * 0.3).to(torch.bfloat16)
    fb = types.SimpleNamespace(forward_mode=_Mode("EXTEND"), seq_lens_cpu=torch.tensor(lens), batch_size=2)
    mdp = types.SimpleNamespace(dsa_cache_seqlens_int32=None, page_table_1=pt,
                                cache_seqlens_int32=torch.tensor(lens, dtype=torch.int32, device=dev))
    run_case("prefill_sequence_512", qp, ptab, "EXTEND", fb=fb, md=mdp)
    os.environ["NUMINOUS_GLM_KV4_ATTN"] = "triton"
    try:
        run_case("triton_fused_48", q48, tab, "DECODE")
    except Exception as e:  # noqa: BLE001
        cases["triton_fused_48"] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
    os.environ["NUMINOUS_GLM_KV4_ATTN"] = "scratch"
    from sgl_kernel.flash_mla import get_mla_metadata as _gmm
    os.environ["NUMINOUS_GLM_KV4_SM90_DECODE"] = "v32"
    be.flashmla_kv_num_q_heads = H
    m48, sp48 = _gmm(cache_seqlens=torch.ones(48, dtype=torch.int32, device=dev), num_q_tokens_per_head_k=H,
                     num_heads_k=1, num_heads_q=H, is_fp8_kvcache=True, topk=K)
    mdv = types.SimpleNamespace(dsa_cache_seqlens_int32=torch.ones(48, dtype=torch.int32, device=dev),
                                page_table_1=None, cache_seqlens_int32=None,
                                flashmla_metadata=types.SimpleNamespace(flashmla_metadata=m48, num_splits=sp48))
    try:
        run_case("v32_decode_48", q48, tab, "TARGET_VERIFY", md=mdv)
        run_case("v32_eager_chunked_48", q48, tab, "DECODE", md=mdv, env={"NUMINOUS_GLM_KV4_SEL_ROWS": "7"})
    except Exception as e:  # noqa: BLE001
        cases["v32_decode_48"] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
    os.environ["NUMINOUS_GLM_KV4_SEL_ROWS"] = "256"
    os.environ["NUMINOUS_GLM_KV4_SM90_DECODE"] = "bf16"
    res["read_path"] = cases

    # ---- 3. timing per layer call
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
    t: dict = {}
    nope_u8, rope_u8 = quantize_k_cache_separate(nope, rope)
    v32 = torch.cat([nope_u8.view(n_slots, -1), rope_u8.view(n_slots, -1)], dim=-1).contiguous()
    res["v32_row_bytes"] = int(v32.shape[-1])
    n_pages = n_slots // 64
    pool8 = v32[: n_pages * 64].view(n_pages, 64, 1, v32.shape[-1])
    tab8 = tab.clamp(max=n_pages * 64 - 1)

    def fp8_decode(tbl, n):
        seql = torch.ones(n, dtype=torch.int32, device=dev)
        meta8, splits = get_mla_metadata(cache_seqlens=seql, num_q_tokens_per_head_k=H, num_heads_k=1, num_heads_q=H,
                                         is_fp8_kvcache=True, topk=K)
        qq = (torch.randn(n, 1, H, D, device=dev) * 0.3).to(torch.bfloat16)
        bt = torch.empty((n, 0), dtype=torch.int32, device=dev)
        return lambda: flash_mla_with_kvcache(q=qq, k_cache=pool8, block_table=bt, cache_seqlens=seql, head_dim_v=DV,
                                              tile_scheduler_metadata=meta8, num_splits=splits, softmax_scale=SCALE,
                                              indices=tbl.unsqueeze(1), is_fp8_kvcache=True)

    try:
        t["decode48_fp8_flashmla_kv_us"] = timeit(fp8_decode(tab8, 48))
    except Exception as e:  # noqa: BLE001
        t["decode48_fp8_flashmla_kv_us"] = f"{type(e).__name__}: {str(e)[:300]}"
    bf16_pool = deq_ref.view(n_slots, 1, D)
    t["decode48_bf16_flashmla_sparse_us"] = timeit(
        lambda: flash_mla_sparse_fwd(q=q48, kv=bf16_pool, indices=tab.unsqueeze(1), sm_scale=SCALE, d_v=DV))

    def kv4_call(mode, dedup, mode_name="DECODE"):
        os.environ["NUMINOUS_GLM_KV4_ATTN"] = mode
        os.environ["NUMINOUS_GLM_KV4_DEDUP"] = "1" if dedup else "0"
        be._numinous_kv4_state = None
        be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode(mode_name), seq_lens_cpu=None, batch_size=48)
        h.forward_flashmla_kv(be, q48, tab, SCALE, DV, layer, meta)       # allocate the state outside the timing
        return lambda: h.forward_flashmla_kv(be, q48, tab, SCALE, DV, layer, meta)

    t["decode48_kv4_scratch_plain_us"] = timeit(kv4_call("scratch", False))
    t["verify48_kv4_scratch_dedup_us"] = timeit(kv4_call("scratch", True, "TARGET_VERIFY"))
    try:
        t["decode48_kv4_triton_us"] = timeit(kv4_call("triton", False))
    except Exception as e:  # noqa: BLE001
        t["decode48_kv4_triton_us"] = f"{type(e).__name__}: {str(e)[:300]}"
    os.environ["NUMINOUS_GLM_KV4_ATTN"] = "scratch"
    os.environ["NUMINOUS_GLM_KV4_SM90_DECODE"] = "v32"
    try:
        be._numinous_kv4_state = None
        be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode("TARGET_VERIFY"), seq_lens_cpu=None,
                                                    batch_size=48)
        h.forward_flashmla_kv(be, q48, tab, SCALE, DV, layer, mdv)
        t["decode48_kv4_v32_us"] = timeit(lambda: h.forward_flashmla_kv(be, q48, tab, SCALE, DV, layer, mdv))
        sc = torch.zeros(48 * K, h.V32_ROW, dtype=torch.uint8, device=dev)
        t["decode48_kv4_to_v32_convert_only_us"] = timeit(lambda: h.kv4_to_v32(rows4, tab, sc))
    except Exception as e:  # noqa: BLE001
        t["decode48_kv4_v32_us"] = f"{type(e).__name__}: {str(e)[:300]}"
    os.environ["NUMINOUS_GLM_KV4_SM90_DECODE"] = "bf16"
    # sweeps that set the next arm's defaults (one GPU run answers them all): dedup forced on a 48-token verify call
    # (the 64-row default left it off above), the dequant / V32 kernels' tile shape, and the 96-token call (P2's graph
    # ceiling: cuda-graph bs 16 x 6 verify tokens)
    sweep: dict = {}
    try:
        h.DEDUP_MIN_ROWS = 1
        sweep["verify48_bf16_dedup_forced_us"] = timeit(kv4_call("scratch", True, "TARGET_VERIFY"))
        h.DEDUP_MIN_ROWS = 64
        dst_bf = torch.empty(48 * K, D, dtype=torch.bfloat16, device=dev)
        sc2 = torch.zeros(48 * K, h.V32_ROW, dtype=torch.uint8, device=dev)
        for br in (8, 16, 32):
            for nw in (2, 4, 8):
                sweep[f"dequant_rows_bf16_br{br}_nw{nw}_us"] = timeit(
                    lambda: kv4.dequant_rows(rows4, tab, dst_bf, block_r=br, num_warps=nw))
                sweep[f"kv4_to_v32_br{br}_nw{nw}_us"] = timeit(
                    lambda: h.kv4_to_v32(rows4, tab, sc2, block_r=br, num_warps=nw))
        tab96, q96 = torch.cat([tab, tab], 0), torch.cat([q48, q48], 0)
        os.environ["NUMINOUS_GLM_KV4_DEDUP"] = "0"
        be._numinous_kv4_state = None
        be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode("DECODE"), seq_lens_cpu=None, batch_size=96)
        md96 = types.SimpleNamespace(dsa_cache_seqlens_int32=None, page_table_1=None, cache_seqlens_int32=None)
        h.forward_flashmla_kv(be, q96, tab96, SCALE, DV, layer, md96)
        sweep["decode96_kv4_bf16_us"] = timeit(lambda: h.forward_flashmla_kv(be, q96, tab96, SCALE, DV, layer, md96))
        sweep["decode96_fp8_flashmla_kv_us"] = timeit(fp8_decode(tab96.clamp(max=n_pages * 64 - 1), 96))
        os.environ["NUMINOUS_GLM_KV4_SM90_DECODE"] = "v32"
        m96, sp96 = _gmm(cache_seqlens=torch.ones(96, dtype=torch.int32, device=dev), num_q_tokens_per_head_k=H,
                         num_heads_k=1, num_heads_q=H, is_fp8_kvcache=True, topk=K)
        md96v = types.SimpleNamespace(dsa_cache_seqlens_int32=torch.ones(96, dtype=torch.int32, device=dev),
                                      page_table_1=None, cache_seqlens_int32=None,
                                      flashmla_metadata=types.SimpleNamespace(flashmla_metadata=m96, num_splits=sp96))
        be._numinous_kv4_state = None
        h.forward_flashmla_kv(be, q96, tab96, SCALE, DV, layer, md96v)
        sweep["decode96_kv4_v32_us"] = timeit(lambda: h.forward_flashmla_kv(be, q96, tab96, SCALE, DV, layer, md96v))
    except Exception as e:  # noqa: BLE001
        sweep["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    os.environ["NUMINOUS_GLM_KV4_SM90_DECODE"] = "bf16"
    h.DEDUP_MIN_ROWS = 64
    res["sweep"] = sweep
    # prefill: 4,096 query tokens over the two ~47K-token sequences (re-using 30K + 17K here: scale by context)
    tq = 4096
    ptab = torch.full((tq, K), -1, dtype=torch.int32, device=dev)
    pos0 = torch.stack([torch.randperm(lens[0], device=dev)[:K] for _ in range(64)])
    pos1 = torch.stack([torch.randperm(lens[1], device=dev)[:K] for _ in range(64)])
    for i in range(tq):
        ptab[i] = pt[0, pos0[i % 64]] if i < 2400 else pt[1, pos1[i % 64]]
    qp = (torch.randn(tq, H, D, device=dev) * 0.3).to(torch.bfloat16)
    be._numinous_kv4_state = None
    be._numinous_kv4_fb = fb
    h.forward_flashmla_kv(be, qp, ptab, SCALE, DV, layer, mdp)
    t["prefill4096_kv4_sequence_us"] = timeit(lambda: h.forward_flashmla_kv(be, qp, ptab, SCALE, DV, layer, mdp),
                                              reps=10)
    try:
        t["prefill4096_fp8_flashmla_kv_us"] = timeit(fp8_decode(ptab.clamp(max=n_pages * 64 - 1), tq), reps=10)
    except Exception as e:  # noqa: BLE001
        t["prefill4096_fp8_flashmla_kv_us"] = f"{type(e).__name__}: {str(e)[:300]}"
    t["prefill4096_bf16_flashmla_sparse_us"] = timeit(
        lambda: flash_mla_sparse_fwd(q=qp, kv=bf16_pool, indices=ptab.unsqueeze(1), sm_scale=SCALE, d_v=DV), reps=10)
    res["timing"] = t
    res["bytes_per_token_layer"] = {"fp8_v32": int(v32.shape[-1]), "kv4": 352, "bf16": 1152}
    kern_ok = all(res[k_] for k_ in ("quant_write_bit_exact", "dequant_rows_bf16_exact", "dequant_seq_bf16_exact",
                                     "dequant_seq_slot_map_ok", "dedup_bf16_rows_exact", "dedup_stamps_clean"))
    path_ok = all(("rel_err" in c and c["rel_err"] < 0.02 and not c["nan"]) for n, c in cases.items()
                  if n != "triton_fused_48" and not n.startswith("v32"))
    res["PASS"] = bool(kern_ok and path_ok)
    v32c = cases.get("v32_decode_48", {})
    res["PASS_v32"] = bool(res.get("v32_bit_exact_vs_stock_quantizer") and "rel_err" in v32c
                           and v32c["rel_err"] < 0.03 and not v32c["nan"])
    s = json.dumps(res, indent=1)
    print(s)
    if a.json:
        open(a.json, "w").write(s)


def profile_main(json_out: str) -> None:
    """Kernel attribution of one DSA decode attention call per path (coordinator 2026-10-05: KV4 decode 115.8 vs
    57.2 us, find why from the kernel). Per path and batch (48 = 8 requests x 6 verify tokens, 96 = P2's graph
    ceiling): (1) CUDA-graph replay time of the whole call (served decode replays graphs: no Python between kernels);
    (2) per-kernel GPU time from torch.profiler over eager calls; (3) the HBM floor of each kernel from its bytes."""
    os.environ.setdefault("NUMINOUS_GLM_KV4", "1")
    from sglang.srt.layers.attention.dsa import glm_kv4 as kv4
    from sglang.srt.layers.attention.dsa import glm_kv4_sm90 as h
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
    from torch.profiler import ProfilerActivity, profile

    dev = torch.device("cuda")
    torch.manual_seed(0)
    hbm = 4.8e12 if "H200" in torch.cuda.get_device_name() else 3.35e12
    n_slots = 120_000
    chan = torch.exp(torch.randn(512, device=dev) * 0.75)
    nope = (torch.randn(n_slots, 512, device=dev) * chan * 0.5).to(torch.bfloat16)
    rope = (torch.randn(n_slots, 64, device=dev) * 0.5).to(torch.bfloat16)
    rows4 = torch.zeros(n_slots, 352, dtype=torch.uint8, device=dev)
    kv4.quant_write(rows4, torch.arange(n_slots, device=dev), nope, rope)
    nu, ru = quantize_k_cache_separate(nope, rope)
    pool8 = torch.cat([nu.view(n_slots, -1), ru.view(n_slots, -1)], -1).contiguous().view(n_slots // 64, 64, 1, -1)
    bf16_pool = kv4.dequantize_rows_ref(rows4, torch.bfloat16).view(n_slots, 1, D)
    layer = types.SimpleNamespace(layer_id=0, tp_q_head_num=H, head_dim=D)
    out: dict = {"gpu": torch.cuda.get_device_name(), "hbm_bytes_per_s": hbm, "graph_us": {}, "kernels": {},
                 "floors_us": {}}

    def graph_us(fn, reps: int = 50) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            g.replay()
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / reps * 1e3

    def kernel_table(fn, iters: int = 20) -> list:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        rows = []
        for ev in prof.key_averages():
            dev_us = getattr(ev, "device_time_total", None)
            if dev_us is None:
                dev_us = getattr(ev, "cuda_time_total", 0)
            if not dev_us:
                continue
            rows.append({"kernel": ev.key[:120], "us_per_call": round(dev_us / iters, 2), "launches": ev.count})
        return sorted(rows, key=lambda r: -r["us_per_call"])[:12]

    for t in (48, 96):
        tab = torch.randint(0, n_slots, (t, K), dtype=torch.int32, device=dev)
        q = (torch.randn(t, H, D, device=dev) * 0.3).to(torch.bfloat16)
        seql = torch.ones(t, dtype=torch.int32, device=dev)
        meta, splits = get_mla_metadata(cache_seqlens=seql, num_q_tokens_per_head_k=H, num_heads_k=1, num_heads_q=H,
                                        is_fp8_kvcache=True, topk=K)
        qq = q.view(t, 1, H, D)
        bt = torch.empty((t, 0), dtype=torch.int32, device=dev)
        fp8 = lambda: flash_mla_with_kvcache(q=qq, k_cache=pool8, block_table=bt, cache_seqlens=seql,  # noqa: E731
                                             head_dim_v=DV, tile_scheduler_metadata=meta, num_splits=splits,
                                             softmax_scale=SCALE, indices=tab.unsqueeze(1), is_fp8_kvcache=True)
        bf16k = lambda: flash_mla_sparse_fwd(q=q, kv=bf16_pool, indices=tab.unsqueeze(1), sm_scale=SCALE,  # noqa: E731
                                             d_v=DV)
        md = types.SimpleNamespace(dsa_cache_seqlens_int32=seql, page_table_1=None, cache_seqlens_int32=None,
                                   flashmla_metadata=types.SimpleNamespace(flashmla_metadata=meta, num_splits=splits))
        paths = {"fp8_stock": fp8, "bf16_kernel_on_bf16_pool": bf16k}
        for dec in ("bf16", "v32"):
            os.environ["NUMINOUS_GLM_KV4_SM90_DECODE"] = dec
            os.environ["NUMINOUS_GLM_KV4_DEDUP"] = "0"
            os.environ["NUMINOUS_GLM_KV4_ATTN"] = "scratch"
            be = backend_shim(rows4, n_slots)
            be.flashmla_kv_num_q_heads = H
            be._numinous_kv4_fb = types.SimpleNamespace(forward_mode=_Mode("DECODE"), seq_lens_cpu=None, batch_size=t)
            h.reserve_graph_scratch(be, t)   # what init_cuda_graph_state does before capture
            paths[f"kv4_{dec}"] = (lambda be=be: h.forward_flashmla_kv(be, q, tab, SCALE, DV, layer, md))
        sc_bf = torch.empty(t * K, D, dtype=torch.bfloat16, device=dev)
        sc_v32 = torch.zeros(t * K, h.V32_ROW, dtype=torch.uint8, device=dev)
        paths["dequant_rows_bf16_only"] = lambda: kv4.dequant_rows(rows4, tab, sc_bf)
        paths["kv4_to_v32_only"] = lambda: h.kv4_to_v32(rows4, tab, sc_v32)
        for name, fn in paths.items():
            try:
                out["graph_us"][f"{name}_{t}"] = round(graph_us(fn), 2)
            except Exception as e:  # noqa: BLE001
                out["graph_us"][f"{name}_{t}"] = f"{type(e).__name__}: {str(e)[:300]}"
            try:
                out["kernels"][f"{name}_{t}"] = kernel_table(fn)
            except Exception as e:  # noqa: BLE001
                out["kernels"][f"{name}_{t}"] = f"{type(e).__name__}: {str(e)[:300]}"
        rows_sel = t * K
        out["floors_us"][str(t)] = {
            "fp8_kernel_reads_656B": round(rows_sel * 656 / hbm * 1e6, 1),
            "bf16_kernel_reads_1152B": round(rows_sel * 1152 / hbm * 1e6, 1),
            "dequant_352B_in_1152B_out": round(rows_sel * (352 + 1152) / hbm * 1e6, 1),
            "v32_convert_352B_in_656B_out": round(rows_sel * (352 + 656) / hbm * 1e6, 1),
            "fused_fp4_kernel_reads_352B": round(rows_sel * 352 / hbm * 1e6, 1),
        }
    s = json.dumps(out, indent=1)
    print(s)
    if json_out:
        open(json_out, "w").write(s)


def splitk_main(json_out: str) -> None:
    """amdpro's dsa_sparse_splitk (installed as sglang.srt.layers.attention.dsa.numinous_sparse_splitk) against the
    REAL unsplit flash_mla_sparse_fwd on H200 (rtxsys 2026-10-05, after arm P5's sanity failure). It records:
    - the kernel's lse convention: an all -1 row (FlashMLA phase1.cuh:448-449 says +INF), and natural vs base-2 units
      against a torch reference;
    - per split factor S and per row valid count, NaN count and rel err of (a) the shipped merge and (b) a natural-log
      merge with an isfinite guard."""
    import math
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    from sglang.srt.layers.attention.dsa import numinous_sparse_splitk as nsk

    dev = torch.device("cuda")
    torch.manual_seed(0)
    n_slots, t, k = 50_000, 32, 2048
    kv = (torch.randn(n_slots, 1, D, device=dev) * 0.5).to(torch.bfloat16)
    q = (torch.randn(t, H, D, device=dev) * 0.3).to(torch.bfloat16)
    counts = [40, 300, 1100, 2048] * (t // 4)
    idx = torch.full((t, 1, k), -1, dtype=torch.int32, device=dev)
    for i, c in enumerate(counts):
        idx[i, 0, :c] = torch.randperm(n_slots, device=dev)[:c].int()
    res: dict = {"gpu": torch.cuda.get_device_name()}
    # peaky rows (real attention concentrates on a few keys): in every 4th row, the first 8 selected keys point along
    # that row's head-0 query, so one split's lse sits far above the others. Random keys keep every split's lse near
    # equal, which hides a wrong merge base (amdpro's GPU check, 0942efe0, used random data)
    for i in range(0, t, 4):
        keys = idx[i, 0, :8].long()
        kv[keys, 0] = (q[i, 0].float() / q[i, 0].float().norm() * 6.0).to(torch.bfloat16)
    o_ref, ml_ref, lse_ref = flash_mla_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=SCALE, d_v=DV)
    # convention probe: one empty row, and units of a full row against torch
    e_idx = torch.full((1, 1, k), -1, dtype=torch.int32, device=dev)
    _, ml_e, lse_e = flash_mla_sparse_fwd(q=q[:1], kv=kv, indices=e_idx, sm_scale=SCALE, d_v=DV)
    res["empty_row_lse"] = float(lse_e.flatten()[0])
    res["empty_row_max_logits"] = float(ml_e.flatten()[0])
    full = 3   # counts[3] == 2048
    sel = kv[idx[full, 0].long(), 0].float()
    logits = (q[full].float() @ sel.T) * SCALE
    res["lse_kernel_row3_head0"] = float(lse_ref[full, 0])
    res["lse_torch_natural"] = float(torch.logsumexp(logits[0], 0))
    res["lse_torch_base2"] = float(torch.logsumexp(logits[0] * math.log2(math.e), 0) / math.log(2))

    def merge_natural(o_s, lse_s, tokens, sp):
        lse = lse_s.view(tokens, sp, -1).float()
        valid = torch.isfinite(lse) & (lse < 1e29)
        m = torch.where(valid, lse, torch.full_like(lse, -1e30)).amax(dim=1, keepdim=True)
        w = torch.where(valid, torch.exp(lse - m), torch.zeros_like(lse))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-30)
        o = o_s.view(tokens, sp, lse.shape[-1], -1).float()
        return (o * w.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)

    rows = {}
    for sp in (2, 4, 8, 16):
        for name, fn in (("shipped_merge", None), ("natural_isfinite_merge", merge_natural)):
            o = nsk.sparse_fwd_splitk(flash_mla_sparse_fwd, q, kv, idx, SCALE, DV, None, sp, merge_fn=fn)
            by = {}
            peaky = [i for i in range(0, t, 4)]
            oo, rr = o[peaky].float(), o_ref[peaky].float()
            by["peaky_rows"] = {"nan": int(torch.isnan(oo).sum()), "rel_err": round(rel(oo.nan_to_num(), rr), 5)}
            for c in sorted(set(counts)):
                r = [i for i, x in enumerate(counts) if x == c and i % 4]
                oo, rr = o[r].float(), o_ref[r].float()
                by[str(c)] = {"nan": int(torch.isnan(oo).sum()), "rel_err": round(rel(oo.nan_to_num(), rr), 5)}
            rows[f"S{sp}_{name}"] = by
    res["splitk"] = rows
    s = json.dumps(res, indent=1)
    print(s)
    if json_out:
        open(json_out, "w").write(s)


if __name__ == "__main__":
    main()
