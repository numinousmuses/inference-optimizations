"""8-GPU DeepGEMM MegaMoE benchmark at DeepSeek V4 Pro shapes, driven the way SGLang v0.5.20 drives it.

What it runs (per rank, EP8, torchrun --nproc-per-node 8):
  SGLang's `_run_mega_routed` (python/sglang/srt/layers/moe/mega_moe.py @ v0.5.20) minus the router:
  pre-dispatch (SGLang's JIT FP8 kernel for `fp8xfp4`, DeepGEMM's `mega_moe_pre_dispatch` for W4A4
  `mxf4xmxf4`) into the symmetric buffer, then `deep_gemm.fp8_fp4_mega_moe(recipe=(1,1,32), swiglu,
  activation_clamp=10, fast_math)` inside `_configure_mega_moe_deep_gemm_num_sms` (SMs minus 2 reserved).
  Shapes: 384 experts (48 per rank), hidden 7168, intermediate 3072, top-6, MXFP4 weights (int8-packed E2M1 +
  E8M0 per 32), no fused shared expert (SGLang runs it separately on an alt stream).

Timing: CUDA-graph replay (a graph holding `--graph-steps` calls, replayed `--replays` times, events around the
replays), all ranks released together after a barrier. Reported per call: pre-dispatch alone, pre-dispatch +
MegaMoE, and their difference (the MegaMoE kernel). Each rank's time includes waiting for the slowest rank,
exactly as in serving (the kernel has NVLink barriers).

Correctness: a torch reference on the gathered tokens (dense dequantized experts, activation quantization emulated
per 32 with UE8M0 scales, top-k weight applied before the L1-output quantization, SwiGLU clamp), summed over
ranks with an all-reduce; `calc_diff` (1 - cosine-like similarity, DeepGEMM's metric) against the graph-replayed
output of this rank.

Floor: bytes of the local experts actually touched (weights + E8M0 scales) over HBM bandwidth, max over ranks
(the kernel ends with a grid-wide barrier). `--hbm-tbps` default 8.0 (B300/B200 HBM3e spec; derived, not measured).

Import order: flashinfer first (it sets TRITON_PTXAS_BLACKWELL_PATH exactly as the engine does).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import flashinfer  # noqa: F401  (engine import order)
import torch
import torch.distributed as dist

HIDDEN, INTER, NUM_EXPERTS, TOPK = 7168, 3072, 384, 6
SWIGLU_LIMIT = 10.0
DEFAULT_TOKENS = (1, 4, 8, 16, 32, 64, 128, 4096)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default=",".join(map(str, DEFAULT_TOKENS)))
    p.add_argument("--mma", default="fp8xfp4", help="comma list of fp8xfp4,mxf4xmxf4")
    p.add_argument("--max-tokens-per-rank", type=int, default=4096,
                   help="SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK (our Pro lane: 4096)")
    p.add_argument("--reserved-sms", type=int, default=2, help="SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS")
    p.add_argument("--graph-steps", type=int, default=10)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--hbm-tbps", type=float, default=8.0)
    p.add_argument("--routing", default="uniform", choices=("uniform", "balanced"))
    p.add_argument("--out", default="", help="rank 0 writes a JSON result here")
    p.add_argument("--label", default="")
    p.add_argument("--configs", default="default",
                   help="';'-separated host-config overrides for a deep_gemm built with the env-override research "
                        "patch, e.g. 'default;BLOCK_K=256;BLOCK_M=32,MAX_STAGES=8' (keys -> DG_MEGA_MOE_<KEY>)")
    p.add_argument("--tokens-by-rank", default="",
                   help="comma list of 8 per-rank token counts (a mixed step, e.g. one rank prefilling); each rank's "
                        "time includes waiting for the others inside the kernel")
    p.add_argument("--pad-policy", default="none", choices=("none", "mean", "max"),
                   help="mixed steps: run the kernel with y padded to max(T, ceil(sum/ranks)) ('mean') or the max "
                        "rank's T ('max') rows, padded routes set to -1, so every rank's host heuristic sees the "
                        "global per-expert load (numinous_megamoe_balance)")
    p.add_argument("--dg-path", default="", help="directory prepended to sys.path to benchmark another deep_gemm")
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------------------ quant helpers
def fp4_decode(codes: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device)
    mag = lut[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def dequant_fp4_packed(packed: torch.Tensor, sf: torch.Tensor, gran: int = 32) -> torch.Tensor:
    """packed int8 [n, k/2] (low nibble = even element) + fp32 scales [n, k/gran] -> fp32 [n, k]."""
    lo, hi = packed & 0x0F, (packed >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return fp4_decode(codes) * sf.repeat_interleave(gran, dim=1)


def ue8m0_ceil(x: torch.Tensor) -> torch.Tensor:
    bits = x.float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def fake_quant(x: torch.Tensor, kind: str, gran: int = 32) -> torch.Tensor:
    """Per-`gran` UE8M0-scaled FP8 E4M3 or FP4 E2M1 round trip (what the kernel's activations carry)."""
    m, n = x.shape
    v = x.float().view(m, n // gran, gran)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    if kind == "fp8":
        sf = ue8m0_ceil(amax / 448.0)
        q = (v / sf).to(torch.float8_e4m3fn).float()
    else:
        sf = ue8m0_ceil(amax / 6.0)
        s = v / sf
        a = s.abs()
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
        idx = torch.bucketize(a, (grid[1:] + grid[:-1]) / 2)
        q = torch.sign(s) * grid[idx]
    return (q * sf).view(m, n)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    den = (x * x + y * y).sum()
    return 0.0 if den == 0 else float(1 - 2 * (x * y).sum() / den)


# ------------------------------------------------------------------------------------------------ setup
def init():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(dist.get_world_size())))


def make_weights(num_local: int, seed: int):
    """Random MXFP4 experts: int8-packed E2M1 codes and power-of-two fp32 scales per 32 (checkpoint format)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randint(-128, 128, (num_local, 2 * INTER, HIDDEN // 2), dtype=torch.int8, device="cuda", generator=g)
    w2 = torch.randint(-128, 128, (num_local, HIDDEN, INTER // 2), dtype=torch.int8, device="cuda", generator=g)
    s13 = torch.pow(2.0, torch.randint(-9, -6, (num_local, 2 * INTER, HIDDEN // 32), device="cuda", generator=g).float())
    s2 = torch.pow(2.0, torch.randint(-8, -5, (num_local, HIDDEN, INTER // 32), device="cuda", generator=g).float())
    return w13, s13, w2, s2


def build_mega_weights(deep_gemm, w13, s13, w2, s2, mma_type):
    """Same layout SGLang's build_mega_moe_experts_weights produces (UE8M0-packed SF, gate/up interleave, UTCCP)."""
    n1, k1 = w13.shape[1], w13.shape[2] * 2
    n2, k2 = w2.shape[1], w2.shape[2] * 2
    sf13 = deep_gemm.transform_sf_into_required_layout(s13, mn=n1, k=k1, recipe=(1, 32), num_groups=w13.shape[0],
                                                       disable_ue8m0_cast=False)
    sf2 = deep_gemm.transform_sf_into_required_layout(s2, mn=n2, k=k2, recipe=(1, 32), num_groups=w2.shape[0],
                                                      disable_ue8m0_cast=False)
    try:
        return deep_gemm.transform_weights_for_mega_moe((w13, sf13), (w2, sf2), activation="swiglu",
                                                        mma_type=mma_type)
    except TypeError:  # older deep_gemm (0.1.x) without the mma_type keyword: FP8xFP4 layout only
        assert mma_type == "fp8xfp4"
        return deep_gemm.transform_weights_for_mega_moe((w13, sf13), (w2, sf2))


def make_routing(num_tokens, rank, mode):
    g = torch.Generator(device="cuda").manual_seed(1000 + rank + 17 * num_tokens)
    if mode == "balanced":  # every expert receives the same count (best case for the tile heuristic)
        base = torch.arange(num_tokens * TOPK, device="cuda") + rank * num_tokens * TOPK
        idx = (base * 7 % NUM_EXPERTS).view(num_tokens, TOPK)
        w = torch.full((num_tokens, TOPK), 1.0 / TOPK, device="cuda")
        return w, idx
    scores = torch.randn((num_tokens, NUM_EXPERTS), device="cuda", generator=g)
    w, idx = torch.topk(scores, TOPK, dim=-1, sorted=False)
    return torch.softmax(w, -1), idx


# ------------------------------------------------------------------------------------------------ reference
def reference(x, topk_w, topk_idx, w13, s13, w2, s2, rank, world, act_kind):
    """Full MoE output for this rank's tokens: each rank computes its local experts' contribution to every
    gathered token, then an all-reduce sums the partials. Ranks may hold different token counts (padded to the
    largest with -1 routes for the gather)."""
    T = x.shape[0]
    t_max = torch.tensor([T], device="cuda")
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    Tm = int(t_max)
    xp = torch.zeros((Tm, HIDDEN), dtype=x.dtype, device="cuda")
    wp = torch.zeros((Tm, TOPK), dtype=topk_w.dtype, device="cuda")
    ip = torch.full((Tm, TOPK), -1, dtype=topk_idx.dtype, device="cuda")
    xp[:T], wp[:T], ip[:T] = x, topk_w, topk_idx
    xs = [torch.empty_like(xp) for _ in range(world)]
    ws = [torch.empty_like(wp) for _ in range(world)]
    ids = [torch.empty_like(ip) for _ in range(world)]
    dist.all_gather(xs, xp)
    dist.all_gather(ws, wp)
    dist.all_gather(ids, ip)
    X, W, I = torch.cat(xs), torch.cat(ws), torch.cat(ids)
    xq = fake_quant(X, act_kind)
    out = torch.zeros((X.shape[0], HIDDEN), dtype=torch.float32, device="cuda")
    nl = w13.shape[0]
    for e in range(nl):
        tok, slot = (I == rank * nl + e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        W13 = dequant_fp4_packed(w13[e], s13[e])
        h = xq[tok] @ W13.t()
        del W13
        gate, up = h[:, :INTER].clamp(max=SWIGLU_LIMIT), h[:, INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        act = torch.nn.functional.silu(gate) * up * W[tok, slot].unsqueeze(1)
        act = fake_quant(act, act_kind)
        W2 = dequant_fp4_packed(w2[e], s2[e])
        out.index_add_(0, tok, act @ W2.t())
        del W2
    dist.all_reduce(out)
    return out[rank * Tm:rank * Tm + T]


# ------------------------------------------------------------------------------------------------ one case
class MegaCase:
    def __init__(self, deep_gemm, group, rank, world, mma_type, weights, max_tokens, reserved_sms):
        self.dg, self.group, self.rank, self.world, self.mma_type = deep_gemm, group, rank, world, mma_type
        phys = torch.cuda.get_device_properties(0).multi_processor_count
        n = max(2, phys - reserved_sms)
        self.num_sms = n - n % 2
        self.phys_sms = phys
        self.max_tokens = max_tokens
        with self.sms():
            self.buf = deep_gemm.get_symm_buffer_for_mega_moe(group, NUM_EXPERTS, max_tokens, TOPK, HIDDEN, INTER,
                                                              mma_type=mma_type, activation="swiglu")
        self.l1, self.l2 = build_mega_weights(deep_gemm, *weights, mma_type)
        if mma_type == "fp8xfp4":
            from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch as sgl_pre
            self.sgl_pre = sgl_pre

    def sms(self):
        dg, target = self.dg, self.num_sms

        class _Ctx:
            def __enter__(s):
                s.prev = dg.get_num_sms()
                dg.set_num_sms(min(target, s.prev) - min(target, s.prev) % 2)

            def __exit__(s, *a):
                dg.set_num_sms(s.prev)
        return _Ctx()

    def pre(self, x, ids, w):
        b = self.buf
        if self.mma_type == "mxf4xmxf4":
            self.dg.mega_moe_pre_dispatch(x, ids, w, b.x, b.x_sf, b.topk_idx, b.topk_weights,
                                          num_tokens=x.shape[0], group_size=32, mma_type=self.mma_type)
        else:
            self.sgl_pre(x, ids, w, b.x, b.x_sf, b.topk_idx, b.topk_weights, quant_group_size=32)

    def moe(self, y):
        with self.sms():
            self.dg.fp8_fp4_mega_moe(y, self.l1, self.l2, self.buf, recipe=(1, 1, 32), activation="swiglu",
                                     activation_clamp=SWIGLU_LIMIT, fast_math=True)

    def time_graph(self, fn, steps, replays):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier(self.group)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(steps):
                fn()
        g.replay()
        torch.cuda.synchronize()
        dist.barrier(self.group)
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(replays):
            g.replay()
        t1.record()
        torch.cuda.synchronize()
        us = t0.elapsed_time(t1) * 1e3 / (steps * replays)
        del g
        return us


def touched_bytes(topk_idx_all: torch.Tensor, rank: int, nl: int) -> tuple[int, int]:
    local = topk_idx_all[(topk_idx_all >= rank * nl) & (topk_idx_all < (rank + 1) * nl)]
    n_touch = int(torch.unique(local).numel())
    per_expert = 3 * INTER * HIDDEN // 2 + 3 * INTER * HIDDEN // 32
    return n_touch, n_touch * per_expert


CONFIG_KEYS = ("BLOCK_M", "BLOCK_K", "STORE_BLOCK_M", "MAX_STAGES", "PULL_BYTES")


def apply_config(cfg: str) -> None:
    """Host config overrides are read by the patched deep_gemm on every call (std::getenv), so they can change
    between calls inside one process."""
    for k in CONFIG_KEYS:
        os.environ.pop(f"DG_MEGA_MOE_{k}", None)
    if cfg != "default":
        for kv in cfg.split(","):
            k, v = kv.split("=")
            assert k in CONFIG_KEYS, k
            os.environ[f"DG_MEGA_MOE_{k}"] = v


def run(args):
    rank, world, group = init()
    configs = args.configs.split(";")
    import deep_gemm
    nl = NUM_EXPERTS // world
    weights = make_weights(nl, seed=rank)
    tokens = [int(t) for t in args.tokens.split(",")]
    if args.tokens_by_rank:
        by_rank = [int(t) for t in args.tokens_by_rank.split(",")]
        assert len(by_rank) == world, "one token count per rank"
        tokens = [by_rank[rank]]
        t_sum = sum(by_rank)
    results = []
    if rank == 0:
        print(f"deep_gemm {deep_gemm.__file__} torch {torch.__version__} "
              f"gpu {torch.cuda.get_device_name()} sms {torch.cuda.get_device_properties(0).multi_processor_count}",
              flush=True)
    for mma in args.mma.split(","):
        case = MegaCase(deep_gemm, group, rank, world, mma, weights, args.max_tokens_per_rank, args.reserved_sms)
        act_kind = "fp4" if mma == "mxf4xmxf4" else "fp8"
        for T in tokens:
            if T > args.max_tokens_per_rank:
                continue
            x = torch.randn((T, HIDDEN), dtype=torch.bfloat16, device="cuda")
            w, idx = make_routing(T, rank, args.routing)
            ids32, w32 = idx.to(torch.int32), w.float()
            T_run = T
            if args.tokens_by_rank and args.pad_policy != "none":
                T_run = max(T, -(-t_sum // world) if args.pad_policy == "mean" else max(by_rank))
            y = torch.empty((max(T_run, 1), HIDDEN), dtype=torch.bfloat16, device="cuda")

            def step_pre():
                case.pre(x, ids32, w32)

            def step_all():
                case.pre(x, ids32, w32)
                if T_run > T:  # padded rows carry no routes; the kernel writes zero rows for them
                    case.buf.topk_idx[T:T_run].fill_(-1)
                case.moe(y)

            ref = None
            for cfg in configs:
                apply_config(cfg)
                get_bm = getattr(deep_gemm, "get_block_m_for_mega_moe", None)  # absent in deep_gemm 0.1.5
                block_m = get_bm(world, NUM_EXPERTS, case.buf.num_max_tokens_per_rank, T_run, TOPK, mma) if get_bm else -1
                try:
                    t_pre = case.time_graph(step_pre, args.graph_steps, args.replays)
                    t_all = case.time_graph(step_all, args.graph_steps, args.replays)
                except Exception as exc:  # an illegal override fails the host-side asserts identically on all ranks
                    if rank == 0:
                        print(json.dumps(dict(config=cfg, mma=mma, tokens_per_rank=T, error=str(exc)[:300])), flush=True)
                    torch.cuda.synchronize()
                    dist.barrier(group)
                    continue
                diff = None
                if not args.no_check:
                    if ref is None:
                        ref = reference(x, w32, idx, *weights, rank, world, act_kind)
                    diff = calc_diff(y[:T].float(), ref)
                    finite = bool(torch.isfinite(y[:T]).all())
                    if not finite:
                        diff = float("inf")
                t_max = torch.tensor([T], device="cuda")
                dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
                idx_pad = torch.full((int(t_max), TOPK), -1, dtype=idx.dtype, device="cuda")
                idx_pad[:T] = idx
                ids_all = [torch.empty_like(idx_pad) for _ in range(world)]
                dist.all_gather(ids_all, idx_pad)
                n_touch, nbytes = touched_bytes(torch.cat(ids_all), rank, nl)
                rec = torch.tensor([t_pre, t_all, float(nbytes), float(n_touch), -1.0 if diff is None else diff,
                                    float(block_m)],
                                   dtype=torch.float64, device="cuda")
                recs = [torch.empty_like(rec) for _ in range(world)]
                dist.all_gather(recs, rec)
                R = torch.stack(recs).cpu()
                if rank == 0:
                    floor_us = float(R[:, 2].max()) / (args.hbm_tbps * 1e12) * 1e6
                    row = dict(label=args.label, config=cfg, pad_policy=args.pad_policy, tokens_run=T_run, mma=mma, tokens_per_rank=T,
                               tokens_by_rank=args.tokens_by_rank or None,
                               total_us_by_rank=[round(float(v), 1) for v in R[:, 1]], block_m=int(block_m),
                               block_m_by_rank=[int(v) for v in R[:, 5]],
                               num_sms=case.num_sms, max_tokens_per_rank=args.max_tokens_per_rank,
                               pre_us=float(R[:, 0].median()), total_us=float(R[:, 1].median()),
                               mega_us=float(R[:, 1].median() - R[:, 0].median()),
                               total_us_max_rank=float(R[:, 1].max()),
                               touched_experts_max=int(R[:, 3].max()), floor_us=floor_us,
                               diff_max=None if diff is None else float(R[:, 4].max()))
                    row["x_floor"] = row["mega_us"] / floor_us if floor_us else None
                    results.append(row)
                    print(json.dumps(row), flush=True)
            apply_config("default")
            del y
        case.buf.destroy()
        del case
        torch.cuda.empty_cache()
    if rank == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), results=results, time=time.time()), f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    _args = parse_args()
    if _args.dg_path:
        sys.path.insert(0, _args.dg_path)
    run(_args)
