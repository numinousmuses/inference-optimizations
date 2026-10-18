"""CPU end-to-end check of the two-tier KDA state (k3state) against stock SGLang, with the REAL stock Triton kernels run
by the Triton interpreter (TRITON_INTERPRET=1). No GPU.

This is the route the ROCm image takes: on HIP the fused Kimi-K3/DSpARK CuTe verify is unavailable
(kda_backend._can_run_dspark_cutedsl_mtp requires CUDA) and the fused chain-verify + ring is gated to SM90/SM100 at
batch 1 (_can_run_fused_chain_verify), so target verify is causal_conv1d_update + fused_sigmoid_gating_delta_rule_update
with the fused ReplaySSM ring write, then commit_kda_replayssm_after_verify. On CPU the same fallbacks are taken.

For one SGLang tree (a git revision of ~/sglang-fork: v0.5.20 = B300 image, 172b1b4825 = ROCm image) and one format:
  1. export python/sglang at the revision, apply operations/patches/k3state_hooks.py to it, import it;
  2. build a stock fp32 MambaPool and a two-tier MambaPool (real classes, K3-shaped per-rank layout reduced to 2 heads);
  3. verify (3 requests x 8 draft tokens, 2 KDA layers) + commit with accept lengths (3, 1, 8) on both; then a
     non-speculative decode step; then a prefill (fresh + resumed-from-pool requests);
  4. report the max abs differences of outputs, working state vs stock state, pool mirror vs ref_quant(W), conv.

    TRITON_INTERPRET=1 python3 operations/patches/k3state_cpu_e2e.py --rev 172b1b4825 --fmt fp8 [--json out.json]
The pytest wrapper is operations/patches/test_k3state_cpu_e2e.py.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(HERE, "..", "kernels")
FORK = os.path.expanduser("~/sglang-fork")
HV, KD, NTOK = 2, 128, 8
DEV = "cpu"          # --device cuda on a GPU box (ROCm torch calls HIP devices "cuda"): real kernels, no shims
DIM = 3 * HV * KD
LAYERS = (1, 2)


def export_tree(rev: str, dst: str) -> str:
    os.makedirs(dst, exist_ok=True)
    arch = subprocess.run(["git", "-C", FORK, "archive", rev, "python/sglang"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", dst], input=arch.stdout, check=True)
    return os.path.join(dst, "python")


def _cache_params():
    import torch
    from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape, Mamba2StateDType
    shape = KimiLinearStateShape.create(tp_world_size=8, num_heads=8 * HV, head_dim=KD, num_k_heads=8 * HV,
                                        head_k_dim=KD, conv_kernel_size=4, num_spec=0)
    return KimiLinearCacheParams(shape=shape, layers=list(LAYERS),
                                 dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32))


def _pool(fmt: str):
    from sglang.srt.mem_cache.memory_pool import MambaPool
    os.environ["NUMINOUS_K3STATE"] = fmt
    return MambaPool(size=10, spec_state_size=6, cache_params=_cache_params(), mamba_layer_ids=list(LAYERS),
                     device=DEV, speculative_num_draft_tokens=NTOK, speculative_eagle_topk=1,
                     enable_linear_replayssm_spec=True, linear_replayssm_cache_len=16)


class _ReqPool:
    def __init__(self, pool):
        self.mamba_pool = pool
        self.map = {lid: i for i, lid in enumerate(LAYERS)}
        self.size = 6

    def mamba2_layer_index(self, layer_id):
        return self.map[layer_id]

    def mamba2_layer_cache(self, layer_id):
        return self.mamba_pool.mamba2_layer_cache(self.map[layer_id])


def _backend(pool, slots, qsl, verify: str):
    import torch
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend, KDAKernelDispatcher
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend as LB
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    be = object.__new__(KDAAttnBackend)
    be.req_to_token_pool = _ReqPool(pool)
    be.kernel_dispatcher = KDAKernelDispatcher(LB.TRITON, LB.TRITON, LB(verify))
    be._fused_chain_verify_fn = None
    be.accept_lens_pool = None
    be.verify_intermediate_state_indices = torch.arange(8, dtype=torch.int32)
    be.forward_metadata = ForwardMetadata(query_start_loc=qsl, mamba_cache_indices=slots)
    be.forward_metadata.has_mamba_track_mask = False
    return be


def _layer(layer_id):
    import torch
    g = torch.Generator(device=DEV).manual_seed(layer_id)
    return types.SimpleNamespace(
        layer_id=layer_id, num_q_heads=HV, num_k_heads=HV, num_v_heads=HV, head_q_dim=KD, head_k_dim=KD,
        head_v_dim=KD, q_dim=HV * KD, k_dim=HV * KD, v_dim=HV * KD, bias=None, lower_bound=-5.0,
        conv_weights=torch.randn(DIM, 4, generator=g) * 0.3,
        A_log=torch.randn(HV, generator=g).reshape(1, 1, HV, 1) * 0.5,
        dt_bias=torch.randn(HV * KD, generator=g) * 0.5)


def reference_extend(*, q, k, v, g, beta, ssm_states, cache_indices, query_start_loc, A_log, dt_bias,
                     lower_bound=None, return_intermediate_states=False, **_):
    """Deterministic torch stand-in for the chunked KDA prefill kernel, used identically by the stock and the two-tier
    runs (the check is the engine's state plumbing: initial state from the right slot / row, final state written back,
    outputs). Triton 3.7's interpreter cannot run the stock chunk kernels (InterpreterBuilder has no get_bf16), so they
    are verified on hardware (research/k3amd.md gate V3). Gated delta rule with a per-K decay, fp32, in place on
    ssm_states[cache_indices[i]]; returns [1, T, HV, V] in q's dtype."""
    import torch
    gate = torch.sigmoid(g.float()) * 0.5 + 0.45                       # per-K decay in (0.45, 0.95)
    bt = torch.sigmoid(beta.float())
    qs = torch.nn.functional.normalize(q.float(), dim=-1)
    ks = torch.nn.functional.normalize(k.float(), dim=-1)
    out = torch.zeros(v.shape, dtype=torch.float32)
    qsl = query_start_loc.tolist()
    for i in range(len(qsl) - 1):
        slot = int(cache_indices[i])
        if slot < 0:
            continue
        h = ssm_states[slot].float().clone()                          # [HV, V, K]
        for t in range(qsl[i], qsl[i + 1]):
            h = h * gate[0, t][:, None, :]
            kv = torch.einsum("hvk,hk->hv", h, ks[0, t])
            h = h + bt[0, t][:, None, None] * torch.einsum("hv,hk->hvk", v[0, t].float() - kv, ks[0, t])
            out[0, t] = torch.einsum("hvk,hk->hv", h, qs[0, t])
        ssm_states[slot] = h.to(ssm_states.dtype)
    out = out.to(q.dtype)
    return (out, None) if return_intermediate_states else out


def mirror_rel_err(q_rows, w_rows, fmt) -> float:
    """max |dequant(pool mirror) - W| / row amax: the interpreter's float casts round ties differently from CUDA/HIP
    (codes may sit one apart from ref_quant), so the mirror is judged by what it decodes to."""
    import sglang.srt.layers.attention.numinous_k3state_quant as kq
    deq = kq.ref_dequant(q_rows.cpu(), fmt)
    w_rows = w_rows.cpu()
    amax = w_rows.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return float(((deq - w_rows).abs() / amax).max())


def _maxdiff(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def interpreter_range_shim(*modules) -> None:
    """Triton's interpreter runs `for t in range(0, n)` with Python ints, so `t.to(tl.int64)` (valid compiled Triton,
    used by SGLang's KDA commit fold and by ours) raises AttributeError there. Shadow `range` in those kernel modules
    only, only under TRITON_INTERPRET=1, with one that yields int32 Triton scalars (what the compiler gives the loop
    variable). Kernel source is untouched."""
    import builtins

    import triton.language as tl

    def _int(x):
        h = getattr(x, "handle", None)
        return int(h.data.item()) if h is not None else int(x)

    def _range(*args):
        for i in builtins.range(*[_int(a) for a in args]):
            yield tl.full([], i, tl.int32)

    for m in modules:
        m.range = _range


def conv_scatter_reference(dst, src, dst_indices_raw, step_indices_raw):
    """torch twin of mamba_state_scatter_triton.fused_conv_window_scatter_with_mask (whose wrapper refuses non-CUDA
    tensors): request i with step >= 0 and a valid slot copies src[:, i, step] into dst[:, slot]."""
    for i in range(step_indices_raw.shape[0]):
        step, slot = int(step_indices_raw[i]), int(dst_indices_raw[i])
        if step < 0 or not (0 <= slot < dst.shape[1]) or i >= src.shape[1] or step >= src.shape[2]:
            continue
        dst[:, slot] = src[:, i, step]


def single_config_autotune() -> None:
    """The interpreter has no benchmarking driver: autotuned kernels (the chunked KDA prefill) run their first config."""
    from triton.runtime.autotuner import Autotuner
    if getattr(Autotuner, "_k3s_single", False):
        return
    orig = Autotuner.run

    def run(self, *args, **kwargs):
        self.configs = self.configs[:1]
        return orig(self, *args, **kwargs)

    Autotuner.run, Autotuner._k3s_single = run, True


def late_bound_math_aliases() -> None:
    """fla/op.py binds `exp = tl.exp` (and exp2 / log / log2) at import; the interpreter patches `tl` only while a
    kernel runs, so the early-bound builtins refuse to run. Rebind those module aliases (op.py and every fla module
    that imported them) to call-time lookups. Compiled kernels are unaffected: this runs only under TRITON_INTERPRET."""
    import triton.language as tl
    names = {"exp": "exp", "exp2": ("math", "exp2"), "log": "log", "log2": "log2"}

    def resolve(spec):
        return getattr(tl.math, spec[1]) if isinstance(spec, tuple) else getattr(tl, spec)

    originals = {n: resolve(spec) for n, spec in names.items()}
    for mname, mod in list(sys.modules.items()):
        if not mname.startswith("sglang.kernels.ops.attention.fla") or mod is None:
            continue
        for n, spec in names.items():
            if getattr(mod, n, None) is originals[n]:
                setattr(mod, n, (lambda sp: (lambda *a, **k: resolve(sp)(*a, **k)))(spec))


def install_cpu_conv_scatter() -> None:
    """Both the stock commit and the engine import the scatter from this module at call time."""
    import sglang.kernels.ops.mamba.mamba_state_scatter_triton as M
    M.fused_conv_window_scatter_with_mask = conv_scatter_reference


def run(fmt: str, verify: str) -> dict:
    import torch
    from sglang.kernels.ops.attention.fla import kda_replayssm_spec_decode as R
    import sglang.srt.layers.attention.numinous_k3state as E
    import sglang.srt.layers.attention.numinous_k3state_quant as kq

    interp = DEV == "cpu"
    if interp:
        interpreter_range_shim(R, kq)
        install_cpu_conv_scatter()
        single_config_autotune()
        import sglang.kernels.ops.attention.fla.kda  # noqa: F401  (prefill chain; aliases rebound below)
        import sglang.srt.layers.attention.linear.kernels.kda_triton  # noqa: F401
        late_bound_math_aliases()
    torch.set_default_device(DEV)

    res: dict = {"fmt": fmt, "verify_backend": verify}
    nl = len(LAYERS)
    # ------------------------------------------------------------------ verify + commit
    bs = 3
    slots = torch.tensor([7, 2, 9], dtype=torch.int32)
    req_rows = torch.tensor([4, 1, 5], dtype=torch.int64)
    qsl = torch.arange(0, (bs + 1) * NTOK, NTOK, dtype=torch.int32)
    ref, two = _pool("off"), _pool(fmt)
    res["pool_dtype"] = str(two.mamba_cache.temporal.dtype)
    res["ring_rows"] = [int(ref.mamba_cache.replayssm_rawv.shape[1]), int(two.mamba_cache.replayssm_rawv.shape[1])]
    g = torch.Generator(device=DEV).manual_seed(0)
    t = ref.mamba_cache.temporal
    for s in slots.tolist():
        t[:, s] = torch.randn(t[:, s].shape, generator=g) * 0.05
    conv = ref.mamba_cache.conv[0]
    conv.copy_(torch.randn(conv.shape, generator=g).to(conv.dtype))
    two.mamba_cache.conv[0].copy_(conv)
    for li in range(nl):
        kq.quant_rows(t[li], slots, two.mamba_cache.temporal[li], slots, fmt)
        two.k3s_w[li][req_rows] = t[li][slots.long()]
    fb = types.SimpleNamespace(spec_info=types.SimpleNamespace(draft_token_num=NTOK, ragged_verify_layout=None),
                               req_pool_indices=req_rows,
                               forward_mode=types.SimpleNamespace(is_target_verify=lambda: True,
                                                                  is_draft_extend_v2=lambda: False))
    outs = {}
    accept = torch.tensor([3, 1, 8], dtype=torch.int32)
    for name, pool in (("stock", ref), ("two", two)):
        os.environ["NUMINOUS_K3STATE"] = "off" if name == "stock" else fmt
        be = _backend(pool, slots, qsl, verify)
        outs[name] = []
        for lid in LAYERS:
            gg = torch.Generator(device=DEV).manual_seed(100 + lid)
            mixed = torch.randn(bs * NTOK, DIM, generator=gg).to(torch.bfloat16)
            a = torch.randn(1, bs * NTOK, HV, KD, generator=gg).to(torch.bfloat16)
            b = torch.randn(1, bs * NTOK, HV, generator=gg).to(torch.bfloat16)
            outs[name].append(be.forward_extend(_layer(lid), fb, mixed, a, b))
        R.commit_kda_replayssm_after_verify(spec_state=pool.mamba_cache, state_batch_indices=slots,
                                            accept_lens=accept, last_correct_step_indices=accept - 1,
                                            null_block_id=-1)
    print("stage verify+commit done", flush=True)
    res["verify_out_maxdiff"] = max(_maxdiff(x, y) for x, y in zip(outs["stock"], outs["two"]))
    res["verify_out_absmax"] = max(float(x.float().abs().max()) for x in outs["stock"])
    res["commit_w_vs_stock_maxdiff"] = max(_maxdiff(two.k3s_w[li][req_rows], t[li][slots.long()]) for li in range(nl))
    res["commit_state_absmax"] = float(t[:, slots.long()].abs().max())
    res["commit_mirror_vs_refquant_mismatch_frac"] = max(
        float((two.mamba_cache.temporal[li][slots.long()].cpu() != kq.ref_quant(two.k3s_w[li][req_rows].cpu(), fmt))
              .float().mean()) for li in range(nl))
    res["commit_mirror_rel_err"] = max(
        mirror_rel_err(two.mamba_cache.temporal[li][slots.long()], two.k3s_w[li][req_rows], fmt) for li in range(nl))
    res["commit_conv_equal"] = bool(torch.equal(ref.mamba_cache.conv[0], two.mamba_cache.conv[0]))

    # ------------------------------------------------------------------ non-speculative decode
    dslots = torch.tensor([6, 1, 9], dtype=torch.int32)
    drows = torch.tensor([2, 5, 3], dtype=torch.int64)
    ref, two = _pool("off"), _pool(fmt)
    t = ref.mamba_cache.temporal
    g = torch.Generator(device=DEV).manual_seed(1)
    for s in dslots.tolist():
        t[:, s] = torch.randn(t[:, s].shape, generator=g) * 0.05
    ref.mamba_cache.conv[0].copy_(torch.randn(ref.mamba_cache.conv[0].shape, generator=g).to(torch.bfloat16))
    two.mamba_cache.conv[0].copy_(ref.mamba_cache.conv[0])
    for li in range(nl):
        kq.quant_rows(t[li], dslots, two.mamba_cache.temporal[li], dslots, fmt)
        two.k3s_w[li][drows] = t[li][dslots.long()]
    fb = types.SimpleNamespace(req_pool_indices=drows, mamba_track_mask=None,
                               forward_mode=types.SimpleNamespace(is_target_verify=lambda: False,
                                                                  is_draft_extend_v2=lambda: False))
    dq = torch.arange(0, bs + 1, dtype=torch.int32)
    for name, pool in (("stock", ref), ("two", two)):
        os.environ["NUMINOUS_K3STATE"] = "off" if name == "stock" else fmt
        be = _backend(pool, dslots, dq, verify)
        outs[name] = []
        for lid in LAYERS:
            gg = torch.Generator(device=DEV).manual_seed(300 + lid)
            mixed = torch.randn(bs, DIM, generator=gg).to(torch.bfloat16)
            a = torch.randn(1, bs, HV, KD, generator=gg).to(torch.bfloat16)
            b = torch.randn(1, bs, HV, generator=gg).to(torch.bfloat16)
            outs[name].append(be.forward_decode(_layer(lid), fb, mixed, a, b))
    res["decode_out_maxdiff"] = max(_maxdiff(x, y) for x, y in zip(outs["stock"], outs["two"]))
    res["decode_w_vs_stock_maxdiff"] = max(_maxdiff(two.k3s_w[li][drows], t[li][dslots.long()]) for li in range(nl))
    res["decode_conv_equal"] = bool(torch.equal(ref.mamba_cache.conv[0], two.mamba_cache.conv[0]))

    # ------------------------------------------------------------------ prefill: fresh, then resumed from the pool
    pslots = torch.tensor([3, 8], dtype=torch.int32)
    prows = torch.tensor([2, 5], dtype=torch.int64)
    lens = [37, 70]
    pq = torch.tensor([0, 37, 107], dtype=torch.int32)
    for resume in (False, True):
        ref, two = _pool("off"), _pool(fmt)
        if resume:
            init = torch.randn(nl, 2, HV, KD, KD, generator=torch.Generator(device=DEV).manual_seed(9)) * 0.05
            for li in range(nl):
                kq.quant_rows(init[li], None, two.mamba_cache.temporal[li], pslots, fmt)
                ref.mamba_cache.temporal[li][pslots.long()] = kq.ref_dequant(
                    two.mamba_cache.temporal[li][pslots.long()].cpu(), fmt).to(DEV)
        fb = types.SimpleNamespace(
            req_pool_indices=prows, extend_prefix_lens=torch.tensor([5 if resume else 0] * 2),
            extend_seq_lens_cpu=lens,
            forward_mode=types.SimpleNamespace(is_target_verify=lambda: False, is_draft_extend_v2=lambda: False))
        for name, pool in (("stock", ref), ("two", two)):
            os.environ["NUMINOUS_K3STATE"] = "off" if name == "stock" else fmt
            be = _backend(pool, pslots, pq, verify)
            if interp:
                be.kernel_dispatcher.extend = reference_extend
                be.kernel_dispatcher.effective_extend_kernel = lambda lb: types.SimpleNamespace(
                    supports_track_state_snapshot=True)
            outs[name] = []
            for lid in LAYERS:
                gg = torch.Generator(device=DEV).manual_seed(200 + lid)
                mixed = torch.randn(sum(lens), DIM, generator=gg).to(torch.bfloat16)
                a = torch.randn(1, sum(lens), HV * KD, generator=gg).to(torch.bfloat16)
                b = torch.randn(1, sum(lens), HV, generator=gg).to(torch.bfloat16)
                outs[name].append(be.forward_extend(_layer(lid), fb, mixed, a, b))
        key = "prefill_resume" if resume else "prefill_fresh"
        res[f"{key}_out_maxdiff"] = max(_maxdiff(x, y) for x, y in zip(outs["stock"], outs["two"]))
        res[f"{key}_w_vs_stock_maxdiff"] = max(
            _maxdiff(two.k3s_w[li][prows], ref.mamba_cache.temporal[li][pslots.long()]) for li in range(nl))
        res[f"{key}_mirror_vs_refquant_mismatch_frac"] = max(
            float((two.mamba_cache.temporal[li][pslots.long()].cpu() != kq.ref_quant(two.k3s_w[li][prows].cpu(), fmt))
                  .float().mean()) for li in range(nl))
        res[f"{key}_mirror_rel_err"] = max(
            mirror_rel_err(two.mamba_cache.temporal[li][pslots.long()], two.k3s_w[li][prows], fmt) for li in range(nl))
        res[f"{key}_conv_equal"] = bool(torch.equal(ref.mamba_cache.conv[0], two.mamba_cache.conv[0]))
    res["prefill_kernel"] = "torch reference (stock chunk kernels need hardware)" if interp else "stock Triton chunk_kda"
    res["device"] = DEV
    res["engine_mode"] = E.mode()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="172b1b4825")
    ap.add_argument("--fmt", default="fp8", choices=["fp8", "fp4", "bf16"])
    ap.add_argument("--verify", default="nv_cutedsl", choices=["nv_cutedsl", "triton"],
                    help="the K3 hook pins nv_cutedsl under spec; off CUDA it runs the Triton verify")
    ap.add_argument("--tree", default="", help="an exported python/ dir to reuse (else a temp export)")
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda: on a GPU box with the real kernels (ROCm torch names HIP devices cuda)")
    ap.add_argument("--installed", action="store_true",
                    help="use the installed (already patched) sglang instead of exporting a tree")
    a = ap.parse_args(argv)
    global DEV
    DEV = a.device
    if DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("set TRITON_INTERPRET=1 (the stock Triton kernels run in the interpreter on CPU)")
    if a.installed:
        import sglang
        res = {"rev": "installed:" + os.path.dirname(sglang.__file__), **run(a.fmt, a.verify)}
        print("@@RESULT@@" + json.dumps(res))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)
        return 0
    tree = a.tree or export_tree(a.rev, tempfile.mkdtemp(prefix="k3s_e2e_"))
    sys.path.insert(0, HERE)
    import k3state_hooks as H
    stamp = os.path.join(tree, ".k3state_applied")
    if not os.path.exists(stamp):
        H.apply(os.path.join(tree, "sglang"))
        open(stamp, "w").write(a.rev)
    sys.path.insert(0, tree)
    importlib.invalidate_caches()
    import sglang
    assert os.path.realpath(sglang.__file__).startswith(os.path.realpath(tree)), sglang.__file__
    res = {"rev": a.rev, **run(a.fmt, a.verify)}
    print("@@RESULT@@" + json.dumps(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
