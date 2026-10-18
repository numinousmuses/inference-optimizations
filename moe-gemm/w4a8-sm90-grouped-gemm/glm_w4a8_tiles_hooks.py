"""Tuned SM90 tiles for SGLang v0.5.20's W4A8 MoE GEMM (cutlass_w4a8_moe_mm), opt-in per server (k3amd, GLM MoE GEMM
owner, coordinator 2026-10-05).

Why (measured, 1x H200, moe-round2 07:15, research/gpu-evidence/hopper/glm_moe_decode_bench_h200_20261005-071528.json):
sglang-kernel 0.4.7 has no tuned branch for GLM-5.3's TP4/TP8 shapes; the generic dispatch (w4a8_grouped_mm_c3x.cu:
160-178) runs CO<128,{16,32,64},512>, i.e. the TP4 down GEMM (K = 512) as one k-tile per CTA. The same kernel template
with other tiles (operations/kernels/w4a8_tiles, built into k3amd_w4a8.so from sglang-kernel's own source and the
CUTLASS commit it pins) timed 1.35-1.43x on decode w2 and 1.49-1.69x on prefill w13. Correctness of each config is
checked by the bench against the stock output before a rule names it.

Env (the server reads them at import and at weight processing):
  NUMINOUS_W4A8_TILES_SO   path of k3amd_w4a8.so (weights-cache:/k3amd_ext/<stamp>/, mounted at /cache in a run);
                           copied to a private local dir (sha256-checked) before torch.ops.load_library, so no volume
                           file stays open (P6 2026-10-05: WEIGHTS.reload() failed on the mapped .so)
  NUMINOUS_W4A8_TILES_SHA256 the library's pinned sha256 (env_for sets it); a mismatch raises
  NUMINOUS_W4A8_TILES      rules "<n_out>x<k>:<m_max>=<cfg>,...;..." per GEMM shape, first m_max >= m wins, e.g.
                           "1024x6144:576=9,1000000=5;6144x512:576=9,1000000=5" (TP4 w13 / w2)
  NUMINOUS_W4A8_TILES_ALIGN "<n_out>x<k>=<alignment>;..." the scale interleave of every shape a rule covers: TileK / 128
                           of ITS configs (all configs of one shape must share it; the scales are interleaved once)
  NUMINOUS_W4A8_DG          "so=<path>;sha=<sha256>;min_m=<tokens>;<n_out>x<k>=<cfg>;...": calls with at least min_m
                           tokens on a listed shape (and alignment-1 scales) run the k3amd W4A8 grouped GEMM
                           (operations/kernels/w4a8dg, torch.ops.k3amd_w4a8dg.mm) instead of cutlass; same staging
                           and sha256 pin as the tiles library
Shapes without a rule run the stock dispatch with the stock interleave.

Hooks (anchored replace, v0.5.20; missing / repeated anchors fail loudly; idempotent; patched files must compile):
  1. cutlass_w4a8_moe.py: the module's cutlass_w4a8_moe_mm becomes a dispatcher (rules -> torch.ops.k3amd_w4a8.mm_cfg,
     else the stock op).
  2. w4afp8.py process_weights_after_loading: interleave each projection's scales with the rule's alignment.

    python3 glm_w4a8_tiles_hooks.py [--root <sglang pkg>] [--check]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sglang_numinous_hooks import sglang_root  # noqa: E402

MARKER = "numinous-hook:glm_w4a8_tiles"


def parse_rules(text: str) -> dict:
    """'1024x6144:576=9,1000000=5;6144x512:...' -> {(1024, 6144): [(576, 9), (1000000, 5)], ...} (m_max ascending)."""
    out = {}
    for part in filter(None, (p.strip() for p in (text or "").split(";"))):
        shape, ranges = part.split(":", 1)
        n, k = (int(x) for x in shape.split("x"))
        out[(n, k)] = sorted((int(a), int(b)) for a, b in (r.split("=") for r in ranges.split(",")))
    return out


# Measured rules (moe-round2b, 1x H200 07:40, research/gpu-evidence/hopper/glm_moe_decode_bench_h200_20261005-074026.json;
# every config exact vs stock: max rel diff 0, cosine 1.0). All use TileK 128, so each shape's scales are interleaved
# with alignment 1 and every m range must be a TileK-128 config (-1 = stock only where the stock tile is TileK 128,
# i.e. TP8 w2). m <= 288: CO<256,16,128> (cfg 9); 288 < m <= 1024: CO<256,32,128> (cfg 13; measured best at 576 in
# round 2, an interpolation above it); larger: CO<128,128,128> (cfg 5) or CO<256,64,128> (cfg 6) as measured.
RULES = {
    "tp4": ("1024x6144:288=9,1024=13,1000000=5;6144x512:288=9,1024=13,8192=6,1000000=5", "1024x6144=1;6144x512=1"),
    # ep4 decode/prefill sweep moe-2xdp4 15:52 (glm_moe_decode_bench_h200_20261005-155249.json): cfg 9 at m 24-288
    # (1.06-1.20x), 13 at 576 (1.07x / 1.10x), 6 at 1024 (1.22x / 1.24x), 5 / 6 at 2048 (1.45x / 1.39x), 5 above
    "ep4": ("4096x6144:288=9,576=13,1024=6,1000000=5;6144x2048:288=9,576=13,2048=6,1000000=5",
            "4096x6144=1;6144x2048=1"),
    "tp8": ("512x6144:288=9,1024=13,1000000=5;6144x256:288=9,1024=13,8192=6,1000000=-1", "512x6144=1;6144x256=1"),
    # EP8 (32 whole experts per rank), all measured on 1xH200: prefill sweep moe-r3 14:12
    # (glm_moe_decode_bench_h200_20261005-141235.json: cfg 6 at m 2048 1.47x / 1.40x vs stock, cfg 5 above 1.42-1.62x)
    # and decode sweep moe-r5 14:57 (glm_moe_decode_bench_h200_20261005-145745.json: cfg 9 at m 24-288 1.10-1.21x,
    # cfg 13 at 576 1.07x / 1.14x, cfg 6 at 1024 1.32x / 1.31x)
    "ep8": ("4096x6144:288=9,576=13,2048=6,1000000=5;6144x2048:288=9,576=13,2048=6,1000000=5",
            "4096x6144=1;6144x2048=1"),
}
# tp8 w2 above m 8192 stays stock: cfg 8 CO<128,128,256> measured 1.107x / 1.127x at 8K / 16K (moe-r3) but needs scale
# alignment 2, and one shape's scales are interleaved once at load for all its m ranges (alignment 1 here)
SO = "/cache/k3amd_ext/20261005-065859/k3amd_w4a8.so"
SO_SHA256 = "3477049ea70243fb44ad5981d6c66c81e86b086a063f8cd4233a43f4c1070b10"   # measured from the volume copy
# k3amd W4A8 grouped GEMM (operations/kernels/w4a8dg; build record research/gpu-evidence/hopper/w4a8dg_build.json)
DG_SO = "/cache/k3amd_ext/w4a8dg_20261005-195707/k3amd_w4a8dg.so"      # v4 RS + kSilu (cfgs 40-44, 50/51)
DG_SHA256 = "4e34c9f26a87503cebe019b373b01b79f67faf29c6f46793ede3235902f11744"


def parse_dg(text: str) -> tuple:
    """'so=..;sha=..;min_m=2048;4096x6144=40;6144x2048=40' -> ({'so':..,'sha':..,'min_m':..}, {(4096, 6144): 40, ..})"""
    kv, shapes = {}, {}
    for part in filter(None, (p.strip() for p in (text or "").split(";"))):
        key, val = part.split("=", 1)
        if "x" in key and key.replace("x", "").isdigit():
            n, k = (int(x) for x in key.split("x"))
            shapes[(n, k)] = int(val)
        else:
            kv[key] = val
    return kv, shapes


def dg_env(layout: str, cfg: int = 40, min_m: int | str = "auto", silu: int | None = None) -> dict:
    """NUMINOUS_W4A8_DG for a server on the given layout: both MoE GEMM shapes on cfg (+ the fused-silu GEMM1).
    silu is off by default: measured on 1xH200 (box 23:00, w4a8dg_box_call_h200_20261005-2300.json) the fused-silu GEMM1
    makes the whole MoE call 7-10% slower than w4a8dg GEMM1 + the served silu kernel (EP8 16K 2,103 vs 1,935 us).
    min_m="auto" = the boot self-test decides per process (numinous_w4a8dg_selftest: bit-exact vs tiles or off, then
    the measured crossover). Needs the tiles env of the same layout (alignment-1 scales, and the tiles below min_m)."""
    rules, align = RULES[layout]
    shapes = ";".join(f"{sh}={cfg}" for sh in (p.split("=")[0] for p in align.split(";")))
    auto = min_m == "auto"
    spec = f"so={DG_SO};sha={DG_SHA256};min_m={'1' if auto else int(min_m)};{shapes}"
    spec += f";silu={silu}" if silu is not None else ""
    spec += ";selftest=1" if auto else ""
    return {"NUMINOUS_W4A8_DG": spec}


def env_for(layout: str) -> dict:
    """The three env values for a server whose MoE runs the given layout (tp4 / ep4 / tp8)."""
    rules, align = RULES[layout]
    return {"NUMINOUS_W4A8_TILES_SO": SO, "NUMINOUS_W4A8_TILES_SHA256": SO_SHA256, "NUMINOUS_W4A8_TILES": rules,
            "NUMINOUS_W4A8_TILES_ALIGN": align}


def moe_gemm_shapes(args: dict, hf_config) -> set:
    """(N, K) of this server's two W4A8 MoE GEMMs (w13: 2 x inter / moe_tp by hidden; w2: hidden by inter / moe_tp),
    moe_tp = tp_size / ep_size as v0.5.20 shards FusedMoE weights."""
    tp, ep = int(args.get("tp_size") or 1), int(args.get("ep_size") or 1)
    inter, hidden = int(hf_config.moe_intermediate_size), int(hf_config.hidden_size)
    moe_tp = max(1, tp // max(1, ep))
    return {(2 * inter // moe_tp, hidden), (hidden, inter // moe_tp)}


def numinous_preflight(ctx) -> list:
    """CPU preflight (operations/hopper/hook_preflight.py): what makes the tiles refuse to load or silently never run.
    The .so must be sha256-pinned to the build record, the rules must parse, and they must name THIS server's MoE GEMM
    shapes (a TP4 rule set on a TP8 server would leave every call on the stock tile), on a W4AFP8 checkpoint."""
    env = ctx.env
    so = env.get("NUMINOUS_W4A8_TILES_SO")
    if not so and not env.get("NUMINOUS_W4A8_TRITON") and not env.get("NUMINOUS_W4A8_DG"):
        return []
    out = dg_preflight(ctx) if env.get("NUMINOUS_W4A8_DG") else []
    sha = env.get("NUMINOUS_W4A8_TILES_SHA256")
    if not so:
        pass
    elif not sha:
        out.append(f"NUMINOUS_W4A8_TILES_SO={so} without NUMINOUS_W4A8_TILES_SHA256: the .so would load unpinned")
    elif so == SO and sha != SO_SHA256:
        out.append(f"NUMINOUS_W4A8_TILES_SHA256 {sha[:16]} != the build record {SO_SHA256[:16]} for {so}: the hook "
                   "refuses to load it")
    try:
        rules = parse_rules(env.get("NUMINOUS_W4A8_TILES", ""))
        align = parse_align(env.get("NUMINOUS_W4A8_TILES_ALIGN", ""))
    except ValueError as e:
        return out + [f"NUMINOUS_W4A8_TILES / _ALIGN do not parse: {e}"]
    quant = str(ctx.args.get("quantization") or getattr(ctx, "checkpoint_quant", "") or "")
    if quant and quant != "w4afp8":
        out.append(f"NUMINOUS_W4A8_TILES on a {quant} server: the W4A8 MoE GEMM never runs, so neither do the tiles")
    try:
        want = moe_gemm_shapes(ctx.args, ctx.hf_config)
    except (AttributeError, TypeError, ValueError):
        return out
    if rules and not (set(rules) & want):
        out.append(f"NUMINOUS_W4A8_TILES rules name GEMM shapes {sorted(rules)} but this server's MoE GEMMs are "
                   f"{sorted(want)} (tp {ctx.args.get('tp_size')}, ep {ctx.args.get('ep_size')}): every call stays on "
                   "the stock tile")
    tri = env.get("NUMINOUS_W4A8_TRITON", "")
    if tri:
        try:
            shapes = {tuple(int(x) for x in p.split("=")[0].split("x")) for p in tri.split(";")
                      if p.strip() and not p.startswith("min_m")}
            if not any(p.startswith("min_m=") for p in tri.split(";")):
                out.append("NUMINOUS_W4A8_TRITON without min_m=<tokens>: the Triton path never runs")
            if shapes and not (shapes & want):
                out.append(f"NUMINOUS_W4A8_TRITON names shapes {sorted(shapes)}, this server's MoE GEMMs are "
                           f"{sorted(want)}: inert")
            if any(align.get(sh) != 1 for sh in shapes & want):
                out.append("NUMINOUS_W4A8_TRITON needs NUMINOUS_W4A8_TILES_ALIGN=1 for its shapes (the kernel reads "
                           "[E, K/128, N] scales)")
        except ValueError as e:
            out.append(f"NUMINOUS_W4A8_TRITON does not parse: {e}")
    if set(rules) - set(align):
        out.append(f"NUMINOUS_W4A8_TILES_ALIGN misses {sorted(set(rules) - set(align))}: those weights keep the stock "
                   "scale interleave, which the TileK-128 configs do not read")
    return out


def dg_preflight(ctx) -> list:
    """What makes NUMINOUS_W4A8_DG refuse to load or never run: an unpinned / unrecorded library, no min_m, shapes that
    are not this server's MoE GEMMs, scales not interleaved with alignment 1 (the kernel reads [E, K/128, N]), a
    non-W4AFP8 checkpoint, or a device that is not SM90 (the kernel is sm_90a TMA + WGMMA)."""
    out = []
    try:
        kv, shapes = parse_dg(ctx.env.get("NUMINOUS_W4A8_DG", ""))
        align = parse_align(ctx.env.get("NUMINOUS_W4A8_TILES_ALIGN", ""))
    except ValueError as e:
        return [f"NUMINOUS_W4A8_DG does not parse: {e}"]
    if not kv.get("so") or not kv.get("sha"):
        out.append("NUMINOUS_W4A8_DG needs so=<path> and sha=<sha256>: the library would not load pinned")
    elif kv["so"] == DG_SO and kv["sha"] != DG_SHA256:
        out.append(f"NUMINOUS_W4A8_DG sha {kv['sha'][:16]} != the build record {DG_SHA256[:16]} for {kv['so']}")
    if "min_m" not in kv:
        out.append("NUMINOUS_W4A8_DG without min_m=<tokens>: the kernel never runs")
    if not shapes:
        out.append("NUMINOUS_W4A8_DG names no <n_out>x<k>=<cfg> shape: inert")
    quant = str(ctx.args.get("quantization") or getattr(ctx, "checkpoint_quant", "") or "")
    if quant and quant != "w4afp8":
        out.append(f"NUMINOUS_W4A8_DG on a {quant} server: the W4A8 MoE GEMM never runs")
    dev = getattr(ctx, "device", None) or {}
    if dev and (dev.get("is_hip") or int(dev.get("sm_major", 9)) != 9):
        out.append(f"NUMINOUS_W4A8_DG on {dev.get('sku')} (sm {dev.get('sm_major')}{dev.get('sm_minor', 0)}): "
                   "k3amd_w4a8dg is built for sm_90a only")
    try:
        want = moe_gemm_shapes(ctx.args, ctx.hf_config)
    except (AttributeError, TypeError, ValueError):
        return out
    if shapes and not (set(shapes) & want):
        out.append(f"NUMINOUS_W4A8_DG names shapes {sorted(shapes)}, this server's MoE GEMMs are {sorted(want)}: "
                   "inert")
    bad = sorted(sh for sh in set(shapes) & want if align.get(sh) != 1)
    if bad:
        out.append(f"NUMINOUS_W4A8_DG needs NUMINOUS_W4A8_TILES_ALIGN=1 for {bad} (the kernel reads [E, K/128, N] "
                   "scales)")
    return out


def parse_align(text: str) -> dict:
    out = {}
    for part in filter(None, (p.strip() for p in (text or "").split(";"))):
        shape, a = part.split("=")
        n, k = (int(x) for x in shape.split("x"))
        out[(n, k)] = int(a)
    return out


def pick(rules: dict, n: int, k: int, m: int):
    """The config id for a call, None for the stock dispatch (no rule, or a -1 range)."""
    for m_max, cfg in rules.get((n, k), []):
        if m <= m_max:
            return cfg if cfg >= 0 else None
    return None


DISPATCH = '''
# {marker}: tuned W4A8 tiles (operations/patches/glm_w4a8_tiles_hooks.py)
import os as _nm_os

_nm_stock_w4a8_mm = cutlass_w4a8_moe_mm
_nm_rules = None
_nm_lib = None          # the staged library path: loaded ONCE per process (a second copy re-registers the
                        # k3amd_w4a8 TORCH_LIBRARY and aborts; moe-dispatch bench 2026-10-05 12:43, rules re-read)


def _nm_stage_so(src, want_sha256=None):
    """Copy the library off the volume before loading it: a library loaded straight from /cache stays mapped and
    holds the volume file open, and every later Volume.reload() in the process then fails ("there are open files
    preventing the operation"; arm P6 2026-10-05). The copy lives in a private mkdtemp dir and is sha256-checked."""
    import hashlib
    import shutil
    import tempfile
    with open(src, "rb") as f:
        blob = f.read()
    got = hashlib.sha256(blob).hexdigest()
    if want_sha256 and got != want_sha256:
        raise RuntimeError(f"NUMINOUS_W4A8_TILES_SO {src}: sha256 {got[:16]} != pinned {want_sha256[:16]}")
    d = tempfile.mkdtemp(prefix="numinous_w4a8_tiles_")
    dst = _nm_os.path.join(d, _nm_os.path.basename(src))
    with open(dst, "wb") as f:
        f.write(blob)
    with open(dst, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != got:
            raise RuntimeError(f"NUMINOUS_W4A8_TILES_SO copy {dst} does not match {src}")
    return dst


def _nm_load_rules():
    global _nm_rules, _nm_lib
    if _nm_rules is None:
        _nm_rules = {}
        so = _nm_os.environ.get("NUMINOUS_W4A8_TILES_SO")
        if so:
            if _nm_lib is None:
                torch.ops.load_library(_nm_lib := _nm_stage_so(so, _nm_os.environ.get("NUMINOUS_W4A8_TILES_SHA256")))
            for part in filter(None, (p.strip() for p in _nm_os.environ.get("NUMINOUS_W4A8_TILES", "").split(";"))):
                shape, ranges = part.split(":", 1)
                n, k = (int(x) for x in shape.split("x"))
                _nm_rules[(n, k)] = sorted((int(a), int(b)) for a, b in (r.split("=") for r in ranges.split(",")))
    return _nm_rules


_nm_triton = None


def _nm_load_triton():
    """NUMINOUS_W4A8_TRITON='min_m=<tokens>;<N>x<K>=<block_n>,<block_t>,<warps>,<stages>;...': calls with at least
    min_m tokens and a listed GEMM shape run glm_w4a8_triton_mm (k3amd prefill kernel) instead of cutlass."""
    global _nm_triton
    if _nm_triton is None:
        spec = _nm_os.environ.get("NUMINOUS_W4A8_TRITON", "")
        min_m, shapes = 1 << 62, {}
        for part in filter(None, (p.strip() for p in spec.split(";"))):
            key, val = part.split("=", 1)
            if key == "min_m":
                min_m = int(val)
            else:
                n, k = (int(x) for x in key.split("x"))
                shapes[(n, k)] = tuple(int(x) for x in val.split(","))
        _nm_triton = (min_m, shapes)
    return _nm_triton


def _nm_expert_bounds(experts_offsets, e):
    """[E+1] row offsets: cutlass_w4a8_moe passes expert_offsets[:-1], a prefix view of its contiguous [E+1] tensor."""
    st = experts_offsets.storage_offset()
    if experts_offsets.dim() != 1 or experts_offsets.stride(0) != 1 or \
            experts_offsets.untyped_storage().nbytes() < (st + e + 1) * experts_offsets.element_size():
        return None
    return experts_offsets.as_strided((e + 1,), (1,), st)


_nm_dg = None
_nm_dg_lib = None       # loaded once per process, like the tiles library


def _nm_load_dg():
    """NUMINOUS_W4A8_DG='so=<path>;sha=<sha256>;min_m=<tokens>;<N>x<K>=<cfg>;...' (k3amd W4A8 grouped GEMM)."""
    global _nm_dg, _nm_dg_lib
    if _nm_dg is None:
        kv, shapes = {}, {}
        for part in filter(None, (p.strip() for p in _nm_os.environ.get("NUMINOUS_W4A8_DG", "").split(";"))):
            key, val = part.split("=", 1)
            if "x" in key and key.replace("x", "").isdigit():
                n, k = (int(x) for x in key.split("x"))
                shapes[(n, k)] = int(val)
            else:
                kv[key] = val
        min_m, silu = int(kv.get("min_m", 1 << 62)) if kv.get("min_m", "").isdigit() else 1 << 62, kv.get("silu")
        if kv.get("selftest") == "1":
            # the boot self-test (numinous_w4a8dg_selftest, run from W4AFp8MoEMethod.process_weights_after_loading)
            # decides per process: not ok / not run -> stay on the tiles path; else its measured min_m and silu
            import json as _nm_json
            dec = _nm_json.loads(_nm_os.environ.get("NUMINOUS_W4A8_DG_DECISION") or "{}")
            if dec.get("ok") and dec.get("min_m"):
                min_m, silu = int(dec["min_m"]), (silu if dec.get("silu") else None)
            else:
                shapes = {}
        if shapes and kv.get("so") and _nm_dg_lib is None:
            torch.ops.load_library(_nm_dg_lib := _nm_stage_so(kv["so"], kv.get("sha")))
        _nm_dg = (min_m, shapes, int(silu) if silu else None)
    return _nm_dg


def _nm_gemm1_silu(out_q, a, w13, a1_scale, w13_scale, a2_scale, expert_offsets, m, topk):
    """GEMM1 + silu*up -> FP8 in one kernel (torch.ops.k3amd_w4a8dg.mm_silu) when NUMINOUS_W4A8_DG has silu=<cfg>,
    names this w13 shape, m >= min_m, the down projection's input scale is static and the scales are alignment-1.
    Returns False (the caller runs the stock GEMM1 + silu kernel) otherwise."""
    dg = _nm_load_dg()
    if dg[2] is None or a2_scale is None or m < dg[0]:
        return False
    e, n, k = w13.shape[0], w13.shape[1], a.shape[-1]
    if (n, k) not in dg[1] or tuple(w13_scale.shape) != (e, k // 128, n) or expert_offsets.numel() != e + 1:
        return False
    torch.ops.k3amd_w4a8dg.mm_silu(out_q, a, w13, w13_scale, expert_offsets, a1_scale.reshape(1).float(),
                                   a2_scale.reshape(1).float(), dg[2])
    return True


def cutlass_w4a8_moe_mm(d, a, b, a_scales, b_scales, experts_offsets, problem_sizes, a_strides, b_strides,
                        d_strides, s_strides, chunk_size=128, topk=8):
    dg = _nm_load_dg()
    if dg[1]:
        n, k, m, e = d.shape[1], a.shape[-1], a.shape[0] // max(1, topk), b.shape[0]
        cfg = dg[1].get((n, k))
        # the kernel reads alignment-1 scales [E, K/128, N] (the tuned-tiles align hook) and [E+1] row offsets
        if cfg is not None and m >= dg[0] and tuple(b_scales.shape) == (e, k // 128, n):
            offs = _nm_expert_bounds(experts_offsets, e)
            if offs is not None:
                torch.ops.k3amd_w4a8dg.mm(d, a, b, b_scales, offs, a_scales.reshape(1).float(), cfg)
                return
    tri = _nm_load_triton()
    if tri[1]:
        n, k, m, e = d.shape[1], a.shape[-1], a.shape[0] // max(1, topk), b.shape[0]
        cfg = tri[1].get((n, k))
        # only alignment-1 scales ([E, K/128, N], the tuned-tiles align hook) are the kernel's scale layout
        if cfg is not None and m >= tri[0] and tuple(b_scales.shape) == (e, k // 128, n):
            offs = _nm_expert_bounds(experts_offsets, e)
            if offs is not None:
                from sglang.srt.layers.moe.glm_w4a8_triton_mm import w4a8_grouped_mm
                w4a8_grouped_mm(d, a, b, b_scales, a_scales.reshape(1), offs, block_n=cfg[0], block_t=cfg[1],
                                num_warps=cfg[2], num_stages=cfg[3])
                return
    rules = _nm_load_rules()
    if rules:
        n, k, m = d.shape[1], a.shape[-1], a.shape[0] // max(1, topk)
        for m_max, cfg in rules.get((n, k), []):
            if m <= m_max:
                if cfg < 0:          # -1: the stock dispatch (only valid where its TileK matches the rule's alignment)
                    break
                torch.ops.k3amd_w4a8.mm_cfg(d, a, b, a_scales, b_scales, experts_offsets, problem_sizes, a_strides,
                                            b_strides, d_strides, s_strides, chunk_size, cfg)
                return
    return _nm_stock_w4a8_mm(d, a, b, a_scales, b_scales, experts_offsets, problem_sizes, a_strides, b_strides,
                             d_strides, s_strides, chunk_size, topk)

'''


@dataclass(frozen=True)
class Replace:
    name: str
    path: str
    anchor: str
    text: str


HOOKS = [
    Replace(
        "dispatch", "srt/layers/moe/cutlass_w4a8_moe.py",
        "def cutlass_w4a8_moe(\n",
        DISPATCH + "\ndef cutlass_w4a8_moe(\n",
    ),
    Replace(
        "align", "srt/layers/quantization/w4afp8.py",
        "        w13_weight_scale = interleave_scales(w13_weight_scale)\n",
        "        import os as _nm_os  # {marker}\n"
        "\n"
        "        _nm_align = {}\n"
        "        for _p in filter(None, (x.strip() for x in _nm_os.environ.get(\"NUMINOUS_W4A8_TILES_ALIGN\", \"\").split(\";\"))):\n"
        "            _s, _a = _p.split(\"=\")\n"
        "            _nm_align[tuple(int(v) for v in _s.split(\"x\"))] = int(_a)\n"
        "\n"
        "        def _nm_interleave(s, n_out, k):\n"
        "            al = _nm_align.get((n_out, k))\n"
        "            if not al:\n"
        "                return interleave_scales(s)\n"
        "            e, n, g = s.shape\n"
        "            return s.reshape(e, n, g // al, al).permute(0, 2, 1, 3).reshape(e, g // al, n * al).contiguous()\n"
        "\n"
        "        w13_weight_scale = _nm_interleave(w13_weight_scale, w13_weight_scale.shape[1],\n"
        "                                          w13_weight_scale.shape[2] * 128)\n",
    ),
    Replace(
        "align2", "srt/layers/quantization/w4afp8.py",
        "        w2_weight_scale = interleave_scales(w2_weight_scale)\n",
        "        w2_weight_scale = _nm_interleave(w2_weight_scale, w2_weight_scale.shape[1],  # {marker}\n"
        "                                         w2_weight_scale.shape[2] * 128)\n",
    ),
]


SILU_ANCHOR = """    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)

    cutlass_w4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        a1_scale.float(),
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides13,
        128,
        topk,
    )

    intermediate_q = torch.empty(
        (m * topk, n), dtype=torch.float8_e4m3fn, device=device
    )

    if a2_scale is None:
        a2_scale = torch.zeros(1, dtype=torch.float32, device=device)
        silu_mul_dynamic_tensorwise_quant_for_cutlass_moe(
            c1, intermediate_q, a2_scale, expert_offsets[-1:], m * topk, n
        )
    else:
        silu_mul_static_tensorwise_quant_for_cutlass_moe(
            c1, intermediate_q, a2_scale.float(), expert_offsets[-1:], m * topk, n
        )
"""
SILU_TEXT = """    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)
    intermediate_q = torch.empty(
        (m * topk, n), dtype=torch.float8_e4m3fn, device=device
    )
    # {marker}-silu: GEMM1 with the silu*up -> FP8 epilogue (k3amd w4a8dg) when NUMINOUS_W4A8_DG has silu=<cfg>;
    # otherwise the stock GEMM1 into bf16 c1 and the silu-quant kernel (c1 is only allocated on that path)
    if not _nm_gemm1_silu(intermediate_q, gateup_input, w1_q, a1_scale, w1_scale, a2_scale, expert_offsets, m, topk):
        c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
        cutlass_w4a8_moe_mm(
            c1,
            gateup_input,
            w1_q,
            a1_scale.float(),
            w1_scale,
            expert_offsets[:-1],
            problem_sizes1,
            a_strides1,
            b_strides1,
            c_strides1,
            s_strides13,
            128,
            topk,
        )
        if a2_scale is None:
            a2_scale = torch.zeros(1, dtype=torch.float32, device=device)
            silu_mul_dynamic_tensorwise_quant_for_cutlass_moe(
                c1, intermediate_q, a2_scale, expert_offsets[-1:], m * topk, n
            )
        else:
            silu_mul_static_tensorwise_quant_for_cutlass_moe(
                c1, intermediate_q, a2_scale.float(), expert_offsets[-1:], m * topk, n
            )
"""
HOOKS.append(Replace("silu", "srt/layers/moe/cutlass_w4a8_moe.py", SILU_ANCHOR, SILU_TEXT))
SELFTEST_ANCHOR = "        layer.w2_weight_scale_inv = Parameter(w2_weight_scale, requires_grad=False)\n"
SELFTEST_TEXT = SELFTEST_ANCHOR + """        if "selftest=1" in _nm_os.environ.get("NUMINOUS_W4A8_DG", ""):  # {marker}
            # once per process, before the KV pool exists: w4a8dg exactness + autotune in a subprocess on this GPU
            try:
                from sglang.srt.layers.moe.numinous_w4a8dg_selftest import selftest_once as _nm_selftest

                _nm_selftest(layer)
            except Exception as _nm_e:  # never fail the weight load; no decision = the tiles path
                import sys as _nm_sys

                print("[numinous w4a8dg selftest] not run:", repr(_nm_e)[:300], file=_nm_sys.stderr)
"""
HOOKS.append(Replace("selftest", "srt/layers/quantization/w4afp8.py", SELFTEST_ANCHOR, SELFTEST_TEXT))
# the self-test module the 'selftest' hook imports (cells: hooks_block install=SELFTEST_INSTALL)
SELFTEST_MODULE = ("../kernels/w4a8dg/numinous_w4a8dg_selftest.py", "srt/layers/moe/numinous_w4a8dg_selftest.py")
SELFTEST_INSTALL = [("operations/kernels/w4a8dg/numinous_w4a8dg_selftest.py", SELFTEST_MODULE[1])]

TRITON_MODULE = ("../kernels/glm_w4a8_triton_mm.py", "srt/layers/moe/glm_w4a8_triton_mm.py")


def apply(root: str, check: bool = False) -> list[str]:
    out, missing, files = [], [], {}
    for rel, dst in (TRITON_MODULE, SELFTEST_MODULE):
        src = os.path.normpath(os.path.join(HERE, rel))
        if not check and os.path.exists(src):    # next to the patcher in a checkout; cells install via hooks_block
            import shutil
            shutil.copy(src, os.path.join(root, dst))
            out.append(f"installed {dst}")
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = files.get(path)
        if src is None:
            src = files[path] = open(path).read()
        tag = f"{MARKER} ({h.name})"
        if tag in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        files[path] = src.replace(h.anchor, h.text.replace("{marker}", tag), 1)
        out.append(f"{h.name}: {'anchor ok' if check else 'applied to ' + h.path}")
    if missing:
        raise SystemExit("glm_w4a8_tiles hooks: " + "; ".join(missing))
    if not check:
        for path, new in files.items():
            if new == open(path).read():
                continue
            compile(new, path, "exec")
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
    return out


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
