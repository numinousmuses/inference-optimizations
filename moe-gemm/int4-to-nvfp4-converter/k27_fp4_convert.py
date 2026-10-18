"""Kimi K2.7 Code: INT4 (QAT, compressed-tensors pack-quantized, symmetric, group 32) routed experts -> FP4.

Two targets, both readable by SGLang v0.5.20 for a DeepseekV3ForCausalLM checkpoint (source_code):
  nvfp4   modelopt NVFP4 layout (hf_quant_config.json quant_algo NVFP4; ModelOptNvFp4FusedMoEMethod, flashinfer
          TRT-LLM FP4 block-scale MoE = W4A4): per expert projection `weight` uint8 [out, in/2] (two E2M1 nibbles,
          low nibble first), `weight_scale` E4M3 [out, in/16], `weight_scale_2` fp32 (global), `input_scale` fp32
          (activation global scale; taken from a calibrated checkpoint, see --input-scales-from)
  mxfp4   E2M1 + E8M0 per 32 (`weight` uint8 [out, in/2], `weight_scale` uint8 E8M0 [out, in/32]); the DeepGEMM
          MegaMoE mxf4xmxf4 / flashinfer_mxfp4 weight format (the DeepSeek V4 Pro W4A4 lead)

Why a converter and not a requantizer from bf16: the QAT model's weights ARE the INT4 grid, w = q * s with q in
[-8, 7] and s the bf16 group scale. E2M1 has 15 levels {0, +-0.5, 1, 1.5, 2, 3, 4, 6}, so a block of 16 (NVFP4) or
32 (MXFP4) INT4 codes fits an FP4 block exactly only when its codes are a subset of a scaled E2M1 grid; otherwise the
block scale is chosen per block by mean-squared-error search over candidates (amax/6 x {1 .. 7/6} on a fine grid for
NVFP4, the two powers of two around amax/6 for MXFP4), so the FP4 block is the MSE-closest to the INT4 reference.
`verify` reports each candidate's error against the INT4 reference per projection (relative RMS in weight space and in
output space for Gaussian activations), next to a third-party NVFP4 checkpoint's (decart-ai/Kimi-K2.7-Code-NVFP4)
error against the same reference. Plain torch (CPU or --device cuda; the host CPUs are shared and slow, the dev box GPU does a projection in
milliseconds); the full conversion streams shard by shard.

    python3 operations/kernels/k27_fp4_convert.py verify --int4 <shard> --nvfp4 <shard> [<shard>] --layer 3 --experts 0-7
    python3 operations/kernels/k27_fp4_convert.py convert --src /cache/moonshotai__Kimi-K2.7-Code --fmt nvfp4 \
        --dst /cache/numinous__Kimi-K2.7-Code-NVFP4 --input-scales-from /cache/decart-ai__Kimi-K2.7-Code-NVFP4
Tests: operations/kernels/test_k27_fp4_convert.py.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import time

import torch

E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# midpoints between consecutive E2M1 magnitudes: round-to-nearest (ties to even code, as the hardware converts)
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
GROUP_INT4 = 32
BLOCK_NVFP4 = 16
BLOCK_MXFP4 = 32
E4M3_MAX = 448.0
EXPERT_RE = re.compile(r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj))\.(?P<leaf>weight_packed|weight_scale|weight_shape)$")


# ---- INT4 reference -------------------------------------------------------------------------------------------
def unpack_int4(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """compressed-tensors pack-quantized int32 [out, in/8] -> int8 codes [out, in] in [-8, 7] (nibble i of a word is
    column 8k+i; codes stored with the +8 offset; sglang gptq_kernels.unpack_from_int32 semantics)."""
    assert packed.dtype == torch.int32
    out, cols = shape
    u = torch.empty((packed.shape[0], packed.shape[1] * 8), dtype=torch.int32, device=packed.device)
    for i in range(8):
        u[:, i::8] = (packed >> (4 * i)) & 0xF
    return (u[:, :cols] - 8).to(torch.int8)


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """fp32 [out, in] = code * group scale (bf16 [out, in/32])."""
    q = unpack_int4(packed, shape).float()
    s = scale.float().repeat_interleave(GROUP_INT4, dim=1)[:, : shape[1]]
    return q * s


# ---- E2M1 codes -----------------------------------------------------------------------------------------------
_DEV_CACHE: dict = {}


def _tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    if key not in _DEV_CACHE:
        _DEV_CACHE[key] = (E2M1_LEVELS.to(device), _E2M1_MID.to(device))
    return _DEV_CACHE[key]


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Round |x| to the nearest E2M1 magnitude (ties to even code), return 4-bit codes (sign in bit 3) as uint8."""
    _, mid = _tables(x.device)
    mag = x.abs().clamp(max=6.0)
    idx = torch.bucketize(mag, mid)                           # level index: mid[idx-1] < mag <= mid[idx]
    # an exact midpoint lands on the lower level; ties-to-even moves it up when that lower code is odd
    tie = (mag == mid[idx.clamp(max=6)]) & (idx < 7) & (idx % 2 == 1)
    idx = torch.where(tie, idx + 1, idx)
    sign = (x < 0) & (idx > 0)
    return (idx + sign.to(torch.int64) * 8).to(torch.uint8)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    levels, _ = _tables(codes.device)
    mag = levels[(codes & 7).long()]
    return torch.where(codes >= 8, -mag, mag)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """[out, in] 4-bit codes -> uint8 [out, in/2], low nibble = even column (TRT-LLM / modelopt layout)."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    out = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = packed & 0xF
    out[:, 1::2] = packed >> 4
    return out


# ---- scales ---------------------------------------------------------------------------------------------------
def to_e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)


def to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """Power-of-two scale as the E8M0 byte (127 = 1.0); x must already be a power of two > 0."""
    return (torch.log2(x).round() + 127).clamp(0, 254).to(torch.uint8)


def from_e8m0(b: torch.Tensor) -> torch.Tensor:
    return torch.exp2(b.float() - 127.0)


def _blocks(w: torch.Tensor, block: int) -> torch.Tensor:
    out, cols = w.shape
    assert cols % block == 0, (w.shape, block)
    return w.reshape(out, cols // block, block)


def _best_scale(blocks: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
    """Per block, the candidate scale (already representable in the target scale format) with the least squared
    error after E2M1 rounding. blocks [out, nb, B], cands [out, nb, C] -> [out, nb]."""
    err = torch.empty(cands.shape, dtype=torch.float32, device=cands.device)
    for c in range(cands.shape[-1]):
        s = cands[..., c].unsqueeze(-1)
        rec = e2m1_decode(e2m1_encode(blocks / s)) * s
        err[..., c] = ((rec - blocks) ** 2).sum(-1)
    return torch.gather(cands, -1, err.argmin(-1, keepdim=True)).squeeze(-1)


def quant_nvfp4(w: torch.Tensor, search: bool = True, global_scale: float | None = None):
    """fp32 [out, in] -> (weight uint8 [out, in/2], weight_scale E4M3 [out, in/16], weight_scale_2 fp32 scalar).
    Dequant: e2m1(code) * float(weight_scale) * weight_scale_2. With search=False the block scale is amax/6
    (modelopt's rule); with search it is the MSE-best of amax/6 x linspace(1, 7/6, 12) per block."""
    blocks = _blocks(w, BLOCK_NVFP4)
    amax = blocks.abs().amax(-1)
    g = float(global_scale) if global_scale else float(w.abs().max()) / (6.0 * E4M3_MAX)
    g = max(g, 1e-12)
    base = amax / 6.0
    if search:
        factors = torch.linspace(1.0, 7.0 / 6.0, 8, device=w.device)
        cands = base.unsqueeze(-1) * factors
    else:
        cands = base.unsqueeze(-1)
    # candidates must be what the E4M3 scale actually stores
    cands = to_e4m3(cands / g).float() * g
    cands = torch.where(cands > 0, cands, torch.full_like(cands, g))   # all-zero blocks: any scale, keep codes 0
    s = _best_scale(blocks, cands) if search else cands[..., 0]
    codes = e2m1_encode(blocks / s.unsqueeze(-1)).reshape(w.shape)
    return pack_nibbles(codes), to_e4m3(s / g), torch.tensor(g, dtype=torch.float32, device=w.device)


def dequant_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor) -> torch.Tensor:
    codes = unpack_nibbles(weight)
    s = (weight_scale.float() * weight_scale_2.float()).repeat_interleave(BLOCK_NVFP4, dim=1)
    return e2m1_decode(codes) * s


def quant_mxfp4(w: torch.Tensor, search: bool = True):
    """fp32 [out, in] -> (weight uint8 [out, in/2], weight_scale E8M0 uint8 [out, in/32]). Block scale = the power of
    two >= amax/6 (no search) or the MSE-best of {that/2, that, that*2} (search)."""
    blocks = _blocks(w, BLOCK_MXFP4)
    amax = blocks.abs().amax(-1).clamp(min=1e-30)
    p = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    if search:
        cands = torch.stack([p / 2, p, p * 2], -1)
        s = _best_scale(blocks, cands)
    else:
        s = p
    codes = e2m1_encode(blocks / s.unsqueeze(-1)).reshape(w.shape)
    return pack_nibbles(codes), to_e8m0(s)


def dequant_mxfp4(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    codes = unpack_nibbles(weight)
    s = from_e8m0(weight_scale).repeat_interleave(BLOCK_MXFP4, dim=1)
    return e2m1_decode(codes) * s


# ---- verification ---------------------------------------------------------------------------------------------
def compare(ref: torch.Tensor, cand: torch.Tensor, n_act: int = 256, seed: int = 0) -> dict:
    """Relative RMS error in weight space, in output space (x W^T for Gaussian x), max abs error, and the fraction of
    weights that moved."""
    d = cand - ref
    g = torch.Generator(device=ref.device).manual_seed(seed)
    x = torch.randn((n_act, ref.shape[1]), generator=g, device=ref.device)
    y_ref, y_d = x @ ref.T, x @ d.T
    return {"rel_rms_w": float(d.norm() / ref.norm()), "rel_rms_out": float(y_d.norm() / y_ref.norm()),
            "max_abs": float(d.abs().max()), "changed_frac": float((d != 0).float().mean())}


def expert_tensors(shard_paths: list[str], layer: int, expert: int) -> dict:
    """{proj: {leaf: tensor}} for one expert across the given safetensors shards (whatever leaves exist)."""
    from safetensors import safe_open
    want = f".layers.{layer}.mlp.experts.{expert}."
    out: dict = {}
    for p in shard_paths:
        with safe_open(p, "pt") as f:
            for k in f.keys():
                if want in k:
                    proj, leaf = k.split(want)[1].split(".", 1)
                    out.setdefault(proj, {})[leaf] = f.get_tensor(k)
    return out


def _to(groups: dict, device: str) -> dict:
    return {p: {leaf: t.to(device) for leaf, t in d.items()} for p, d in groups.items()}


def verify(int4_shards: list[str], nvfp4_shards: list[str], layer: int, experts: list[int], device: str = "cpu") -> dict:
    rows = []
    for e in experts:
        ref_t = _to(expert_tensors(int4_shards, layer, e), device)
        cand_t = _to(expert_tensors(nvfp4_shards, layer, e), device) if nvfp4_shards else {}
        for proj, t in ref_t.items():
            shape = tuple(int(v) for v in t["weight_shape"])
            ref = dequant_int4(t["weight_packed"], t["weight_scale"], shape)
            s_int4 = t["weight_scale"].float().repeat_interleave(GROUP_INT4, dim=1)[:, : shape[1]]
            row = {"layer": layer, "expert": e, "proj": proj, "shape": shape,
                   # the INT4 grid's own step relative to the weights: a uniform rounding error of one step is
                   # step/sqrt(12) rms, so this is the scale any FP4 error should be read against (derived)
                   "int4_step_rel_rms": float((s_int4 / 12 ** 0.5).norm() / ref.norm())}
            if proj in cand_t and "weight_scale_2" in cand_t[proj]:
                c = cand_t[proj]
                row["decart_nvfp4"] = compare(ref, dequant_nvfp4(c["weight"], c["weight_scale"], c["weight_scale_2"]))
                row["decart_input_scale"] = float(c["input_scale"]) if "input_scale" in c else None
            w, sc, g = quant_nvfp4(ref, search=False)
            row["ours_nvfp4_amax"] = compare(ref, dequant_nvfp4(w, sc, g))
            w, sc, g = quant_nvfp4(ref, search=True)
            row["ours_nvfp4_search"] = compare(ref, dequant_nvfp4(w, sc, g))
            w, sc = quant_mxfp4(ref, search=False)
            row["ours_mxfp4_pow2"] = compare(ref, dequant_mxfp4(w, sc))
            w, sc = quant_mxfp4(ref, search=True)
            row["ours_mxfp4_search"] = compare(ref, dequant_mxfp4(w, sc))
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {"layer": layer, "experts": experts, "rows": rows}


# ---- full conversion ------------------------------------------------------------------------------------------
def _load_input_scales(src_dir: str) -> dict[str, torch.Tensor]:
    """{'<prefix>.input_scale': tensor} for every expert projection of a calibrated modelopt checkpoint."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, sh in idx.items():
        if k.endswith(".input_scale") and ".mlp.experts." in k:
            by_shard.setdefault(sh, []).append(k)
    out = {}
    for sh, keys in by_shard.items():
        with safe_open(os.path.join(src_dir, sh), "pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    return out


def convert_shard(src: str, dst: str, shard: str, fmt: str, search: bool = True,
                  in_scales: dict[str, torch.Tensor] | None = None, device: str = "cpu") -> tuple[dict[str, str], dict]:
    """One safetensors shard of src -> the same file name in dst: expert INT4 tensors become FP4 tensors, everything
    else is copied. Returns (key -> shard for the new index, stats). Independent per shard, so shards convert in
    parallel (operations/modal/k27_fp4_build.py)."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(dst, exist_ok=True)
    in_scales = in_scales or {}
    t0 = time.time()
    out_t: dict[str, torch.Tensor] = {}
    groups: dict[str, dict] = {}
    with safe_open(os.path.join(src, shard), "pt") as f:
        for k in f.keys():
            m = EXPERT_RE.match(k)
            if not m:
                out_t[k] = f.get_tensor(k)
                continue
            groups.setdefault(m["prefix"], {})[m["leaf"]] = f.get_tensor(k)
    err = 0.0
    for prefix, t in groups.items():
        shape = tuple(int(v) for v in t["weight_shape"])
        ref = dequant_int4(t["weight_packed"].to(device), t["weight_scale"].to(device), shape)
        if fmt == "nvfp4":
            w, sc, g = quant_nvfp4(ref, search=search)
            err += compare(ref, dequant_nvfp4(w, sc, g), n_act=8)["rel_rms_w"]
            out_t[prefix + ".weight"], out_t[prefix + ".weight_scale"], out_t[prefix + ".weight_scale_2"] = w.cpu(), sc.cpu(), g.cpu()
            key = prefix + ".input_scale"
            if key in in_scales:
                out_t[key] = in_scales[key].to(torch.float32).reshape(())
        else:
            w, sc = quant_mxfp4(ref, search=search)
            err += compare(ref, dequant_mxfp4(w, sc), n_act=8)["rel_rms_w"]
            out_t[prefix + ".weight"], out_t[prefix + ".weight_scale"] = w.cpu(), sc.cpu()
    save_file(out_t, os.path.join(dst, shard), metadata={"format": "pt"})
    stats = {"shard": shard, "experts": len(groups), "rel_rms_w_sum": err, "minutes": (time.time() - t0) / 60}
    return {k: shard for k in out_t}, stats


def finalize(src: str, dst: str, fmt: str, new_map: dict[str, str], stats: dict) -> None:
    """Index, config (+ hf_quant_config.json for nvfp4), tokenizer and template files, conversion record."""
    total = sum(os.path.getsize(os.path.join(dst, s)) for s in set(new_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or fn.startswith("."):
            continue
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            shutil.copy(p, os.path.join(dst, fn))
    write_config(src, dst, fmt)
    json.dump(stats, open(os.path.join(dst, "numinous_convert.json"), "w"), indent=1)


def shards_of(src: str, layers: str = "") -> list[str]:
    """The shards to convert: all, or only those holding experts of the given layers ("3,4")."""
    wmap = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    if not layers:
        return sorted(set(wmap.values()))
    only = {int(x) for x in layers.split(",") if x}
    return sorted({sh for k, sh in wmap.items() if (m := EXPERT_RE.match(k)) and int(m["layer"]) in only})


def convert(src: str, dst: str, fmt: str, search: bool = True, input_scales_from: str = "", layers: str = "",
            device: str = "cpu") -> dict:
    """Sequential conversion of src into dst (every shard, or --layers "3,4" for a smoke subset that is not loadable):
    fmt nvfp4 writes hf_quant_config.json (modelopt NVFP4, KV FP8) and strips the compressed-tensors
    quantization_config; fmt mxfp4 writes a quantization_config of quant_method mxfp4 over the experts."""
    in_scales = _load_input_scales(input_scales_from) if (fmt == "nvfp4" and input_scales_from) else {}
    t0 = time.time()
    new_map: dict[str, str] = {}
    stats = {"experts": 0, "shards": 0, "rel_rms_w_sum": 0.0}
    for sh in shards_of(src, layers):
        keys, st = convert_shard(src, dst, sh, fmt, search, in_scales, device)
        new_map.update(keys)
        stats["experts"] += st["experts"]
        stats["shards"] += 1
        stats["rel_rms_w_sum"] += st["rel_rms_w_sum"]
        print(f"[{fmt}] {sh}: {st['experts']} expert projections, {stats['experts']} total, {(time.time() - t0) / 60:.1f} min", flush=True)
    stats["mean_rel_rms_w"] = stats["rel_rms_w_sum"] / max(stats["experts"], 1)
    stats["minutes"] = (time.time() - t0) / 60
    finalize(src, dst, fmt, new_map, stats)
    return stats


def write_config(src: str, dst: str, fmt: str) -> None:
    cfg = json.load(open(os.path.join(src, "config.json")))
    text = cfg.get("text_config", cfg)
    n_layers = text["num_hidden_layers"]
    first_dense = text.get("first_k_dense_replace", 1)
    text.pop("quantization_config", None)
    cfg.pop("quantization_config", None)
    if fmt == "nvfp4":
        # modelopt convention: everything not an expert stays bf16 (attention, shared experts, dense MLP, gate, head)
        exclude = ["language_model.lm_head"] + [f"language_model.model.layers.{i}*" for i in range(first_dense)]
        for i in range(first_dense, n_layers):
            exclude += [f"language_model.model.layers.{i}.mlp.shared_experts*", f"language_model.model.layers.{i}.mlp.gate",
                        f"language_model.model.layers.{i}.self_attn*"]
        json.dump({"producer": {"name": "numinous k27_fp4_convert", "version": "1"},
                   "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": BLOCK_NVFP4,
                                    "exclude_modules": exclude}},
                  open(os.path.join(dst, "hf_quant_config.json"), "w"), indent=4)
        # the top-level block modelopt exports next to hf_quant_config.json (decart / nvidia Kimi NVFP4 checkpoints
        # carry it; loaders that read config.json first see quant_method modelopt and the same ignore list)
        cfg["quantization_config"] = {
            "config_groups": {"group_0": {
                "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": BLOCK_NVFP4},
                "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": BLOCK_NVFP4},
                "targets": ["Linear"]}},
            "ignore": exclude, "quant_algo": "NVFP4",
            "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
            "producer": {"name": "modelopt", "version": "numinous k27_fp4_convert 1"}, "quant_method": "modelopt"}
    else:
        text["quantization_config"] = {"quant_method": "mxfp4", "modules_to_not_convert": [
            "self_attn", "shared_experts", "lm_head", "mlp.gate"] + [f"layers.{i}.mlp" for i in range(first_dense)]}
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)


def _parse_experts(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--int4", nargs="+", required=True)
    v.add_argument("--nvfp4", nargs="*", default=[])
    v.add_argument("--layer", type=int, default=3)
    v.add_argument("--experts", default="0-3")
    v.add_argument("--out", default="")
    v.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c = sub.add_parser("convert")
    c.add_argument("--src", required=True)
    c.add_argument("--dst", required=True)
    c.add_argument("--fmt", choices=["nvfp4", "mxfp4"], default="nvfp4")
    c.add_argument("--no-search", action="store_true")
    c.add_argument("--input-scales-from", default="")
    c.add_argument("--layers", default="")
    c.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.cmd == "verify":
        res = verify(a.int4, a.nvfp4, a.layer, _parse_experts(a.experts), a.device)
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    else:
        print(json.dumps(convert(a.src, a.dst, a.fmt, not a.no_search, a.input_scales_from, a.layers, a.device), indent=1))


if __name__ == "__main__":
    main()
