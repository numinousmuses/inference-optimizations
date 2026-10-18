"""Dispatch AITER's tuned CK-Tile stage-1 + OPUS stage-2 MXFP4 a8w4 MoE rows at inter_dim % 256 != 0
(k3amd for amdpro, 2026-10-03). Target: DeepSeek-V4 Pro on 8x MI355X, MoE TP8 slice inter_dim 384 (385 experts incl.
the fused shared one, top-7), AITER e7d2453f in the 20261002 image root.

Measured problem (amdpro's server logs): every prefill-sized MoE call (T = 2048..32768) logs "no tuned FlyDSL config
... using heuristic FlyDSL fallback" although AITER's dsv4_fp8fp4_tuned_fmoe.csv has those rows:
stage 1 = cktile_a8w4_bm64, stage 2 = opus_moe2_afp8_wfp4_fp8_*. get_2stage_cfgs only builds tuned metadata when stage
1 is flydsl_/opus_; a cktile_ stage 1 falls through to the generic CK-Tile route (CK-Tile for BOTH stages), which is
disabled at inter_dim % 256 != 0 because CK-Tile *stage 2* mis-indexes e8m0 scale groups there. AITER's own comment:
stage 1 "is unaffected -- it reduces over model_dim". The fallback costs ~657 ms per 65K-token forward against the
tuned rows' ~350 ms (amdpro).

install() wraps aiter.fused_moe.get_2stage_cfgs: when the call is MXFP4-weight / FP8-activation, inter_dim is not a
multiple of 256, and the tuned row is (cktile_* stage 1, OPUS a8w4 stage 2), it returns metadata with stage 1 =
cktile_moe_stage1 (CK-Tile, safe) and stage 2 = the OPUS a8w4 stage-2 wrapper. The stage-1 output is bf16 and goes
through fused_moe's standard a8w4 inter-quant (fused_dynamic_mxfp8_quant_moe_sort) into the OPUS FP8 operand, the same
hand-off fused_moe uses for every bf16 stage 1 (fuse_quant ""). Everything else returns the original metadata.

Inert unless NUMINOUS_AITER_CKTILE_S1_OPUS_S2=1. Correctness is checked on the GPU by
operations/amdflash/box/pro_moe_dispatch_check.py (AITER's own test_fmoe reference at Pro shapes) before any arm.
"""
from __future__ import annotations

import functools
import os

ENV = "NUMINOUS_AITER_CKTILE_S1_OPUS_S2"
_STATE = {"installed": False, "hits": 0}


def _lookup(fm, key_args: dict):
    """The tuned row for these call args, read from the same parsed table get_2stage_cfgs filled (primary dict)."""
    c2s = getattr(fm, "cfg_2stages", None)
    if not c2s:
        return None
    primary = c2s[0]
    from aiter.jit.utils.chip_info import get_cu_num
    gfx = fm.get_gfx_runtime()
    keys = (gfx, get_cu_num(), key_args["token"], key_args["model_dim"], key_args["inter_dim"], key_args["expert"],
            key_args["topk"], str(key_args["activation"]), str(key_args["dtype"]), str(key_args["q_dtype_a"]),
            str(key_args["q_dtype_w"]), str(key_args["q_type"]), key_args["use_g1u1"], key_args["doweight_stage1"])
    row = primary.get(keys)
    if row is None and key_args["token"] > 32768:   # upstream's tier fallback: 131072 -> 32768
        row = primary.get(keys[:2] + (32768,) + keys[3:])
    return row


def build_metadata(fm, cfg: dict, *, activation, dtype, use_g1u1, hidden_pad, intermediate_pad):
    kn1 = str(cfg.get("kernelName1", "") or "").strip()
    kn2 = str(cfg.get("kernelName2", "") or "").strip()
    if not kn1.startswith("cktile_") or not fm._opus_a8w4.is_opus_a8w4_stage2_kernel(kn2):
        return None
    block_m = int(cfg.get("block_m", 0) or 0) or int(kn1.rsplit("bm", 1)[-1])
    stage1 = functools.partial(
        fm.cktile_moe_stage1,
        n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
        k_pad_zeros=hidden_pad // 128 * 128,
        activation=activation,
        split_k=1,
        dtype=dtype,
    )
    stage2 = functools.partial(
        fm._opus_a8w4.opus_a8w4_stage2_wrapper,
        kernelName=kn2,
        inter_dim_pad=intermediate_pad,
        model_dim_pad=hidden_pad,
        launch=fm._opus_a8w4.parse_stage2_config(cfg, block_m),
    )
    return fm.MOEMetadata(stage1, stage2, block_m, int(cfg.get("ksplit", 0) or 0), run_1stage=False,
                          has_bias=True, stage2_has_bias=False, fuse_quant="")


def install() -> bool:
    if os.environ.get(ENV) != "1" or _STATE["installed"]:
        return False
    import torch  # noqa: F401
    import aiter.fused_moe as fm
    from aiter import dtypes

    orig = fm.get_2stage_cfgs
    names = ["token", "model_dim", "inter_dim", "expert", "topk", "dtype", "q_dtype_a", "q_dtype_w", "q_type",
             "use_g1u1", "activation", "doweight_stage1", "hidden_pad", "intermediate_pad"]

    @functools.lru_cache(maxsize=2048)
    def get_2stage_cfgs(*args, **kwargs):
        md = orig(*args, **kwargs)
        a = dict(zip(names, args))
        a.update({k: v for k, v in kwargs.items() if k in names})
        try:
            if not (a["q_dtype_w"] == dtypes.fp4x2 and a["q_dtype_a"] == dtypes.fp8 and a["inter_dim"] % 256 != 0):
                return md
            if kwargs.get("is_ep") or kwargs.get("config_file"):
                return md
            cfg = _lookup(fm, a)
            if cfg is None:
                return md
            new = build_metadata(fm, cfg, activation=a["activation"], dtype=a["dtype"], use_g1u1=a["use_g1u1"],
                                 hidden_pad=a["hidden_pad"], intermediate_pad=a["intermediate_pad"])
        except Exception as ex:  # never break serving: keep upstream's (safe) metadata
            fm.logger.warning(f"[numinous] cktile-s1/opus-s2 dispatch skipped: {ex!r}")
            return md
        if new is None:
            return md
        _STATE["hits"] += 1
        fm.logger.info(f"[numinous] tuned cktile stage1 + opus stage2 for token={a['token']} inter={a['inter_dim']}")
        return new

    get_2stage_cfgs.cache_clear_orig = orig.cache_clear
    fm.get_2stage_cfgs = get_2stage_cfgs
    _STATE["installed"] = True
    return True
