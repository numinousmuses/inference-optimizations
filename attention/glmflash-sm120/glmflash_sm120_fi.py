"""FlashInfer's SM120 sparse MLA kernels serving GLM-5.3-Flash (rope-free 512 latent, kpool top-k 2051).

What FlashInfer 0.6.18 ships for SM120 (source_code, flashinfer/data/include/flashinfer/attention/sparse_mla_sm120/,
csrc/sparse_mla_sm120_*.cu): warp-specialized decode (split-K + merge) and prefill kernels for three row types.
`GLM_NSA` = d_qk 576 (512 nope + 64 rope), 656-byte inline rows [512 e4m3 | 4 fp32 arbitrary scales | 64 bf16
rope], d_v 512, FP8 compute (Q quantized on the fly, block-scaled MXFP8 mma.sync, P quantized to FP8 for PV).

How Flash fits it without touching a kernel (derived):
  * rope-free latent = GLM_NSA with a zero rope: q576 = [q512 | 0], row rope bytes 0 -> the rope MMA adds exactly 0
    and V is the 512 nope part, which is Flash's whole latent. glmflash_sm120_attn.FMT_FP8_656 is that row.
  * width: Flash's kpool table is 2051 columns (2048 pooled + up to 3 tail, compact, -1 padded). The kernels are
    TOPK-generic C++ templates (NI = TOPK / 64, loops over runtime tile counts; odd NI already instantiated at
    topk 192), but the dispatch tables stop at 2048. `jit` mode builds FlashInfer's own five source files with the
    GLM_NSA dispatch widened to TOPK = 2112 (= 33 x 64) under a separate module name, pads the table to 2112 with -1
    (the kernels mask idx < 0 to -1e30 before softmax) and makes one call. `stock` mode uses the installed module's
    top-2048 kernels on columns [0, 2048) plus our Triton kernel on the tail columns and an LSE merge (both kernels'
    LSE are log2-domain).
  * -1 entries: the kernels gather slot 0 for them and zero them in softmax, but P = 0 times a NaN row is NaN, so
    slot 0 must stay finite: the 656 writer (glmflash_sm120_attn.nsa656_quant_write) never writes loc 0 and never
    emits a NaN code (the stock writer does for an all-zero 128-block).

Numbers (derived): 656 B/token/layer = 64% of bf16's 1024 (pool x1.56), 2.28x the 288-B FP4 row.

CPU tests: test_glmflash_sm120_fi.py (source patches against the 0.6.18 wheel's csrc, the call plumbing with the
FlashInfer module replaced by a torch emulation, padding / split / merge math). GPU: test_glmflash_sm120_gpu.py.
"""
from __future__ import annotations

import logging
import os
import pathlib
import re
from typing import Optional

import torch
import torch.nn.functional as F

try:  # installed next to this module inside sglang/srt/layers/attention/dsa/ by rtxflash_attn_hooks.py
    from . import glmflash_sm120_attn as A
except ImportError:
    import glmflash_sm120_attn as A

logger = logging.getLogger("sglang.numinous")

MODEL_TYPE_DSV3_2 = 0   # ModelType::DSV3_2: 656-B rows with power-of-two fp32 scales (UE8M0-exact)
MODEL_TYPE_GLM_NSA = 2  # flashinfer.mla._sparse_mla_sm120._MODEL_TYPE_GLM_NSA / ModelType::GLM_NSA (arbitrary scales)
D_QK = 576
TOPK_STOCK = 2048
TOPK_JIT = 2112
DECODE_MAX_TOKENS = 64  # flashinfer _DECODE_MAX_TOKENS: T <= 64 -> split-K decode kernel, else prefill kernel
# rtxsys 2026-10-02 (decode-cycle work; box 6b58cae2 profile: sparse_mla_decode 0.25-1.0 ms/call at T~40, latency
# bound): env knobs for the split-K decode launch, swept by operations/kernels/rtxflash_mla_decode_bench.py.
#   NUMINOUS_RTXFLASH_DEC_CPB    chunks_per_block override (-1 = the kernel's wave heuristic, the default)
#   NUMINOUS_RTXFLASH_DEC_MAX_T  largest T served by the split-K decode kernel (default 64; v3's 6-token verify puts
#                                10+ requests per rank above 64, into the prefill kernel)
#   NUMINOUS_RTXFLASH_DEC_CPB_TABLE  "maxT:cpb,..." per-T override (first entry with T <= maxT; else DEC_CPB)
#   NUMINOUS_RTXFLASH_DEC_PAD_MIN_T  T in (this, 64] is padded to 65 rows and run by the prefill kernel (0 = off)
# Measured on box 5e89aa43 (1 x RTX PRO 6000, H=64, top-k 2112, CUDA-graph replay; evidence mla_decode_bench.json):
# split-K decode at T=48/60/64 takes 242/244/247 us with the heuristic, 185/195/207 us at its best cpb, while the
# prefill kernel takes ~128 us at T=72-80 (and 254 us at 96-128 vs 323-406 for split-K): above ~40 rows the
# prefill kernel is the faster one, and the stock 64-row cutoff sends 41-64-row batches to the slow path.
DEC_CPB = int(os.environ.get("NUMINOUS_RTXFLASH_DEC_CPB", "-1"))
DEC_MAX_T = int(os.environ.get("NUMINOUS_RTXFLASH_DEC_MAX_T", str(DECODE_MAX_TOKENS)))
DEC_PAD_MIN_T = int(os.environ.get("NUMINOUS_RTXFLASH_DEC_PAD_MIN_T", "0"))


def parse_cpb_table(raw: str) -> list:
    out = []
    for item in (x for x in raw.split(",") if x.strip()):
        t, c = item.split(":")
        out.append((int(t), int(c)))
    return sorted(out)


DEC_CPB_TABLE = parse_cpb_table(os.environ.get("NUMINOUS_RTXFLASH_DEC_CPB_TABLE", ""))


def cpb_for(t: int) -> int:
    for max_t, cpb in DEC_CPB_TABLE:
        if t <= max_t:
            return cpb
    return DEC_CPB
BI = 64
HEADS = (8, 16, 32, 64, 128)
MODULE_NAME = "numinous_glm5next_sparse_mla_sm120_tk2112_v2"  # v2: trimmed build carries DSV3_2 + GLM_NSA
SOURCES = ("sparse_mla_sm120.cu", "sparse_mla_sm120_decode_dsv3_2.cu", "sparse_mla_sm120_decode_dsv4.cu",
           "sparse_mla_sm120_prefill.cu", "sparse_mla_sm120_jit_binding.cu")


# ------------------------------------------------------------------------------------------------ source patches


class PatchError(RuntimeError):
    pass


def _sub_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise PatchError(f"{what}: anchor found {n} times")
    return text.replace(old, new, 1)


_DECODE_LINE = re.compile(r"^  DSV3_2_DISPATCH\((\d+), (\d+)\)\n", re.M)
_DSV4_LINE = re.compile(r"^  DSV4_DISPATCH\((\d+), (\d+)\)\n", re.M)


def patch_decode_dsv3_2(text: str, trim: bool) -> str:
    lines = _DECODE_LINE.findall(text)
    if len(lines) != 20 or ("128", "2048") not in lines:
        raise PatchError(f"decode_dsv3_2: expected the 20-entry 0.6.18 dispatch table, found {lines}")
    block_start = text.index("  DSV3_2_DISPATCH(8, 128)\n")
    block_end = text.index("#undef DSV3_2_DISPATCH\n")
    if trim:  # DSV3_2 (pow2 rows) and GLM_NSA (arbitrary rows), top-k 2048 and 2112
        new = "".join(f"  if (mt == ModelType::{m}) {{ DSV3_2_DISPATCH_MT(ModelType::{m}, {h}, {k}) }}\n"
                      for m in ("DSV3_2", "GLM_NSA") for h in HEADS for k in (TOPK_STOCK, TOPK_JIT))
    else:
        new = text[block_start:block_end] + "".join(f"  DSV3_2_DISPATCH({h}, {TOPK_JIT})\n" for h in HEADS)
    return text[:block_start] + new + text[block_end:]


def patch_decode_dsv4(text: str, trim: bool) -> str:
    n = len(_DSV4_LINE.findall(text))
    if n != 25:
        raise PatchError(f"decode_dsv4: expected 25 DSV4_DISPATCH entries, found {n}")
    return _DSV4_LINE.sub("", text) if trim else text


def patch_prefill(text: str, trim: bool) -> str:
    start = text.index("template <ModelType MT>\ninline bool dispatch_v32(")
    end = text.index("#undef DISPATCH_DSV3_2_MG\n}\n", start) + len("#undef DISPATCH_DSV3_2_MG\n}\n")
    body = text[start:end]
    if body.count("2048") != 4:
        raise PatchError(f"prefill dispatch_v32: expected 4 literal 2048s, found {body.count('2048')}")
    tk = body.replace("template <ModelType MT>\ninline bool dispatch_v32(",
                      "template <ModelType MT, int TK>\ninline bool dispatch_v32_tk(").replace("2048", "TK")
    call = (f"  if (topk == {TOPK_JIT})\n    return dispatch_v32_tk<MT, {TOPK_JIT}>(num_heads, topk, Q, KV, indices, "
            "attn_sink, output, out_lse, sm_scale,\n                                        num_tokens, "
            "stride_kv_block, topk_length_ptr, stream);\n")
    new_body = _sub_once(body, "  if (topk != 2048) return false;\n", call + "  if (topk != 2048) return false;\n",
                         "prefill dispatch_v32")
    text = text[:start] + tk + "\n" + new_body + text[end:]
    if trim:
        a = text.index("inline bool dispatch_dsv4_single(")
        b = text.index("}  // namespace\n\n// Public dispatcher")
        text = text[:a] + text[b:]
        s = text.index("  if (extra_KV_cache != nullptr) {\n")
        e = text.index("  return false;\n}\n\n}  // namespace flashinfer::sparse_mla_sm120")
        text = text[:s] + (
            "  (void)topk_extra; (void)extra_page_block_size; (void)extra_indices; (void)stride_kv_block_extra;\n"
            "  (void)extra_topk_length; (void)page_block_size;\n"
            "  if (extra_KV_cache != nullptr) return false;\n"
            "  if (mt == ModelType::DSV3_2)  // numinous trimmed build: the two 656-B row types only\n"
            "    return dispatch_v32<ModelType::DSV3_2>(num_heads, topk, Q, KV_cache, indices, attn_sink, output,\n"
            "                                           out_lse, sm_scale, num_tokens, stride_kv_block, topk_length,\n"
            "                                           stream);\n"
            "  if (mt != ModelType::GLM_NSA) return false;\n"
            "  return dispatch_v32<ModelType::GLM_NSA>(num_heads, topk, Q, KV_cache, indices, attn_sink, output,\n"
            "                                          out_lse, sm_scale, num_tokens, stride_kv_block, topk_length,\n"
            "                                          stream);\n") + text[e:]
    return text


def patched_sources(csrc_dir: pathlib.Path, trim: bool = True) -> dict:
    """{file name: patched text} for FlashInfer's five SM120 sparse-MLA sources."""
    src = {n: (pathlib.Path(csrc_dir) / n).read_text() for n in SOURCES}
    src["sparse_mla_sm120_decode_dsv3_2.cu"] = patch_decode_dsv3_2(src["sparse_mla_sm120_decode_dsv3_2.cu"], trim)
    src["sparse_mla_sm120_decode_dsv4.cu"] = patch_decode_dsv4(src["sparse_mla_sm120_decode_dsv4.cu"], trim)
    src["sparse_mla_sm120_prefill.cu"] = patch_prefill(src["sparse_mla_sm120_prefill.cu"], trim)
    for n, t in src.items():
        t = f"// numinous: generated from flashinfer csrc/{n} by glmflash_sm120_fi.py (trim={trim})\n" + t
        src[n] = t
    return src


_MODULE = None


def build_module(trim: Optional[bool] = None, verbose: bool = False):
    """JIT-build (cached by FlashInfer under ~/.cache/flashinfer) and load the widened module. Needs nvcc + SM120."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    if trim is None:
        trim = os.environ.get("NUMINOUS_RTXFLASH_FI_TRIM", "1") == "1"
    name = MODULE_NAME + ("_trim" if trim else "_full")
    gen_dir = jit_env.FLASHINFER_GEN_SRC_DIR / name
    gen_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, t in patched_sources(jit_env.FLASHINFER_CSRC_DIR, trim).items():
        p = gen_dir / n
        if not p.exists() or p.read_text() != t:
            p.write_text(t)
        paths.append(p)
    flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(name, paths, extra_cuda_cflags=flags)
    _MODULE = spec.build_and_load()
    logger.warning("numinous rtxflash: FlashInfer SM120 sparse MLA widened to top-k %d built (%s)", TOPK_JIT, name)
    return _MODULE


def _stock_paged_attention():
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention
    return _sparse_mla_sm120_paged_attention


# ------------------------------------------------------------------------------------------------ runtime


def kv_pages_view(kv_buffer: torch.Tensor, page: int = 64) -> torch.Tensor:
    """SGLang DSA pool buffer [rows, 1, 656] (any dtype view) -> uint8 [rows / 64, 64, 656] (FlashInfer 3-D form:
    slot s is byte s * 656, the same address as the token-major pool)."""
    b = kv_buffer if kv_buffer.dtype == torch.uint8 else kv_buffer.view(torch.uint8)
    b = b.reshape(b.shape[0], -1)
    assert b.shape[1] == A.NSA_ROW, b.shape
    assert b.shape[0] % page == 0, f"pool rows {b.shape[0]} not a multiple of page {page}"
    return b.view(b.shape[0] // page, page, A.NSA_ROW)


def pad_q576(q512: torch.Tensor) -> torch.Tensor:
    """[T, H, 512] -> contiguous [T, H, 576] bf16 with a zero rope tail."""
    return F.pad(q512.to(torch.bfloat16), (0, D_QK - A.D))


def pad_idx(idx: torch.Tensor, width: int) -> torch.Tensor:
    k = idx.shape[1]
    assert k <= width, (k, width)
    idx = idx.to(torch.int32)
    return idx.contiguous() if k == width else F.pad(idx, (0, width - k), value=-1)


def decode_scratch(workspace: Optional[torch.Tensor], t: int, h: int, splits: int, device):
    """mid_out [T, H, splits, 512] bf16 + mid_lse [T, H, splits] fp32 carved from the backend workspace (as SGLang's
    stock path does), or fresh tensors when no workspace is given / it is too small."""
    n_out = t * h * splits * A.D * 2
    n_lse = t * h * splits * 4
    if workspace is not None and workspace.numel() * workspace.element_size() >= n_out + n_lse + 16:
        w = workspace.view(torch.uint8)
        mid_out = w[:n_out].view(torch.bfloat16).view(t, h, splits, A.D)
        off = (n_out + 15) // 16 * 16
        mid_lse = w[off:off + n_lse].view(torch.float32).view(t, h, splits)
        return mid_out, mid_lse
    return (torch.empty(t, h, splits, A.D, dtype=torch.bfloat16, device=device),
            torch.empty(t, h, splits, dtype=torch.float32, device=device))


class FlashInferGlm5Next:
    """q512 [T, H, 512] + 656-byte pool rows + kpool table [T, K<=2112] -> out [T, H, 512] bf16, lse [T, H] (log2).

    mode "jit": one call to the widened module (top-k 2112). mode "stock": the installed FlashInfer module on columns
    [0, 2048) + glmflash_sm120_attn on the tail + merge. `module` / `stock_fn` are injectable for CPU tests."""

    def __init__(self, mode: str = "jit", workspace: Optional[torch.Tensor] = None, module=None, stock_fn=None,
                 pow2: bool = False):
        assert mode in ("jit", "stock"), mode
        self.mode = mode
        # pow2 rows -> DSV3_2 kernels (block-scaled QK, single-pass FP8 P); arbitrary rows -> GLM_NSA kernels
        self.pow2 = bool(pow2)
        self.model_type = MODEL_TYPE_DSV3_2 if self.pow2 else MODEL_TYPE_GLM_NSA
        self.kv_scale_format = "pow2_fp32" if self.pow2 else "arbitrary_fp32"
        self.workspace = workspace
        self._module = module
        self._stock_fn = stock_fn

    @property
    def module(self):
        if self._module is None:
            self._module = build_module()
        return self._module

    @property
    def stock_fn(self):
        if self._stock_fn is None:
            self._stock_fn = _stock_paged_attention()
        return self._stock_fn

    def forward(self, q512: torch.Tensor, kv_buffer: torch.Tensor, idx: torch.Tensor, sm_scale: float,
                out: Optional[torch.Tensor] = None):
        t, h, _ = q512.shape
        assert h in HEADS, f"FlashInfer SM120 GLM_NSA kernels exist for heads {HEADS}, got {h}"
        k = idx.shape[1]
        kv = kv_pages_view(kv_buffer)
        q = pad_q576(q512)
        if out is None:
            out = torch.empty(t, h, A.D, dtype=torch.bfloat16, device=q.device)
        lse = torch.empty(t, h, dtype=torch.float32, device=q.device)
        if t == 0:
            return out, lse
        if k <= TOPK_STOCK or self.mode == "stock":
            main = pad_idx(idx[:, :TOPK_STOCK], TOPK_STOCK)
            self._stock_call(q, kv, main, out, lse, sm_scale)
            if k > TOPK_STOCK:
                tail = idx[:, TOPK_STOCK:].contiguous().to(torch.int32)
                ob, lb = A.sparse_mla(q512, kv_buffer, tail, sm_scale, A.FMT_FP8_656, splits=1, block_n=16)
                A.merge_attn(out, lse, ob, lb)
            return out, lse
        assert k <= TOPK_JIT, f"kpool table width {k} > {TOPK_JIT}: widen TOPK_JIT"
        ip = pad_idx(idx, TOPK_JIT)
        if DEC_PAD_MIN_T and DEC_PAD_MIN_T < t <= DECODE_MAX_TOKENS:
            pad = DECODE_MAX_TOKENS + 1 - t       # dummy rows: idx -1 everywhere (masked), outputs dropped
            qp = F.pad(q, (0, 0, 0, 0, 0, pad))
            ipp = F.pad(ip, (0, 0, 0, pad), value=-1)
            op = torch.empty(t + pad, h, A.D, dtype=torch.bfloat16, device=q.device)
            lp = torch.empty(t + pad, h, dtype=torch.float32, device=q.device)
            self.module.sparse_mla_sm120_paged_attention(qp, kv, ipp, op, lp, float(sm_scale), self.model_type,
                                                        None, None, None, None, None)
            out.copy_(op[:t])
            lse.copy_(lp[:t])
        elif t <= DEC_MAX_T:
            splits = TOPK_JIT // BI
            mid_out, mid_lse = decode_scratch(self.workspace, t, h, splits, q.device)
            self.module.sparse_mla_sm120_decode_dsv3_2(q, kv, ip, mid_out, mid_lse, out, lse, splits, float(sm_scale),
                                                      None, None, self.model_type, cpb_for(t))
        else:
            self.module.sparse_mla_sm120_paged_attention(q, kv, ip, out, lse, float(sm_scale), self.model_type,
                                                        None, None, None, None, None)
        return out, lse

    def _stock_call(self, q, kv, idx2048, out, lse, sm_scale):
        t, h, _ = q.shape
        mid_out = mid_lse = None
        if t <= DECODE_MAX_TOKENS:
            mid_out, mid_lse = decode_scratch(self.workspace, t, h, TOPK_STOCK // BI, q.device)
        self.stock_fn(q, kv, idx2048, out, lse, float(sm_scale), kv_scale_format=self.kv_scale_format,
                      mid_out=mid_out, mid_lse=mid_lse)
