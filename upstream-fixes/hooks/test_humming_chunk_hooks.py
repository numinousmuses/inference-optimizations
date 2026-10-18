"""CPU tests for operations/patches/humming_chunk_hooks.py.

1. The hook applies once to the SGLang v0.5.20 file, is idempotent, and the patched module still parses.
2. Chunked preparation equals the unchunked call on FlashInfer 0.6.18's Humming math (the scale clamp and payload
   rewrite copied from flashinfer/fused_moe/prepare.py without the CUDA-only checks; the CUDA interleave kernels are
   modelled as per-expert reshapes, which is what their [E, ...] -> [E, ...] contract states).

    python3 -m pytest -q operations/patches/test_humming_chunk_hooks.py
"""
from __future__ import annotations

import os
import pathlib
import shutil
import struct
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import humming_chunk_hooks as hc  # noqa: E402

SGL = pathlib.Path(os.environ.get("SGL_V0520_ROOT", "/tmp/sgl-v0520-wt/python/sglang"))


@pytest.mark.skipif(not (SGL / hc.PATH).exists(), reason="SGLang v0.5.20 source tree not present")
def test_hook_applies_once_and_parses(tmp_path):
    root = tmp_path / "sglang"
    (root / os.path.dirname(hc.PATH)).mkdir(parents=True)
    shutil.copy(SGL / hc.PATH, root / hc.PATH)
    assert all(line.endswith("anchor ok") for line in hc.apply(str(root), check=True))
    first = hc.apply(str(root))
    assert all("applied to" in line for line in first)
    second = hc.apply(str(root))
    assert all(line.endswith("already applied") for line in second)
    src = (root / hc.PATH).read_text()
    compile(src, str(root / hc.PATH), "exec")
    # the binding sits right after the import inside the Humming branch, the wrapper at module level
    assert src.count("_numinous_humming_chunked(") == 2
    assert src.index("def _numinous_humming_chunked") < src.index("class Mxfp4FlashinferCutlassMoEMethod")


# ---- FlashInfer 0.6.18 Humming math (prepare.py), CPU copy for the exactness test -------------------------------
def _lut() -> torch.Tensor:
    def fbits(b):
        return struct.unpack("f", struct.pack("I", b & 0xFFFFFFFF))[0]

    def bits(v):
        return struct.unpack("I", struct.pack("f", v))[0]

    def deq(code):
        return fbits(((code & 0x8) << 28) | ((code & 0x7) << 22))

    def q(v):
        vb = bits(v)
        mask = 0x81C00000
        rz, ru = vb & mask, (vb + 0x00200000) & mask
        rb = ru if abs(v - fbits(rz)) >= abs(v - fbits(ru)) else rz
        return ((rb & 0x80000000) >> 28) | ((rb & 0x01C00000) >> 22)

    lut = torch.empty((256, 16), dtype=torch.uint8)
    for d in range(256):
        sf = fbits(0x3F800000 - (d << 23))
        for c in range(16):
            n = 0 if c == 8 else c
            if d:
                n = q(deq(n) * sf)
            lut[d, c] = n
    return lut


LUT = _lut()


def _scale(raw: torch.Tensor, max_range: int = 11):
    e = raw.shape[0]
    v = raw.contiguous().view(e, -1)
    smax = v.max(dim=1, keepdim=True).values
    smin = v.min(dim=1, keepdim=True).values
    rng = torch.minimum(smax - smin, torch.tensor(max_range, dtype=torch.uint8))
    smin_new = smax - rng
    clamped = v.maximum(smin_new)
    delta = (clamped - v).to(torch.uint8)
    offset = torch.bitwise_and(clamped - smin_new + 1, 0x0F).to(torch.uint8)
    residual = torch.exp2(smin_new.squeeze(1).to(torch.float32) - 127.0) * 0.5
    return offset.view_as(raw), residual, delta.view_as(raw)


def _payload(w: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    lo, hi = w & 0x0F, (w >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(*w.shape[:-1], -1)
    d = delta.repeat_interleave(32, dim=-1).to(torch.long)
    rw = LUT[d, codes.to(torch.long)]
    return (rw[..., 0::2] | (rw[..., 1::2] << 4)).contiguous()


def _prepare(weight: torch.Tensor, raw_scale: torch.Tensor):
    offset, residual, delta = _scale(raw_scale)
    pw = _payload(weight, delta)
    # interleave kernels: per-expert layout changes; modelled as a per-expert flip / reshape
    w_il = pw.flip(-1).contiguous()
    e, rows, kgs = offset.shape
    s_il = offset.reshape(e, rows // 64, 64, kgs // 4, 4).permute(0, 1, 3, 2, 4).contiguous()
    return w_il, s_il, residual


@pytest.mark.parametrize("experts,chunk", [(40, 16), (48, 16), (7, 3), (16, 16), (5, 0)])
def test_chunked_equals_unchunked(experts, chunk, monkeypatch):
    monkeypatch.setenv("NUMINOUS_HUMMING_CHUNK", str(chunk))
    # the wrapper is defined inside the patched SGLang module; exec its source text with torch/os in scope
    ns = {"torch": torch, "os": os}
    exec(hc.WRAPPER, ns)  # noqa: S102
    cuda_cache = torch.cuda.empty_cache
    torch.cuda.empty_cache = lambda: None   # CPU run
    try:
        g = torch.Generator().manual_seed(experts * 31 + chunk)
        rows, k = 128, 256
        w = torch.randint(0, 256, (experts, rows, k // 2), dtype=torch.uint8, generator=g)
        # E8M0 scales spread wider than max_range so the clamp and payload rewrite are exercised
        s = torch.randint(100, 130, (experts, rows, k // 32), dtype=torch.uint8, generator=g)
        ref = _prepare(w, s)
        got = ns["_numinous_humming_chunked"](_prepare)(w, s)
    finally:
        torch.cuda.empty_cache = cuda_cache
    for a, b in zip(ref, got):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a, b)
