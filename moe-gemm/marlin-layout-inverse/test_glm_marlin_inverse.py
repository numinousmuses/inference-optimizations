"""CPU tests of the Marlin -> cutlass_w4a8 inverse (glm_marlin_inverse.py): the Triton kernel under TRITON_INTERPRET
against a numpy transcription of the CUDA repack loops, the scale inverse against marlin_permute_scales, and the
interleave against w4afp8.interleave_scales (both copied from SGLang v0.5.20 below)."""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "patches"))
import glm_marlin_inverse as gmi  # noqa: E402
from glm_w4a8_marlin import pack_int4_pairs, scales_to_marlin_input  # noqa: E402


def interleave_scales_v0520(scales: torch.Tensor) -> torch.Tensor:
    """w4afp8.py:111 verbatim (alignment 4 if the group count divides by 4)."""
    s_shape = scales.shape
    alignment = 4 if s_shape[2] % 4 == 0 else 1
    x = scales.reshape(s_shape[0], s_shape[1], (s_shape[2] // alignment), alignment)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(s_shape[0], s_shape[2] // alignment, s_shape[1] * alignment).contiguous()


def marlin_permute_scales_v0520(s: torch.Tensor, size_k: int, size_n: int, group_size: int) -> torch.Tensor:
    """marlin_utils.py:313 (group_size < size_k branch)."""
    assert group_size < size_k
    return s.reshape((-1, 64))[:, gmi.scale_perm()].reshape((-1, size_n)).contiguous()


def _marlin_words(q_enk: np.ndarray) -> torch.Tensor:
    """q [E, N, K] -> Marlin int32 [E, K/16, 2N] through the transcribed forward loops."""
    e, n, k = q_enk.shape
    words = np.stack([gmi.marlin_forward_reference(q_enk[x].T) for x in range(e)])
    return torch.from_numpy(words.view(np.int32).reshape(e, k // 16, 2 * n).copy())


@pytest.mark.parametrize("n,k", [(64, 256), (128, 128), (192, 384)])
def test_inverse_matches_the_transcribed_forward(n, k):
    rng = np.random.default_rng(n * 7 + k)
    q = rng.integers(-8, 8, size=(2, n, k), dtype=np.int64)
    w_marlin = _marlin_words(q)
    got = gmi.marlin_to_cutlass_weights(w_marlin, n, k)
    want = pack_int4_pairs(torch.from_numpy(q).to(torch.int8))
    assert got.dtype == torch.int8 and got.shape == (2, n, k // 2)
    assert torch.equal(got, want)


def test_source_positions_are_a_bijection():
    n_size, k_size = 128, 64
    k, n = np.meshgrid(np.arange(k_size), np.arange(n_size), indexing="ij")
    word, nib = gmi.marlin_source(k, n, n_size)
    flat = (word * 8 + nib).ravel()
    assert len(set(flat.tolist())) == k_size * n_size and flat.max() == k_size * n_size - 1


def test_into_a_scratch_buffer():
    rng = np.random.default_rng(3)
    q = rng.integers(-8, 8, size=(3, 64, 128), dtype=np.int64)
    scratch = torch.full((3, 64, 64), 77, dtype=torch.int8)
    out = gmi.marlin_to_cutlass_weights(_marlin_words(q), 64, 128, out=scratch)
    assert out.data_ptr() == scratch.data_ptr()
    assert torch.equal(scratch, pack_int4_pairs(torch.from_numpy(q).to(torch.int8)))


@pytest.mark.parametrize("n,k", [(512, 6144), (6144, 256), (128, 1024)])
def test_scales_inverse_and_interleave(n, k):
    g = k // 128
    raw = torch.randn(2, n, g, dtype=torch.float32).to(torch.bfloat16)       # checkpoint layout [E, N, K/128]
    marlin_in = scales_to_marlin_input(raw)                                   # [E, K/128, N]
    s_marlin = torch.stack([marlin_permute_scales_v0520(marlin_in[x], k, n, 128) for x in range(2)])
    back = gmi.marlin_to_raw_scales(s_marlin)
    assert torch.equal(back, marlin_in)
    assert torch.equal(gmi.cutlass_scales(back, gmi.stock_alignment(g)), interleave_scales_v0520(raw))
    one = gmi.cutlass_scales(back, 1)                                         # the tuned tiles' alignment 1
    assert torch.equal(one, raw.permute(0, 2, 1))


@pytest.mark.parametrize("n,k,kt", [(64, 256, 16), (128, 128, 4), (192, 384, 16)])
def test_v2_word_parallel_matches_v1_and_reference(n, k, kt):
    rng = np.random.default_rng(n + k + kt)
    q = rng.integers(-8, 8, size=(2, n, k), dtype=np.int64)
    w = _marlin_words(q)
    v2 = gmi.marlin_to_cutlass_weights_v2(w, n, k, kt_block=kt)
    assert torch.equal(v2, pack_int4_pairs(torch.from_numpy(q).to(torch.int8)))
    assert torch.equal(v2, gmi.marlin_to_cutlass_weights_v1(w, n, k))


def test_marlin_to_fp8_is_exact():
    rng = np.random.default_rng(9)
    n, k = 128, 256
    q = rng.integers(-8, 8, size=(2, n, k), dtype=np.int64)
    f8 = gmi.marlin_to_fp8_weights(_marlin_words(q), n, k)
    assert f8.dtype == torch.float8_e4m3fn
    assert torch.equal(f8.to(torch.float32), torch.from_numpy(q).to(torch.float32))


def test_int4_pairs_to_fp8_is_exact():
    rng = np.random.default_rng(13)
    q = torch.from_numpy(rng.integers(-8, 8, size=(3, 64, 256))).to(torch.int8)
    f8 = gmi.int4_pairs_to_fp8(pack_int4_pairs(q))
    assert f8.shape == (3, 64, 256) and torch.equal(f8.to(torch.float32), q.to(torch.float32))


@pytest.mark.parametrize("n,k,kt", [(64, 256, 16), (128, 128, 4), (192, 384, 8)])
def test_v3_chunk_kernel(n, k, kt):
    rng = np.random.default_rng(n * 3 + k + kt)
    q = rng.integers(-8, 8, size=(2, n, k), dtype=np.int64)
    got = gmi.marlin_to_cutlass_weights_v3(_marlin_words(q), n, k, kt_block=kt)
    assert torch.equal(got, pack_int4_pairs(torch.from_numpy(q).to(torch.int8)))
