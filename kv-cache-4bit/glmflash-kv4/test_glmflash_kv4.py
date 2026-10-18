"""CPU tests: the rope-free FP4 row equals glm_kv4's row with the rope tail removed, byte for byte."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glm_kv4  # noqa: E402
import glmflash_kv4 as fk  # noqa: E402

torch.manual_seed(0)


def _latent(t=37, scale=3.0):
    x = torch.randn(t, 512) * scale
    x[0] = 0                                   # all-zero row: scale clamps to 2^-9, codes 0, no negative zero
    x[1, :16] = 1e-6                           # tiny block: subnormal-scale path
    x[2, 16:32] = torch.linspace(-3000, 3000, 16)  # amax/6 above 448: scale saturates, codes saturate at 6
    return x.to(torch.bfloat16)


def test_row_geometry():
    assert fk.ROW_BYTES == 288 and fk.FP8_ROW_BYTES == 512 and fk.DIM_ROPE == 0
    assert fk.bytes_per_token() == {"bf16": 11 * 1024, "fp8": 11 * 512, "fp4": 11 * 288}
    assert fk.bytes_per_token(indexer_bytes=1400)["fp4"] == 11 * 288 + 1400


def test_rows_equal_glm_kv4_rows_without_rope_tail():
    x = _latent()
    rope = torch.zeros(x.shape[0], 64, dtype=torch.bfloat16)
    ours = fk.quantize_rows_ref(x)
    theirs = glm_kv4.quantize_rows_ref(x, rope)
    assert ours.shape == (x.shape[0], 288)
    assert torch.equal(ours, theirs[:, :288])


def test_dequant_matches_glm_kv4_latent_part():
    x = _latent()
    rows = fk.quantize_rows_ref(x)
    full = glm_kv4.dequantize_rows_ref(torch.cat([rows, torch.zeros(rows.shape[0], 64, dtype=torch.uint8)], 1), torch.float32)
    ours = fk.dequantize_rows_ref(rows, torch.float32)
    assert ours.shape == (x.shape[0], 512)
    assert torch.equal(ours, full[:, :512])
    # quantization error bound: half the largest E2M1 step (2, between 4 and 6) at the block scale
    s = rows[:, 256:].view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    err = (ours - x.float()).abs()
    sat = x.float().abs() > 6 * s        # saturated values are clamped by design
    assert bool((err[~sat] <= 1.0 * s[~sat] + 1e-6).all())


def test_fp8_output_is_the_trtllm_raw_row():
    rows = fk.quantize_rows_ref(_latent(8))
    out = fk.dequantize_rows_ref(rows)
    assert out.dtype == torch.float8_e4m3fn and out.shape == (8, 512)


def test_sparse_mla_ref_matches_dense_softmax_attention():
    t, h, n, k = 3, 4, 50, 12
    x = torch.randn(n, 512)
    rows = fk.quantize_rows_ref(x)
    q = torch.randn(t, h, 512)
    table = torch.stack([torch.randperm(n)[:k] for _ in range(t)])
    table[0, -3:] = -1
    out = fk.sparse_mla_ref(q, rows, table, 0.1)
    kv = fk.dequantize_rows_ref(rows, torch.float32)
    for i in range(t):
        idx = table[i][table[i] >= 0].long()
        ref = torch.softmax(q[i] @ kv[idx].T * 0.1, -1) @ kv[idx]
        assert torch.allclose(out[i], ref, atol=1e-5)
