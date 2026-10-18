"""CPU tests of the W4AFP8 -> Marlin packing (operations/patches/glm_w4a8_marlin.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_w4a8_marlin as gm  # noqa: E402


def test_unpack_matches_the_cutlass_test_packer():
    torch.manual_seed(0)
    q = torch.randint(-8, 8, (3, 5, 16), dtype=torch.int8)
    # test/manual/quant/test_cutlass_w4a8_moe.py pack_int4_values_to_int8
    low, high = q[..., 0::2], q[..., 1::2]
    packed = ((high << 4) | (low & 0x0F)).to(torch.int8)
    assert torch.equal(gm.pack_int4_pairs(q), packed)
    assert torch.equal(gm.unpack_int4_pairs(packed), q)


def test_gptq_packing_bit_positions():
    torch.manual_seed(1)
    q = torch.randint(-8, 8, (2, 3, 16), dtype=torch.int8)          # [E, N, K]
    g = gm.to_gptq_int32(q)
    assert g.shape == (2, 2, 3) and g.dtype == torch.int32
    for e in range(2):
        for kb in range(2):
            for n in range(3):
                word = int(g[e, kb, n]) & 0xFFFFFFFF
                for i in range(8):
                    assert ((word >> (4 * i)) & 0xF) == int(q[e, n, kb * 8 + i]) + 8


def test_scales_and_reference_dequant():
    torch.manual_seed(2)
    q = torch.randint(-8, 8, (2, 4, 256), dtype=torch.int8)
    s = torch.rand(2, 4, 2) + 0.5
    w = gm.dequant_reference(gm.pack_int4_pairs(q), s)
    assert torch.allclose(w[:, :, :128], q[:, :, :128].float() * s[:, :, :1])
    assert torch.allclose(w[:, :, 128:], q[:, :, 128:].float() * s[:, :, 1:])
    m = gm.scales_to_marlin_input(s)
    assert m.shape == (2, 2, 4) and m.dtype == torch.bfloat16 and torch.allclose(m.float(), s.permute(0, 2, 1), rtol=1e-2)
