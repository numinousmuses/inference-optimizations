"""CPU tests for the HIP sparse MLA decode kernel's index algebra (rocm_sparse_mla_decode_hip.py)."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rocm_sparse_mla_decode_hip as m  # noqa: E402


def slot_of(key: int) -> int:  # mirrors the device function
    return ((key & 15) >> 2) * 8 + (key >> 4) * 4 + (key & 3)


def test_slot_is_a_bijection_matching_the_p_fragment():
    assert sorted(slot_of(k) for k in range(32)) == list(range(32))
    # lane group g holds S^T rows 4g+i of tile 0 (keys 4g+i) then of tile 1 (keys 16+4g+i), as P fragment
    # bytes j = 0..7; the MFMA A operand byte j of lane group g is k index 8g+j.
    for g in range(4):
        for j in range(8):
            key = 4 * g + j if j < 4 else 16 + 4 * g + (j - 4)
            assert slot_of(key) == 8 * g + j


def test_source_mirrors_slot_formula_and_constants():
    src = m.HIP_SRC
    assert "((key & 15) >> 2) * 8 + (key >> 4) * 4 + (key & 3)" in src
    assert re.search(r"#define BN 32\b", src) and re.search(r"#define D 576\b", src)
    # LDS fits a static allocation (<64 KB) and the strides keep 8-B alignment for ds_read_b64
    kst = int(re.search(r"#define KST (\d+)", src).group(1))
    assert kst % 8 == 0 and kst >= 576
    assert 32 * kst + 2048 * 4 < 64 * 1024


def test_lds_bank_spread():
    # 16 lanes (c16) of one group read 8 B at row c16: distinct 2-word bank pairs within 64 banks
    for stride in (584,):
        words = [(r * stride // 4) % 64 for r in range(16)]
        pairs = set()
        for w in words:
            pairs.add(w)
            pairs.add((w + 1) % 64)
        assert len(pairs) == 32, (stride, words)


def test_wrapper_rejects_unsupported_shapes():
    import pytest
    import torch
    q = torch.zeros(2, 32, 576)
    with pytest.raises(AssertionError):
        m.sparse_mla_decode_hip(q, torch.zeros(10, 1, 576), torch.zeros(2, 1, 2048, dtype=torch.int32), 0.1)


def test_pv_row_map_inverts_slot():
    # lane (g, c16): transpose-read row r = c16 // 2 must be the key whose slot is 8g + r
    for g in range(4):
        for c16 in range(16):
            r = c16 >> 1
            key = 4 * g + r if r < 4 else 16 + 4 * g + (r - 4)
            assert slot_of(key) == 8 * g + r
    assert "const int vkey = vr < 4 ? 4 * g + vr : 16 + 4 * g + (vr - 4);" in m.HIP_SRC
