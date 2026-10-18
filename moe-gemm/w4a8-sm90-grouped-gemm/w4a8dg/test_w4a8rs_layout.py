"""CPU checks of the k3amd W4A8 v4 kernel (w4a8rs_kernel.cuh): every index the device code computes, emulated.

- TMA CU_TENSOR_MAP_SWIZZLE_64B placement of the 64-byte int4 rows (cute Swizzle<2,4,3>) and the kernel's sw64().
- load_frags(): each thread's 16 LDS.U16 + PRMT-LUT cvt4 reproduce the WGMMA m64n128k32 RS A fragment for that thread
  (cute ALayout_64x32, taken from cute/atom/mma_traits_sm90_gmma.hpp), byte for byte, for both math warpgroups.
- every one of those LDS.U16 warp instructions touches each shared-memory bank with at most one 32-bit word.
- the epilogue's (accumulator index -> token, weight) map equals cute CLayout_64x128 and covers the tile once.
- the math role's stage walk (next_of/advance, two fragment buffers) matches the TMA producer's for 1..9 K blocks.

    python3 -m pytest -q operations/kernels/w4a8dg/test_w4a8rs_layout.py
"""
import itertools

import numpy as np

E4M3 = {0: 0x00, 1: 0x38, 2: 0x40, 3: 0x44, 4: 0x48, 5: 0x4A, 6: 0x4C, 7: 0x4E,
        -1: 0xB8, -2: 0xC0, -3: 0xC4, -4: 0xC8, -5: 0xCA, -6: 0xCC, -7: 0xCE, -8: 0xD0}


def prmt(a, b, sel):
    data = a | (b << 32)
    out = 0
    for i in range(4):
        s = (sel >> (4 * i)) & 0xF
        byte = (data >> (8 * (s & 7))) & 0xFF
        if s & 8:
            byte = 0xFF if byte & 0x80 else 0x00
        out |= byte << (8 * i)
    return out


def cvt4(idx, sign):
    pos = prmt(0x44403800, 0x4E4C4A48, idx)
    neg = prmt(0xCACCCED0, 0xB8C0C4C8, idx)
    return prmt(pos, neg, 0x3210 | sign)


def sw64(row, byte):
    """the kernel's sw64()"""
    return row * 64 + ((((byte >> 4) ^ (row >> 1)) & 3) << 4) + (byte & 15)


def tma_sw64(tile):
    """TMA SWIZZLE_64B: logical (row, byte) lands at cute Swizzle<2,4,3>(row*64 + byte) (bits 4-5 ^= bits 7-8)."""
    rows = tile.shape[0]
    smem = np.zeros(rows * 64, dtype=np.uint8)
    for r in range(rows):
        for b in range(64):
            lin = r * 64 + b
            phys = lin ^ (((lin >> 7) & 3) << 4)
            smem[phys] = tile[r, b]
    return smem


def nib(tile, row, k):
    """signed int4 value of (row, k) in the row-major pair layout (low nibble = even k)"""
    byte = int(tile[row, k // 2])
    n = (byte >> 4) if k & 1 else (byte & 0xF)
    return n - 16 if n >= 8 else n


def a_layout_64x32(t, v):
    """cute ALayout_64x32 = ((4,8,4),(4,2,2)):((256,1,16),(64,8,1024)) -> (m, k) of thread t, value v"""
    t0, t1, t2 = t % 4, (t // 4) % 8, t // 32
    v0, v1, v2 = v % 4, (v // 4) % 2, v // 8
    lin = t0 * 256 + t1 * 1 + t2 * 16 + v0 * 64 + v1 * 8 + v2 * 1024
    return lin % 64, lin // 64


def c_layout_64x128(t, v):
    """cute CLayout_64x128 = ((4,8,4),(2,2,16)):((128,1,16),(64,8,512)) -> (m, n)"""
    t0, t1, t2 = t % 4, (t // 4) % 8, t // 32
    v0, v1, v2 = v % 2, (v // 2) % 2, v // 4
    lin = t0 * 128 + t1 * 1 + t2 * 16 + v0 * 64 + v1 * 8 + v2 * 512
    return lin % 64, lin // 64


def test_sw64_matches_tma_placement():
    rng = np.random.default_rng(0)
    tile = rng.integers(0, 256, size=(128, 64), dtype=np.uint8)
    smem = tma_sw64(tile)
    for r in range(128):
        for b in range(64):
            assert smem[sw64(r, b)] == tile[r, b]


def frags_of(smem, wg, wl, lane):
    """the kernel's load_frags for one thread: rl = wl*16 + g + wg*64, c = lane % 4"""
    g, c = lane // 4, lane % 4
    rl = wl * 16 + g + wg * 64
    f = []
    for j in range(4):
        b0, b1 = 16 * j + 2 * c, 16 * j + 2 * c + 8
        for (r, b) in ((rl, b0), (rl + 8, b0), (rl, b1), (rl + 8, b1)):
            u = int(smem[sw64(r, b)]) | (int(smem[sw64(r, b) + 1]) << 8)
            f.append(cvt4(u & 0x7777, (u & 0x8888) >> 1))
    return f


def test_fragments_equal_rs_a_layout():
    rng = np.random.default_rng(1)
    tile = rng.integers(0, 256, size=(128, 64), dtype=np.uint8)     # one stage: 128 weight rows x 128 k (int4)
    smem = tma_sw64(tile)
    for wg in range(2):
        for wl in range(4):
            for lane in range(32):
                f = frags_of(smem, wg, wl, lane)
                t = wl * 32 + lane                                    # thread index inside the warpgroup
                for j in range(4):                                    # k32 step = one WGMMA
                    for v in range(16):
                        m, k = a_layout_64x32(t, v)
                        want = E4M3[nib(tile, wg * 64 + m, 32 * j + k)]
                        reg, byte = v // 4, v % 4
                        got = (f[4 * j + reg] >> (8 * byte)) & 0xFF
                        assert got == want, (wg, wl, lane, j, v, hex(got), hex(want))


def test_fragment_loads_are_bank_conflict_free():
    for wg, wl, j, which in itertools.product(range(2), range(4), range(4), range(4)):
        words_per_bank = {}
        for lane in range(32):
            g, c = lane // 4, lane % 4
            rl = wl * 16 + g + wg * 64
            r = rl + (8 if which in (1, 3) else 0)
            b = 16 * j + 2 * c + (8 if which >= 2 else 0)
            addr = sw64(r, b)
            words_per_bank.setdefault((addr // 4) % 32, set()).add(addr // 4)
        assert max(len(s) for s in words_per_bank.values()) == 1, (wg, wl, j, which, words_per_bank)


def test_epilogue_map_is_clayout_and_covers_tile():
    seen = set()
    for wg in range(2):
        for wl in range(4):
            for lane in range(32):
                g, c = lane // 4, lane % 4
                wr = wg * 64 + wl * 16 + g
                t = wl * 32 + lane
                for i in range(16):
                    for q in range(4):
                        tok = i * 8 + c * 2 + (q & 1)
                        wrow = wr + (8 if q >= 2 else 0)
                        m, n = c_layout_64x128(t, i * 4 + q)
                        assert (wg * 64 + m, n) == (wrow, tok)
                        seen.add((tok, wrow))
    assert len(seen) == 128 * 128


def walk_producer(kbs, stages):
    stage = phase = 0
    out = []
    for k_blocks in kbs:
        for _ in range(k_blocks):
            out.append((stage, phase))
            stage = 0 if stage + 1 == stages else stage + 1
            phase ^= stage == 0
    return out


def walk_math(kbs, stages):
    """the kernel's math loop: waits (fragment loads) and releases (promote) in order"""
    stage = phase = 0
    waits, rels = [], []

    def nxt(s, p):
        ns = 0 if s + 1 == stages else s + 1
        return ns, p ^ (ns == 0)
    for k_blocks in kbs:
        waits.append((stage, phase))
        kb = 0
        while kb < k_blocks:
            if kb + 1 < k_blocks:
                waits.append(nxt(stage, phase))
            rels.append(stage)
            stage, phase = nxt(stage, phase)
            if kb + 1 >= k_blocks:
                break
            if kb + 2 < k_blocks:
                waits.append(nxt(stage, phase))
            rels.append(stage)
            stage, phase = nxt(stage, phase)
            kb += 2
    return waits, rels


def test_math_stage_walk_matches_producer():
    for stages in (4, 5, 6):
        for kbs in itertools.product(range(1, 10), repeat=2):
            prod = walk_producer(list(kbs) * 2, stages)
            waits, rels = walk_math(list(kbs) * 2, stages)
            assert waits == prod
            assert rels == [s for s, _ in prod]


def test_silu_two_half_boxes_land_like_one_tile():
    """kSilu TMA-loads the 64 gate rows to tile rows 0..63 and the 64 up rows to rows 64..127 as two boxes; the 64B
    swizzle phase comes from the smem address, so the second box at +4096 B lands exactly where one 128-row box
    would, and load_frags/sw64 need no change."""
    rng = np.random.default_rng(2)
    gate = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    up = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    smem = np.zeros(128 * 64, dtype=np.uint8)
    for base, box in ((0, gate), (4096, up)):
        for r in range(64):
            for b in range(64):
                lin = base + r * 64 + b
                smem[lin ^ (((lin >> 7) & 3) << 4)] = box[r, b]
    assert np.array_equal(smem, tma_sw64(np.concatenate([gate, up])))


def test_silu_epilogue_pairs_gate_and_up():
    """Warpgroup 0's accumulator holds gate(t, j) and warpgroup 1's holds up(t, j) at the same (thread, register):
    the exchange tile written by thread T of warpgroup 0 is read only by thread T of warpgroup 1, and the 128 x 64
    FP8 output tile is covered exactly once."""
    out = {}
    for wl in range(4):
        for lane in range(32):
            g, c = lane // 4, lane % 4
            t_idx = wl * 32 + lane
            for i in range(16):
                for q in range(4):
                    tok, j = i * 8 + c * 2 + (q & 1), wl * 16 + g + (q >> 1) * 8
                    m, n = c_layout_64x128(t_idx, i * 4 + q)
                    assert (m, n) == (j, tok)          # same (row-in-warpgroup, token) for gate (wg0) and up (wg1)
                    assert (tok, j) not in out
                    out[(tok, j)] = t_idx
    assert len(out) == 128 * 64


def test_silu_epilogue_numerics_match_the_served_kernel_math():
    """The fused epilogue's arithmetic (CPU model, exact exp) equals the served path: GEMM1 writes c1 in bf16, then
    silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe computes gate / (1 + exp(-gate)) * up * (1 / a2)
    in fp32 and stores e4m3 (ep_moe_kernels.py:766-795). The kernel rounds fin * a_scale to bf16 for both operands
    before the element op, which is what makes the two agree."""
    import torch
    g = torch.Generator().manual_seed(3)
    fin_gate = torch.randn(4096, generator=g) * 40
    fin_up = torch.randn(4096, generator=g) * 40
    a_scale, a2 = 0.05, 0.02
    # served: c1 = bf16(fin * a_scale); silu kernel in fp32; cast to e4m3
    gate = (fin_gate * a_scale).to(torch.bfloat16).float()
    up = (fin_up * a_scale).to(torch.bfloat16).float()
    inv = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(a2, dtype=torch.float32)
    served = (gate / (1 + torch.exp(-gate)) * up * inv).to(torch.float8_e4m3fn)
    # fused model: identical op order on the same bf16-rounded operands
    fused = []
    for fg, fu in zip(fin_gate.tolist(), fin_up.tolist()):
        gg = torch.tensor(fg * a_scale, dtype=torch.float32).to(torch.bfloat16).float()
        uu = torch.tensor(fu * a_scale, dtype=torch.float32).to(torch.bfloat16).float()
        fused.append(float(gg / (1 + torch.exp(-gg)) * uu * inv))
    fused = torch.tensor(fused, dtype=torch.float32).to(torch.float8_e4m3fn)
    assert torch.equal(served.view(torch.uint8), fused.view(torch.uint8))
