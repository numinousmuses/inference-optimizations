"""kvmix_glm kernels vs the torch references, bit for bit (GPU), plus CPU checks of the layout and references.

    python3 -m pytest -q operations/kernels/test_kvmix_glm.py            # CPU: layout + reference consistency
    devbox.py test <box> operations/kernels/test_kvmix_glm.py           # GPU: every kernel, CUDA-graph replay
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm_kv4 as kv4  # noqa: E402
import kvmix_glm as mix  # noqa: E402

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
PS = 64
L = 3


def _import_flashinfer_first():
    """The engine always has flashinfer loaded (it sets TRITON_PTXAS_BLACKWELL_PATH): test the same assembler."""
    try:
        import flashinfer  # noqa: F401
    except Exception:
        pass


def _lay(n8=PS * 30, n4=PS * 20):
    return mix.Layout(n8, n4, PS)


def _fp8_bytes(n, dev):
    return (torch.randn(n, 576, device=dev) * 1.5).to(torch.float8_e4m3fn).view(torch.uint8)


def _fp4_rows(n, dev):
    nope = torch.randn(n, 512, device=dev) * 1.5
    rope = (torch.randn(n, 64, device=dev) * 1.5).to(torch.float8_e4m3fn)
    return kv4.quantize_rows_ref(nope.bfloat16(), rope)


def _pools(lay, dev, scratch=PS * 8):
    fp8_all = _fp8_bytes(L * lay.fp8_rows + scratch, dev)
    fp4_all = _fp4_rows(L * lay.fp4_rows, dev)
    return fp8_all, fp4_all


def _slots(lay, n, dev, frac4=0.5, with_pad=False, unique=True):
    """Random valid unified slots (no slot 0 / FP4 dummy rows), optionally with -1 padding. Writes need unique slots
    (a duplicate makes the winning writer order-dependent); read tables may repeat slots."""
    n4 = int(n * frac4)
    if unique:
        assert n - n4 <= lay.boundary - 1 and n4 <= lay.n4, (n, lay.boundary, lay.n4)
        s8 = torch.randperm(lay.boundary - 1, device=dev)[: n - n4] + 1
        s4 = torch.randperm(lay.n4, device=dev)[:n4] + lay.boundary
    else:
        s8 = torch.randint(1, lay.boundary, (n - n4,), device=dev)
        s4 = torch.randint(lay.boundary, lay.n_slots, (n4,), device=dev)
    s = torch.cat([s8, s4])[torch.randperm(n, device=dev)]
    if with_pad:
        s[torch.rand(n, device=dev) < 0.1] = -1
    return s


# ------------------------------------------------------------------------------------------------ CPU


def test_layout_math():
    lay = _lay()
    assert lay.boundary == PS * 31 and lay.n_slots == PS * 51 and lay.fp8_rows == PS * 31 and lay.fp4_rows == PS * 21
    assert lay.fp8_pages == 30 and lay.fp4_pages == 20
    assert lay.fp4_row(lay.boundary) == PS  # first FP4 slot -> FP4 row 64 (rows [0, 64) are the dummy page)
    assert lay.fp4_page_to_unified(1) == 31 and lay.unified_page_to_fp4(50) == 20
    assert lay.is_fp4_slot(lay.boundary) and not lay.is_fp4_slot(lay.boundary - 1)


def test_references_agree_on_cpu():
    torch.manual_seed(0)
    lay = _lay(PS * 3, PS * 2)
    fp8 = _fp8_bytes(lay.fp8_rows, "cpu")
    fp4 = _fp4_rows(lay.fp4_rows, "cpu")
    t = 6
    loc = torch.tensor([1, lay.boundary, 70, lay.boundary + 70, 0, lay.n_slots - 1])
    nope_bf16 = torch.randn(t, 512).bfloat16()
    nope_fp8 = nope_bf16.to(torch.float8_e4m3fn)
    rope = torch.randn(t, 64).to(torch.float8_e4m3fn)
    fp8_before = fp8.clone()
    mix.write_mixed_ref(lay, fp8, fp4, loc, nope_fp8, rope, nope_bf16)
    assert torch.equal(fp8[0], fp8_before[0])  # slot 0 is never written
    assert torch.equal(fp8[1, :512], nope_fp8[0].view(torch.uint8))
    q = kv4.quantize_rows_ref(nope_bf16[1:2], rope[1:2])[0]
    assert torch.equal(fp4[PS], q)
    g = mix.gather_bf16_ref(lay, fp8, fp4, loc[:2])
    assert torch.equal(g[0, :512], nope_fp8[0].float().bfloat16())
    assert torch.equal(g[1], kv4.dequantize_rows_ref(q[None], torch.float32)[0].bfloat16())
    table = torch.tensor([[1, lay.boundary, -1, 70], [lay.boundary + 5, -1, -1, 2]], dtype=torch.int32)
    out, rows = mix.redirect_gather_ref(lay, table, fp4, 1000)
    assert out.tolist() == [[1, 1001, -1, 70], [1004, -1, -1, 2]] and set(rows) == {1, 4}


# ------------------------------------------------------------------------------------------------ GPU


@gpu
def test_write_mixed_bit_exact():
    _import_flashinfer_first()
    torch.manual_seed(1)
    dev = "cuda"
    lay = _lay()
    fp8_all, fp4_all = _pools(lay, dev)
    fp8 = fp8_all[: lay.fp8_rows].clone()
    fp4 = fp4_all[: lay.fp4_rows].clone()
    t = 300
    loc = _slots(lay, t, dev)
    loc[0] = 0  # dummy slot: skipped
    nope_bf16 = torch.randn(t, 512, device=dev).bfloat16()
    nope_fp8 = nope_bf16.to(torch.float8_e4m3fn)
    rope = torch.randn(t, 64, device=dev).to(torch.float8_e4m3fn)
    for have_bf16 in (True, False):
        a8, a4 = fp8.clone(), fp4.clone()
        r8, r4 = fp8.clone(), fp4.clone()
        mix.write_mixed(lay, a8, a4, loc, nope_fp8, rope, nope_bf16 if have_bf16 else None)
        mix.write_mixed_ref(lay, r8, r4, loc, nope_fp8, rope, nope_bf16 if have_bf16 else None)
        torch.cuda.synchronize()
        assert torch.equal(a8, r8) and torch.equal(a4, r4), have_bf16
        assert torch.equal(a8[0], fp8[0])
    # int32 loc and the [T, 1, D] shapes the backend passes
    a8, a4 = fp8.clone(), fp4.clone()
    mix.write_mixed(lay, a8.view(-1, 1, 576), a4.view(-1, 1, 352), loc.int(), nope_fp8.view(t, 1, 512),
                    rope.view(t, 1, 64), nope_bf16.view(t, 1, 512))
    r8, r4 = fp8.clone(), fp4.clone()
    mix.write_mixed_ref(lay, r8, r4, loc, nope_fp8, rope, nope_bf16)
    assert torch.equal(a8, r8) and torch.equal(a4, r4)


@gpu
def test_redirect_gather_bit_exact():
    _import_flashinfer_first()
    torch.manual_seed(2)
    dev = "cuda"
    lay = _lay()
    fp8_all, fp4_all = _pools(lay, dev, scratch=PS * 40)
    fp4 = fp4_all[lay.fp4_rows: 2 * lay.fp4_rows]
    t, k = 5, 512
    table = _slots(lay, t * k, dev, with_pad=True, unique=False).int().view(t, k)
    out = torch.empty_like(table)
    scratch = torch.zeros(t * k, 576, dtype=torch.uint8, device=dev)
    base = 777
    mix.redirect_gather(lay, table, out, fp4, scratch, base)
    ref_out, rows = mix.redirect_gather_ref(lay, table.cpu(), fp4.cpu(), base)
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), ref_out)
    idx = torch.tensor(sorted(rows), dtype=torch.int64)
    got = scratch[idx.to(dev)].cpu()
    want = torch.stack([rows[i] for i in idx.tolist()]).view(torch.uint8)
    assert torch.equal(got, want)
    untouched = torch.ones(t * k, dtype=torch.bool)
    untouched[idx] = False
    assert scratch.cpu()[untouched].abs().sum().item() == 0
    # fp8-dtype views (the pool's scratch and layer views are fp8 views of uint8 storage)
    out2 = torch.empty_like(table)
    scratch2 = torch.zeros(t * k, 576, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    mix.redirect_gather(lay, table, out2, fp4.view(torch.float8_e4m3fn), scratch2, base)
    assert torch.equal(out2, out) and torch.equal(scratch2.view(torch.uint8), scratch)


@gpu
def test_seq_dequant_and_redirect_seq_bit_exact():
    _import_flashinfer_first()
    torch.manual_seed(3)
    dev = "cuda"
    lay = _lay()
    fp8_all, fp4_all = _pools(lay, dev)
    fp4 = fp4_all[: lay.fp4_rows]
    bs, max_len, k = 4, 150, 64
    seq_lens = torch.tensor([150, 20, 97, 130], dtype=torch.int32, device=dev)
    pt = torch.full((bs, max_len), -1, dtype=torch.int32, device=dev)
    all_slots = _slots(lay, int(seq_lens.sum()), dev, frac4=0.6)  # unique across the batch (each slot has one owner)
    all_slots[150:170] = torch.randperm(lay.boundary - 1, device=dev)[:20] + 1  # request 1 holds only FP8 rows
    used = 0
    for b in range(bs):
        n = int(seq_lens[b])
        pt[b, :n] = all_slots[used:used + n].int()
        used += n
    fp4_reqs = [True, False, True, True]  # request 1 holds only FP8 rows
    req_ids = torch.tensor([b for b in range(bs) if fp4_reqs[b]], dtype=torch.int32, device=dev)
    bases = [0]
    for b in req_ids.tolist()[:-1]:
        bases.append(bases[-1] + int(seq_lens[b]))
    base_t = torch.tensor(bases, dtype=torch.int32, device=dev)
    rows_total = bases[-1] + int(seq_lens[req_ids[-1]])
    scratch = torch.zeros(rows_total + 8, 576, dtype=torch.uint8, device=dev)
    slot_to_pos = torch.full((lay.n_slots,), -1, dtype=torch.int32, device=dev)
    mix.seq_dequant(lay, fp4, pt, req_ids, seq_lens, base_t, max_len, scratch, slot_to_pos)
    # a per-token table picking positions inside each request's own sequence
    toks = [3, 1, 2, 2]
    table = torch.full((sum(toks), k), -1, dtype=torch.int32, device=dev)
    row = 0
    for b in range(bs):
        n = int(seq_lens[b])
        for _ in range(toks[b]):
            picks = torch.randperm(n, device=dev)[: min(k, n)]
            table[row, : picks.numel()] = pt[b, picks]
            row += 1
    out = torch.empty_like(table)
    sbase = 4321
    mix.redirect_seq(lay, table, slot_to_pos, sbase, out)
    ref_out, ref_rows = mix.seq_ref(lay, fp4.cpu(), pt.cpu(), req_ids.cpu(), seq_lens.cpu(), base_t.cpu(),
                                    table.cpu(), sbase)
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), ref_out)
    for r, want in ref_rows.items():
        assert torch.equal(scratch[r].cpu(), want.view(torch.uint8)), r


@gpu
def test_promote_and_move_bit_exact():
    _import_flashinfer_first()
    torch.manual_seed(4)
    dev = "cuda"
    lay = _lay()
    fp8_all, fp4_all = _pools(lay, dev)
    n_pages = 3
    src_pages = torch.tensor([2, 17, 29], device=dev)
    dst_pages = torch.tensor([31, 40, 50], device=dev)  # unified FP4 pages
    src = (src_pages[:, None] * PS + torch.arange(PS, device=dev)).reshape(-1)
    dst = (dst_pages[:, None] * PS + torch.arange(PS, device=dev)).reshape(-1)
    a8, a4 = fp8_all.clone(), fp4_all.clone()
    r8, r4 = fp8_all.clone(), fp4_all.clone()
    mix.promote_rows(lay, a8, a4, L, src, dst)
    mix.promote_ref(lay, r8, r4, L, src.cpu(), dst.cpu())
    torch.cuda.synchronize()
    assert torch.equal(a8, r8) and torch.equal(a4, r4)
    assert not torch.equal(a4, fp4_all)  # rows changed
    # move across the four format combinations, all layers
    tgt = torch.tensor([5, lay.boundary + 70, 9, lay.boundary + 200], device=dev)
    srcm = torch.tensor([6, 7, lay.boundary + 80, lay.boundary + 90], device=dev)
    a8, a4 = fp8_all.clone(), fp4_all.clone()
    r8, r4 = fp8_all.clone(), fp4_all.clone()
    mix.move_rows(lay, a8, a4, L, tgt, srcm)
    mix.move_ref(lay, r8, r4, L, tgt.cpu(), srcm.cpu())
    torch.cuda.synchronize()
    assert torch.equal(a8, r8) and torch.equal(a4, r4)
    for l in range(L):
        assert torch.equal(a8[l * lay.fp8_rows + 5], fp8_all[l * lay.fp8_rows + 6])  # FP8 -> FP8 byte copy
        r_t, r_s = lay.fp4_row(lay.boundary + 200), lay.fp4_row(lay.boundary + 90)
        assert torch.equal(a4[l * lay.fp4_rows + r_t], fp4_all[l * lay.fp4_rows + r_s])  # FP4 -> FP4 byte copy


@gpu
def test_gather_bf16_and_index_move():
    _import_flashinfer_first()
    torch.manual_seed(5)
    dev = "cuda"
    lay = _lay()
    fp8_all, fp4_all = _pools(lay, dev)
    fp8, fp4 = fp8_all[: lay.fp8_rows], fp4_all[: lay.fp4_rows]
    loc = _slots(lay, 200, dev)
    out = torch.empty(200, 576, dtype=torch.bfloat16, device=dev)
    mix.gather_bf16(lay, fp8, fp4, loc, out)
    assert torch.equal(out.cpu(), mix.gather_bf16_ref(lay, fp8.cpu(), fp4.cpu(), loc.cpu()))
    pages = lay.fp8_pages + lay.fp4_pages + 2
    buf = torch.randint(0, 255, (pages, PS * 132), dtype=torch.uint8, device=dev)
    ref = buf.clone()
    tgt = torch.tensor([5, lay.boundary + 3, 200, lay.n_slots - 1], device=dev)
    src = torch.tensor([lay.boundary + 66, 7, 201, 1], device=dev)
    mix.index_move(buf, PS, tgt, src)
    torch.cuda.synchronize()
    for t_, s_ in zip(tgt.tolist(), src.tolist()):
        tp, to, sp, so = t_ // PS, t_ % PS, s_ // PS, s_ % PS
        ref[tp, to * 128:(to + 1) * 128] = ref[sp, so * 128:(so + 1) * 128]
        ref[tp, PS * 128 + to * 4: PS * 128 + (to + 1) * 4] = ref[sp, PS * 128 + so * 4: PS * 128 + (so + 1) * 4]
    assert torch.equal(buf, ref)


@gpu
def test_redirect_gather_under_cuda_graph_reads_live_data():
    _import_flashinfer_first()
    torch.manual_seed(6)
    dev = "cuda"
    lay = _lay()
    fp8_all, fp4_all = _pools(lay, dev)
    fp4 = fp4_all[: lay.fp4_rows]
    t, k = 4, 256
    table = _slots(lay, t * k, dev, with_pad=True, unique=False).int().view(t, k)
    out = torch.empty_like(table)
    scratch = torch.zeros(t * k, 576, dtype=torch.uint8, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mix.redirect_gather(lay, table, out, fp4, scratch, 99)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            mix.redirect_gather(lay, table, out, fp4, scratch, 99)
    torch.cuda.current_stream().wait_stream(s)
    # new table and new pool contents, replay, compare with the reference on the new data
    table.copy_(_slots(lay, t * k, dev, with_pad=True, unique=False).int().view(t, k))
    fp4.copy_(_fp4_rows(lay.fp4_rows, dev))
    g.replay()
    torch.cuda.synchronize()
    ref_out, rows = mix.redirect_gather_ref(lay, table.cpu(), fp4.cpu(), 99)
    assert torch.equal(out.cpu(), ref_out)
    for i, want in rows.items():
        assert torch.equal(scratch[i].cpu(), want.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
