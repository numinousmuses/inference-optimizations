"""CPU (Triton interpreter) tests for symm_collectives: eight peers are eight plain tensors in one process."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

from symm_collectives import pull_reduce_rows, push_rows  # noqa: E402


def _offsets(counts):
    offs, acc = [], 0
    for c in counts:
        offs.append(acc)
        acc += c
    return offs, acc


@pytest.mark.parametrize("counts", [[3, 1, 4, 1, 5, 9, 2, 6], [0, 7, 0, 0, 2, 0, 1, 0], [4] * 8])
def test_push_gather_is_variable_length_and_exact(counts):
    """Every rank pushes its own rows at its SUM_LEN offset; afterwards every peer holds the concatenation."""
    torch.manual_seed(0)
    world, width = 8, 7440                      # MXFP8 row + scales + top-k, bytes
    offs, total = _offsets(counts)
    local = [torch.randint(0, 256, (c, width), dtype=torch.uint8) for c in counts]
    peers = [torch.zeros(total + 3, width, dtype=torch.uint8) for _ in range(world)]
    for r in range(world):
        push_rows(local[r], peers, offs[r])
    want = torch.cat(local) if total else torch.empty(0, width, dtype=torch.uint8)
    for p in peers:
        assert torch.equal(p[:total], want)
        assert torch.count_nonzero(p[total:]) == 0          # nothing written past the gathered rows


def test_push_rows_carries_any_dtype_bitwise():
    """fp8 payloads, e8m0 scales and int32/fp32 top-k all travel as bytes."""
    world = 8
    counts = [2] * world
    offs, total = _offsets(counts)
    ids = [torch.randint(0, 384, (2, 6), dtype=torch.int32) for _ in range(world)]
    wts = [torch.rand(2, 6, dtype=torch.float32) for _ in range(world)]
    pid = [torch.zeros(total, 6, dtype=torch.int32) for _ in range(world)]
    pw = [torch.zeros(total, 6, dtype=torch.float32) for _ in range(world)]
    for r in range(world):
        push_rows(ids[r], pid, offs[r])
        push_rows(wts[r], pw, offs[r])
    for p, q in zip(pid, pw):
        assert torch.equal(p, torch.cat(ids))
        assert torch.equal(q, torch.cat(wts))


@pytest.mark.parametrize("with_shared", [False, True])
def test_pull_reduce_is_scaled_sum_plus_shared(with_shared):
    torch.manual_seed(1)
    world, hidden = 8, 7168
    counts = [5, 0, 3, 8, 1, 2, 7, 4]
    offs, total = _offsets(counts)
    partial = [torch.randn(total, hidden).to(torch.bfloat16) for _ in range(world)]
    scale = 2.5                                   # DeepSeek V4's routed_scaling_factor
    ref_sum = torch.stack([p.float() for p in partial]).sum(0) * scale
    for r in range(world):
        n = counts[r]
        shared = torch.randn(n, hidden).to(torch.bfloat16) if with_shared else None
        out = torch.empty(n, hidden, dtype=torch.bfloat16)
        pull_reduce_rows(partial, offs[r], n, out, scale=scale, shared=shared)
        want = ref_sum[offs[r]:offs[r] + n] + (shared.float() if with_shared else 0)
        torch.testing.assert_close(out.float(), want, rtol=1e-2, atol=2e-2)


def test_pull_reduce_order_is_identical_on_every_rank():
    """The same rows reduced by two different ranks' calls give bit-identical results (fixed peer order)."""
    torch.manual_seed(2)
    partial = [torch.randn(6, 256).to(torch.bfloat16) for _ in range(8)]
    a = pull_reduce_rows(partial, 0, 6, torch.empty(6, 256, dtype=torch.bfloat16), scale=1.0)
    b = pull_reduce_rows(partial, 0, 6, torch.empty(6, 256, dtype=torch.bfloat16), scale=1.0)
    assert torch.equal(a, b)


def test_bad_shapes_are_refused():
    with pytest.raises(ValueError):
        push_rows(torch.zeros(2, 3, dtype=torch.uint8), [torch.zeros(4, 3, dtype=torch.uint8)], 0)   # 3-byte rows
    with pytest.raises(ValueError):
        push_rows(torch.zeros(2, 8, dtype=torch.uint8), [torch.zeros(2, 8, dtype=torch.uint8)], 1)   # past the end
    with pytest.raises(ValueError):
        pull_reduce_rows([torch.zeros(2, 8, dtype=torch.bfloat16)], 1, 2, torch.empty(2, 8, dtype=torch.bfloat16))


def test_pull_reduce_zeroes_rows_past_the_device_count():
    """A graph captured for a padded block of 6 rows, replayed with 4 real rows: rows 4-5 come out zero."""
    partial = [torch.randn(12, 256).to(torch.bfloat16) for _ in range(8)]
    out = torch.full((6, 256), 7.0, dtype=torch.bfloat16)
    pull_reduce_rows(partial, 6, 6, out, scale=1.0, valid=torch.tensor([4], dtype=torch.int32))
    want = torch.stack([p.float() for p in partial]).sum(0)[6:10]
    torch.testing.assert_close(out[:4].float(), want, rtol=1e-2, atol=2e-2)
    assert torch.count_nonzero(out[4:]) == 0
