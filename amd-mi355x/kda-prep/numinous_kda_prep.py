"""One-pass KDA prefill input prep: split the conv output into q / k / v, L2-normalize q and k per head, write all three
contiguous (k3amd for amdflash, 2026-10-04; coordinator 01:50 "KDA chunk pipeline, if fusable").

Stock path (SGLang 1330e437e1, kda_backend.forward_extend + kda_triton.extend + fla chunk_kda): qkv.split gives strided
views of the [T, 3*H*D] conv output; chunk_kda then runs q.contiguous(), k.contiguous() (copies), l2norm_fwd on each
(two more read+write passes) and v.contiguous() (a third copy). Measured on Flash prefill (stack profile, 35K-token
prompt): the chunk_kda copies are 20.4 ms (1.6% of prefill), l2norm ~1.2%. This kernel reads the conv output once and
writes the three contiguous tensors once: ~1.6 GB of traffic per 16K-token layer-chunk instead of ~2.7 GB, and one
launch instead of five. Numerics equal l2norm_fwd's (fp32, x / sqrt(sum(x^2) + eps), cast to the input dtype); v is a copy.

Env NUMINOUS_KDA_PREP=1; the backend then tells the Triton extend kernel that q / k are already normalized
(use_qk_l2norm_in_kernel=False). A kernel the dispatcher reroutes to that ignores the flag normalizes again, which is
harmless (l2norm of a unit vector is the vector, to rounding).
"""
from __future__ import annotations

import os

import triton
import triton.language as tl


def enabled() -> bool:
    return os.environ.get("NUMINOUS_KDA_PREP") == "1"


@triton.jit
def _kda_split_l2norm_kernel(x, out, eps, T, stride_x, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    j = tl.program_id(1)                       # 0 .. 3H-1: q heads, k heads, v heads
    rows = i_t * BT + tl.arange(0, BT)
    cols = tl.arange(0, D)
    mask = rows[:, None] < T
    b = tl.load(x + rows[:, None] * stride_x + j * D + cols[None, :], mask=mask, other=0.0)
    part = j // H
    h = j % H
    if part < 2:
        bf = b.to(tl.float32)
        var = tl.sum(bf * bf, axis=1)
        b = (bf / tl.sqrt(var + eps)[:, None]).to(out.dtype.element_ty)
    dst = out + part * T * H * D + rows[:, None] * (H * D) + h * D + cols[None, :]
    tl.store(dst, b.to(out.dtype.element_ty), mask=mask)


def split_l2norm(qkv, q_dim: int, k_dim: int, v_dim: int, head_dim: int, eps: float = 1e-6):
    """qkv [T, q_dim + k_dim + v_dim] (unit column stride) -> (q_n, k_n, v) each [T, dim], contiguous."""
    import torch

    assert q_dim == k_dim == v_dim and q_dim % head_dim == 0 and qkv.stride(-1) == 1
    T = qkv.shape[0]
    H = q_dim // head_dim
    out = torch.empty((3, T, q_dim), dtype=qkv.dtype, device=qkv.device)
    BT = 16
    _kda_split_l2norm_kernel[(triton.cdiv(T, BT), 3 * H)](
        qkv, out, eps, T, qkv.stride(0), H=H, D=head_dim, BT=BT, num_warps=4)
    return out[0], out[1], out[2]
