"""Fused KDA prefill front for GLM-5.3-Flash: causal conv1d + SiLU + per-head q/k l2norm + token-major q/k/v store, in
ONE Triton kernel (amdflash, Hopper kernel program 2026-10-05; rtxsys findings Part 6.3 fusion #4).

Stock v0.5.20 path per KDA layer per extend (layers/attention/linear/kda_backend.py:843-857, fla/kda.py:1230-1238):
  causal_conv1d_fn(mixed_qkv.T)      kernels/ops/mamba/causal_conv1d_triton.py  (writes [n, 3*H*D] bf16)
  qkv.split -> q, k, v (strided column slices of the packed output)
  l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous()), v.contiguous()   (3 copies + 2 norm kernels)
F4 prefill profile (TP0, 4,096 tokens per rank per step, ms per step over 34 layers): conv1d 5.46, direct_copy 8.32,
l2norm 2.19 = ~16 ms. This kernel reads mixed_qkv once and writes q, k, v once (~402 MB per layer at 4,096 tokens):
about 3.6 ms per step at 80% of H200 HBM bandwidth (derived; measured by flash_kda_prefill_bench.py).

Semantics match the stock pair exactly, including the bf16 rounding of the conv output before the l2norm and the
conv-state update:
  y[t, c] = silu(bias[c] + sum_j w[c, j] * x_ext[t - (W-1) + j, c]),  x_ext = [conv state (W-1 rows, oldest first) if
            has_initial_state else zeros, x[seq]]
  q/k heads: y / sqrt(sum(y^2) + eps) over the head's D channels (fla l2norm_fwd_kernel, eps 1e-6)
  conv_states[slot] <- the last W-1 rows of x_ext (stock: the program of the sequence's first chunk writes it after
            reading the old state; here the same: the sequence's first token block reads, then writes)

    q, k, v = conv_l2norm_qkv(mixed_qkv, conv_w, conv_b, conv_states, cache_indices, has_initial_state,
                              query_start_loc, H, D)          # each [1, n, H, D] bf16, contiguous

conv_states layout: [slots, W-1, dim] (the KDA pool's raw layout, kda_backend.py:812 passes .transpose(-1, -2) to the
stock kernel). Reference: conv_l2norm_ref (pure torch, CPU-testable).
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001  (CPU-only import of the reference)
    triton = None
    tl = None

EPS = 1e-6


def l2norm_ref(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """fla l2norm_fwd semantics on the last dim (fp32 math, input dtype out)."""
    xf = x.float()
    return (xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + eps)).to(x.dtype)


def conv_ref(x: torch.Tensor, w: torch.Tensor, b, conv_states: torch.Tensor, cache_indices: torch.Tensor,
             has_initial_state: torch.Tensor, query_start_loc: torch.Tensor) -> torch.Tensor:
    """Stock causal_conv1d_fn semantics (varlen, conv-state in/out, SiLU), token-major [n, dim] in and out.
    Updates conv_states in place. Output rounded to x.dtype like the stock kernel's store."""
    n, dim = x.shape
    W = w.shape[1]
    out = torch.empty_like(x)
    cu = query_start_loc.tolist()
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        slot = int(cache_indices[i])
        prior = conv_states[slot].to(torch.float32) if bool(has_initial_state[i]) else \
            torch.zeros(W - 1, dim, dtype=torch.float32, device=x.device)
        xe = torch.cat([prior, x[s:e].to(torch.float32)], 0)              # [W-1 + len, dim]
        acc = torch.zeros(e - s, dim, dtype=torch.float32, device=x.device)
        for j in range(W):
            acc += xe[j:j + (e - s)] * w[:, j].to(torch.float32)[None, :]
        if b is not None:
            acc += b.to(torch.float32)[None, :]
        out[s:e] = (acc * torch.sigmoid(acc)).to(x.dtype)
        conv_states[slot] = xe[-(W - 1):].to(conv_states.dtype)
    return out


def conv_l2norm_ref(x, w, b, conv_states, cache_indices, has_initial_state, query_start_loc, H: int, D: int):
    """The stock chain end to end: conv -> split -> l2norm(q), l2norm(k), v; each [1, n, H, D] contiguous."""
    y = conv_ref(x, w, b, conv_states, cache_indices, has_initial_state, query_start_loc)
    q, k, v = y.split([H * D, H * D, H * D], dim=-1)
    q = l2norm_ref(q.unflatten(-1, (H, D))).unsqueeze(0).contiguous()
    k = l2norm_ref(k.unflatten(-1, (H, D))).unsqueeze(0).contiguous()
    v = v.unflatten(-1, (H, D)).unsqueeze(0).contiguous()
    return q, k, v


def block_plan(seq_lens: list[int], bt: int) -> tuple[list[int], list[int]]:
    """(sequence id, block start token) for every BT-token block; blocks never cross a sequence boundary."""
    seqs, starts, off = [], [], 0
    for i, ln in enumerate(seq_lens):
        for t0 in range(0, ln, bt):
            seqs.append(i)
            starts.append(off + t0)
        off += ln
    return seqs, starts


if triton is not None:
    @triton.jit
    def _conv_l2norm_qkv_kernel(
        x_ptr, w_ptr, b_ptr, cs_ptr, ci_ptr, his_ptr, cu_ptr, bseq_ptr, bstart_ptr,
        q_ptr, k_ptr, v_ptr,
        stride_x_tok, stride_cs_slot, stride_cs_row,
        H: tl.constexpr, D: tl.constexpr, W: tl.constexpr, BT: tl.constexpr, HAS_BIAS: tl.constexpr,
        EPS_C: tl.constexpr, SP: tl.constexpr,
    ):
        pid_b = tl.program_id(0)          # token block
        hs = tl.program_id(1)             # head slot in [0, 3H): q heads, k heads, v heads
        seq = tl.load(bseq_ptr + pid_b)
        t0 = tl.load(bstart_ptr + pid_b)
        s0 = tl.load(cu_ptr + seq)
        s1 = tl.load(cu_ptr + seq + 1)
        slot = tl.load(ci_ptr + seq).to(tl.int64)
        slot_ok = slot >= 0                               # PAD_SLOT_ID (-1): no state read, no state write (stock)
        has_init = tl.load(his_ptr + seq) * slot_ok.to(tl.int32)
        ch = hs * D + tl.arange(0, D)     # channel ids of this head in the packed [q | k | v] width
        rows = t0 + tl.arange(0, BT)      # global token ids of this block
        live = rows < s1
        # taps: x_ext[t - (W-1) + j]; rows before s0 come from the conv state (oldest first) or zeros
        acc = tl.zeros((BT, D), dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(b_ptr + ch).to(tl.float32)[None, :]
        for j in tl.static_range(W):
            src = rows - (W - 1) + j                      # token id of this tap
            in_x = (src >= s0) & live
            xv = tl.load(x_ptr + src.to(tl.int64)[:, None] * stride_x_tok + ch[None, :],
                         mask=in_x[:, None], other=0.0).to(tl.float32)
            # from the state: src in [s0-(W-1), s0) -> state row (W-1) - (s0 - src)
            srow = (W - 1) - (s0 - src)
            in_s = (src < s0) & (srow >= 0) & live & (has_init != 0)
            sv = tl.load(cs_ptr + slot * stride_cs_slot + srow.to(tl.int64)[:, None] * stride_cs_row + ch[None, :],
                         mask=in_s[:, None], other=0.0).to(tl.float32)
            wj = tl.load(w_ptr + ch * W + j).to(tl.float32)
            acc += (xv + sv) * wj[None, :]
        y = acc / (1.0 + tl.exp(-acc))                    # SiLU
        y = y.to(tl.bfloat16).to(tl.float32)              # the stock conv stores bf16 before the l2norm reads it
        is_v = hs >= 2 * H
        if not is_v:
            ss = tl.sum(y * y, axis=1)
            y = y / tl.sqrt(ss + EPS_C)[:, None]
        h = hs % H
        dst = rows.to(tl.int64)[:, None] * (H * D) + h * D + tl.arange(0, D)[None, :]
        if hs < H:
            tl.store(q_ptr + dst, y.to(tl.bfloat16), mask=live[:, None])
        elif hs < 2 * H:
            tl.store(k_ptr + dst, y.to(tl.bfloat16), mask=live[:, None])
        else:
            tl.store(v_ptr + dst, y.to(tl.bfloat16), mask=live[:, None])
        # conv-state update by the sequence's first block (it has read the old state above): new state = last W-1
        # rows of x_ext. Every head slot writes its own channels.
        if (t0 == s0) & slot_ok:
            # (names differ from the tap loop's: Triton requires one type per variable across branches)
            r_st = tl.arange(0, SP)                       # new state rows (oldest first), padded to a power of 2
            rok = r_st < (W - 1)
            src_st = s1 - (W - 1) + r_st                  # token id feeding new row r
            fx = (src_st >= s0) & rok
            xv_st = tl.load(x_ptr + src_st.to(tl.int64)[:, None] * stride_x_tok + ch[None, :], mask=fx[:, None],
                            other=0.0)
            orow = (W - 1) - (s0 - src_st)                # old-state row when the sequence is shorter than W-1
            fs = (src_st < s0) & (orow >= 0) & rok & (has_init != 0)
            sv_st = tl.load(cs_ptr + slot * stride_cs_slot + orow.to(tl.int64)[:, None] * stride_cs_row + ch[None, :],
                            mask=fs[:, None], other=0.0)
            tl.debug_barrier()
            new_st = tl.where(fx[:, None], xv_st, sv_st)
            tl.store(cs_ptr + slot * stride_cs_slot + r_st.to(tl.int64)[:, None] * stride_cs_row + ch[None, :], new_st,
                     mask=rok[:, None])


if triton is not None:
    @triton.jit
    def _conv_l2norm_qkv_rowloop_kernel(
        x_ptr, w_ptr, b_ptr, cs_ptr, ci_ptr, his_ptr, cu_ptr, bseq_ptr, bstart_ptr,
        q_ptr, k_ptr, v_ptr,
        stride_x_tok, stride_cs_slot, stride_cs_row,
        H: tl.constexpr, D: tl.constexpr, SEG: tl.constexpr, HAS_BIAS: tl.constexpr, EPS_C: tl.constexpr,
    ):
        """Row-loop variant (W = 4): one program per (segment of SEG tokens inside one sequence, head slot); the three
        previous input rows roll through registers, so every x row is loaded once (the stock conv kernel's pattern),
        and the per-head l2norm is a warp-level reduction over D channels."""
        pid_b = tl.program_id(0)
        hs = tl.program_id(1)
        seq = tl.load(bseq_ptr + pid_b)
        t0 = tl.load(bstart_ptr + pid_b)
        s0 = tl.load(cu_ptr + seq)
        s1 = tl.load(cu_ptr + seq + 1)
        slot = tl.load(ci_ptr + seq).to(tl.int64)
        slot_ok = slot >= 0
        has_init = (tl.load(his_ptr + seq) != 0) & slot_ok
        ch = hs * D + tl.arange(0, D)
        w0 = tl.load(w_ptr + ch * 4 + 0).to(tl.float32)
        w1 = tl.load(w_ptr + ch * 4 + 1).to(tl.float32)
        w2 = tl.load(w_ptr + ch * 4 + 2).to(tl.float32)
        w3 = tl.load(w_ptr + ch * 4 + 3).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(b_ptr + ch).to(tl.float32)
        else:
            bias = tl.zeros((D,), dtype=tl.float32)
        cs_base = cs_ptr + slot * stride_cs_slot + ch
        # previous rows p1 (t-1), p2 (t-2), p3 (t-3)
        if t0 == s0:
            p3 = tl.load(cs_base + 0 * stride_cs_row, mask=(ch >= 0) & has_init, other=0.0).to(tl.float32)
            p2 = tl.load(cs_base + 1 * stride_cs_row, mask=(ch >= 0) & has_init, other=0.0).to(tl.float32)
            p1 = tl.load(cs_base + 2 * stride_cs_row, mask=(ch >= 0) & has_init, other=0.0).to(tl.float32)
        else:
            p3 = tl.load(x_ptr + (t0 - 3).to(tl.int64) * stride_x_tok + ch).to(tl.float32)
            p2 = tl.load(x_ptr + (t0 - 2).to(tl.int64) * stride_x_tok + ch).to(tl.float32)
            p1 = tl.load(x_ptr + (t0 - 1).to(tl.int64) * stride_x_tok + ch).to(tl.float32)
        h = hs % H
        is_q = hs < H
        is_k = (hs >= H) & (hs < 2 * H)
        is_qk = hs < 2 * H
        n_tok = tl.minimum(SEG, s1 - t0)
        for i in range(0, n_tok):
            t = t0 + i
            xv = tl.load(x_ptr + t.to(tl.int64) * stride_x_tok + ch).to(tl.float32)
            acc = bias + w0 * p3 + w1 * p2 + w2 * p1 + w3 * xv
            y = acc / (1.0 + tl.exp(-acc))
            y = y.to(tl.bfloat16).to(tl.float32)
            if is_qk:
                y = y / tl.sqrt(tl.sum(y * y, axis=0) + EPS_C)
            dst = t.to(tl.int64) * (H * D) + h * D + tl.arange(0, D)
            if is_q:
                tl.store(q_ptr + dst, y.to(tl.bfloat16))
            elif is_k:
                tl.store(k_ptr + dst, y.to(tl.bfloat16))
            else:
                tl.store(v_ptr + dst, y.to(tl.bfloat16))
            p3 = p2
            p2 = p1
            p1 = xv
        # conv-state update by the sequence's first segment (it has read the old state above)
        if (t0 == s0) & slot_ok:
            ln = s1 - s0
            # new rows r = 0, 1, 2 hold x_ext[len-3+r]: from x when that index >= 0, else from the old state
            for r in tl.static_range(3):
                src = s1 - 3 + r
                if src >= s0:
                    val = tl.load(x_ptr + src.to(tl.int64) * stride_x_tok + ch)
                else:
                    orow = 3 - (s0 - src)
                    val = tl.load(cs_base + orow * stride_cs_row, mask=(ch >= 0) & has_init, other=0.0)
                tl.debug_barrier()
                tl.store(cs_base + r * stride_cs_row, val)


_PLAN_CACHE: dict = {}


def _plan_tensors(seq_lens: list[int], bt: int, device) -> tuple:
    """Device copies of the block plan, cached per (seq lens, bt, device): the 34 KDA layers of one forward share one
    batch, so only the first layer pays the host list build + H2D copy (measured ~0.14 ms per call uncached)."""
    key = (tuple(seq_lens), bt, str(device))
    hit = _PLAN_CACHE.get(key)
    if hit is None:
        bseq, bstart = block_plan(seq_lens, bt)
        hit = (torch.tensor(bseq, dtype=torch.int32, device=device), torch.tensor(bstart, dtype=torch.int32, device=device))
        if len(_PLAN_CACHE) > 64:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = hit
    return hit


def usable(layer, x: torch.Tensor, conv_raw: torch.Tensor) -> bool:
    """The serving hook's gate (flash_kda_front_hooks): env NUMINOUS_KDA_FUSED_FRONT=1 and the shapes this kernel
    covers. Anything else keeps the stock conv + split + l2norm path."""
    import os
    if os.environ.get("NUMINOUS_KDA_FUSED_FRONT") != "1" or triton is None:
        return False
    try:
        d = int(layer.head_q_dim)
        h = int(layer.q_dim) // d
        w = layer.conv_weights
        return (int(layer.q_dim) == int(layer.k_dim) == int(layer.v_dim)
                and d == int(layer.head_k_dim) == int(layer.head_v_dim) and d & (d - 1) == 0
                and x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1 and x.shape[1] == 3 * h * d
                and w.dim() == 2 and w.shape[0] == 3 * h * d and w.shape[1] == 4
                and conv_raw.dim() == 3 and conv_raw.shape[1] == 3 and conv_raw.shape[2] == 3 * h * d
                and conv_raw.stride(2) == 1 and conv_raw.dtype == x.dtype)
    except Exception:  # noqa: BLE001
        return False


def conv_l2norm_qkv(x: torch.Tensor, w: torch.Tensor, b, conv_states: torch.Tensor, cache_indices: torch.Tensor,
                    has_initial_state: torch.Tensor, query_start_loc: torch.Tensor, H: int, D: int,
                    seq_lens: list[int] | None = None, bt: int = 32, num_warps: int | None = None,
                    rowloop_seg: int | None = 64):
    """Fused front (see module doc). x [n, 3*H*D] token-major; returns q, k, v each [1, n, H, D] bf16 contiguous.
    seq_lens (host list) avoids a device sync; otherwise it is read from query_start_loc.
    Default: the row-loop kernel, 64-token segments, 1 warp. Measured on 1x H200 (flash_kda_prefill_bench_h200.json):
    0.178 ms at 4,096 tokens and 0.340 ms at 8,192, against the tile kernel's 0.224 / 0.438 ms and the stock chain's
    0.462 / 0.897 ms. rowloop_seg=0 selects the tile kernel (bt rows, 4 warps)."""
    if num_warps is None:
        num_warps = 1 if rowloop_seg else 4
    n, dim = x.shape
    assert dim == 3 * H * D and x.stride(1) == 1 and w.shape[0] == dim
    W = w.shape[1]
    if seq_lens is None:
        cu = query_start_loc.tolist()
        seq_lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    dev = x.device
    bseq_t, bstart_t = _plan_tensors(seq_lens, bt, dev)
    q = torch.empty(1, n, H, D, dtype=x.dtype, device=dev)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    wc = w.contiguous()
    his = has_initial_state.to(torch.int32)
    if rowloop_seg and W == 4:
        bseq_r, bstart_r = _plan_tensors(seq_lens, rowloop_seg, dev)
        _conv_l2norm_qkv_rowloop_kernel[(bseq_r.numel(), 3 * H)](
            x, wc, b if b is not None else wc, conv_states, cache_indices, his, query_start_loc.to(torch.int32),
            bseq_r, bstart_r, q, k, v, x.stride(0), conv_states.stride(0), conv_states.stride(1),
            H=H, D=D, SEG=rowloop_seg, HAS_BIAS=b is not None, EPS_C=EPS, num_warps=num_warps)
        return q, k, v
    grid = (bseq_t.numel(), 3 * H)
    _conv_l2norm_qkv_kernel[grid](
        x, wc, b if b is not None else wc, conv_states, cache_indices, his, query_start_loc.to(torch.int32),
        bseq_t, bstart_t, q, k, v,
        x.stride(0), conv_states.stride(0), conv_states.stride(1),
        H=H, D=D, W=W, BT=bt, HAS_BIAS=b is not None, EPS_C=EPS, SP=triton.next_power_of_2(W - 1), num_warps=num_warps)
    return q, k, v
