"""Exact chain speculative sampling for a deterministic (greedy, topk 1) EAGLE draft on ROCm.

Why: SGLang on HIP has no tree_speculative_sampling_target_only kernel, so EAGLE with temperature > 0 auto-enables
rejection sampling (arg_groups/speculative_hook._should_auto_enable_hip_rejection_sampling). That makes the draft
SAMPLE every step from a full-vocab softmax (spec_utils.sample_draft_proposal: softmax + exponential race + PyTorch
argmax over 154,880 logits per row, plus a bs x steps x vocab fp32 probs stack) and forbids a hot-vocab token map.
In the GLM-5.3 MI355X decode profile (amdopt P8) the PyTorch argmax reduce + softmax were 12.8 + 6.3 ms of 153 ms
traced GPU time. The CUDA path that the B300 cells serve instead drafts greedily (draft_topk1 argmax kernel) and
verifies target-only.

What: the draft proposes d_i = argmax q_i, i.e. q_i is one-hot at d_i. Leviathan rejection sampling with that q is
  accept d_i  with probability min(1, p_i(d_i) / q_i(d_i)) = p_i(d_i)      (coin < p_i(d_i))
  on the first rejection at row r: sample from norm(relu(p_r - onehot(d_r))) = p_r with d_r's mass removed
  all accepted: sample the bonus token from p_last
which leaves the output distribution exactly p (target after temperature / top-k / top-p renorm). CUDA's
target-only kernel resamples from p_r itself (it passes zeros for q), slightly over-weighting the rejected token;
this kernel is the exact form. A greedy row (renormed p one-hot) accepts iff d == argmax p and otherwise commits
argmax p, which is greedy decoding.

Same call signature as sglang.kernels.ops.speculative.reject_sampling.chain_speculative_sampling_triton (draft_probs,
retrive_next_token / sibling, thresholds and `deterministic` are accepted and ignored; thresholds must be 1.0, as the
auto-enable predicate already requires for rejection sampling). Installed by operations/patches/rocm_spec_hooks.py as
sglang/kernels/ops/speculative/numinous_onehot_verify.py.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _onehot_chain_kernel(
    Predicts, AcceptIndex, AcceptTokenNum, Candidates, RetriveIndex, UniformSamples, UniformSamplesFinal, TargetProbs,
    stride_cand_b, stride_cand_s, stride_idx_b, stride_idx_s, stride_uni_b, stride_uni_s,
    stride_tp_b, stride_tp_s, stride_tp_v,
    NUM_SLOTS: tl.constexpr, VOCAB_SIZE: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    cand_base = Candidates + pid * stride_cand_b
    idx_base = RetriveIndex + pid * stride_idx_b
    uni_base = UniformSamples + pid * stride_uni_b

    root = tl.load(idx_base)
    tl.store(AcceptIndex + pid * stride_idx_b, root)
    last = root
    num_accept = 0
    row = 0
    step = 1
    going = 1
    rejected = -1
    while (step < NUM_SLOTS) and (going == 1):
        tok = tl.load(cand_base + step * stride_cand_s)
        p = tl.load(TargetProbs + pid * stride_tp_b + row * stride_tp_s + tok.to(tl.int64) * stride_tp_v)
        coin = tl.load(uni_base + (step - 1) * stride_uni_s)
        if coin < p:
            num_accept += 1
            row = step
            tl.store(Predicts + last, tok)
            cur = tl.load(idx_base + step * stride_idx_s)
            tl.store(AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s, cur)
            last = cur
            step += 1
        else:
            rejected = tok.to(tl.int32)
            going = 0
    tl.store(AcceptTokenNum + pid, num_accept)

    # final token: residual p_row minus the rejected token (rejected == -1 when every draft was accepted -> p_row)
    u = tl.load(UniformSamplesFinal + pid)
    base = TargetProbs + pid * stride_tp_b + row * stride_tp_s
    total = 0.0
    for v0 in range(0, VOCAB_SIZE, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        val = tl.load(base + offs.to(tl.int64) * stride_tp_v, mask=offs < VOCAB_SIZE, other=0.0)
        val = tl.where((offs == rejected) | (val < 0.0), 0.0, val)
        total += tl.sum(val)
    target = u * total
    cum = 0.0
    final = -1
    best = -1
    best_p = -1.0
    for v0 in range(0, VOCAB_SIZE, BLOCK_V):
        if final < 0:
            offs = v0 + tl.arange(0, BLOCK_V)
            val = tl.load(base + offs.to(tl.int64) * stride_tp_v, mask=offs < VOCAB_SIZE, other=0.0)
            val = tl.where((offs == rejected) | (val < 0.0), 0.0, val)
            # fallback for a degenerate residual (all mass on the rejected token, numerically ~impossible): the
            # largest remaining entry, never an arbitrary vocab id
            bmax = tl.max(val, axis=0)
            if bmax > best_p:
                best_p = bmax
                best = v0 + tl.argmax(val, axis=0)
            run = cum + tl.cumsum(val, axis=0)
            hit = (run > target) & (val > 0.0)
            if tl.max(hit.to(tl.int32), axis=0) > 0:
                final = v0 + tl.argmax(hit.to(tl.int32), axis=0)
            cum += tl.sum(val)
    if final < 0:
        final = best
    if final < 0:
        final = 0
    tl.store(Predicts + last, final)


def chain_speculative_sampling_onehot(
    predicts, accept_index, accept_token_num, candidates, retrive_index, retrive_next_token, retrive_next_sibling,
    uniform_samples, uniform_samples_for_final_sampling, target_probs, draft_probs=None, threshold_single=1.0,
    threshold_acc=1.0, deterministic=True,
):
    """Drop-in for chain_speculative_sampling_triton / tree_speculative_sampling_target_only with a greedy chain
    draft (topk 1). target_probs: [bs, num_slots, vocab] fp32 (after temperature and top-k / top-p renorm)."""
    if threshold_single != 1.0 or threshold_acc != 1.0:
        raise ValueError("numinous one-hot verify is exact rejection sampling; accept thresholds must be 1.0")
    bs, num_slots = candidates.shape
    if bs == 0:
        return
    assert target_probs.dim() == 3 and target_probs.shape[0] == bs and target_probs.shape[1] == num_slots, \
        (tuple(target_probs.shape), bs, num_slots)
    _onehot_chain_kernel[(bs,)](
        predicts, accept_index, accept_token_num, candidates, retrive_index, uniform_samples,
        uniform_samples_for_final_sampling, target_probs,
        candidates.stride(0), candidates.stride(1), retrive_index.stride(0), retrive_index.stride(1),
        uniform_samples.stride(0), uniform_samples.stride(1),
        target_probs.stride(0), target_probs.stride(1), target_probs.stride(2),
        NUM_SLOTS=num_slots, VOCAB_SIZE=target_probs.shape[-1], BLOCK_V=4096,
    )


def reference(candidates, retrive_index, coins, coin_final, target_probs):
    """Plain PyTorch reference of the same rule (tests): returns predicts [bs*num_slots], accept_index [bs, num_slots]
    (-1 padded), accept_token_num [bs]."""
    import torch
    bs, n = candidates.shape
    predicts = torch.zeros(bs * n, dtype=torch.int32)
    acc_idx = torch.full((bs, n), -1, dtype=torch.int32)
    acc_num = torch.zeros(bs, dtype=torch.int32)
    tp = target_probs.double().cpu()
    for b in range(bs):
        last = int(retrive_index[b, 0])
        acc_idx[b, 0] = last
        row, k, rejected = 0, 0, -1
        for s in range(1, n):
            tok = int(candidates[b, s])
            if float(coins[b, s - 1]) < float(tp[b, row, tok]):
                k += 1
                row = s
                predicts[last] = tok
                last = int(retrive_index[b, s])
                acc_idx[b, k] = last
            else:
                rejected = tok
                break
        acc_num[b] = k
        res = tp[b, row].clone().clamp_min(0)
        if rejected >= 0:
            res[rejected] = 0
        cdf = torch.cumsum(res, 0)
        t = float(coin_final[b]) * float(cdf[-1])
        hit = torch.nonzero((cdf > t) & (res > 0))
        predicts[last] = int(hit[0]) if len(hit) else int(torch.argmax(res))
    return predicts, acc_idx, acc_num
