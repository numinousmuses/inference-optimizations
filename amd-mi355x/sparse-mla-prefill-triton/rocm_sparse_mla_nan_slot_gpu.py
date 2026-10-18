"""GPU regression check (agent rtxsys, 2026-10-03) for the GLM-5.3 Value idle-rank garbage root cause found by k3b300:
sparse MLA kernels in rocm_sparse_mla_prefill.py loading KV slot 0 unmasked for invalid (-1) top-k picks.

Slot 0 is the padded rows' KV write target, so with idle DP ranks and graph padding it can hold any bytes. On gfx950,
with NaN bytes (0x7F / 0xFF) in slot 0, the unmasked kernels return finite but WRONG outputs: rel err 1.0 vs the fp32
reference, measured on mi355x8b GPU 7, 2026-10-03, so isnan checks miss it. The masked loads give 0.0296 (the FP8 P
quantization level). The Triton interpreter does not reproduce this, which makes it a GPU check rather than a CPU
unit test.

    HIP_VISIBLE_DEVICES=<g> bash operations/amdpro/chroot_exec.sh <root> -- \\
        python3 operations/kernels/rocm_sparse_mla_nan_slot_gpu.py      # exit 1 on failure
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rocm_sparse_mla_prefill as M  # noqa: E402


def ref(q, kv, idx, sm):
    q8 = q.cpu().to(torch.float8_e4m3fn).float()
    kk = kv.cpu().view(kv.shape[0], 576).float()
    ii = idx.cpu()
    out = torch.zeros(q.shape[0], q.shape[1], 512)
    for i in range(q.shape[0]):
        sel = ii[i, 0]
        sel = sel[sel >= 0].long()
        keys = kk[sel]
        out[i] = torch.softmax((q8[i] @ keys.T) * sm, -1) @ keys[:, :512]
    return out


def main() -> int:
    g = torch.Generator().manual_seed(0)
    t, h, k, slots = 3, 16, 128, 2000
    q = (torch.randn(t, h, 576, generator=g) * 0.3).to(torch.bfloat16).cuda()
    kv = (torch.randn(slots, 1, 576, generator=g) * 0.5).to(torch.float8_e4m3fn)
    idx = torch.randint(1, slots, (t, 1, k), generator=g, dtype=torch.int32)
    idx[torch.rand(t, 1, k, generator=g) < 0.3] = -1
    idx = idx.cuda()
    sm = 576 ** -0.5
    bad = 0
    for pat in (0x7F, 0xFF):
        kv.view(torch.uint8)[0].fill_(pat)
        kvd = kv.cuda()
        r = ref(q, kvd, idx, sm)
        for name, fn in (("prefill", lambda: M.sparse_mla_prefill_fp8(q, kvd, idx, sm, 512, block_n=64)),
                         ("split", lambda: M.sparse_mla_split_fp8(q, kvd, idx, sm, 512, split=2, block_n=64)),
                         ("split_pf", lambda: M.sparse_mla_split_fp8(q, kvd, idx, sm, 512, split=2, block_n=64,
                                                                    prefetch=True))):
            o = fn()[0].float().cpu()
            err = float((o - r).abs().max() / r.abs().max())
            ok = err < 0.08 and not torch.isnan(o).any()
            bad += not ok
            print(f"slot0={pat:#x} {name}: rel err {err:.4f} {'ok' if ok else 'FAIL'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
