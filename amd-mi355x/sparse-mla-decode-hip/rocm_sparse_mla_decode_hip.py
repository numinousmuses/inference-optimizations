#!/usr/bin/env python3
"""Hand-written HIP sparse MLA decode kernel for gfx950 (GLM-5.3 DSA decode/verify) (rtxsys, 2026-10-04).

Why: the Triton split-K kernel (rocm_sparse_mla_prefill._sparse_mla_split_kernel) needs 442 VGPRs, so it runs one
wave per SIMD. Each warp holds the whole [BN, 512] K tile as a dot operand twice, for QK and for PV. At 72 rows it
takes 70-78 us, against 17 us for the bare 576-B row gather (claim amd.glm.dsa_decode.gather_headroom). Six Triton
restructures gained at most 4-7%.

Design (one 256-thread workgroup per (row, split); 4 waves x 16 heads = 64 heads):
  * Block of BN=32 picked keys: the 32 x 576 B rows are gathered by all 256 threads with 16-B loads into registers
    (issued one block ahead; the split's pick list sits in LDS), then written once to LDS row-major
    (Ks [key][byte], stride 584 B). Ks is the A operand of S^T = K q^T, and through the gfx950 transpose read
    ds_read_b64_tr_b8 also the B operand of O = P V (no V^T copy).
  * QK: per wave, S^T [32 keys x 16 heads] = two 16x16 tiles, 18 MFMA_F32_16x16x32_FP8 k-steps each. The q fragment
    (16 heads x 576 B, 36 VGPRs) stays in registers. Lane l holds S^T[key 4g+i (+16)][head l%16], g = l/16.
  * Online softmax per head: 8 values in the lane, then xor-shuffles over the 4 lane groups. All lanes with the
    same l%16 hold the head's running max / sum.
  * PV: the lane's 8 probabilities ARE its A fragment (row head l%16, k slots 8g..8g+7), if the key order inside the
    32-key k step is permuted to slot(key) = 8*(key%16 / 4) + 4*(key / 16) + key%4. The transpose read takes
    its 8 rows in that slot order, so each B fragment is one ds_read_b64_tr_b8. 32 MFMAs per block into 32 float4 accumulators (128 VGPRs).
  * P is FP8 x 448 with fp32 accumulation, as in the Triton kernel. Partials are written as po [T, S, H, 512] fp32
    plus plse [T, S, H] log2-sum-exp, the Triton kernel's layout, and combined by the same Triton combine kernel.
Invalid picks (index < 0) load zeros and get -inf scores (no slot-0 garbage reads).

    HIP_VISIBLE_DEVICES=<g> bash operations/amdpro/chroot_exec.sh /root/img/20261002-rocm724 -- \\
        python3 operations/kernels/rocm_sparse_mla_decode_hip.py [--check-only]
"""
import argparse
import hashlib
import json
import os
import sys

import torch

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

typedef float f4 __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));

#ifndef WPE
#define WPE 2
#endif
#define NH 64
#define BN 32
#define D 576
#define DV 512
#define KST 584
#define CHUNKS (BN * (D / 16))        // 16-B pieces per block = 1152
#define PER_T ((CHUNKS + 255) / 256)  // 5
#define MAXK 2048

__device__ __forceinline__ int slot_of(int key) {
    return ((key & 15) >> 2) * 8 + (key >> 4) * 4 + (key & 3);
}

__device__ __forceinline__ long pack8(const float* p) {
    int lo = 0, hi = 0;
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(p[0], p[1], lo, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(p[2], p[3], lo, true);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(p[4], p[5], hi, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(p[6], p[7], hi, true);
    return (long)(((unsigned long)(unsigned)hi << 32) | (unsigned)lo);
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(WPE, WPE)))
nm_sparse_mla_decode(const uint8_t* __restrict__ q, long stride_qt, long stride_qh,
                     const uint8_t* __restrict__ kv, long stride_kv,
                     const int* __restrict__ idx, long stride_it, int topk, int split,
                     float* __restrict__ po, float* __restrict__ plse, float sm_scale_log2) {
    __shared__ __attribute__((aligned(16))) uint8_t Ks[BN * KST];
    __shared__ int idx_all[MAXK];
    const long t = blockIdx.x;
    const int sp = blockIdx.y;
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    const int g = lane >> 4, c16 = lane & 15;
    const int per = topk / split;
    const int nblk = per / BN;
    {   // this split's picks -> LDS once (the gathers then wait on one memory latency, not two)
        const int* idx_row = idx + t * stride_it + (long)sp * per;
        for (int i = tid; i < per; i += 256) idx_all[i] = idx_row[i];
    }

#ifdef NM_Q_LDS
    // q in LDS (36 VGPRs back for the prefetch ring); stride 584 B spreads the 16 head rows over the banks
    __shared__ __attribute__((aligned(16))) uint8_t Qs[NH * KST];
    for (int i = tid; i < NH * 36; i += 256) {
        const int h = i / 36, part = i - h * 36;
        const uint4 v = *(const uint4*)(q + t * stride_qt + (long)h * stride_qh + part * 16);
        *(uint2*)(Qs + h * KST + part * 16) = make_uint2(v.x, v.y);
        *(uint2*)(Qs + h * KST + part * 16 + 8) = make_uint2(v.z, v.w);
    }
    const uint8_t* qlds = Qs + (wave * 16 + c16) * KST + 8 * g;
#define QF(c) (*(const long*)(qlds + 32 * (c)))
#else
    long qf[18];
    {
        const uint8_t* qrow = q + t * stride_qt + (long)(wave * 16 + c16) * stride_qh;
#pragma unroll
        for (int c = 0; c < 18; ++c) qf[c] = *(const long*)(qrow + 32 * c + 8 * g);
    }
#define QF(c) qf[c]
#endif
    __syncthreads();

    // chunk r of this thread: c = tid + 256 r; key = c % 32 (lanes 4m..4m+3 = keys 4m..4m+3), part = c / 32
    auto gload = [&](uint4* pre, int b) {
#pragma unroll
        for (int r = 0; r < PER_T; ++r) {
            const int c = tid + 256 * r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c < CHUNKS) {
                const int key = c & (BN - 1), part = c / BN;
                const int s = idx_all[b * BN + key];
                if (s >= 0) v = *(const uint4*)(kv + (long)s * stride_kv + part * 16);
            }
            pre[r] = v;
        }
    };
    auto lstore = [&](const uint4* pre) {
        const int j = lane & 3;
#pragma unroll
        for (int r = 0; r < PER_T; ++r) {
            const int c = tid + 256 * r;
            const int key = c & (BN - 1), part = c / BN;
            const unsigned w[4] = {pre[r].x, pre[r].y, pre[r].z, pre[r].w};
            if (c < CHUNKS) {
                uint8_t* kd = Ks + key * KST + part * 16;
                *(uint2*)kd = make_uint2(w[0], w[1]);
                *(uint2*)(kd + 8) = make_uint2(w[2], w[3]);
            }
        }
    };

    f4 acc[32];
#pragma unroll
    for (int d = 0; d < 32; ++d) acc[d] = (f4){0.f, 0.f, 0.f, 0.f};
    // PV B-operand row address of this lane: row r = (lane % 16) / 2 is slot 8g + r -> key; half = lane & 1
    const int vr = c16 >> 1;
    const int vkey = vr < 4 ? 4 * g + vr : 16 + 4 * g + (vr - 4);
    const uint8_t* vrow = Ks + vkey * KST + (lane & 1) * 8;
    float m_run = -INFINITY, l_run = 0.f;

    uint4 preA[PER_T];
#ifdef NM_DEPTH2
    uint4 preB[PER_T];
#endif
    gload(preA, 0);
    lstore(preA);
#ifdef NM_DEPTH2
    if (nblk > 1) gload(preA, 1);
#endif
    __syncthreads();
    // default depth 1: one register buffer; block b+1 is issued at the top of body(b) (244 VGPRs, no spill).
    // NM_DEPTH2: entry to body(b): LDS holds block b, `nxt` holds b+1 (in flight), `fre` is free (spills 27).
    auto body = [&](int b, uint4* nxt, uint4* fre) {
#ifndef NM_DEPTH2
        if (b + 1 < nblk) gload(nxt, b + 1);
#else
        if (b + 2 < nblk) gload(fre, b + 2);
#endif
        const int* idx_s = idx_all + b * BN;
        f4 s0 = (f4){0.f, 0.f, 0.f, 0.f}, s1 = s0;
#pragma unroll
        for (int c = 0; c < 18; ++c) {
            const long a0 = *(const long*)(Ks + c16 * KST + 32 * c + 8 * g);
            const long a1 = *(const long*)(Ks + (16 + c16) * KST + 32 * c + 8 * g);
            const long qc = QF(c);
            s0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a0, qc, s0, 0, 0, 0);
            s1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a1, qc, s1, 0, 0, 0);
        }
        float sv[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sv[i] = idx_s[4 * g + i] >= 0 ? s0[i] * sm_scale_log2 : -INFINITY;
            sv[4 + i] = idx_s[16 + 4 * g + i] >= 0 ? s1[i] * sm_scale_log2 : -INFINITY;
        }
        float mx = sv[0];
#pragma unroll
        for (int i = 1; i < 8; ++i) mx = fmaxf(mx, sv[i]);
        mx = fmaxf(mx, __shfl_xor(mx, 16));
        mx = fmaxf(mx, __shfl_xor(mx, 32));
        const float m_new = fmaxf(m_run, mx);
        const float m_safe = m_new == -INFINITY ? 0.f : m_new;
        const float alpha = exp2f(m_run - m_safe);
        float p[8], ps = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) { p[i] = exp2f(sv[i] - m_safe); ps += p[i]; }
        ps += __shfl_xor(ps, 16);
        ps += __shfl_xor(ps, 32);
        l_run = l_run * alpha + ps;
        m_run = m_new;
        float p8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) p8[i] = p[i] * 448.f;
        const long pa = pack8(p8);
        f4 al;
        al[0] = __shfl(alpha, 4 * g + 0);
        al[1] = __shfl(alpha, 4 * g + 1);
        al[2] = __shfl(alpha, 4 * g + 2);
        al[3] = __shfl(alpha, 4 * g + 3);
#ifndef NM_NO_PV
#pragma unroll
        for (int d = 0; d < 32; ++d) {
            // ds_read_b64_tr_b8 (probe: rocm_ds_read_tr_probe.py): in each 16-lane group, lanes 2r / 2r+1 address
            // the low / high 8 B of row r; lane o receives byte o of rows 0..7. Rows = the 8 keys of slot
            // 8g..8g+7, bytes = dv d*16..d*16+15, so lane (g, c16) gets V[keys of its slots][d*16 + c16].
            const v2i tr = __builtin_amdgcn_ds_read_tr8_b64_v2i32(
                (__attribute__((address_space(3))) v2i*)(vrow + d * 16));
            const long bv = (long)(((unsigned long)(unsigned)tr[1] << 32) | (unsigned)tr[0]);
            acc[d] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(pa, bv, acc[d] * al, 0, 0, 0);
        }
#else
        acc[0][0] += (float)(pa & 1) * al[0];
#endif
        __syncthreads();
        if (b + 1 < nblk) {
            lstore(nxt);
            __syncthreads();
        }
    };
#ifndef NM_DEPTH2
    for (int b = 0; b < nblk; ++b) body(b, preA, preA);
#else
    for (int b = 0; b < nblk; b += 2) {
        body(b, preA, preB);
        if (b + 1 < nblk) body(b + 1, preB, preA);
    }
#endif
    float li[4], inv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        li[i] = __shfl(l_run, 4 * g + i);
        inv[i] = li[i] == 0.f ? 0.f : 1.f / (li[i] * 448.f);
    }
    const int H = NH;
#ifdef NM_PO_BF16
    uint16_t* pbase = (uint16_t*)po + ((t * split + sp) * H + wave * 16 + 4 * g) * (long)DV + c16;
#pragma unroll
    for (int d = 0; d < 32; ++d) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned u = __float_as_uint(acc[d][i] * inv[i]);
            pbase[(long)i * DV + d * 16] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        }
    }
#else
    float* pbase = po + ((t * split + sp) * H + wave * 16 + 4 * g) * (long)DV + c16;
#pragma unroll
    for (int d = 0; d < 32; ++d) {
#pragma unroll
        for (int i = 0; i < 4; ++i) pbase[(long)i * DV + d * 16] = acc[d][i] * inv[i];
    }
#endif
    if (g == 0) {
        const float lse = l_run == 0.f ? -INFINITY : log2f(l_run) + (m_run == -INFINITY ? 0.f : m_run);
        plse[(t * split + sp) * H + wave * 16 + c16] = lse;
    }
}

void sparse_mla_decode_launch(torch::Tensor q, torch::Tensor kv, torch::Tensor idx, torch::Tensor po,
                              torch::Tensor plse, int64_t split, double sm_scale_log2) {
    const int T = q.size(0);
    const int topk = idx.size(1);
    dim3 grid(T, split);
    nm_sparse_mla_decode<<<grid, dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
                       (const uint8_t*)q.data_ptr(), (long)q.stride(0), (long)q.stride(1),
                       (const uint8_t*)kv.data_ptr(), (long)kv.stride(0),
                       (const int*)idx.data_ptr(), (long)idx.stride(0), topk, (int)split,
                       (float*)po.data_ptr(), (float*)plse.data_ptr(), (float)sm_scale_log2);
}
"""

CPP_SRC = ("void sparse_mla_decode_launch(torch::Tensor q, torch::Tensor kv, torch::Tensor idx, torch::Tensor po, "
           "torch::Tensor plse, int64_t split, double sm_scale_log2);")
SRC_SHA = hashlib.sha256(HIP_SRC.encode()).hexdigest()[:12]
_EXT = {}


def _ext(defines: tuple = ()):
    if defines not in _EXT:
        from torch.utils.cpp_extension import load_inline
        tag = SRC_SHA + "".join("_" + d.lower().replace("=", "") for d in defines)
        bdir = os.path.join(os.environ.get("NUMINOUS_HIP_BUILD_DIR", os.path.expanduser("~/.cache/numinous_hip")),
                            f"nm_smla_{tag}")
        os.makedirs(bdir, exist_ok=True)
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"   # torch's default list adds gfx942/gfx1250 (no this MFMA)
        _EXT[defines] = load_inline(name=f"nm_smla_{tag}", cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                                    functions=["sparse_mla_decode_launch"], build_directory=bdir,
                                    extra_cuda_cflags=["-O3", "-Rpass-analysis=kernel-resource-usage"]
                                    + [f"-D{d}" for d in defines],
                                    verbose=os.environ.get("NUMINOUS_HIP_VERBOSE") == "1")
    return _EXT[defines]


import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _combine_kernel(po_ptr, plse_ptr, o_ptr, stride_pt, stride_ps, stride_ph, stride_lt, stride_ls,
                    stride_ot, stride_oh, SPLIT: tl.constexpr, DV: tl.constexpr):
    """Flash-decoding combine of the per-split partials (same math as rocm_sparse_mla_prefill's combine)."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLIT)
    offs_d = tl.arange(0, DV)
    lse = tl.load(plse_ptr + t * stride_lt + offs_s * stride_ls + h)
    m = tl.max(lse, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse - m_safe)
    wsum = tl.sum(w, 0)
    w = w / tl.where(wsum == 0.0, 1.0, wsum)
    po = tl.load(po_ptr + t * stride_pt + offs_s[:, None] * stride_ps + h * stride_ph + offs_d[None, :])
    out = tl.sum(po.to(tl.float32) * w[:, None], 0)
    tl.store(o_ptr + t * stride_ot + h * stride_oh + offs_d, out.to(tl.bfloat16))


# best split per row count, bf16 partials (rocm_sparse_mla_decode_hip.py on mi355x8b GPU 4, 2026-10-04,
# research/gpu-evidence/rtxsys/amd/smla_hip_mi355x8b.json): rows -> split for rows <= key
SPLIT_TABLE = ((48, 8), (160, 4), (224, 2), (1 << 30, 4))
DEFAULT_DEFINES = ("NM_PO_BF16",)


def split_for_rows(rows: int, table=None) -> int:
    if table is None:
        env = os.environ.get("NUMINOUS_SPARSE_DECODE_HIP_SPLIT_TABLE", "")
        table = tuple(tuple(int(x) for x in kv.split(":")) for kv in env.split(",") if kv) or SPLIT_TABLE
    for limit, split in table:
        if rows <= limit:
            return split
    return table[-1][1]


def sparse_mla_decode_hip(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float,
                          d_v: int = 512, split: int = 0, defines: tuple = DEFAULT_DEFINES) -> torch.Tensor:
    """Drop-in for rocm_sparse_mla_prefill.sparse_mla_split_fp8 (64 heads, d 576, d_v 512, FP8 e4m3fn KV);
    returns [1, T, H, d_v] bf16. split 0 = split_for_rows(T). CUDA-graph safe (no host sync)."""
    t, h, d = q.shape
    k = indices.shape[-1]
    if not split:
        split = split_for_rows(t)
    assert h == 64 and d == 576 and d_v == 512, (h, d, d_v)
    assert k % (split * 32) == 0 and k // split <= 2048, (k, split)
    out = torch.empty(1, t, h, d_v, dtype=torch.bfloat16, device=q.device)
    if t == 0:
        return out
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
    q = q.contiguous()
    idx = indices.reshape(t, k)
    if idx.dtype != torch.int32:
        idx = idx.to(torch.int32)
    idx = idx.contiguous()
    kv2 = kv.view(kv.shape[0], d)
    po = torch.empty(t, split, h, d_v, dtype=torch.bfloat16 if "NM_PO_BF16" in defines else torch.float32,
                     device=q.device)
    plse = torch.empty(t, split, h, dtype=torch.float32, device=q.device)
    _ext(tuple(defines)).sparse_mla_decode_launch(q, kv2, idx, po, plse, split, sm_scale * 1.4426950408889634)
    _combine_kernel[(t, h)](po, plse, out, po.stride(0), po.stride(1), po.stride(2), plse.stride(0), plse.stride(1),
                            out.stride(1), out.stride(2), SPLIT=split, DV=d_v, num_warps=4)
    return out


def reference(q8: torch.Tensor, kv8: torch.Tensor, idx: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """fp32 attention over the picked rows (invalid picks masked)."""
    t = q8.shape[0]
    outs = []
    for r in range(t):
        ii = idx[r].long()
        valid = ii >= 0
        k = kv8[ii.clamp(min=0)].float()                    # [K, 576]
        s = (q8[r].float() @ k.T) * sm_scale                # [H, K]
        s = s.masked_fill(~valid[None, :], float("-inf"))
        p = torch.softmax(s, -1)
        outs.append(p @ k[:, :512])
    return torch.stack(outs)


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rocm_fp8_smallm_bench import graph_us
    from rocm_sparse_mla_prefill import sparse_mla_prefill_fp8, sparse_mla_split_fp8
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--rows", default="12,24,48,72,96,128,192,288")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    g = torch.Generator().manual_seed(0)
    slots = 2_000_000
    kv = (torch.randn(slots, 1, 576, generator=g) * 0.5).to(torch.float8_e4m3fn).cuda()
    sm = 576 ** -0.5
    res = {"src_sha": SRC_SHA, "checks": [], "timing": []}
    # correctness: random picks, -1 tail padding, an all-invalid split, duplicate picks
    for rows, split in ((4, 1), (5, 2), (7, 4), (3, 8)):
        q = (torch.randn(rows, 64, 576, generator=g) * 0.3).to(torch.float8_e4m3fn).cuda()
        idx = torch.randint(0, slots, (rows, 1, 2048), generator=g, dtype=torch.int32)
        idx[1, 0, 1500:] = -1
        idx[2, 0, :1024] = -1
        idx[0, 0, 100:200] = idx[0, 0, 0]
        idx = idx.cuda()
        o = sparse_mla_decode_hip(q, kv, idx, sm, split=split, defines=())[0].float()
        for dfs in (("NM_DEPTH2",), ("NM_Q_LDS",)):
            o1 = sparse_mla_decode_hip(q, kv, idx, sm, split=split, defines=dfs)[0].float()
            assert torch.equal(o, o1), f"{dfs} build differs"
        o2 = sparse_mla_decode_hip(q, kv, idx, sm, split=split, defines=("NM_PO_BF16",))[0].float()
        ref = reference(q, kv.view(slots, 576), idx.view(rows, 2048), sm)
        rel_bf = float((o2 - ref).abs().max() / ref.abs().max())
        tri = sparse_mla_split_fp8(q, kv, idx, sm, 512, split=max(2, split))[0].float()
        rel = float((o - ref).abs().max() / ref.abs().max())
        rel_tri = float((tri - ref).abs().max() / ref.abs().max())
        res["checks"].append({"rows": rows, "split": split, "rel_vs_fp32": round(rel, 4),
                              "triton_rel_vs_fp32": round(rel_tri, 4), "bf16po_rel_vs_fp32": round(rel_bf, 4)})
        print(json.dumps(res["checks"][-1]), flush=True)
    if a.check_only:
        return 0
    for rows in (int(x) for x in a.rows.split(",")):
        q = (torch.randn(rows, 64, 576, generator=g) * 0.3).to(torch.float8_e4m3fn).cuda()
        idx = torch.randint(0, slots, (rows, 1, 2048), generator=g, dtype=torch.int32).cuda()
        row = {"rows": rows}
        if rows >= 128:   # production picks the one-pass kernel here (rocm_prefill_hooks tl_sparse_prefill)
            row["triton_shipped"] = graph_us(lambda: sparse_mla_prefill_fp8(q, kv, idx, sm, 512), iters=20)
        else:
            tsplit = 8 if rows <= 32 else (4 if rows <= 64 else 2)
            row["triton_shipped"] = graph_us(lambda: sparse_mla_split_fp8(q, kv, idx, sm, 512, split=tsplit),
                                             iters=20)
        for split in (1, 2, 4, 8):
            row[f"hip_split{split}"] = graph_us(lambda s=split: sparse_mla_decode_hip(q, kv, idx, sm, split=s, defines=()),
                                                iters=20)
        row["hip_auto"] = graph_us(lambda: sparse_mla_decode_hip(q, kv, idx, sm), iters=20)
        for dfs in (("NM_PO_BF16",),):
            for split in (2, 4, 8):
                row[f"hip_split{split}_" + "+".join(dfs)] = graph_us(
                    lambda d=dfs, s=split: sparse_mla_decode_hip(q, kv, idx, sm, split=s, defines=d), iters=20)
        print(json.dumps(row), flush=True)
        res["timing"].append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
