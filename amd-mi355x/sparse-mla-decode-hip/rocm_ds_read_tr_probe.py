#!/usr/bin/env python3
"""Probe the gfx950 LDS transpose read ds_read_b64_tr_b8 (rtxsys, 2026-10-04): which bytes does each lane get?

LDS is filled with byte value = (address % 251). Lane l passes address A(l) = l * 8 (case 'lin') or
(l % 16) * 64 + (l / 16) * 8 (case 'rows'). The kernel returns the 8 bytes each lane received, decoded back to
source addresses. Used to design the V operand path of the HIP sparse MLA decode kernel.
"""
import json
import os

import torch

SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
typedef int v2i __attribute__((ext_vector_type(2)));
extern "C" __global__ void probe(int* out, int mode) {
    __shared__ __attribute__((aligned(16))) unsigned char L[4096];
    for (int i = threadIdx.x; i < 4096; i += 64) L[i] = (unsigned char)(i % 251);
    __syncthreads();
    const int l = threadIdx.x;
    int addr = mode == 0 ? l * 8 : ((l % 16) * 64 + (l / 16) * 8);
    v2i r = __builtin_amdgcn_ds_read_tr8_b64_v2i32((__attribute__((address_space(3))) v2i*)(L + addr));
    out[l * 2] = r[0];
    out[l * 2 + 1] = r[1];
}
void run(torch::Tensor out, int64_t mode) { probe<<<1, 64, 0, at::cuda::getCurrentCUDAStream()>>>((int*)out.data_ptr(), (int)mode); }
"""


def main():
    from torch.utils.cpp_extension import load_inline
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    ext = load_inline(name="nm_trprobe", cpp_sources="void run(torch::Tensor out, int64_t mode);", cuda_sources=SRC,
                      functions=["run"], build_directory=os.path.expanduser("~/.cache/nm_trprobe"), verbose=False)
    res = {}
    for mode, name in ((0, "lin"), (1, "rows")):
        out = torch.zeros(128, dtype=torch.int32, device="cuda")
        ext.run(out, mode)
        b = out.cpu().view(torch.uint8).view(64, 8).tolist()
        res[name] = {l: b[l] for l in (0, 1, 2, 7, 8, 15, 16, 17, 32, 63)}
    print(json.dumps(res))


if __name__ == "__main__":
    os.makedirs(os.path.expanduser("~/.cache/nm_trprobe"), exist_ok=True)
    main()
