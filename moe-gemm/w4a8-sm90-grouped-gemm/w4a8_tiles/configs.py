"""Tile / schedule candidates for sglang-kernel 0.4.7's SM90 W4A8 grouped GEMM (k3amd, GLM MoE GEMM owner).

TileShape <M, N, K>: M = output features (the int4 weight rows), N = tokens (activation rows), K = reduction, as in
w4a8_grouped_mm_c3x.cu (the dispatch grows N with the token count). K / 128 packed scales per tile: the weight
scales must be interleaved with alignment K / 128 (w4afp8.py interleave_scales uses 4 when the group count allows it,
i.e. TileK 512). Cluster <A, B, C>. Each config is one translation unit of the extension (parallel build).
"""
# id: (schedule, M, N, K, cluster_m, cluster_n, cluster_k)
CFGS = {
    0: ("CO", 128, 16, 512, 1, 1, 1),     # stock: decode m <= 32 (generic k % 512 branch)
    1: ("CO", 128, 32, 512, 1, 1, 1),     # stock: decode / small prefill m <= 1024
    2: ("CO", 128, 64, 512, 1, 1, 1),     # stock: prefill m > 1024
    3: ("CO", 128, 128, 512, 1, 1, 1),
    4: ("CO", 128, 64, 128, 1, 1, 1),
    5: ("CO", 128, 128, 128, 1, 1, 1),
    6: ("CO", 256, 64, 128, 1, 1, 1),
    7: ("CO", 256, 128, 128, 1, 1, 1),
    8: ("CO", 128, 128, 256, 1, 1, 1),
    9: ("CO", 256, 16, 128, 1, 1, 1),     # sglang main's H200 GLM-5.2 TP8 down-proj decode tile
    10: ("CO", 128, 16, 128, 1, 1, 1),
    11: ("PP", 128, 64, 128, 1, 1, 1),
    12: ("CO", 128, 128, 512, 2, 1, 1),
    13: ("CO", 256, 32, 128, 1, 1, 1),
    14: ("CO", 128, 32, 128, 1, 1, 1),
    15: ("CO", 128, 16, 256, 1, 1, 1),
    16: ("CO", 128, 32, 256, 1, 1, 1),
    # prefill round 2 (k3amd 2026-10-05 14:0x): token tile 256 and TMA multicast clusters, never tried in round 1
    17: ("CO", 128, 256, 128, 1, 1, 1),
    18: ("CO", 128, 128, 128, 1, 2, 1),
    19: ("CO", 128, 128, 128, 2, 1, 1),
    20: ("CO", 256, 128, 128, 1, 2, 1),
    21: ("CO", 128, 256, 128, 2, 1, 1),
    22: ("PP", 128, 128, 128, 1, 1, 1),
    23: ("CO", 256, 64, 128, 1, 2, 1),
    24: ("CO", 256, 128, 128, 2, 1, 1),
}


def scale_alignment(cfg_id: int) -> int:
    return CFGS[cfg_id][3] // 128


def tu_source(cfg_id: int) -> str:
    s, m, n, k, a, b, c = CFGS[cfg_id]
    return (f'#include "w4a8_cfg_common.cuh"\n'
            f"void k3amd_w4a8_run_cfg_{cfg_id}(W4A8_ARGS) {{\n"
            f"  invoke_cfg<SM90W4A8Config<{m}, {n}, {k}, {a}, {b}, {c}, Sched::{s}>>(W4A8_PASS);\n}}\n")


def entry_source() -> str:
    decls = "".join(f"void k3amd_w4a8_run_cfg_{i}(W4A8_ARGS);\n" for i in CFGS)
    cases = "".join(f"    case {i}: k3amd_w4a8_run_cfg_{i}(W4A8_PASS); return;\n" for i in CFGS)
    return ('#include "w4a8_cfg_decl.h"\n' + decls +
            "void mm_cfg(torch::Tensor d_tensors, torch::Tensor a_tensors, torch::Tensor b_tensors, torch::Tensor a_scales,\n"
            "            torch::Tensor b_scales, torch::Tensor expert_offsets, torch::Tensor problem_sizes,\n"
            "            torch::Tensor a_strides, torch::Tensor b_strides, torch::Tensor d_strides, torch::Tensor s_strides,\n"
            "            int64_t chunk_size, int64_t cfg) {\n"
            "  const c10::cuda::CUDAGuard device_guard(a_tensors.device());\n"
            "  switch (cfg) {\n" + cases +
            '    default: TORCH_CHECK(false, "unknown w4a8 tile config ", cfg);\n  }\n}\n'
            "TORCH_LIBRARY(k3amd_w4a8, m) {\n"
            '  m.def("mm_cfg(Tensor(a!) d, Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor expert_offsets, '
            'Tensor problem_sizes, Tensor a_strides, Tensor b_strides, Tensor d_strides, Tensor s_strides, int chunk_size, '
            'int cfg) -> ()");\n'
            '  m.impl("mm_cfg", torch::kCUDA, &mm_cfg);\n}\n')
