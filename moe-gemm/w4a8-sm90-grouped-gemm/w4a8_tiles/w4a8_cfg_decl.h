#pragma once
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#define W4A8_ARGS                                                                                            \
  torch::Tensor &d_tensors, torch::Tensor const &a_tensors, torch::Tensor const &b_tensors,                  \
      torch::Tensor const &a_scales, torch::Tensor const &b_scales, torch::Tensor const &expert_offsets,     \
      torch::Tensor const &problem_sizes, torch::Tensor const &a_strides, torch::Tensor const &b_strides,    \
      torch::Tensor const &d_strides, torch::Tensor const &s_strides, int64_t chunk_size
#define W4A8_PASS                                                                                            \
  d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets, problem_sizes, a_strides, b_strides, \
      d_strides, s_strides, chunk_size
