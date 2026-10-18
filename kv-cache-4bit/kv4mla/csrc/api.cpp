// kv4mla torch bindings: GLM-5.3 FP4 sparse MLA decode (glm_fp4_decode.cu) + FlashMLA's split-KV scheduler and combine.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/bfloat16.h>

#include "kernels/params.h"
#include "kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "kernels/smxx/decode/combine/combine.h"

namespace kv4mla {
void run_glm_fp4_sparse_decode(const SparseAttnDecodeParams &params);
void run_glm_fp4_sparse_decode_fp8(const SparseAttnDecodeParams &params);
void trace_read(unsigned long long *host, unsigned int *n);
void trace_read_fp8(unsigned long long *host);
}

// Debug timeline (only with -DKV4MLA_TRACE_ON): [10, 64] int64 clock64 per (event, block) of CTA (0, 0)
at::Tensor trace() {
    cudaDeviceSynchronize();
    auto buf = torch::zeros({10, 64}, torch::dtype(torch::kInt64));
    unsigned int n = 0;
    kv4mla::trace_read((unsigned long long*)buf.data_ptr(), &n);
    if (buf.abs().sum().item<int64_t>() == 0) {   // the fp8-MMA kernel keeps its own buffer
        kv4mla::trace_read_fp8((unsigned long long*)buf.data_ptr());
        n = 640;
    }
    return n ? buf : buf.narrow(0, 0, 0);
}

static constexpr float LOG_2_E = 1.44269504f;
static constexpr int ROW_BYTES = 352, H_Q = 64, D_QK = 576, D_V = 512, B_TOPK = 64;
static constexpr int FIXED_OVERHEAD_BLOCKS = 5;   // FlashMLA's head64 scheduler constant

static int to_int(int64_t v) {
    TORCH_CHECK(v <= std::numeric_limits<int>::max(), "kv4mla: stride exceeds int32: ", v);
    return (int)v;
}

// Number of scheduler partitions FlashMLA's head64 decode uses: one per SM per q position
int64_t num_sm_parts(int64_t s_q) {
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return std::max<int64_t>(sms / s_q, 1);
}

// Split-KV schedule for (b, s_q, topk[, topk_length]). Writes sched_meta [num_sm_parts, 8] int32 and num_splits [b+1].
void sched_meta(int64_t b, int64_t s_q, int64_t topk, const std::optional<at::Tensor> &topk_length,
                at::Tensor &meta, at::Tensor &num_splits, int64_t fixed_overhead_blocks) {
    TORCH_CHECK(meta.dtype() == torch::kInt32 && meta.is_contiguous() && meta.dim() == 2 &&
                meta.size(1) == (int64_t)(sizeof(DecodingSchedMeta) / 4), "kv4mla: bad sched_meta tensor");
    TORCH_CHECK(num_splits.dtype() == torch::kInt32 && num_splits.is_contiguous() && num_splits.numel() == b + 1,
                "kv4mla: bad num_splits tensor");
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length->dtype() == torch::kInt32 && topk_length->is_contiguous() && topk_length->numel() == b);
    }
    at::cuda::CUDAGuard guard{(char)meta.get_device()};
    GetDecodeSchedMetaParams p = {
        (int)b, (int)s_q, B_TOPK, (int)(fixed_overhead_blocks > 0 ? fixed_overhead_blocks : FIXED_OVERHEAD_BLOCKS), (int)topk, 0,
        topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr, nullptr, nullptr,
        (DecodingSchedMeta*)meta.data_ptr(), num_splits.data_ptr<int>(), (int)meta.size(0),
        at::cuda::getCurrentCUDAStream().stream()};
    smxx::decode::run_get_decoding_sched_meta_kernel(p);
}

// out [b, s_q, 64, 512] bf16 <- sparse MLA over the 352-byte FP4 rows.
//   q        [b, s_q, 64, 576] float8_e4m3fn (last dim contiguous)
//   kv       [rows, 352] uint8 (one row per pool slot, contiguous rows)
//   indices  [b, s_q, topk] int32 pool slots; -1 or >= rows masked; topk % 64 == 0
//   lse      [b, s_q, 64] fp32 (natural-log LSE, +inf for rows without a valid slot)
//   split-KV (meta/num_splits from sched_meta for the same b, s_q, topk, topk_length): o_accum
//   [b + num_sm_parts, s_q, 64, 512] fp32 and lse_accum [b + num_sm_parts, s_q, 64] fp32 workspaces.
void sparse_decode(const at::Tensor &q, const at::Tensor &kv, const at::Tensor &indices,
                   const std::optional<at::Tensor> &topk_length, double sm_scale, at::Tensor &out, at::Tensor &lse,
                   const std::optional<at::Tensor> &meta, const std::optional<at::Tensor> &num_splits,
                   const std::optional<at::Tensor> &o_accum, const std::optional<at::Tensor> &lse_accum,
                   bool fp8_mma) {
    TORCH_CHECK(q.dim() == 4 && kv.dim() == 2 && indices.dim() == 3, "kv4mla: bad ranks");
    const int b = q.size(0), s_q = q.size(1), h_q = q.size(2), d_qk = q.size(3);
    const int rows = kv.size(0), topk = indices.size(2);
    TORCH_CHECK(h_q == H_Q && d_qk == D_QK, "kv4mla: q must be [b, s_q, 64, 576]");
    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn && q.stride(3) == 1, "kv4mla: q must be fp8 e4m3, last dim contiguous");
    TORCH_CHECK(kv.dtype() == torch::kUInt8 && kv.size(1) == ROW_BYTES && kv.stride(1) == 1 && kv.stride(0) == ROW_BYTES,
                "kv4mla: kv must be contiguous [rows, 352] uint8");
    TORCH_CHECK(indices.dtype() == torch::kInt32 && indices.stride(2) == 1 && indices.size(0) == b && indices.size(1) == s_q,
                "kv4mla: indices must be [b, s_q, topk] int32, last dim contiguous");
    TORCH_CHECK(topk > 0 && topk % B_TOPK == 0, "kv4mla: topk must be a positive multiple of 64");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.dim() == 4 && out.size(0) == b && out.size(1) == s_q &&
                out.size(2) == H_Q && out.size(3) == D_V && out.stride(3) == 1, "kv4mla: bad out");
    TORCH_CHECK(lse.dtype() == torch::kFloat32 && lse.dim() == 3 && lse.size(0) == b && lse.size(1) == s_q &&
                lse.size(2) == H_Q && lse.stride(2) == 1, "kv4mla: bad lse");
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length->dtype() == torch::kInt32 && topk_length->is_contiguous() && topk_length->numel() == b);
    }
    for (const at::Tensor *t : {&q, &kv, &indices, (const at::Tensor*)&out, (const at::Tensor*)&lse})
        TORCH_CHECK(t->is_cuda() && t->get_device() == q.get_device(), "kv4mla: tensors must be on q's device");
    // TMA row coordinates are int32
    TORCH_CHECK(rows > 0, "kv4mla: empty kv");
    const bool split = meta.has_value();
    at::cuda::CUDAGuard guard{(char)q.get_device()};
    if (b == 0) return;

    SparseAttnDecodeParams p{};
    p.b = b; p.s_q = s_q; p.h_q = h_q; p.h_kv = 1; p.d_qk = d_qk; p.d_v = D_V;
    p.sm_scale = (float)sm_scale; p.sm_scale_div_log2 = (float)sm_scale * LOG_2_E;
    p.num_blocks = rows; p.page_block_size = 1; p.topk = topk;
    p.model_type = ModelType::V32; p.extra_model_type = ModelType::V32;
    p.q = (cutlass::bfloat16_t*)q.data_ptr();
    p.kv = (cutlass::bfloat16_t*)kv.data_ptr();
    p.indices = indices.data_ptr<int>();
    p.topk_length = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;
    p.attn_sink = nullptr;
    p.lse = lse.data_ptr<float>();
    p.out = (cutlass::bfloat16_t*)out.data_ptr();
    p.stride_q_b = to_int(q.stride(0)); p.stride_q_s_q = to_int(q.stride(1)); p.stride_q_h_q = to_int(q.stride(2));
    p.stride_kv_block = ROW_BYTES; p.stride_kv_row = ROW_BYTES;
    p.stride_indices_b = to_int(indices.stride(0)); p.stride_indices_s_q = to_int(indices.stride(1));
    p.stride_lse_b = to_int(lse.stride(0)); p.stride_lse_s_q = to_int(lse.stride(1));
    p.stride_o_b = to_int(out.stride(0)); p.stride_o_s_q = to_int(out.stride(1)); p.stride_o_h_q = to_int(out.stride(2));
    p.stream = at::cuda::getCurrentCUDAStream().stream();
    p.enable_split_kv = split;

    if (split) {
        TORCH_CHECK(num_splits.has_value() && o_accum.has_value() && lse_accum.has_value(), "kv4mla: split-KV needs all workspaces");
        const int parts = meta->size(0);
        TORCH_CHECK(meta->dtype() == torch::kInt32 && meta->is_contiguous() && meta->size(1) == (int64_t)(sizeof(DecodingSchedMeta) / 4));
        TORCH_CHECK(num_splits->dtype() == torch::kInt32 && num_splits->is_contiguous() && num_splits->numel() == b + 1);
        TORCH_CHECK(o_accum->dtype() == torch::kFloat32 && o_accum->is_contiguous() && o_accum->dim() == 4 &&
                    o_accum->size(0) >= b + parts && o_accum->size(1) == s_q && o_accum->size(2) == H_Q && o_accum->size(3) == D_V,
                    "kv4mla: o_accum must be contiguous [>= b + num_sm_parts, s_q, 64, 512] fp32");
        TORCH_CHECK(lse_accum->dtype() == torch::kFloat32 && lse_accum->is_contiguous() && lse_accum->dim() == 3 &&
                    lse_accum->size(0) >= b + parts && lse_accum->size(1) == s_q && lse_accum->size(2) == H_Q,
                    "kv4mla: lse_accum must be contiguous [>= b + num_sm_parts, s_q, 64] fp32");
        p.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)meta->data_ptr();
        p.num_splits_ptr = num_splits->data_ptr<int>();
        p.num_sm_parts = parts;
        p.lse_accum = lse_accum->data_ptr<float>();
        p.o_accum = o_accum->data_ptr<float>();
        p.stride_lse_accum_split = to_int(lse_accum->stride(0)); p.stride_lse_accum_s_q = to_int(lse_accum->stride(1));
        p.stride_o_accum_split = to_int(o_accum->stride(0)); p.stride_o_accum_s_q = to_int(o_accum->stride(1));
        p.stride_o_accum_h_q = to_int(o_accum->stride(2));
    }

    if (fp8_mma) kv4mla::run_glm_fp4_sparse_decode_fp8(p);
    else kv4mla::run_glm_fp4_sparse_decode(p);

    if (split) {
        CombineParams c = {
            b, s_q, H_Q, D_V,
            p.lse, p.out, p.stride_lse_b, p.stride_lse_s_q, p.stride_o_b, p.stride_o_s_q, p.stride_o_h_q,
            p.lse_accum, p.o_accum, p.stride_lse_accum_split, p.stride_lse_accum_s_q,
            p.stride_o_accum_split, p.stride_o_accum_s_q, p.stride_o_accum_h_q,
            p.tile_scheduler_metadata_ptr, p.num_splits_ptr, p.num_sm_parts,
            nullptr, p.stream};
        smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(c);
    }
}

// k27 dense mode: out [b, 64, 512] bf16 <- dense MLA over a paged FP4 pool (fp8-MMA kernel only).
//   q           [b, 64, 576] float8_e4m3fn: 64 q rows per sequence = tokens x heads (row t * n_heads + h), padded
//   kv          [rows, 352] uint8 pool (row = page * page_size + offset)
//   page_table  [b, n_pages] int32 page ids (SGLang block_kv_indices); seq_lens [b] int32 = positions to attend,
//               including the sequence's own draft rows; max_seq_len bounds the block loop (multiple of 64 after
//               padding); row_limit [b, 64] int32 (optional): q row h attends positions < row_limit[b, h]
//   lse         [b, 64] fp32; split-KV workspaces as for sparse_decode with s_q = 1 (meta from sched_meta(b, 1,
//               topk = round_up(max_seq_len, 64), seq_lens)).
void dense_decode(const at::Tensor &q, const at::Tensor &kv, const at::Tensor &page_table, const at::Tensor &seq_lens,
                  int64_t page_size, int64_t max_seq_len, const std::optional<at::Tensor> &row_limit, double sm_scale,
                  at::Tensor &out, at::Tensor &lse, const std::optional<at::Tensor> &meta,
                  const std::optional<at::Tensor> &num_splits, const std::optional<at::Tensor> &o_accum,
                  const std::optional<at::Tensor> &lse_accum) {
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 2 && page_table.dim() == 2, "kv4mla dense: bad ranks");
    const int b = q.size(0), h_q = q.size(1), d_qk = q.size(2);
    const int rows = kv.size(0);
    TORCH_CHECK(h_q == H_Q && d_qk == D_QK, "kv4mla dense: q must be [b, 64, 576]");
    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn && q.stride(2) == 1, "kv4mla dense: q must be fp8 e4m3, last dim contiguous");
    TORCH_CHECK(kv.dtype() == torch::kUInt8 && kv.size(1) == ROW_BYTES && kv.stride(1) == 1 && kv.stride(0) == ROW_BYTES,
                "kv4mla dense: kv must be contiguous [rows, 352] uint8");
    TORCH_CHECK(page_table.dtype() == torch::kInt32 && page_table.stride(1) == 1 && page_table.size(0) == b,
                "kv4mla dense: page_table must be [b, n_pages] int32, last dim contiguous");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32 && seq_lens.is_contiguous() && seq_lens.numel() == b, "kv4mla dense: bad seq_lens");
    TORCH_CHECK(page_size > 1 && page_size <= 1024, "kv4mla dense: page_size must be in (1, 1024]");
    // max_seq_len only bounds the block loop; the table is read for positions < seq_lens[b], which the caller
    // guarantees it covers (SGLang's block_kv_indices span every sequence's pages; graph buffers span the context)
    TORCH_CHECK(max_seq_len > 0, "kv4mla dense: max_seq_len must be positive");
    const int topk = (int)((max_seq_len + B_TOPK - 1) / B_TOPK * B_TOPK);
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.dim() == 3 && out.size(0) == b && out.size(1) == H_Q &&
                out.size(2) == D_V && out.stride(2) == 1, "kv4mla dense: bad out");
    TORCH_CHECK(lse.dtype() == torch::kFloat32 && lse.dim() == 2 && lse.size(0) == b && lse.size(1) == H_Q &&
                lse.stride(1) == 1, "kv4mla dense: bad lse");
    if (row_limit.has_value()) {
        TORCH_CHECK(row_limit->dtype() == torch::kInt32 && row_limit->dim() == 2 && row_limit->size(0) == b &&
                    row_limit->size(1) == H_Q && row_limit->stride(1) == 1, "kv4mla dense: row_limit must be [b, 64] int32");
    }
    for (const at::Tensor *t : {&q, &kv, &page_table, &seq_lens, (const at::Tensor*)&out, (const at::Tensor*)&lse})
        TORCH_CHECK(t->is_cuda() && t->get_device() == q.get_device(), "kv4mla dense: tensors must be on q's device");
    TORCH_CHECK(rows > 0, "kv4mla dense: empty kv");
    const bool split = meta.has_value();
    at::cuda::CUDAGuard guard{(char)q.get_device()};
    if (b == 0) return;

    SparseAttnDecodeParams p{};
    p.b = b; p.s_q = 1; p.h_q = h_q; p.h_kv = 1; p.d_qk = d_qk; p.d_v = D_V;
    p.sm_scale = (float)sm_scale; p.sm_scale_div_log2 = (float)sm_scale * LOG_2_E;
    p.num_blocks = rows; p.page_block_size = (int)page_size; p.topk = topk;
    p.model_type = ModelType::V32; p.extra_model_type = ModelType::V32;
    p.q = (cutlass::bfloat16_t*)q.data_ptr();
    p.kv = (cutlass::bfloat16_t*)kv.data_ptr();
    p.indices = page_table.data_ptr<int>();
    p.topk_length = seq_lens.data_ptr<int>();
    p.attn_sink = nullptr;
    p.lse = lse.data_ptr<float>();
    p.out = (cutlass::bfloat16_t*)out.data_ptr();
    p.stride_q_b = to_int(q.stride(0)); p.stride_q_s_q = to_int(q.stride(0)); p.stride_q_h_q = to_int(q.stride(1));
    p.stride_kv_block = ROW_BYTES; p.stride_kv_row = ROW_BYTES;
    p.stride_indices_b = to_int(page_table.stride(0)); p.stride_indices_s_q = 0;
    p.stride_lse_b = to_int(lse.stride(0)); p.stride_lse_s_q = 0;
    p.stride_o_b = to_int(out.stride(0)); p.stride_o_s_q = to_int(out.stride(0)); p.stride_o_h_q = to_int(out.stride(1));
    p.row_limit = row_limit.has_value() ? row_limit->data_ptr<int>() : nullptr;
    p.stride_row_limit_b = row_limit.has_value() ? to_int(row_limit->stride(0)) : 0;
    p.stream = at::cuda::getCurrentCUDAStream().stream();
    p.enable_split_kv = split;

    if (split) {
        TORCH_CHECK(num_splits.has_value() && o_accum.has_value() && lse_accum.has_value(), "kv4mla dense: split-KV needs all workspaces");
        const int parts = meta->size(0);
        TORCH_CHECK(meta->dtype() == torch::kInt32 && meta->is_contiguous() && meta->size(1) == (int64_t)(sizeof(DecodingSchedMeta) / 4));
        TORCH_CHECK(num_splits->dtype() == torch::kInt32 && num_splits->is_contiguous() && num_splits->numel() == b + 1);
        TORCH_CHECK(o_accum->dtype() == torch::kFloat32 && o_accum->is_contiguous() && o_accum->dim() == 4 &&
                    o_accum->size(0) >= b + parts && o_accum->size(1) == 1 && o_accum->size(2) == H_Q && o_accum->size(3) == D_V,
                    "kv4mla dense: o_accum must be contiguous [>= b + num_sm_parts, 1, 64, 512] fp32");
        TORCH_CHECK(lse_accum->dtype() == torch::kFloat32 && lse_accum->is_contiguous() && lse_accum->dim() == 3 &&
                    lse_accum->size(0) >= b + parts && lse_accum->size(1) == 1 && lse_accum->size(2) == H_Q,
                    "kv4mla dense: lse_accum must be contiguous [>= b + num_sm_parts, 1, 64] fp32");
        p.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)meta->data_ptr();
        p.num_splits_ptr = num_splits->data_ptr<int>();
        p.num_sm_parts = parts;
        p.lse_accum = lse_accum->data_ptr<float>();
        p.o_accum = o_accum->data_ptr<float>();
        p.stride_lse_accum_split = to_int(lse_accum->stride(0)); p.stride_lse_accum_s_q = to_int(lse_accum->stride(1));
        p.stride_o_accum_split = to_int(o_accum->stride(0)); p.stride_o_accum_s_q = to_int(o_accum->stride(1));
        p.stride_o_accum_h_q = to_int(o_accum->stride(2));
    }

    kv4mla::run_glm_fp4_sparse_decode_fp8(p);

    if (split) {
        CombineParams c = {
            b, 1, H_Q, D_V,
            p.lse, p.out, p.stride_lse_b, p.stride_lse_s_q, p.stride_o_b, p.stride_o_s_q, p.stride_o_h_q,
            p.lse_accum, p.o_accum, p.stride_lse_accum_split, p.stride_lse_accum_s_q,
            p.stride_o_accum_split, p.stride_o_accum_s_q, p.stride_o_accum_h_q,
            p.tile_scheduler_metadata_ptr, p.num_splits_ptr, p.num_sm_parts,
            nullptr, p.stream};
        smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(c);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "kv4mla: GLM-5.3 NVFP4 sparse MLA decode (SM100); k27 dense mode over a paged FP4 pool";
    m.def("num_sm_parts", &num_sm_parts);
    m.def("sched_meta", &sched_meta);
    m.def("sparse_decode", &sparse_decode);
    m.def("dense_decode", &dense_decode);
    m.def("trace", &trace);
}
