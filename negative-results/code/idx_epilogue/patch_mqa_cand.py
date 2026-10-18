"""Anchored edit of DeepGEMM 0.2.0's JIT header deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh: a fused
"candidate" epilogue for the ragged (prefill) MQA-logits kernel (agent glmidx).

The kernel is untouched in every mode SGLang's GLM path uses (non-compressed logits). In compressed-logits mode
(max_seqlen_k > 0, which GLM's DSA path never uses) the epilogue writes only the scores above a per-row threshold
(sparse stores into the rows x stride buffer, relative to ks) and sets their bit in a per-row bitmap with a
non-returning atomicOr (a first version appended to per-row lists with returning atomicAdd: correct, but the atomic
latency on the epilogue's critical path cost +1.5 ms per 2048 x 131k call, measured):

    aux = cu_seq_len_k_start (int32), R4 = align(num_q_tokens, 4), W = logits_stride / 32
    aux[0 : R)                  ks (the real window starts, read by the scheduler as before)
    aux[R4 : R4 + R)            thr (float bits), per-row threshold set by the caller (loaded once per Q block)
    aux[2 R4 + r W : + W)       bitmap of row r over relative key offsets (zeroed by the caller)
The caller passes ks as a view of the first R entries of aux, so DeepGEMM's host checks see an ordinary ks tensor.
Exact top-k: if row r has >= k bits set, its true top-k all exceed thr[r], hence all are marked.

    python3 patch_mqa_cand.py <path to sm100_mqa_logits.cuh> [--check]
"""
import sys

CAP = 8192
MARK = "// numinous-hook:mqa_logits_cand"

A_CORE = ("                                               const cute::TmaDescriptor& tensor_map_weights,\n"
          "                                               const MakeScheduler& make_scheduler) {\n")
R_CORE = ("                                               const cute::TmaDescriptor& tensor_map_weights,\n"
          "                                               const MakeScheduler& make_scheduler,\n"
          "                                               const uint32_t* numinous_aux = nullptr,\n"
          "                                               const uint32_t numinous_rows = 0) {\n")

A_EPI = ("                        if constexpr (kIsCompressedLogits) {\n"
         "                            const uint32_t rel_kv = kv_offset - seq_k_start[i];\n"
         "                            const uint32_t len = seq_k_end[i] - seq_k_start[i];\n"
         "                            if (rel_kv < len)\n"
         "                                logits[q_offset + rel_kv] = result;\n")
R_EPI = (f"                        if constexpr (kIsCompressedLogits) {{\n"
         f"                            {MARK}\n"
         "                            const uint32_t rel_kv = kv_offset - seq_k_start[i];\n"
         "                            const uint32_t len = seq_k_end[i] - seq_k_start[i];\n"
         "                            if (rel_kv < len and static_cast<float>(result) > numinous_thr[i]) {\n"
         "                                const uint32_t row = scheduler.get_logits_row(q_block_idx, i);\n"
         "                                const uint32_t r4 = (numinous_rows + 3u) & ~3u;\n"
         "                                uint32_t* bitmap = const_cast<uint32_t*>(numinous_aux) + 2 * r4 +\n"
         "                                                   static_cast<uint64_t>(row) * (logits_stride / 32u);\n"
         "                                atomicOr(bitmap + (rel_kv >> 5), 1u << (rel_kv & 31u));\n"
         "                                logits[q_offset + rel_kv] = result;\n"
         "                            }\n")

A_THR = ("            const auto process_q_block = [&](auto num_valid_tokens_t) {\n"
         "                constexpr uint32_t kNumValidTokens = decltype(num_valid_tokens_t)::value;\n")
R_THR = (A_THR +
         "                float numinous_thr[kNumValidTokens];\n"
         "                if constexpr (kIsCompressedLogits) {\n"
         "                    const uint32_t r4 = (numinous_rows + 3u) & ~3u;\n"
         "                    #pragma unroll\n"
         "                    for (uint32_t i = 0; i < kNumValidTokens; ++ i)\n"
         "                        numinous_thr[i] = __ldg(reinterpret_cast<const float*>(numinous_aux + r4) +\n"
         "                                                scheduler.get_logits_row(q_block_idx, i));\n"
         "                }\n")

ENTRY_TAIL = ("        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,\n"
              "        make_scheduler);\n")
ENTRY_TAIL_NEW = ("        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,\n"
                  "        make_scheduler, cu_seq_len_k_start, num_q_tokens);\n")


def patch(src: str) -> str:
    if MARK in src:
        return src
    for a in (A_CORE, A_EPI, A_THR):
        if src.count(a) != 1:
            raise SystemExit(f"anchor found {src.count(a)} times: {a[:90]!r}")
    src = src.replace(A_CORE, R_CORE, 1).replace(A_EPI, R_EPI, 1).replace(A_THR, R_THR, 1)
    # the two ragged entries (sm100_mqa_logits, sm100_mqa_logits_unrolled) forward ks / num_q_tokens; paged keeps
    # the defaults
    out, pos = [], 0
    for name in ("void sm100_mqa_logits(", "void sm100_mqa_logits_unrolled("):
        start = src.index(name, pos)
        tail = src.index(ENTRY_TAIL, start)
        if src.find("void sm100_paged_mqa_logits(", start, tail) != -1:
            raise SystemExit("entry layout changed")
        out.append(src[pos:tail])
        out.append(ENTRY_TAIL_NEW)
        pos = tail + len(ENTRY_TAIL)
    out.append(src[pos:])
    return "".join(out)


if __name__ == "__main__":
    path = sys.argv[1]
    src = open(path).read()
    new = patch(src)
    if "--check" in sys.argv:
        print("ok", new.count(ENTRY_TAIL_NEW), "entries")
    else:
        open(path, "w").write(new)
        print("patched" if new != src else "already patched", path)
