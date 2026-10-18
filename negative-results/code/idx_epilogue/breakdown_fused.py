"""Stage timing of indexer_topk_fused (agent glmidx)."""
import json, math, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import flashinfer  # noqa: F401
import deep_gemm
from operations.kernels.idx_epilogue.fused_topk import aux_buffer, _topk_rows, CAP
from operations.kernels.idx_epilogue.bench_fused_topk import ev_time  # noqa
