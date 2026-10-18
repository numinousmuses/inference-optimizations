"""CUTLASS native-FP4 integration (Blackwell sm_100).

The Triton compiler cannot lower the e2m1 MMA (documented MLIR bug), so our Triton GEMMs
dequant to bf16 first. The CUTLASS CuTe-DSL block-scaled GEMM DOES reach the native FP4
tensor cores: bench/cutlass_probe_modal.py measured a dense NVFP4 GEMM at 7.45 PFLOPS
(~83% of FP4 peak, 3.1x our dequant path) and a MoE-shaped grouped NVFP4 GEMM at 1742 TF/s
(9.6x our Triton grouped). This package wires that kernel onto our data.

nvfp4_layout.py  -- the format bridge (our packed layout <-> CUTLASS's expected layout)
grouped_gemm.py  -- the kernel binding (drives the DSL grouped GEMM); GPU/B200 only
"""
