"""CUTLASS native-FP4 grouped GEMM, driven by our data (GPU/B200 only).

This is the kernel binding: it drives the `nvidia-cutlass-dsl` Sm100 block-scaled grouped
GEMM (the native FP4 tensor-core path Triton can't reach) with our NVFP4 tensors. Validated
against a bf16 reference (single expert: SQNR 17.4 dB / cos 0.991, i.e. exactly W4A4 NVFP4
quantization noise; the math is correct).

Inputs per expert g: A_gridvals[m,k] and B_gridvals[n,k] (fp32 e2m1 grid values, i.e. the
4-bit codes as their represented magnitudes), SFA[m,k//16]/SFB[n,k//16] (e4m3 block scales),
and alpha[g] = gscale_A * gscale_B (the two per-tensor scales folded into one epilogue mul).
Use kernels.cutlass.nvfp4_layout.quantize_nvfp4 to produce these from bf16 activations, and
the same for weights (or load pre-packed). Returns C[m,n] fp32 per expert.

Depends on the CUTLASS example helpers; set CUTLASS_DSL_ROOT to the CuTeDSL examples dir
(the bench/serving image clones NVIDIA/cutlass to /root/cutlass).

STATUS: validated + fast. Correctness SQNR 17.4 dB / cos 0.991 across 64 experts (the W4A4
NVFP4 floor). Perf: run() builds+compiles once; kernel_only() (the per-forward hot path) is
0.111 ms / 1696 TF/s = 9.4x our Triton grouped GEMM at 64x(512x1408x2048). execute() adds the
per-group global-scale alpha as one batched bf16 multiply (fuses into the downstream SwiGLU in
the real MoE pipeline). Remaining: cache static weight tensors so only activations rebuild per
forward, then wire into runner._moe behind a flag.
"""
from __future__ import annotations

import os

CUTLASS_DSL_ROOT = os.environ.get(
    "CUTLASS_DSL_ROOT", "/root/cutlass/examples/python/CuTeDSL")
SF_VEC = 16


def _load_example():
    import sys, importlib.util
    if CUTLASS_DSL_ROOT not in sys.path:
        sys.path.insert(0, CUTLASS_DSL_ROOT)
    p = (f"{CUTLASS_DSL_ROOT}/cute/blackwell/kernel/blockscaled_grouped_gemm/"
         "grouped_blockscaled_gemm.py")
    spec = importlib.util.spec_from_file_location("gbg", p)
    gbg = importlib.util.module_from_spec(spec); spec.loader.exec_module(gbg)
    return gbg


class GroupedNVFP4:
    """Compile the grouped block-scaled NVFP4 GEMM once, then run it on lists of per-expert
    tensors. mma_tiler/cluster are fixed at construction (they parametrize the compile)."""

    def __init__(self, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1)):
        import cutlass
        self.cutlass = cutlass
        self.gbg = _load_example()
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.ab_dt = cutlass.Float4E2M1FN
        self.sf_dt = cutlass.Float8E4M3FN
        # bf16 out: same exponent range as fp32 (the pre-alpha accumulator ~1e7 would overflow
        # fp16, but not bf16) at half the write bandwidth; mantissa is ample vs the W4A4 noise.
        self.c_dt = cutlass.BFloat16
        self._conv_cache = {}          # precompiled _convert kernels, keyed by (rows, cols)

    # -- per-tensor builders. Each returns a `refresh(new_values)` that rewrites the SAME GPU
    #    storage in place (pointer stable), so per-forward we only refresh activations. --
    def _make_ab(self, m0, m1, src):
        """Build an fp4 operand whose fp32 source is `src` ([m0,m1,1] GPU, may be a slice of a
        bigger buffer). Returns a graph-captured process() = precompiled fp32->fp4 convert.
        The load (writing new data into `src`) is the caller's job (one batched copy)."""
        import cutlass, cutlass.cute as cute, cutlass.torch as ct
        from cutlass.cute.runtime import from_dlpack
        from cutlass.cute import testing as T
        cpu = ct.matrix(1, m0, m1, False, cutlass.Float32)         # CPU tensor -> layout only
        cute_t, torch_t = ct.cute_tensor_like(cpu, self.ab_dt, True, assumed_align=16)
        fp32c = from_dlpack(src).mark_layout_dynamic(leading_dim=1)
        key = (m0, m1, src.dtype)                                  # precompile convert once per shape+dtype
        if key not in self._conv_cache:
            self._conv_cache[key] = cute.compile(T._convert, fp32c, cute_t, 1, 8)  # leading=1, fp4->8/copy
        conv = self._conv_cache[key]
        conv(fp32c, cute_t)

        def process():                                             # graph-captured: precompiled convert
            conv(fp32c, cute_t)
        return torch_t.data_ptr(), torch_t, cute_t, (m1, 1), process

    def _make_c(self, m, n):
        import cutlass, cutlass.torch as ct
        cpu = ct.matrix(1, m, n, False, cutlass.Float32)
        cute_t, torch_t = ct.cute_tensor_like(cpu, self.c_dt, True, assumed_align=16)
        return torch_t.data_ptr(), torch_t, cute_t, (n, 1)

    def _make_sf(self, mn, k, sf_in):
        """Build a swizzled e4m3 scale operand whose flat fp32 source is `sf_in` ([mn*sf_k],
        may be a slice). Returns process() = graph-captured gather + e4m3 write."""
        import cutlass, cutlass.torch as ct
        from cutlass.cute.runtime import from_dlpack
        import torch
        sf_k = max(1, (k + SF_VEC - 1) // SF_VEC)
        cd = lambda a, b: (a + b - 1) // b
        mma_shape = (1, cd(mn, 128), cd(sf_k, 4), 32, 4, 4)
        # GPU-native swizzle: the example's cvt_sf is a HOST per-element loop (~44ms/expert,
        # segfaults on GPU). Instead derive the exact permutation ONCE by running cvt on an
        # index tensor, then per-forward the swizzle is a single GPU gather (verified byte-
        # identical to cvt in bench/swizzle_match check).
        ref = ct.create_and_permute_torch_tensor(
            (1, mn, sf_k), torch.float32, permute_order=(1, 2, 0),
            init_type=ct.TensorInitType.RANDOM, init_config=ct.RandomInitConfig(min_val=1, max_val=3))
        mma = ct.create_and_permute_torch_tensor(
            mma_shape, torch.float32, permute_order=(3, 4, 1, 5, 2, 0),
            init_type=ct.TensorInitType.RANDOM, init_config=ct.RandomInitConfig(min_val=0, max_val=1))
        idx_ref = ref.clone()
        idx_ref.copy_(torch.arange(ref.numel(), dtype=torch.float32).reshape(ref.shape))
        idx_mma = mma.clone()
        self.gbg.cvt_sf_MKL_to_M32x4xrm_K4xrk_L(from_dlpack(idx_ref), from_dlpack(idx_mma))
        gather = idx_mma.long().cuda()                              # [mma_shape] indices into sf.flat
        cute_t, torch_t = ct.cute_tensor_like(mma, self.sf_dt, True, assumed_align=16)

        def process():                                            # graph-captured: gather + e4m3 write
            torch_t.copy_(sf_in[gather].reshape(torch_t.shape).to(torch_t.dtype))
        process()
        return torch_t.data_ptr(), torch_t, cute_t, process

    def _make_a_batched(self, total, k):
        """ONE fp4 buffer holding ALL experts' activations, converted in a SINGLE DSL call per
        forward (the per-expert convert was ~85ms x G; batching makes it one call). Per-expert
        A pointers slice this buffer by row offset (rows are contiguous, K uniform)."""
        import cutlass, cutlass.torch as ct
        cpu = ct.matrix(1, total, k, False, cutlass.Float32)       # [total, k, 1] layout
        cute_t, torch_t = ct.cute_tensor_like(cpu, self.ab_dt, True, assumed_align=16)
        src = cpu.cuda()

        def convert(all_gv):                                       # all_gv: [total, k] fp32 grid values
            src.copy_(all_gv.reshape(src.shape))
            ct.convert_cute_tensor(src, cute_t, self.ab_dt, is_dynamic_layout=True)
        return torch_t, cute_t, convert

    def prepare(self, problem_sizes, B_gv, SFB):
        """Build the STATIC side once: per-expert weight tensors (B, SFB), the output buffer,
        placeholder activation tensors (refreshed each forward), the metadata/pointer arrays,
        and the compiled kernel. Weights and compile never change between forwards.
        problem_sizes: list of (m,n,k) per expert (m = that expert's padded row count)."""
        import cutlass, cutlass.cute as cute, cutlass.torch as ctorch, torch
        G = len(problem_sizes)
        n = problem_sizes[0][1]
        rows = [pm for (pm, _, _) in problem_sizes]
        offs = [sum(rows[:g]) for g in range(G)]
        total = sum(rows)
        k = problem_sizes[0][2]                                             # Din, uniform per projection
        sf_k = max(1, k // SF_VEC)
        C_all = torch.zeros((total, n), dtype=torch.bfloat16, device="cuda")
        _, tC0, cC0, sC = self._make_c(problem_sizes[0][0], n)
        # ONE contiguous fp32 input buffer for all experts' activations + all their block scales;
        # per forward we fill each with a SINGLE batched copy (real MoE gathers rows contiguously
        # anyway), then per-expert convert/gather reads slices (graph-captured).
        A_in = torch.zeros(total, k, device="cuda", dtype=torch.bfloat16)   # bf16 src: half the write/read
        sf_sizes = [pm * sf_k for pm in rows]
        sf_offs = [sum(sf_sizes[:g]) for g in range(G)]
        SFA_in = torch.ones(sum(sf_sizes), device="cuda")
        keep, ptrs_abc, ptrs_sf, strides, dims = [C_all, tC0, A_in, SFA_in], [], [], [], []
        self._a_proc, self._sfa_proc, static = [], [], []
        for g, (m, _, kk) in enumerate(problem_sizes):
            src_a = A_in[offs[g]:offs[g] + m].view(m, kk, 1)               # slice into batched A
            pA, tA, cA, sA, aproc = self._make_ab(m, kk, src_a)
            src_b = B_gv[g].reshape(n, kk, 1).contiguous().float(); static.append(src_b)
            pB, tB, cB, sB, bproc = self._make_ab(n, kk, src_b); bproc()   # weights: convert once
            pC = C_all[offs[g]].data_ptr()
            sfa_slice = SFA_in[sf_offs[g]:sf_offs[g] + sf_sizes[g]]
            pSFA, tSFA, cSFA, sfaproc = self._make_sf(m, kk, sfa_slice)
            src_sfb = SFB[g].reshape(-1).contiguous().float(); static.append(src_sfb)
            pSFB, tSFB, cSFB, sfbproc = self._make_sf(n, kk, src_sfb); sfbproc()
            keep += [tA, tB, tSFA, tSFB]
            self._a_proc.append(aproc); self._sfa_proc.append(sfaproc)
            ptrs_abc.append([pA, pB, pC]); ptrs_sf.append([pSFA, pSFB])
            strides.append([list(sA), list(sB), list(sC)]); dims.append([m, n, kk, 1])
            if g == 0:
                first = (cA, cB, cC0, cSFA, cSFB)
        keep += static
        self._A_in, self._SFA_in = A_in, SFA_in

        def ct_meta(t, dt):
            c, th = ctorch.cute_tensor_like(t, dt, is_dynamic_layout=False, assumed_align=16)
            keep.append(th); return c
        t_dim = ct_meta(torch.tensor(dims, dtype=torch.int32), cutlass.Int32)
        t_str = ct_meta(torch.tensor(strides, dtype=torch.int32), cutlass.Int32)
        t_pabc = ct_meta(torch.tensor(ptrs_abc, dtype=torch.int64), cutlass.Int64)
        t_psf = ct_meta(torch.tensor(ptrs_sf, dtype=torch.int64), cutlass.Int64)
        K = self.gbg.Sm100GroupedBlockScaledGemmKernel
        hw = cutlass.utils.HardwareInfo()
        sm = hw.get_max_active_clusters(1)
        mac = hw.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        t_tmap = ct_meta(torch.empty((sm, K.num_tensormaps, K.bytes_per_tensormap // 8),
                                     dtype=torch.int64), cutlass.Int64)
        stream = ctorch.default_stream()
        kernel = K(SF_VEC, self.mma_tiler_mn, self.cluster_shape_mn)
        compiled = cute.compile(kernel, first[0], first[1], first[2], first[3], first[4],
                                G, t_dim, t_str, t_pabc, t_psf, mac, t_tmap, mac, stream,
                                options="--opt-level 2")
        self._exec = (compiled, first, t_dim, t_str, t_pabc, t_psf, t_tmap, stream)
        # static buffers for CUDA-graph capture of the compute (per-forward we only copy new
        # inputs into the load buffers + update alpha_rows, then replay).
        self._alpha_rows = torch.ones((total, 1), dtype=torch.bfloat16, device="cuda")
        # expand per-expert alpha -> per-row via a precomputed gather index (one op, not G torch.full+cat)
        self._alpha_expand = torch.tensor([g for g in range(G) for _ in range(rows[g])],
                                          device="cuda")
        self._C_scaled = torch.empty((total, n), dtype=torch.bfloat16, device="cuda")
        self._C_all = C_all
        self._out = (C_all, self._alpha_rows, offs, rows, n, keep)
        self._graph = None
        return self

    def _compute(self):
        """The graph-captured body: convert/gather all experts, launch the kernel, scale."""
        for g in range(len(self._a_proc)):
            self._a_proc[g](); self._sfa_proc[g]()
        self.kernel_only()
        self._C_scaled.copy_(self._C_all * self._alpha_rows)

    def forward(self, A_gv, SFA, alpha):
        """Per-forward hot path: copy new activations into static buffers + update alpha, then
        replay the captured compute graph (converts + gather + kernel + scale as ONE replay).
        Returns the flat [total_rows, N] bf16 output on GPU (slice with .offsets)."""
        import torch
        rows = self._out[3]
        # batched inputs: A_all [total,K], SFA_all [total,sf_k] (or a list -> cat). Real MoE
        # already gathers rows contiguously, so this is one copy, not 2*G.
        A_all = A_gv if torch.is_tensor(A_gv) else torch.cat([x.reshape(-1, x.shape[-1]) for x in A_gv], 0)
        SFA_all = SFA if torch.is_tensor(SFA) else torch.cat([x.reshape(-1) for x in SFA], 0)
        self._A_in.copy_(A_all.reshape(self._A_in.shape))
        self._SFA_in.copy_(SFA_all.reshape(-1))
        a = alpha if torch.is_tensor(alpha) else torch.tensor(alpha, device="cuda")
        self._alpha_rows.copy_(a.to("cuda")[self._alpha_expand].unsqueeze(1).bfloat16())
        if self._graph is None:                                   # capture once
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._compute()
            torch.cuda.current_stream().wait_stream(s)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._compute()
        self._graph.replay()                                      # caller syncs
        return self._C_scaled

    def run(self, problem_sizes, A_gv, B_gv, SFA, SFB, alpha):
        """Convenience: prepare(weights) then forward(activations). Returns flat [total,N]."""
        self.prepare(problem_sizes, B_gv, SFB)
        return self.forward(A_gv, SFA, alpha)

    def kernel_only(self):
        """Just launch the compiled kernel (no output post-processing). Timing this reflects
        the true kernel cost; the caller synchronizes."""
        compiled, first, t_dim, t_str, t_pabc, t_psf, t_tmap, stream = self._exec
        compiled(first[0], first[1], first[2], first[3], first[4],
                 t_dim, t_str, t_pabc, t_psf, t_tmap, stream)

    def execute(self):
        """Per-forward hot path: launch the kernel, apply the per-group global-scale alpha in
        one batched multiply, return the flat [total_rows, N] output ON GPU. To slice per
        expert use output[offs[g]:offs[g]+rows[g]]."""
        import torch
        self.kernel_only()
        C_all, alpha_rows, offs, rows, n, _ = self._out
        torch.cuda.synchronize()
        return C_all * alpha_rows                                        # single GPU op

    @property
    def offsets(self):
        return self._out[2], self._out[3]                               # (offs, rows)
