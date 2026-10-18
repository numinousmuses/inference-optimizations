"""kv4mla: native NVFP4 sparse MLA decode for GLM-5.3's 4-bit DSA KV cache (SM100, CUDA, no Triton).

Reads kv4glm's 352-byte rows (operations/kernels/glm_kv4.py: 512 E2M1 latent + 32 E4M3 block scales + 64 E4M3 rope)
straight from the pool and dequantizes them in shared memory inside a FlashMLA-derived warp-specialized kernel
(csrc/glm_fp4_decode.cu; fp16 UMMA, fp32 accumulation, exact dequantization), instead of dequantizing the selected rows
into an FP8 scratch for the TRT-LLM op. Same inputs as the served TRT-LLM call: q [T, 64, 576] FP8 with RoPE applied,
a [T, K] table of pool slots (-1 padded), bmm1 scale; output [T, 64, 512] bf16.

Build: `load()` JIT-compiles the extension with torch.utils.cpp_extension (sm_100a) the first time and caches it in
KV4MLA_BUILD_DIR (default ~/.cache/kv4mla/<hash>). A prebuilt module can be pointed to with KV4MLA_SO. CUTLASS headers:
KV4MLA_CUTLASS (an include/ dir), else FlashMLA's pinned submodule at /root/src/FlashMLA/csrc/cutlass/include, else the
copy flashinfer bundles (flashinfer/data/cutlass/include, CUTLASS 4.5 in the v0.5.20 image). The build needs no GPU
(explicit -gencode sm_100a), so the hook patcher compiles it at image build time.

    dec = Decoder(device)
    out = dec(q_fp8 [T, 64, 576], kv_rows [rows, 352] uint8, table [T, K] int32, sm_scale)   # [T, 64, 512] bf16
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CSRC = HERE / "csrc"
ROW_BYTES, H_Q, D_QK, D_V, B_TOPK = 352, 64, 576, 512, 64
SPLIT_MIN_TOPK = 640  # FlashMLA's rule: split-KV only pays off above this many selected rows per query


def sched_overhead(t: int) -> int:
    """FlashMLA scheduler's per-split fixed cost in 64-token blocks (larger = fewer, longer splits). Measured on B200
    (top-2048): 10 for T <= 16 (T=8 22.4 -> 19.3 us, T=16 25.9 -> 22.3), FlashMLA's 5 above (10 costs T=48 30.4 -> 32.7).
    KV4MLA_SCHED_OVERHEAD overrides for every T."""
    env = int(os.environ.get("KV4MLA_SCHED_OVERHEAD", "0"))
    if env > 0:
        return env
    return 10 if t <= 16 else 5


def split_policy(t: int, k: int) -> bool:
    """Split-KV or one CTA per token, from the measured B200 table (bench_kv4mla.py, top-2048): one CTA per token wins
    when T tokens fill one wave (96..136) or several (>= 224); split-KV wins below 96 and between 137 and 223. Split-KV
    workspace is therefore bounded by (223 + num_sms) partial outputs."""
    if k <= SPLIT_MIN_TOPK:
        return False
    return not (96 <= t <= 136 or t >= 224)
_MOD = None


def _cutlass_include() -> str:
    cands = [os.environ.get("KV4MLA_CUTLASS", ""), "/root/src/FlashMLA/csrc/cutlass/include"]
    # the engine image ships CUTLASS 4.5 headers inside flashinfer (measured to build and pass the tests)
    spec = importlib.util.find_spec("flashinfer")
    if spec is not None and spec.submodule_search_locations:
        cands.append(str(Path(list(spec.submodule_search_locations)[0]) / "data" / "cutlass" / "include"))
    for c in cands:
        if c and (Path(c) / "cutlass" / "cutlass.h").exists():
            return c
    raise RuntimeError("kv4mla: CUTLASS headers not found (set KV4MLA_CUTLASS to a cutlass include/ dir)")


def _sources() -> list[str]:
    return [str(CSRC / "api.cpp"), str(CSRC / "glm_fp4_decode.cu"), str(CSRC / "glm_fp4_decode_fp8.cu"),
            str(CSRC / "flashmla/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu"),
            str(CSRC / "flashmla/kernels/smxx/decode/combine/combine.cu")]


def _source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(CSRC.rglob("*")):
        if p.is_file() and p.suffix in (".cu", ".cuh", ".h", ".cpp", ".hpp"):
            h.update(str(p.relative_to(CSRC)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def load(verbose: bool = False):
    """The compiled extension (JIT build on first use, cached by source hash)."""
    global _MOD
    if _MOD is not None:
        return _MOD
    so = os.environ.get("KV4MLA_SO")
    if so:
        spec = importlib.util.spec_from_file_location("kv4mla_ext", so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD = mod
        return mod
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")   # no GPU at image build: do not probe
    from torch.utils.cpp_extension import load as _load

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    extra = os.environ.get("KV4MLA_EXTRA_CFLAGS", "").split()   # experiments only (e.g. -DKV4MLA_SKIP_LOAD)
    tag = _source_hash() + ("_" + hashlib.sha256(" ".join(extra).encode()).hexdigest()[:8] if extra else "")
    build = Path(os.environ.get("KV4MLA_BUILD_DIR", str(Path.home() / ".cache" / "kv4mla"))) / tag
    build.mkdir(parents=True, exist_ok=True)
    inc = [str(CSRC / "flashmla"), str(CSRC / "flashmla" / "kerutils" / "include"), _cutlass_include(),
           str(Path(cuda_home) / "targets" / "x86_64-linux" / "include" / "cccl")]
    nvcc = ["-O3", "-std=c++20", "-DNDEBUG", "-D_USE_MATH_DEFINES", "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "--use_fast_math", "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage", "-lineinfo",
            "-gencode", "arch=compute_100a,code=sm_100a", "--threads", "8"] + extra
    _MOD = _load(name=f"kv4mla_ext_{tag}", sources=_sources(), extra_include_paths=inc,
                 extra_cflags=["-O3", "-std=c++20", "-DNDEBUG"], extra_cuda_cflags=nvcc, build_directory=str(build),
                 verbose=verbose)
    return _MOD


class Decoder:
    """Holds the split-KV schedule per (T, K) and reusable workspaces, so calls are allocation-free after warm-up and
    CUDA-graph capturable. The schedule depends only on the shapes (no topk_length), so a graph captured at batch T
    replays with any table contents."""

    def __init__(self, device, num_sms: int | None = None, mode: str | None = None):
        """mode "fp8": E4M3 tensor-core math (dequantized tiles rounded to E4M3, P in E4M3, as the TRT-LLM FP8 op);
        "exact": fp16 math over exactly dequantized rows. Default from KV4MLA_MODE (fp8)."""
        self.mode = mode or os.environ.get("KV4MLA_MODE", "fp8")
        assert self.mode in ("fp8", "exact"), self.mode
        self.dev = torch.device(device)
        self.mod = load()
        with torch.cuda.device(self.dev):
            self.parts = int(self.mod.num_sm_parts(1)) if num_sms is None else int(num_sms)
        self._sched: dict = {}
        self._ws: dict = {}

    def schedule(self, t: int, k: int, split: bool | None = None):
        split = split_policy(t, k) if split is None else split
        if not split:
            return None
        key = (t, k)
        s = self._sched.get(key)
        if s is None:
            meta = torch.empty(self.parts, 8, dtype=torch.int32, device=self.dev)
            ns = torch.empty(t + 1, dtype=torch.int32, device=self.dev)
            self.mod.sched_meta(t, 1, k, None, meta, ns, sched_overhead(t))
            s = (meta, ns)
            self._sched[key] = s
        return s

    def workspace(self, t: int):
        n = t + self.parts
        ws = self._ws.get("acc")
        if ws is None or ws[0].shape[0] < n:
            ws = (torch.empty(n, 1, H_Q, D_V, dtype=torch.float32, device=self.dev),
                  torch.empty(n, 1, H_Q, dtype=torch.float32, device=self.dev))
            self._ws["acc"] = ws
        return ws

    def prepare(self, t: int, k: int, split: bool | None = None) -> None:
        """Build the schedule and workspaces for batch t outside graph capture."""
        if self.schedule(t, k, split) is not None:
            self.workspace(t)

    def __call__(self, q: torch.Tensor, kv: torch.Tensor, table: torch.Tensor, sm_scale: float,
                 out: torch.Tensor | None = None, lse: torch.Tensor | None = None, split: bool | None = None):
        t = q.shape[0]
        assert q.dim() == 3 and q.shape[1:] == (H_Q, D_QK), q.shape
        if kv.dim() == 3:
            kv = kv.view(kv.shape[0], kv.shape[-1])
        assert kv.shape[-1] == ROW_BYTES, kv.shape
        if kv.dtype != torch.uint8:
            kv = kv.view(torch.uint8)
        k = table.shape[1]
        if out is None:
            out = torch.empty(t, H_Q, D_V, dtype=torch.bfloat16, device=q.device)
        if t == 0:
            return out
        if lse is None:
            lse = torch.empty(t, 1, H_Q, dtype=torch.float32, device=q.device)
        if k % B_TOPK:
            pad = torch.full((t, B_TOPK - k % B_TOPK), -1, dtype=torch.int32, device=q.device)
            table = torch.cat([table, pad], dim=1)
            k = table.shape[1]
        sched = self.schedule(t, k, split)
        o4 = out.view(t, 1, H_Q, D_V)
        if sched is None:
            self.mod.sparse_decode(q.view(t, 1, H_Q, D_QK), kv, table.view(t, 1, k), None, float(sm_scale), o4, lse,
                                   None, None, None, None, self.mode == "fp8")
        else:
            oa, la = self.workspace(t)
            self.mod.sparse_decode(q.view(t, 1, H_Q, D_QK), kv, table.view(t, 1, k), None, float(sm_scale), o4, lse,
                                   sched[0], sched[1], oa, la, self.mode == "fp8")
        return out


class DenseDecoder:
    """k27 dense mode: dense MLA decode over a paged FP4 pool (352-byte rows, row = page * page_size + offset), one q
    tile of 64 rows per sequence. Rows are tokens x heads (row t * n_heads + h; K2.7 under TP8 = 8 heads per rank, so
    up to 8 tokens of one sequence share one KV read), padded with zero rows to 64; row_limit [b, 64] gives every row the
    number of positions it may attend (context + its own token index + 1), so verify batches are causal over the draft
    rows. The split-KV schedule follows the real lengths (sched_meta with topk_length), so it is a kernel launch per
    call and graph-capturable; `topk` is max_seq_len rounded up to 64 (the block loop bound).

        dec = DenseDecoder(device)
        out = dec(q_fp8 [b, 64, 576], kv_rows, page_table [b, n_pages] int32, seq_lens [b] int32, page_size,
                  max_seq_len, row_limit [b, 64] int32 | None, sm_scale)   # [b, 64, 512] bf16
    """

    def __init__(self, device, num_sms: int | None = None):
        self.dev = torch.device(device)
        self.mod = load()
        with torch.cuda.device(self.dev):
            self.parts = int(self.mod.num_sm_parts(1)) if num_sms is None else int(num_sms)
        self._meta: dict = {}
        self._ws = None

    def _buffers(self, b: int):
        s = self._meta.get(b)
        if s is None:
            s = (torch.empty(self.parts, 8, dtype=torch.int32, device=self.dev),
                 torch.empty(b + 1, dtype=torch.int32, device=self.dev))
            self._meta[b] = s
        return s

    def workspace(self, b: int):
        n = b + self.parts
        if self._ws is None or self._ws[0].shape[0] < n:
            self._ws = (torch.empty(n, 1, H_Q, D_V, dtype=torch.float32, device=self.dev),
                        torch.empty(n, 1, H_Q, dtype=torch.float32, device=self.dev))
        return self._ws

    def prepare(self, b: int) -> None:
        """Allocate the schedule buffers and workspaces for batch b outside graph capture."""
        self._buffers(b)
        self.workspace(b)

    def __call__(self, q: torch.Tensor, kv: torch.Tensor, page_table: torch.Tensor, seq_lens: torch.Tensor,
                 page_size: int, max_seq_len: int, row_limit: torch.Tensor | None, sm_scale: float,
                 out: torch.Tensor | None = None, lse: torch.Tensor | None = None, split: bool | None = None):
        b = q.shape[0]
        assert q.dim() == 3 and q.shape[1:] == (H_Q, D_QK), q.shape
        if kv.dim() == 3:
            kv = kv.view(kv.shape[0], kv.shape[-1])
        if kv.dtype != torch.uint8:
            kv = kv.view(torch.uint8)
        assert kv.shape[-1] == ROW_BYTES, kv.shape
        if out is None:
            out = torch.empty(b, H_Q, D_V, dtype=torch.bfloat16, device=q.device)
        if b == 0:
            return out
        if lse is None:
            lse = torch.empty(b, H_Q, dtype=torch.float32, device=q.device)
        seq_lens = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        seq_lens = seq_lens.contiguous()
        k = (int(max_seq_len) + B_TOPK - 1) // B_TOPK * B_TOPK
        split = split_policy(b, k) if split is None else split
        if not split:
            self.mod.dense_decode(q, kv, page_table, seq_lens, int(page_size), int(max_seq_len), row_limit,
                                  float(sm_scale), out, lse, None, None, None, None)
        else:
            meta, ns = self._buffers(b)
            self.mod.sched_meta(b, 1, k, seq_lens, meta, ns, sched_overhead(b))
            oa, la = self.workspace(b)
            self.mod.dense_decode(q, kv, page_table, seq_lens, int(page_size), int(max_seq_len), row_limit,
                                  float(sm_scale), out, lse, meta, ns, oa, la)
        return out


def _glm_kv4():
    """glm_kv4 (the row format and its reference dequantizer): from operations/kernels when run from the repo, else
    from where the hooks install it in the engine."""
    try:
        import glm_kv4  # noqa: F401  (operations/kernels on sys.path, as the tests set it)
        return glm_kv4
    except ImportError:
        from sglang.srt.layers.attention.dsa import glm_kv4  # type: ignore
        return glm_kv4


def dense_reference(q: torch.Tensor, kv_rows: torch.Tensor, page_table: torch.Tensor, seq_lens: torch.Tensor,
                    page_size: int, row_limit: torch.Tensor | None, sm_scale: float) -> torch.Tensor:
    """fp32 reference of DenseDecoder: dequantize (glm_kv4 row format) the attended rows of every sequence and run
    softmax attention per q row with the row's position limit. q may be fp8 (E4M3) or bf16; returns [b, 64, 512]."""
    kv4 = _glm_kv4()
    b = q.shape[0]
    out = torch.zeros(b, H_Q, D_V, dtype=torch.float32, device=q.device)
    for i in range(b):
        n = int(seq_lens[i])
        pos = torch.arange(n, device=q.device)
        slots = page_table[i, pos // page_size].long() * page_size + (pos % page_size)
        rows = kv4.dequantize_rows_ref(kv_rows[slots], out_dtype=torch.float32)          # [n, 576]
        s = q[i].float() @ rows.T * sm_scale                                             # [64, n]
        if row_limit is not None:
            s = s.masked_fill(pos[None, :] >= row_limit[i][:, None], float("-inf"))
        p = torch.softmax(s, dim=-1).nan_to_num(0.0)
        out[i] = p @ rows[:, :D_V]
    return out
