"""glm_kv4_hooks: anchors, idempotence, clean import, the real pool class, and the production call sequence.

CPU (needs the local SGLang fork with tag v0.5.20): every hook applies to v0.5.20 exactly once, twice is a no-op,
both files compile, the write hook replaces the stock write, the route sits before the TRT-LLM call, the dim hook
sits before `return kv_cache_dim` on the trtllm branch.
Box (installed sglang + GPU): a copy of the installed package is patched and imported in a clean interpreter; the
real GLMKV4DSATokenToKVPool is built (352-byte rows, indexer cache, page size 64) and its write door checked bit
for bit; then DeepseekSparseAttnBackend._forward_trtllm is called on a stub backend with a two-request prefill
batch (FP8 quantize + RoPE, KV write, top-k -> page table: real SGLang code) with the FP8 pool (stock) and with the
FP4 pool on all three read paths (sequence scratch, selected-row scratch, fused Triton), and as a decode batch under
CUDA-graph capture + replay; every FP4 output is checked against the fp32 reference over the dequantized FP4 pool,
and the FP4-vs-FP8 gap is printed (measured, random Gaussian rows).
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import glm_kv4_hooks as hooks  # noqa: E402

FORK = pathlib.Path("/home/ubuntu/sglang-fork")
sglang_spec = importlib.util.find_spec("sglang")


def _v0520(path: str) -> str:
    if not (FORK / ".git").exists():
        pytest.skip("local SGLang fork not present")
    r = subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{path}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        pytest.skip("tag v0.5.20 not available")
    return r.stdout


def test_backend_hooks_apply_once_and_compile():
    src = _v0520(hooks.BACKEND)
    new, lines = hooks.patch_source(src, hooks.BACKEND)
    assert lines == [f"{h.name}: applied to {hooks.BACKEND}" for h in hooks.HOOKS if h.path == hooks.BACKEND]
    compile(new, "dsa_backend.py", "exec")
    again, lines2 = hooks.patch_source(new, hooks.BACKEND)
    assert again == new and all(ln.endswith("already applied") for ln in lines2)
    body = new[new.index("    def _forward_trtllm("):]
    body = body[: body.index("\n    def ", 10)]
    assert body.count(hooks.WRITE_ANCHOR) == 1  # the stock write survives only inside the else branch
    assert body.index("numinous-hook:glm_kv4_stash") < body.index("if self.kv_cache_dtype == torch.float8_e4m3fn:")
    assert body.index("numinous-hook:glm_kv4_write") < body.index("numinous-hook:glm_kv4_route")
    assert body.index("_pad_trtllm_sparse_page_table(") < body.index("numinous-hook:glm_kv4_route")
    assert body.index("numinous-hook:glm_kv4_route") < body.index("trtllm_batch_decode_with_kv_cache_mla")
    init = new[new.index("    def init_cuda_graph_state("):]
    init = init[: init.index("\n    def ", 10)]
    assert "numinous-hook:glm_kv4_graph" in init and "max_num_tokens" in init


def test_configurator_hooks_apply_once_and_compile():
    src = _v0520(hooks.CONFIGURATOR)
    new, lines = hooks.patch_source(src, hooks.CONFIGURATOR)
    assert lines == [f"{h.name}: applied to {hooks.CONFIGURATOR}" for h in hooks.HOOKS
                     if h.path == hooks.CONFIGURATOR]
    compile(new, "kv_cache_configurator.py", "exec")
    fn = new[new.index("def calculate_mla_kv_cache_dim("):]
    end = fn.find("\ndef ", 10)
    fn = fn[:end] if end > 0 else fn
    i = fn.index("numinous-hook:glm_kv4_dim")
    assert fn.index("if uses_trtllm_kv_layout:") < i < fn.index("return kv_cache_dim", i)
    build = new[new.index("    def _build_dsa_kv_pool("):]
    build = build[: build.index("\n    def ", 10)]
    assert build.index("PoolCls = DSATokenToKVPool") < build.index("numinous-hook:glm_kv4_pool") < build.index(
        "token_to_kv_pool = PoolCls(")


def test_missing_anchor_fails_loudly():
    with pytest.raises(SystemExit, match="anchor found 0 times"):
        hooks.patch_source("class X:\n    pass\n", hooks.BACKEND)


def test_modules_exist_and_compile():
    for local in hooks.MODULES:
        p = hooks.REPO / local
        compile(p.read_text(), str(p), "exec")


# ------------------------------------------------------------------------------------------------------ box tests

box = pytest.mark.skipif(sglang_spec is None, reason="needs an installed sglang (dev box)")


@pytest.fixture(scope="module")
def patched(tmp_path_factory) -> pathlib.Path:
    src = list(sglang_spec.submodule_search_locations)[0]
    base = tmp_path_factory.mktemp("sgl")
    shutil.copytree(src, base / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
    hooks.apply(str(base / "sglang"))
    return base


@box
def test_anchors_resolve_on_installed_sglang():
    lines = hooks.apply(list(sglang_spec.submodule_search_locations)[0], check=True)
    assert all(("anchor ok" in ln) or ln.endswith("already applied") for ln in lines), lines
    assert len(lines) == len(hooks.HOOKS)


@box
def test_apply_is_idempotent(patched):
    lines = hooks.apply(str(patched / "sglang"))
    assert [ln for ln in lines if not ln.startswith("installed")] == [f"{h.name}: already applied"
                                                                       for h in hooks.HOOKS]


@box
def test_patched_modules_import_in_clean_interpreter(patched):
    code = ("import sglang, sys, inspect; assert sglang.__file__.startswith(sys.argv[1]), sglang.__file__\n"
            "import sglang.srt.layers.attention.dsa_backend as m\n"
            "import sglang.srt.mem_cache.kv_cache_configurator as c\n"
            "src = inspect.getsource(m) + inspect.getsource(c)\n"
            f"assert all(m in src for m in {[h.marker for h in hooks.HOOKS]!r})\n"
            "from sglang.srt.layers.attention.dsa import glm_kv4_engine as e, glm_kv4 as k\n"
            "assert e.kv4 is k and e.ROW_BYTES == 352\n"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code, str(patched)], env={**os.environ, "PYTHONPATH": str(patched)},
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


POOL = r'''
import logging, sys, types
import flashinfer, torch
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(name)s %(message)s")
from sglang.srt.layers.attention.dsa import glm_kv4_engine as e, glm_kv4 as kv4
from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
hf = types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
assert e.enabled_for(hf, torch.float8_e4m3fn) is True
assert e.enabled_for(types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"]), torch.float8_e4m3fn) is False
try:
    e.enabled_for(hf, torch.bfloat16); raise SystemExit("bf16 must be refused")
except RuntimeError as ex:
    assert "fp8_e4m3" in str(ex)
dev = "cuda"
size, page = 64 * 100, 64
from sglang.srt.runtime_context import get_parallel
get_parallel().override_permanently(attn_dcp_size=1, dcp_enabled=False)  # no process groups in this test
pool = e.GLMKV4DSATokenToKVPool(size, page_size=page, kv_lora_rank=512, dtype=torch.float8_e4m3fn,
                               qk_rope_head_dim=64, layer_num=2, device=dev, index_head_dim=128,
                               enable_memory_saver=False, kv_cache_dim=352)
assert pool.kv_cache_dim == 352 and pool.dsa_kv_cache_store_fp8 is False and pool.numinous_kv4
assert len(pool.kv_buffer) == 2 and tuple(pool.kv_buffer[0].shape) == (size + page, 1, 352)
assert pool.kv_buffer[0].dtype == torch.uint8
assert pool.get_key_buffer(1).dtype == torch.float8_e4m3fn and pool.get_key_buffer(1).shape[-1] == 352
kv_bytes = pool.get_kv_size_bytes()  # latent rows + the unchanged indexer cache (128 fp8 + 4 scale bytes per layer)
assert kv_bytes == 2 * (size + page) * (352 + 132), kv_bytes
layer = types.SimpleNamespace(layer_id=1)
T = 300
k = torch.randn(T, 1, 512, device=dev).bfloat16()
k_rope = torch.randn(T, 1, 64, device=dev).to(torch.float8_e4m3fn)
loc = torch.randperm(size, device=dev)[:T] + page
pool.set_mla_kv_buffer(layer, loc, k, k_rope)
ref = kv4.quantize_rows_ref(k.view(T, 512), k_rope.view(T, 64))
assert torch.equal(pool.kv_buffer[1][loc, 0], ref)
assert pool.kv_buffer[0].abs().sum().item() == 0
n_bf16, r_bf16 = pool.get_mla_kv_buffer(layer, loc)
dq = kv4.dequantize_rows_ref(ref, torch.bfloat16)
assert torch.equal(n_bf16.view(T, 512), dq[:, :512]) and torch.equal(r_bf16.view(T, 64), dq[:, 512:])
# move (spec accept relocation): the latent rows move byte-exact (the indexer cache move is stock code, unchanged)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
tgt = torch.arange(T, device=dev) + page
MLATokenToKVPool.move_kv_cache(pool, tgt, loc)
assert torch.equal(pool.kv_buffer[1][tgt, 0], ref)
for bad in ("set_kv_buffer", "get_value_buffer"):
    try:
        getattr(pool, bad)(layer, loc, k, None) if bad == "set_kv_buffer" else pool.get_value_buffer(1)
        raise SystemExit(bad + " must refuse")
    except RuntimeError:
        pass
print("ok")
'''


@box
def test_real_pool_class(patched):
    env = {**os.environ, "PYTHONPATH": str(patched), "NUMINOUS_GLM_KV4": "1"}
    r = subprocess.run([sys.executable, "-c", POOL], env=env, capture_output=True, text=True, timeout=900)
    print(r.stdout[-2000:])
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-2000:] + r.stderr[-4000:]
    assert "numinous glm_kv4 pool: 2 layers x 352 B/token" in r.stdout


SEQUENCE = r'''
import logging, sys, types
import flashinfer, torch
from sglang.srt.layers.attention import dsa_backend as db
from sglang.srt.layers.attention.dsa import glm_kv4_engine as e, glm_kv4 as kv4
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.layers.attention.trtllm_mla_backend import make_persistent_multi_ctas_kv_counter_buffer

logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(name)s %(message)s")
torch.manual_seed(0)
dev, H, K, PAGE = "cuda", 64, 2048, 64
prefix, extend = [1500, 5000], [200, 312]
seqs = [p + e for p, e in zip(prefix, extend)]
T, max_seq = sum(extend), max(seqs)
n_pages = 400
N = (n_pages + 1) * PAGE
src_nope = torch.randn(N, 512, device=dev) * 1.5
src_rope = torch.randn(N, 64, device=dev) * 1.5
pool8 = torch.zeros(N, 1, 576, dtype=torch.uint8, device=dev)
pool8[:, 0, :512] = src_nope.to(torch.float8_e4m3fn).view(torch.uint8)
pool8[:, 0, 512:] = src_rope.to(torch.float8_e4m3fn).view(torch.uint8)
pool4 = kv4.quantize_rows_ref(src_nope.bfloat16(), src_rope.to(torch.float8_e4m3fn)).view(N, 1, 352)
perm = torch.randperm(n_pages, device=dev)
req_to_token = torch.zeros(2, max_seq, dtype=torch.int32, device=dev)
start = 0
for r, s in enumerate(seqs):
    np_ = -(-s // PAGE)
    slots = (perm[start:start + np_, None] * PAGE + torch.arange(PAGE, device=dev)).reshape(-1)[:s]
    req_to_token[r, :s] = slots.int()
    start += np_
positions = torch.cat([torch.arange(p, p + e, device=dev) for p, e in zip(prefix, extend)])
out_cache_loc = torch.cat([req_to_token[r, p:p + e] for r, (p, e) in enumerate(zip(prefix, extend))]).long()
topk = torch.full((T, K), -1, dtype=torch.int32, device=dev)
for t in range(T):
    n = min(int(positions[t]) + 1, K)
    topk[t, :n] = torch.randperm(int(positions[t]) + 1, device=dev)[:n].int()
dsa_seq = torch.clamp(positions + 1, max=K).int()

class Pool8:
    kv_cache_dim = 576
    def set_mla_kv_buffer(self, layer, loc, k, k_rope):
        pool8[loc, 0, :512] = k.view(torch.uint8).view(-1, 512)
        pool8[loc, 0, 512:] = k_rope.view(torch.uint8).view(-1, 64)
    def get_key_buffer(self, layer_id):
        return pool8.view(torch.float8_e4m3fn)

class Pool4:
    numinous_kv4 = True
    kv_cache_dim = 352
    start_layer = 0
    kv_buffer = [pool4]
    def set_mla_kv_buffer(self, layer, loc, k, k_rope):
        assert k.dtype == torch.bfloat16, k.dtype  # the hook hands the bf16 latent over
        kv4.quant_write(pool4, loc, k, k_rope)
    def get_key_buffer(self, layer_id):
        return pool4.view(torch.float8_e4m3fn)

def make_stub(pool):
    stub = types.SimpleNamespace(
        forward_metadata=types.SimpleNamespace(
            page_table_1=req_to_token, dsa_extend_seq_lens_list=extend,
            cu_seqlens_q=torch.tensor([0, extend[0], T], dtype=torch.int32, device=dev),
            cache_seqlens_int32=torch.tensor(seqs, dtype=torch.int32, device=dev), max_seq_len_k=max_seq),
        kv_cache_dtype=torch.float8_e4m3fn, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=192,
        token_to_kv_pool=pool, real_page_size=PAGE, kv_cache_dim=pool.kv_cache_dim, use_fused_topk=False, device=dev,
        num_q_heads=H, workspace_buffer=torch.empty(512 << 20, dtype=torch.uint8, device=dev),
        _multi_ctas_kv_counter_buffer=make_persistent_multi_ctas_kv_counter_buffer(torch.device(dev), H, 8192),
        dsa_prefill_impl="trtllm", dsa_decode_impl="trtllm", device_sm_major=torch.cuda.get_device_capability()[0],
        dsa_index_topk=K, use_mha=False,
    )
    stub._pad_trtllm_sparse_page_table = types.MethodType(db.DeepseekSparseAttnBackend._pad_trtllm_sparse_page_table, stub)
    stub._pad_topk_indices = types.MethodType(db.DeepseekSparseAttnBackend._pad_topk_indices, stub)
    return stub

layer = types.SimpleNamespace(tp_q_head_num=H, head_dim=576, v_head_dim=512, k_scale_float=None,
                              scaling=576 ** -0.5, layer_id=0, is_cross_attention=False)
fb = types.SimpleNamespace(positions=positions, out_cache_loc=out_cache_loc, forward_mode=ForwardMode.EXTEND,
                           seq_lens_sum=sum(seqs), attn_cp_metadata=None, batch_size=2,
                           seq_lens_cpu=torch.tensor(seqs, dtype=torch.int32))
q_nope = torch.randn(T, H, 512, device=dev).bfloat16()
q_rope = torch.randn(T, H, 64, device=dev).bfloat16()
k = torch.randn(T, 1, 512, device=dev).bfloat16()
k_rope = torch.randn(T, 1, 64, device=dev).bfloat16()

def run(stub):
    return db.DeepseekSparseAttnBackend._forward_trtllm(
        stub, q_nope, k, None, layer, fb, dsa_seq, True, q_rope, k_rope, topk, None, False, None, is_prefill=True)

# stock: FP8 pool
s8 = make_stub(Pool8()); s8._numinous_kv4 = False
o8 = run(s8)
assert not hasattr(s8, "_numinous_kv4_state")

# FP4 pool through attach (the init hook) and the three read paths
s4 = make_stub(Pool4())
mr = types.SimpleNamespace(token_to_kv_pool=s4.token_to_kv_pool)
s4._numinous_kv4 = e.attach(s4, mr)
assert s4._numinous_kv4 is True
st = e._state(s4)
assert st.mode == "scratch"
o4_seq = run(s4)
assert st.calls == {"seq": 1, "sel": 0, "triton": 0}, st.calls
assert o4_seq.shape == o8.shape and o4_seq.dtype == o8.dtype
# the FP4 write door was used for the 512 new rows: pool4 holds their quantized bf16 latent
new_rows = pool4[out_cache_loc, 0]
assert torch.equal(new_rows, kv4.quantize_rows_ref(k.view(T, 512), k_rope.view(T, 64).to(torch.float8_e4m3fn)))
st.seq_rows_cap = 0  # force the selected-row path (chunked: 512 tokens > sel_tokens 256 -> 2 chunks), with dedup
o4_sel = run(s4)
assert st.calls["sel"] == 1 and st.calls["seq"] == 1 and st.dedup and T >= e.DEDUP_MIN_ROWS
assert (st.stamp == kv4.STAMP_CLEAN).all()
st.dedup = False
o4_sel_plain = run(s4)
assert st.calls["sel"] == 2
assert torch.equal(o4_sel, o4_sel_plain), "dedup only moves rows; the op sees the same FP8 values in the same order"
st.dedup = True
st.mode = "triton"
o4_tri = run(s4)
assert st.calls["triton"] == 1

# references: fp32 attention over the dequantized FP4 pool with the same fp8 q and pool-slot table
# q with RoPE skipped (cos_sin_cache None -> mla_quantize_for_fp8_no_rope): the fp8 q the kernel saw
from sglang.kernels.ops.attention.utils import mla_quantize_for_fp8_no_rope
q8, _, _ = mla_quantize_for_fp8_no_rope(q_nope, q_rope, k.squeeze(1), k_rope.squeeze(1), 512, 64)
from sglang.kernels.ops.attention.dsa.transform_index import transform_index_page_table_prefill
table = transform_index_page_table_prefill(page_table=req_to_token, topk_indices=topk, extend_lens_cpu=extend,
                                           page_size=1, output_num_tokens=T, page_table_is_expanded=False,
                                           cu_seqlens_q=s4.forward_metadata.cu_seqlens_q)
sub = list(range(0, T, 4))
ref4 = kv4.sparse_mla_ref(q8[sub], pool4.view(N, 352), table[sub], layer.scaling)
# the scratch paths feed FP8-requantized rows to the TRT-LLM op: their kernel error is measured against that
rows4_8 = kv4.dequantize_rows_ref(pool4.view(N, 352), torch.float8_e4m3fn).float()
def ref_rows(rows_f32, idx):
    out = torch.zeros(len(idx), H, 512, device=dev)
    for i, t in enumerate(idx):
        sl = table[t].long(); v = sl >= 0
        kk = rows_f32[sl[v]]
        p = torch.softmax(q8[t].float() @ kk.T * layer.scaling, -1)
        out[i] = p @ kk[:, :512]
    return out
ref4_8 = ref_rows(rows4_8, sub)
ref8_rows = pool8.view(N, 576).view(torch.float8_e4m3fn).float()
def ref_fp8(idx):
    out = torch.zeros(len(idx), H, 512, device=dev)
    for i, t in enumerate(idx):
        sl = table[t].long(); v = sl >= 0
        kk = ref8_rows[sl[v]]
        p = torch.softmax(q8[t].float() @ kk.T * layer.scaling, -1)
        out[i] = p @ kk[:, :512]
    return out
ref8 = ref_fp8(sub)
def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
r_trt8 = rel(o8[sub, 0], ref8)
r_seq, r_sel, r_tri = rel(o4_seq[sub, 0], ref4_8), rel(o4_sel[sub, 0], ref4_8), rel(o4_tri[sub, 0], ref4)
r_seq4 = rel(o4_seq[sub, 0], ref4)  # includes the FP4 -> FP8 requantization of the scratch (E4M3 rounding)
gap = rel(o4_seq[sub, 0], o8[sub, 0])
print(f"kernel-vs-fp32 (max err / max |ref|): trtllm fp8 pool {r_trt8:.4f}; fp4 scratch seq {r_seq:.4f} sel {r_sel:.4f} "
      f"(vs exact fp4 rows {r_seq4:.4f}); fused triton {r_tri:.4f}; fp4 vs fp8 output gap {gap:.4f}; "
      f"seq vs sel {rel(o4_seq, o4_sel):.5f}")
assert r_seq < 0.03 and r_sel < 0.03 and r_tri < 0.03, (r_seq, r_sel, r_tri)
assert r_seq4 < 0.1, r_seq4
assert torch.equal(o4_seq, o4_sel), "both scratch paths feed identical FP8 rows to the same op"

# decode under CUDA-graph capture: bs rows, one token each, graph scratch reserved by the init_cuda_graph_state hook
bs = 8
st.mode = "scratch"
e.reserve_graph_scratch(s4, bs)
assert st.graph_scratch.shape[0] == bs * K
dec_topk = torch.full((bs, K), -1, dtype=torch.int32, device=dev)
for b in range(bs):
    s = seqs[b % 2]
    n = min(s, K); dec_topk[b, :n] = torch.randperm(s, device=dev)[:n].int()
dec_pt = req_to_token[[b % 2 for b in range(bs)]]
s4.forward_metadata = types.SimpleNamespace(page_table_1=dec_pt, cache_seqlens_int32=torch.tensor(
    [seqs[b % 2] for b in range(bs)], dtype=torch.int32, device=dev), max_seq_len_k=max_seq)
dq_nope = torch.randn(bs, H, 512, device=dev).bfloat16(); dq_rope = torch.randn(bs, H, 64, device=dev).bfloat16()
dk = torch.randn(bs, 1, 512, device=dev).bfloat16(); dk_rope = torch.randn(bs, 1, 64, device=dev).bfloat16()
dloc = torch.arange(bs, device=dev) + N - PAGE
dfb = types.SimpleNamespace(positions=torch.tensor([seqs[b % 2] for b in range(bs)], device=dev), out_cache_loc=dloc,
                            forward_mode=ForwardMode.DECODE, seq_lens_sum=0, attn_cp_metadata=None, batch_size=bs,
                            seq_lens_cpu=None)
def run_dec():
    return db.DeepseekSparseAttnBackend._forward_trtllm(
        s4, dq_nope, dk, None, layer, dfb, None, True, dq_rope, dk_rope, dec_topk, None, False, None)
o_eager = run_dec()
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    run_dec()
    with torch.cuda.graph(g):
        o_graph = run_dec()
torch.cuda.current_stream().wait_stream(s)
# new table contents in place: the same slots permuted inside each row's packed valid prefix (the TRT-LLM
# contract: valid entries first, then -1; the op takes min(seq_len, top-k) entries), so the softmax is the same
# set and the replayed graph must reproduce the eager result on the new table
for b in range(bs):
    n = min(seqs[b % 2], K)
    dec_topk[b, :n] = dec_topk[b, :n][torch.randperm(n, device=dev)]
g.replay(); torch.cuda.synchronize()
o_after = run_dec()
print(f"graph replay vs eager {rel(o_graph, o_after):.5f}; eager before the permutation {rel(o_eager, o_after):.5f}")
assert rel(o_graph, o_after) < 0.02 and rel(o_eager, o_after) < 0.02 and torch.isfinite(o_graph.float()).all()
print("ok")
'''


@box
def test_production_sequence_fp8_vs_fp4_and_graph(patched):
    env = {**os.environ, "PYTHONPATH": str(patched), "NUMINOUS_GLM_KV4": "1"}
    r = subprocess.run([sys.executable, "-c", SEQUENCE], env=env, capture_output=True, text=True, timeout=1200)
    print(r.stdout[-3000:])
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-3000:] + r.stderr[-5000:]
    assert r.stdout.count("numinous glm_kv4 attention active") == 1


# ---------------------------------------------------------------- composition with the final GLM stack's patchers

KERNELS = pathlib.Path.home() / "numinous-kernels"
FINAL_GLM_PATCHERS = [("glm_tp1_shared_hooks.py", []), ("glm_dpmoe_hooks.py", []),
                      ("sglang_numinous_hooks.py", ["tactics"]), ("glm_kv4_hooks.py", [])]


def test_kv4_hooks_compose_with_the_final_glm_stack(tmp_path):
    """The decode-burst series, the TP1 fix, the fused DP-MoE hooks, the tactics runner and glm_kv4 all apply to one
    v0.5.20 tree (the final_glm_kv4 image build) and the touched modules byte-compile."""
    if not (FORK / ".git").exists() or not (KERNELS / ".git").exists():
        pytest.skip("needs ~/sglang-fork and ~/numinous-kernels")
    sys.path.insert(0, str(HERE.parent / "modal"))
    from gate_arms import GLM_V20_PATCHES

    tar = subprocess.run(["git", "-C", str(FORK), "archive", "v0.5.20", "python/sglang"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=tar.stdout, check=True)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for p in GLM_V20_PATCHES[0]["files"]:
        blob = subprocess.run(["git", "-C", str(KERNELS), "show",
                               f"{GLM_V20_PATCHES[0]['ref']}:patches/sglang-v0.5.20/{p}"], capture_output=True,
                              check=True).stdout
        subprocess.run(["git", "apply"], cwd=tmp_path, input=blob, check=True)
    tree = tmp_path / "python/sglang"
    logs = []
    for name, args in FINAL_GLM_PATCHERS:
        r = subprocess.run([sys.executable, str(HERE / name), *args, "--root", str(tree)], capture_output=True,
                           text=True, timeout=300)
        assert r.returncode == 0, (name, r.stdout[-800:], r.stderr[-800:])
        logs.append(r.stdout)
    log = "\n".join(logs)
    for hook in ("tp1_shared_once_normal", "glm_dpmoe: applied", "tactics_runner") + tuple(f"{h.name}: applied" for h in hooks.HOOKS):
        assert hook in log, (hook, log[-1500:])
    r = subprocess.run([sys.executable, "-m", "py_compile", *[str(tree / p) for p in (
        "srt/layers/attention/dsa_backend.py", "srt/mem_cache/kv_cache_configurator.py", "srt/models/deepseek_v2.py",
        "srt/layers/moe/moe_runner/flashinfer_trtllm.py", "srt/layers/attention/dsa/glm_kv4.py",
        "srt/layers/attention/dsa/glm_kv4_engine.py")]], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
