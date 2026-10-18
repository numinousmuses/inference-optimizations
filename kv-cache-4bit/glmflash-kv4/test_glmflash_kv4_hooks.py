"""glmflash_kv4_hooks: anchors on the fork's v0.5.20 sources, idempotence, compile, hook ordering, and that the
modules it installs import together on CPU (glmflash_kv4 over glm_kv4). The engine module needs sglang (box).
Box tests of the real pool / attention path: to be added with test_glm_kv4_hooks.py's fixtures once a B200 is
available (2026-09-29: none)."""
from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kernels"))

import glm_kv4_hooks as glm_hooks  # noqa: E402
import glmflash_kv4_hooks as hooks  # noqa: E402

FORK = pathlib.Path("/home/ubuntu/sglang-fork")


def _v0520(path: str) -> str:
    if not (FORK / ".git").exists():
        pytest.skip("local SGLang fork not present")
    r = subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{path}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        pytest.skip("tag v0.5.20 not available")
    return r.stdout


def test_hook_names_and_modules_are_flash_specific():
    assert all(h.name.startswith("glmflash_kv4_") for h in hooks.HOOKS)
    shared = {"glmflash_kv4_" + h.name[len("glm_kv4_"):]: h for h in glm_hooks.HOOKS}
    assert set(shared) < {h.name for h in hooks.HOOKS}
    assert {h.name for h in hooks.HOOKS} - set(shared) == {"glmflash_kv4_hybrid_pool"}   # Flash's extra seam
    for h in hooks.HOOKS:
        if h.name in shared:
            g = shared[h.name]
            assert h.anchor == g.anchor and h.path == g.path and h.replace == g.replace   # same v0.5.20 seams
        assert "glm_kv4_engine" not in h.text and "_numinous_kv4" not in h.text
    assert set(hooks.MODULES) == {"operations/kernels/glm_kv4.py", "operations/kernels/glmflash_kv4.py",
                                  "operations/kernels/glmflash_kv4_engine.py"}
    for local in hooks.MODULES:
        assert (hooks.REPO / local).exists(), local


def test_backend_hooks_apply_once_and_compile():
    src = _v0520(hooks.BACKEND)
    new, lines = hooks.patch_source(src, hooks.BACKEND)
    assert lines == [f"{h.name}: applied to {hooks.BACKEND}" for h in hooks.HOOKS if h.path == hooks.BACKEND]
    compile(new, "dsa_backend.py", "exec")
    again, lines2 = hooks.patch_source(new, hooks.BACKEND)
    assert again == new and all(ln.endswith("already applied") for ln in lines2)
    body = new[new.index("    def _forward_trtllm("):]
    body = body[: body.index("\n    def ", 10)]
    assert body.count(hooks.WRITE_ANCHOR) == 1
    assert body.index("numinous-hook:glmflash_kv4_stash") < body.index("if self.kv_cache_dtype == torch.float8_e4m3fn:")
    assert body.index("numinous-hook:glmflash_kv4_write") < body.index("numinous-hook:glmflash_kv4_route")
    assert body.index("numinous-hook:glmflash_kv4_route") < body.index("trtllm_batch_decode_with_kv_cache_mla")
    # the rope-0 metadata the stock path computes sits after our route: engine_forward computes its own
    assert body.index("numinous-hook:glmflash_kv4_route") < body.index("prepare_trtllm_nope_sparse_metadata(page_table_1)")


def test_memory_pool_hybrid_hook_applies_once_and_compiles():
    src = _v0520(hooks.MEMORY_POOL)
    new, lines = hooks.patch_source(src, hooks.MEMORY_POOL)
    assert lines == [f"glmflash_kv4_hybrid_pool: applied to {hooks.MEMORY_POOL}"]
    compile(new, "memory_pool.py", "exec")
    again, lines2 = hooks.patch_source(new, hooks.MEMORY_POOL)
    assert again == new and lines2 == ["glmflash_kv4_hybrid_pool: already applied"]
    assert src.count(hooks.HYBRID_ANCHOR) == 1 and hooks.HYBRID_ANCHOR not in new
    body = new[new.index("class HybridLinearKVPool"):]
    assert "pool_class(DSATokenToKVPool, kv_cache_dim)(" in body


def test_configurator_hooks_apply_once_and_compile():
    src = _v0520(hooks.CONFIGURATOR)
    new, lines = hooks.patch_source(src, hooks.CONFIGURATOR)
    assert len(lines) == 2
    compile(new, "kv_cache_configurator.py", "exec")
    fn = new[new.index("def calculate_mla_kv_cache_dim("):]
    nxt = fn.find("\ndef ", 10)
    fn = fn if nxt < 0 else fn[:nxt]
    # the hook sits right after the trtllm-layout test, before that branch's `return kv_cache_dim`
    branch = fn[fn.index("    if uses_trtllm_kv_layout:\n"):]
    assert branch.index("numinous-hook:glmflash_kv4_dim") < branch.index("return kv_cache_dim")


def test_glm_and_flash_hooks_do_not_share_markers():
    src = _v0520(hooks.BACKEND)
    flash, _ = hooks.patch_source(src, hooks.BACKEND)
    assert "numinous-hook:glm_kv4_" not in flash and "glm_kv4_engine" not in flash
    glm, _ = glm_hooks.patch_source(src, hooks.BACKEND)
    assert "numinous-hook:glmflash_kv4_" not in glm
    # the two patchers target different models and never share an image (glmflash_arms builds its own overlay)


def test_kernel_module_imports_on_cpu():
    fk = importlib.import_module("glmflash_kv4")
    assert fk.ROW_BYTES == 288 and fk.FP8_ROW_BYTES == 512
    eng = (hooks.REPO / "operations/kernels/glmflash_kv4_engine.py").read_text()
    compile(eng, "glmflash_kv4_engine.py", "exec")
    assert 'ARCH_PREFIX = "Glm5Next"' in eng and 'ENV = "NUMINOUS_GLMFLASH_KV4"' in eng
    assert "sparse_mla_top_k_lens=prepare_trtllm_nope_sparse_metadata(table)" in eng
    assert "def pool_class(default_cls, kv_cache_dim)" in eng and "_dense_layer(pool, layer.layer_id)" in eng
    assert "kv4.quant_write(dst_buffer, loc, cache_k_nope)" in eng


# ------------------------------------------------------------------------------------------------------ box tests
# test_glm_kv4_hooks.py's box tests at the Flash geometry (rope 0, 288-B rows, 512-B scratch, head_dim 512). NOT RUN
# yet (no B200 2026-09-29). Run: python3 operations/devbox/devbox.py test <box> operations/patches/test_glmflash_kv4_hooks.py

import importlib.util
import os
import shutil

sglang_spec = importlib.util.find_spec("sglang")

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
            "from sglang.srt.layers.attention.dsa import glmflash_kv4_engine as e, glmflash_kv4 as k\n"
            "assert e.kv4 is k and e.ROW_BYTES == 288 and k.DIM_ROPE == 0\n"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code, str(patched)], env={**os.environ, "PYTHONPATH": str(patched)},
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


POOL = r'''
import logging, sys, types
import flashinfer, torch
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(name)s %(message)s")
from sglang.srt.layers.attention.dsa import glmflash_kv4_engine as e, glmflash_kv4 as kv4
from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
hf = types.SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"])
assert e.enabled_for(hf, torch.float8_e4m3fn) is True
assert e.enabled_for(types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]), torch.float8_e4m3fn) is False
try:
    e.enabled_for(hf, torch.bfloat16); raise SystemExit("bf16 must be refused")
except RuntimeError as ex:
    assert "fp8_e4m3" in str(ex)
dev = "cuda"
size, page = 64 * 100, 64
from sglang.srt.runtime_context import get_parallel
get_parallel().override_permanently(attn_dcp_size=1, dcp_enabled=False)  # no process groups in this test
pool = e.GLMFlashKV4DSATokenToKVPool(size, page_size=page, kv_lora_rank=512, dtype=torch.float8_e4m3fn,
                                    qk_rope_head_dim=0, layer_num=2, device=dev, index_head_dim=128,
                                    enable_memory_saver=False, kv_cache_dim=288)
assert pool.kv_cache_dim == 288 and pool.dsa_kv_cache_store_fp8 is False and pool.numinous_kv4
assert len(pool.kv_buffer) == 2 and tuple(pool.kv_buffer[0].shape) == (size + page, 1, 288)
assert pool.kv_buffer[0].dtype == torch.uint8
assert pool.get_key_buffer(1).dtype == torch.float8_e4m3fn and pool.get_key_buffer(1).shape[-1] == 288
kv_bytes = pool.get_kv_size_bytes()  # latent rows + the unchanged indexer cache (128 fp8 + 4 scale bytes per layer)
assert kv_bytes == 2 * (size + page) * (288 + 132), kv_bytes
layer = types.SimpleNamespace(layer_id=1)
T = 300
k = torch.randn(T, 1, 512, device=dev).bfloat16()
k_rope = torch.empty(T, 1, 0, device=dev, dtype=torch.float8_e4m3fn)  # the zero-width rope the FP8 path passes
loc = torch.randperm(size, device=dev)[:T] + page
pool.set_mla_kv_buffer(layer, loc, k, k_rope)
ref = kv4.quantize_rows_ref(k.view(T, 512))
assert torch.equal(pool.kv_buffer[1][loc, 0], ref)
assert pool.kv_buffer[0].abs().sum().item() == 0
n_bf16, r_bf16 = pool.get_mla_kv_buffer(layer, loc)
dq = kv4.dequantize_rows_ref(ref, torch.bfloat16)
assert torch.equal(n_bf16.view(T, 512), dq) and r_bf16.shape == (T, 1, 0)
# move (spec accept relocation): the latent rows move byte-exact (the indexer cache move is stock code, unchanged)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
tgt = torch.arange(T, device=dev) + page
MLATokenToKVPool.move_kv_cache(pool, tgt, loc)
assert torch.equal(pool.kv_buffer[1][tgt, 0], ref)
# the Flash path: HybridLinearKVPool builds the DSA pool inside its __init__ (memory_pool hook) and maps layer ids
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
hyb = HybridLinearKVPool(size, dtype=torch.float8_e4m3fn, page_size=page, head_num=1, head_dim=1,
                         full_attention_layer_ids=[3, 7], device=dev, mamba_pool=None, use_mla=True, kv_lora_rank=512,
                         qk_rope_head_dim=0, use_dsa=True, index_head_dim=128, kv_cache_dim=288)
assert type(hyb.full_kv_pool).__name__ == "GLMFlashKV4DSATokenToKVPool", type(hyb.full_kv_pool).__name__
assert e._inner(hyb) is hyb.full_kv_pool and e._dense_layer(hyb, 7) == 1 and e._dense_layer(hyb, 3) == 0
layer7 = types.SimpleNamespace(layer_id=7)
hyb.set_mla_kv_buffer(layer7, loc, k, k_rope)          # the wrapper maps layer 7 -> dense 1 and hands the bf16 latent over
assert torch.equal(hyb.full_kv_pool.kv_buffer[1][loc, 0], ref)
stock = HybridLinearKVPool(size, dtype=torch.float8_e4m3fn, page_size=page, head_num=1, head_dim=1,
                           full_attention_layer_ids=[3], device=dev, mamba_pool=None, use_mla=True, kv_lora_rank=512,
                           qk_rope_head_dim=0, use_dsa=True, index_head_dim=128, kv_cache_dim=512)
assert type(stock.full_kv_pool).__name__ == "DSATokenToKVPool"   # stock dim -> stock class, even with the env on
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
    env = {**os.environ, "PYTHONPATH": str(patched), "NUMINOUS_GLMFLASH_KV4": "1"}
    r = subprocess.run([sys.executable, "-c", POOL], env=env, capture_output=True, text=True, timeout=900)
    print(r.stdout[-2000:])
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-2000:] + r.stderr[-4000:]
    assert "numinous glmflash_kv4 pool: 2 layers x 288 B/token" in r.stdout


SEQUENCE = r'''
import logging, sys, types
import flashinfer, torch
from sglang.srt.layers.attention import dsa_backend as db
from sglang.srt.layers.attention.dsa import glmflash_kv4_engine as e, glmflash_kv4 as kv4
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
pool8 = torch.zeros(N, 1, 512, dtype=torch.uint8, device=dev)
pool8[:, 0, :] = src_nope.to(torch.float8_e4m3fn).view(torch.uint8)
pool4 = kv4.quantize_rows_ref(src_nope.bfloat16()).view(N, 1, 288)
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
    kv_cache_dim = 512
    def set_mla_kv_buffer(self, layer, loc, k, k_rope):
        pool8[loc, 0, :] = k.view(torch.uint8).view(-1, 512)
    def get_key_buffer(self, layer_id):
        return pool8.view(torch.float8_e4m3fn)

class Pool4:
    numinous_kv4 = True
    kv_cache_dim = 288
    start_layer = 0
    kv_buffer = [pool4]
    def set_mla_kv_buffer(self, layer, loc, k, k_rope):
        assert k.dtype == torch.bfloat16, k.dtype  # the hook hands the bf16 latent over
        assert k_rope is None or k_rope.numel() == 0
        kv4.quant_write(pool4, loc, k)
    def get_key_buffer(self, layer_id):
        return pool4.view(torch.float8_e4m3fn)

def make_stub(pool):
    stub = types.SimpleNamespace(
        forward_metadata=types.SimpleNamespace(
            page_table_1=req_to_token, dsa_extend_seq_lens_list=extend,
            cu_seqlens_q=torch.tensor([0, extend[0], T], dtype=torch.int32, device=dev),
            cache_seqlens_int32=torch.tensor(seqs, dtype=torch.int32, device=dev), max_seq_len_k=max_seq),
        kv_cache_dtype=torch.float8_e4m3fn, kv_lora_rank=512, qk_rope_head_dim=0, qk_nope_head_dim=256,
        token_to_kv_pool=pool, real_page_size=PAGE, kv_cache_dim=pool.kv_cache_dim, use_fused_topk=False, device=dev,
        num_q_heads=H, workspace_buffer=torch.empty(512 << 20, dtype=torch.uint8, device=dev),
        _multi_ctas_kv_counter_buffer=make_persistent_multi_ctas_kv_counter_buffer(torch.device(dev), H, 8192),
        dsa_prefill_impl="trtllm", dsa_decode_impl="trtllm", device_sm_major=torch.cuda.get_device_capability()[0],
        dsa_index_topk=K, use_mha=False,
    )
    stub._pad_trtllm_sparse_page_table = types.MethodType(db.DeepseekSparseAttnBackend._pad_trtllm_sparse_page_table, stub)
    stub._pad_topk_indices = types.MethodType(db.DeepseekSparseAttnBackend._pad_topk_indices, stub)
    return stub

layer = types.SimpleNamespace(tp_q_head_num=H, head_dim=512, v_head_dim=512, k_scale_float=None,
                              scaling=512 ** -0.5, layer_id=0, is_cross_attention=False)
fb = types.SimpleNamespace(positions=positions, out_cache_loc=out_cache_loc, forward_mode=ForwardMode.EXTEND,
                           seq_lens_sum=sum(seqs), attn_cp_metadata=None, batch_size=2,
                           seq_lens_cpu=torch.tensor(seqs, dtype=torch.int32))
q_nope = torch.randn(T, H, 512, device=dev).bfloat16()
q_rope = torch.empty(T, H, 0, device=dev).bfloat16()  # the zero-width rope tensors the no-RoPE path passes
k = torch.randn(T, 1, 512, device=dev).bfloat16()
k_rope = torch.empty(T, 1, 0, device=dev).bfloat16()

def run(stub):
    return db.DeepseekSparseAttnBackend._forward_trtllm(
        stub, q_nope, k, None, layer, fb, dsa_seq, True, q_rope, k_rope, topk, None, False, None, is_prefill=True)

# stock: FP8 pool
s8 = make_stub(Pool8()); s8._numinous_fkv4 = False
o8 = run(s8)
assert not hasattr(s8, "_numinous_fkv4_state")

# FP4 pool through attach (the init hook) and the three read paths
s4 = make_stub(Pool4())
mr = types.SimpleNamespace(token_to_kv_pool=s4.token_to_kv_pool)
s4._numinous_fkv4 = e.attach(s4, mr)
assert s4._numinous_fkv4 is True
st = e._state(s4)
assert st.mode == "scratch"
o4_seq = run(s4)
assert st.calls == {"seq": 1, "sel": 0, "triton": 0}, st.calls
assert o4_seq.shape == o8.shape and o4_seq.dtype == o8.dtype
# the FP4 write door was used for the new rows: pool4 holds their quantized bf16 latent
new_rows = pool4[out_cache_loc, 0]
assert torch.equal(new_rows, kv4.quantize_rows_ref(k.view(T, 512)))
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
q8, _, _ = mla_quantize_for_fp8_no_rope(q_nope, q_rope, k.squeeze(1), k_rope.squeeze(1), 512, 0)
from sglang.kernels.ops.attention.dsa.transform_index import transform_index_page_table_prefill
table = transform_index_page_table_prefill(page_table=req_to_token, topk_indices=topk, extend_lens_cpu=extend,
                                           page_size=1, output_num_tokens=T, page_table_is_expanded=False,
                                           cu_seqlens_q=s4.forward_metadata.cu_seqlens_q)
sub = list(range(0, T, 4))
ref4 = kv4.sparse_mla_ref(q8[sub], pool4.view(N, 288), table[sub], layer.scaling)
# the scratch paths feed FP8-requantized rows to the TRT-LLM op: their kernel error is measured against that
rows4_8 = kv4.dequantize_rows_ref(pool4.view(N, 288), torch.float8_e4m3fn).float()
def ref_rows(rows_f32, idx):
    out = torch.zeros(len(idx), H, 512, device=dev)
    for i, t in enumerate(idx):
        sl = table[t].long(); v = sl >= 0
        kk = rows_f32[sl[v]]
        p = torch.softmax(q8[t].float() @ kk.T * layer.scaling, -1)
        out[i] = p @ kk[:, :512]
    return out
ref4_8 = ref_rows(rows4_8, sub)
ref8_rows = pool8.view(N, 512).view(torch.float8_e4m3fn).float()
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
dq_nope = torch.randn(bs, H, 512, device=dev).bfloat16(); dq_rope = torch.empty(bs, H, 0, device=dev).bfloat16()
dk = torch.randn(bs, 1, 512, device=dev).bfloat16(); dk_rope = torch.empty(bs, 1, 0, device=dev).bfloat16()
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
    env = {**os.environ, "PYTHONPATH": str(patched), "NUMINOUS_GLMFLASH_KV4": "1"}
    r = subprocess.run([sys.executable, "-c", SEQUENCE], env=env, capture_output=True, text=True, timeout=1200)
    print(r.stdout[-3000:])
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-3000:] + r.stderr[-5000:]
    assert r.stdout.count("numinous glmflash_kv4 attention active") == 1




# ---------------------------------------------------------------- composition with the Flash stack's other pieces

import subprocess  # noqa: E402  (re-import for readers of this section)

KERNELS = pathlib.Path.home() / "numinous-kernels"


def test_flash_kv4_hooks_compose_with_burst_series_and_tactics(tmp_path):
    """The decode-burst series (glmflash_arms.BURST_PATCHES), the tactics runner hook and glmflash_kv4 all apply to
    one v0.5.20 tree (a future flash_full image) and the touched modules byte-compile."""
    if not (FORK / ".git").exists() or not (KERNELS / ".git").exists():
        pytest.skip("needs ~/sglang-fork and ~/numinous-kernels")
    sys.path.insert(0, str(HERE.parent / "modal"))
    import glmflash_arms as fa

    tar = subprocess.run(["git", "-C", str(FORK), "archive", "v0.5.20", "python/sglang"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=tar.stdout, check=True)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    series = fa.BURST_PATCHES[0]
    for p in series["files"]:
        blob = subprocess.run(["git", "-C", str(KERNELS), "show", f"{series['ref']}:patches/{series['series']}/{p}"],
                              capture_output=True, check=True).stdout
        subprocess.run(["git", "apply"], cwd=tmp_path, input=blob, check=True)
    tree = tmp_path / "python/sglang"
    logs = []
    for name, args in (("sglang_numinous_hooks.py", ["tactics"]), ("glmflash_kv4_hooks.py", [])):
        r = subprocess.run([sys.executable, str(HERE / name), *args, "--root", str(tree)], capture_output=True,
                           text=True, timeout=300)
        assert r.returncode == 0, (name, r.stdout[-800:], r.stderr[-800:])
        logs.append(r.stdout)
    log = "\n".join(logs)
    for hook in ("tactics_runner",) + tuple(f"{h.name}: applied" for h in hooks.HOOKS):
        assert hook in log, (hook, log[-1500:])
    r = subprocess.run([sys.executable, "-m", "py_compile", *[str(tree / p) for p in (
        "srt/layers/attention/dsa_backend.py", "srt/mem_cache/kv_cache_configurator.py",
        "srt/layers/moe/moe_runner/flashinfer_trtllm.py", "srt/layers/attention/dsa/glm_kv4.py",
        "srt/layers/attention/dsa/glmflash_kv4.py", "srt/layers/attention/dsa/glmflash_kv4_engine.py",
        "srt/managers/scheduler.py")]], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
