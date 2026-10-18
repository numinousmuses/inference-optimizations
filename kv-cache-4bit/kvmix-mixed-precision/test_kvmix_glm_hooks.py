"""kvmix_glm_hooks: anchors, idempotence, clean import, the real mixed pool, promotion on the real pool, and the
production call sequence (DeepseekSparseAttnBackend._forward_trtllm) over mixed FP8 / FP4 requests.

CPU (needs the local SGLang fork with tag v0.5.20): every hook applies to v0.5.20 exactly once, twice is a no-op,
all four files compile, the hooks sit where the design says (before the prefix rows / last_loc are read, before
the TRT-LLM call, in place of the stock write).
Box (installed sglang + GPU): a copy of the installed package is patched and imported in a clean interpreter; the
real KVMixDSATokenToKVPool + KVMixAllocator are built (two storages, one FP8 tensor with per-layer views and the
scratch tail, indexer cache, page 64); write doors, get_mla_kv_buffer, move, promotion (rows + indexer pages) are
checked bit for bit; then _forward_trtllm runs on a stub backend with an FP8-only batch (stock path), a mixed
prefill batch (sequence path, and the gather path forced), a promoted request, and a mixed decode batch under
CUDA-graph capture + replay; every output is checked against an fp32 reference over the rows the op should see.
A separate box test drives the TRT-LLM op through a > 4 GiB FP8 tensor to confirm 64-bit addressing (the design
hands the op the tail of the multi-layer tensor).
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

import kvmix_glm_hooks as hooks  # noqa: E402

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


def _body(src: str, head: str) -> str:
    b = src[src.index(head):]
    end = b.find("\n    def ", 10) if head.startswith("    def") else b.find("\ndef ", 10)
    return b[:end] if end > 0 else b


@pytest.mark.parametrize("path", hooks.FILES)
def test_hooks_apply_once_and_compile(path):
    src = _v0520(path)
    new, lines = hooks.patch_source(src, path)
    assert lines == [f"{h.name}: applied to {path}" for h in hooks.HOOKS if h.path == path]
    compile(new, path, "exec")
    again, lines2 = hooks.patch_source(new, path)
    assert again == new and all(ln.endswith("already applied") for ln in lines2)


def test_backend_hook_placement():
    new, _ = hooks.patch_source(_v0520(hooks.BACKEND), hooks.BACKEND)
    body = _body(new, "    def _forward_trtllm(")
    assert body.count("self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)") == 1  # else branch only
    assert body.index("numinous-hook:kvmix_stash") < body.index("if self.kv_cache_dtype == torch.float8_e4m3fn:")
    assert body.index("numinous-hook:kvmix_write") < body.index("numinous-hook:kvmix_route")
    assert body.index("_pad_trtllm_sparse_page_table(") < body.index("numinous-hook:kvmix_route")
    assert body.index("numinous-hook:kvmix_route") < body.index("trtllm_batch_decode_with_kv_cache_mla")
    init = _body(new, "    def init_cuda_graph_state(")
    assert "numinous-hook:kvmix_graph" in init and "max_num_tokens" in init


def test_configurator_hook_placement():
    new, _ = hooks.patch_source(_v0520(hooks.CONFIGURATOR), hooks.CONFIGURATOR)
    fn = _body(new, "def calculate_mla_kv_cache_dim(")
    i = fn.index("numinous-hook:kvmix_dim")
    assert fn.index("if uses_trtllm_kv_layout:") < i < fn.index("return kv_cache_dim", i)
    build = _body(new, "    def _build_dsa_kv_pool(")
    assert build.index("PoolCls = DSATokenToKVPool") < build.index("numinous-hook:kvmix_pool") < build.index(
        "token_to_kv_pool = PoolCls(")
    alloc = _body(new, "    def _build_token_to_kv_pool_allocator(")
    assert alloc.index("token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(") < alloc.index(
        "numinous-hook:kvmix_alloc") < alloc.index("if get_memory().enable_hisparse and is_dsv4_model:")
    assert alloc.index("numinous-hook:kvmix_alloc") < alloc.index("assert self.is_draft_worker")


def test_allocation_and_common_hook_placement():
    new, _ = hooks.patch_source(_v0520(hooks.ALLOCATION), hooks.ALLOCATION)
    ext = _body(new, "def alloc_for_extend(")
    assert ext.index("numinous-hook:kvmix_extend") < ext.index("prefix_tensors = [r.prefix_indices") < ext.index(
        "alloc_req_slots(")
    dec = _body(new, "def alloc_for_decode(")
    assert dec.index("numinous-hook:kvmix_decode") < dec.index("last_loc = batch.req_to_token_pool.req_to_token[")
    spec = _body(new, "def alloc_for_spec_decode(")
    assert spec.index("numinous-hook:kvmix_spec") < spec.index("last_loc = get_last_loc(") < spec.index(
        "ALLOC_EXTEND_FUNCS")
    common, _ = hooks.patch_source(_v0520(hooks.COMMON), hooks.COMMON)
    ev = _body(common, "def evict_from_tree_cache(")
    assert ev.index("allocator = tree_cache.token_to_kv_pool_allocator") < ev.index("numinous-hook:kvmix_evict") < \
        ev.index("isinstance(allocator, SWATokenToKVPoolAllocator)")


def test_missing_anchor_fails_loudly():
    with pytest.raises(SystemExit, match="anchor found 0 times"):
        hooks.patch_source("class X:\n    pass\n", hooks.BACKEND)


def test_exclusive_with_glm_kv4_hooks():
    """Applying kvmix on top of glm_kv4 (all-FP4 pool) must fail at the write anchor: the two pools are exclusive."""
    import glm_kv4_hooks as kv4h

    src = _v0520(hooks.BACKEND)
    kv4_patched, _ = kv4h.patch_source(src, kv4h.BACKEND)
    with pytest.raises(SystemExit, match="kvmix_write: anchor found 0 times"):
        hooks.patch_source(kv4_patched, hooks.BACKEND)


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


def _run(script: str, patched: pathlib.Path, timeout: int = 1200, **env):
    e = {**os.environ, "PYTHONPATH": str(patched), "NUMINOUS_KVMIX": "1", **env}
    r = subprocess.run([sys.executable, "-c", script], env=e, capture_output=True, text=True, timeout=timeout)
    print(r.stdout[-4000:])
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-3000:] + r.stderr[-6000:]
    return r.stdout


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
            "import sglang.srt.mem_cache.allocation as a, sglang.srt.mem_cache.common as cm\n"
            "src = inspect.getsource(m) + inspect.getsource(c) + inspect.getsource(a) + inspect.getsource(cm)\n"
            f"assert all(m in src for m in {[h.marker for h in hooks.HOOKS]!r})\n"
            "from sglang.srt.layers.attention.dsa import kvmix_glm_engine as e, kvmix_glm as mx, glm_kv4 as k\n"
            "from sglang.srt.mem_cache import kvmix_core as core\n"
            "assert e.mix is mx and mx.kv4 is k and e.core is core and e.ROW_BYTES == 352\n"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code, str(patched)], env={**os.environ, "PYTHONPATH": str(patched)},
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


POOL = r'''
import logging, sys, types
import flashinfer, torch
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(name)s %(message)s")
from sglang.srt.layers.attention.dsa import kvmix_glm_engine as e, kvmix_glm as mix, glm_kv4 as kv4
from sglang.srt.mem_cache import kvmix_core as core
hf = types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
assert e.enabled_for(hf, torch.float8_e4m3fn) is True
assert e.enabled_for(types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"]), torch.float8_e4m3fn) is False
try:
    e.enabled_for(hf, torch.bfloat16); raise SystemExit("bf16 must be refused")
except RuntimeError as ex:
    assert "fp8_e4m3" in str(ex)
dev = "cuda"
size, page, L = 64 * 100, 64, 2
from sglang.srt.runtime_context import get_parallel
get_parallel().override_permanently(attn_dcp_size=1, dcp_enabled=False)
blended = e.blended_dim()
assert blended == 437, blended
pool = e.KVMixDSATokenToKVPool(size, page_size=page, kv_lora_rank=512, dtype=torch.float8_e4m3fn,
                               qk_rope_head_dim=64, layer_num=L, device=dev, index_head_dim=128,
                               enable_memory_saver=False, kv_cache_dim=blended)
lay = pool.lay
assert pool.n8 + pool.n4 == size and pool.n8 % page == 0 and pool.n4 % page == 0
assert pool.n8 * 576 + pool.n4 * 352 <= size * blended, (pool.n8, pool.n4)
assert pool.kv_cache_dim == 576 and pool.dsa_kv_cache_store_fp8 is False and pool.numinous_kvmix
assert len(pool.kv_buffer) == L and tuple(pool.kv_buffer[0].shape) == (pool.n8 + page, 1, 576)
assert tuple(pool.fp4_buffer[1].shape) == (pool.n4 + page, 1, 352)
assert pool.kv_buffer[1].data_ptr() == pool.fp8_all.data_ptr() + (pool.n8 + page) * 576  # views into one tensor
assert pool.scratch.data_ptr() == pool.fp8_all.data_ptr() + L * (pool.n8 + page) * 576
assert pool.layer_view(1).shape[0] * 64 == (pool.n8 + page) + pool.scratch_rows and pool.scratch_base(1) == pool.n8 + page
assert pool.get_key_buffer(1).dtype == torch.float8_e4m3fn and pool.get_key_buffer(1).shape[-1] == 576
kv_bytes = pool.get_kv_size_bytes()
assert kv_bytes == pool.fp8_all.numel() + pool.fp4_all.numel() + sum(b.numel() for b in pool.index_key_cache.buffer), kv_bytes
alloc = e.build_allocator(pool, page, torch.float8_e4m3fn, dev, False)
assert (alloc.n8, alloc.n4) == (pool.n8, pool.n4) and alloc.boundary == lay.boundary

layer = types.SimpleNamespace(layer_id=1)
T = 300
loc = torch.cat([torch.randperm(pool.n8, device=dev)[:150] + page,
                 torch.randperm(pool.n4, device=dev)[:150] + lay.boundary])
k = torch.randn(T, 1, 512, device=dev).bfloat16()
k_rope = torch.randn(T, 1, 64, device=dev).bfloat16()
pool.set_mla_kv_buffer(layer, loc, k, k_rope)  # the generic door: bf16 in
f8 = pool.kv_buffer[1][loc[:150], 0]
assert torch.equal(f8[:, :512], k[:150, 0].to(torch.float8_e4m3fn).view(torch.uint8))
assert torch.equal(f8[:, 512:], k_rope[:150, 0].to(torch.float8_e4m3fn).view(torch.uint8))
ref4 = kv4.quantize_rows_ref(k[150:, 0], k_rope[150:, 0].to(torch.float8_e4m3fn))
assert torch.equal(pool.fp4_buffer[1][loc[150:] - pool.n8, 0], ref4)
assert pool.kv_buffer[0].abs().sum().item() == 0 and pool.fp4_buffer[0].abs().sum().item() == 0
# the trtllm door: fp8 rows + the bf16 latent
k8 = k[:, 0].to(torch.float8_e4m3fn); r8 = k_rope[:, 0].to(torch.float8_e4m3fn)
pool.write_mixed(types.SimpleNamespace(layer_id=0), loc, k8, r8, k[:, 0])
assert torch.equal(pool.kv_buffer[0][loc[:150], 0], f8) and torch.equal(pool.fp4_buffer[0][loc[150:] - pool.n8, 0], ref4)
n_bf16, r_bf16 = pool.get_mla_kv_buffer(layer, loc)
want = mix.gather_bf16_ref(lay, pool.kv_buffer[1].cpu(), pool.fp4_buffer[1].cpu(), loc.cpu())
assert torch.equal(n_bf16.view(T, 512).cpu(), want[:, :512]) and torch.equal(r_bf16.view(T, 64).cpu(), want[:, 512:])
# move: FP8 -> FP4 and FP4 -> FP8 rows, indexer tokens with them
for l in range(L):
    pool.index_key_cache.buffer[l].copy_(torch.randint(0, 255, pool.index_key_cache.buffer[l].shape, dtype=torch.uint8, device=dev))
idx_before = [b.clone() for b in pool.index_key_cache.buffer]
# two moves (a swap in one launch would race, as the stock kv[tgt] = kv[src] does): FP8 -> FP4, then FP4 -> FP8
d = kv4.dequantize_rows_ref(ref4[:10].cpu(), torch.float8_e4m3fn).view(torch.uint8)  # the FP4 rows before the move
pool.move_kv_cache(loc[150:160], loc[:10])
assert torch.equal(pool.fp4_buffer[1][loc[150:160] - pool.n8, 0], kv4.quantize_rows_ref(k8[:10], r8[:10]))
pool.move_kv_cache(loc[:10], loc[160:170])
d = kv4.dequantize_rows_ref(ref4[10:20].cpu(), torch.float8_e4m3fn).view(torch.uint8)
assert torch.equal(pool.kv_buffer[1][loc[:10], 0].cpu(), d)
tgt = torch.cat([loc[150:160], loc[:10]]); src = torch.cat([loc[:10], loc[160:170]])
for l in range(L):
    buf, ref = pool.index_key_cache.buffer[l], idx_before[l]
    for t_, s_ in zip(tgt.tolist(), src.tolist()):
        assert torch.equal(buf[t_ // 64, (t_ % 64) * 128:(t_ % 64 + 1) * 128], ref[s_ // 64, (s_ % 64) * 128:(s_ % 64 + 1) * 128])
        assert torch.equal(buf[t_ // 64, 64 * 128 + (t_ % 64) * 4: 64 * 128 + (t_ % 64 + 1) * 4], ref[s_ // 64, 64 * 128 + (s_ % 64) * 4: 64 * 128 + (s_ % 64 + 1) * 4])
# promotion: two FP8 pages -> two FP4 pages, all layers + indexer pages
src_pages = torch.tensor([3, 7], device=dev); dst_pages = torch.tensor([lay.fp8_pages + 2, lay.fp8_pages + 9], device=dev)
src_s = (src_pages[:, None] * 64 + torch.arange(64, device=dev)).reshape(-1)
dst_s = (dst_pages[:, None] * 64 + torch.arange(64, device=dev)).reshape(-1)
before8 = pool.fp8_all.clone(); before4 = pool.fp4_all.clone(); idx_before = [b.clone() for b in pool.index_key_cache.buffer]
pool.kvmix_promote_pages(src_s, dst_s)
mix.promote_ref(lay, before8, before4, L, src_s.cpu(), dst_s.cpu())
assert torch.equal(pool.fp8_all, before8) and torch.equal(pool.fp4_all, before4)
for l in range(L):
    assert torch.equal(pool.index_key_cache.buffer[l][dst_pages], idx_before[l][src_pages])
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
    out = _run(POOL, patched)
    assert "numinous kvmix pool: 2 layers" in out and "numinous kvmix allocator" in out


SEQUENCE = r'''
import logging, sys, types
import flashinfer, torch
from sglang.srt.layers.attention import dsa_backend as db
from sglang.srt.layers.attention.dsa import kvmix_glm_engine as e, kvmix_glm as mix, glm_kv4 as kv4
from sglang.srt.mem_cache import kvmix_core as core
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.layers.attention.trtllm_mla_backend import make_persistent_multi_ctas_kv_counter_buffer
from sglang.srt.runtime_context import get_parallel
get_parallel().override_permanently(attn_dcp_size=1, dcp_enabled=False)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(name)s %(message)s")
torch.manual_seed(0)
dev, H, K, PAGE = "cuda", 64, 2048, 64
size = 64 * 400
pool = e.KVMixDSATokenToKVPool(size, page_size=PAGE, kv_lora_rank=512, dtype=torch.float8_e4m3fn, qk_rope_head_dim=64,
                               layer_num=1, device=dev, index_head_dim=128, enable_memory_saver=False,
                               kv_cache_dim=e.blended_dim())
lay = pool.lay
alloc = e.build_allocator(pool, PAGE, torch.float8_e4m3fn, dev, False)
# three prefill requests: FP8, FP4, and mixed (FP8 prefix pages + FP4 extension); one more FP8 for the stock path
prefix, extend = [1500, 3000, 2100, 700], [200, 312, 250, 100]
flags = [False, True, True, False]
seqs = [p + x for p, x in zip(prefix, extend)]
bs, T, max_seq = len(seqs), sum(extend), max(seqs)
req_to_token = torch.zeros(bs, max_seq, dtype=torch.int32, device=dev)
# prefix pages: request 2's prefix is FP8 (a cached, shared prefix), request 1's is FP4 (a long request)
pre_flags = [False, True, False, False]
alloc.pending = list(pre_flags)
z = torch.zeros(bs, dtype=torch.int64, device=dev)
pre_t = torch.tensor(prefix, dtype=torch.int64, device=dev)
pre = alloc.alloc_extend(z, z.cpu(), pre_t, pre_t.cpu(), torch.full((bs,), -1, dtype=torch.int64, device=dev), sum(prefix))
p0 = 0
for r in range(bs):
    req_to_token[r, :prefix[r]] = pre[p0:p0 + prefix[r]].int(); p0 += prefix[r]
last = torch.stack([req_to_token[r, prefix[r] - 1] for r in range(bs)]).long()
alloc.pending = list(flags)
seq_t = torch.tensor(seqs, dtype=torch.int64, device=dev)
out_cache_loc = alloc.alloc_extend(pre_t, pre_t.cpu(), seq_t, seq_t.cpu(), last, T)
p0 = 0
for r in range(bs):
    req_to_token[r, prefix[r]:seqs[r]] = out_cache_loc[p0:p0 + extend[r]].int(); p0 += extend[r]
for r in range(bs):
    row = req_to_token[r, :seqs[r]]
    has4 = bool((row >= lay.boundary).any()); all4 = bool((row >= lay.boundary).all())
    assert has4 == (flags[r] or pre_flags[r]) and all4 == (flags[r] and pre_flags[r]), (r, has4, all4)
# fill every prefix row through the generic door (bf16 in -> FP8 or FP4 by slot)
layer = types.SimpleNamespace(tp_q_head_num=H, head_dim=576, v_head_dim=512, k_scale_float=None,
                              scaling=576 ** -0.5, layer_id=0, is_cross_attention=False)
pre_nope = torch.randn(sum(prefix), 1, 512, device=dev).bfloat16() * 1.5
pre_rope = torch.randn(sum(prefix), 1, 64, device=dev).bfloat16() * 1.5
pool.set_mla_kv_buffer(layer, pre, pre_nope, pre_rope)
positions = torch.cat([torch.arange(p, p + x, device=dev) for p, x in zip(prefix, extend)])
topk = torch.full((T, K), -1, dtype=torch.int32, device=dev)
for t in range(T):
    n = min(int(positions[t]) + 1, K)
    topk[t, :n] = torch.randperm(int(positions[t]) + 1, device=dev)[:n].int()
dsa_seq = torch.clamp(positions + 1, max=K).int()
cu_q = [0]
for x in extend: cu_q.append(cu_q[-1] + x)

def make_stub(rows, ext, cache_lens, max_len, cu):
    stub = types.SimpleNamespace(
        forward_metadata=types.SimpleNamespace(
            page_table_1=rows, dsa_extend_seq_lens_list=ext,
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=dev),
            cache_seqlens_int32=torch.tensor(cache_lens, dtype=torch.int32, device=dev), max_seq_len_k=max_len),
        kv_cache_dtype=torch.float8_e4m3fn, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=192,
        token_to_kv_pool=pool, real_page_size=PAGE, kv_cache_dim=576, use_fused_topk=False, device=dev,
        num_q_heads=H, workspace_buffer=torch.empty(512 << 20, dtype=torch.uint8, device=dev),
        _multi_ctas_kv_counter_buffer=make_persistent_multi_ctas_kv_counter_buffer(torch.device(dev), H, 8192),
        dsa_prefill_impl="trtllm", dsa_decode_impl="trtllm", device_sm_major=torch.cuda.get_device_capability()[0],
        dsa_index_topk=K, use_mha=False,
    )
    stub._pad_trtllm_sparse_page_table = types.MethodType(db.DeepseekSparseAttnBackend._pad_trtllm_sparse_page_table, stub)
    stub._pad_topk_indices = types.MethodType(db.DeepseekSparseAttnBackend._pad_topk_indices, stub)
    return stub

stub = make_stub(req_to_token, extend, seqs, max_seq, cu_q)
mr = types.SimpleNamespace(token_to_kv_pool=pool, token_to_kv_pool_allocator=alloc)
stub._numinous_kvmix = e.attach(stub, mr)
assert stub._numinous_kvmix is True
st = e._state(stub)
fb = types.SimpleNamespace(positions=positions, out_cache_loc=out_cache_loc, forward_mode=ForwardMode.EXTEND,
                           seq_lens_sum=sum(seqs), attn_cp_metadata=None, batch_size=bs,
                           seq_lens_cpu=torch.tensor(seqs, dtype=torch.int32),
                           extend_seq_lens_cpu=torch.tensor(extend, dtype=torch.int32))
q_nope = torch.randn(T, H, 512, device=dev).bfloat16()
q_rope = torch.randn(T, H, 64, device=dev).bfloat16()
k = torch.randn(T, 1, 512, device=dev).bfloat16() * 1.5
k_rope = torch.randn(T, 1, 64, device=dev).bfloat16() * 1.5

def run(s, fb_, q_n, q_r, kk, kr, tk, seq, is_prefill=True):
    return db.DeepseekSparseAttnBackend._forward_trtllm(
        s, q_n, kk, None, layer, fb_, seq, True, q_r, kr, tk, None, False, None, is_prefill=is_prefill)

o_seq = run(stub, fb, q_nope, q_rope, k, k_rope, topk, dsa_seq)
assert st.calls == {"stock": 0, "gather": 0, "seq": 1}, st.calls
assert st.batch_fp4_reqs == [False, True, True, False], st.batch_fp4_reqs
# the write door: FP8 slots hold the fp8 rows, FP4 slots a quantization of the bf16 latent
from sglang.kernels.ops.attention.utils import mla_quantize_for_fp8_no_rope
q8, k8, r8 = mla_quantize_for_fp8_no_rope(q_nope, q_rope, k.squeeze(1), k_rope.squeeze(1), 512, 64)
is4 = out_cache_loc >= lay.boundary
assert torch.equal(pool.kv_buffer[0][out_cache_loc[~is4], 0, :512], k8[~is4].view(torch.uint8))
assert torch.equal(pool.kv_buffer[0][out_cache_loc[~is4], 0, 512:], r8[~is4].view(torch.uint8))
assert torch.equal(pool.fp4_buffer[0][out_cache_loc[is4] - pool.n8, 0], kv4.quantize_rows_ref(k.view(T, 512)[is4], r8[is4]))

# fp32 reference over the rows the op should see: FP8 rows as stored, FP4 rows dequantized then E4M3-rounded (the scratch)
rows_f32 = torch.zeros(lay.n_slots, 576, device=dev)
rows_f32[:lay.boundary] = pool.kv_buffer[0][:, 0].view(torch.float8_e4m3fn).float()
rows_f32[lay.boundary:] = kv4.dequantize_rows_ref(pool.fp4_buffer[0][PAGE:, 0], torch.float8_e4m3fn).float()
from sglang.kernels.ops.attention.dsa.transform_index import transform_index_page_table_prefill
table = transform_index_page_table_prefill(page_table=req_to_token, topk_indices=topk, extend_lens_cpu=extend,
                                           page_size=1, output_num_tokens=T, page_table_is_expanded=False,
                                           cu_seqlens_q=stub.forward_metadata.cu_seqlens_q)
def ref_rows(idx, tab, qq):
    out = torch.zeros(len(idx), H, 512, device=dev)
    for i, t in enumerate(idx):
        sl = tab[t].long(); v = sl >= 0
        kk = rows_f32[sl[v]]
        p = torch.softmax(qq[t].float() @ kk.T * layer.scaling, -1)
        out[i] = p @ kk[:, :512]
    return out
def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
sub = list(range(0, T, 5))
ref = ref_rows(sub, table, q8)
r_seq = rel(o_seq[sub, 0], ref)
# gather path forced (no extend_seq_lens_cpu -> the sequence path declines), same rows -> same result
fb2 = types.SimpleNamespace(**{**vars(fb), "extend_seq_lens_cpu": None})
o_gather = run(stub, fb2, q_nope, q_rope, k, k_rope, topk, dsa_seq)
assert st.calls["gather"] == 1, st.calls
r_gather = rel(o_gather[sub, 0], ref)
# a request longer than the scratch: the sequence path declines, the gather path serves the batch (same rows)
saved_rows = pool.scratch_rows
pool.scratch_rows = 2048  # < request 1's 3312 tokens; gather chunks of 2048 // 2048 = 1 token
o_small = run(stub, fb, q_nope, q_rope, k, k_rope, topk, dsa_seq)
assert st.calls["gather"] == 2 and st.calls["seq"] == 1, st.calls
r_small = rel(o_small[sub, 0], ref)  # one-token TRT-LLM calls schedule differently: same rows, kernel-level tolerance
assert r_small < 0.03, r_small
pool.scratch_rows = saved_rows
# stock path: an FP8-only batch (request 3 alone) takes the stock call on layer 0's view
r3 = 3
rows3 = req_to_token[r3:r3 + 1]
stub3 = make_stub(rows3, extend[r3:], seqs[r3:], seqs[r3], [0, extend[r3]])
stub3._numinous_kvmix = e.attach(stub3, mr); st3 = e._state(stub3)
t0, t1 = cu_q[r3], cu_q[r3 + 1]
fb3 = types.SimpleNamespace(**{**vars(fb), "positions": positions[t0:t1], "out_cache_loc": out_cache_loc[t0:t1],
                               "batch_size": 1, "seq_lens_cpu": torch.tensor(seqs[r3:], dtype=torch.int32),
                               "extend_seq_lens_cpu": torch.tensor(extend[r3:], dtype=torch.int32)})
o_stock = run(stub3, fb3, q_nope[t0:t1], q_rope[t0:t1], k[t0:t1], k_rope[t0:t1], topk[t0:t1], dsa_seq[t0:t1])
assert st3.calls == {"stock": 1, "gather": 0, "seq": 0}, st3.calls
r_stock = rel(o_stock[:, 0], o_seq[t0:t1, 0])
print(f"prefill kernel-vs-fp32 (max err / max |ref|): seq path {r_seq:.4f}, gather path {r_gather:.4f}; "
      f"FP8-only request via the stock call vs the mixed batch {r_stock:.5f}")
assert r_seq < 0.03 and r_gather < 0.03 and r_stock < 0.02, (r_seq, r_gather, r_stock)

# promotion of request 0 (all FP8) on the real pool: pages move to FP4, the same query now attends FP4 rows
kv = types.SimpleNamespace(req_pool_idx=0, cache_protected_len=0, kv_committed_len=seqs[0], kv_allocated_len=seqs[0], holds_kv=True)
class R2T:
    req_to_token = req_to_token
    def write(self, idx, vals): req_to_token[idx] = vals
req0 = types.SimpleNamespace(rid="r0", origin_input_ids=[0] * seqs[0], output_ids=[], prefix_indices=None, last_node=None, kv=kv)
class ChunkLike:
    def is_chunk_cache(self): return True
before_fp8 = alloc.fp8.available_size(); before_fp4 = alloc.fp4.available_size()
assert core.promote_request(alloc, ChunkLike(), req0, R2T(), "decode")
n_pages0 = -(-seqs[0] // PAGE)
assert alloc.fp8.available_size() == before_fp8 + n_pages0 * PAGE and alloc.fp4.available_size() == before_fp4 - n_pages0 * PAGE
row0 = req_to_token[0, :seqs[0]]
assert bool((row0 >= lay.boundary).all())
rows_f32[lay.boundary:] = kv4.dequantize_rows_ref(pool.fp4_buffer[0][PAGE:, 0], torch.float8_e4m3fn).float()
tab0 = transform_index_page_table_prefill(page_table=req_to_token, topk_indices=topk, extend_lens_cpu=extend, page_size=1,
                                          output_num_tokens=T, page_table_is_expanded=False, cu_seqlens_q=stub.forward_metadata.cu_seqlens_q)
fb4 = types.SimpleNamespace(**vars(fb))
o_prom = run(stub, fb4, q_nope, q_rope, k, k_rope, topk, dsa_seq)
sub0 = list(range(0, extend[0], 4))
ref0 = ref_rows(sub0, tab0, q8)
r_prom = rel(o_prom[sub0, 0], ref0)
gap0 = rel(o_prom[sub0, 0], o_seq[sub0, 0])
print(f"promoted request: kernel-vs-fp32 {r_prom:.4f}; FP4-vs-FP8 output gap for the same rows {gap0:.4f}")
assert r_prom < 0.03 and 0 < gap0 < 0.3

# decode under CUDA-graph capture over a mixed batch (rows of requests 0..3), replay after a table permutation.
# A FRESH backend state: its first mixed-path call happens inside the capture, as in the engine (the capture
# warm-up batches hold no FP4 slot and take the stock call) -> reserve_graph_scratch must have warmed the kernels
bs_d = 8
stub = make_stub(req_to_token, extend, seqs, max_seq, cu_q)
stub._numinous_kvmix = e.attach(stub, mr); st = e._state(stub)
kv4._LUTS.clear()
e.reserve_graph_scratch(stub, bs_d)
dec_topk = torch.full((bs_d, K), -1, dtype=torch.int32, device=dev)
for b in range(bs_d):
    s = seqs[b % bs]; n = min(s, K); dec_topk[b, :n] = torch.randperm(s, device=dev)[:n].int()
dec_pt = req_to_token[[b % bs for b in range(bs_d)]]
stub.forward_metadata = types.SimpleNamespace(page_table_1=dec_pt, cache_seqlens_int32=torch.tensor(
    [seqs[b % bs] for b in range(bs_d)], dtype=torch.int32, device=dev), max_seq_len_k=max_seq)
dq_nope = torch.randn(bs_d, H, 512, device=dev).bfloat16(); dq_rope = torch.randn(bs_d, H, 64, device=dev).bfloat16()
dk = torch.randn(bs_d, 1, 512, device=dev).bfloat16(); dk_rope = torch.randn(bs_d, 1, 64, device=dev).bfloat16()
alloc.pending = [b % 2 == 0 for b in range(bs_d)]
dloc = alloc.alloc_extend(torch.zeros(bs_d, dtype=torch.int64, device=dev), torch.zeros(bs_d, dtype=torch.int64),
                          torch.ones(bs_d, dtype=torch.int64, device=dev), torch.ones(bs_d, dtype=torch.int64),
                          torch.full((bs_d,), -1, dtype=torch.int64, device=dev), bs_d)
dfb = types.SimpleNamespace(positions=torch.tensor([seqs[b % bs] for b in range(bs_d)], device=dev), out_cache_loc=dloc,
                            forward_mode=ForwardMode.DECODE, seq_lens_sum=0, attn_cp_metadata=None, batch_size=bs_d,
                            seq_lens_cpu=None)
def run_dec():
    return run(stub, dfb, dq_nope, dq_rope, dk, dk_rope, dec_topk, None, is_prefill=False)
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    with torch.cuda.graph(g):
        o_graph = run_dec()
torch.cuda.current_stream().wait_stream(s)
assert st.calls["gather"] == 1, st.calls  # the capture itself was the first mixed call of this state
for b in range(bs_d):
    n = min(seqs[b % bs], K)
    dec_topk[b, :n] = dec_topk[b, :n][torch.randperm(n, device=dev)]
g.replay(); torch.cuda.synchronize()
o_after = run_dec()
print(f"decode graph replay vs eager on the permuted table {rel(o_graph, o_after):.5f}")
assert rel(o_graph, o_after) < 0.02 and torch.isfinite(o_graph.float()).all()
print("ok")
'''


@box
def test_production_sequence_mixed_requests_promotion_and_graph(patched):
    out = _run(SEQUENCE, patched, timeout=1500)
    assert out.count("numinous kvmix attention active") == 1


BIG = r'''
import flashinfer, torch, types
from sglang.srt.layers.attention.trtllm_mla_backend import make_persistent_multi_ctas_kv_counter_buffer
from sglang.srt.environ import envs
dev, H, K, PAGE = "cuda", 64, 2048, 64
rows = 8 * 1024 * 1024  # 8M x 576 B = 4.5 GiB: row offsets past 2^32 bytes
buf = torch.zeros(rows, 576, dtype=torch.uint8, device=dev)
hi = rows - 64 * 64  # rows placed in the last 64 pages (byte offset > 4 GiB)
S = 3000
vals = (torch.randn(S, 576, device=dev) * 1.5).to(torch.float8_e4m3fn)
slots = torch.arange(hi, hi + S, device=dev)
buf[slots] = vals.view(torch.uint8)
T = 4
q = (torch.randn(T, H, 576, device=dev)).to(torch.float8_e4m3fn)
table = torch.full((T, K), -1, dtype=torch.int32, device=dev)
for t in range(T):
    table[t] = slots[torch.randperm(S, device=dev)[:K]].int()
seq_lens = torch.full((T,), S, dtype=torch.int32, device=dev)
out = torch.empty(T, 1, H, 512, dtype=torch.bfloat16, device=dev)
ctr = make_persistent_multi_ctas_kv_counter_buffer(torch.device(dev), H, 8192)
flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
    query=q.view(T, 1, H, 576), kv_cache=buf.view(torch.float8_e4m3fn).view(-1, 1, PAGE, 576),
    workspace_buffer=torch.empty(512 << 20, dtype=torch.uint8, device=dev), qk_nope_head_dim=192, kv_lora_rank=512,
    qk_rope_head_dim=64, block_tables=table.unsqueeze(1), seq_lens=seq_lens, max_seq_len=S, sparse_mla_top_k=K,
    bmm1_scale=576 ** -0.5, backend="trtllm-gen",
    skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
    multi_ctas_kv_counter_buffer=ctr, out=out)
torch.cuda.synchronize()
worst = 0.0
for t in range(T):
    kk = buf[table[t].long()].view(torch.float8_e4m3fn).float()
    p = torch.softmax(q[t].float() @ kk.T * 576 ** -0.5, -1)
    ref = p @ kk[:, :512]
    worst = max(worst, ((out[t, 0].float() - ref).abs().max() / ref.abs().max()).item())
print(f"trtllm sparse MLA over a {buf.numel() / 2**30:.2f} GiB tensor, rows at byte offset {hi * 576 / 2**30:.2f} GiB: max rel err {worst:.4f}")
assert worst < 0.03, worst
print("ok")
'''


@box
def test_trtllm_reads_beyond_4gib():
    """The design hands the TRT-LLM op the tail of one multi-layer tensor (tens of GB): 64-bit row addressing."""
    e = {**os.environ}
    r = subprocess.run([sys.executable, "-c", BIG], env=e, capture_output=True, text=True, timeout=900)
    print(r.stdout[-2000:])
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-2000:] + r.stderr[-4000:]
