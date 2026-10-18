"""kv4mla_hooks: anchors on top of glm_kv4, idempotence, clean import + extension build, production call sequence.

CPU (local SGLang fork, tag v0.5.20): the kv4mla hooks apply to the glm_kv4-patched v0.5.20 dsa_backend exactly once,
twice is a no-op, the result compiles, the route sits before kv4glm's engine_forward.
Box (installed sglang + SM100 GPU): a copy of the installed package is patched (+ extension built), imported in a clean
interpreter, and DeepseekSparseAttnBackend._forward_trtllm (real SGLang: FP8 quantize, FP4 KV write door, top-k ->
page table) runs with NUMINOUS_GLM_KV4=1 NUMINOUS_GLM_KV4MLA=1 on kv4glm's test batch: plain prefill takes kv4glm's
sequence path, the forced-fallback prefill and a decode batch take the native kernel, checked against the fp32 reference
over the exactly dequantized FP4 pool; the decode runs under CUDA-graph capture + replay with new table contents.
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

import glm_kv4_hooks as base  # noqa: E402
import kv4mla_hooks as hooks  # noqa: E402
import test_glm_kv4_hooks as tg  # noqa: E402

sglang_spec = importlib.util.find_spec("sglang")
box = pytest.mark.skipif(
    sglang_spec is None or not os.path.exists("/root/work") or not __import__("torch").cuda.is_available(),
    reason="needs the box (installed sglang + GPU)")


def test_hooks_apply_once_on_top_of_glm_kv4_and_compile():
    src = tg._v0520(base.BACKEND)
    src, _ = base.patch_source(src, base.BACKEND)
    new, lines = hooks.patch_source(src)
    assert lines == [f"{h.name}: applied to {h.path}" for h in hooks.HOOKS]
    again, lines2 = hooks.patch_source(new)
    assert again == new and all(ln.endswith("already applied") for ln in lines2)
    compile(new, "dsa_backend.py", "exec")
    assert new.index("numinous-hook:kv4mla_route") < new.index("return _nkv4.engine_forward(")
    assert new.index("numinous-hook:kv4mla_graph") < new.index("_nkv4.reserve_graph_scratch(self, max_num_tokens)")


def test_missing_anchor_fails_loudly():
    with pytest.raises(SystemExit):
        hooks.patch_source("nothing here\n")


def test_modules_exist_and_compile():
    for p in (hooks.KDIR / "kv4mla.py", hooks.KDIR / "kv4mla_engine.py"):
        compile(p.read_text(), str(p), "exec")
    for f in ("glm_fp4_decode.cu", "glm_fp4_decode.cuh", "api.cpp"):
        assert (hooks.KDIR / "csrc" / f).exists()


@pytest.fixture(scope="module")
def patched(tmp_path_factory) -> pathlib.Path:
    src = list(sglang_spec.submodule_search_locations)[0]
    root = tmp_path_factory.mktemp("sgl")
    shutil.copytree(src, root / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
    lines = hooks.apply(str(root / "sglang"))
    assert any(ln.startswith("built ") for ln in lines), lines
    return root


@box
def test_anchors_resolve_on_installed_sglang():
    lines = hooks.apply(list(sglang_spec.submodule_search_locations)[0], check=True)
    assert all(("anchor ok" in ln) or ln.endswith("already applied") for ln in lines), lines


@box
def test_apply_is_idempotent(patched):
    lines = hooks.apply(str(patched / "sglang"), build=False)
    names = [h.name for h in base.HOOKS] + [h.name for h in hooks.HOOKS]
    assert [ln for ln in lines if not ln.startswith("installed")] == [f"{n}: already applied" for n in names]


@box
def test_patched_modules_import_in_clean_interpreter(patched):
    code = ("import sglang, sys, inspect; assert sglang.__file__.startswith(sys.argv[1]), sglang.__file__\n"
            "import sglang.srt.layers.attention.dsa_backend as m\n"
            "src = inspect.getsource(m)\n"
            "assert 'numinous-hook:kv4mla_route' in src and 'numinous-hook:kv4mla_graph' in src\n"
            "from sglang.srt.layers.attention.dsa import kv4mla_engine as e\n"
            "from sglang.srt.layers.attention.dsa.kv4mla import kv4mla as k\n"
            "assert e.native is k and e.enabled()\n"
            "mod = k.load(); assert hasattr(mod, 'sparse_decode')\n"
            "print('ok')")
    env = {**os.environ, "PYTHONPATH": str(patched), "NUMINOUS_GLM_KV4": "1", "NUMINOUS_GLM_KV4MLA": "1"}
    r = subprocess.run([sys.executable, "-c", code, str(patched)], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


SEQUENCE = tg.SEQUENCE[: tg.SEQUENCE.index("# stock: FP8 pool")] + r'''
from sglang.srt.layers.attention.dsa import kv4mla_engine as m
assert m.enabled()
s4 = make_stub(Pool4())
s4._numinous_kv4 = e.attach(s4, types.SimpleNamespace(token_to_kv_pool=s4.token_to_kv_pool))
assert s4._numinous_kv4 is True
kst = e._state(s4)
o_seq = run(s4)                      # plain prefill -> kv4glm's sequence path via the kv4mla route
mst = m._state(s4)
assert mst.calls == {"native": 0, "seq": 1} and kst.calls["seq"] == 1, (mst.calls, kst.calls)
kst.seq_rows_cap = 0                 # the sequence scratch does not fit -> native kernel
o_nat = run(s4)
assert mst.calls == {"native": 1, "seq": 1}, mst.calls
assert kst.calls["sel"] == 0, "the FP8 scratch path must not run"

from sglang.kernels.ops.attention.utils import mla_quantize_for_fp8_no_rope
q8, _, _ = mla_quantize_for_fp8_no_rope(q_nope, q_rope, k.squeeze(1), k_rope.squeeze(1), 512, 64)
from sglang.kernels.ops.attention.dsa.transform_index import transform_index_page_table_prefill
table = transform_index_page_table_prefill(page_table=req_to_token, topk_indices=topk, extend_lens_cpu=extend,
                                           page_size=1, output_num_tokens=T, page_table_is_expanded=False,
                                           cu_seqlens_q=s4.forward_metadata.cu_seqlens_q)
sub = list(range(0, T, 4))
ref4 = kv4.sparse_mla_ref(q8[sub], pool4.view(N, 352), table[sub], layer.scaling)
def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
rows48 = kv4.dequantize_rows_ref(pool4.view(N, 352), torch.float8_e4m3fn).float()
ref48 = torch.zeros(len(sub), H, 512, device=dev)
for ii, tt in enumerate(sub):
    sl = table[tt].long(); v = sl >= 0
    kk = rows48[sl[v]]
    ref48[ii] = torch.softmax(q8[tt].float() @ kk.T * layer.scaling, -1) @ kk[:, :512]
mode = mst.dec.mode
r_nat, r_seq = rel(o_nat[sub, 0], ref4), rel(o_seq[sub, 0], ref4)
r_nat48, r_seq48 = rel(o_nat[sub, 0], ref48), rel(o_seq[sub, 0], ref48)
print(f"mode {mode}: vs fp32 over exact FP4 rows: native {r_nat:.4f}, kv4glm sequence scratch {r_seq:.4f}; "
      f"vs fp32 over E4M3-rounded rows: native {r_nat48:.4f}, scratch {r_seq48:.4f}")
if mode == "exact":
    assert r_nat < 0.006, r_nat
else:
    assert r_nat48 < 0.03 and r_nat < 0.12, (r_nat48, r_nat)
assert torch.isfinite(o_nat.float()).all()

# decode batch under CUDA-graph capture: the graph hook's prepare_graph builds the schedules before capture
bs = 8
assert m.prepare_graph(s4, bs) is True
small = m.MIN_T >= bs   # NUMINOUS_GLM_KV4MLA_MIN_T: this decode batch takes kv4glm's scratch path
assert (kst.graph_scratch is not None) == small, "FP8 graph scratch only for the small batches routed to scratch"
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
n0, s0 = mst.calls["native"], kst.calls["sel"]
o_eager = run_dec()
assert (mst.calls["native"], kst.calls["sel"]) == ((n0, s0 + 1) if small else (n0 + 1, s0)), (mst.calls, kst.calls)
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    run_dec()
    with torch.cuda.graph(g):
        o_graph = run_dec()
torch.cuda.current_stream().wait_stream(s)
g.replay(); torch.cuda.synchronize()
assert rel(o_graph, run_dec()) < (1e-3 if small else 1e-9), "replay on the same inputs must match eager"
for b in range(bs):
    n = min(seqs[b % 2], K)
    dec_topk[b, :n] = torch.randperm(seqs[b % 2], device=dev)[:n].int()   # a different selection
g.replay(); torch.cuda.synchronize()
o_after = run_dec()
print(f"graph replay vs eager on a new selection {rel(o_graph, o_after):.6f}")
assert (rel(o_graph, o_after) < 1e-3 if small else torch.equal(o_graph, o_after)) and torch.isfinite(o_graph.float()).all()
print("ok")
'''


@box
@pytest.mark.parametrize("mode,min_t", [("fp8", "0"), ("exact", "0"), ("fp8", "16")])
def test_production_sequence_native_and_graph(patched, mode, min_t):
    env = {**os.environ, "PYTHONPATH": str(patched), "NUMINOUS_GLM_KV4": "1", "NUMINOUS_GLM_KV4MLA": "1",
           "KV4MLA_MODE": mode, "NUMINOUS_GLM_KV4MLA_MIN_T": min_t}
    r = subprocess.run([sys.executable, "-c", SEQUENCE], env=env, capture_output=True, text=True, timeout=1500)
    print(r.stdout[-3000:])
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-3000:] + r.stderr[-5000:]
    assert r.stdout.count("numinous kv4mla native FP4 sparse MLA active") == 1
