"""k27_kv4_hooks: anchors, idempotence, compile, engine module import shape (CPU, local SGLang fork tag v0.5.20);
on the box (installed sglang + SM100): patch a copy of the installed package, import the patched modules in a clean
interpreter, and exercise the pool class + decode/verify glue on a synthetic backend against dense_reference.

    python3 -m pytest operations/patches/test_k27_kv4_hooks.py -q
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
import k27_kv4_hooks as hooks  # noqa: E402

FORK = pathlib.Path("/home/ubuntu/sglang-fork")
sglang_spec = importlib.util.find_spec("sglang")
box = pytest.mark.skipif(
    sglang_spec is None or not os.path.exists("/root/work") or not __import__("torch").cuda.is_available(),
    reason="needs the box (installed sglang + GPU)")


def _v0520(path: str) -> str:
    if not (FORK / ".git").exists():
        pytest.skip("local SGLang fork not present")
    r = subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{path}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        pytest.skip("tag v0.5.20 not available")
    return r.stdout


@pytest.mark.parametrize("path", hooks.FILES)
def test_hooks_apply_once_and_compile(path):
    src = _v0520(path)
    new, lines = hooks.patch_source(src, path)
    mine = [h for h in hooks.HOOKS if h.path == path]
    assert lines == [f"{h.name}: applied to {h.path}" for h in mine]
    compile(new, path, "exec")
    again, lines2 = hooks.patch_source(new, path)
    assert again == new and all(ln.endswith("already applied") for ln in lines2)
    for h in mine:
        assert h.marker in new
        if h.replace:
            assert src.count(h.anchor) == 1 and h.anchor not in new or h.anchor in h.text


def test_decode_hook_orders_metadata_before_route():
    new, _ = hooks.patch_source(_v0520(hooks.TRT), hooks.TRT)
    i_meta = new.index("# numinous-hook:k27_kv4_decode")
    seg = new[i_meta - 600:i_meta]
    assert "self.init_forward_metadata(forward_batch)" in seg          # metadata resolved before the route
    tail = new[i_meta:i_meta + 700]
    assert "return _nk27.decode(self, query, layer, forward_batch, metadata)" in tail
    assert "k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)" in tail   # FP8 path kept after it


def test_verify_hooks_route_before_kernel_and_reuse_unpad():
    new, _ = hooks.patch_source(_v0520(hooks.TRT), hooks.TRT)
    a, b = new.index("# numinous-hook:k27_kv4_verify_a"), new.index("# numinous-hook:k27_kv4_verify_b")
    assert a < b
    body = new[b:new.index("assert kv_cache.dtype == self.data_type", b)]
    assert "unpad_draft_extend_output" in body and "unpad_zero_uncovered" in body


def test_missing_anchor_fails_loudly():
    with pytest.raises(SystemExit):
        hooks.patch_source("def nothing():\n    pass\n", hooks.TRT)


def test_engine_module_compiles_and_names():
    src = (hooks.REPO / "operations/kernels/k27_kv4_engine.py").read_text()
    compile(src, "k27_kv4_engine.py", "exec")
    for name in ("K27KV4MLATokenToKVPool", "def attach(", "def decode(", "def verify(", "def prepare_graph(",
                 "def enabled_for(", "numinous_kv4 = True"):
        assert name in src
    assert "void dense_decode(" in (hooks.KDIR / "csrc/api.cpp").read_text()
    assert "class DenseDecoder" in (hooks.KDIR / "kv4mla.py").read_text()
    assert "row_limit" in (hooks.KDIR / "csrc/flashmla/kernels/params.h").read_text()


def test_apply_check_on_fork_copy(tmp_path):
    root = tmp_path / "sglang"
    for path in hooks.FILES:
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_v0520(path))
    lines = hooks.apply(str(root), check=True)
    assert all("anchor ok" in ln for ln in lines) and len(lines) == len(hooks.HOOKS)
    lines = hooks.apply(str(root), check=False, build=False)
    assert any("installed" in ln for ln in lines)
    assert (root / hooks.PKG_DIR / "csrc" / "api.cpp").exists()
    assert (root / "srt/layers/attention/dsa/k27_kv4_engine.py").exists()
    for path in hooks.FILES:
        compile((root / path).read_text(), path, "exec")


# ------------------------------------------------------------------------------------------------ box tests
@pytest.fixture(scope="module")
def patched(tmp_path_factory) -> pathlib.Path:
    src = pathlib.Path(list(sglang_spec.submodule_search_locations)[0])
    dst = tmp_path_factory.mktemp("pkg") / "sglang"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    hooks.apply(str(dst), build=True)
    return dst.parent


@box
def test_patched_modules_import_and_pool_roundtrip(patched):
    code = r'''
import os, sys, torch
sys.path.insert(0, %r)
os.environ["NUMINOUS_K27_KV4"] = "1"
import flashinfer  # noqa
import sglang.srt.layers.attention.trtllm_mla_backend
import sglang.srt.layers.attention.tokenspeed_mla_backend
import sglang.srt.mem_cache.kv_cache_configurator as kc
from sglang.srt.layers.attention.dsa import k27_kv4_engine as e, glm_kv4 as kv4
from sglang.srt.layers.attention.dsa.kv4mla import kv4mla
assert "k27_kv4_dim" in open(kc.__file__).read()
from sglang.srt.runtime_context import get_parallel
get_parallel().override_permanently(attn_dcp_size=1, dcp_enabled=False)  # no process groups in this test
# pool: build directly (MLATokenToKVPool signature), write bf16 rows, read them back through both doors
pool = e.K27KV4MLATokenToKVPool(1024, page_size=64, dtype=torch.float8_e4m3fn, kv_lora_rank=512, qk_rope_head_dim=64,
                                layer_num=2, device="cuda", enable_memory_saver=False, start_layer=0, end_layer=2)
assert pool.kv_buffer[0].shape[-1] == 352 and pool.kv_buffer[0].dtype == torch.uint8
class L: layer_id = 1
loc = torch.tensor([5, 70, 700], device="cuda")
nope = torch.randn(3, 1, 512, device="cuda").bfloat16(); rope = torch.randn(3, 1, 64, device="cuda").to(torch.float8_e4m3fn)
pool.set_mla_kv_buffer(L(), loc, nope, rope)
n, r = pool.get_mla_kv_buffer(L(), loc)
assert n.shape == (3, 1, 512) and r.shape == (3, 1, 64)
err = (n.float() - nope.float()).norm() / nope.float().norm()
assert err < 0.12, float(err)
assert torch.equal(r.to(torch.float8_e4m3fn).view(torch.uint8), rope.view(torch.uint8))
# dense decode through the engine on a synthetic backend
class MD: pass
md = MD(); md.seq_lens_k = torch.tensor([701, 71], dtype=torch.int32, device="cuda")
md.block_kv_indices = torch.arange(0, 16, dtype=torch.int32, device="cuda").repeat(2, 1)
class B: pass
b = B(); b.token_to_kv_pool = pool; b.num_q_heads = 8; b.page_size = 64; b.data_type = torch.float8_e4m3fn
b.max_context_len = 1024; b.disable_chunked_prefix_cache = False; b._fused_set_kv_concat_q_fp8 = True; b._fused_set_kv_concat_q = False
class MR: pass
mr = MR(); mr.token_to_kv_pool = pool
class MC: context_len = 1024
mr.model_config = MC()
assert e.attach(b, mr) and b._fused_set_kv_concat_q_fp8 is False
# fill the whole pool with data so every attended row is real
all_loc = torch.arange(1024 + 64, device="cuda")
pool.set_mla_kv_buffer(L(), all_loc, torch.randn(1088, 1, 512, device="cuda").bfloat16(),
                       torch.randn(1088, 1, 64, device="cuda").to(torch.float8_e4m3fn))
class Layer: layer_id = 1; scaling = 576 ** -0.5
q = (torch.randn(2, 1, 8, 576, device="cuda") * 1.2).to(torch.float8_e4m3fn)
out = e.decode(b, q, Layer(), None, md)
assert out.shape == (2, 8 * 512)
tile = torch.zeros(2, 64, 576, dtype=torch.float8_e4m3fn, device="cuda"); tile[:, :8] = q[:, 0]
ref = kv4mla.dense_reference(tile, kv4.pool_rows(pool.kv_buffer[1]), md.block_kv_indices, md.seq_lens_k, 64, None, Layer.scaling)
rel = (out.float().view(2, 8, 512) - ref[:, :8]).norm() / ref[:, :8].norm()
assert rel < 0.08, float(rel)
# verify fold: 4 tokens x 8 heads
qv = (torch.randn(2, 4, 8, 576, device="cuda") * 1.2).to(torch.float8_e4m3fn)
md.seq_lens_k = torch.tensor([704, 74], dtype=torch.int32, device="cuda")
ov = e.verify(b, qv, Layer(), None, md, 4, None)
assert ov.shape == (2, 4, 8, 512)
lim = torch.zeros(2, 64, dtype=torch.int32, device="cuda")
for t in range(4): lim[:, t*8:(t+1)*8] = (md.seq_lens_k - 4 + t + 1)[:, None]
tile = torch.zeros(2, 64, 576, dtype=torch.float8_e4m3fn, device="cuda"); tile[:, :32] = qv.reshape(2, 32, 576)
ref = kv4mla.dense_reference(tile, kv4.pool_rows(pool.kv_buffer[1]), md.block_kv_indices, md.seq_lens_k, 64, lim, Layer.scaling)
rel = (ov.float().reshape(2, 32, 512) - ref[:, :32]).norm() / ref[:, :32].norm()
assert rel < 0.08, float(rel)
print("OK", float(err), float(rel))
''' % str(patched)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-6000:]
    assert "OK" in r.stdout
