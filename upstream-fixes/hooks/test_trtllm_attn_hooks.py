"""trtllm_attn_hooks: anchors, idempotence, snippet semantics, and kernel parity of the DSpark wide-table path.

CPU (needs the local SGLang fork with tag v0.5.20): the hooks apply to v0.5.20's deepseek_v4_backend.py and
deepseek_v4_trtllm_backend.py exactly once, twice is a no-op, both results compile, and the inserted blocks sit
before the shape asserts they relax. The tables and in-graph snippets are executed with stub locals on CPU
tensors: wide DSpark tables get 64-row tile-aligned parents, per-row lens clamp(count, min=128), live counts
preserved; narrow standard tables are untouched by the new branches.
Box (installed sglang + one GPU): a copy of the installed package is patched and imported in a clean
interpreter; the patched DSV4AttnMetadata.init_trtllm_sparse_buffers builds the wide-table metadata, and the
patched DeepseekV4TrtllmAttnBackend._forward_trtllm_decode runs the real trtllm-gen kernel on a stub backend
over a uniform-FP8 pool; the output must match an fp32 reference over exactly the live entries per row
(counts straddling column 128 included) within FP8 tolerance.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import trtllm_attn_hooks as hooks  # noqa: E402

FORK = pathlib.Path("/home/ubuntu/sglang-fork")
sglang_spec = importlib.util.find_spec("sglang")


def _v0520_source(path: str) -> str:
    if not (FORK / ".git").exists():
        pytest.skip("local SGLang fork not present")
    r = subprocess.run(["git", "-C", str(FORK), "show", f"v0.5.20:python/sglang/{path}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        pytest.skip("tag v0.5.20 not available")
    return r.stdout


# ------------------------------------------------------------------------------------------------------ CPU tests

def test_hooks_apply_once_to_v0520_and_compile():
    for path in (hooks.BACKEND, hooks.TRTLLM):
        src = _v0520_source(path)
        new, lines = hooks.patch_source(src, path)
        expect = [h for h in hooks.HOOKS if h.path == path]
        assert lines == [f"{h.name}: applied to {path}" for h in expect]
        compile(new, path, "exec")
        again, lines2 = hooks.patch_source(new, path)
        assert again == new and all(ln.endswith("already applied") for ln in lines2)
    # the tables hook precedes the shape assert it relaxes
    backend = hooks.patch_source(_v0520_source(hooks.BACKEND), hooks.BACKEND)[0]
    body = backend[backend.index("    def init_trtllm_sparse_buffers("):]
    body = body[: body.index("\n    def ", 10)]
    assert body.index("numinous-hook:trtllm_dspark_tables") < body.index(
        "assert self.swa_page_indices.shape == (num_tokens, SWA_WINDOW)")
    # the decode hook precedes the standard-table assert
    trtllm = hooks.patch_source(_v0520_source(hooks.TRTLLM), hooks.TRTLLM)[0]
    body = trtllm[trtllm.index("    def _forward_trtllm_decode("):]
    body = body[: body.index("\n    def ", 10)]
    assert body.index("numinous-hook:trtllm_dspark_decode") < body.index(
        "assert swa_page_indices.shape == (bs, SWA_WINDOW)")


def test_missing_anchor_fails_loudly():
    with pytest.raises(SystemExit, match="anchor found 0 times"):
        hooks.patch_source("class X:\n    pass\n", hooks.BACKEND)


def _dspark_table(torch, counts, width=192, device="cpu"):
    """Compact DSpark-layout table: row i has counts[i] live entries, -1 after."""
    n = len(counts)
    tbl = torch.full((n, width), -1, dtype=torch.int32, device=device)
    for i, c in enumerate(counts):
        tbl[i, :c] = torch.arange(10 * i, 10 * i + c, dtype=torch.int32, device=device)
    return tbl


def _run_tables_snippet(torch, meta_ns, num_tokens):
    code = "def _run(self, num_tokens, torch, SWA_WINDOW):\n" + hooks.TABLES
    g: dict = {}
    exec(code, g)  # noqa: S102 (executing our own hook text under test)
    return g["_run"](meta_ns, num_tokens, torch, 128)


def test_tables_snippet_wide_and_narrow():
    torch = pytest.importorskip("torch")
    counts = [53, 90, 128, 131, 140]
    n = len(counts)
    cnt = torch.tensor(counts, dtype=torch.int32)
    ns = types.SimpleNamespace(
        swa_page_indices=_dspark_table(torch, counts),
        swa_topk_lengths=cnt.clone(),
        seq_lens_casual=torch.tensor([c + 7 for c in counts], dtype=torch.int32),
        c4_sparse_page_indices=None,
        c128_page_indices=None,
        cuda_int32_kwargs={"dtype": torch.int32, "device": "cpu"},
        trtllm_swa_lens=None,
    )
    before = ns.swa_page_indices.clone()
    _run_tables_snippet(torch, ns, n)
    # views over 64-row tile-aligned parents, contents preserved
    assert ns.swa_page_indices.shape == (n, 192) and ns.swa_page_indices._base.shape == (64, 192)
    assert torch.equal(ns.swa_page_indices, before)
    assert torch.equal(ns.swa_page_indices._base[n:], torch.full((64 - n, 192), -1, dtype=torch.int32))
    assert ns.swa_topk_lengths._base.shape == (64,) and torch.equal(ns.swa_topk_lengths, cnt)
    assert ns.seq_lens_casual._base.shape == (64,)
    # lens clamp to at least the fixed 128 SWA slots (kernel contract), per row
    assert ns.trtllm_swa_lens.tolist() == [128, 128, 128, 131, 140]
    assert ns.trtllm_swa_lens._base.shape == (64,)
    # narrow standard table: the new branch must not touch anything
    ns2 = types.SimpleNamespace(
        swa_page_indices=torch.full((5, 128), 3, dtype=torch.int32),
        swa_topk_lengths=torch.full((5,), 128, dtype=torch.int32),
        seq_lens_casual=torch.full((5,), 100, dtype=torch.int32),
        c4_sparse_page_indices=None,
        c128_page_indices=None,
        cuda_int32_kwargs={"dtype": torch.int32, "device": "cpu"},
        trtllm_swa_lens=None,
    )
    _run_tables_snippet(torch, ns2, 5)
    assert ns2.trtllm_swa_lens is None and ns2.swa_page_indices._base is None


def test_ingraph_snippet_semantics():
    torch = pytest.importorskip("torch")
    counts = [131, 53]
    cam = types.SimpleNamespace(cuda_int32_kwargs={"dtype": torch.int32, "device": "cpu"},
                                swa_page_indices=None, swa_topk_lengths=None, trtllm_swa_lens=None)
    g = {
        "torch": torch, "SWA_WINDOW": 128,
        "self": types.SimpleNamespace(trtllm_attn=True),
        "metadata": types.SimpleNamespace(core_attn_metadata=cam),
        "swa_page_indices": _dspark_table(torch, counts),
        "swa_topk_lengths": torch.tensor(counts, dtype=torch.int32),
    }
    exec(textwrap.dedent(hooks.INGRAPH), g)  # noqa: S102
    assert cam.swa_page_indices.shape == (2, 192) and cam.swa_page_indices._base.shape == (64, 192)
    assert torch.equal(cam.swa_page_indices, g["swa_page_indices"])
    assert cam.swa_topk_lengths.tolist() == counts
    assert cam.trtllm_swa_lens.tolist() == [131, 128]
    assert cam.trtllm_swa_lens._base.shape == (64,)
    # flashmla backend (trtllm_attn False): inert
    cam2 = types.SimpleNamespace(cuda_int32_kwargs=cam.cuda_int32_kwargs,
                                 swa_page_indices=None, swa_topk_lengths=None, trtllm_swa_lens=None)
    g2 = {**g, "self": types.SimpleNamespace(trtllm_attn=False),
          "metadata": types.SimpleNamespace(core_attn_metadata=cam2)}
    exec(textwrap.dedent(hooks.INGRAPH), g2)  # noqa: S102
    assert cam2.swa_page_indices is None and cam2.trtllm_swa_lens is None


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
    assert lines == [f"{h.name}: already applied" for h in hooks.HOOKS]


PARITY = r'''
import sys, types
import flashinfer, torch  # flashinfer first: TRITON_PTXAS_BLACKWELL_PATH
import sglang
assert sglang.__file__.startswith(sys.argv[1]), sglang.__file__
from sglang.srt.layers.attention import deepseek_v4_backend as vb
from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as tb
import inspect
assert "numinous-hook:trtllm_dspark_tables" in inspect.getsource(vb)
assert "numinous-hook:trtllm_dspark_ingraph" in inspect.getsource(vb)
assert "numinous-hook:trtllm_dspark_decode" in inspect.getsource(tb)

torch.manual_seed(0)
DEV, HEADS, DIM, PAGE, N = "cuda", 128, 512, 128, 4096
SCALE = DIM ** -0.5
counts = [53, 90, 128, 131, 140]  # straddles column 128 (probe P3/P4)
bs = len(counts)

pool = (torch.randn(N, DIM, device=DEV) * 0.05).to(torch.float8_e4m3fn)
pool_view = pool.view(N // PAGE, 1, PAGE, DIM)
q = (torch.randn(bs, HEADS, DIM, device=DEV) * 0.05).to(torch.float8_e4m3fn)
sink = torch.full((HEADS,), -10000.0, dtype=torch.float32, device=DEV)

rows = [list(range(11 * i, 11 * i + c)) for i, c in enumerate(counts)]
tbl = torch.full((bs, 192), -1, dtype=torch.int32, device=DEV)
for i, locs in enumerate(rows):
    tbl[i, : len(locs)] = torch.tensor(locs, dtype=torch.int32, device=DEV)

# the patched metadata builder handles the wide table (this assert was the boot crash)
meta = vb.DSV4AttnMetadata(
    page_size=256,
    page_table=torch.zeros(bs, 1, dtype=torch.int32, device=DEV),
    raw_out_loc=torch.zeros(bs, dtype=torch.int64, device=DEV),
    cuda_int32_kwargs={"dtype": torch.int32, "device": torch.device(DEV)},
    seq_lens_casual=torch.tensor([c + 7 for c in counts], dtype=torch.int32, device=DEV),
    positions_casual=torch.arange(bs, dtype=torch.int32, device=DEV),
    swa_page_indices=tbl.clone(),
    swa_topk_lengths=torch.tensor(counts, dtype=torch.int32, device=DEV),
    c4_sparse_topk=1024,
)
meta.c4_sparse_page_indices = None
meta.c128_page_indices = None
meta.init_trtllm_sparse_buffers()
assert meta.swa_page_indices.shape == (bs, 192) and meta.swa_page_indices._base.shape[0] == 64
assert meta.trtllm_swa_lens.tolist() == [max(c, 128) for c in counts]
assert meta.swa_topk_lengths.tolist() == counts

# the patched decode forward serves it through the real trtllm-gen kernel
tb._trtllm_semaphore_rows = 1 << 20
ns = types.SimpleNamespace(
    _trtllm_kv_cache_views=lambda layer_id, cr: (pool_view, pool_view),
    _get_trtllm_bmm_scales=lambda layer: (SCALE, 1.0),
    trtllm_workspace_buffer=torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=DEV),
)
layer = types.SimpleNamespace(layer_id=0, k_scale_float=None)
out = tb.DeepseekV4TrtllmAttnBackend._forward_trtllm_decode(
    ns, q=q.to(torch.bfloat16), layer=layer, compress_ratio=0, core_attn_metadata=meta,
    attn_sink=sink, swa_page_indices=meta.swa_page_indices, extra_indices=None, extra_topk_lengths=None,
).to(torch.float32)

flat = pool.to(torch.float32)
qf = q.to(torch.float32)
errs = []
for i, locs in enumerate(rows):
    k = flat[torch.tensor(locs, device=DEV)]
    s = torch.einsum("hd,nd->hn", qf[i], k) * SCALE
    ref = torch.einsum("hn,nd->hd", torch.softmax(s, dim=-1), k)
    errs.append(((out[i] - ref).norm() / ref.norm()).item())
assert all(e < 0.05 for e in errs), errs
print("parity ok", [round(e, 4) for e in errs])
'''


@box
def test_wide_table_kernel_parity(patched):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    r = subprocess.run([sys.executable, "-c", PARITY, str(patched)],
                       env={**os.environ, "PYTHONPATH": str(patched)},
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "parity ok" in r.stdout, (r.stdout[-2000:], r.stderr[-4000:])
