"""CPU test: the kpool logits row-chunking hook is EXACT. The patched `_get_topk_ragged_kpool_plan` (SGLang v0.5.20,
patched in memory from the stock file) runs with a torch fp8_mqa_logits and a torch row-wise top-k on a fake plan;
budget unset (stock path) and a small budget (chunked path) must give identical indices.

    python3 -m pytest -q operations/patches/test_kpool_logits_chunk_hooks.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SGLANG_SRC = os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python")
sys.path.insert(0, str(HERE))
import kpool_logits_chunk_hooks as H  # noqa: E402

_CHILD = r'''
import importlib.abc, importlib.machinery, json, os, sys, types
import torch
SRC, HOOKS = sys.argv[1], sys.argv[2]
torch.cuda.is_available = lambda: False
STUB = {"flashinfer", "sgl_kernel", "deep_gemm", "tilelang", "cutlass", "flash_mla", "vllm", "aiter", "mooncake",
        "nixl", "triton_kernels", "flashkda", "helion", "quack", "decord", "humming"}
class _Stub(types.ModuleType):
    __path__ = []
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        m = _Stub(self.__name__ + "." + n); setattr(self, n, m); return m
    def __call__(self, *a, **k): return _Stub(self.__name__ + "()")
    def __iter__(self): return iter(())
    def __mro_entries__(self, bases): return (object,)
class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] not in STUB:
            return None
        return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec): return _Stub(spec.name)
    def exec_module(self, m): pass
sys.meta_path.insert(0, _Finder())
sys.path.insert(0, SRC); sys.path.insert(0, HOOKS)
import kpool_logits_chunk_hooks as H
name = "sglang.srt.layers.attention.dsa.dsa_indexer_kpool"
src = open(os.path.join(SRC, "sglang", H.TARGET)).read()
mod = types.ModuleType(name); mod.__file__ = "patched"
importlib.import_module("sglang.srt.layers.attention.dsa")
exec(compile(H.patch_text(src), "patched_dsa_indexer_kpool", "exec"), mod.__dict__)
sys.modules[name] = mod

# --- fakes -------------------------------------------------------------------------------------------------------
g = torch.Generator().manual_seed(0)
n_req_rows = [700, 500]          # query rows of two requests in one extend batch
pooled = [900, 1300]             # pooled key rows per request
total_q = sum(n_req_rows) + 8    # padded q rows (beyond n_real)
n_real = sum(n_req_rows)
H_IDX, D = 4, 16
q = torch.randn(total_q, H_IDX, D, generator=g)
k = (torch.randn(sum(pooled), D, generator=g) * 0.5).to(torch.float8_e4m3fn)
w = torch.rand(total_q, H_IDX, 1, generator=g)
ks, ke, pool_lens, seq_lens = [], [], [], []
off = 0
for r, p in zip(n_req_rows, pooled):
    for i in range(r):
        ks.append(off); ke.append(off + p); pool_lens.append(p); seq_lens.append(4 * p + 1)
    off += p
ks, ke = torch.tensor(ks), torch.tensor(ke)
pool_lens, seq_lens = torch.tensor(pool_lens), torch.tensor(seq_lens)

class FakeDG:
    calls = []
    @staticmethod
    def fp8_mqa_logits(qq, kk, ww, a, b, clean_logits=True):
        kmat = kk[0].float()
        s = torch.einsum("thd,kd->thk", qq.float(), kmat).relu()
        s = (s * ww.float().unsqueeze(-1)).sum(1)            # [rows, total_k]
        col = torch.arange(kmat.shape[0])[None, :]
        mask = (col >= a[:, None]) & (col < b[:, None])
        FakeDG.calls.append(tuple(s.shape))
        return torch.where(mask, s, torch.full_like(s, float("-inf")))
mod.deep_gemm = FakeDG

import sglang.srt.layers.attention.dsa.kpool_fp8_index as kfi
kfi.gather_index_k_scale_prefix_into = lambda **kw: None
mod.get_token_to_kv_pool = lambda: None

class Plan:
    seq_lens_expanded = seq_lens; pooled_seq_lens_expanded = pool_lens; ragged_q_ks = ks; ragged_q_ke = ke
    ragged_total_k_rows = int(k.shape[0]); ragged_k_u8 = k.view(torch.uint8); ragged_k_scale = torch.ones(k.shape[0])
    ragged_concat_page_table = None; ragged_paged_page_table = None; ragged_paged_page_table_row_index = None
class AM:
    kpool_extend_plan = Plan(); topk_indices_offset = None
class MD:
    attn_metadata = AM(); topk_transform_method = None

class Self:
    index_kpool = 4; index_topk = 64
    def _get_index_k_read_buffer(self, pool, layer_id): return None
    def _topk_from_kpool_logits(self, logits, pool_lens, seq_lens=None, page_table=None, topk_offsets=None,
                                row_starts=None, out_rows=None, page_table_row_index=None):
        kk = min(16, logits.shape[1])
        idx = torch.topk(logits, kk, dim=1).indices - row_starts[:, None]
        if out_rows is not None and out_rows > idx.shape[0]:
            pad = torch.full((out_rows, kk), -1, dtype=idx.dtype); pad[: idx.shape[0]] = idx; idx = pad
        return idx
fn = mod.DSAIndexerKPool._get_topk_ragged_kpool_plan if hasattr(mod, "DSAIndexerKPool") else None
if fn is None:
    fn = next(v._get_topk_ragged_kpool_plan for v in mod.__dict__.values()
              if isinstance(v, type) and hasattr(v, "_get_topk_ragged_kpool_plan"))
q8 = q
os.environ.pop("NUMINOUS_KPOOL_LOGITS_BUDGET_MB", None)
FakeDG.calls.clear()
ref = fn(Self(), None, 0, q8, w, MD())
stock_calls = list(FakeDG.calls)
os.environ["NUMINOUS_KPOOL_LOGITS_BUDGET_MB"] = "1"     # 1 MiB -> rows per chunk = 2^20 // (2200*4) = 119
FakeDG.calls.clear()
got = fn(Self(), None, 0, q8, w, MD())
print("@@" + json.dumps({"equal": bool(torch.equal(ref, got)), "shape": list(ref.shape),
                         "stock_calls": stock_calls, "chunk_calls": len(FakeDG.calls),
                         "max_chunk_rows": max(c[0] for c in FakeDG.calls),
                         "pad_rows_minus1": bool((got[n_real:] == -1).all())}))
'''


@pytest.mark.skipif(not pathlib.Path(SGLANG_SRC, "sglang", H.TARGET).exists(), reason="no SGLang v0.5.20 tree")
def test_chunked_equals_stock():
    p = subprocess.run([sys.executable, "-c", _CHILD, SGLANG_SRC, str(HERE)], capture_output=True, text=True,
                       timeout=600, env=dict(os.environ, HF_HUB_OFFLINE="1"))
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@")), None)
    assert line, p.stderr[-3000:]
    r = json.loads(line[2:])
    assert r["equal"], r
    assert r["stock_calls"] == [[1200, 2200]], r            # stock: one dense [n_real x total_k] logits
    assert r["chunk_calls"] == 11 and r["max_chunk_rows"] <= 119, r   # 1200 rows / 119 per chunk
    assert r["pad_rows_minus1"] and r["shape"] == [1208, 16], r


def test_anchors_apply_once():
    src = pathlib.Path(SGLANG_SRC, "sglang", H.TARGET)
    if not src.exists():
        pytest.skip("no tree")
    t = src.read_text()
    new = H.patch_text(t)
    assert H.MARKER in new and H.patch_text(new) == new   # idempotent
