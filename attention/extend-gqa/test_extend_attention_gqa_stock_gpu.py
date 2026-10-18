"""Box test: the Gemma GQA extend overlay (extend_attention_gqa) matches SGLang's stock Triton extend kernel,
including the sliding-window boundary (Gemma 4 sliding layers: W = 1023, 16 query / 8 KV heads x 256) and the
global-layer shape (16 / 2 x 512), fp8 and bf16 caches. Before 2026-09-28 the overlay's window was one key short.

    python3 operations/devbox/devbox.py test rtx-gemma operations/kernels/test_extend_attention_gqa_stock_gpu.py -q
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("flashinfer")  # engine ptxas
import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
stock = pytest.importorskip("sglang.kernels.ops.attention.extend_attention")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extend_attention_gqa as eg  # noqa: E402


@pytest.mark.parametrize("H,KVH,D,W", [(16, 8, 256, 1023), (16, 8, 256, 64), (16, 2, 512, -1)])
@pytest.mark.parametrize("cache", [torch.bfloat16, torch.float8_e4m3fn])
def test_matches_stock(H, KVH, D, W, cache):
    torch.manual_seed(0)
    dev = "cuda"
    prefixes, news = [1500, 30, 0], [200, 70, 33]
    P = sum(prefixes)
    kb = (torch.randn(P + 1, KVH, D, device=dev) * 0.5).to(cache)
    vb = (torch.randn(P + 1, KVH, D, device=dev) * 0.5).to(cache)
    T = sum(news)
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    ke = (torch.randn(T, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
    ve = (torch.randn(T, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
    qo = torch.tensor([0] + list(torch.cumsum(torch.tensor(news), 0)), dtype=torch.int32, device=dev)
    kvp = torch.tensor([0] + list(torch.cumsum(torch.tensor(prefixes), 0)), dtype=torch.int32, device=dev)
    idx = (torch.randperm(P, device=dev) + 1).to(torch.int64)
    o_ref = torch.empty_like(q)
    stock.extend_attention_fwd(q, ke, ve, o_ref, kb, vb, qo, kvp, idx, None, True, None, max(news), 1.0, 1.0,
                               D ** -0.5, sliding_window_size=W)
    o = torch.empty_like(q)
    eg.extend_attention_gqa(q, ke, ve, o, kb, vb, qo, kvp, idx, max(news), D ** -0.5, sliding_window_size=W)
    # fp8 path quantizes q per row in the overlay (stock casts q to the cache dtype): looser bound there
    tol = 0.02 if cache == torch.bfloat16 else 0.08
    assert (o.float() - o_ref.float()).abs().max() < tol
