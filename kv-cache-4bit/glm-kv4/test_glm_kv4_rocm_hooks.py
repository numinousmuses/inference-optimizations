"""glm_kv4_rocm_hooks + glm_kv4_rocm on the ROCm SGLang tree (MI355X box).

1. anchors resolve on the installed ROCm tree; applying to a copy compiles, is idempotent, and the patched package
   imports with the three modules installed (subprocess with PYTHONPATH at the copy);
2. functional (GPU): forward_tilelang over an FP4 pool equals the stock tilelang_sparse_fwd over the FP8 pool that
   holds exactly the dequantized rows (same keys, same order) -- plain gather, dedup, chunked eager, -1 rows, and a
   table whose width is not a multiple of 64.

    HIP_VISIBLE_DEVICES=3 python3 -m pytest -q operations/patches/test_glm_kv4_rocm_hooks.py
"""
import os
import shutil
import subprocess
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import glm_kv4_rocm_hooks as hooks  # noqa: E402

pytestmark = pytest.mark.skipif(torch.version.hip is None, reason="ROCm tree / MI355X only")


@pytest.fixture(scope="module")
def patched_root(tmp_path_factory):
    src = hooks.sglang_root()
    out = tmp_path_factory.mktemp("sgl") / "python"
    shutil.copytree(src, out / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
    return str(out)


def test_anchors_resolve_on_installed_tree():
    lines = hooks.apply(hooks.sglang_root(), check=True)
    assert len(lines) == len(hooks.HOOKS), lines
    assert all(("anchor ok in" in l) or l.endswith("already applied") for l in lines), lines


def test_apply_idempotent_and_imports(patched_root):
    root = os.path.join(patched_root, "sglang")
    first = hooks.apply(root)
    assert sum("applied to" in l for l in first) == len(hooks.HOOKS)
    assert sum(l.startswith("installed") for l in first) == 3
    conf = open(os.path.join(root, hooks.CONFIGURATOR)).read()
    fn = conf[conf.index("def calculate_mla_kv_cache_dim("):]
    # the FP4 return must precede the HIP raw-layout return (the first ROCm boot hit the trtllm-branch anchor: 576)
    assert fn.index("numinous-hook:kv4r_dim") < fn.index("if _is_hip and (")
    second = hooks.apply(root, install_modules=False)
    assert all(l.endswith("already applied") for l in second)
    code = ("import sglang, sglang.srt.layers.attention.dsa.glm_kv4_rocm as r, "
            "sglang.srt.layers.attention.dsa_backend as b, sglang.srt.mem_cache.kv_cache_configurator as c; "
            "import inspect; assert 'numinous-hook:kv4r_route' in inspect.getsource(b); print(sglang.__file__)")
    res = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": patched_root},
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip().startswith(patched_root), res.stdout


def _backend(pool_buf, topk, dedup):
    os.environ["NUMINOUS_GLM_KV4_DEDUP"] = "1" if dedup else "0"
    os.environ["NUMINOUS_GLM_KV4_SEL_ROWS"] = "8"  # force chunking in the eager path
    pool = types.SimpleNamespace(kv_buffer=[pool_buf])
    return types.SimpleNamespace(token_to_kv_pool=pool, device="cuda", dsa_index_topk=topk)


@pytest.mark.parametrize("t,k,dedup", [(5, 2048, False), (20, 2048, False), (200, 2048, True), (3, 2000, False)])
def test_forward_tilelang_equals_fp8_pool(patched_root, t, k, dedup):
    sys.path.insert(0, patched_root)
    for m in [m for m in sys.modules if m == "sglang" or m.startswith("sglang.")]:
        del sys.modules[m]
    from sglang.srt.layers.attention.dsa import glm_kv4 as kv4
    from sglang.srt.layers.attention.dsa import glm_kv4_rocm as r
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    torch.manual_seed(0)
    dev = torch.device("cuda")
    n_slots, heads = 30000, 16
    nope = (torch.randn(n_slots, 512, device=dev) * 0.5).to(torch.bfloat16)
    rope = (torch.randn(n_slots, 64, device=dev) * 0.5).to(torch.bfloat16)
    pool4 = torch.zeros(n_slots, kv4.ROW_BYTES, dtype=torch.uint8, device=dev)
    kv4.quant_write(pool4, torch.arange(n_slots, device=dev), nope, rope)
    pool8 = kv4.dequantize_rows_ref(pool4).view(torch.float8_e4m3fn)  # [slots, 576] raw layout, exact dequant
    table = torch.randint(0, n_slots, (t, k), dtype=torch.int32, device=dev)
    if dedup:  # verify-like: consecutive tokens share most rows
        table[1::2] = table[0::2][: table[1::2].shape[0]]
    table[0, 100:] = -1
    q = (torch.randn(t, heads, 576, device=dev) * 0.3).to(torch.bfloat16)
    be = _backend(pool4.view(n_slots, 1, kv4.ROW_BYTES), k, dedup)
    out = r.forward_tilelang(be, q, pool4.view(n_slots, 1, kv4.ROW_BYTES), table, 576 ** -0.5, 512)
    kp = -(-k // 64) * 64
    ref_tab = torch.cat((table, table.new_full((t, kp - k), -1)), 1) if kp != k else table
    ref = tilelang_sparse_fwd(q=q, kv=pool8.view(n_slots, 1, 576), indices=ref_tab.unsqueeze(1), sm_scale=576 ** -0.5,
                              d_v=512)
    torch.cuda.synchronize()
    assert out.shape == ref.shape, (out.shape, ref.shape)
    err = (out.float() - ref.float()).abs().max().item()
    assert torch.isfinite(out.float()).all()
    assert err <= 1e-2 * max(1.0, ref.float().abs().max().item()), err
