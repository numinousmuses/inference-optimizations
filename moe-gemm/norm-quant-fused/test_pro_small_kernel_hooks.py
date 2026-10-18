"""The Pro small-kernel hook patcher against the SGLang installed on the box (skips without sglang).

Copies the installed package, applies every hook (which also installs sglang.srt.layers.numinous_pro), then in
clean interpreters: imports every patched module, and drives three hooks through the patched engine code with the
env flags on: the mHC boundary, the FP8 LM head in LogitsProcessor._compute_lm_head, and the pre-quantized input
in Fp8LinearMethod.apply. Each checks the result against the kernel called directly and the activation log line.
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

import pro_small_kernel_hooks as hooks  # noqa: E402

sglang_spec = importlib.util.find_spec("sglang")
pytestmark = pytest.mark.skipif(sglang_spec is None, reason="needs an installed sglang (dev box)")


@pytest.fixture(scope="module")
def patched(tmp_path_factory) -> pathlib.Path:
    src = list(sglang_spec.submodule_search_locations)[0]
    base = tmp_path_factory.mktemp("sgl")
    shutil.copytree(src, base / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
    lines = hooks.apply(str(base / "sglang"), [])
    assert any("modules installed" in x for x in lines), lines
    return base


def _run(patched, code, **env):
    e = {**os.environ, "PYTHONPATH": str(patched), **env}
    r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-4000:])
    return r.stdout + r.stderr


def test_every_anchor_resolves_once_on_the_installed_sglang():
    lines = hooks.apply(list(sglang_spec.submodule_search_locations)[0], [], check=True)
    assert all(x.endswith(("anchor ok", "already applied")) for x in lines), lines
    assert len(lines) == len(hooks.HOOKS)


def test_apply_is_idempotent(patched):
    again = hooks.apply(str(patched / "sglang"), [], modules=False)
    assert all(x.endswith("already applied") for x in again), again


def test_patched_modules_import_in_a_clean_interpreter(patched):
    code = ("import sglang, inspect; assert sglang.__file__.startswith('%s')\n"
            "import sglang.srt.models.deepseek_v4 as m\n"
            "import sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc as f\n"
            "import sglang.srt.layers.attention.dsv4.indexer as ix\n"
            "import sglang.srt.layers.logits_processor as lp\n"
            "import sglang.srt.layers.quantization.fp8 as q\n"
            "from sglang.srt.layers.numinous_pro import pro_mhc_fused, pro_norm_quant, pro_indexer_q, pro_fp8_lm_head\n"
            "for mod, name in [(f, 'mhc_fused'), (m, 'norm_quant'), (m, 'pro_post_load'), (ix, 'indexer_q'),\n"
            "                  (lp, 'fp8_lm_head'), (q, 'norm_quant_linear')]:\n"
            "    assert 'numinous-hook:' + name in inspect.getsource(mod), name\n"
            "print('ok')" % patched)
    assert "ok" in _run(patched, code).split()


MHC_CODE = r"""
import flashinfer, contextlib, types, torch
import sglang.srt.distributed.parallel_state as ps
import sglang.srt.distributed.device_communicators.pynccl_allocator as pa
import sglang.srt.layers.dp_attention as da
ps.get_tp_group = lambda: types.SimpleNamespace(world_size=1)   # no process group in a unit test
pa.use_symmetric_memory = lambda *a, **k: contextlib.nullcontext()
da.is_allocation_symmetric = lambda: False
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import apply_mhc_post_pre_boundary
from sglang.srt.layers.numinous_pro import pro_mhc_fused as M
x, res, post, comb, fn, sc, base, nw = M.make_inputs(96, 7168, device="cuda", seed=3)
r = apply_mhc_post_pre_boundary(x, res, post, comb, fn, sc, base, 4, 1e-6, 1e-6, 2.0, 20, nw, 1e-6, fn_transpose=True)
d = M.mhc_post_pre_fused(x, res, post, comb, fn, sc, base, 1e-6, 1e-6, 1e-6, 2.0, 20, nw, 1e-6)
assert r[4] is True
for a, b in zip((r[0], r[2], r[3], r[1]), d):
    assert torch.equal(a, b)
print("ok")
"""


def test_mhc_hook_drives_the_fused_kernel(patched):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    out = _run(patched, MHC_CODE, NUMINOUS_MHC_FUSED="1")
    assert "numinous mhc_fused active" in out and "ok" in out.split(), out[-2000:]


LM_CODE = r"""
import flashinfer, types, torch
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.numinous_pro import pro_fp8_lm_head as LM
torch.manual_seed(0)
head = torch.nn.Module(); head.weight = torch.nn.Parameter((torch.randn(2048, 7168, device="cuda") * 0.02).bfloat16(), requires_grad=False)
model = types.SimpleNamespace(lm_head=head)
LM.engine_post_load(model)
h = torch.randn(96, 7168, device="cuda").bfloat16()
stub = types.SimpleNamespace(use_fp32_lm_head=False)
out = LogitsProcessor._compute_lm_head(stub, h, head)
assert torch.equal(out, LM.fp8_logits(h, head))
stub.use_fp32_lm_head = True
assert LogitsProcessor._compute_lm_head(stub, h, head).dtype == torch.float32
print("ok")
"""


def test_fp8_lm_head_hook(patched):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    out = _run(patched, LM_CODE, NUMINOUS_FP8_LM_HEAD="1")
    assert "numinous fp8_lm_head active" in out and "ok" in out.split(), out[-2000:]


LINEAR_CODE = r"""
import flashinfer, types, torch
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0, deepgemm_w8a8_block_fp8_linear_with_fallback
from sglang.srt.layers.numinous_pro import pro_norm_quant as NQ
torch.manual_seed(0)
wq, ws = quant_weight_ue8m0((torch.randn(8192, 1536, device="cuda") * 0.02).bfloat16(), weight_block_size=[128, 128])
ws = transform_scale_ue8m0(ws, mn=8192)
layer = types.SimpleNamespace(weight=wq, weight_scale_inv=ws, bias=None)
qn = types.SimpleNamespace(weight=torch.ones(1536, device="cuda", dtype=torch.bfloat16), variance_epsilon=1e-6)
x = torch.randn(96, 2048, device="cuda").bfloat16()[:, :1536]
y = NQ.engine_q_norm(qn, x)
assert hasattr(y, "_numinous_fp8")
stub = types.SimpleNamespace(use_marlin=False, use_mxfp8=False, block_quant=True)
out = Fp8LinearMethod.apply(stub, layer, y)
ref = deepgemm_w8a8_block_fp8_linear_with_fallback(y, wq, [128, 128], ws)
assert torch.equal(out, ref)
print("ok")
"""


def test_norm_quant_hooks(patched):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    out = _run(patched, LINEAR_CODE, NUMINOUS_NORM_QUANT="1")
    assert "numinous norm_quant active" in out and "ok" in out.split(), out[-2000:]


INDEXER_CODE = r"""
import flashinfer, types, torch
from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
from sglang.srt.layers.numinous_pro import pro_norm_quant as NQ, pro_indexer_q as IQ
torch.manual_seed(0)
wq, ws = quant_weight_ue8m0((torch.randn(8192, 1536, device="cuda") * 0.03).bfloat16(), weight_block_size=[128, 128])
ws = transform_scale_ue8m0(ws, mn=8192)
ang = torch.outer(torch.arange(65536.0), 1.0 / (10000 ** (torch.arange(0, 64, 2) / 64)))
stub = types.SimpleNamespace(wq_b=types.SimpleNamespace(weight=wq, weight_scale_inv=ws),
                             freqs_cis=torch.polar(torch.ones_like(ang), ang).cuda(), weight_scale=0.011,
                             use_fp4_indexer=False)
assert IQ.engine_prepare(stub)
qn = types.SimpleNamespace(weight=torch.ones(1536, device="cuda", dtype=torch.bfloat16), variance_epsilon=1e-6)
y = NQ.engine_q_norm(qn, torch.randn(96, 1536, device="cuda").bfloat16())
w = torch.randn(96, 64, device="cuda").bfloat16()
pos = torch.randint(0, 65536, (96,), device="cuda")
q8, wo = C4Indexer.compute_q(stub, y, pos, w)
r8, rwo = IQ.indexer_q_fused(*y._numinous_fp8, wq, stub._numinous_wbs, w, 0.011, stub._numinous_freqs, pos)
assert torch.equal(q8.view(torch.uint8), r8.view(torch.uint8)) and torch.equal(wo, rwo)
print("ok")
"""


def test_indexer_q_hook(patched):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    out = _run(patched, INDEXER_CODE, NUMINOUS_NORM_QUANT="1", NUMINOUS_INDEXER_Q="1")
    assert "numinous indexer_q active" in out and "ok" in out.split(), out[-2000:]
