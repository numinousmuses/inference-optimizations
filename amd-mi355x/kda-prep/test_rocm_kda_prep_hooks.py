"""rocm_kda_prep_hooks: anchors / compile / idempotent on 1330e437e1; module sha pinned; the fused split + L2 norm
kernel equals split + l2norm (TRITON_INTERPRET on CPU) (k3amd)."""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TREE = pathlib.Path("/tmp/sgl-1002/python/sglang")
sys.path.insert(0, str(ROOT / "operations/patches"))
import rocm_kda_prep_hooks as hk  # noqa: E402


def test_apply_idempotent_compiles(tmp_path):
    if not TREE.exists():
        pytest.skip("1330e437e1 tree not unpacked")
    dst = tmp_path / "sglang"
    for f in (hk.KB, hk.KT):
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(TREE / f, dst / f)
    assert all("anchor ok" in x for x in hk.apply(str(dst), check=True))
    out = hk.apply(str(dst))
    assert sum("applied to" in x for x in out) == 3
    assert all("already applied" in x for x in hk.apply(str(dst)) if not x.startswith("module"))
    for f in (hk.KB, hk.KT):
        compile((dst / f).read_text(), f, "exec")


def test_module_sha_pinned():
    assert hk.module_sha16() == hk.MODULE_SHA256


KERNEL_CHECK = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
import numinous_kda_prep as m
torch.manual_seed(0)
T, H, D = 37, 4, 128
qkv = torch.randn(T, 3 * H * D, dtype=torch.float32) * 3
q, k, v = m.split_l2norm(qkv, H * D, H * D, H * D, D)
rq, rk, rv = qkv.split([H * D] * 3, dim=-1)
def l2(x):
    x = x.view(T, H, D).float()
    return (x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)).view(T, H * D)
assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
assert torch.allclose(q, l2(rq), atol=1e-5) and torch.allclose(k, l2(rk), atol=1e-5)
assert torch.equal(v, rv)
print("OK")
"""


def test_kernel_matches_reference_interpreted():
    pytest.importorskip("triton")
    pytest.importorskip("torch")
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    p = subprocess.run([sys.executable, "-c", KERNEL_CHECK, str(ROOT / "operations/kernels")], env=env,
                       capture_output=True, text=True, timeout=300)
    assert "OK" in p.stdout, p.stderr[-2000:]
