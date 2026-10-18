import hashlib
import os

import rocm_sparse_decode_hip_hooks as H
import rocm_prefill_hooks as P

TL_1002 = "/tmp/rtxsys/amdsrc/tilelang_kernel.py"


def test_kernel_hash_pin():
    assert hashlib.sha256(open(H.MODULE_SRC, "rb").read()).hexdigest()[:16] == H.KERNEL_SHA256


def test_runs_before_amdglm_sparse_decode_branch_and_is_idempotent():
    if not os.path.exists(TL_1002):
        return
    src = open(TL_1002).read()
    hook = next(h for h in P.HOOKS if h.name == "tl_sparse_prefill")
    a = src.replace(hook.anchor, hook.replacement, 1)            # amdglm's hook first (overlay order)
    b, msg = H.patch_source(a)
    assert "applied" in msg
    compile(b, "tl", "exec")
    assert b.index("numinous-hook:sparse_decode_hip") < b.index("numinous-hook:tl_sparse_prefill")
    c, msg2 = H.patch_source(b)
    assert c == b and "already" in msg2


def test_routing_condition_text():
    for s in ("NUMINOUS_SPARSE_DECODE_HIP", "num_heads == 64", "tail_dim == 64", "is_current_stream_capturing()",
              "topk <= 2048", "NUMINOUS_SPARSE_DECODE_HIP_MAX_ROWS"):
        assert s in H.TEXT, s


def test_installed_module_is_self_contained():
    src = open(H.MODULE_SRC).read()
    assert "from rocm_sparse_mla_prefill import _sparse_mla_combine_kernel" not in src.split("def main")[0]
