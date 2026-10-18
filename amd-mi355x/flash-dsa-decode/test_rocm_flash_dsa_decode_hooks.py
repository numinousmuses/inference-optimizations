import hashlib
import os

import rocm_flash_dsa_decode_hooks as H

TL_1002 = "/tmp/rtxsys/amdsrc/tilelang_kernel.py"


def test_kernel_hash_pin():
    assert hashlib.sha256(open(H.MODULE_SRC, "rb").read()).hexdigest()[:16] == H.KERNEL_SHA256


def test_coexists_with_flash_prefill_hook_and_runs_first():
    if not os.path.exists(TL_1002):
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("fph", os.path.join(os.path.dirname(H.HERE), "amdflash",
                                                                       "flash_prefill_hooks.py"))
    fph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fph)
    src = open(TL_1002).read()
    a = src.replace(fph.ANCHOR, fph.TEXT, 1)            # amdflash's hook first (overlay order)
    b, msg = H.patch_source(a)
    assert "applied" in msg
    compile(b, "tl", "exec")
    assert b.index("numinous-hook:flash_dsa_decode") < b.index("numinous-hook:flash_sparse_prefill")
    c, msg2 = H.patch_source(b)
    assert c == b and "already" in msg2


def test_routing_condition_text():
    assert "is_current_stream_capturing()" in H.TEXT and "NUMINOUS_FLASH_DSA_DECODE" in H.TEXT
