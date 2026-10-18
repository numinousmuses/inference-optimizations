import os

import rocm_dp_empty_idle_hooks as H

SRC = "/tmp/rtxsys/amdsrc/forward_batch_info.py"   # 1330e437e1 copy of model_executor/forward_batch_info.py


def test_applies_compiles_idempotent():
    if not os.path.exists(SRC):
        return
    new, msg = H.patch_source(open(SRC).read())
    assert "applied" in msg
    compile(new, "forward_batch_info.py", "exec")
    again, msg2 = H.patch_source(new)
    assert again == new and "already" in msg2


class _M:
    def __init__(self, n):
        self.n = n

    def is_idle(self):
        return self.n == "IDLE"


def _run(mode, num_tokens, env=None):
    """Execute the patched branch on a stub batch."""
    body = "\n".join(ln[16:] for ln in H.TEXT.splitlines())
    ns = {"ForwardMode": type("FM", (), {"TARGET_VERIFY": _M("TARGET_VERIFY")}),
          "self": type("B", (), {"forward_mode": _M(mode), "_original_forward_mode": None})(),
          "num_tokens": num_tokens}
    old = os.environ.get("NUMINOUS_DP_EMPTY_IDLE")
    if env is not None:
        os.environ["NUMINOUS_DP_EMPTY_IDLE"] = env
    try:
        exec(body, ns)
    finally:
        if env is not None:
            if old is None:
                del os.environ["NUMINOUS_DP_EMPTY_IDLE"]
            else:
                os.environ["NUMINOUS_DP_EMPTY_IDLE"] = old
    return ns["self"]


def test_zero_token_idle_stays_idle():
    b = _run("IDLE", 0)
    assert b.forward_mode.n == "IDLE" and b._original_forward_mode is None


def test_padded_idle_still_becomes_verify():
    b = _run("IDLE", 8)
    assert b.forward_mode.n == "TARGET_VERIFY" and b._original_forward_mode.n == "IDLE"


def test_switch_off_restores_stock():
    b = _run("IDLE", 0, env="0")
    assert b.forward_mode.n == "TARGET_VERIFY"
