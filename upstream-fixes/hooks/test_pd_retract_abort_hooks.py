"""pd_retract_abort_hooks: anchors on the v0.5.20 tree, idempotent, and the patched retraction_backup returns False
(the abort-this-request result) when the pool has no CPU copy, re-raises with NUMINOUS_PD_RETRACT_ABORT=0."""
import ast
import importlib.util
import pathlib
import shutil

import pytest

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("pd_retract_abort_hooks", HERE / "pd_retract_abort_hooks.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

TREE = pathlib.Path("/tmp/sgl-v0520-wt/python/sglang")


def _tree(tmp_path):
    if not (TREE / h.PATH).exists():
        pytest.skip("v0.5.20 tree not checked out at /tmp/sgl-v0520-wt")
    dst = tmp_path / "sglang"
    (dst / h.PATH).parent.mkdir(parents=True)
    shutil.copy(TREE / h.PATH, dst / h.PATH)
    return dst


def _retraction_backup(root):
    """The patched function, compiled alone with stub globals."""
    src = (root / h.PATH).read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "retraction_backup")
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"Req": object, "BasePrefixCache": object, "ReqToTokenPool": object, "BaseTokenToKVPoolAllocator": object,
          "cast": lambda t, v: v}
    exec(compile(mod, "patched", "exec"), ns)
    return ns["retraction_backup"]


class _KV:
    retraction_backup = "stale"


class _Req:
    rid = "r1"

    def __init__(self, fail):
        self.kv = _KV()
        self.fail = fail

    def offload_kv_cache(self, *a):
        if self.fail:
            raise NotImplementedError("DeepSeekV4TokenToKVPool has no get_cpu_copy")
        self.kv.retraction_backup = "saved"


def test_check_apply_idempotent(tmp_path):
    root = _tree(tmp_path)
    assert h.apply(str(root), check=True) == [f"{h.NAME}: anchor ok in {h.PATH}"]
    assert h.apply(str(root)) == [f"{h.NAME}: applied to {h.PATH}"]
    assert h.apply(str(root)) == [f"{h.NAME}: already applied"]
    assert h.apply(str(root), check=True) == [f"{h.NAME}: already applied"]
    compile((root / h.PATH).read_text(), h.PATH, "exec")


def test_patched_backup_aborts_one_request_instead_of_raising(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    h.apply(str(root))
    rb = _retraction_backup(root)
    ok = _Req(fail=False)
    assert rb(ok, None, None, None, "cpu_tensor") is True and ok.kv.retraction_backup == "saved"
    bad = _Req(fail=True)
    assert rb(bad, None, None, None, "cpu_tensor") is False      # retract_decode: FINISH_ABORT for this request
    assert bad.kv.retraction_backup is None
    monkeypatch.setenv("NUMINOUS_PD_RETRACT_ABORT", "0")
    with pytest.raises(NotImplementedError):
        rb(_Req(fail=True), None, None, None, "cpu_tensor")


def test_stock_tree_raises(tmp_path):
    root = _tree(tmp_path)
    rb = _retraction_backup(root)
    with pytest.raises(NotImplementedError):
        rb(_Req(fail=True), None, None, None, "cpu_tensor")
