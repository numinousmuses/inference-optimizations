"""CPU tests for the sync-free IMA locator (operations/kernels/ima_locator.py) and its anchored hooks
(operations/patches/ima_locator_hooks.py) on SGLang v0.5.20 (94602c9c2b) and main 172b1b4825.

    python3 -m pytest -q operations/patches/test_ima_locator.py
"""
from __future__ import annotations

import ast
import builtins
import importlib
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "operations", "kernels"))

TREES = {"v0.5.20": "/tmp/sgl-v0520-wt/python/sglang", "main-172b": "/tmp/sgl-rocm-172b/python/sglang"}


@pytest.fixture()
def loc(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMINOUS_IMA_LOCATOR", "1")
    monkeypatch.setenv("NUMINOUS_IMA_LOCATOR_DIR", str(tmp_path))
    monkeypatch.setenv("NUMINOUS_IMA_LOCATOR_NO_THREAD", "1")
    import ima_locator
    m = importlib.reload(ima_locator)
    yield m
    m._STATES.clear()
    m._SEEN.clear()


def _rec(m, site):
    return {h["site"]: h for h in m.snapshot()}.get(site)


def test_in_range_records_nothing(loc):
    loc.check("mamba_eager_state", torch.tensor([0, 3, 80, -1]), 0, 81, allow_neg1=True)
    loc.check("dsa_page_table", torch.arange(64).view(4, 16), 0, 64)
    assert loc.snapshot() == []


def test_first_violation_kept_and_count_accumulates(loc):
    loc.tick("cpu")
    loc.check("mamba_cow_src", torch.tensor([5, 81, 7, 900]), 0, 81)
    r = _rec(loc, "mamba_cow_src")
    assert (r["count"], r["first_bad_value"], r["first_bad_pos"], r["lo"], r["hi"], r["numel"], r["step"]) == \
        (2, 81, 1, 0, 81, 4, 1)
    loc.tick("cpu")
    loc.check("mamba_cow_src", torch.tensor([-3]), 0, 81)
    r = _rec(loc, "mamba_cow_src")
    assert r["count"] == 3 and r["first_bad_value"] == 81 and r["step"] == 1  # first hit is sticky


def test_neg1_policy_and_negative_values(loc):
    loc.check("mamba_replay_state", torch.tensor([1, -1, -1], dtype=torch.int32), 0, 81, allow_neg1=True)
    assert loc.snapshot() == []
    loc.check("mamba_clear", torch.tensor([-1]), 0, 81)  # clear slots never legitimately -1
    assert _rec(loc, "mamba_clear")["first_bad_value"] == -1
    loc.check("mamba_eager_state", torch.tensor([-2]), 0, 81, allow_neg1=True)
    assert _rec(loc, "mamba_eager_state")["first_bad_value"] == -2


def test_mask_ignores_untracked_rows_and_truncates_to_mask(loc):
    idx = torch.tensor([10**6, 4, 10**6, 10**6])          # garbage in rows that are not tracked
    loc.check("mamba_eager_track", idx, 0, 81, mask=torch.tensor([False, True, False]))
    assert loc.snapshot() == []
    loc.check("verify_track", idx, 0, 81, mask=torch.tensor([0, 1, 1, 0]) >= 1)
    r = _rec(loc, "verify_track")
    assert r["count"] == 1 and r["first_bad_pos"] == 2


def test_skips_when_off_empty_or_unbounded(monkeypatch, loc):
    loc.check("fwd_out_cache_loc", torch.tensor([5]), 0, None)
    loc.check("fwd_out_cache_loc", torch.empty(0, dtype=torch.int64), 0, 1)
    loc.check("fwd_out_cache_loc", None, 0, 1)
    assert loc.snapshot() == []
    monkeypatch.setattr(loc, "ENABLED", False)
    loc.check("fwd_out_cache_loc", torch.tensor([5]), 0, 1)
    assert loc.snapshot() == []


def test_no_host_sync_on_the_check_path(monkeypatch, loc):
    """The locator must not serialize: no .item()/.tolist()/.cpu()/bool()/int() on device data, no synchronize."""
    def boom(*a, **k):
        raise AssertionError("host sync on the locator path")
    for name in ("item", "tolist", "cpu", "__bool__", "__int__", "__index__", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    monkeypatch.setattr(torch.cuda, "synchronize", boom)
    loc.tick("cpu")
    loc.check("verify_step", torch.tensor([0, 1, 7]), 0, 4, mask=torch.tensor([True, True, True]))
    loc.check("dsa_page_table", torch.tensor([[1, -1, 99]], dtype=torch.int32), 0, 64, allow_neg1=True)
    monkeypatch.undo()
    assert {h["site"] for h in loc.snapshot()} == {"verify_step", "dsa_page_table"}


def test_poll_dumps_json_once_per_change(loc, tmp_path, capsys):
    assert loc.poll_once() == []
    loc.check("fwd_req_pool_indices", torch.tensor([3, 70]), 0, 65)
    ch = loc.poll_once()
    assert [c["site"] for c in ch] == ["fwd_req_pool_indices"]
    assert "numinous ima_locator HIT" in capsys.readouterr().err
    payload = json.load(open(tmp_path / f"ima_locator_{os.getpid()}.json"))
    assert payload["hits"][0]["first_bad_value"] == 70 and payload["hits"][0]["hi"] == 65
    assert loc.poll_once() == []                      # unchanged -> silent
    loc.check("fwd_req_pool_indices", torch.tensor([99]), 0, 65)
    assert [c["count"] for c in loc.poll_once()] == [2]


def test_bound_helpers(loc):
    class P:
        size, page_size = 1000, 64
    assert loc.kv_rows(P()) == 1064 and loc.kv_rows(object()) is None
    class C:
        conv = [torch.zeros(3, 129, 2)]
    assert loc.ckpt_rows(C()) == 129 and loc.ckpt_rows(None) is None
    assert loc.token_rows(torch.zeros(10, 1, 656, dtype=torch.uint8)) == 10
    assert loc.token_rows(torch.zeros(4, 64, 576)) == 256


# ----------------------------------------------------------------------------------------------- hooks on real trees

def _tree(name):
    root = TREES[name]
    if not os.path.isdir(root):
        pytest.skip(f"{root} missing")
    return root


@pytest.fixture(params=list(TREES))
def patched(request, tmp_path):
    src = _tree(request.param)
    dst = tmp_path / "sglang"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    import ima_locator_hooks as H
    lines = H.apply(str(dst))
    return request.param, str(dst), lines


def test_hooks_apply_once_compile_and_install_module(patched):
    import ima_locator_hooks as H
    _, root, lines = patched
    assert sum("applied to" in ln for ln in lines) == len(H.HOOKS)
    assert os.path.isfile(os.path.join(root, "srt", "numinous_ima_locator.py"))
    again = H.apply(root)
    assert sum("already applied" in ln for ln in again) == len(H.HOOKS)
    for h in H.HOOKS:
        src = open(os.path.join(root, h.path)).read()
        assert src.count(h.marker) == 1
        compile(src, h.path, "exec")


_EXPECT_FN = {"ima_fwd": "_forward_raw", "ima_cow": "_maybe_execute_deferred_mamba_cow_and_clear",
              "ima_eager": "_forward_metadata", "ima_replay": "_replay_metadata",
              "ima_verify": "commit_mamba_states_after_verify", "ima_kdaverify": "commit_mamba_states_after_verify",
              "ima_dsa": "_forward_flashinfer_sparse_mla", "ima_moe": "select_experts"}


def _enclosing(tree, line):
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= line <= node.end_lineno:
            if best is None or node.lineno > best.lineno:
                best = node
    return best


def _bound_before(fn, line):
    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    for node in ast.walk(fn):
        if getattr(node, "lineno", 10**9) >= line:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def _module_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def test_hook_code_lands_in_the_right_function_and_uses_only_bound_names(patched):
    """Every name a hook reads is a parameter / local assigned above it (or a builtin / its own _nil names)."""
    import ima_locator_hooks as H
    _, root, _ = patched
    for h in H.HOOKS:
        src = open(os.path.join(root, h.path)).read()
        tree = ast.parse(src)
        mline = next(i for i, ln in enumerate(src.splitlines(), 1) if h.marker in ln)
        fn = _enclosing(tree, mline)
        assert fn is not None and fn.name == _EXPECT_FN[h.name], (h.name, fn and fn.name)
        body = h.text[h.text.index("# numinous-hook"):]
        if h.replace:  # drop the re-emitted tail of the anchor (the longest common suffix)
            k = 0
            while k < min(len(body), len(h.anchor)) and body[-1 - k] == h.anchor[-1 - k]:
                k += 1
            body = body[:len(body) - k]
            body = body[:body.rfind("\n") + 1]
        hook_lines = body.count("\n")
        bound = _bound_before(fn, mline) | _module_names(tree) | set(dir(builtins))
        loads = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and mline <= node.lineno < mline + hook_lines:
                loads.add(node.id)
        hook_locals = {n for n in loads if n.startswith("_nil")}
        missing = loads - bound - hook_locals
        assert not missing, (h.name, sorted(missing))


def test_dsa_check_runs_before_the_rtxattn_route_in_either_order(tmp_path):
    src = _tree("v0.5.20")
    import ima_locator_hooks as H
    import rtxflash_attn_hooks as R
    for order in ("ima_first", "rtx_first"):
        dst = tmp_path / order / "sglang"
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.so"))
        steps = [H.apply, R.apply] if order == "ima_first" else [R.apply, H.apply]
        for step in steps:
            step(str(dst))
        s = open(dst / H.DSA).read()
        assert s.index("numinous-hook:ima_dsa") < s.index("numinous-hook:rtxattn_route"), order


def test_check_mode_cli_on_both_trees():
    for name in TREES:
        root = TREES[name]
        if not os.path.isdir(root):
            continue
        r = subprocess.run([sys.executable, os.path.join(HERE, "ima_locator_hooks.py"), "--root", root, "--check"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert r.stdout.count("anchor ok in") == 8, (name, r.stdout)
