"""Box test: apply pro_skew_budget_hooks to a copy of the installed sglang package and import it.

Run on a dev box:
    python3 operations/devbox/devbox.py test <box> operations/patches/test_pro_skew_budget_box.py

Or locally (if sglang is installed):
    python3 -m pytest operations/patches/test_pro_skew_budget_box.py -v
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


@pytest.fixture(scope="module")
def patched_root():
    """Copy the installed sglang package to a temp dir, apply hooks, return the path."""
    try:
        import pro_skew_budget_hooks as hooks
        src_root = hooks.sglang_root()
    except SystemExit:
        pytest.skip("sglang not importable")
    tmpdir = tempfile.mkdtemp(prefix="sglang_skew_test_")
    dst_root = os.path.join(tmpdir, "sglang")
    # Copy only the files we need (scheduler.py, schedule_policy.py, prefill_delayer.py, and the managers init)
    for subpath in [
        "srt/managers/scheduler.py",
        "srt/managers/schedule_policy.py",
        "srt/managers/prefill_delayer.py",
        "srt/managers/__init__.py",
        "srt/__init__.py",
        "__init__.py",
    ]:
        src = os.path.join(src_root, subpath)
        dst = os.path.join(dst_root, subpath)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    yield dst_root
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_apply_and_import(patched_root):
    """Apply the hooks and verify the module source compiles."""
    import pro_skew_budget_hooks as hooks
    # Apply to the copy
    lines = hooks.apply(patched_root)
    for line in lines:
        print(line)
        assert "error" not in line.lower(), f"unexpected error: {line}"
    # Verify module was written
    mod_path = os.path.join(patched_root, hooks.MODULE_PATH)
    assert os.path.exists(mod_path), f"module not written at {mod_path}"
    # Verify it compiles
    src = open(mod_path).read()
    compile(src, mod_path, "exec")
    # Verify the scheduler.py hooks were applied
    sched_path = os.path.join(patched_root, "srt/managers/scheduler.py")
    sched_src = open(sched_path).read()
    assert "numinous-hook:pro_skew_budget_install" in sched_src
    assert "numinous-hook:pro_skew_budget_chunk" in sched_src
    # Verify the schedule_policy.py hook was applied
    policy_path = os.path.join(patched_root, "srt/managers/schedule_policy.py")
    policy_src = open(policy_path).read()
    assert "numinous-hook:pro_skew_budget_chunked" in policy_src


def test_apply_idempotent(patched_root):
    """Applying twice is safe (already-applied detection)."""
    import pro_skew_budget_hooks as hooks
    # First apply (may already be applied from test_apply_and_import)
    hooks.apply(patched_root)
    # Second apply should report "already applied"
    lines = hooks.apply(patched_root)
    for line in lines:
        assert "already applied" in line or "up to date" in line or "module:" in line, f"unexpected on re-apply: {line}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def _names_defined_before(func, line: int) -> set:
    """Parameters plus names bound by statements that start before `line` in `func`."""
    import ast
    out = {a.arg for a in func.args.args + func.args.kwonlyargs}
    for node in ast.walk(func):
        if getattr(node, "lineno", line) >= line:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in node.names)
    return out


@pytest.mark.parametrize("rel,func_name,marker", [
    ("srt/managers/schedule_policy.py", "add_chunked_req", "numinous-hook:pro_skew_budget_chunked"),
    ("srt/managers/scheduler.py", "_get_new_batch_prefill_raw", "numinous-hook:pro_skew_budget_chunk"),
])
def test_hook_reads_only_bound_names(patched_root, rel, func_name, marker):
    """Every name the inserted hook reads must be bound before it (the first gate crashed on
    cand_extend_input_len, which v0.5.20 assigns only after the delayer block)."""
    import ast
    import builtins
    import pro_skew_budget_hooks as hooks
    hooks.apply(patched_root)
    src = open(os.path.join(patched_root, rel)).read()
    tree = ast.parse(src)
    hook_line = next(i for i, ln in enumerate(src.splitlines(), 1) if marker in ln)
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name
                and n.lineno < hook_line <= n.end_lineno)
    hook_stmts = [n for n in ast.walk(func) if isinstance(n, ast.stmt) and n.lineno > hook_line
                  and n.lineno <= hook_line + 25 and n is not func]
    # names bound inside the hook itself count once they are assigned
    known = _names_defined_before(func, hook_line + 1) | set(dir(builtins))
    module_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    and n.col_offset == 0} | {(a.asname or a.name).split(".")[0] for n in tree.body
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    bad = []
    for st in sorted(hook_stmts, key=lambda n: n.lineno):
        seg = ast.get_source_segment(src, st) or ""
        if "numinous" not in seg and "_psb" not in seg:
            continue
        # compound statements: only their own expression (the nested statements are visited on their own)
        roots = [st.test] if isinstance(st, (ast.If, ast.While)) else (
            [] if isinstance(st, (ast.For, ast.With, ast.Try, ast.FunctionDef)) else [st])
        for root in roots:
            for n in ast.walk(root):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in known | module_names:
                    bad.append((st.lineno, n.id))
            known |= {n.id for n in ast.walk(root) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            known |= {(a.asname or a.name) for n in ast.walk(root) if isinstance(n, (ast.Import, ast.ImportFrom))
                      for a in n.names}
    assert not bad, f"{rel}:{func_name}: hook reads unbound names {bad}"
