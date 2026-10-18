"""git_series_hooks.py on a throwaway git tree: forward apply, idempotent re-run, --check, wrong tree fails loudly; and
aiter_cfg_hooks.py's check/idempotence contract (the build itself runs on the MI355X box against the image's AITER)."""
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kernels" / "aiter_tuning"))
import aiter_cfg_hooks as ac  # noqa: E402
import git_series_hooks as gs  # noqa: E402


@pytest.fixture()
def tree(tmp_path):
    t = tmp_path / "t"
    (t / "python").mkdir(parents=True)
    (t / "python" / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=t, check=True)
    subprocess.run(["git", "add", "."], cwd=t, check=True)
    subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "user.email=a@b", "-c", "user.name=n", "commit", "-qm", "base"], cwd=t, check=True)
    (t / "python" / "a.py").write_text("x = 2\n")
    (t / "test").mkdir()
    (t / "test" / "new.py").write_text("ok = True\n")
    subprocess.run(["git", "add", "-A"], cwd=t, check=True)
    patch = tmp_path / "0001-x.patch"
    patch.write_text(subprocess.run(["git", "diff", "--cached"], cwd=t, check=True, capture_output=True, text=True).stdout)
    subprocess.run(["git", "reset", "-q", "--hard"], cwd=t, check=True)
    subprocess.run(["git", "clean", "-qfd"], cwd=t, check=True)
    return t, patch


def test_apply_then_idempotent_and_check(tree):
    t, p = tree
    assert gs.apply(str(t), [str(p)], check=True) == ["0001-x.patch: anchor ok"]
    assert gs.apply(str(t), [str(p)]) == ["0001-x.patch: applied"]
    assert (t / "python" / "a.py").read_text() == "x = 2\n" and (t / "test" / "new.py").exists()
    assert gs.apply(str(t), [str(p)]) == ["0001-x.patch: already applied"]
    assert gs.apply(str(t), [str(p)], check=True) == ["0001-x.patch: already applied"]


def test_include_limits_paths(tree):
    t, p = tree
    gs.apply(str(t), [str(p)], include="python/*")
    assert (t / "python" / "a.py").read_text() == "x = 2\n" and not (t / "test" / "new.py").exists()


def test_wrong_tree_fails_loudly(tree):
    t, p = tree
    (t / "python" / "a.py").write_text("y = 9\n")
    with pytest.raises(SystemExit):
        gs.apply(str(t), [str(p)])


def test_aiter_cfg_check_contract(tmp_path):
    out = tmp_path / "cfg.csv"
    assert ac.apply(str(out), "/nonexistent", check=True) == [f"aiter_cfg {out}: anchor ok"]
    out.write_text("gfx,cu_num,M,N,K\n")
    assert ac.apply(str(out), "/nonexistent", check=True) == [f"aiter_cfg {out}: anchor ok"]
    out.write_text("gfx,cu_num,M,N,K\ngfx950,256,1,2,3\n")
    assert ac.apply(str(out), "/nonexistent") == [f"aiter_cfg {out}: already applied (1 rows)"]
    assert ac.apply(str(out), "/nonexistent", check=True) == [f"aiter_cfg {out}: already applied (1 rows)"]


def test_aiter_cfg_refuses_other_bytes(tmp_path):
    out = tmp_path / "cfg.csv"
    out.write_text("gfx,cu_num,M,N,K\ngfx950,256,1,2,3\n")
    good = ac.sha256(str(out))
    assert ac.apply(str(out), "/x", expect=good) == [f"aiter_cfg {out}: already applied (1 rows)"]
    with pytest.raises(SystemExit):
        ac.apply(str(out), "/x", expect="0" * 64)


def test_tuning_build_is_decided_by_commit_not_path(tmp_path, monkeypatch):
    import build_merged_config as bm
    a = tmp_path / "aiter"
    (a / "aiter").mkdir(parents=True)
    (a / "aiter" / "_version.py").write_text("__version__ = '0.1.23.dev183+ge7d2453f2.d20261002'\n")
    assert bm.aiter_commit(a) == "e7d2453f2" and not bm.is_tuning_build(a)
    (a / "aiter" / "_version.py").write_text("__version__ = '0.1.22.dev120+gacf8fdf93.d20260923'\n")
    assert bm.is_tuning_build(a)


def test_created_file_edited_later_counts_as_applied(tree):
    """0004-style: the patch creates test/new.py, a later patcher edits it; re-runs and --check must say applied."""
    t, p = tree
    gs.apply(str(t), [str(p)])
    (t / "test" / "new.py").write_text("ok = True\n# numinous-hook: edited by a later patcher\n")
    out = gs.apply(str(t), [str(p)], check=True)
    assert out == ["0001-x.patch: already applied (files it creates were edited later: test/new.py)"]
    assert gs.apply(str(t), [str(p)])[0].startswith("0001-x.patch: already applied")


def test_modified_file_drift_still_fails(tree):
    """Only created files may differ: drift in a file the patch modifies is still a wrong tree."""
    t, p = tree
    gs.apply(str(t), [str(p)])
    (t / "python" / "a.py").write_text("x = 3\n")
    with pytest.raises(SystemExit):
        gs.apply(str(t), [str(p)], check=True)
