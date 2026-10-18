"""Apply git-format patch files to the engine checkout as a hooks-series entry (the production start script runs every
hooks entry once, then again with --check, and refuses to launch unless each says "already applied").

The ROCm production recipes (k3b300 2026-10-03, AMD push D4: GLM-5.3 Value on 4x MI355X) carry their scheduler series
(operations/patches/rocm_burst_1002, rebased by amdpro for the lmsysorg/sglang-rocm 20261002 image) in the same pinned
numinous-inference hooks series as their anchored patchers, so one series ref pins all of the engine's changes. On the
gate box the series is applied to an overlay copy of python/sglang (amd_gate.set_git_patches, --include=python/*); a
production node owns its container, so the image's checkout is patched in place, like the CUDA recipes' series.

    python3 git_series_hooks.py [--tree /sgl-workspace/sglang] [--include 'python/*'] <patch> [<patch> ...]
    python3 git_series_hooks.py --check ... <patch> ...     # report, change nothing

Per patch, one line: "<name>: already applied" when the reverse applies cleanly (idempotent re-run), "<name>: applied"
after a clean forward apply, and a SystemExit naming the patch when neither direction applies (the tree is not the one
the series was cut for). A patch whose created files were edited later by an anchored patcher of the same series
(e.g. pro1002/0004 creates numinous_dp_fp8_combine.py, amdflash's fp8_combine_hooks edits it) reports "already applied
(files it creates were edited later: ...)" when every file it modifies still reverse-applies. --check prints "<name>: anchor ok" for a patch that would apply but is not applied, which the
start script treats as a refusal to start.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _git(tree: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "apply", *args], cwd=tree, capture_output=True, text=True)


def _patch_files(patch: str) -> list[tuple[str, bool]]:
    """(path, created) per file of a git-format patch."""
    files: list[tuple[str, bool]] = []
    for line in open(patch, encoding="utf-8", errors="replace"):
        if line.startswith("diff --git a/"):
            files.append((line.rstrip("\n").split(" b/", 1)[1], False))
        elif line.startswith("new file mode") and files:
            files[-1] = (files[-1][0], True)
    return files


def _applied_with_edited_new_files(tree: str, patch: str, include: str | None) -> list[str] | None:
    """A patch that creates a file which a later series entry (an anchored patcher) edits no longer reverse-applies as
    a whole. Treat it as applied when every file it modifies reverse-applies on its own and every file it creates
    exists; return the created files that differ from the patch (edited later). None when that does not hold."""
    import fnmatch
    edited = []
    files = [(f, c) for f, c in _patch_files(patch) if include is None or fnmatch.fnmatch(f, include)]
    if not files:
        return None
    for f, created in files:
        if _git(tree, "-R", "--check", f"--include={f}", patch).returncode == 0:
            continue
        if created and os.path.isfile(os.path.join(tree, f)):
            edited.append(f)
            continue
        return None
    return edited or None


def apply(tree: str, patches: list[str], include: str | None = None, check: bool = False) -> list[str]:
    inc = [f"--include={include}"] if include else []
    out = []
    for p in patches:
        p = os.path.abspath(p)
        name = os.path.basename(p)
        if _git(tree, "-R", "--check", *inc, p).returncode == 0:
            out.append(f"{name}: already applied")
            continue
        fwd = _git(tree, "--check", *inc, p)
        if fwd.returncode != 0:
            edited = _applied_with_edited_new_files(tree, p, include)
            if edited is not None:
                out.append(f"{name}: already applied (files it creates were edited later: {', '.join(edited)})")
                continue
            raise SystemExit(f"git_series_hooks: {name} applies in neither direction on {tree}: {fwd.stderr.strip()[:400]}")
        if check:
            out.append(f"{name}: anchor ok")
            continue
        r = _git(tree, *inc, p)
        if r.returncode != 0:
            raise SystemExit(f"git_series_hooks: {name} failed to apply: {r.stderr.strip()[:400]}")
        out.append(f"{name}: applied")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/sgl-workspace/sglang")
    ap.add_argument("--include", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("patches", nargs="+")
    a = ap.parse_args(argv)
    for line in apply(a.tree, a.patches, a.include, a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
