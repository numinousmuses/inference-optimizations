"""kv4mla: native NVFP4 sparse MLA attention for GLM-5.3's FP4 DSA pool (SGLang v0.5.20), as anchored hooks.

Builds on kv4glm (operations/patches/glm_kv4_hooks.py: FP4 pool, write door, attach, route). `apply`:
  1. applies the glm_kv4 hooks and installs its modules,
  2. installs operations/kernels/kv4mla/{kv4mla.py, csrc/} -> sglang/srt/layers/attention/dsa/kv4mla/ and
     operations/kernels/kv4mla/kv4mla_engine.py -> sglang/srt/layers/attention/dsa/kv4mla_engine.py,
  3. applies two hooks on top of the glm_kv4 ones in dsa_backend.py:
       kv4mla_route  inside glm_kv4_route: NUMINOUS_GLM_KV4MLA=1 sends the attention call to kv4mla_engine
       kv4mla_graph  inside glm_kv4_graph: build the native schedules instead of reserving the FP8 graph scratch
  4. compiles the CUDA extension (no GPU needed; --no-build to skip), so engine processes load it from the cache.

Inert unless NUMINOUS_GLM_KV4=1 and NUMINOUS_GLM_KV4MLA=1 (then kv4glm's scratch path is replaced for decode / verify /
draft extend; plain prefill keeps kv4glm's sequence path).

    python3 kv4mla_hooks.py                  # patch the installed sglang + build
    python3 kv4mla_hooks.py --root /tmp/sgl --no-build
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import glm_kv4_hooks as base  # noqa: E402

BACKEND = base.BACKEND
KDIR = REPO / "operations/kernels/kv4mla"
PKG_DIR = "srt/layers/attention/dsa/kv4mla"
ENGINE_DST = "srt/layers/attention/dsa/kv4mla_engine.py"

ROUTE_ANCHOR = '''            return _nkv4.engine_forward(
                self, q_all, page_table_1, sparse_mla_top_k, layer, forward_batch, metadata, seq_lens, is_prefill
            )
'''
ROUTE = '''            # numinous-hook:kv4mla_route
            from sglang.srt.layers.attention.dsa import kv4mla_engine as _nmla

            if _nmla.enabled():
                return _nmla.engine_forward(
                    self, q_all, page_table_1, sparse_mla_top_k, layer, forward_batch, metadata, seq_lens, is_prefill
                )
''' + ROUTE_ANCHOR

GRAPH_ANCHOR = "            _nkv4.reserve_graph_scratch(self, max_num_tokens)\n"
GRAPH = '''            # numinous-hook:kv4mla_graph
            from sglang.srt.layers.attention.dsa import kv4mla_engine as _nmla

            if not (_nmla.enabled() and _nmla.prepare_graph(self, max_num_tokens)):
                _nkv4.reserve_graph_scratch(self, max_num_tokens)
'''

HOOKS = [
    base.Hook("kv4mla_route", BACKEND, ROUTE_ANCHOR, ROUTE, replace=True),
    base.Hook("kv4mla_graph", BACKEND, GRAPH_ANCHOR, GRAPH, replace=True),
]


def patch_source(src: str) -> tuple[str, list[str]]:
    out, missing = [], []
    for h in HOOKS:
        if h.marker + "\n" in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        src = src.replace(h.anchor, h.text if h.replace else h.anchor + h.text, 1)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("kv4mla hooks: " + "; ".join(missing))
    return src, out


def install_modules(root: str) -> list[str]:
    dst = pathlib.Path(root) / PKG_DIR
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "csrc").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(KDIR / "csrc", dst / "csrc", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(KDIR / "kv4mla.py", dst / "kv4mla.py")
    (dst / "__init__.py").write_text("")
    shutil.copy(KDIR / "kv4mla_engine.py", pathlib.Path(root) / ENGINE_DST)
    return [f"installed {PKG_DIR}/ (kv4mla.py, csrc)", f"installed {ENGINE_DST}"]


def apply(root: str, check: bool = False, build: bool = True) -> list[str]:
    lines = base.apply(root, check=check)
    full = os.path.join(root, BACKEND)
    src = open(full).read()
    if check:
        # the kv4mla anchors live in text the glm_kv4 hooks insert: check against the base-patched source
        src, _ = base.patch_source(src, BACKEND)
        _, ls = patch_source(src)
        return lines + [ln.replace("applied to", "anchor ok in") for ln in ls]
    new, ls = patch_source(src)
    compile(new, full, "exec")
    if new != src:
        tmp = f"{full}.numinous.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, full)
    lines += ls + install_modules(root)
    if build:
        mod_path = pathlib.Path(root) / PKG_DIR / "kv4mla.py"
        code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('kv4mla_build', %r); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print('built', m.load().__file__)"
                % str(mod_path))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("kv4mla: extension build failed:\n" + r.stdout[-3000:] + r.stderr[-6000:])
        lines.append(r.stdout.strip().splitlines()[-1])
    return lines


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or base.sglang_root(), a.check, build=not a.no_build):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
