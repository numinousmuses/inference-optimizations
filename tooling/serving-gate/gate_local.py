"""cell_gate arms on this machine (Daytona B300 / MI355X boxes), no Modal.

The box runs one engine image. Before each arm's server starts, the engine checkout is reset and the arm's patch
series (numinous-kernels at its pinned ref) and overlays (module files, then the anchored hook patcher) are applied;
cell_gate's own _run_seq then runs the arms back to back on this node, alternating order per repeat.

    GATE_CACHE=/root/cache python3 operations/modal/gate_local.py --cell glm-5.3/standard --arms glm_v20,glm_v20_k \\
        --ladder 64,96 --repeats 2 [--probes] [--sessions real_sessions_t8_300.json] [--profile-steps 30]

Expects on the box: this repo at /root/work, numinous-kernels (with .git) at /root/numinous-kernels, the sessions
file under operations/modal/data/. Evidence goes to research/gpu-evidence/gate/ like cell_gate's.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

os.environ["GATE_LOCAL"] = "1"
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import cell_gate as cg  # noqa: E402
from gate_arms import ARMS  # noqa: E402

KERNELS = pathlib.Path(os.environ.get("GATE_KERNELS", "/root/numinous-kernels"))


def engine_root(pkg: str = "sglang") -> pathlib.Path:
    import importlib.util
    return pathlib.Path(list(importlib.util.find_spec(pkg).submodule_search_locations)[0])


def checkout_of(pkg_dir: pathlib.Path) -> pathlib.Path:
    """The git checkout the editable engine install lives in (/sgl-workspace/sglang for the SGLang images)."""
    r = subprocess.run(["git", "-C", str(pkg_dir), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{pkg_dir} is not in a git checkout: cannot reset between arms")
    return pathlib.Path(r.stdout.strip())


def reset_engine(repo: pathlib.Path) -> None:
    """Back to the image's checkout: tracked files restored, and the untracked files a patch series creates removed
    (the decode-burst series adds python/sglang/srt/managers/decode_burst.py and test/numinous/*)."""
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "clean", "-qfd", "python/sglang", "test"], check=True)


def apply_patches(repo: pathlib.Path, pkg_dir: pathlib.Path, patches: list) -> list[str]:
    done = []
    for p in patches or []:
        for f in p["files"]:
            src = f.split(":")[0]
            if p["ref"] == "local":
                # a series kept in this repo (operations/patches/<series>/; see registry_image.materialize)
                blob = (ROOT / "operations/patches" / p["series"] / src).read_bytes()
            else:
                blob = subprocess.run(["git", "-C", str(KERNELS), "show", f"{p['ref']}:patches/{p['series']}/{src}"],
                                      capture_output=True, check=True).stdout
            if p.get("overlay"):
                dst = pkg_dir / f.split(":")[1]
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(blob)
            else:
                tmp = pathlib.Path("/tmp") / src
                tmp.write_bytes(blob)
                subprocess.run(["git", "-C", str(repo), "apply", "--check", str(tmp)], check=True)
                subprocess.run(["git", "-C", str(repo), "apply", str(tmp)], check=True)
            done.append(f"{p['series']}/{src}")
    return done


def apply_overlays(pkg_dir: pathlib.Path, overlays: dict) -> list[str]:
    done = []
    for local, dst in sorted(overlays.items(), key=lambda kv: kv[1].startswith("!run")):
        if dst.startswith("!run"):
            subprocess.run([sys.executable, str(ROOT / local), *dst[4:].split()], check=True)
        else:
            (pkg_dir / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / local, pkg_dir / dst)
        done.append(f"{local} -> {dst}")
    return done


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--arms", default="control")
    ap.add_argument("--base", default="")
    ap.add_argument("--ladder", default="64")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--profile-steps", type=int, default=0)
    ap.add_argument("--sessions", default="")
    ap.add_argument("--image", default=os.environ.get("BOX_IMAGE", ""), help="the image this box runs")
    a = ap.parse_args(argv)

    cell = cg.cell_by_key(a.cell)
    base = [x for x in a.base.split(",") if x in ARMS]
    arms = [x for x in a.arms.split(",") if x in ARMS]
    unknown = set(a.arms.split(",")) - set(arms)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")
    deltas = {x: cg._compose(base + [x]) for x in arms}
    for x, d in deltas.items():
        img = d.get("image") or cell["config"]["image"]
        if a.image and img != a.image:
            raise SystemExit(f"arm {x} runs {img}; this box runs {a.image}")
    sessions = a.sessions or cg.SESSIONS.get(a.cell, cg.SESSIONS["default"])
    shutil.copy(HERE / sessions, f"/root/{pathlib.Path(sessions).name}")
    shutil.copy(HERE / "dc_bench.py", "/root/dc_bench.py")
    shutil.copy(cg.REGISTRY, "/root/serving_registry.json")

    pkg_dir = engine_root("vllm" if cell["config"]["engine"] == "vllm" else "sglang")
    repo = checkout_of(pkg_dir)
    original = cg._run_arm

    def prepared(cell_, arm, delta, *args, **kw):
        reset_engine(repo)
        cfg_patches = delta["patches"] if "patches" in delta else cell_["config"].get("patches")
        pl = cfg_patches if isinstance(cfg_patches, list) else ([cfg_patches] if cfg_patches else [])
        applied = apply_patches(repo, pkg_dir, pl) + apply_overlays(pkg_dir, delta.get("overlay") or {})
        print(f"[{arm}] engine prepared: {len(applied)} patch/overlay steps", flush=True)
        try:
            return original(cell_, arm, delta, *args, **kw)
        finally:
            reset_engine(repo)

    cg._run_arm = prepared
    ladder = [int(x) for x in a.ladder.split(",")]
    extra = {"_same_sessions": os.environ.get("GATE_SAME_SESSIONS") == "1",
             "_probe_repeats": int(os.environ.get("GATE_PROBE_REPEATS", "1")),
             "_monitor": os.environ.get("GATE_MONITOR") == "1",
             "_max_chars": int(os.environ.get("GATE_MAX_CHARS", "600000")),
             "_gpu_label": os.environ.get("GATE_GPU_LABEL", "box"), "_stagger_s": 0.0, "_nsys": False}
    labels = {x: "+".join(base + [x]) for x in arms}
    # GATE_ACCURACY=1 (as cell_gate on Modal): every arm records the accuracy gate after boot into one directory; the
    # first arm is the control and the others are compared with it at the end
    acc = os.environ.get("GATE_ACCURACY") == "1"
    if acc:
        acc_dir = os.path.join(cg.CACHE, "accuracy",
                               f"{a.cell.replace('/', '_')}_{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}")
        extra.update(_acc_dir=acc_dir, _acc_control=labels[arms[0]])
        print(f"accuracy gate: control {labels[arms[0]]}, records in {acc_dir}", flush=True)
    t0 = time.time()
    res = cg._run_seq(cell, [(labels[x], {**deltas[x], **extra}) for x in arms], ladder, a.repeats, sessions,
                      a.probes, a.profile_steps)
    out_dir = ROOT / "research/gpu-evidence/gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path = out_dir / f"{a.cell.replace('/', '_')}_{extra['_gpu_label']}_local_{stamp}.json"
    path.write_text(json.dumps({"cell": a.cell, "arms": arms, "base": base, "ladder": ladder, "repeats": a.repeats,
                                "minutes": round((time.time() - t0) / 60, 1), "results": res}, indent=1,
                               default=str))
    print(f"evidence: {path}", flush=True)
    if acc:
        import accuracy_gate as ag
        load = lambda lb: json.load(open(p)) if os.path.exists(p := os.path.join(acc_dir, f"accuracy_{lb}.json")) else None  # noqa: E731
        ctl = load(labels[arms[0]])
        noise_p = os.path.join(acc_dir, "accuracy_noise.json")
        noise = json.load(open(noise_p)) if os.path.exists(noise_p) else None
        cmp = {labels[x]: (ag.compare(ctl, rec, noise) if ctl and (rec := load(labels[x])) else {"error": "missing record"})
               for x in arms[1:]}
        apath = out_dir / f"{a.cell.replace('/', '_')}_accuracy_{stamp}.json"
        apath.write_text(json.dumps({"control": labels[arms[0]], "arms": cmp}, indent=1, default=str))
        for lb, v in cmp.items():
            print(f"[accuracy] {lb} vs {labels[arms[0]]}: {json.dumps(v.get('verdict', v), default=str)[:300]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
