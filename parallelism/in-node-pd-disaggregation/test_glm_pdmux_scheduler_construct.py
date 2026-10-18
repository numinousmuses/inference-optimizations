"""CPU test that CONSTRUCTS the hooked v0.5.20 Scheduler through its pdmux init path (coordinator 2026-10-06 04:1x,
after the final run's GX0 died at boot on `'Scheduler' object has no attribute 'gpu_id'` in init_pdmux).

A child process gets a mocked SM90 GPU and stubbed GPU-only packages, the same way the preflight resolver does
(hopper_preflight_cpu._CHILD). In the child:
  1. resolve the server args of the GX0 arm itself: its flags on the hooked tree (hopper_cells.patched_src: every hook
     of the arm applied as the boot applies them);
  2. publish them as the scheduler process does (run_scheduler_process, scheduler.py:5805-5808);
  3. construct Scheduler(server_args, port_args, gpu_id=5, tp_rank=5, ...) through the real __init__.
     - The init_* steps before init_pdmux that need a GPU or IPC (model config, metrics, IPC, idle sleeper, ZBAL,
       watchdog) are no-ops.
     - initialize_stream_groups is a recorder, since green contexts need a GPU.
     - init_tokenizer, the first step after init_pdmux, raises a sentinel.
     So the constructor runs every line up to and including init_pdmux.
  4. check init_pdmux ran with ps.gpu_id = 5 and stream groups, and that the hooked event_loop_pdmux hands over to the
     DP loop.
The negative control runs the same child on the STOCK v0.5.20 tree and must reproduce the boot crash."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hopper"))

CHILD = r'''
import importlib.abc, importlib.machinery, json, os, sys, types
import torch
SRC, argv = sys.argv[1], json.loads(sys.argv[2])
MEM = 143771 << 20
class Props:
    name = "NVIDIA H200"; total_memory = MEM; major = 9; minor = 0
    multi_processor_count = 132; L2_cache_size = 50 << 20; uuid = "GPU-mock"; shared_memory_per_block_optin = 232448
torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 8
torch.cuda.get_device_capability = lambda d=None: (9, 0)
torch.cuda.get_device_properties = lambda d=None: Props()
torch.cuda.get_device_name = lambda d=None: Props.name
torch.cuda.mem_get_info = lambda d=None: (MEM - (600 << 20), MEM)
torch.cuda.current_device = lambda: 0
torch.version.cuda = "12.9"
STUB = {"flashinfer", "sgl_kernel", "deep_gemm", "tilelang", "cutlass", "flash_mla", "vllm", "aiter", "mooncake",
        "nixl", "triton_kernels", "flashkda", "helion", "quack", "decord", "humming"}
class _Stub(types.ModuleType):
    __path__ = []
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        m = _Stub(self.__name__ + "." + n); setattr(self, n, m); return m
    def __call__(self, *a, **k): return _Stub(self.__name__ + "()")
    def __iter__(self): return iter(())
    def __mro_entries__(self, bases): return (object,)
class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] not in STUB:
            return None
        for f in sys.meta_path:
            if f is self: continue
            sp = getattr(f, "find_spec", None)
            try:
                if sp and sp(name, path, target): return None
            except Exception:
                pass
        return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec): return _Stub(spec.name)
    def exec_module(self, m): pass
sys.meta_path.insert(0, _Finder())
sys.path.insert(0, SRC)
out = {"stage": "resolve"}
try:
    from sglang.srt.server_args import prepare_server_args
    sa = prepare_server_args(argv)
    sa.resolve_once()
    out["stage"] = "publish"
    from sglang.srt.runtime_context import publish
    publish(sa, role="scheduler")
    out["stage"] = "import scheduler"
    import sglang.srt.managers.scheduler as S
    import sglang.srt.multiplex.multiplexing_mixin as MM
    import sglang.srt.multiplex.pdmux_context as PC
    rec = {}
    def fake_groups(gpu_id, cfg):
        rec["gpu_id"] = gpu_id
        rec["sm_group_num"] = cfg.sm_group_num
        PC.STREAM_GROUPS = [("p0", "d0"), ("p1", "d1"), ("p2", "d2")]
        PC.SM_COUNTS = [(132, 0), (72, 60), (0, 132)]
    MM.initialize_stream_groups = fake_groups
    class Reached(Exception):
        pass
    for name in ("init_startup_timing_begin", "init_soft_watchdog", "init_model_config", "init_metrics_collector",
                 "init_ipc_channels", "init_idle_sleeper", "init_zbal_on_npu"):
        setattr(S.Scheduler, name, lambda self, *a, **k: None)
    def stop(self, *a, **k):
        raise Reached()
    S.Scheduler.init_tokenizer = stop
    port_args = types.SimpleNamespace(nccl_port=0)
    out["stage"] = "construct"
    sched = S.Scheduler.__new__(S.Scheduler)
    try:
        S.Scheduler.__init__(sched, sa, port_args, gpu_id=5, tp_rank=5, moe_ep_rank=5, pp_rank=0, attn_cp_rank=0,
                             moe_dp_rank=0, dp_rank=5)
        out["reached_init_tokenizer"] = False
    except Reached:
        out["reached_init_tokenizer"] = True
    out.update(enable_pdmux=sched.enable_pdmux, ps_gpu_id=sched.ps.gpu_id, groups_gpu_id=rec.get("gpu_id"),
               sm_group_num=rec.get("sm_group_num"), stream_groups=len(getattr(sched, "stream_groups", []) or []),
               split_prefill_batch_none=getattr(sched, "split_prefill_batch", "missing") is None)
    out["stage"] = "delegate"
    import sglang.srt.multiplex.numinous_pdmux_dp as DP
    called = []
    DP.event_loop_pdmux_dp = lambda s: called.append(s)
    os.environ["NUMINOUS_PDMUX_DP"] = "1"
    sched.require_mlp_sync = True
    sched.event_loop_pdmux()
    out["delegated"] = called == [sched]
    out["ok"] = True
except BaseException as e:
    out["ok"] = False
    out["error"] = f"{type(e).__name__}: {str(e)[:400]}"
print("@@JSON@@" + json.dumps(out))
'''


def _run_child(src: str, argv: list, env: dict) -> dict:
    e = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", **env)
    e["SGLANG_CACHE_DIR"] = os.path.expanduser("~/.cache/numinous/preflight_sglang_cache")
    p = subprocess.run([sys.executable, "-c", CHILD, src, json.dumps(argv)], capture_output=True, text=True,
                       timeout=900, env=e)
    for line in p.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    raise AssertionError(f"child died: {(p.stderr or p.stdout)[-2000:]}")


@pytest.fixture(scope="module")
def gx0():
    import glm_hopper_cells as g
    import hopper_cells as hc
    import hopper_preflight_cpu as hp
    if not pathlib.Path(hc.SGL_SRC).is_dir():
        pytest.skip("needs operations/hopper/restore_cpu_preflight.sh (CPU v0.5.20 tree)")
    cell = g.arm_gx0_fix()
    cfg = cell["config"]
    mdir = hc.RESOLVE_CFG_BY_REPO.get(cfg["repo"]) or hc.RESOLVE_CFG[cell["model"]]
    mdir = hp.model_dir(mdir) if mdir in hp.CONFIGS else mdir
    env = {k: v for k, v in (cfg.get("env") or {}).items() if not k.startswith("NUMINOUS_SCHED_TRACE")}
    flags = list(cfg["flags"])
    # the container path (/root/numinous/...) -> this repo's copy of the same file
    flags[flags.index("--pdmux-config-path") + 1] = str(HERE / "glm_pdmux" / "glm_pdmux.yaml")
    return cell, hc.patched_src(cfg), ["--model-path", mdir] + flags, env, hc


def test_hooked_scheduler_constructs_through_init_pdmux(gx0):
    cell, src, argv, env, _hc = gx0
    r = _run_child(src, argv, env)
    assert r.get("ok"), r
    assert r["reached_init_tokenizer"] and r["enable_pdmux"]
    assert r["ps_gpu_id"] == 5 and r["groups_gpu_id"] == 5 and r["sm_group_num"] == 3
    assert r["stream_groups"] == 3 and r["split_prefill_batch_none"]
    assert r["delegated"], "the hooked event_loop_pdmux did not hand over to the DP loop"


def test_stock_tree_reproduces_the_boot_crash(gx0):
    """Negative control: the same constructor path on the stock v0.5.20 tree fails exactly as GX0 did at 04:01."""
    cell, _src, argv, env, hc = gx0
    a = list(argv)
    a[a.index("--chunked-prefill-size") + 1] = "-1"      # stock validation refuses chunked prefill under pdmux
    r = _run_child(str(pathlib.Path(hc.SGL_SRC)), a, {k: v for k, v in env.items() if not k.startswith("NUMINOUS")})
    assert not r.get("ok") and r["stage"] == "construct" and "gpu_id" in r.get("error", ""), r
