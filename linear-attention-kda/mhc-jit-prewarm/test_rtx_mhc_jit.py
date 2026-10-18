"""CPU tests: GLM mHC JIT stall fix (prewarm + pow2 split buckets) and the per-step scheduler trace."""
import json
import os
import shutil
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kernels"))
sys.path.insert(0, HERE)
import numinous_mhc_jit as mj  # noqa: E402
import numinous_step_trace as st  # noqa: E402
import rtx_mhc_jit_hooks as hooks  # noqa: E402

V0520 = os.environ.get("SGLANG_V0520", "/tmp/sgl-v0520-wt/python/sglang")


def upstream_split(num_tokens, hc_hidden_size, n_sms=188):
    """v0.5.20 _compute_num_split_for_mhc_pre with the RTX PRO 6000's SM count."""
    grid = (num_tokens + 63) // 64
    nbk = (hc_hidden_size + 63) // 64
    return max(1, min(n_sms // max(grid, 1), nbk // 4))


def test_bucket_counts_on_rtx():
    # GLM-5.3-Flash: hidden 4096 x hc_mult 4; per-rank chunk 4096 (v3) / 8192 (v2)
    b = mj.bucket_token_counts(4096, 16384, upstream_split)
    assert len(b) == 24, b  # the compile count a cold rank can hit, one ~11.6 s compile each (measured)
    pow2 = mj.bucket_token_counts(4096, 16384, lambda t, k: mj.nsplit_pow2(upstream_split(t, k)))
    assert len(pow2) == 6  # {64,32,16,8,4,2}; 1 appears only above ~6K tokens
    assert len(mj.bucket_token_counts(8192, 16384, lambda t, k: mj.nsplit_pow2(upstream_split(t, k)))) == 7
    assert [mj.nsplit_pow2(n) for n in (1, 2, 3, 47, 62, 64)] == [1, 2, 2, 32, 32, 64]


class _Layer(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.config = types.SimpleNamespace(mhc=True, hc_mult=4, hidden_size=4096, rms_norm_eps=1e-5)
        self.is_linear_attn = linear
        self.hc_attn_fn = torch.nn.Parameter(torch.zeros(24, 16384))
        self.input_layernorm = types.SimpleNamespace(weight=types.SimpleNamespace(data=torch.ones(4096)),
                                                     variance_epsilon=1e-5)
        self.calls = []

    def hc_attn_pre(self, x, w, eps):
        self.calls.append((x.shape[0], w is not None, eps))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(True), _Layer(True), _Layer(False)])


def _fake_mhc(monkeypatch, split):
    mod = types.ModuleType("sglang.kernels.ops.layernorm.mhc")
    mod._compute_num_split_for_mhc_pre = split
    pkg = types.ModuleType("sglang.kernels.ops.layernorm")
    pkg.mhc = mod
    for name, m in (("sglang", types.ModuleType("sglang")), ("sglang.kernels", types.ModuleType("sglang.kernels")),
                    ("sglang.kernels.ops", types.ModuleType("sglang.kernels.ops")),
                    ("sglang.kernels.ops.layernorm", pkg), ("sglang.kernels.ops.layernorm.mhc", mod)):
        monkeypatch.setitem(sys.modules, name, m)


def test_prewarm_covers_every_bucket_once_per_kind(monkeypatch):
    _fake_mhc(monkeypatch, upstream_split)
    monkeypatch.setattr(mj, "_PREWARMED", False)
    m = _Model()
    monkeypatch.delenv("NUMINOUS_MHC_PREWARM", raising=False)
    assert mj.prewarm_glm5next(m, max_num_tokens=4096) == {"skipped": True}  # env off: no-op
    monkeypatch.setenv("NUMINOUS_MHC_PREWARM", "1")
    rep = mj.prewarm_glm5next(m, max_num_tokens=4096)
    assert rep["layers"] == 2 and rep["calls"] == 2 * 24 * 2  # 2 kinds x 24 buckets x (no norm, fused norm)
    lin, dsa = m.layers[0], m.layers[2]
    assert m.layers[1].calls == []  # one representative per kind
    for layer in (lin, dsa):
        ts = sorted({c[0] for c in layer.calls})
        assert ts == list(mj.bucket_token_counts(4096, 16384, upstream_split))
        assert {(c[1], c[2]) for c in layer.calls} == {(False, None), (True, 1e-5)}
        # every bucket's split value is hit, i.e. every kernel specialization compiled
        assert {upstream_split(t, 16384) for t in ts} == {upstream_split(t, 16384) for t in range(1, 4097)}
    assert mj.prewarm_glm5next(m, max_num_tokens=4096) == {"skipped": True}  # once per process


@pytest.mark.skipif(not os.path.isdir(V0520), reason="no v0.5.20 tree")
def test_hooks_apply_compile_and_gate(tmp_path):
    root = tmp_path / "sglang"
    for rel in ("srt/models/glm5_next.py", "kernels/ops/layernorm/mhc.py", "srt/managers/scheduler.py"):
        (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(V0520, rel), root / rel)
    (root / "srt/layers").mkdir(parents=True, exist_ok=True)
    hooks.apply(str(root))
    for rel in ("srt/models/glm5_next.py", "kernels/ops/layernorm/mhc.py", "srt/managers/scheduler.py"):
        compile((root / rel).read_text(), rel, "exec")
    assert all(x.endswith("already applied") for x in hooks.apply(str(root))[:3])
    # the patched split function: stock unless NUMINOUS_MHC_NSPLIT_POW2=1
    src = (root / "kernels/ops/layernorm/mhc.py").read_text()
    start = src.index("def _compute_num_split_for_mhc_pre")
    end = src.index("\ndef ", start + 10)
    ns = {"torch": types.SimpleNamespace(cuda=types.SimpleNamespace(
        get_device_properties=lambda i: types.SimpleNamespace(multi_processor_count=188)))}
    exec(src[start:end], ns)
    f = ns["_compute_num_split_for_mhc_pre"]
    os.environ.pop("NUMINOUS_MHC_NSPLIT_POW2", None)
    assert [f(t, 16384) for t in (64, 640, 3000, 4096)] == [upstream_split(t, 16384) for t in (64, 640, 3000, 4096)]
    os.environ["NUMINOUS_MHC_NSPLIT_POW2"] = "1"
    try:
        assert [f(t, 16384) for t in (64, 640, 3000, 4096)] == [64, 16, 4, 2]
    finally:
        os.environ.pop("NUMINOUS_MHC_NSPLIT_POW2")


def test_step_trace_writes_per_phase_records(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMINOUS_STEP_TRACE", str(tmp_path))
    monkeypatch.setattr(st, "_fh", None)
    monkeypatch.setattr(st, "_rank", None)
    st._buf.clear()

    class Sched:
        ps = types.SimpleNamespace(dp_rank=2)

        def ingest_requests(self):
            return None

        def get_next_batch_to_run(self, running_batch=None, last_batch=None):
            b = types.SimpleNamespace(forward_mode="ForwardMode.EXTEND", reqs=[1, 2], extend_num_tokens=4096)
            return types.SimpleNamespace(batch_to_run=b)

        def run_batch(self, batch):
            return "r"

        def process_batch_result(self, batch, result):
            return None

    assert set(st.install(Sched)) == set(st.PHASES)
    assert st.install(Sched) == []  # idempotent
    s = Sched()
    for _ in range(3):
        s.ingest_requests()
        plan = s.get_next_batch_to_run()
        s.process_batch_result(plan.batch_to_run, s.run_batch(plan.batch_to_run))
    st._flush(True)
    files = list(tmp_path.glob("steps_dp2_*.jsonl"))
    assert len(files) == 1
    recs = [json.loads(x) for x in files[0].read_text().splitlines()]
    assert len(recs) == 12 and {r["ph"] for r in recs} == set(st.PHASES)
    rb = [r for r in recs if r["ph"] == "run_batch"]
    assert rb[0]["mode"] == "EXTEND" and rb[0]["tok"] == 4096 and rb[0]["bs"] == 2


@pytest.mark.skipif(not os.path.isdir(V0520), reason="no v0.5.20 tree")
def test_prewarm_hook_is_on_the_real_load_path(tmp_path):
    """Regression (composed run g156o, 2026-10-01): the first anchor sat in Glm5NextForConditionalGeneration.
    post_load_weights, which load_weights never calls (it calls the mixin's post_load_weights directly), so the
    prewarm never ran. The hook must sit inside load_weights of the EntryClass, after the run_post call, at the
    method's top indentation (not inside the `if run_post:` block), and the method must not call
    self.post_load_weights."""
    import ast

    root = tmp_path / "sglang"
    for rel in ("srt/models/glm5_next.py", "kernels/ops/layernorm/mhc.py", "srt/managers/scheduler.py"):
        (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(V0520, rel), root / rel)
    (root / "srt/layers").mkdir(parents=True, exist_ok=True)
    hooks.apply(str(root))
    src = (root / "srt/models/glm5_next.py").read_text()
    tree = ast.parse(src)
    entry = [ln for ln in src.splitlines() if ln.startswith("EntryClass")]
    assert entry == ["EntryClass = [Glm5NextForConditionalGeneration]"]
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Glm5NextForConditionalGeneration")
    lw = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "load_weights")
    seg = ast.get_source_segment(src, lw)
    assert "numinous-hook:glm5next_mhc_prewarm" in seg
    assert "self.post_load_weights(" not in seg  # i.e. the method-level hook site would be dead code
    calls = [n for n in ast.walk(lw) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "prewarm_glm5next"]
    assert len(calls) == 1
    # the prewarm `if` is a direct statement of load_weights (runs whether or not run_post was taken)
    top_ifs = [s for s in lw.body if isinstance(s, ast.If) and "NUMINOUS_MHC_PREWARM" in ast.get_source_segment(src, s)]
    assert len(top_ifs) == 1
