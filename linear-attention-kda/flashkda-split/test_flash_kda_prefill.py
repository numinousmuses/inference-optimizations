"""CPU tests for flash_kda_prefill.py (L4): the tracked split plan reproduces the stock tracked path exactly. An fp64
torch KDA recurrence stands in for both FlashKDA and Triton chunk_kda. Plus the hook anchors on the v0.5.20 tree."""
import os
import pathlib
import shutil
import sys
import types

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "patches"))
import flash_kda_prefill as fkp  # noqa: E402
import flash_kda_flashkda_hooks as H  # noqa: E402

SRC = pathlib.Path(os.environ.get("SGLANG_SRC", "/tmp/sgl-v0520-wt/python")) / "sglang"
NH, D, LB = 2, 8, -5.0


def _step_all(q, k, v, g, beta, A_log, dt_bias, S, raw_beta=True):
    """fp64 recurrence over one sequence. q,k,v,g [T,H,D]; beta [T,H]; S [H,V,K] (updated copy returned)."""
    qn = q / q.norm(dim=-1, keepdim=True) * D ** -0.5
    kn = k / k.norm(dim=-1, keepdim=True)
    ga = LB * torch.sigmoid(torch.exp(A_log).view(NH, 1) * (g + dt_bias.view(NH, D)))
    b = torch.sigmoid(beta) if raw_beta else beta
    o = torch.empty_like(v)
    states = []
    for t in range(q.shape[0]):
        S = S * torch.exp(ga[t]).unsqueeze(1)
        u = b[t].unsqueeze(-1) * (v[t] - torch.einsum("hvk,hk->hv", S, kn[t]))
        S = S + u.unsqueeze(-1) * kn[t].unsqueeze(1)
        o[t] = torch.einsum("hvk,hk->hv", S, qn[t])
        states.append(S)
    return o, S, states


def fake_flash_kda():
    def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lb, initial_state=None, final_state=None, cu_seqlens=None):
        cu = cu_seqlens.tolist()
        for j in range(len(cu) - 1):
            a, b = cu[j], cu[j + 1]
            o, S, _ = _step_all(q[0, a:b], k[0, a:b], v[0, a:b], g[0, a:b], beta[0, a:b], A_log, dt_bias.reshape(-1),
                                initial_state[j])
            out[0, a:b] = o
            final_state[j] = S
    return types.SimpleNamespace(fwd=fwd)


def stock_triton_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, A_log=None, dt_bias=None,
                        lower_bound=None, extend_seq_lens_cpu=None, beta_is_raw=False, return_intermediate_states=False,
                        track_state=None, track_chunk_idx=None, **kw):
    """Reference for the stock tracked path: per request, snapshot at track_chunk_idx * 64 into track_state."""
    out = torch.empty_like(v)
    cu = query_start_loc.tolist()
    for i in range(len(cu) - 1):
        a, b = cu[i], cu[i + 1]
        S0 = ssm_states[cache_indices[i]].clone()
        o, S, states = _step_all(q[0, a:b], k[0, a:b], v[0, a:b], g[0, a:b], beta[0, a:b], A_log, dt_bias, S0,
                                 raw_beta=beta_is_raw)
        out[0, a:b] = o
        ssm_states[cache_indices[i]] = S
        if track_state is not None and track_chunk_idx is not None and int(track_chunk_idx[i]) >= 0:
            p = 64 * int(track_chunk_idx[i])
            track_state[i] = states[p - 1] if p > 0 else S0
    return (out, None) if return_intermediate_states else out


def _batch(lens, seed=0):
    g0 = torch.Generator().manual_seed(seed)
    T = sum(lens)
    mk = lambda *s: torch.randn(*s, generator=g0, dtype=torch.float64)  # noqa: E731
    q, k, v, g = (mk(1, T, NH, D) for _ in range(4))
    beta = mk(1, T, NH)
    A_log = torch.log(torch.rand(NH, generator=g0, dtype=torch.float64) * 15 + 1)
    dt_bias = mk(NH * D) * 0.1
    pool = mk(len(lens) + 3, NH, D, D) * 0.05
    ci = torch.arange(1, 1 + len(lens))
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()))
    return q, k, v, g, beta, A_log, dt_bias, pool, ci, cu


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    monkeypatch.setitem(fkp._FK, "mod", fake_flash_kda())
    monkeypatch.setitem(fkp._FK, "err", None)


def test_plan_splits_only_interior_points():
    segs, snap0 = fkp.plan([100, 64, 300, 7, 130], [1, -1, 4, -1, 0])
    assert [(a, b, kd) for a, b, kd, _ in segs] == [(0, 64, "head"), (64, 100, "tail"), (100, 164, "full"),
                                                    (164, 420, "head"), (420, 464, "tail"), (464, 471, "full"),
                                                    (471, 601, "full")]
    assert snap0 == [4]


@pytest.mark.parametrize("lens,track", [([100, 64, 300, 7], [1, -1, 4, -1]), ([129, 70], [2, 1]), ([65], [-1])])
def test_split_matches_stock_tracked_path(lens, track):
    q, k, v, g, beta, A_log, dt_bias, pool, ci, cu = _batch(lens)
    tci = torch.tensor(track, dtype=torch.int32)
    ref_pool, ref_snap = pool.clone(), torch.zeros(len(lens), NH, D, D, dtype=torch.float64)
    o_ref, _ = stock_triton_extend(q, k, v, g, beta, ssm_states=ref_pool, cache_indices=ci, query_start_loc=cu,
                                   A_log=A_log, dt_bias=dt_bias, lower_bound=LB, beta_is_raw=True,
                                   return_intermediate_states=True, track_state=ref_snap, track_chunk_idx=tci)
    my_pool, my_snap = pool.clone(), torch.zeros_like(ref_snap)
    dev_tci = tci.clone()
    fkp.stash_track_idx(dev_tci, tci)
    o, h = fkp.extend(q, k, v, g, beta, ssm_states=my_pool, cache_indices=ci, query_start_loc=cu, A_log=A_log,
                      dt_bias=dt_bias, lower_bound=LB, extend_seq_lens_cpu=lens, beta_is_raw=True,
                      return_intermediate_states=True, triton_extend=stock_triton_extend, track_state=my_snap,
                      track_chunk_idx=dev_tci)
    assert h is None
    assert torch.allclose(o, o_ref, atol=1e-10, rtol=1e-9)
    assert torch.allclose(my_pool, ref_pool, atol=1e-10, rtol=1e-9)
    rows = [i for i, t in enumerate(track) if t >= 0]
    assert torch.allclose(my_snap[rows], ref_snap[rows], atol=1e-10, rtol=1e-9)


def test_untracked_batch_returns_tensor_and_matches():
    lens = [200, 90]
    q, k, v, g, beta, A_log, dt_bias, pool, ci, cu = _batch(lens, seed=3)
    ref_pool = pool.clone()
    o_ref = stock_triton_extend(q, k, v, g, beta, ssm_states=ref_pool, cache_indices=ci, query_start_loc=cu,
                                A_log=A_log, dt_bias=dt_bias, lower_bound=LB, beta_is_raw=True)
    my_pool = pool.clone()
    o = fkp.extend(q, k, v, g, beta, ssm_states=my_pool, cache_indices=ci, query_start_loc=cu, A_log=A_log,
                   dt_bias=dt_bias, lower_bound=LB, extend_seq_lens_cpu=lens, beta_is_raw=True,
                   triton_extend=stock_triton_extend)
    assert torch.is_tensor(o) and torch.allclose(o, o_ref, atol=1e-10) and torch.allclose(my_pool, ref_pool, atol=1e-10)


def test_fallbacks(monkeypatch):
    lens = [100]
    q, k, v, g, beta, A_log, dt_bias, pool, ci, cu = _batch(lens, seed=5)
    calls = []

    def spy(*a, **kw):
        calls.append(kw.get("track_chunk_idx"))
        return stock_triton_extend(*a, **kw)
    snap = torch.zeros(1, NH, D, D, dtype=torch.float64)
    stale = torch.tensor([1], dtype=torch.int32)                       # never stashed: no CPU copy -> Triton
    fkp.stash_track_idx(None, torch.tensor([1]))
    fkp.extend(q, k, v, g, beta, ssm_states=pool.clone(), cache_indices=ci, query_start_loc=cu, A_log=A_log,
               dt_bias=dt_bias, lower_bound=LB, extend_seq_lens_cpu=lens, beta_is_raw=True,
               return_intermediate_states=True, triton_extend=spy, track_state=snap, track_chunk_idx=stale)
    fkp.extend(q, k, v, g, beta, ssm_states=pool.clone(), cache_indices=ci, query_start_loc=cu, A_log=A_log,
               dt_bias=dt_bias, lower_bound=None, extend_seq_lens_cpu=lens, beta_is_raw=True, triton_extend=spy)
    fkp.extend(q, k, v, g, beta, ssm_states=pool.clone(), cache_indices=ci, query_start_loc=cu, A_log=A_log,
               dt_bias=dt_bias, lower_bound=LB, extend_seq_lens_cpu=lens, beta_is_raw=True, is_spec_decode=True,
               triton_extend=spy)
    assert len(calls) == 3 and calls[0] is stale


@pytest.mark.skipif(not (SRC / H.HOOKS[0][1]).exists(), reason="SGLang v0.5.20 source absent")
def test_hooks_apply_once_and_compile(tmp_path):
    pkg = tmp_path / "sglang"
    for _, rel, _, _ in H.HOOKS:
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / rel, pkg / rel)
    (pkg / H.MODULE_DST).parent.mkdir(parents=True, exist_ok=True)
    assert H.apply(str(pkg), check=True) == ["extend: anchor ok", "stash: anchor ok"]
    assert H.apply(str(pkg))[-1] == f"{H.NAME}: installed {H.MODULE_DST}"
    assert H.apply(str(pkg), install_module=False) == ["extend: already applied", "stash: already applied"]
    for _, rel, _, _ in H.HOOKS:
        compile((pkg / rel).read_text(), rel, "exec")


def test_package_dir_prefers_volume_then_baked(tmp_path, monkeypatch):
    import json as _json
    vol, baked = tmp_path / "vol", tmp_path / "baked"
    for d in (vol, baked):
        (d / "flash_kda").mkdir(parents=True)
        (d / "flash_kda" / "__init__.py").write_text("")
        (d / "manifest.json").write_text(_json.dumps({"files": {"flash_kda/__init__.py": "x"}}))
    monkeypatch.setattr(fkp, "BAKED_DIR", str(baked))
    monkeypatch.setenv(fkp.DIR_ENV, str(vol))
    assert fkp.package_dir() == str(vol)
    monkeypatch.setenv(fkp.DIR_ENV, str(tmp_path / "missing"))
    assert fkp.package_dir() == str(baked)
    (baked / "flash_kda" / "__init__.py").unlink()                     # incomplete copy is not used
    assert fkp.package_dir() is None


def test_hook_ensure_package_builds_once_and_never_raises(tmp_path):
    calls = []

    def fake_run(c, **kw):
        calls.append(c[:3])
        if "--target" in c:
            tgt = c[c.index("--target") + 1]
            os.makedirs(os.path.join(tgt, "flash_kda"), exist_ok=True)
            open(os.path.join(tgt, "flash_kda", "__init__.py"), "w").write("x")
            open(os.path.join(tgt, "flash_kda_C.cpython-312-x86_64-linux-gnu.so"), "wb").write(b"\0")
        return types.SimpleNamespace(returncode=0, stderr="")
    dst = str(tmp_path / "opt" / "flash_kda-7afb9f45")
    msg = H.ensure_package(dst, run=fake_run)
    assert "built into" in msg and os.path.exists(os.path.join(dst, "manifest.json"))
    assert os.path.exists(os.path.join(dst, "flash_kda_C.cpython-312-x86_64-linux-gnu.so"))
    n = len(calls)
    assert "present" in H.ensure_package(dst, run=fake_run) and len(calls) == n          # idempotent
    fail = lambda c, **kw: types.SimpleNamespace(returncode=128, stderr="no network")  # noqa: E731
    msg2 = H.ensure_package(str(tmp_path / "other"), run=fail)
    assert "falls back to Triton" in msg2 and ": anchor ok" not in msg2
