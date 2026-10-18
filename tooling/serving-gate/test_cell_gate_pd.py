"""cell_gate in-box PD mode: role commands (pd_launch), the process group wrapper, the shipped proxy copy, and the Pro PD
arms' role flags."""
import importlib
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("GATE_LOCAL", "1")
os.environ.setdefault("SVC_NAME", "svc-test")
HERE = pathlib.Path(__file__).resolve().parent
GATEWAY = pathlib.Path("/home/ubuntu/numitechnology/numitechnology/platform/apps/inference/gateway/agent/pd_proxy.py")


def _cg():
    import cell_gate
    return importlib.reload(cell_gate)


def _vals(flags, name):
    return [flags[i + 1] for i, f in enumerate(flags) if f == name and i + 1 < len(flags)]


def test_pd_launch_roles():
    cg = _cg()
    base = ["--tp", "8", "--dp", "8", "--ep-size", "8", "--ep-size", "8", "--chunked-prefill-size", "65536", "--x"]
    pd = {"dp": 4, "decode_base": 4, "prefill": {"flags_remove": ["--tp", "--dp", "--ep-size"],
                                                 "flags_add": ["--tp", "4", "--dp", "4", "--ep-size", "4"],
                                                 "env": {"A": "1"}},
          "decode": {"flags_remove": ["--tp", "--dp", "--ep-size", "--chunked-prefill-size"],
                     "flags_add": ["--tp", "4", "--dp", "4", "--ep-size", "4", "--disable-radix-cache"]}}
    c = cg.pd_launch("/m", base, pd)
    p, d = c["prefill"]["flags"], c["decode"]["flags"]
    assert _vals(p, "--ep-size") == ["4"] and _vals(d, "--ep-size") == ["4"]      # every repeated occurrence dropped
    assert _vals(p, "--disaggregation-mode") == ["prefill"] and _vals(d, "--disaggregation-mode") == ["decode"]
    assert _vals(p, "--base-gpu-id") == ["0"] and _vals(d, "--base-gpu-id") == ["4"]
    assert _vals(p, "--disaggregation-bootstrap-port") == ["8998"] and "--disaggregation-bootstrap-port" not in d
    assert "--chunked-prefill-size" in p and "--chunked-prefill-size" not in d and "--disable-radix-cache" in d
    assert c["prefill"]["env"] == {"A": "1"} and c["decode"]["env"] == {}
    assert _vals(c["prefill"]["cmd"], "--port") == [str(cg.PORT + 100)]
    assert _vals(c["decode"]["cmd"], "--port") == [str(cg.PORT + 200)]
    px = c["proxy"]["cmd"]
    assert px[1].endswith("gateway_pd_proxy.py") and _vals(px, "--dp") == ["4"] and _vals(px, "--port") == [str(cg.PORT)]


def test_pdgroup_poll_and_stop(tmp_path):
    cg = _cg()
    cmds = {"prefill": {"cmd": [sys.executable, "-c", "import time; time.sleep(30)"], "env": {}},
            "decode": {"cmd": [sys.executable, "-c", "import sys; sys.exit(3)"], "env": {}},
            "proxy": {"cmd": [sys.executable, "-c", "import time; time.sleep(30)"]}}
    g = cg.PDGroup(cmds, str(tmp_path / "arm.log"), dict(os.environ))
    g.procs[1].wait(timeout=10)
    assert g.poll() == 3                                  # a dead role is reported at once
    g.kill()
    for p in g.procs:
        p.wait(timeout=10)
    assert all(p.poll() is not None for p in g.procs)
    assert (tmp_path / "arm_prefill.log").exists() and (tmp_path / "arm_proxy.log").exists()


def test_proxy_copy_matches_gateway():
    ours = (HERE / "gateway_pd_proxy.py").read_text()
    body = ours.split('test_cell_gate_pd.py checks it still matches the gateway file when that checkout is present.\n\n', 1)[1]
    if GATEWAY.exists():
        assert '"""' + body == GATEWAY.read_text()
    subprocess.run([sys.executable, "-m", "py_compile", str(HERE / "gateway_pd_proxy.py")], check=True)


def test_pro_pd_arms():
    import gate_arms
    importlib.reload(gate_arms)
    cg = _cg()
    from gate_arms import ARMS, compose_flags
    cell = cg.cell_by_key("deepseek-v4-pro/standard")
    for arm, hs in (("pro_pd", True), ("pro_pd_nohs", False), ("pro_pd_tcp", True)):
        d = ARMS[arm]
        c = cg.pd_launch("/m", compose_flags(cell["config"]["flags"], d, "/m"), d["_pd"])
        p, dec = c["prefill"]["flags"], c["decode"]["flags"]
        for f in (p, dec):
            assert _vals(f, "--tp") == ["4"] and _vals(f, "--dp") == ["4"] and _vals(f, "--ep-size") == ["4"]
            assert "--max-total-tokens" not in f and _vals(f, "--mem-fraction-static") == ["0.95"]
        assert "--enable-hierarchical-cache" in p and "--hicache-size" not in p and _vals(p, "--hicache-ratio") == ["10"] and _vals(p, "--chunked-prefill-size") == ["32768"]
        assert "--num-continuous-decode-steps" not in p
        assert c["decode"]["env"]["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK"] == "2048"
        assert _vals(dec, "--cuda-graph-max-bs-decode") == ["48"]
        assert "--disable-radix-cache" in dec and "--enable-prefill-delayer" not in dec
        assert ("--enable-hisparse" in dec) == hs
        for r in ("prefill", "decode"):
            if arm == "pro_pd_tcp":
                assert c[r]["env"]["MC_FORCE_TCP"] == "1" and "MC_INTRANODE_NVLINK" not in c[r]["env"]
            else:
                assert c[r]["env"]["MOONCAKE_PROTOCOL"] == "nvlink_intra"
