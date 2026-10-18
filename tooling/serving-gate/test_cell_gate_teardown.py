"""Sequential-gate teardown: a leftover engine child that holds an RPC port is killed and the port range frees up."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("GATE_LOCAL", "1")
import cell_gate  # noqa: E402

HOLDER = ("import socket, time, sys; s = socket.socket(); s.bind(('', int(sys.argv[1]))); s.listen(); "
          "time.sleep(600)")


def test_leftover_child_killed_and_port_freed():
    port = 30236
    # its own session, like a scheduler child that escaped the server's process group; argv names the engine
    p = subprocess.Popen([sys.executable, "-c", HOLDER, str(port), "sglang::scheduler_DP0_TP0"], start_new_session=True)
    try:
        time.sleep(1.0)
        assert port in cell_gate._wait_ports_free(port, port + 1, timeout_s=0.1)
        killed = cell_gate._kill_leftovers()
        assert p.pid in killed
        p.wait(timeout=10)
        assert cell_gate._wait_ports_free(port, port + 1, timeout_s=10) == []
    finally:
        if p.poll() is None:
            p.kill()


def test_kill_leftovers_spares_non_engine_processes():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        assert p.pid not in cell_gate._kill_leftovers()
        assert p.poll() is None
    finally:
        p.kill()
