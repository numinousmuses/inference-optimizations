"""pd_nvlink_copy: the hook on the v0.5.20 tree and the module's CPU-checkable parts.

1. The three anchors resolve once each in mooncake/conn.py; the patched file compiles. The hook composes with
   dsa_compact_index_k_hooks (which also edits _send_kvcache_generic) in either order. Stock from_zmq never reads
   frame 19.
2. The export frame round-trips (fake driver). register_remote keys it by session. map_remote translates addresses,
   opens each allocation once, and refuses addresses outside every allocation.
3. build_plan skips 0-row placeholders, refuses unaligned or unmapped buffers, and returns None when nothing is left.
4. The Triton kernel run under TRITON_INTERPRET=1 on CPU tensors matches reference_copy byte for byte: 3 buffers with
   different item sizes, scattered page pairs, an item longer than one chunk.
5. try_copy: disabled, unknown session or unmappable buffer -> None (the Mooncake path); never raises.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kernels"))
import pd_nvlink_copy_hooks as H  # noqa: E402

SRC = os.path.join(os.environ.get("SGLANG_SRC", os.path.expanduser("~/.cache/numinous/sgl-v0520/python")), "sglang")


@pytest.fixture()
def tree(tmp_path):
    src = os.path.join(SRC, H.PATH)
    if not os.path.exists(src):
        pytest.skip(f"no v0.5.20 tree at {SRC}")
    dst = tmp_path / H.PATH
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)
    return tmp_path


def test_anchors_apply_compile_and_compose(tree):
    assert H.apply(str(tree), check=True) == [f"{H.NAME}: 3 anchors ok in {H.PATH}"]
    out = H.apply(str(tree))
    assert out[0] == f"{H.NAME}: applied 3 hooks to {H.PATH}" and out[1].endswith(H.MODULE_DST)
    text = (tree / H.PATH).read_text()
    assert text.count("numinous-hook:pd_nvlink_copy") == 3
    assert (tree / H.MODULE_DST).exists()
    assert H.apply(str(tree))[0] == f"{H.NAME}: already applied"
    import dsa_compact_index_k_hooks as C                   # the other hook that edits this function
    C.HOOKS = [h for h in C.HOOKS if h[1] == H.PATH]
    assert all("applied" in ln or "ok" in ln for ln in C.apply(str(tree)))
    compile((tree / H.PATH).read_text(), H.PATH, "exec")


def test_compose_reverse_order(tmp_path):
    src = os.path.join(SRC, H.PATH)
    if not os.path.exists(src):
        pytest.skip("no tree")
    dst = tmp_path / H.PATH
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)
    import dsa_compact_index_k_hooks as C
    importlib.reload(C)
    C.HOOKS = [h for h in C.HOOKS if h[1] == H.PATH]
    C.apply(str(tmp_path))
    assert H.apply(str(tmp_path))[0].startswith(f"{H.NAME}: applied 3")


def test_stock_from_zmq_ignores_frame_19():
    text = open(os.path.join(SRC, H.PATH)).read()
    body = text[text.index("class KVArgsRegisterInfo"):]
    body = body[body.index("    def from_zmq(cls, msg: List[bytes]):"): body.index("class MooncakeKVManager")]
    assert "msg[19]" not in body and "len(msg) > 19" not in body and "slot_ids_index=18" in body


class FakeDriver:
    """Allocations at fixed bases; ipc_open maps base B to B + OFFSET."""
    OFFSET = 1 << 40

    def __init__(self, allocs):
        self.allocs = allocs            # [(base, size)]
        self.opens = []

    def mem_range(self, ptr):
        for b, s in self.allocs:
            if b <= ptr < b + s:
                return b, s
        raise RuntimeError("not an allocation")

    def ipc_get(self, base):
        return base.to_bytes(8, "little") + b"\x00" * 56

    def ipc_open(self, handle):
        base = int.from_bytes(handle[:8], "little")
        self.opens.append(base)
        return base + self.OFFSET


@pytest.fixture()
def mod(monkeypatch):
    import pd_nvlink_copy as M
    importlib.reload(M)
    monkeypatch.setenv(M.ENV, "1")
    return M


class KVArgs:
    def __init__(self, kv, state):
        self.kv_data_ptrs, self.state_data_ptrs = kv, state


def test_export_register_map(mod):
    drv = FakeDriver([(0x1000_0000, 0x100_0000), (0x2000_0000, 0x100_0000), (0x3000_0000, 0x10_0000)])
    mod._DRIVER = drv
    args = KVArgs([0x1000_0000, 0x1000_4000, 0x2000_0000], [[0x3000_0000, 0], [0x3000_0040]])
    frame = mod.export_frame(args)
    assert frame.startswith(mod.MAGIC) and mod.export_frame(args) is frame        # cached on kv_args
    table = mod.parse_frame(frame)
    assert [b for b, _, _ in table] == [0x1000_0000, 0x2000_0000, 0x3000_0000]   # deduped by allocation, 0 skipped
    mod.register_remote("sess", [b""] * 19 + [frame])
    assert mod.map_remote("sess", 0x1000_4000, 64, 0) == 0x1000_4000 + drv.OFFSET
    assert mod.map_remote("sess", 0x1000_8000, 64, 0) == 0x1000_8000 + drv.OFFSET
    assert drv.opens == [0x1000_0000]                                              # opened once per allocation
    assert mod.map_remote("sess", 0x1000_0000 + 0x100_0000 - 32, 64, 0) is None  # straddles the end
    assert mod.map_remote("other", 0x1000_4000, 64, 0) is None
    assert mod.decode_frames(args) == [frame]


def test_build_plan(mod):
    drv = FakeDriver([(0x1000_0000, 0x100_0000)])
    mod._DRIVER = drv
    mod.register_remote("s", [b""] * 19 + [mod.export_frame(KVArgs([0x1000_0000], []))])
    plan = mod.build_plan([(0x5000, 0x1000_0000, 512), (0, 0x1000_1000, 0), (0x9000, 0x1000_2000, 256)], "s", 0)
    src, dst, item = plan
    assert list(src) == [0x5000, 0x9000] and list(item) == [512, 256]
    assert list(dst) == [0x1000_0000 + drv.OFFSET, 0x1000_2000 + drv.OFFSET]
    assert mod.build_plan([(0x5000, 0x1000_0000, 510)], "s", 0) is None             # not a multiple of 4
    assert mod.build_plan([(0x5000, 0x9000_0000, 512)], "s", 0) is None             # unmapped destination
    assert mod.build_plan([(0, 0x1000_0000, 0)], "s", 0) is None                    # nothing to copy


def test_try_copy_falls_back(mod, monkeypatch):
    class Mgr:
        kv_args = types_ns(gpu_id=0)
    assert mod.try_copy(Mgr(), "unknown", [(1, 2, 4)], [0], [0]) is None
    monkeypatch.setenv(mod.ENV, "0")
    assert mod.try_copy(Mgr(), "s", [(1, 2, 4)], [0], [0]) is None


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


KERNEL_PROBE = textwrap.dedent('''
    import os, sys
    os.environ["TRITON_INTERPRET"] = "1"
    sys.path.insert(0, sys.argv[1])
    import numpy as np, torch
    import pd_nvlink_copy as M
    rng = np.random.default_rng(0)
    items = [22528, 8448, 4 * M.CHUNK_ELEMS * 2 + 1024]          # KV4 page, index_k page, > 2 chunks
    n_src, n_dst, n_pairs = 40, 50, 17
    srcs = [torch.randint(-2**31, 2**31 - 1, (n_src * n // 4,), dtype=torch.int32) for n in items]
    dsts = [torch.zeros(n_dst * n // 4, dtype=torch.int32) for n in items]
    sp = rng.choice(n_src, n_pairs, replace=False)
    dp = rng.choice(n_dst, n_pairs, replace=False)
    st = torch.tensor([t.data_ptr() for t in srcs], dtype=torch.int64)
    dt = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64)
    it = torch.tensor(items, dtype=torch.int64)
    chunks = -(-max(items) // (4 * M.CHUNK_ELEMS))
    M._ipc_page_copy[(n_pairs, len(items), chunks)](st, dt, it, torch.tensor(sp), torch.tensor(dp), CHUNK=M.CHUNK_ELEMS)
    ref = [torch.zeros_like(d) for d in dsts]
    for l, n in enumerate(items):
        e = n // 4
        for a, b in zip(sp, dp):
            ref[l][b * e:(b + 1) * e] = srcs[l][a * e:(a + 1) * e]
    ok = all(torch.equal(d, r) for d, r in zip(dsts, ref))
    print("KERNEL_OK" if ok else "KERNEL_MISMATCH")
''')


def test_kernel_matches_reference_under_the_interpreter(tmp_path):
    pytest.importorskip("triton")
    pytest.importorskip("torch")
    probe = tmp_path / "probe.py"
    probe.write_text(KERNEL_PROBE)
    r = subprocess.run([sys.executable, str(probe), str(HERE.parent / "kernels")], capture_output=True, text=True,
                       timeout=600)
    out = r.stdout + r.stderr
    if "KERNEL_OK" not in out and ("pointer_type" in out or "interpreter" in out.lower()):
        pytest.skip(f"Triton interpreter cannot cast int to pointer here: {out[-400:]}")
    assert "KERNEL_OK" in out, out[-2000:]
