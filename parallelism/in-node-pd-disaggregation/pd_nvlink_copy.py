"""One-kernel intra-node PD KV transfer over CUDA IPC (rtxsys, 2026-10-05; coordinator item "P7 KV transfer 1.3 GB/s").

Installed by operations/patches/pd_nvlink_copy_hooks.py as sglang/srt/disaggregation/mooncake/numinous_nvlink_copy.py.

Why the stock path is slow (source_code; SGLang v0.5.20 + Mooncake 0.3.13.post1, whose
intranode_nvlink_transport.cpp:83/:197 log lines match the image's logs):
- _send_kvcache_generic (mooncake/conn.py:650-862) builds one Mooncake entry per (layer buffer x contiguous page
  run). It does this in a Python loop (set_transfer_blocks, conn.py:755-776) and hands the whole list to one
  batch_transfer_sync (conn.py:789-807).
- Mooncake then works per entry:
  - a linear scan of the peer's registered buffers under a lock (relocateSharedMemoryAddress, cpp:706-760);
  - one copy-engine job per entry in cudaMemcpyBatchAsync on a single thread-local stream (cpp:175-260, 419-502);
  - a busy completion loop that calls cudaStreamQuery once per entry per round (getTransferStatus cpp:504-550,
    MultiTransport::getBatchTransferStatus multi_transport.cpp:363-400, transfer_engine_py.cpp:646-668).
- With page-fragmented pools a GLM KV4 page is 22.5 KB (index_k 8.4 KB), so a 339 MB turn is tens of thousands of
  entries.
- P7 measured 1.3 GB/s p50 (pd_time_stats over ReqTimeStats; amdglm findings 9e). The p90 of 18 GB/s is the
  contiguous case.

What this does instead:
- The decode server exports CUDA IPC handles for its KV/state allocations. They ride as one extra frame on its
  existing KV-args registration message, after the 19 stock frames.
- The prefill server opens each allocation once, mapped into its own device with lazy peer access (no context on the
  decode GPU).
- Every send is one Triton launch. Each program copies one (page pair x buffer x chunk) straight from the prefill
  pool into the decode pool over NVLink. Addresses are computed on the GPU from per-buffer base/item tables and the
  page index arrays; no per-entry Python, no per-entry DMA job, no per-entry polling.
- Then a stream sync, and the stock completion path continues.
- Fallback to the stock Mooncake path on anything unexpected: no table for the session, an unmapped address, an
  item length not a multiple of 4, no Triton, or an exception. It logs once.
Env: NUMINOUS_PD_NVLINK_COPY=1 on both roles (the decode role exports; the prefill role copies).
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MAGIC = b"NUMIPC1:"
ENV = "NUMINOUS_PD_NVLINK_COPY"
CHUNK_ELEMS = 2048          # int32 elements per program: 8 KB
_LOCK = threading.Lock()
_TLS = threading.local()                                     # one non-blocking copy stream per transfer thread
_REMOTE: Dict[str, List[Tuple[int, int, bytes]]] = {}       # session -> [(base, size, handle)]
_MAPPED: Dict[Tuple[str, int, int], int] = {}                # (session, base, device) -> local mapped pointer
_WARNED: set = set()


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning("pd_nvlink_copy: " + msg, *args)


# ------------------------------------------------------------------------------------------------ CUDA driver (ctypes)
class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


class Driver:
    """The four driver calls this needs. libcuda.so.1 ships with the NVIDIA driver in every GPU container."""

    def __init__(self, lib=None):
        self.lib = lib or ctypes.CDLL("libcuda.so.1")

    def _ok(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what} failed: CUresult {rc}")

    def mem_range(self, ptr: int) -> Tuple[int, int]:
        base, size = ctypes.c_uint64(0), ctypes.c_size_t(0)
        self._ok(self.lib.cuMemGetAddressRange_v2(ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(ptr)),
                 "cuMemGetAddressRange_v2")
        return int(base.value), int(size.value)

    def ipc_get(self, base: int) -> bytes:
        h = _Handle()
        self._ok(self.lib.cuIpcGetMemHandle(ctypes.byref(h), ctypes.c_uint64(base)), "cuIpcGetMemHandle")
        return ctypes.string_at(ctypes.addressof(h), 64)      # .reserved would stop at the first NUL byte

    def ipc_open(self, handle: bytes) -> int:
        h = _Handle()
        ctypes.memmove(ctypes.byref(h), handle, 64)
        d = ctypes.c_uint64(0)
        self._ok(self.lib.cuIpcOpenMemHandle_v2(ctypes.byref(d), h, ctypes.c_uint(1)),   # LAZY_ENABLE_PEER_ACCESS
                 "cuIpcOpenMemHandle_v2")
        return int(d.value)


_DRIVER: Optional[Driver] = None


def driver() -> Driver:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = Driver()
    return _DRIVER


# ------------------------------------------------------------------------------------------------ decode side: export
def export_allocations(ptrs: Sequence[int], drv: Optional[Driver] = None) -> List[Tuple[int, int, bytes]]:
    """(base, size, ipc handle) per distinct allocation behind `ptrs` (a pool's per-layer / per-state data ptrs)."""
    drv = drv or driver()
    seen: Dict[int, Tuple[int, int, bytes]] = {}
    for p in ptrs:
        if not p:
            continue
        base, size = drv.mem_range(int(p))
        if base not in seen:
            seen[base] = (base, size, drv.ipc_get(base))
    return sorted(seen.values())


def _flat(xs) -> List[int]:
    out: List[int] = []
    for x in xs or []:
        if isinstance(x, (list, tuple)):
            out.extend(_flat(x))
        else:
            out.append(int(x))
    return out


def export_frame(kv_args, drv: Optional[Driver] = None) -> bytes:
    """The extra registration frame for this decode rank's pools (cached on kv_args)."""
    cached = getattr(kv_args, "_numinous_ipc_frame", None)
    if cached is not None:
        return cached
    ptrs = _flat(getattr(kv_args, "kv_data_ptrs", [])) + _flat(getattr(kv_args, "state_data_ptrs", []))
    allocs = export_allocations(ptrs, drv)
    frame = MAGIC + json.dumps([[b, s, h.hex()] for b, s, h in allocs]).encode()
    try:
        kv_args._numinous_ipc_frame = frame
    except Exception:  # noqa: BLE001
        pass
    return frame


def decode_frames(kv_args) -> List[bytes]:
    """Frames to append to the decode KV-args registration message (empty unless enabled; never raises)."""
    if not enabled():
        return []
    try:
        return [export_frame(kv_args)]
    except Exception as e:  # noqa: BLE001
        _warn_once("export", "IPC export failed (%s); prefill keeps the Mooncake path", e)
        return []


# ------------------------------------------------------------------------------------------------ prefill side: import
def parse_frame(frame: bytes) -> Optional[List[Tuple[int, int, bytes]]]:
    if not frame.startswith(MAGIC):
        return None
    return [(int(b), int(s), bytes.fromhex(h)) for b, s, h in json.loads(frame[len(MAGIC):].decode())]


def register_remote(session_id: str, msg: Sequence[bytes]) -> None:
    """Called on the prefill side with a decode rank's raw registration message."""
    if not enabled():
        return
    for frame in list(msg)[19:]:
        try:
            table = parse_frame(frame)
        except Exception as e:  # noqa: BLE001
            _warn_once("parse", "bad IPC frame from %s: %s", session_id, e)
            table = None
        if table is not None:
            with _LOCK:
                _REMOTE[session_id] = table
                for k in [k for k in _MAPPED if k[0] == session_id]:
                    _MAPPED.pop(k, None)      # a re-registered decode rank: reopen lazily (CUDA refcounts handles)
            return


def map_remote(session_id: str, addr: int, length: int, device: int, drv: Optional[Driver] = None) -> Optional[int]:
    """The prefill-local pointer for a decode-pool address, or None if no exported allocation covers it."""
    table = _REMOTE.get(session_id)
    if not table:
        return None
    for base, size, handle in table:
        if base <= addr and addr + length <= base + size:
            key = (session_id, base, device)
            mapped = _MAPPED.get(key)
            if mapped is None:
                with _LOCK:
                    mapped = _MAPPED.get(key)
                    if mapped is None:
                        mapped = (drv or driver()).ipc_open(handle)
                        _MAPPED[key] = mapped
            return mapped + (addr - base)
    return None


# ------------------------------------------------------------------------------------------------ the copy plan
def build_plan(layers_params: Sequence[Tuple[int, int, int]], session_id: str, device: int,
               drv: Optional[Driver] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-buffer (src base, mapped dst base, item bytes), or None when any buffer cannot take the kernel path."""
    src, dst, item = [], [], []
    for s, d, n in layers_params:
        n = int(n)
        if n == 0 or not s or not d:
            continue                  # a 0-row placeholder carries no data (dsa_compact_index_k drops it the same way)
        if n < 0 or n % 4 or int(s) % 4 or int(d) % 4:
            return None
        m = map_remote(session_id, int(d), n, device, drv)
        if m is None:
            return None
        src.append(int(s))
        dst.append(m)
        item.append(n)
    if not item:
        return None
    return np.asarray(src, np.int64), np.asarray(dst, np.int64), np.asarray(item, np.int64)


def reference_copy(read, write, src_bases, dst_bases, items, src_pages, dst_pages) -> None:
    """The plan's semantics in plain Python (tests): buffer l, page pair i moves items[l] bytes from
    src_bases[l] + src_pages[i] * items[l] to dst_bases[l] + dst_pages[i] * items[l]."""
    for l in range(len(items)):
        n = int(items[l])
        for sp, dp in zip(src_pages, dst_pages):
            write(int(dst_bases[l]) + int(dp) * n, read(int(src_bases[l]) + int(sp) * n, n))


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _ipc_page_copy(src_tab, dst_tab, item_tab, src_idx, dst_idx, CHUNK: tl.constexpr):
        p = tl.program_id(0)                      # page pair
        b = tl.program_id(1)                      # buffer (layer / state tensor)
        c = tl.program_id(2)                      # CHUNK-element chunk of the item
        n_el = tl.load(item_tab + b) // 4         # int32 elements per item
        off = c * CHUNK + tl.arange(0, CHUNK)
        mask = off < n_el
        sp = tl.load(src_idx + p).to(tl.int64)
        dp = tl.load(dst_idx + p).to(tl.int64)
        src = tl.load(src_tab + b).to(tl.pointer_type(tl.int32))
        dst = tl.load(dst_tab + b).to(tl.pointer_type(tl.int32))
        v = tl.load(src + sp * n_el + off, mask=mask)
        tl.store(dst + dp * n_el + off, v, mask=mask)
except Exception:  # noqa: BLE001
    triton = None


def launch(src_bases: np.ndarray, dst_bases: np.ndarray, items: np.ndarray, src_pages, dst_pages, device) -> None:
    import torch
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.int64)).to(dev, non_blocking=False)  # noqa: E731
    st, dt, it = t(src_bases), t(dst_bases), t(items)
    sp, dp = t(src_pages), t(dst_pages)
    n_pages, n_buf = int(sp.numel()), int(it.numel())
    if n_pages == 0 or n_buf == 0:
        return
    chunks = int(-(-int(items.max()) // (4 * CHUNK_ELEMS)))
    _ipc_page_copy[(n_pages, n_buf, chunks)](st, dt, it, sp, dp, CHUNK=CHUNK_ELEMS)


def try_copy(mgr, session_id: str, layers_params, prefill_indices, dst_indices) -> Optional[int]:
    """The kernel path for one _send_kvcache_generic call. Returns 0 on success, None to fall back to Mooncake."""
    if not enabled() or triton is None or session_id not in _REMOTE:
        return None
    try:
        import torch
        device = int(getattr(mgr.kv_args, "gpu_id", torch.cuda.current_device()))
        src_pages = np.asarray(prefill_indices, dtype=np.int64)
        dst_pages = np.asarray(dst_indices, dtype=np.int64)
        if src_pages.shape != dst_pages.shape:
            return None
        with torch.cuda.device(device):
            plan = build_plan(layers_params, session_id, device)
            if plan is None:
                _warn_once(f"plan:{session_id}", "a buffer of session %s cannot take the kernel path; Mooncake path",
                           session_id)
                return None
            # The source KV is complete: the scheduler queues a chunk for transfer only after its forward's result is
            # processed (the same ordering the Mooncake path relies on). A private non-blocking stream keeps the copy
            # off the forward stream (the legacy default stream would serialize with it).
            streams = getattr(_TLS, "streams", None)
            if streams is None:
                streams = _TLS.streams = {}
            stream = streams.get(device)
            if stream is None:
                stream = streams[device] = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                launch(*plan, src_pages, dst_pages, device)
            stream.synchronize()
        return 0
    except Exception as e:  # noqa: BLE001
        _warn_once(f"err:{type(e).__name__}", "kernel path failed (%s: %s); Mooncake path", type(e).__name__, e)
        return None
