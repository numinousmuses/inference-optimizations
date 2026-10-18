"""Which inter-GPU memory mechanisms work on this ROCm box (torchrun --nproc-per-node 2).

The Daytona MI355X box exposes SR-IOV virtual functions ("AMD Instinct MI355X VF"); peer mappings may be limited.
Probes, each isolated and reported with its exact error:
  1. device peer access (hipDeviceCanAccessPeer via torch)
  2. torch symmetric memory: symm_mem.empty, then rendezvous (the CUDA/HIP backend: hipMemCreate + exported
     shareable handles), with the Python traceback of the failing call
  3. legacy HIP IPC (hipIpcGetMemHandle / hipIpcOpenMemHandle through torch's storage sharing): peer reads/writes
  4. hipMemCreate / hipMemExportToShareableHandle (POSIX FD) directly via ctypes: the call symmetric memory needs
  5. RCCL all_reduce (reference transport)
Prints one JSON report per rank (@@PROBE@@).
"""
from __future__ import annotations

import ctypes
import json
import os
import traceback

import torch
import torch.distributed as dist


def _tb(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))[-2500:]


def probe_vmm(dev: int) -> dict:
    """hipMemCreate with POSIX-FD shareable handle type, then export (ctypes against libamdhip64)."""
    out = {}
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError as e:
        return {"load": str(e)}

    class Loc(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

    class Prop(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("requestedHandleType", ctypes.c_int), ("location", Loc),
                    ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", ctypes.c_ubyte * 8)]
    prop = Prop()
    prop.type = 1                     # hipMemAllocationTypePinned
    prop.requestedHandleType = 1      # hipMemHandleTypePosixFileDescriptor
    prop.location.type = 1            # hipMemLocationTypeDevice
    prop.location.id = dev
    gran = ctypes.c_size_t(0)
    out["granularity_rc"] = hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0)
    out["granularity"] = gran.value
    size = max(gran.value, 2 << 20)
    handle = ctypes.c_uint64(0)
    out["hipMemCreate_rc"] = hip.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(prop), 0)
    if out["hipMemCreate_rc"] == 0:
        fd = ctypes.c_int(-1)
        out["hipMemExportToShareableHandle_rc"] = hip.hipMemExportToShareableHandle(
            ctypes.byref(fd), handle, 1, 0)
        out["fd"] = fd.value
        hip.hipMemRelease(handle)
    return out


def main() -> int:
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    rep: dict = {"rank": rank, "gpu": torch.cuda.get_device_name(local)}
    # 1
    rep["can_access_peer"] = {str(p): torch.cuda.can_device_access_peer(local, p) for p in range(world) if p != local}
    # 5
    try:
        x = torch.full((1024,), float(rank + 1), device="cuda")
        dist.all_reduce(x)
        torch.cuda.synchronize()
        rep["rccl_all_reduce"] = "ok" if float(x[0]) == sum(range(1, world + 1)) else f"wrong {float(x[0])}"
    except Exception as e:  # noqa: BLE001
        rep["rccl_all_reduce"] = _tb(e)
    # 2
    import torch.distributed._symmetric_memory as symm_mem
    s: dict = {}
    try:
        t = symm_mem.empty(4096, dtype=torch.float32, device="cuda")
        s["empty"] = "ok"
        try:
            h = symm_mem.rendezvous(t, dist.group.WORLD.group_name)
            s["rendezvous"] = "ok"
            s["multicast_ptr"] = int(getattr(h, "multicast_ptr", 0) or 0)
        except Exception as e:  # noqa: BLE001
            s["rendezvous"] = _tb(e)
    except Exception as e:  # noqa: BLE001
        s["empty"] = _tb(e)
    for be in ("CUDA", "NVSHMEM"):
        try:
            symm_mem.set_backend(be)
            s[f"set_backend_{be}"] = "ok"
        except Exception as e:  # noqa: BLE001
            s[f"set_backend_{be}"] = f"{type(e).__name__}: {str(e)[:200]}"
    rep["symm_mem"] = s
    # 3 legacy IPC through torch storage sharing
    ipc: dict = {}
    try:
        buf = torch.full((1 << 20,), float(rank), device="cuda")
        torch.cuda.synchronize()
        handle = buf.untyped_storage()._share_cuda_()
        hs = [None] * world
        dist.all_gather_object(hs, handle)
        peer = (rank + 1) % world
        st = torch.UntypedStorage._new_shared_cuda(*hs[peer])
        pv = torch.empty(0, dtype=torch.float32, device=st.device).set_(st, 0, (1 << 20,))  # opened on the owner's
        ipc["opened_on"] = str(st.device)
        ipc["open"] = "ok"
        ipc["peer_read_ok"] = bool(float(pv[0]) == float(peer))
        dist.barrier()
        with torch.cuda.device(st.device):
            pv[:16].fill_(100.0 + rank)              # write into the peer's buffer
            torch.cuda.synchronize()
        dist.barrier()
        src = (rank - 1) % world
        ipc["peer_write_seen"] = bool(float(buf[0]) == 100.0 + src)
    except Exception as e:  # noqa: BLE001
        ipc["error"] = _tb(e)
    rep["legacy_ipc"] = ipc
    # 4
    try:
        rep["vmm_posix_fd"] = probe_vmm(local)
    except Exception as e:  # noqa: BLE001
        rep["vmm_posix_fd"] = _tb(e)
    print("@@PROBE@@" + json.dumps(rep), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
