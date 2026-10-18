"""Symmetric buffers over legacy HIP IPC, for ROCm builds where torch symmetric memory cannot rendezvous.

Why (measured on the mi355x box, torch 2.9.1+rocm7.2.0, HIP 7.2.26015, MI355X VF): symm_mem.empty works and so do
hipMemCreate / hipMemExportToShareableHandle, but symm_mem.rendezvous fails with "HIP error: invalid argument" in
hipMemImportFromShareableHandle: the HIP log shows the import receiving the low 32 bits of a host pointer as the
POSIX fd (os_handle 0x996bccf4 for an argument 0x641f996bccf4), i.e. torch passes a pointer to the fd where this HIP
takes the fd value, and hsa_amd_vmem_import_shareable_handle rejects it (status 4097). Legacy IPC
(hipIpcGetMemHandle / hipIpcOpenMemHandle) works on the same box, peer reads and writes included.

IpcSymmetricBuffer gives the subset of the torch symmetric-memory handle our kernels use: a local tensor, the
peers' base addresses as a host list (buffer_ptrs) and as a device int64 array (buffer_ptrs_dev, an integer
address as in torch), read_peer(rank, n) copies for host-side checks, multicast_ptr = 0, and a device-side barrier(channel) with torch's
algorithm (per channel and peer: CAS 0 -> 1 into the peer's signal pad with release at system scope, then CAS 1 -> 0
on the own pad with acquire at system scope), so a push followed by barrier() is ordered without a host sync.

Peer mappings come from torch's storage IPC (opened under the owner's device), and the local device is given peer
access to the owner, so the addresses are valid in kernels running on the local GPU. (hipIpcOpenMemHandle called
directly through ctypes under the local device returned hipErrorInvalidValue on this box.)
"""
from __future__ import annotations

import ctypes

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_HIP = None


def _hip():
    global _HIP
    if _HIP is None:
        _HIP = ctypes.CDLL("libamdhip64.so")
    return _HIP


def _check(rc: int, what: str) -> None:
    if rc != 0:
        name = _hip().hipGetErrorName
        name.restype = ctypes.c_char_p
        raise RuntimeError(f"{what} failed: {name(rc).decode()} ({rc})")


class IpcSymmetricBuffer:
    """One symmetric allocation per rank of `group` (collective: every rank constructs it together).

    The memory is torch's caching allocator (coarse-grained). Known hazard, measured on the mi355x box: after a
    rank rewrites its own receive buffer locally (zero_ then a host sync) and a peer then pushes into it, the owner
    read back the previous round's bytes for a whole row range, persisting over 2 ms, in 5 of 18 runs of an
    earlier single-round version of rocm_gpu_test_symm.py (0 of 60 rounds of the later 10-round loop) (the barrier's ISA does carry buffer_wbl2 sc0 sc1 / buffer_inv sc0 sc1). The standard fix
    is fine-grained or uncached receive buffers (what RCCL and SGLang's ROCm custom all-reduce allocate); raw
    hipIpcOpenMemHandle through ctypes returned hipErrorInvalidValue on this box for hipMalloc and
    hipExtMallocWithFlags memory alike, so that path is not provided here."""

    def __init__(self, shape, dtype: torch.dtype, group=None, signal_channels: int = 4):
        self.group = group or dist.group.WORLD
        self.rank = dist.get_rank(self.group)
        self.world = dist.get_world_size(self.group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.shape, self.dtype = tuple(shape), dtype
        self.channels = signal_channels
        self._peer_storages: list = []
        self.local = torch.zeros(self.shape, dtype=dtype, device=self.device)
        self.pad = torch.zeros(signal_channels * self.world, dtype=torch.int32, device=self.device)
        torch.cuda.synchronize()
        self.buffer_ptrs = self._exchange(self.local)
        self.signal_pad_ptrs = self._exchange(self.pad)
        self._ptrs_dev = torch.tensor(self.buffer_ptrs, dtype=torch.int64, device=self.device)
        self._pads_dev = torch.tensor(self.signal_pad_ptrs, dtype=torch.int64, device=self.device)
        self.buffer_ptrs_dev = self._ptrs_dev.data_ptr()
        self.signal_pad_ptrs_dev = self._pads_dev.data_ptr()
        self.multicast_ptr = 0
        dist.barrier(self.group)

    def _exchange(self, t: torch.Tensor) -> list[int]:
        """Every rank's address of its `t`, valid on this device. The mapping is torch's storage IPC
        (hipIpcGetMemHandle of the caching-allocator segment + offset; opened by torch under the owner's device, the
        only way this HIP accepts), then the local device is given peer access to the owner."""
        hip = _hip()
        mine = (t.untyped_storage()._share_cuda_(), t.storage_offset() * t.element_size(), self.device.index)
        allh: list = [None] * self.world
        dist.all_gather_object(allh, mine, group=self.group)
        out = []
        for r, (handle, off, dev) in enumerate(allh):
            if r == self.rank:
                out.append(t.data_ptr())
                continue
            st = torch.UntypedStorage._new_shared_cuda(*handle)
            self._peer_storages.append(st)
            rc = hip.hipDeviceEnablePeerAccess(dev, 0)
            if rc not in (0, 704):  # 704: hipErrorPeerAccessAlreadyEnabled
                _check(rc, "hipDeviceEnablePeerAccess")
            hip.hipGetLastError()
            out.append(st.data_ptr() + off)
        return out

    def read_peer(self, rank: int, n_elems: int, offset_elems: int = 0) -> torch.Tensor:
        """Copy n_elems of `rank`'s buffer starting at offset_elems into a local tensor (a device kernel load)."""
        out = torch.empty(n_elems, dtype=self.dtype, device=self.device)
        esz = self.local.element_size()
        _copy_from_addr(self.buffer_ptrs[rank] + offset_elems * esz, out)
        return out

    def barrier(self, channel: int = 0) -> None:
        _barrier_kernel[(self.world,)](self.signal_pad_ptrs_dev, self.rank, channel, WORLD=self.world)

    def close(self) -> None:
        torch.cuda.synchronize()
        dist.barrier(self.group)
        self._peer_storages = []


@triton.jit
def _barrier_kernel(pads_addr, rank, channel, WORLD: tl.constexpr):
    p = tl.program_id(0)
    pads = pads_addr.to(tl.pointer_type(tl.uint64))
    peer_pad = tl.load(pads + p).to(tl.pointer_type(tl.int32))
    own_pad = tl.load(pads + rank).to(tl.pointer_type(tl.int32))
    sig = peer_pad + channel * WORLD + rank
    while tl.atomic_cas(sig, 0, 1, sem="release", scope="sys") != 0:
        pass
    wt = own_pad + channel * WORLD + p
    while tl.atomic_cas(wt, 1, 0, sem="acquire", scope="sys") != 1:
        pass


@triton.jit
def _release_kernel(word):
    tl.atomic_add(word, 0, sem="release", scope="sys")


_RELEASE_WORD: dict = {}


def system_release() -> None:
    """A system-scope release on the current stream: writes before it (local L2 included) become visible to peers."""
    dev = torch.cuda.current_device()
    w = _RELEASE_WORD.get(dev)
    if w is None:
        w = _RELEASE_WORD[dev] = torch.zeros(1, dtype=torch.int32, device="cuda")
    _release_kernel[(1,)](w)


@triton.jit
def _copy_from_addr_kernel(src_addr, dst, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    src = src_addr.to(tl.pointer_type(tl.uint8))
    tl.store(dst + i, tl.load(src + i, mask=m), mask=m)


def _copy_from_addr(addr: int, out: torch.Tensor) -> None:
    b = out.view(torch.uint8)
    n = b.numel()
    _copy_from_addr_kernel[(triton.cdiv(n, 4096),)](addr, b, n, BLOCK=4096)
