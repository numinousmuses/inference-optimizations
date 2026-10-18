"""ROCm stand-in for sgl_kernel.spatial (CUDA green contexts) used by SGLang's PD-multiplexing (agent k3amd,
2026-10-01). Installed as sglang/srt/multiplex/numinous_rocm_spatial.py by operations/patches/k3amd_pdmux_rocm_hooks.py.

pdmux runs prefill and decode on two streams that own disjoint compute partitions. CUDA builds them with green
contexts (create_greenctx_stream_by_value(prefill_sms, decode_sms, gpu)). HIP has no green contexts. Its equivalent
is a CU-masked stream: hipExtStreamCreateWithCUMask(&stream, mask_words, mask), and work on that stream is dispatched
only to the CUs whose bits are set. This module builds the two masks and wraps the HIP streams as
torch.cuda.ExternalStream.

CU layout. A multi-XCD part (MI300X / MI355X: 8 XCDs; MI355X has 256 CUs, 32 per XCD) must split every XCD, or one
side gets whole dies and the other side loses their L2. The default "xcd_block" layout assumes CU ids are numbered
die by die (XCD x owns ids [x*per, (x+1)*per)) and gives prefill the first p/num_xcd CUs of every XCD.
"interleave" assumes ids round-robin across XCDs (id % num_xcd = XCD). "contiguous" takes the first p ids.
Which numbering the driver uses must be checked on the box: rocprofv3 dispatch traces show CU/SE ids per kernel.
NUMINOUS_ROCM_CU_LAYOUT picks the layout and NUMINOUS_ROCM_NUM_XCD the die count (default 8).
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional

_STREAMS: list = []          # keep every created HIP stream alive for the process lifetime
_HIP: Optional[ctypes.CDLL] = None


def cu_masks(total: int, prefill: int, decode: int, layout: str = "xcd_block", num_xcd: int = 8) -> tuple[list, list]:
    """Two disjoint CU bit masks (lists of uint32 words) with prefill / decode CU counts."""
    if prefill < 0 or decode < 0 or prefill + decode > total:
        raise ValueError(f"cannot split {total} CUs into {prefill} + {decode}")
    ids = list(range(total))
    if layout == "contiguous":
        p_ids, d_ids = ids[:prefill], ids[prefill:prefill + decode]
    elif layout in ("xcd_block", "interleave"):
        if total % num_xcd or prefill % num_xcd or decode % num_xcd:
            raise ValueError(f"{layout}: {total} / {prefill} / {decode} CUs do not split evenly over {num_xcd} XCDs")
        per = total // num_xcd
        p_per, d_per = prefill // num_xcd, decode // num_xcd
        if layout == "xcd_block":
            die = [[x * per + j for j in range(per)] for x in range(num_xcd)]
        else:
            die = [[j * num_xcd + x for j in range(per)] for x in range(num_xcd)]
        p_ids = [c for d in die for c in d[:p_per]]
        d_ids = [c for d in die for c in d[p_per:p_per + d_per]]
    else:
        raise ValueError(f"unknown CU layout {layout!r}")
    words = (total + 31) // 32

    def pack(sel):
        m = [0] * words
        for c in sel:
            m[c // 32] |= 1 << (c % 32)
        return m
    return pack(p_ids), pack(d_ids)


def _hip() -> ctypes.CDLL:
    global _HIP
    if _HIP is None:
        _HIP = ctypes.CDLL(os.environ.get("NUMINOUS_HIP_LIB", "libamdhip64.so"))
        _HIP.hipExtStreamCreateWithCUMask.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
                                                      ctypes.POINTER(ctypes.c_uint32)]
        _HIP.hipExtStreamCreateWithCUMask.restype = ctypes.c_int
        _HIP.hipSetDevice.argtypes = [ctypes.c_int]
        _HIP.hipSetDevice.restype = ctypes.c_int
    return _HIP


def _masked_stream(mask: list, gpu_id: int):
    import torch
    hip = _hip()
    rc = hip.hipSetDevice(int(gpu_id))
    if rc != 0:
        raise RuntimeError(f"hipSetDevice({gpu_id}) failed: {rc}")
    handle = ctypes.c_void_p()
    arr = (ctypes.c_uint32 * len(mask))(*mask)
    rc = hip.hipExtStreamCreateWithCUMask(ctypes.byref(handle), len(mask), arr)
    if rc != 0:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: {rc}")
    s = torch.cuda.ExternalStream(handle.value, device=torch.device("cuda", int(gpu_id)))
    _STREAMS.append((handle, s))
    return s


def get_sm_available(gpu_id: int) -> int:
    import torch
    return int(torch.cuda.get_device_properties(int(gpu_id)).multi_processor_count)


def create_greenctx_stream_by_value(prefill_sms: int, decode_sms: int, gpu_id: int):
    """Same contract as sgl_kernel.spatial: (prefill_stream, decode_stream) over disjoint compute partitions."""
    total = get_sm_available(gpu_id)
    layout = os.environ.get("NUMINOUS_ROCM_CU_LAYOUT", "xcd_block")
    num_xcd = int(os.environ.get("NUMINOUS_ROCM_NUM_XCD", "8"))
    pm, dm = cu_masks(total, int(prefill_sms), int(decode_sms), layout, num_xcd)
    return _masked_stream(pm, gpu_id), _masked_stream(dm, gpu_id)
