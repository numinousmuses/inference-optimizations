"""kvmix core (model-agnostic): two KV pools in one slot space, the precision policy, and promotion.

Installed as sglang/srt/mem_cache/kvmix_core.py by operations/patches/kvmix_*_hooks.py. The model-specific pool
(operations/kernels/kvmix_glm_engine.py) supplies the storage, the read/write kernels and `kvmix_promote_pages`;
this module owns everything the scheduler sees:

KVMixAllocator (a PagedTokenToKVPoolAllocator by interface)
    One paged allocator per pool. Unified slot space: FP8 slots [0, B) with B = N8 + page_size (page 0 is the
    dummy page, as in stock), FP4 slots [B, N8 + N4 + page_size) (FP4-native slot = unified - N8, FP4-native page =
    unified page - N8 / page_size). alloc_extend / alloc_decode split the batch by the per-request precision flags
    the promotion pass left in `pending` (else inferred from last_loc), run the stock kernels once per group, and
    scatter the results back in request order. When the flagged pool is exhausted the group falls back to the other
    pool (counted in `stats`, logged) instead of failing the step: the read path accepts mixed rows per request.
    available_size() is the total; evict_to_free_tokens / check_decode_capacity evict the total shortfall (one
    call, stock semantics) and leave per-pool shortage to the fallback: the pool split is a preference, capacity
    is the sum.

Policy (before_alloc, called from alloc_for_extend / alloc_for_decode / alloc_for_spec_decode before last_loc is
read): a request is FP4 once max(prompt length, current KV length) >= NUMINOUS_KVMIX_THRESHOLD (default 262144).
The first time a request becomes FP4 its existing FP8 pages are promoted: pages of radix nodes on its path with
lock_ref == 1 (no other running request holds them) and its own pages beyond the tree-owned prefix are requantized
FP8 -> FP4 in place (new FP4 pages, one multi-layer kernel, FP8 pages freed), node.value / req_to_token /
req.prefix_indices are rewritten. Nodes shared with another running request (lock_ref > 1) stay FP8. Nodes shorter
than NUMINOUS_KVMIX_MIN_PROMOTE_TOKENS (default 4096, i.e. small shared-looking prefixes such as system prompts)
stay FP8 too. Node promotion needs the plain RadixCache; with UnifiedRadixCache (the v0.5.20 default for the served
GLM cell) or a chunk cache only the request's own pages beyond cache_protected_len are promoted and cached prefixes
keep the precision they were written with. The GPU work is ordered after the previous forward (cur_stream.wait_stream(forward_stream)) because
the overlap scheduler runs this while the previous step may still be writing the request's last rows.

Env: NUMINOUS_KVMIX=1 enables; NUMINOUS_KVMIX_THRESHOLD, NUMINOUS_KVMIX_FP4_FRAC (fraction of the latent KV byte
budget given to the FP4 pool, default 0.5), NUMINOUS_KVMIX_MIN_PROMOTE_TOKENS, NUMINOUS_KVMIX_FALLBACK (1 default:
cross-pool fallback; 0: an exhausted pool fails the allocation like stock), NUMINOUS_KVMIX_PROMOTE (1 default).
"""
from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Sequence

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)

logger = logging.getLogger("sglang.numinous")

ENV = "NUMINOUS_KVMIX"
THRESHOLD_ENV = "NUMINOUS_KVMIX_THRESHOLD"
FRAC_ENV = "NUMINOUS_KVMIX_FP4_FRAC"
MIN_PROMOTE_ENV = "NUMINOUS_KVMIX_MIN_PROMOTE_TOKENS"
FALLBACK_ENV = "NUMINOUS_KVMIX_FALLBACK"
PROMOTE_ENV = "NUMINOUS_KVMIX_PROMOTE"
STATS_DIR_ENV = "NUMINOUS_KVMIX_STATS_DIR"  # when set: the allocator appends a JSON line of its stats there periodically
DEFAULT_THRESHOLD = 262144
DEFAULT_FRAC = 0.5
DEFAULT_MIN_PROMOTE = 4096
_logged: set = set()


def log_once(key: str, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.warning(msg, *args)


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def threshold() -> int:
    return int(os.environ.get(THRESHOLD_ENV, str(DEFAULT_THRESHOLD)))


def fp4_frac() -> float:
    f = float(os.environ.get(FRAC_ENV, str(DEFAULT_FRAC)))
    if not (0.0 < f < 1.0):
        raise ValueError(f"{FRAC_ENV}={f}: expected a fraction strictly between 0 and 1")
    return f


def min_promote_tokens() -> int:
    return int(os.environ.get(MIN_PROMOTE_ENV, str(DEFAULT_MIN_PROMOTE)))


def fallback_enabled() -> bool:
    return os.environ.get(FALLBACK_ENV, "1") == "1"


def promote_enabled() -> bool:
    return os.environ.get(PROMOTE_ENV, "1") == "1"


# ------------------------------------------------------------------------------------------------ pool sizing


def blended_row_bytes(fp8_row: int, fp4_row: int, frac: float) -> int:
    """Per-token, per-layer bytes the configurator prices when a fraction `frac` of the latent byte budget is FP4:
    tokens per byte = (1 - f) / fp8_row + f / fp4_row, so the blended row is the harmonic mix (rounded up)."""
    return int(math.ceil(1.0 / ((1.0 - frac) / fp8_row + frac / fp4_row)))


def split_tokens(total_tokens: int, fp8_row: int, fp4_row: int, blended: int, frac: float, page_size: int):
    """(N8, N4) with N8 + N4 == total_tokens, both multiples of page_size, N8 * fp8_row + N4 * fp4_row <= total *
    blended (the bytes the configurator reserved), and N4 as close to the requested byte fraction as that allows."""
    assert total_tokens % page_size == 0, total_tokens
    budget = total_tokens * blended
    want4 = int(round(frac * budget / fp4_row / page_size)) * page_size
    n4 = max(0, min(total_tokens, want4))
    while n4 < total_tokens and (total_tokens - n4) * fp8_row + n4 * fp4_row > budget:
        n4 += page_size
    n8 = total_tokens - n4
    assert n8 * fp8_row + n4 * fp4_row <= budget, (n8, n4, budget)
    return n8, n4


# ------------------------------------------------------------------------------------------------ allocator


class KVMixAllocator(PagedTokenToKVPoolAllocator):
    """Two paged pools behind the stock paged-allocator interface (see module doc)."""

    kvmix = True

    def __init__(self, n8: int, n4: int, page_size: int, dtype: torch.dtype, device, kvcache, need_sort: bool,
                 forward_stream=None):
        BaseTokenToKVPoolAllocator.__init__(self, n8 + n4, page_size, dtype, device, kvcache, need_sort)
        assert n8 % page_size == 0 and n4 % page_size == 0 and n8 > 0 and n4 > 0, (n8, n4, page_size)
        self.n8, self.n4 = n8, n4
        self.boundary = n8 + page_size
        self.n_slots = n8 + n4 + page_size
        self.fp8_pages = n8 // page_size
        self.fp4_pages = n4 // page_size
        self.num_pages = self.fp8_pages + self.fp4_pages
        self.debug_mode = os.environ.get("SGLANG_DEBUG_MEMORY_POOL", "").lower() in ("1", "true")
        self.fp8 = PagedTokenToKVPoolAllocator(n8, page_size, dtype, device, kvcache, need_sort)
        self.fp4 = PagedTokenToKVPoolAllocator(n4, page_size, dtype, device, kvcache, need_sort)
        self.forward_stream = forward_stream
        self.threshold = threshold()
        self.min_promote = min_promote_tokens()
        self.fallback = fallback_enabled()
        self.promote = promote_enabled()
        self.pending: Optional[List[bool]] = None  # per-request FP4 flags for the next alloc_extend/alloc_decode
        self.pending_need = None  # (fp8 tokens, fp4 tokens) the coming allocation asks for, from before_alloc
        self.stats = {"alloc8": 0, "alloc4": 0, "fallback_4to8_pages": 0, "fallback_8to4_pages": 0,
                      "inferred_flags": 0, "promotions": 0, "promoted_pages": 0, "promotion_skipped": 0,
                      "promotion_no_fp4_pages": 0}
        self._grouping = False
        logger.warning("numinous kvmix allocator: FP8 %d tokens (%d pages) + FP4 %d tokens (%d pages), page %d, "
                       "threshold %d tokens, fallback %s, promote %s", n8, self.fp8_pages, n4, self.fp4_pages,
                       page_size, self.threshold, self.fallback, self.promote)

    # ---- unified <-> native --------------------------------------------------------------------------------
    def _to_unified_pages(self, native_pages4: torch.Tensor) -> torch.Tensor:
        return native_pages4 + self.fp8_pages

    def is_fp4_slot(self, slot: int) -> bool:
        return slot >= self.boundary

    # ---- capacity ------------------------------------------------------------------------------------------
    def available_size(self):
        return self.fp8.available_size() + self.fp4.available_size()

    def available_fp8(self):
        return self.fp8.available_size()

    def available_fp4(self):
        return self.fp4.available_size()

    def get_all_free_pages(self):
        return torch.cat((self.fp8.get_all_free_pages(), self._to_unified_pages(self.fp4.get_all_free_pages())))

    @property
    def free_pages(self):  # debug census only; the hot paths use the sub-allocators
        return self.get_all_free_pages()

    @free_pages.setter
    def free_pages(self, value):  # BaseTokenToKVPoolAllocator.__init__ assigns None
        assert value is None

    def merge_and_sort_free(self):
        self.fp8.merge_and_sort_free()
        self.fp4.merge_and_sort_free()

    def debug_print(self) -> str:
        return (f"kvmix fp8 free {self.fp8.available_size()} / {self.n8}, fp4 free {self.fp4.available_size()} / "
                f"{self.n4}, stats {self.stats}")

    def _needs(self, requests, num_tokens: int):
        """(need8, need4) tokens for the next decode step of `requests`, split by their (future) precision."""
        if not requests:
            return num_tokens, 0
        per = max(1, num_tokens // max(1, len(requests)))
        need4 = sum(per for r in requests if will_be_fp4(r, self.threshold, "decode"))
        return num_tokens - need4, need4

    def _evict_split(self, tree_cache, need8: int, need4: int) -> None:
        """Evict the TOTAL shortfall (stock semantics), then at most one more round for a per-pool shortfall.
        Never a loop: with a precision-blind LRU (and UnifiedRadixCache's consensus eviction) a loop that insists on
        one pool evicts every other session's cache and stalls decode (measured: 5.5 vs 30 tok/s/user at 64
        sessions). Per-pool shortage after that is served by the cross-pool fallback in _alloc_group."""
        if tree_cache is None or tree_cache.is_chunk_cache():
            return
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        total_short = max(0, need8 + need4 - self.available_size())
        if total_short > 0:
            tree_cache.evict_for_alloc(EvictParams(num_tokens=total_short))
        if not self.fallback:
            short = max(0, need8 - self.fp8.available_size()) + max(0, need4 - self.fp4.available_size())
            if short > 0:
                tree_cache.evict_for_alloc(EvictParams(num_tokens=short))

    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> None:
        """Called by evict_from_tree_cache (hooked) before every allocation: evict per-pool shortfalls, splitting
        num_tokens the way before_alloc measured the coming batch (pending_need), else all FP8."""
        need = self.pending_need
        if need and (need[0] + need[1]) > 0 and self.pending is not None:
            scale = num_tokens / float(need[0] + need[1])
            need4 = min(num_tokens, int(math.ceil(need[1] * scale)))
            self._evict_split(tree_cache, num_tokens - need4, need4)
        else:
            self._evict_split(tree_cache, num_tokens, 0)

    def check_decode_capacity(self, *, num_tokens: int, tree_cache, requests=None, spec_algorithm=None) -> bool:
        need8, need4 = self._needs(requests, num_tokens)
        self._evict_split(tree_cache, need8, need4)
        if self.fallback:
            return self.available_size() >= num_tokens
        return self.fp8.available_size() >= need8 and self.fp4.available_size() >= need4

    # ---- diagnostics -----------------------------------------------------------------------------------------
    def _dump_stats(self) -> None:
        """Every 500 allocations, append one JSON line (stats + free tokens per pool) to
        $NUMINOUS_KVMIX_STATS_DIR/<host>_<pid>.jsonl; the gate's /cache volume makes it readable after the run."""
        d = os.environ.get(STATS_DIR_ENV)
        if not d:
            return
        self._calls = getattr(self, "_calls", 0) + 1
        if self._calls % 500 != 1:
            return
        try:
            import json
            import socket
            import time

            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{socket.gethostname()}_{os.getpid()}.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps({"t": time.time(), "calls": self._calls, "free8": self.fp8.available_size(),
                                    "free4": self.fp4.available_size(), "n8": self.n8, "n4": self.n4,
                                    **self.stats}) + "\n")
        except OSError as e:  # diagnostics never fail the step
            log_once("stats", "numinous kvmix: stats dump failed: %s", e)

    # ---- allocation ------------------------------------------------------------------------------------------
    def _take_flags(self, bs: int, last_loc: torch.Tensor) -> List[bool]:
        self._dump_stats()
        flags, self.pending = self.pending, None
        if flags is not None and len(flags) == bs:
            return flags
        if flags is not None:
            log_once("flags_len", "numinous kvmix: pending flags for %d requests but the batch has %d; inferring "
                     "precision from last_loc", len(flags), bs)
        self.stats["inferred_flags"] += 1
        ll = last_loc.reshape(-1).tolist()
        return [s >= self.boundary for s in ll]

    def _free_list(self, sub: PagedTokenToKVPoolAllocator, need: int, is_fp4: bool) -> torch.Tensor:
        if need > len(sub.free_pages):
            sub.merge_and_sort_free()
        free = sub.free_pages
        return self._to_unified_pages(free) if is_fp4 else free

    def _consume(self, sub: PagedTokenToKVPoolAllocator, k: int) -> None:
        sub.free_pages = sub.free_pages[k:]

    def _run_extend(self, free: torch.Tensor, prefix_lens, seq_lens, last_loc, extend_num_tokens: int):
        out = torch.empty((extend_num_tokens,), dtype=torch.int64, device=self.device)
        if extend_num_tokens == 0:
            return out
        if free.device.type == "cuda":
            from sglang.kernels.ops.memory.allocator import alloc_extend_kernel
            from sglang.srt.utils import next_power_of_2

            alloc_extend_kernel[(len(prefix_lens),)](prefix_lens, seq_lens, last_loc, free, out,
                                                     next_power_of_2(len(prefix_lens)), self.page_size)
        else:
            alloc_extend_naive(prefix_lens, seq_lens, last_loc, free, out, self.page_size, self.device)
        return out

    def _run_decode(self, free: torch.Tensor, seq_lens, last_loc):
        bs = len(seq_lens)
        out = torch.empty((bs,), dtype=torch.int64, device=self.device)
        if free.device.type == "cuda":
            from sglang.kernels.ops.memory.allocator import alloc_decode_kernel
            from sglang.srt.utils import next_power_of_2

            alloc_decode_kernel[(bs,)](seq_lens, last_loc, free, out, next_power_of_2(bs), self.page_size)
        else:
            new_page = (seq_lens - 1) % self.page_size == 0
            idx = torch.cumsum(new_page.to(torch.int64), 0) - 1
            page = free[idx.clamp(min=0, max=max(0, free.numel() - 1))] if free.numel() else torch.zeros_like(idx)
            out.copy_(torch.where(new_page, page * self.page_size, last_loc + 1))
        return out

    def _alloc_group(self, is_fp4: bool, need_pages: int, run):
        """Run the stock kernel for one precision group with that pool's free list (or a cross-pool fallback)."""
        primary, other = (self.fp4, self.fp8) if is_fp4 else (self.fp8, self.fp4)
        free = self._free_list(primary, need_pages, is_fp4)
        if len(free) >= need_pages:
            out = run(free)
            self._consume(primary, need_pages)
            self.stats["alloc4" if is_fp4 else "alloc8"] += need_pages
            return out
        if not self.fallback:
            return None
        short = need_pages - len(free)
        other_free = self._free_list(other, short, not is_fp4)
        if len(other_free) < short:
            return None
        out = run(torch.cat((free, other_free[:short])))
        self._consume(primary, len(free))
        self._consume(other, short)
        key = "fallback_4to8_pages" if is_fp4 else "fallback_8to4_pages"
        self.stats[key] += short
        self.stats["alloc4" if is_fp4 else "alloc8"] += len(free)
        if self.stats[key] in (short, 1024, 16384) or self.stats[key] % 65536 < short:
            logger.warning("numinous kvmix: %s pool exhausted, %d pages taken from the %s pool (total %d); "
                           "consider changing %s", "FP4" if is_fp4 else "FP8", short, "FP8" if is_fp4 else "FP4",
                           self.stats[key], FRAC_ENV)
        return out

    def alloc_extend(self, prefix_lens: torch.Tensor, prefix_lens_cpu: torch.Tensor, seq_lens: torch.Tensor,
                     seq_lens_cpu: torch.Tensor, last_loc: torch.Tensor, extend_num_tokens: int,
                     num_new_pages: int = None, **_ignored):
        bs = len(prefix_lens_cpu)
        flags = self._take_flags(bs, last_loc)
        ps = self.page_size
        pages = ((seq_lens_cpu + ps - 1) // ps - (prefix_lens_cpu + ps - 1) // ps).tolist()
        ext = (seq_lens_cpu - prefix_lens_cpu).tolist()
        groups = {False: [i for i in range(bs) if not flags[i]], True: [i for i in range(bs) if flags[i]]}
        if not groups[True] or not groups[False]:
            is4 = bool(groups[True])
            return self._alloc_group(is4, sum(pages), lambda free: self._run_extend(
                free, prefix_lens, seq_lens, last_loc, extend_num_tokens))
        out = torch.empty((extend_num_tokens,), dtype=torch.int64, device=self.device)
        starts = [0] * bs
        for i in range(1, bs):
            starts[i] = starts[i - 1] + ext[i - 1]
        for is4, idx in groups.items():
            idx_t = torch.tensor(idx, dtype=torch.int64, device=self.device)
            n_tok = sum(ext[i] for i in idx)
            sub = self._alloc_group(is4, sum(pages[i] for i in idx), lambda free: self._run_extend(
                free, prefix_lens[idx_t], seq_lens[idx_t], last_loc[idx_t], n_tok))
            if sub is None:
                return None  # the other group's pages are lost only on this failing step; stock raises here too
            pos = torch.cat([torch.arange(starts[i], starts[i] + ext[i], dtype=torch.int64) for i in idx]) \
                if n_tok else torch.empty(0, dtype=torch.int64)
            out[pos.to(self.device)] = sub
        return out

    def alloc_decode(self, seq_lens: torch.Tensor, seq_lens_cpu: torch.Tensor, last_loc: torch.Tensor, **_ignored):
        bs = len(seq_lens_cpu)
        flags = self._take_flags(bs, last_loc)
        new_page = (seq_lens_cpu % self.page_size == 1).tolist()
        groups = {False: [i for i in range(bs) if not flags[i]], True: [i for i in range(bs) if flags[i]]}
        if not groups[True] or not groups[False]:
            is4 = bool(groups[True])
            return self._alloc_group(is4, sum(1 for x in new_page if x),
                                     lambda free: self._run_decode(free, seq_lens, last_loc))
        out = torch.empty((bs,), dtype=torch.int64, device=self.device)
        for is4, idx in groups.items():
            idx_t = torch.tensor(idx, dtype=torch.int64, device=self.device)
            sub = self._alloc_group(is4, sum(1 for i in idx if new_page[i]),
                                    lambda free: self._run_decode(free, seq_lens[idx_t], last_loc[idx_t]))
            if sub is None:
                return None
            out[idx_t] = sub
        return out

    def alloc(self, need_size: int):
        """Page-aligned contiguous slots; FP8 first, FP4 when FP8 is exhausted (page_size == 1 paths only)."""
        out = self.fp8.alloc(need_size)
        if out is None and self.fallback:
            out = self.fp4.alloc(need_size)
            if out is not None:
                out = out + self.n8
                self.stats["fallback_8to4_pages"] += need_size // self.page_size
        return out

    def alloc_fp4_slots(self, num_pages: int) -> Optional[torch.Tensor]:
        """num_pages whole FP4 pages as unified slots [num_pages * page_size] (None when the FP4 pool is short)."""
        out = self.fp4.alloc(num_pages * self.page_size)
        return None if out is None else out + self.n8

    # ---- free ------------------------------------------------------------------------------------------------
    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        m = free_index < self.boundary
        if bool(m.all()):
            self.fp8.free(free_index)
        elif not bool(m.any()):
            self.fp4.free(free_index - self.n8)
        else:
            self.fp8.free(free_index[m])
            self.fp4.free(free_index[~m] - self.n8)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        if free_index.numel() == 0:
            return
        assert start_pos % self.page_size == 0, f"segment start {start_pos} is not page-aligned"
        self.free_page_ids(free_index[:: self.page_size] // self.page_size)

    def free_page_ids(self, page_ids: torch.Tensor):
        if page_ids.numel() == 0:
            return
        m = page_ids <= self.fp8_pages
        if bool(m.all()):
            self.fp8.free_page_ids(page_ids)
        elif not bool(m.any()):
            self.fp4.free_page_ids(page_ids - self.fp8_pages)
        else:
            self.fp8.free_page_ids(page_ids[m])
            self.fp4.free_page_ids(page_ids[~m] - self.fp8_pages)

    def free_group_begin(self):
        assert not self._grouping, "free groups cannot be nested"
        self._grouping = True
        self.fp8.free_group_begin()
        self.fp4.free_group_begin()

    def free_group_end(self):
        self._grouping = False
        self.fp8.free_group_end()
        self.fp4.free_group_end()

    def clear(self):
        if hasattr(self, "fp8"):
            self.fp8.clear()
            self.fp4.clear()
        self.free_group = None

    def resize(self, config) -> None:
        raise RuntimeError("kvmix allocator: resize is not supported (the pool split is fixed at boot)")


# ------------------------------------------------------------------------------------------------ policy


def current_len(req, phase: str) -> int:
    """The length that decides a request's precision: the whole prompt (known before prefill) or the KV it holds."""
    n = len(getattr(req, "origin_input_ids", ()) or ())
    kv = getattr(req, "kv", None)
    if kv is not None:
        n = max(n, int(getattr(kv, "kv_committed_len", 0)), int(getattr(kv, "kv_allocated_len", 0)))
    if phase == "extend":
        er = getattr(req, "extend_range", None)
        if er is not None:
            n = max(n, int(getattr(er, "end", 0)))
    else:
        n += 1  # the token about to be allocated
    return n


def will_be_fp4(req, thr: int, phase: str) -> bool:
    return bool(getattr(req, "kvmix_fp4", False)) or current_len(req, phase) >= thr


def before_alloc(allocator: KVMixAllocator, tree_cache, reqs: Sequence, req_to_token_pool, phase: str) -> None:
    """Decide each request's pool for the coming allocation and promote newly-long requests. `phase` is "extend"
    (alloc_for_extend: rows come from req.prefix_indices) or "decode" (rows come from req_to_token)."""
    flags: List[bool] = []
    need = [0, 0]
    ps = allocator.page_size
    for req in reqs:
        fp4 = will_be_fp4(req, allocator.threshold, phase)
        er = getattr(req, "extend_range", None)
        need[int(fp4)] += (int(getattr(er, "length", 0)) if (phase == "extend" and er is not None) else 1) + ps
        if fp4 and not getattr(req, "kvmix_fp4", False):
            req.kvmix_fp4 = True
            req.kvmix_promoted = False
            logger.warning("numinous kvmix: request %s crosses %d tokens (%s); new KV pages are FP4",
                        getattr(req, "rid", "?"), allocator.threshold, phase)
        if fp4 and allocator.promote and not getattr(req, "kvmix_promoted", False):
            req.kvmix_promoted = promote_request(allocator, tree_cache, req, req_to_token_pool, phase)
        flags.append(fp4)
    allocator.pending = flags
    allocator.pending_need = (need[0], need[1])


def _path_nodes(tree_cache, req) -> list:
    """Radix nodes from the root's child down to req.last_node (empty for chunk caches / no match)."""
    node = getattr(req, "last_node", None)
    root = getattr(tree_cache, "root_node", None)
    if node is None or root is None or node is root:
        return []
    path = []
    while node is not None and node is not root:
        path.append(node)
        node = node.parent
    path.reverse()
    return path


def _supports_node_promotion(tree_cache) -> bool:
    """True for the plain RadixCache (node values are the request's slots, rewritable in place). Other caches
    (UnifiedRadixCache with its tree core and components, HiRadixCache) keep their cached prefixes FP8: only the
    request's own pages beyond cache_protected_len are promoted."""
    if tree_cache is None or tree_cache.is_chunk_cache() or getattr(tree_cache, "disable", False):
        return False
    from sglang.srt.mem_cache.radix_cache import RadixCache

    if type(tree_cache) is not RadixCache:
        log_once("tree_kind", "numinous kvmix: %s is not the plain RadixCache; promotion covers a request's own pages "
                 "only, cached prefixes stay in the pool they were written to", type(tree_cache).__name__)
        return False
    return True


def promote_request(allocator: KVMixAllocator, tree_cache, req, req_to_token_pool, phase: str) -> bool:
    """Requantize the request's exclusive FP8 pages to FP4. Returns True (done for this request: promoted, nothing
    to promote, or no FP4 room, in which case its existing pages stay FP8)."""
    ps = allocator.page_size
    kv = req.kv
    holds_row = kv.holds_kv and kv.req_pool_idx is not None
    if phase == "extend":
        row = req.prefix_indices
        if row is None or row.numel() == 0:
            return True
        row = row.to(allocator.device)
        L = int(row.numel())
    else:
        if not holds_row:
            return True
        L = int(kv.kv_allocated_len)
        if L == 0:
            return True
        row = req_to_token_pool.req_to_token[kv.req_pool_idx, :L].to(torch.int64)
    n_pages = (L + ps - 1) // ps
    first_slots = row[::ps].tolist()  # one sync per promotion event (rare: once per long request)
    assert len(first_slots) == n_pages
    promote = [0 < s < allocator.boundary for s in first_slots]  # only FP8 pages move
    covered = 0
    node_pages: list = []  # (node, page_lo, page_hi)
    protected = int(kv.cache_protected_len)
    if not _supports_node_promotion(tree_cache):
        # tree-owned pages (positions [0, protected)) belong to the cache and stay: only own pages move
        for j in range(0, min(n_pages, (protected + ps - 1) // ps)):
            promote[j] = False
        covered = protected
    else:
        pos = 0
        for node in _path_nodes(tree_cache, req):
            n_len = len(node.key)
            if n_len == 0 or node.value is None:
                continue
            assert n_len % ps == 0, f"kvmix: radix node of {n_len} tokens is not page-aligned (page {ps})"
            lo, hi = pos // ps, min((pos + n_len) // ps, n_pages)
            if node.lock_ref > 1 or n_len < allocator.min_promote:
                for j in range(lo, hi):
                    promote[j] = False  # shared with another running request, or too small to matter
            else:
                node_pages.append((node, lo, hi))
            pos += n_len
        covered = pos
        if protected and pos != protected:
            log_once("protected", "numinous kvmix: path length %d != cache_protected_len %d for request %s; "
                     "promoting only the request's own pages", pos, protected, getattr(req, "rid", "?"))
            for j in range(0, min(n_pages, (max(pos, protected) + ps - 1) // ps)):
                promote[j] = False
            node_pages = []
    pages = [j for j in range(n_pages) if promote[j]]
    if not pages:
        return True
    new_slots = allocator.alloc_fp4_slots(len(pages))
    if new_slots is None:
        allocator._evict_split(tree_cache, 0, len(pages) * ps)
        new_slots = allocator.alloc_fp4_slots(len(pages))
    if new_slots is None:
        # no room: leave the request's existing pages FP8 (its new pages still prefer FP4). Not retried: a retry
        # per step would cost one host sync per pending request per step.
        allocator.stats["promotion_no_fp4_pages"] += 1
        log_once("nofp4", "numinous kvmix: FP4 pool has no room to promote %d pages; the request keeps its FP8 pages",
                 len(pages))
        return True
    new_slots = new_slots.view(len(pages), ps)
    src_pages = torch.tensor([first_slots[j] // ps for j in pages], dtype=torch.int64, device=allocator.device)
    src_slots = (src_pages[:, None] * ps + torch.arange(ps, device=allocator.device)).reshape(-1)
    new_row = row.clone()
    page_idx = torch.tensor(pages, dtype=torch.int64, device=allocator.device)
    # positions of the promoted pages inside the row: page j covers [j * ps, min((j + 1) * ps, L))
    pos = (page_idx[:, None] * ps + torch.arange(ps, device=allocator.device)).reshape(-1)
    keep = pos < L
    new_row[pos[keep]] = new_slots.reshape(-1)[keep]
    stream = torch.cuda.current_stream() if allocator.device != "cpu" and torch.cuda.is_available() else None
    if stream is not None and allocator.forward_stream is not None:
        stream.wait_stream(allocator.forward_stream)  # the previous step may still be writing this row's tail
    allocator.get_kvcache().kvmix_promote_pages(src_slots, new_slots.reshape(-1))
    allocator.free_page_ids(src_pages)
    for node, lo, hi in node_pages:
        a, b = lo * ps, min(hi * ps, L)
        if b > a:
            node.value[: b - a] = new_row[a:b].to(node.value.dtype)
    if holds_row:
        req_to_token_pool.write((kv.req_pool_idx, slice(0, L)), new_row.to(req_to_token_pool.req_to_token.dtype))
    if phase == "extend":
        req.prefix_indices = new_row.to(req.prefix_indices.dtype)
    allocator.stats["promotions"] += 1
    allocator.stats["promoted_pages"] += len(pages)
    logger.warning("numinous kvmix: promoted %d of %d pages of request %s (%d tokens, %d covered by the tree) to FP4",
                   len(pages), n_pages, getattr(req, "rid", "?"), L, covered)
    return True
