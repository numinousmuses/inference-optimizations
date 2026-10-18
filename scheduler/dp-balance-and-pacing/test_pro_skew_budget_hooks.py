"""CPU unit tests for the balanced prefill budget (pro_skew_budget_hooks.py).

Tests:
1. Pure cap computation (no GPU, no imports beyond stdlib)
2. Anchor verification against the installed sglang package
3. Parse-safety of the module source
"""
from __future__ import annotations

import importlib
import os
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---- 1. Pure cap computation tests ----------------------------------------

def _load_module_source():
    """Extract MODULE_SOURCE from the hook file and exec it in a namespace."""
    import pro_skew_budget_hooks as hooks
    ns = {}
    exec(hooks.MODULE_SOURCE, ns)
    return ns


@pytest.fixture
def mod():
    return _load_module_source()


class TestParseFloor:
    def test_none(self, mod):
        assert mod["parse_floor"](None) == 0

    def test_empty(self, mod):
        assert mod["parse_floor"]("") == 0

    def test_zero(self, mod):
        assert mod["parse_floor"]("0") == 0

    def test_one_is_default(self, mod):
        assert mod["parse_floor"]("1") == mod["DEFAULT_FLOOR"]

    def test_explicit(self, mod):
        assert mod["parse_floor"]("512") == 512

    def test_negative_raises(self, mod):
        with pytest.raises(ValueError):
            mod["parse_floor"]("-1")


class TestComputeCap:
    def setup_method(self):
        self.mod = _load_module_source()
        self.compute_cap = self.mod["compute_cap"]

    def test_no_floor(self):
        """Floor 0 means off."""
        assert self.compute_cap([0, 0, 0, 0, 0, 0, 0, 4096], 4096, 0) is None

    def test_this_rank_not_extending(self):
        """If this rank is not extending, no cap."""
        assert self.compute_cap([4096, 0, 0, 0, 0, 0, 0, 0], 0, 512) is None

    def test_lone_extend_caps_to_floor(self):
        """One rank extends 4096, 7 decode. Cap = floor."""
        cap = self.compute_cap([4096, 0, 0, 0, 0, 0, 0, 0], 4096, 512)
        assert cap == 512

    def test_all_extend_no_cap(self):
        """All 8 ranks extend. No decode ranks = no skew to fix."""
        cap = self.compute_cap([4096, 2048, 3000, 1024, 4096, 2048, 3000, 1024], 4096, 512)
        assert cap is None

    def test_two_extend_cap_is_max_other(self):
        """Two ranks extend: this=4096, other=2048. Cap = max(512, 2048) = 2048."""
        cap = self.compute_cap([4096, 2048, 0, 0, 0, 0, 0, 0], 4096, 512)
        assert cap == 2048

    def test_two_extend_cap_when_other_larger(self):
        """Two ranks extend: this=2048, other=4096. Cap = max(512, 4096) = 4096 >= this, so None."""
        cap = self.compute_cap([4096, 2048, 0, 0, 0, 0, 0, 0], 2048, 512)
        assert cap is None  # 4096 >= 2048, no cap

    def test_cap_with_alignment(self):
        """Cap with alignment rounds UP (ceiling)."""
        cap = self.compute_cap([4096, 0, 0, 0, 0, 0, 0, 0], 4096, 500, align=256)
        # floor=500, max_other=0 -> cap=500. ceil-align to 256: ceil(500/256)*256 = 512
        assert cap == 512

    def test_cap_none_when_at_or_above_planned(self):
        """If cap >= this_rank_planned, return None (no cap needed)."""
        cap = self.compute_cap([1024, 0, 0, 0, 0, 0, 0, 0], 1024, 1024)
        assert cap is None

    def test_single_rank(self):
        """With only 1 rank, no cap."""
        cap = self.compute_cap([4096], 4096, 512)
        assert cap is None  # Can't remove this_rank and have others

    def test_two_ranks_one_decode(self):
        """Two ranks: this extends 4096, other decodes 0. Cap = floor."""
        cap = self.compute_cap([4096, 0], 4096, 1024)
        assert cap == 1024

    def test_floor_1_maps_to_default(self):
        """parse_floor('1') returns DEFAULT_FLOOR."""
        floor = self.mod["parse_floor"]("1")
        cap = self.compute_cap([4096, 0, 0, 0, 0, 0, 0, 0], 4096, floor)
        assert cap == self.mod["DEFAULT_FLOOR"]


class TestSchedulerChunkSize:
    def setup_method(self):
        self.mod = _load_module_source()
        self.scs = self.mod["scheduler_chunk_size"]

    def test_no_cap(self):
        """When _cap is None, returns base."""
        self.mod["_cap"] = None
        assert self.scs(4096, 256, 64) == 4096

    def test_applies_cap(self):
        """When _cap is set, returns min(base, aligned_cap)."""
        self.mod["_cap"] = 512
        result = self.scs(4096, 256, 64)
        assert result == 512

    def test_base_none(self):
        """When base is None (chunked prefill disabled), returns None."""
        self.mod["_cap"] = 512
        assert self.scs(None, 256, 64) is None

    def test_alignment(self):
        """Cap is aligned to lcm(page_size, truncation_align_size) using ceiling."""
        self.mod["_cap"] = 500
        result = self.scs(4096, 256, 64)
        # lcm(256, 64) = 256. cap ceil-aligned = max(256, ceil(500/256)*256) = max(256, 512) = 512
        assert result == 512

    def test_cap_larger_than_base(self):
        """Cap larger than base: returns base."""
        self.mod["_cap"] = 8192
        assert self.scs(4096, 256, 64) == 4096


# ---- 2. Anchor verification -----------------------------------------------

def test_anchors_match():
    """Every anchored hook resolves exactly once in the installed sglang package."""
    try:
        import pro_skew_budget_hooks as hooks
        root = hooks.sglang_root()
    except SystemExit:
        pytest.skip("sglang not importable")
    for h in hooks.HOOKS:
        path = os.path.join(root, h.path)
        if not os.path.exists(path):
            pytest.skip(f"{h.path} not found at {path}")
        src = open(path).read()
        if h.anchor == "":
            # End-of-file hook: just check the file exists
            continue
        count = src.count(h.anchor)
        assert count == 1, f"hook {h.name}: anchor in {h.path} found {count} times (expected 1)"


# ---- 3. Module source is valid Python --------------------------------------

def test_module_source_parses():
    """The MODULE_SOURCE compiles without syntax errors."""
    import pro_skew_budget_hooks as hooks
    compile(hooks.MODULE_SOURCE, "numinous_pro_skew_budget.py", "exec")


# ---- 4. Full apply --check on a copy of sglang -----------------------------

def test_apply_check():
    """apply(root, check=True) succeeds on a copy of the installed sglang package."""
    try:
        import pro_skew_budget_hooks as hooks
        root = hooks.sglang_root()
    except SystemExit:
        pytest.skip("sglang not importable")
    # check mode does not write, so we can run it on the installed package
    lines = hooks.apply(root, check=True)
    for line in lines:
        assert "anchor" not in line.lower() or "ok" in line.lower(), f"unexpected: {line}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
