"""PD decode-side radix cache for hybrid SSM/KDA models (GLM-5.3-Flash) in SGLang v0.5.20 (rtxattn, coordinator task
2026-10-05). Design with file:line: research/pd-decode-radix-hybrid.md.

Why (measured): Flash arm F at 32 agent sessions holds 473 warm tok/s with ~10 s TTFT per short turn. Every turn hands
the whole prefix from prefill to decode, because v0.5.20 refuses the decode radix cache for hybrid SSM models
(mem_cache/kv_cache_builder.py:293-297; arm F3 died there).

Why no per-prefix SSM state is needed on decode (source): every handoff ships the request's final recurrent state whole
(disaggregation/prefill.py:1318-1327 mamba slot), the DSA index cache for the full sequence (:1343-1347 / :1398) and the
DSA tail (:1349-1354); only the attention KV starts at decode_prefix_len (prefill.py:385-395, decode.py:1404-1409). A
decode server never resumes compute from a prefix, so its tree caches attention KV only: a FULL-only UnifiedRadixCache.
With no MAMBA component, supports_mamba() is False and v0.5.20 already runs the mamba slots outside the tree, as for the
decode ChunkCache: alloc per request at prealloc (decode.py:1831), free in release_kv_cache (common.py:306-314), the
no-cache admission factor (allocation.py:308-316), no copy-on-match (decode.py:686).

Hooks (anchored replace, idempotent by marker, a missing or repeated anchor fails loudly, a patched file must compile).
All are inert unless NUMINOUS_PD_DECODE_RADIX_HYBRID=1 AND the server is the PD decode role with the decode radix on
(5. skip_validation, amdflash 2026-10-05: srt/arg_groups/model_hook.py handle_mamba_radix_cache skips the mamba
radix-strategy validation there, since no_extra_buffer otherwise lands the page-64 decode role in
validate_mamba_no_buffer):
  1. allow            srt/mem_cache/kv_cache_builder.py: the hybrid-SSM refusal becomes a warning
  2. full_only        srt/mem_cache/registry.py (_create_unified_radix_cache): ComponentType.MAMBA is not added
  3. no_extra_buffer  srt/arg_groups/model_override_base.py (mamba_extra_buffer_of): False. The extra state buffer
                      tracks prefix states FOR THE RADIX (exec_.py:329-333 derives it from "radix on + strategy
                      extra_buffer"); a FULL-only tree stores none. Measured why (arm F5, 2026-10-05 10:15): with the
                      radix on, the derived flag flipped True on decode, HybridMambaDecodeReqToTokenPool sized
                      (size + pre_alloc) x 3 slots (decode.py:271-291) above the configured 160, and the decode role
                      OOMed in pool init (+~5 GB/GPU vs F4)
  4. mamba_ratio      srt/mem_cache/kv_cache_configurator.py (_calculate_mamba_ratio): 1, the no-radix value
                      (:2203-2205), so request capping and spec reservations match a decode server whose tree holds no
                      mamba state
  6. stats_full_only  srt/managers/scheduler_components/pool_stats_observer.py (_get_mamba_token_info): count the
                      FULL-only tree's cached pages as full-evictable. The stock line (:250-253) counts 0 for any tree
                      without mamba support. Root cause of arm F6-radix's 503 (2026-10-05), source_code: the idle
                      invariant (invariant_checker._check_full_pool) saw available = total - 64 with evictable 0 once
                      a finished request cached one page, raised "pool memory leak detected" and killed the decode
                      scheduler. The other accessors are already right for this tree:
                        - invariant protected (protected_size when not supports_mamba);
                        - PrefillAdder (is_hybrid_ssm_cache = supports_mamba, so the generic branch);
                        - decode prealloc (_radix_full_evictable: evictable_size off SWA).

    python3 pd_decode_radix_hybrid_hooks.py                           # patch the installed sglang package
    python3 pd_decode_radix_hybrid_hooks.py --root /tmp/sgl --check   # report which anchors resolve, change nothing
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sglang_numinous_hooks import sglang_root  # noqa: E402

MARKER = "numinous-hook:pd_decode_radix_hybrid"
ENV = "NUMINOUS_PD_DECODE_RADIX_HYBRID"


@dataclass(frozen=True)
class Replace:
    name: str
    path: str
    anchor: str
    text: str


HOOKS = [
    Replace(
        "allow", "srt/mem_cache/kv_cache_builder.py",
        "        if is_hybrid_ssm:\n"
        "            raise ValueError(\n"
        "                \"--disaggregation-decode-enable-radix-cache is incompatible \"\n"
        "                \"with Mamba/SSM models\"\n"
        "            )\n",
        "        if is_hybrid_ssm and __import__(\"os\").environ.get(\"" + ENV + "\") == \"1\":\n"
        "            # {marker}: the decode tree is built FULL-only (registry hook): prefill\n"
        "            # ships the final SSM/conv state whole on every handoff, so only the\n"
        "            # attention KV of a cached prefix is reused on decode.\n"
        "            logger.warning(\n"
        "                \"numinous pd_decode_radix_hybrid: decode-side radix cache on a hybrid \"\n"
        "                \"SSM model with a FULL-only tree (attention KV cached; SSM/conv state \"\n"
        "                \"always transferred from prefill)\"\n"
        "            )\n"
        "        elif is_hybrid_ssm:\n"
        "            raise ValueError(\n"
        "                \"--disaggregation-decode-enable-radix-cache is incompatible \"\n"
        "                \"with Mamba/SSM models\"\n"
        "            )\n",
    ),
    Replace(
        "full_only", "srt/mem_cache/registry.py",
        "    if ctx.is_hybrid_ssm:\n"
        "        tree_components.append(ComponentType.MAMBA)\n",
        "    # {marker}: a PD decode server never resumes compute from a cached prefix\n"
        "    # (prefill ships the final SSM/conv state), so its tree caches attention KV only\n"
        "    _numinous_full_only = (\n"
        "        __import__(\"os\").environ.get(\"" + ENV + "\") == \"1\"\n"
        "        and get_disagg().disaggregation_mode == \"decode\"\n"
        "        and get_disagg().disaggregation_decode_enable_radix_cache\n"
        "    )\n"
        "    if ctx.is_hybrid_ssm and not _numinous_full_only:\n"
        "        tree_components.append(ComponentType.MAMBA)\n",
    ),
    Replace(
        "no_extra_buffer", "srt/arg_groups/model_override_base.py",
        "    return cfg.disable_radix_cache is False and cfg.mamba_radix_cache_strategy in (\n"
        "        \"extra_buffer\",\n"
        "        \"extra_buffer_lazy\",\n"
        "    )\n",
        "    # {marker}: the PD decode role's FULL-only tree stores no mamba prefix state\n"
        "    if (\n"
        "        __import__(\"os\").environ.get(\"" + ENV + "\") == \"1\"\n"
        "        and getattr(cfg, \"disaggregation_mode\", None) == \"decode\"\n"
        "        and getattr(cfg, \"disaggregation_decode_enable_radix_cache\", False)\n"
        "    ):\n"
        "        return False\n"
        "    return cfg.disable_radix_cache is False and cfg.mamba_radix_cache_strategy in (\n"
        "        \"extra_buffer\",\n"
        "        \"extra_buffer_lazy\",\n"
        "    )\n",
    ),
    Replace(
        "mamba_ratio", "srt/mem_cache/kv_cache_configurator.py",
        "    def _calculate_mamba_ratio(self) -> int:\n"
        "        if get_memory().disable_radix_cache:\n"
        "            return 1\n",
        "    def _calculate_mamba_ratio(self) -> int:\n"
        "        if get_memory().disable_radix_cache:\n"
        "            return 1\n"
        "        # {marker}: the PD decode role's FULL-only tree caches no mamba state: no-radix ratio\n"
        "        if (\n"
        "            __import__(\"os\").environ.get(\"" + ENV + "\") == \"1\"\n"
        "            and get_disagg().disaggregation_mode == \"decode\"\n"
        "            and get_disagg().disaggregation_decode_enable_radix_cache\n"
        "        ):\n"
        "            return 1\n",
    ),
    Replace(
        "skip_validation", "srt/arg_groups/model_hook.py",
        "    view = resolved_view(server_args)\n"
        "    if not view.uses_mamba_radix_cache:\n"
        "        return\n"
        "\n"
        "    if mamba_extra_buffer_of(view):\n",
        "    view = resolved_view(server_args)\n"
        "    if not view.uses_mamba_radix_cache:\n"
        "        return\n"
        "    # {marker}: the PD decode role's FULL-only tree stores no mamba prefix state, so neither\n"
        "    # the extra-buffer contract nor the no-buffer one (page_size 1, no overlap) applies there.\n"
        "    # Without this, no_extra_buffer sends a page-64 / overlap decode role into\n"
        "    # validate_mamba_no_buffer (arm F6-radix preflight, 2026-10-05).\n"
        "    if (\n"
        "        __import__(\"os\").environ.get(\"" + ENV + "\") == \"1\"\n"
        "        and getattr(view, \"disaggregation_mode\", None) == \"decode\"\n"
        "        and getattr(view, \"disaggregation_decode_enable_radix_cache\", False)\n"
        "    ):\n"
        "        return\n"
        "\n"
        "    if mamba_extra_buffer_of(view):\n",
    ),
    Replace(
        "stats_full_only", "srt/managers/scheduler_components/pool_stats_observer.py",
        "        full_evictable_size = (\n"
        "            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0\n"
        "        )\n",
        "        full_evictable_size = (\n"
        "            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0\n"
        "        )\n"
        "        # {marker}: the decode role's FULL-only tree caches attention pages but\n"
        "        # supports_mamba() is False, so the stock line counts them as 0 evictable. The idle\n"
        "        # invariant then sees available = total - cached and kills the scheduler (arm F6-radix,\n"
        "        # 2026-10-05: 'pool memory leak detected! [full] total=5657024, available=5656960,\n"
        "        # evictable=0' after one 64-token page was cached), and token usage counts cached pages\n"
        "        # as used.\n"
        "        if (\n"
        "            not is_mamba_radix_cache\n"
        "            and __import__(\"os\").environ.get(\"" + ENV + "\") == \"1\"\n"
        "            and self.tree_cache.is_tree_cache()\n"
        "        ):\n"
        "            full_evictable_size = self.tree_cache.full_evictable_size()\n",
    ),
]


def apply(root: str, check: bool = False) -> list[str]:
    """One status line per hook; SystemExit if an anchor is missing or repeated. All hooks of a file are applied to
    its text together and the file is written once (never half-patched)."""
    out, missing, files = [], [], {}
    for h in HOOKS:
        path = os.path.join(root, h.path)
        src = files.get(path)
        if src is None:
            src = files[path] = open(path).read()
        tag = f"{MARKER} ({h.name})"
        if tag in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        files[path] = src.replace(h.anchor, h.text.replace("{marker}", tag), 1)
        out.append(f"{h.name}: {'anchor ok' if check else 'applied to ' + h.path}")
    if missing:
        raise SystemExit("pd_decode_radix_hybrid hooks: " + "; ".join(missing))
    if not check:
        for path, new in files.items():
            if new == open(path).read():
                continue
            compile(new, path, "exec")  # never leave a file that does not parse
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
