"""Bound the DSA kpool indexer's prefill logits transient (GLM-5.3-Flash) by chunking over query rows: an anchored
hook into SGLang v0.5.20 `layers/attention/dsa/dsa_indexer_kpool.py:_get_topk_ragged_kpool_plan`.

Why (measured 2026-10-04 23:14 UTC, flash-h200-fp8kv cell H, 4xH200, OR shape at 48 sessions; agents/findings/
hopper_fp8kv.md): every rank died with CUDA OOM in `deep_gemm.fp8_mqa_logits` called from `_get_topk_ragged_kpool_plan`
("Tried to allocate 3.37 GiB"; the requests grew 2.45 -> 3.76 GB with context). The stock code materializes the dense
[query rows x pooled key rows] fp32 logits for the WHOLE prefill chunk (16384 rows x ~50K pooled keys = 3.3 GB) in the
memory left outside --mem-fraction-static. The top-k that consumes it is row-wise (per-row lengths, row starts, seq lens,
page-table rows, offsets: kpool_fp8_index.topk_from_pooled_history_logits), so computing the logits and the top-k for
row slices and concatenating is EXACT and bounds the transient to rows_per_chunk x pooled keys x 4 B.

    NUMINOUS_KPOOL_LOGITS_BUDGET_MB=1024 ...   # chunk when the logits would exceed this (unset/0 = stock behaviour)

    python3 kpool_logits_chunk_hooks.py                   # patch the installed sglang
    python3 kpool_logits_chunk_hooks.py --root /tmp/sgl --check

Same contract as the other hook files: exact anchors (each must occur once), a marker makes re-application a no-op,
the patched file must compile before it replaces the original.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import sys
import tempfile

TARGET = "srt/layers/attention/dsa/dsa_indexer_kpool.py"
MARKER = "# numinous: kpool_logits_chunk"

# 1) compute logits lazily when the dense matrix would exceed the budget
ANCHOR_LOGITS = """            k_fp8 = k_u8.view(torch.float8_e4m3fn)
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8[:n_real].contiguous(),
                (k_fp8.contiguous(), k_scale.contiguous()),
                weights[:n_real].contiguous(),
                ks_per_q,
                ke_per_q,
                clean_logits=True,
            )
        else:
            logits = torch.empty((n_real, 0), dtype=torch.float32, device=device)
"""
REPLACE_LOGITS = """            k_fp8 = k_u8.view(torch.float8_e4m3fn)
            _nk_budget = int(os.environ.get("NUMINOUS_KPOOL_LOGITS_BUDGET_MB", "0") or 0) << 20  # numinous: kpool_logits_chunk
            if _nk_budget > 0 and n_real * int(total_k_rows) * 4 > _nk_budget:
                _nk_k = (k_fp8.contiguous(), k_scale.contiguous())
                _nk_q = q_fp8[:n_real].contiguous()
                _nk_w = weights[:n_real].contiguous()
                _nk_rows = max(1, _nk_budget // (int(total_k_rows) * 4))
                _nk_logits = lambda s, e: deep_gemm.fp8_mqa_logits(  # noqa: E731
                    _nk_q[s:e], _nk_k, _nk_w[s:e], ks_per_q[s:e], ke_per_q[s:e], clean_logits=True)
                logits = None
            else:
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8[:n_real].contiguous(),
                    (k_fp8.contiguous(), k_scale.contiguous()),
                    weights[:n_real].contiguous(),
                    ks_per_q,
                    ke_per_q,
                    clean_logits=True,
                )
        else:
            logits = torch.empty((n_real, 0), dtype=torch.float32, device=device)
"""

# 2) row-chunked top-k when the logits were not materialized
ANCHOR_TOPK = """        return self._topk_from_kpool_logits(
            logits,
            pool_lens,
            seq_lens=seq_lens_expanded,
            page_table=page_table_all,
            topk_offsets=topk_offsets_all,
            row_starts=ks_per_q,
            out_rows=total_q,
            page_table_row_index=page_table_row_index_all,
        )

    def _get_topk_ragged_kpool(
"""
REPLACE_TOPK = """        if logits is None:  # numinous: kpool_logits_chunk -- exact: the kpool top-k is row-wise
            _nk_out = []
            for _s in range(0, n_real, _nk_rows):
                _e = min(n_real, _s + _nk_rows)
                _pt = page_table_all
                if _pt is not None and page_table_row_index_all is None and _pt.shape[0] >= n_real:
                    _pt = _pt[_s:_e]
                _nk_out.append(self._topk_from_kpool_logits(
                    _nk_logits(_s, _e),
                    pool_lens[_s:_e],
                    seq_lens=seq_lens_expanded[_s:_e],
                    page_table=_pt,
                    topk_offsets=topk_offsets_all[_s:_e] if topk_offsets_all is not None else None,
                    row_starts=ks_per_q[_s:_e],
                    out_rows=None,
                    page_table_row_index=(page_table_row_index_all[_s:_e]
                                          if page_table_row_index_all is not None else None),
                ))
            _nk_res = torch.cat(_nk_out, dim=0)
            if _nk_res.shape[0] < total_q:
                _nk_pad = torch.full((total_q, _nk_res.shape[1]), -1, dtype=_nk_res.dtype, device=_nk_res.device)
                _nk_pad[: _nk_res.shape[0]] = _nk_res
                _nk_res = _nk_pad
            return _nk_res
        return self._topk_from_kpool_logits(
            logits,
            pool_lens,
            seq_lens=seq_lens_expanded,
            page_table=page_table_all,
            topk_offsets=topk_offsets_all,
            row_starts=ks_per_q,
            out_rows=total_q,
            page_table_row_index=page_table_row_index_all,
        )

    def _get_topk_ragged_kpool(
"""
ANCHOR_IMPORT = "from __future__ import annotations\n"
REPLACE_IMPORT = "from __future__ import annotations\n\nimport os  # numinous: kpool_logits_chunk\n"

EDITS = [(ANCHOR_IMPORT, REPLACE_IMPORT), (ANCHOR_LOGITS, REPLACE_LOGITS), (ANCHOR_TOPK, REPLACE_TOPK)]


def sglang_root() -> str:
    import sglang
    return os.path.dirname(sglang.__file__)


def patch_text(src: str) -> str:
    if MARKER in src:
        return src
    for anchor, repl in EDITS:
        n = src.count(anchor)
        if n != 1:
            raise RuntimeError(f"kpool_logits_chunk: anchor found {n} times (need 1): {anchor[:60]!r}")
        src = src.replace(anchor, repl, 1)
    return src


def apply(root: str, check: bool = False) -> list[str]:
    path = pathlib.Path(root) / TARGET
    src = path.read_text()
    if MARKER in src:
        return [f"kpool_logits_chunk: already applied to {TARGET}"]
    new = patch_text(src)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(new)
    py_compile.compile(f.name, doraise=True)
    if check:
        return [f"kpool_logits_chunk: would apply to {TARGET} (compiles)"]
    path.write_text(new)
    return [f"kpool_logits_chunk: applied to {TARGET}"]


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
