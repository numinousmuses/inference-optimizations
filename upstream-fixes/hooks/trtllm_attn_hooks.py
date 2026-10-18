"""Make --dsv4-attn-backend trtllm serve DeepSeek V4 Pro with DSPARK speculation (SGLang v0.5.20, SM100).

Root cause (proattn, 2026-09-28): the served Pro cell (DP8 attention + DSPARK) fails to boot with the trtllm
DSV4 backend because the DSpark draft worker's TARGET_VERIFY graph capture builds draft-block metadata whose
SWA table is the compact DSpark layout (get_dspark_swa_page_indices): per row, the first swa_topk_lengths[i]
columns are live token rows (the causal window, then the draft block), -1 after, width
ceil64(SWA_WINDOW + gamma) = 192. Upstream's init_trtllm_sparse_buffers asserts the standard causal shape
(num_tokens, SWA_WINDOW) and dies: "AssertionError ... line 442" -> rank scheduler exits -3 with no visible
traceback (captured with PYTHONFAULTHANDLER on the b200 box, /root/proattn_repro.log).

Kernel facts the fix builds on (measured: operations/modal/proattn_trtllm_kernel_probe.py, B200,
flashinfer 0.6.18 trtllm-gen):
  P1  SWA validity is clipped solely by seq_lens (entries past seq_lens are ignored)      rel_err 0.0067
  P2  -1 columns inside the seq_lens window ARE gathered (garbage), never skipped         rel_err 0.53
  P3  compact table + seq_lens := live count + lens := max(count, 128) is exact,
      including rows whose live entries straddle column 128                               rel_err 0.0069
  P4  columns >= 128 read the compressed cache argument; for compress_ratio 0 that is
      the same SWA pool view, so straddling block entries resolve correctly               rel_err 0.0074
  P5  gather order does not matter                                                        rel_err 3e-6

So the compact DSpark table serves the trtllm-gen kernel verbatim when the per-row live count is passed as
seq_lens and the lens include the fixed 128 SWA slots. Three anchored hooks (marker, idempotent, loud,
compile-checked; style of sglang_numinous_hooks.py):

  trtllm_dspark_tables   init_trtllm_sparse_buffers accepts the wide table: tile-aligned parents (the VarSeq
                         kernel reads table rows to the 64-row boundary), per-row lens clamp(count, min=128)
                         in trtllm_swa_lens; covers graph-capture prep and the eager draft-block path.
  trtllm_dspark_ingraph  init_forward_metadata_in_graph rebuilds the DSpark tables inside the captured graph
                         (recorded ops); give those the same tile-aligned parents and per-row lens so replays
                         refresh them.
  trtllm_dspark_decode   _forward_trtllm_decode serves the wide table directly: seq_lens := live counts
                         (metadata.swa_topk_lengths), lens := trtllm_swa_lens; compress_ratio 0 only, which
                         is the only ratio the draft-block path produces (need_compress=False).

No behavior change without DSPARK or without --dsv4-attn-backend trtllm: every hook is conditioned on the
wide table (only the DSpark draft builds one) or on self.trtllm_attn.

    python3 trtllm_attn_hooks.py                     # patch the installed sglang (v0.5.20)
    python3 trtllm_attn_hooks.py --root /tmp/sgl --check
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

BACKEND = "srt/layers/attention/deepseek_v4_backend.py"
TRTLLM = "srt/layers/attention/deepseek_v4_trtllm_backend.py"


@dataclass(frozen=True)
class Hook:
    name: str
    path: str
    anchor: str
    text: str

    @property
    def marker(self) -> str:
        return f"# numinous-hook:{self.name}"


TABLES = '''        # numinous-hook:trtllm_dspark_tables
        # Numinous: DSpark draft-block metadata (get_dspark_swa_page_indices) is a
        # compact table: per row, the first swa_topk_lengths[i] columns are live
        # token rows (causal window, then the draft block), -1 after, width
        # ceil64(SWA_WINDOW + gamma). trtllm-gen clips SWA validity solely by
        # seq_lens and does not skip -1 columns (measured:
        # operations/modal/proattn_trtllm_kernel_probe.py, B200, flashinfer
        # 0.6.18), so the table serves verbatim with per-row lens
        # max(count, SWA_WINDOW) and the live count as the kernel's seq_lens;
        # columns >= SWA_WINDOW read the compressed cache argument, which
        # _trtllm_kv_cache_views maps to the same SWA pool for compress_ratio 0,
        # the only ratio on this path (need_compress=False).
        if self.swa_page_indices.shape[1] != SWA_WINDOW:
            width = self.swa_page_indices.shape[1]
            assert self.swa_page_indices.shape == (num_tokens, width) and width % 64 == 0, (
                f"{self.swa_page_indices.shape=} {num_tokens=}"
            )
            assert self.c4_sparse_page_indices is None and self.c128_page_indices is None, (
                "a compact DSpark table carries no c4/c128 region"
            )
            assert self.swa_topk_lengths is not None
            n_pad = (num_tokens + 63) // 64 * 64

            def _tile_padded_wide(fill, src=None, w=None):
                shape = (n_pad,) if w is None else (n_pad, w)
                buf = torch.full(shape, fill, **self.cuda_int32_kwargs)
                if src is not None:
                    buf[:num_tokens].copy_(src)
                return buf[:num_tokens]

            if n_pad != num_tokens:
                self.seq_lens_casual = _tile_padded_wide(1, self.seq_lens_casual)
                self.swa_page_indices = _tile_padded_wide(-1, self.swa_page_indices, w=width)
                self.swa_topk_lengths = _tile_padded_wide(1, self.swa_topk_lengths)
            self.trtllm_swa_lens = _tile_padded_wide(
                SWA_WINDOW, self.swa_topk_lengths.to(torch.int32).clamp(min=SWA_WINDOW)
            )
            return
'''

INGRAPH = '''                # numinous-hook:trtllm_dspark_ingraph
                # Numinous: the trtllm-gen VarSeq kernel reads table rows and lens
                # vectors to the 64-row tile boundary, so the DSpark tables rebuilt
                # inside the captured graph get tile-aligned parents here (recorded
                # ops: replays refresh their contents); per-row lens and live-count
                # views land in the fields the trtllm decode forward reads.
                if self.trtllm_attn:
                    _cam = metadata.core_attn_metadata
                    _nt, _w = swa_page_indices.shape
                    _npad = (_nt + 63) // 64 * 64
                    _tbl = torch.full((_npad, _w), -1, **_cam.cuda_int32_kwargs)
                    _tbl[:_nt].copy_(swa_page_indices)
                    _cam.swa_page_indices = _tbl[:_nt]
                    _cnt = torch.full((_npad,), 1, **_cam.cuda_int32_kwargs)
                    _cnt[:_nt].copy_(swa_topk_lengths)
                    _cam.swa_topk_lengths = _cnt[:_nt]
                    _lens = torch.full((_npad,), SWA_WINDOW, **_cam.cuda_int32_kwargs)
                    torch.clamp(_cnt[:_nt], min=SWA_WINDOW, out=_lens[:_nt])
                    _cam.trtllm_swa_lens = _lens[:_nt]
'''

DECODE = '''        # numinous-hook:trtllm_dspark_decode
        # Numinous: DSpark draft-block rows carry the compact wide table (see the
        # trtllm_dspark_tables hook); serve it directly with the live count as
        # seq_lens. Only compress_ratio 0 exists on the draft-block path; both
        # cache arguments then view the same SWA pool, so live entries that
        # straddle column SWA_WINDOW resolve correctly (probe P3/P4).
        if core_attn_metadata.swa_page_indices.shape[1] != SWA_WINDOW:
            assert compress_ratio == 0, (
                f"DSpark compact tables serve only the SWA/c0 route, got {compress_ratio=}"
            )
            sparse_indices = core_attn_metadata.swa_page_indices
            sparse_topk_lens = core_attn_metadata.trtllm_swa_lens
            seq_lens = core_attn_metadata.swa_topk_lengths
            assert sparse_topk_lens is not None and seq_lens.dtype == torch.int32
            if sparse_indices.shape[0] != bs:
                assert sparse_indices.shape[0] > bs, f"{sparse_indices.shape=} {bs=}"
                sparse_indices = sparse_indices[:bs]
            if sparse_topk_lens.shape[0] != bs:
                sparse_topk_lens = sparse_topk_lens[:bs]
            if seq_lens.shape[0] != bs:
                seq_lens = seq_lens[:bs]
            swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
                layer.layer_id, 0
            )
            q_fp8 = q.to(torch.float8_e4m3fn).view(bs, 1, num_heads, 512)
            bmm1_scale, bmm2_scale = self._get_trtllm_bmm_scales(layer)
            assert attn_sink is not None and attn_sink.dtype == torch.float32
            assert self.trtllm_workspace_buffer is not None
            _check_trtllm_query_rows(bs)
            out = trtllm_batch_decode_sparse_mla_dsv4(
                query=q_fp8,
                swa_kv_cache=swa_kv_cache,
                workspace_buffer=self.trtllm_workspace_buffer,
                sparse_indices=sparse_indices,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens,
                seq_lens=seq_lens,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=attn_sink,
                kv_layout="HND",
            )
            if out_pad_tail is not None:
                out_pad_tail[:bs] = out.view(bs, num_heads, 512)
                return out_pad_tail
            return out.view(bs, num_heads, 512)
'''

HOOKS = [
    Hook(
        "trtllm_dspark_tables",
        BACKEND,
        '        Only the c4 tail and lens are filled per layer.\n'
        '        """\n'
        "\n"
        "        num_tokens = self.seq_lens_casual.shape[0]\n",
        TABLES,
    ),
    Hook(
        "trtllm_dspark_ingraph",
        BACKEND,
        "                metadata.core_attn_metadata.swa_page_indices = swa_page_indices\n"
        "                metadata.core_attn_metadata.swa_topk_lengths = swa_topk_lengths\n",
        INGRAPH,
    ),
    Hook(
        "trtllm_dspark_decode",
        TRTLLM,
        "            bs = n_meta_rows\n",
        DECODE,
    ),
]


def sglang_root() -> str:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable")
    return list(spec.submodule_search_locations)[0]


def patch_source(src: str, path: str = BACKEND) -> tuple[str, list[str]]:
    """Apply every hook of one file to its text. Returns (new text, status lines); SystemExit on a bad anchor."""
    out, missing = [], []
    for h in HOOKS:
        if h.path != path:
            continue
        if h.marker + "\n" in src:
            out.append(f"{h.name}: already applied")
            continue
        n = src.count(h.anchor)
        if n != 1:
            missing.append(f"{h.name}: anchor found {n} times in {h.path}")
            continue
        src = src.replace(h.anchor, h.anchor + h.text, 1)
        out.append(f"{h.name}: applied to {h.path}")
    if missing:
        raise SystemExit("trtllm_attn hooks: " + "; ".join(missing))
    return src, out


def apply(root: str, check: bool = False) -> list[str]:
    lines = []
    for path in (BACKEND, TRTLLM):
        full = os.path.join(root, path)
        src = open(full).read()
        new, out = patch_source(src, path)
        if check:
            lines += [ln.replace("applied to", "anchor ok in") for ln in out]
            continue
        compile(new, full, "exec")  # never leave a file that does not parse
        if new != src:
            tmp = f"{full}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, full)
        lines += out
    return lines


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
