"""Skip the q concat of absorbed MLA on DSA prefill: the w_kc bmm writes q_nope_out straight into a [T, H, 576] buffer,
the roped q_pe is copied into its last 64 columns, and the DSA backend takes the concatenated q with q_rope=None (its
cat-skip path, dsa_backend.py:1942-1953). rtxattn, Hopper kernel program 2026-10-05, lever "GLM fusions".

SUPERSEDED (duplicate): amdglm built the same lever first as operations/patches/glm_absorb_q_fold_hooks.py
(NUMINOUS_ABSORB_Q_FOLD=1, engages at T >= 512, green in operations/kernels/glm_kernel_set.py row absorb_q_concat). Do
not apply both: they patch the same forward_mla lines. This file stays as the independent measurement (claim
glm.h200.mla_q_cat_skip: -48.3 us/layer at 2,048 tokens, bit-identical; amdglm measured 36% of the 0.1355 ms/layer
stock).

Profile row (measured: M5-spec314 rung-32 cold-wave prefill, rank 4, 8 forwards of 2,048 tok/rank):
sglang::concat_mla_absorb_q_kernel, 5.89 ms per prefill forward = 78 calls x 75 us. It reads q_nope_out [T,64,512] and
q_pe [T,64,64] and writes [T,64,576], all bf16, from the concat_mla_absorb_q_general call before
_forward_flashmla_kv (dsa_backend.py:2122-2124). The bmm's write into the strided view costs the same bytes as today's
contiguous write. The replacement copies only the 64 rope columns (~1/9 of the concat's traffic).

Hooks (anchored replace, idempotent by marker, fail loudly on a missing/repeated anchor, compile-checked). They are
inert unless NUMINOUS_MLA_Q_CAT=1, and they act only on EXTEND / MIXED (not verify, not draft, not under CUDA-graph
capture). They also need DSA layers with rope (qk_rope_head_dim > 0), full local heads (no DCP gather) and the bf16
w_kc bmm:
  bmm   forward_mla.py absorbed prepare: torch.bmm(q_nope^T, w_kc, out=<q_cat[:, :, :512] transposed>) and stash
        (q_cat, its data_ptr) on the attention module
  core  forward_mla.forward_absorb_core: when the stash matches the q_nope_out it is handed, copy q_pe into
        q_cat[..., 512:] and call attn_mqa(q_cat, ..., q_rope=None, k_rope=k_pe). Any mismatch runs the stock call.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import py_compile
import sys
from dataclasses import dataclass

MARKER = "numinous-hook:mla_q_cat_skip"
ENV = "NUMINOUS_MLA_Q_CAT"
REL = "srt/models/deepseek_common/attention_forward_methods/forward_mla.py"


@dataclass(frozen=True)
class Replace:
    name: str
    rel: str
    old: str
    new: str


BMM_OLD = '''            else:
                q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

            q_nope_out = q_nope_out.transpose(0, 1)
'''

BMM_NEW = '''            else:
                # {marker}: on DSA prefill, write the absorbed q_nope straight into the 576-wide q buffer
                self._nm_q_cat = None
                if (
                    __import__("os").environ.get("''' + ENV + '''") == "1"
                    and getattr(self, "use_dsa", False)
                    and self.qk_rope_head_dim > 0
                    and q_nope.dim() == 3
                    and q_nope.shape[1] == self.num_local_heads
                    and forward_batch.forward_mode.is_extend_without_speculative()
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    _nm_cat = q_nope.new_empty(
                        q_nope.shape[0],
                        self.num_local_heads,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                    )
                    _nm_view = _nm_cat[:, :, : self.kv_lora_rank].transpose(0, 1)
                    torch.bmm(q_nope.transpose(0, 1), self.w_kc, out=_nm_view)
                    q_nope_out = _nm_view
                    self._nm_q_cat = (_nm_cat, _nm_cat.data_ptr())
                else:
                    q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

            q_nope_out = q_nope_out.transpose(0, 1)
'''

CORE_OLD = '''            else:
                attn_output = self.attn_mqa(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
'''

CORE_NEW = '''            elif (
                getattr(self, "_nm_q_cat", None) is not None
                and q_nope_out is not None
                and q_nope_out.data_ptr() == self._nm_q_cat[1]
                and q_nope_out.shape[0] == self._nm_q_cat[0].shape[0]
                and not extra_args
                and q_pe is not None
            ):
                # {marker}: q_nope_out already sits in the 576-wide buffer; add the roped part, skip the concat
                _nm_cat = self._nm_q_cat[0]
                self._nm_q_cat = None
                _nm_cat[:, :, self.kv_lora_rank :].copy_(q_pe.view(_nm_cat.shape[0], _nm_cat.shape[1], -1))
                attn_output = self.attn_mqa(
                    _nm_cat,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=None,
                    k_rope=k_pe,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
''' + CORE_OLD

HOOKS = [Replace("bmm", REL, BMM_OLD, BMM_NEW), Replace("core", REL, CORE_OLD, CORE_NEW)]


def apply(root: str, check: bool = False) -> list[str]:
    out = []
    base = pathlib.Path(root)
    path = base / REL
    text = path.read_text()
    for h in HOOKS:
        tag = f"{MARKER} ({h.name})"
        if tag in text:
            out.append(f"{h.name}: already applied")
            continue
        n = text.count(h.old)
        if n != 1:
            raise RuntimeError(f"{h.name}: anchor found {n} times in {h.rel} (need exactly 1)")
        if check:
            out.append(f"{h.name}: anchor ok in {h.rel}")
            continue
        text = text.replace(h.old, h.new.replace("{marker}", tag))
        out.append(f"{h.name}: applied to {h.rel}")
    if not check and any("applied to" in o for o in out):
        tmp = path.with_suffix(path.suffix + f".numinous.{os.getpid()}")
        tmp.write_text(text)
        py_compile.compile(str(tmp), doraise=True)
        os.replace(tmp, path)
    return out


def _default_root() -> str:
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang not importable; pass --root")
    return list(spec.submodule_search_locations)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or _default_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
