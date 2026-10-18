"""rtxsys D2-b hook (Gemma 4 31B on H100, kernel program 2026-10-06): fused softmax top-1 in the frozen-KV MTP draft.

FrozenKVMTPDraftWorker.draft_forward (frozen_kv_mtp_worker_v2.py:577-578 and :617-618, v0.5.20) runs softmax over
[bs, 262144] then fast_topk(probs, 1) on every draft step. With NUMINOUS_DRAFT_TOP1=1 and topk == 1, both become
operations/kernels/gemma31/rtx_draft_top1.py (installed as srt/layers/numinous_draft_top1.py).
- The kernel reads the logits once, split across CTAs, and gives (1 / sum exp(x - max), argmax) in fp32.
- That is the same pair the stock path produces, up to fp32 summation order.
- The draft runs inside the frozen-KV MTP CUDA graph. The kernels allocate and launch only: no host sync.

Measured rows it removes (rtxv41 prof_decode32_composed, 32 sessions): the 5 draft cunn_SoftMaxForward calls (part
of 0.862 ms for 6 calls) and the 5 MaxOps reduces (0.117 ms). Derived: ~-0.6 ms per verify cycle at 32 sessions.

Hooks (anchored replace, idempotent, the patched files must compile): the seed-iteration and the step-loop softmax +
fast_topk pairs (frozen_kv_mtp_worker_v2.py), and the assistant head (gemma4_mtp.py forward's logits_processor call,
D2-c, env NUMINOUS_DRAFT_HEAD_FUSED=1; NUMINOUS_DRAFT_HEAD_FP8=1 for the FP8 head copy).

    python3 gemma31_draft_top1_hooks.py [--root <sglang pkg dir>] [--check]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sglang_numinous_hooks import sglang_root  # noqa: E402

MARKER = "numinous-hook:gemma31_draft_top1"
ENV = "NUMINOUS_DRAFT_TOP1"
PATH = "srt/speculative/frozen_kv_mtp_worker_v2.py"
INSTALL = [("../kernels/gemma31/rtx_draft_top1.py", "srt/layers/numinous_draft_top1.py")]
CELL_INSTALL = [("operations/kernels/gemma31/rtx_draft_top1.py", "srt/layers/numinous_draft_top1.py")]

MTP = "srt/models/gemma4_mtp.py"
HOOKS = [
    ("seed",
     "        probs = torch.softmax(seed_next_logits, dim=-1)\n"
     "        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)\n",
     "        from sglang.srt.layers import numinous_draft_top1 as _ndt  # {marker} (seed)\n\n"
     "        _ndt_out = _ndt.draft_top1(seed_next_logits, self.topk)\n"
     "        if _ndt_out is not None:\n"
     "            topk_p, topk_index = _ndt_out\n"
     "        else:\n"
     "            probs = torch.softmax(seed_next_logits, dim=-1)\n"
     "            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)\n"),
    ("step",
     "            probs = torch.softmax(logits_output.next_token_logits, dim=-1)\n"
     "            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)\n",
     "            _ndt_out = _ndt.draft_top1(logits_output.next_token_logits, self.topk)  # {marker} (step)\n"
     "            if _ndt_out is not None:\n"
     "                topk_p, topk_index = _ndt_out\n"
     "            else:\n"
     "                probs = torch.softmax(logits_output.next_token_logits, dim=-1)\n"
     "                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)\n"),
    ("head",
     "        return self.logits_processor(\n"
     "            input_ids,\n"
     "            hidden_states,\n"
     "            self.lm_head,\n"
     "            forward_batch,\n"
     "            hidden_states_before_norm=projected_states,\n"
     "        )\n",
     "        from sglang.srt.layers import numinous_draft_top1 as _ndt  # {marker} (head)\n"
     "\n"
     "        _ndt_head = _ndt.draft_head(\n"
     "            self, input_ids, hidden_states, projected_states, forward_batch\n"
     "        )\n"
     "        if _ndt_head is not None:\n"
     "            return _ndt_head\n"
     "        return self.logits_processor(\n"
     "            input_ids,\n"
     "            hidden_states,\n"
     "            self.lm_head,\n"
     "            forward_batch,\n"
     "            hidden_states_before_norm=projected_states,\n"
     "        )\n"),
]
FILE_OF = {"seed": PATH, "step": PATH, "head": MTP}


def numinous_preflight(ctx) -> list:
    """Where NUMINOUS_DRAFT_TOP1=1 (and the D2-c head envs) never engage."""
    env = ctx.env or {}
    if env.get(ENV) != "1":
        if env.get("NUMINOUS_DRAFT_HEAD_FUSED") == "1":
            return ["NUMINOUS_DRAFT_HEAD_FUSED=1 needs NUMINOUS_DRAFT_TOP1=1 (the worker consumes the head's top-1): inert"]
        return []
    out, a = [], ctx.args or {}
    algo = str(a.get("speculative_algorithm") or "").upper()
    if algo not in ("NEXTN", "FROZEN_KV_MTP"):
        out.append(f"{ENV}=1 with speculative algorithm {algo or 'none'}: the hook patches the frozen-KV MTP draft only")
    if int(a.get("speculative_eagle_topk") or 1) != 1:
        out.append(f"{ENV}=1 with --speculative-eagle-topk {a.get('speculative_eagle_topk')}: topk == 1 only, inert")
    if env.get("NUMINOUS_DRAFT_HEAD_FP8") == "1" and env.get("NUMINOUS_DRAFT_HEAD_FUSED") != "1":
        out.append("NUMINOUS_DRAFT_HEAD_FP8=1 without NUMINOUS_DRAFT_HEAD_FUSED=1: inert")
    return out


def apply(root: str, check: bool = False) -> list:
    out = []
    if not check:
        for rel, dst in INSTALL:
            shutil.copy(os.path.normpath(os.path.join(HERE, rel)), os.path.join(root, dst))
            out.append(f"installed {dst}")
    for rel in (PATH, MTP):
        path = os.path.join(root, rel)
        src = open(path).read()
        new = src
        for name, anchor, text in HOOKS:
            if FILE_OF[name] != rel:
                continue
            tag = f"{MARKER} ({name})"
            if tag in new:
                out.append(f"{name}: already applied")
                continue
            n = new.count(anchor)
            if n != 1:
                raise SystemExit(f"gemma31_draft_top1 hooks: {name} anchor found {n} times in {rel}")
            new = new.replace(anchor, text.replace("{marker}", MARKER), 1)
            out.append(f"{name}: {'anchor ok' if check else 'applied to ' + rel}")
        if not check and new != src:
            compile(new, path, "exec")
            tmp = f"{path}.numinous.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for line in apply(a.root or sglang_root(), a.check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
