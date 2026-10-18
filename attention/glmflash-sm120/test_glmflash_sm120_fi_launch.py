"""rtxsys launch knobs in glmflash_sm120_fi (per-T cpb table, pad-to-prefill for 41-64-row batches)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")   # as test_glmflash_sm120_fi.py (CPU Triton)

import torch  # noqa: E402

import glmflash_sm120_attn as A  # noqa: E402
import glmflash_sm120_fi as FI  # noqa: E402


class Stub:
    def __init__(self):
        self.calls = []

    def sparse_mla_sm120_decode_dsv3_2(self, q, kv, ip, mo, ml, out, lse, splits, sm, a, b, mt, cpb):
        self.calls.append(("decode", q.shape[0], cpb))
        out.fill_(1.0)
        lse.fill_(2.0)

    def sparse_mla_sm120_paged_attention(self, q, kv, ip, out, lse, sm, mt, *rest):
        self.calls.append(("prefill", q.shape[0], int((ip[q.shape[0] - 1] >= 0).sum())))
        out.fill_(3.0)
        lse.fill_(4.0)


def _run(t, monkeypatch, pad_min=0, table=""):
    monkeypatch.setattr(FI, "DEC_PAD_MIN_T", pad_min)
    monkeypatch.setattr(FI, "DEC_CPB_TABLE", FI.parse_cpb_table(table))
    stub = Stub()
    fi = FI.FlashInferGlm5Next("jit", module=stub, pow2=True)
    q = torch.zeros(t, 64, A.D, dtype=torch.bfloat16)
    kv = torch.zeros(128, 1, A.NSA_ROW, dtype=torch.uint8)
    idx = torch.zeros(t, 2051, dtype=torch.int32)
    out, lse = fi.forward(q, kv, idx, 0.1)
    return stub.calls, out, lse


def test_default_unchanged(monkeypatch):
    calls, out, _ = _run(48, monkeypatch)
    assert calls == [("decode", 48, -1)] and float(out[0, 0, 0]) == 1.0


def test_cpb_table(monkeypatch):
    assert FI.parse_cpb_table("64:16,12:11, 48:8") == [(12, 11), (48, 8), (64, 16)]
    assert _run(10, monkeypatch, table="12:11,48:8,64:16")[0] == [("decode", 10, 11)]
    assert _run(60, monkeypatch, table="12:11,48:8,64:16")[0] == [("decode", 60, 16)]
    assert _run(40, monkeypatch, table="12:11")[0] == [("decode", 40, -1)]


def test_pad_to_prefill(monkeypatch):
    calls, out, lse = _run(48, monkeypatch, pad_min=40)
    assert calls == [("prefill", 65, 0)]            # 65 rows, the dummy rows carry only masked (-1) indices
    assert out.shape[0] == 48 and float(out[47, 0, 0]) == 3.0 and float(lse[0, 0]) == 4.0
    assert _run(40, monkeypatch, pad_min=40)[0][0][0] == "decode"
    assert _run(80, monkeypatch, pad_min=40)[0] == [("prefill", 80, 2051)]   # 2051 real columns, padded to 2112 with -1
