"""Per-layer timing of GLM-5.3 sparse MLA attention: FP8 pool (served TRT-LLM op) vs the FP4 pool read paths.

CUDA-graph replay timing (N launches captured, replayed, events; never eager loops). Shapes are GLM-5.3's: 64 local
heads, top-2048, 576-wide fp8 q, pool slots spread at random over a pool larger than L2. Rows:
  decode  bs 1 / 8 / 16 / 32 / 48 (one token per request)
  verify  bs 16 / 48 x 4 draft tokens (192 rows: the DP8 standard cell's largest graph)
  prefill 512 / 2048 tokens (two requests, 64k context each: the sequence path dequantizes 128k rows once)
Paths: fp8_trtllm (control), fp4_scratch (selected rows, selected rows with dedup, or sequence -> FP8 scratch -> the
same op), fp4_triton (fused, in-kernel dequant; best of a few tile configs). Verify and prefill rows model the
overlap of a request's tokens (90% shared top-k, 10% own picks) that the dedup path exploits.

    python3 glm_kv4_bench.py [--pool 600000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics

import flashinfer  # noqa: F401  (sets the engine's ptxas for Triton)
import flashinfer.decode
import torch

import glm_kv4 as kv4

H, K, CTX = 64, 2048, 65536
SCALE = 576 ** -0.5


def graph_time_us(fn, n_launch: int = 20, replays: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(n_launch):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    times = []
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000 / n_launch)
    return statistics.median(times)


class Bench:
    def __init__(self, n_pool: int, device="cuda"):
        self.dev = torch.device(device)
        self.n = n_pool
        g = torch.Generator(device=device).manual_seed(0)
        nope = (torch.randn(n_pool, 512, device=device, generator=g) * 1.5).bfloat16()
        rope = (torch.randn(n_pool, 64, device=device, generator=g) * 1.5).to(torch.float8_e4m3fn)
        self.pool8 = torch.empty(n_pool, 1, 576, dtype=torch.float8_e4m3fn, device=device)
        self.pool8[:, 0, :512] = nope.to(torch.float8_e4m3fn)
        self.pool8[:, 0, 512:] = rope
        self.pool4 = torch.empty(n_pool, 1, 352, dtype=torch.uint8, device=device)
        kv4.quant_write(self.pool4, torch.arange(n_pool, device=device), nope, rope)
        self.workspace = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
        from sglang.srt.layers.attention.trtllm_mla_backend import make_persistent_multi_ctas_kv_counter_buffer

        self.counter = make_persistent_multi_ctas_kv_counter_buffer(self.dev, H, 8192)
        self.g = g

    def trtllm(self, q, kv, table, seq_lens, out):
        t = q.shape[0]
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q.view(t, 1, H, 576), kv_cache=kv.view(-1, 1, 64, 576), workspace_buffer=self.workspace,
            qk_nope_head_dim=192, kv_lora_rank=512, qk_rope_head_dim=64, block_tables=table.unsqueeze(1),
            seq_lens=seq_lens, max_seq_len=CTX, sparse_mla_top_k=K, bmm1_scale=SCALE, backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=self.counter, out=out)

    def case(self, rows: int):
        q = (torch.randn(rows, H, 576, device=self.dev, generator=self.g) * 1.5).to(torch.float8_e4m3fn)
        table = torch.randint(0, self.n, (rows, K), device=self.dev, generator=self.g, dtype=torch.int32)
        seq_lens = torch.full((rows,), K, dtype=torch.int32, device=self.dev)
        out = torch.empty(rows, 1, H, 512, dtype=torch.bfloat16, device=self.dev)
        return q, table, seq_lens, out

    def run_rows(self, rows: int, tag: str, prefill_reqs: int = 0, tokens_per_req: int = 1) -> dict:
        q, table, seq_lens, out = self.case(rows)
        if tokens_per_req > 1:
            # a request's draft tokens share most of their top-k (adjacent positions): 90% shared, 10% own picks
            for r0 in range(0, rows, tokens_per_req):
                base = table[r0]
                for i in range(r0 + 1, min(rows, r0 + tokens_per_req)):
                    j = torch.randperm(K, device=self.dev, generator=self.g)[: K // 10]
                    table[i] = base
                    table[i, j] = torch.randint(0, self.n, (K // 10,), device=self.dev, generator=self.g,
                                                dtype=torch.int32)
        res = {"case": tag, "rows": rows}
        res["fp8_trtllm"] = graph_time_us(lambda: self.trtllm(q, self.pool8, table, seq_lens, out))
        # selected-row scratch
        scratch = torch.empty(rows * K, 576, dtype=torch.float8_e4m3fn, device=self.dev)
        ident = torch.empty_like(table)

        def sel():
            kv4.dequant_rows(self.pool4, table, scratch)
            kv4.identity_table(table, ident)
            self.trtllm(q, scratch, ident, seq_lens, out)

        res["fp4_scratch_selected"] = graph_time_us(sel)
        best_dq = None
        for br, nw in ((8, 4), (16, 4), (16, 8), (32, 8), (32, 4), (64, 8)):
            us = graph_time_us(lambda: kv4.dequant_rows(self.pool4, table, scratch, block_r=br, num_warps=nw))
            if best_dq is None or us < best_dq[0]:
                best_dq = (us, (br, nw))
        res["fp4_dequant_selected_only"] = best_dq[0]
        res["fp4_dequant_cfg"] = best_dq[1]
        stamp = torch.full((self.n,), kv4.STAMP_CLEAN, dtype=torch.int32, device=self.dev)
        work = (torch.empty(-(-rows * K // 1024), dtype=torch.int32, device=self.dev),
                torch.empty(rows * K, dtype=torch.int32, device=self.dev),
                torch.empty(rows * K, dtype=torch.int32, device=self.dev))

        def dedup():
            kv4.dequant_rows_dedup(self.pool4, table, stamp, scratch, ident, work=work)
            self.trtllm(q, scratch, ident, seq_lens, out)

        res["fp4_scratch_dedup"] = graph_time_us(dedup)
        res["fp4_dedup_only"] = graph_time_us(lambda: kv4.dequant_rows_dedup(self.pool4, table, stamp, scratch, ident,
                                                                              work=work))
        if prefill_reqs:
            # sequence path: prefill_reqs requests of CTX tokens each; the table's slots must lie inside them
            pt = torch.stack([torch.randperm(self.n, device=self.dev, generator=self.g)[:CTX]
                              for _ in range(prefill_reqs)]).int()
            lens = torch.full((prefill_reqs,), CTX, dtype=torch.int32, device=self.dev)
            bases = (torch.arange(prefill_reqs, device=self.dev) * CTX).int()
            per = rows // prefill_reqs
            tab = torch.stack([pt[i // per if i // per < prefill_reqs else prefill_reqs - 1][
                torch.randint(0, CTX, (K,), device=self.dev, generator=self.g)] for i in range(rows)]).int()
            sscratch = torch.empty(prefill_reqs * CTX, 576, dtype=torch.float8_e4m3fn, device=self.dev)  # CTX % 64 == 0
            s2p = torch.full((self.n,), -1, dtype=torch.int32, device=self.dev)
            remap = torch.empty_like(tab)

            def seq():
                kv4.dequant_seq(self.pool4, pt, lens, bases, CTX, sscratch, s2p)
                kv4.remap_table(tab, s2p, self.n, remap)
                self.trtllm(q, sscratch, remap, seq_lens, out)

            res["fp4_scratch_sequence"] = graph_time_us(seq, n_launch=5)
            res["fp4_dequant_sequence_only"] = graph_time_us(
                lambda: kv4.dequant_seq(self.pool4, pt, lens, bases, CTX, sscratch, s2p), n_launch=5)
            res["sequence_rows"] = prefill_reqs * CTX
        best = None
        for cfg in (dict(block_h=32, block_n=64, num_warps=8, splits=8), dict(block_h=32, block_n=64, num_warps=8, splits=4),
                    dict(block_h=32, block_n=64, num_warps=4, splits=8), dict(block_h=16, block_n=64, num_warps=4, splits=8),
                    dict(block_h=32, block_n=32, num_warps=4, splits=16), dict(block_h=32, block_n=64, num_warps=8, splits=16),
                    dict(block_h=32, block_n=64, num_warps=8, splits=1)):
            try:
                us = graph_time_us(lambda: kv4.sparse_mla(q, self.pool4, table, SCALE, out=out.view(rows, H, 512),
                                                          **cfg), n_launch=5 if rows > 256 else 20)
            except Exception as ex:  # a config that does not compile at this shape
                res.setdefault("triton_errors", []).append(f"{cfg}: {type(ex).__name__}")
                continue
            if best is None or us < best[0]:
                best = (us, cfg)
        if best:
            res["fp4_triton"] = best[0]
            res["fp4_triton_cfg"] = best[1]
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=600_000)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    b = Bench(a.pool)
    rows = []
    for bs in (1, 8, 16, 32, 48):
        rows.append(b.run_rows(bs, f"decode bs{bs}"))
    for bs in (16, 48):
        rows.append(b.run_rows(bs * 4, f"verify bs{bs}x4", tokens_per_req=4))
    for t in (512, 2048):
        rows.append(b.run_rows(t, f"prefill {t} tok", prefill_reqs=2, tokens_per_req=16))
    print(f"{'case':<18}{'rows':>6}{'fp8_trtllm':>12}{'fp4_sel':>10}{'(dq)':>8}{'fp4_dedup':>10}{'(dq)':>8}"
          f"{'fp4_seq':>10}{'(dq)':>8}{'fp4_triton':>12}  cfg")
    for r in rows:
        print(f"{r['case']:<18}{r['rows']:>6}{r['fp8_trtllm']:>12.1f}{r['fp4_scratch_selected']:>10.1f}"
              f"{r['fp4_dequant_selected_only']:>8.1f}{r['fp4_scratch_dedup']:>10.1f}{r['fp4_dedup_only']:>8.1f}"
              f"{r.get('fp4_scratch_sequence', float('nan')):>10.1f}"
              f"{r.get('fp4_dequant_sequence_only', float('nan')):>8.1f}{r.get('fp4_triton', float('nan')):>12.1f}"
              f"  {r.get('fp4_triton_cfg', r.get('triton_errors'))} dq{r['fp4_dequant_cfg']}")
    print(f"(us per layer call, CUDA-graph replay medians; GPU {torch.cuda.get_device_name()}, pool {a.pool} slots)")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
