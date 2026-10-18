# Scheduler

## [decode-burst/](decode-burst/)

SGLang defines `--num-continuous-decode-steps` at `server_args.py:971`, but nothing reads it. On agent traffic with short turns, every new turn's prefill stalls decode for everyone on the node. Kimi K2.7 ran at 15 tok/s per user on 200-token turns against 67 on long outputs.

These 5 patches, against SGLang v0.5.18, also shipped on v0.5.20, make the flag real:

1. After a prefill batch, decode runs up to N steps before the next prefill is admitted while requests are running. The rule is a pure function in [`decode_burst.py`](decode-burst/decode_burst.py).
2. A request is never held mid chunked prefill.
3. Under DP attention, the burst streak joins the prefill delayer's negotiation as a sixth gathered field. Every rank holds prefill together while any rank is mid-streak, bounded by `max_delay_passes`.
4. A queue cap (`SGLANG_DECODE_BURST_QUEUE_CAP`) holds prefill only while the busiest rank's waiting queue is short.

At the default N=1 nothing changes. Tests: [`test_decode_burst.py`](decode-burst/test_decode_burst.py), [`test_delayer_burst.py`](decode-burst/test_delayer_burst.py).

Measured: Kimi K2.7 short turns 15 to 67 tok/s per user. GLM-5.3 at 96 sessions x 96k: stock 3,022 tok/s, cap 4 gives 4,050, cap 8 gives 4,670 (+55%). On mixed agent traffic, cap 2 is neutral and cap 8 is -10%, so the cap is tuned per traffic shape.

## [dp-balance-and-pacing/](dp-balance-and-pacing/)

Under DP attention, every rank waits for the slowest rank's prefill chunk. This folder tries to fix that:

- Cache-aware, prefill-work-balanced DP dispatch, with a replay simulator.
- A per-rank prefill budget.
- Skew alignment and skew budgets.
- Prefill pacing.

Measured on 8xB200 and MI355X. The extend cap came out at -12% / -7% and was dropped. Prefill balancing was neutral on RTX (115 vs 118 tok/s per user). Kept for the record and for the simulator.
