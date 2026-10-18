"""The decode-burst rule inside the prefill delayer's negotiation must be uniform across ranks and bounded."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
import torch  # noqa: E402

from sglang.srt.managers.prefill_delayer import PrefillDelayer, _State  # noqa: E402


def _delayer(fake_gathered):
    d = PrefillDelayer.__new__(PrefillDelayer)
    d._token_usage_low_watermark = None
    d._max_delay_passes = 4
    d._max_delay_ms = 10_000
    d._queue_trigger_enabled = False
    d._queue_min_ratio = 0.0
    d.enable_dp_attention = True
    d.dp_size = 2
    d.skip_first_delayer = False
    d._metrics_collector = None
    d._burst_queue_cap = 2
    d._gather_info = lambda **kw: torch.tensor(fake_gathered, dtype=torch.int64)
    return d


def test_burst_holds_prefill_while_a_rank_is_mid_streak():
    # rank0 has decodes running and 2 decode steps since its last prefill, rank1 idle
    d = _delayer([[1, 0, 8, 4, 2, 2], [1, 0, 0, 4, 0, 9]])
    out = d._negotiate_should_allow_prefill_pure(None, True, 0.5, running_batch=8, max_prefill_bs=4,
                                                 max_running_requests=64, waiting_queue_len=2, decode_streak=2, decode_burst=8)
    assert out.output_allow is False and out.output_reason == "delay"


def test_burst_releases_after_the_streak_and_is_bounded():
    d = _delayer([[1, 0, 8, 4, 2, 9], [1, 0, 8, 4, 2, 9]])
    out = d._negotiate_should_allow_prefill_pure(None, True, 0.5, running_batch=8, max_prefill_bs=4,
                                                 max_running_requests=64, waiting_queue_len=2, decode_streak=9, decode_burst=8)
    assert out.output_allow is True
    d2 = _delayer([[1, 0, 8, 4, 2, 1], [1, 0, 8, 4, 2, 1]])
    st = _State(); st = st.bump_delayed_count(); st = st.bump_delayed_count(); st = st.bump_delayed_count()
    out2 = d2._negotiate_should_allow_prefill_pure(st, True, 0.5, running_batch=8, max_prefill_bs=4,
                                                   max_running_requests=64, waiting_queue_len=2, decode_streak=1, decode_burst=8)
    assert out2.output_allow is True and out2.output_reason == "wait_timeout"


def test_burst_off_by_default():
    d = _delayer([[1, 0, 8, 4, 2, 0], [1, 0, 8, 4, 2, 0]])
    out = d._negotiate_should_allow_prefill_pure(None, True, 0.5, running_batch=8, max_prefill_bs=4,
                                                 max_running_requests=64, waiting_queue_len=2, decode_streak=0, decode_burst=1)
    assert out.output_allow is True


def test_burst_does_not_hold_when_turns_are_queued():
    # busiest rank has 5 waiting turns: prefill must not be held even mid-streak
    d = _delayer([[1, 0, 8, 4, 5, 2], [1, 0, 8, 4, 1, 2]])
    out = d._negotiate_should_allow_prefill_pure(None, True, 0.5, running_batch=8, max_prefill_bs=4,
                                                 max_running_requests=64, waiting_queue_len=5, decode_streak=2, decode_burst=8)
    assert out.output_allow is True
