import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from sglang.srt.managers.decode_burst import hold_prefill, next_streak  # noqa: E402


def test_disabled_at_one():
    assert hold_prefill(1, 0, False, True, False) is False


def test_holds_for_burst_then_releases():
    burst, streak = 4, 0
    held = []
    for _ in range(6):
        h = hold_prefill(burst, streak, False, True, False)
        held.append(h)
        streak = next_streak(burst, streak, prefill_admitted=not h)
    assert held == [True, True, True, True, False, True]  # four decode steps, a prefill pass (streak resets), decode again


def test_idle_scheduler_never_holds():
    assert hold_prefill(8, 0, False, False, False) is False
    assert hold_prefill(8, 0, False, True, True) is False   # prefill-only running batch has nothing to decode


def test_after_prefill_pass_no_hold():
    assert hold_prefill(8, 0, True, True, False) is False
    assert hold_prefill(8, 0, None, True, False) is False


def test_streak_resets_on_prefill():
    assert next_streak(4, 3, prefill_admitted=True) == 0
    assert next_streak(4, 3, prefill_admitted=False) == 4
    assert next_streak(1, 5, prefill_admitted=False) == 0


def test_chunked_prefill_in_flight_is_never_held():
    assert hold_prefill(8, 0, False, True, False, chunked_prefill_pending=True) is False
    assert hold_prefill(8, 0, False, True, False, chunked_prefill_pending=False) is True
