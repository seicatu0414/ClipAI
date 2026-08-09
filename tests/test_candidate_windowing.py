from uuid import UUID

from clipai.candidates.domain import CandidateEvent
from clipai.candidates.windowing import construct_windows, overlap_ratio, reduce_windows
from clipai.events.domain import EventType


def event(index: int, start: float, end: float, confidence: float) -> CandidateEvent:
    return CandidateEvent(
        UUID(int=index),
        EventType.LOUD_REACTION,
        start,
        end,
        confidence,
        {},
        "reaction",
    )


def test_construct_windows_enforces_duration_and_merges_setup_payoff() -> None:
    windows = construct_windows(
        [event(1, 20, 22, 0.7), event(2, 27, 29, 0.8)],
        minimum_seconds=15,
        maximum_seconds=120,
        padding_before_seconds=8,
        padding_after_seconds=12,
        merge_gap_seconds=8,
    )

    assert len(windows) == 1
    assert windows[0].start_seconds == 12
    assert windows[0].end_seconds == 41
    assert len(windows[0].events) == 2


def test_reduce_windows_is_stable_and_suppresses_overlap() -> None:
    windows = construct_windows(
        [
            event(1, 10, 12, 0.7),
            event(2, 11, 13, 0.9),
            event(3, 100, 102, 0.8),
        ],
        minimum_seconds=15,
        maximum_seconds=120,
        padding_before_seconds=2,
        padding_after_seconds=13,
        merge_gap_seconds=0,
    )
    reduced = reduce_windows(windows, target_count=2, overlap_threshold=0.5)

    assert [window.start_seconds for window in reduced] == [8, 98]
    assert overlap_ratio(reduced[0], reduced[1]) == 0


def test_constructed_window_never_exceeds_maximum() -> None:
    window = construct_windows(
        [event(1, 5, 300, 0.8)],
        minimum_seconds=15,
        maximum_seconds=120,
        padding_before_seconds=8,
        padding_after_seconds=12,
        merge_gap_seconds=8,
    )[0]

    assert window.start_seconds == 0
    assert window.end_seconds == 120
