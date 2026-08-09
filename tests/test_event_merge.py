from clipai.events.domain import DetectedEvent, EventType
from clipai.events.merge import merge_events


def _event(start: float, end: float, confidence: float = 0.7) -> DetectedEvent:
    return DetectedEvent(
        EventType.LAUGHTER,
        start,
        end,
        confidence,
        {"keyword": "ｗｗ"},
        "Laughter marker found.",
    )


def test_merges_overlapping_and_nearby_duplicates() -> None:
    events = merge_events(
        [_event(10, 12), _event(11, 13), _event(14, 15)],
        minimum_confidence=0.5,
        merge_gap_seconds=1,
    )

    assert len(events) == 1
    assert events[0].start_seconds == 10
    assert events[0].end_seconds == 15
    assert events[0].source_signals["merged_count"] == 3


def test_suppresses_events_below_threshold() -> None:
    events = merge_events(
        [_event(0, 1, 0.49), _event(2, 3, 0.5)],
        minimum_confidence=0.5,
        merge_gap_seconds=0,
    )

    assert [(event.start_seconds, event.confidence) for event in events] == [(2, 0.5)]


def test_does_not_merge_different_event_types() -> None:
    reaction = DetectedEvent(
        EventType.LOUD_REACTION, 0, 2, 0.8, {"delta_db": 15}, "Loudness rose."
    )

    events = merge_events(
        [_event(0, 2), reaction], minimum_confidence=0.5, merge_gap_seconds=2
    )

    assert len(events) == 2
