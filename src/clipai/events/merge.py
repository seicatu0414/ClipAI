from collections.abc import Iterable

from clipai.events.domain import DetectedEvent


def merge_events(
    events: Iterable[DetectedEvent],
    *,
    minimum_confidence: float,
    merge_gap_seconds: float,
) -> list[DetectedEvent]:
    accepted = sorted(
        (event for event in events if event.confidence >= minimum_confidence),
        key=lambda event: (event.event_type.value, event.start_seconds, event.end_seconds),
    )
    merged: list[DetectedEvent] = []
    for event in accepted:
        if not merged:
            merged.append(event)
            continue
        previous = merged[-1]
        same_type = previous.event_type is event.event_type
        nearby = event.start_seconds <= previous.end_seconds + merge_gap_seconds
        if not (same_type and nearby):
            merged.append(event)
            continue
        explanations = list(
            dict.fromkeys([previous.explanation, event.explanation])
        )
        merged.append(
            DetectedEvent(
                event_type=previous.event_type,
                start_seconds=min(previous.start_seconds, event.start_seconds),
                end_seconds=max(previous.end_seconds, event.end_seconds),
                confidence=max(previous.confidence, event.confidence),
                source_signals={
                    "merged_count": int(previous.source_signals.get("merged_count", 1)) + 1,
                    "signals": [previous.source_signals, event.source_signals],
                },
                explanation=" ".join(explanations),
            )
        )
        merged.pop(-2)
    return sorted(merged, key=lambda event: (event.start_seconds, event.end_seconds))
