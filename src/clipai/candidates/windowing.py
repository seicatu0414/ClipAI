from clipai.candidates.domain import CandidateEvent, CandidateWindow


def overlap_ratio(first: CandidateWindow, second: CandidateWindow) -> float:
    overlap = max(
        0.0,
        min(first.end_seconds, second.end_seconds)
        - max(first.start_seconds, second.start_seconds),
    )
    shorter = min(
        first.end_seconds - first.start_seconds,
        second.end_seconds - second.start_seconds,
    )
    return 0.0 if shorter <= 0 else overlap / shorter


def construct_windows(
    events: list[CandidateEvent],
    *,
    minimum_seconds: float,
    maximum_seconds: float,
    padding_before_seconds: float,
    padding_after_seconds: float,
    merge_gap_seconds: float,
) -> list[CandidateWindow]:
    windows: list[CandidateWindow] = []
    for event in sorted(events, key=lambda item: (item.start_seconds, item.end_seconds)):
        start = max(0.0, event.start_seconds - padding_before_seconds)
        end = max(event.end_seconds + padding_after_seconds, start + minimum_seconds)
        end = min(end, start + maximum_seconds)
        if windows and start - windows[-1].end_seconds <= merge_gap_seconds:
            previous = windows[-1]
            merged_end = max(previous.end_seconds, end)
            if merged_end - previous.start_seconds <= maximum_seconds:
                combined = previous.events + (event,)
                windows[-1] = CandidateWindow(
                    previous.start_seconds,
                    merged_end,
                    combined,
                    max(item.confidence for item in combined)
                    + min(0.12, 0.03 * (len(combined) - 1)),
                )
                continue
        windows.append(CandidateWindow(start, end, (event,), event.confidence))
    return windows


def reduce_windows(
    windows: list[CandidateWindow],
    *,
    target_count: int,
    overlap_threshold: float,
) -> list[CandidateWindow]:
    selected: list[CandidateWindow] = []
    ordered = sorted(
        windows,
        key=lambda item: (-item.preliminary_score, item.start_seconds, item.end_seconds),
    )
    for window in ordered:
        if all(overlap_ratio(window, kept) < overlap_threshold for kept in selected):
            selected.append(window)
        if len(selected) >= target_count:
            break
    return sorted(selected, key=lambda item: item.start_seconds)
