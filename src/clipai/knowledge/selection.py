from collections.abc import Iterable

from clipai.knowledge.domain import HistoricalStream


def select_historical_streams(
    streams: Iterable[HistoricalStream],
    *,
    max_recent_hours: float,
    max_representative_streams: int,
) -> list[HistoricalStream]:
    available = list(streams)
    recent: list[HistoricalStream] = []
    duration = 0.0
    for stream in sorted(available, key=lambda item: item.published_at, reverse=True):
        if duration >= max_recent_hours * 3600:
            break
        recent.append(stream)
        duration += stream.duration_seconds

    recent_ids = {stream.id for stream in recent}
    representatives = sorted(
        (stream for stream in available if stream.id not in recent_ids),
        key=lambda item: (
            item.manually_selected,
            item.view_count,
            item.comment_count,
            item.published_at,
        ),
        reverse=True,
    )[:max_representative_streams]
    return sorted([*recent, *representatives], key=lambda item: item.published_at)
