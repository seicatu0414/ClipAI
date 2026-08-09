from datetime import UTC, datetime, timedelta
from uuid import uuid4

from clipai.knowledge.domain import HistoricalStream
from clipai.knowledge.selection import select_historical_streams


def _stream(
    age_days: int,
    duration_hours: float,
    *,
    views: int = 0,
    manual: bool = False,
) -> HistoricalStream:
    return HistoricalStream(
        id=uuid4(),
        streamer_id=uuid4(),
        transcript_id=uuid4(),
        title=f"stream-{age_days}",
        published_at=datetime(2026, 1, 10, tzinfo=UTC) - timedelta(days=age_days),
        duration_seconds=duration_hours * 3600,
        view_count=views,
        comment_count=0,
        manually_selected=manual,
    )


def test_selects_recent_hours_then_representative_streams() -> None:
    recent = [_stream(0, 2), _stream(1, 2), _stream(2, 2)]
    popular = _stream(20, 1, views=10_000)
    manual = _stream(30, 1, manual=True)

    selected = select_historical_streams(
        [*recent, popular, manual],
        max_recent_hours=4,
        max_representative_streams=1,
    )

    selected_ids = {stream.id for stream in selected}
    assert recent[0].id in selected_ids
    assert recent[1].id in selected_ids
    assert manual.id in selected_ids
    assert popular.id not in selected_ids
