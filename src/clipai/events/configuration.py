from clipai.config import Settings
from clipai.events.domain import JsonValue


def event_configuration(settings: Settings) -> dict[str, JsonValue]:
    return {
        "minimum_confidence": settings.event_min_confidence,
        "loudness_delta_db": settings.event_loudness_delta_db,
        "silence_db": settings.event_silence_db,
        "silence_minimum_seconds": settings.event_silence_min_seconds,
        "merge_gap_seconds": settings.event_merge_gap_seconds,
    }
