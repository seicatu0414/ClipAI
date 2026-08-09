from clipai.config import Settings
from clipai.events.domain import JsonValue


def knowledge_configuration(settings: Settings) -> dict[str, JsonValue]:
    return {
        "max_historical_hours": settings.knowledge_max_historical_hours,
        "max_representative_streams": settings.knowledge_max_representative_streams,
        "chunk_characters": settings.knowledge_chunk_characters,
    }
