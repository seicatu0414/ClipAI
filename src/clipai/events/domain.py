from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias
from uuid import UUID

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class EventType(StrEnum):
    LAUGHTER = "laughter"
    LOUD_REACTION = "loud_reaction"
    SINGING = "singing"
    EMOTIONAL_VOICE = "emotional_voice"
    UNUSUAL_SILENCE = "unusual_silence"
    VICTORY_DEFEAT = "victory_defeat"
    MEMORABLE_STATEMENT = "memorable_statement"
    VIEWER_RESPONSE = "viewer_response"
    CALLBACK_CONTRADICTION = "callback_contradiction"


class EventJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class AudioFeature:
    start_seconds: float
    end_seconds: float
    rms_dbfs: float


@dataclass(frozen=True)
class DetectedEvent:
    event_type: EventType
    start_seconds: float
    end_seconds: float
    confidence: float
    source_signals: dict[str, JsonValue]
    explanation: str


@dataclass(frozen=True)
class EventDetectionJob:
    id: UUID
    transcript_id: UUID
    status: EventJobStatus
    progress: int
    detector_version: str
    configuration: dict[str, JsonValue]
    error: str | None = None
