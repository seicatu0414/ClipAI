from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from clipai.events.domain import JsonValue


class KnowledgeCategory(StrEnum):
    RECURRING_PHRASE = "recurring_phrase"
    RECURRING_JOKE = "recurring_joke"
    SPEECH_PATTERN = "speech_pattern"
    EMOTIONAL_BASELINE = "emotional_baseline"
    CONTENT_STRENGTH = "content_strength"
    COLLABORATION_PATTERN = "collaboration_pattern"
    CALLBACK = "callback"


class ObservationOrigin(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"


class KnowledgeJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class Streamer:
    id: UUID
    channel_url: str
    display_name: str
    youtube_channel_id: str | None = None


@dataclass(frozen=True)
class HistoricalStream:
    id: UUID
    streamer_id: UUID
    transcript_id: UUID
    title: str
    published_at: datetime
    duration_seconds: float
    view_count: int
    comment_count: int
    manually_selected: bool


@dataclass(frozen=True)
class Evidence:
    transcript_id: UUID
    segment_index: int
    start_seconds: float
    end_seconds: float
    quote: str


@dataclass(frozen=True)
class KnowledgeObservation:
    category: KnowledgeCategory
    statement: str
    origin: ObservationOrigin
    confidence: float
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class KnowledgeJob:
    id: UUID
    streamer_id: UUID
    status: KnowledgeJobStatus
    progress: int
    provider: str
    model: str
    prompt_version: str
    configuration: dict[str, JsonValue]
    error: str | None = None


@dataclass(frozen=True)
class KnowledgeVersion:
    id: UUID
    streamer_id: UUID
    version_number: int
    previous_version_id: UUID | None
    provider: str
    model: str
    prompt_version: str
    configuration: dict[str, JsonValue]
    observations: tuple[KnowledgeObservation, ...]
