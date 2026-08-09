from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from clipai.events.domain import EventType, JsonValue
from clipai.knowledge.domain import KnowledgeObservation


class CandidateCategory(StrEnum):
    HUMOR = "humor"
    GREAT_PLAY = "great_play"
    EMOTIONAL_MOMENT = "emotional_moment"
    MEMORABLE_QUOTE = "memorable_quote"
    STRONG_REACTION = "strong_reaction"
    STORY_PAYOFF = "story_payoff"
    VIEWER_INTERACTION = "viewer_interaction"
    CALLBACK_RUNNING_JOKE = "callback_running_joke"


class CandidateJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class CandidateEvent:
    id: UUID
    event_type: EventType
    start_seconds: float
    end_seconds: float
    confidence: float
    source_signals: dict[str, JsonValue]
    explanation: str


@dataclass(frozen=True)
class CandidateWindow:
    start_seconds: float
    end_seconds: float
    events: tuple[CandidateEvent, ...]
    preliminary_score: float


@dataclass(frozen=True)
class ClipCandidate:
    id: UUID | None
    rank: int
    start_seconds: float
    end_seconds: float
    category_scores: dict[CandidateCategory, float]
    overall_score: float
    confidence: float
    reasons: tuple[str, ...]
    event_ids: tuple[UUID, ...]
    knowledge_observation_ids: tuple[UUID, ...]
    knowledge: tuple[KnowledgeObservation, ...]


@dataclass(frozen=True)
class CandidateJob:
    id: UUID
    streamer_id: UUID
    transcript_id: UUID
    event_detection_job_id: UUID
    knowledge_version_id: UUID
    preference_version_id: UUID | None
    status: CandidateJobStatus
    progress: int
    pipeline_version: str
    provider: str
    model: str
    prompt_version: str
    configuration: dict[str, JsonValue]
    error: str | None = None
