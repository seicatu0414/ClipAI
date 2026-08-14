from dataclasses import dataclass, field
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


class ScenePhase(StrEnum):
    SETUP = "setup"
    DEVELOPMENT = "development"
    CLIMAX = "climax"
    AFTERMATH = "aftermath"
    TRANSITION = "transition"


class ThreadStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


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
class TopicWindow:
    start_seconds: float
    end_seconds: float
    confidence: float
    source_signals: tuple[str, ...]


@dataclass(frozen=True)
class OpenThread:
    thread: str
    status: ThreadStatus
    confidence: float
    opened_at: float
    resolved_at: float | None = None


@dataclass(frozen=True)
class SemanticChunk:
    start_seconds: float
    end_seconds: float
    text: str
    phase: ScenePhase
    source_signals: tuple[str, ...]


@dataclass(frozen=True)
class SceneWindow:
    start_seconds: float
    end_seconds: float
    phase: ScenePhase
    phases: tuple[ScenePhase, ...]
    primary_goal: str | None
    open_threads: tuple[OpenThread, ...]
    resolved_threads: tuple[OpenThread, ...]
    emotional_state: str
    reaction_state: str
    transition_signal: str | None
    confidence: float
    completion_confidence: float
    source_signals: tuple[str, ...]


@dataclass(frozen=True)
class EndBoundaryCandidate:
    id: str
    timestamp: float
    confidence: float
    reason: str
    source_signals: tuple[str, ...]


@dataclass(frozen=True)
class EndBoundarySelection:
    timestamp: float
    confidence: float
    reason: str
    source_signals: tuple[str, ...]
    topic_window: TopicWindow
    scene_window: SceneWindow
    candidates: tuple[EndBoundaryCandidate, ...]
    context_start_seconds: float
    context_end_seconds: float
    llm_used: bool
    detailed_analysis_used: bool


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
    boundary_analysis: dict[str, JsonValue] = field(default_factory=dict)


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
