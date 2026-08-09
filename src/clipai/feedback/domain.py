from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from clipai.candidates.domain import CandidateCategory


class FeedbackRating(StrEnum):
    EXCELLENT = "excellent"
    USABLE = "usable"
    REJECT = "reject"


class FeedbackReasonTag(StrEnum):
    HUMOR = "humor"
    GREAT_PLAY = "great_play"
    EMOTIONAL = "emotional"
    QUOTE = "quote"
    REACTION = "reaction"
    STORY = "story"
    VIEWER_INTERACTION = "viewer_interaction"
    CALLBACK = "callback"
    OTHER = "other"


@dataclass(frozen=True)
class CandidateFeedback:
    id: UUID
    candidate_id: UUID
    streamer_id: UUID
    rating: FeedbackRating
    reason_tags: tuple[FeedbackReasonTag, ...]
    note: str | None
    preference_version_id: UUID
    created_at: datetime


@dataclass(frozen=True)
class PreferenceVersion:
    id: UUID
    streamer_id: UUID
    version_number: int
    previous_version_id: UUID | None
    source_feedback_id: UUID | None
    rollback_of_version_id: UUID | None
    category_weights: dict[CandidateCategory, float]
    explanation: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True)
class EvaluationCandidate:
    candidate_id: UUID
    original_rank: int
    category_scores: dict[CandidateCategory, float]
    rating: FeedbackRating | None


@dataclass(frozen=True)
class PreferenceEvaluation:
    preference_version_id: UUID
    ranked_candidate_ids: tuple[UUID, ...]
    average_accepted_rank: float | None
    precision_at_20: float | None
    precision_at_30: float | None
