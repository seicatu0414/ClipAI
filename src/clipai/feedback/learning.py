from uuid import UUID

from clipai.candidates.domain import CandidateCategory
from clipai.feedback.domain import (
    EvaluationCandidate,
    FeedbackRating,
    FeedbackReasonTag,
    PreferenceEvaluation,
)

DEFAULT_WEIGHT = 1.0
MINIMUM_WEIGHT = 0.5
MAXIMUM_WEIGHT = 1.5
LEARNING_RATE = 0.12
STRATEGY_VERSION = "bounded-category-weights-v1"

_RATING_TARGET = {
    FeedbackRating.EXCELLENT: 1.0,
    FeedbackRating.USABLE: 0.65,
    FeedbackRating.REJECT: 0.0,
}
_TAG_CATEGORY = {
    FeedbackReasonTag.HUMOR: CandidateCategory.HUMOR,
    FeedbackReasonTag.GREAT_PLAY: CandidateCategory.GREAT_PLAY,
    FeedbackReasonTag.EMOTIONAL: CandidateCategory.EMOTIONAL_MOMENT,
    FeedbackReasonTag.QUOTE: CandidateCategory.MEMORABLE_QUOTE,
    FeedbackReasonTag.REACTION: CandidateCategory.STRONG_REACTION,
    FeedbackReasonTag.STORY: CandidateCategory.STORY_PAYOFF,
    FeedbackReasonTag.VIEWER_INTERACTION: CandidateCategory.VIEWER_INTERACTION,
    FeedbackReasonTag.CALLBACK: CandidateCategory.CALLBACK_RUNNING_JOKE,
}


def default_weights() -> dict[CandidateCategory, float]:
    return {category: DEFAULT_WEIGHT for category in CandidateCategory}


def next_version_number(current: int) -> int:
    return current + 1


def rollback_weights(
    target: dict[CandidateCategory, float],
) -> dict[CandidateCategory, float]:
    return target.copy()


def update_weights(
    current: dict[CandidateCategory, float],
    scores: dict[CandidateCategory, float],
    rating: FeedbackRating,
    reason_tags: tuple[FeedbackReasonTag, ...],
) -> tuple[dict[CandidateCategory, float], tuple[str, ...]]:
    signal = _RATING_TARGET[rating] - 0.5
    tagged = {_TAG_CATEGORY[tag] for tag in reason_tags if tag in _TAG_CATEGORY}
    updated: dict[CandidateCategory, float] = {}
    explanations: list[str] = [f"strategy={STRATEGY_VERSION}"]
    for category in CandidateCategory:
        emphasis = 1.5 if category in tagged else 1.0
        delta = LEARNING_RATE * signal * scores.get(category, 0.0) * emphasis
        before = current.get(category, DEFAULT_WEIGHT)
        after = round(max(MINIMUM_WEIGHT, min(MAXIMUM_WEIGHT, before + delta)), 4)
        updated[category] = after
        if after != before:
            explanations.append(
                f"{category.value}: {before:.4f} -> {after:.4f} "
                f"({rating.value}, score={scores.get(category, 0.0):.4f})"
            )
    if len(explanations) == 1:
        explanations.append("no category weight changed")
    return updated, tuple(explanations)


def personalized_score(
    scores: dict[CandidateCategory, float],
    weights: dict[CandidateCategory, float],
) -> float:
    weighted = sorted(
        (score * weights.get(category, DEFAULT_WEIGHT) for category, score in scores.items()),
        reverse=True,
    )
    normalization = max(weights.values(), default=DEFAULT_WEIGHT)
    return round(min(1.0, (weighted[0] * 0.7 + weighted[1] * 0.3) / normalization), 4)


def evaluate_preferences(
    candidates: list[EvaluationCandidate],
    weights: dict[CandidateCategory, float],
    preference_version_id: UUID,
) -> PreferenceEvaluation:
    reviewed = [item for item in candidates if item.rating is not None]
    ordered = sorted(
        reviewed,
        key=lambda item: (
            -personalized_score(item.category_scores, weights),
            item.original_rank,
            str(item.candidate_id),
        ),
    )
    accepted_ranks = [
        index
        for index, item in enumerate(ordered, start=1)
        if item.rating in {FeedbackRating.EXCELLENT, FeedbackRating.USABLE}
    ]
    return PreferenceEvaluation(
        preference_version_id,
        tuple(item.candidate_id for item in ordered),
        None if not accepted_ranks else sum(accepted_ranks) / len(accepted_ranks),
        _precision(ordered, 20),
        _precision(ordered, 30),
    )


def _precision(candidates: list[EvaluationCandidate], limit: int) -> float | None:
    reviewed = candidates[:limit]
    if not reviewed:
        return None
    accepted = sum(
        item.rating in {FeedbackRating.EXCELLENT, FeedbackRating.USABLE}
        for item in reviewed
    )
    return round(accepted / len(reviewed), 4)
