from uuid import UUID

from clipai.candidates.domain import CandidateCategory
from clipai.feedback.domain import (
    EvaluationCandidate,
    FeedbackRating,
    FeedbackReasonTag,
)
from clipai.feedback.learning import (
    default_weights,
    evaluate_preferences,
    next_version_number,
    personalized_score,
    rollback_weights,
    update_weights,
)


def scores(primary: CandidateCategory) -> dict[CandidateCategory, float]:
    result = {category: 0.1 for category in CandidateCategory}
    result[primary] = 0.9
    return result


def test_excellent_feedback_increases_explainable_tagged_preference() -> None:
    original = default_weights()
    updated, explanation = update_weights(
        original,
        scores(CandidateCategory.HUMOR),
        FeedbackRating.EXCELLENT,
        (FeedbackReasonTag.HUMOR,),
    )

    assert original[CandidateCategory.HUMOR] == 1
    assert updated[CandidateCategory.HUMOR] > 1
    assert any("humor: 1.0000 ->" in item for item in explanation)


def test_reject_feedback_reduces_future_category_score() -> None:
    category_scores = scores(CandidateCategory.STRONG_REACTION)
    original = default_weights()
    updated, _ = update_weights(
        original,
        category_scores,
        FeedbackRating.REJECT,
        (FeedbackReasonTag.REACTION,),
    )

    assert personalized_score(category_scores, updated) < personalized_score(
        category_scores, original
    )


def test_preference_versions_are_monotonic_and_rollback_copies_history() -> None:
    target = default_weights()
    restored = rollback_weights(target)
    restored[CandidateCategory.HUMOR] = 1.2

    assert next_version_number(4) == 5
    assert target[CandidateCategory.HUMOR] == 1


def test_before_after_evaluation_does_not_mutate_candidates() -> None:
    candidates = [
        EvaluationCandidate(
            UUID(int=1),
            1,
            scores(CandidateCategory.HUMOR),
            FeedbackRating.EXCELLENT,
        ),
        EvaluationCandidate(
            UUID(int=2),
            2,
            scores(CandidateCategory.GREAT_PLAY),
            FeedbackRating.REJECT,
        ),
        EvaluationCandidate(
            UUID(int=3),
            3,
            scores(CandidateCategory.EMOTIONAL_MOMENT),
            None,
        ),
    ]
    before_weights = default_weights()
    after_weights = before_weights | {CandidateCategory.HUMOR: 1.3}

    before = evaluate_preferences(candidates, before_weights, UUID(int=10))
    after = evaluate_preferences(candidates, after_weights, UUID(int=11))

    assert candidates[0].original_rank == 1
    assert before.preference_version_id != after.preference_version_id
    assert after.ranked_candidate_ids[0] == UUID(int=1)
    assert UUID(int=3) not in after.ranked_candidate_ids
