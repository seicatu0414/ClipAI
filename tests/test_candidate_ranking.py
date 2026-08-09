import json

import pytest

from clipai.candidates.domain import CandidateCategory
from clipai.candidates.ranking import BalancedCandidateScorer, parse_ranking_response


def response(**overrides: object) -> str:
    data: dict[str, object] = {
        "category_scores": {category.value: 0.1 for category in CandidateCategory},
        "confidence": 0.8,
        "reasons": ["イベントと発言が一致している"],
    }
    data.update(overrides)
    return json.dumps(data)


def test_score_composition_uses_two_strongest_dimensions() -> None:
    scores = {category: 0.1 for category in CandidateCategory}
    scores[CandidateCategory.HUMOR] = 0.9
    scores[CandidateCategory.STRONG_REACTION] = 0.5

    assert BalancedCandidateScorer().score(scores) == 0.78


def test_response_requires_explanation() -> None:
    with pytest.raises(ValueError, match="reasons"):
        parse_ranking_response(response(reasons=[]))


def test_extensions_are_bounded() -> None:
    result = parse_ranking_response(
        response(extend_before_seconds=100, extend_after_seconds=-5)
    )

    assert result.extend_before_seconds == 30
    assert result.extend_after_seconds == 0


def test_response_bounds_numeric_scores_to_probabilities() -> None:
    scores = {category.value: 0.1 for category in CandidateCategory}
    scores[CandidateCategory.HUMOR.value] = 2
    scores[CandidateCategory.GREAT_PLAY.value] = -1

    result = parse_ranking_response(
        response(category_scores=scores, confidence=3)
    )

    assert result.category_scores[CandidateCategory.HUMOR] == 1
    assert result.category_scores[CandidateCategory.GREAT_PLAY] == 0
    assert result.confidence == 1

