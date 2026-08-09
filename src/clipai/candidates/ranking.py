import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from clipai.candidates.domain import (
    CandidateCategory,
    CandidateWindow,
    ClipCandidate,
)
from clipai.domain import TranscriptSegment
from clipai.knowledge.domain import KnowledgeCategory, KnowledgeObservation
from clipai.knowledge.provider import LlmProvider


class CandidateScorer(Protocol):
    def score(self, scores: dict[CandidateCategory, float]) -> float: ...


class BalancedCandidateScorer:
    def score(self, scores: dict[CandidateCategory, float]) -> float:
        ordered = sorted(scores.values(), reverse=True)
        return round(ordered[0] * 0.7 + ordered[1] * 0.3, 4)


@dataclass(frozen=True)
class RankingResult:
    category_scores: dict[CandidateCategory, float]
    confidence: float
    reasons: tuple[str, ...]
    extend_before_seconds: float = 0
    extend_after_seconds: float = 0


_KNOWLEDGE_RELATIONS = {
    CandidateCategory.HUMOR: {KnowledgeCategory.RECURRING_JOKE},
    CandidateCategory.MEMORABLE_QUOTE: {
        KnowledgeCategory.RECURRING_PHRASE,
        KnowledgeCategory.SPEECH_PATTERN,
    },
    CandidateCategory.CALLBACK_RUNNING_JOKE: {
        KnowledgeCategory.CALLBACK,
        KnowledgeCategory.RECURRING_JOKE,
    },
    CandidateCategory.EMOTIONAL_MOMENT: {KnowledgeCategory.EMOTIONAL_BASELINE},
    CandidateCategory.STORY_PAYOFF: {KnowledgeCategory.CONTENT_STRENGTH},
}


def relevant_knowledge(
    observations: tuple[KnowledgeObservation, ...],
    categories: set[CandidateCategory],
    maximum: int,
) -> tuple[KnowledgeObservation, ...]:
    related = set().union(*(_KNOWLEDGE_RELATIONS.get(item, set()) for item in categories))
    matches = [item for item in observations if item.category in related]
    return tuple(sorted(matches, key=lambda item: -item.confidence)[:maximum])


class LlmCandidateRanker:
    def __init__(self, provider: LlmProvider, model: str, prompt_path: Path) -> None:
        self._provider = provider
        self._model = model
        self._template = prompt_path.read_text(encoding="utf-8")

    def rank(
        self,
        window: CandidateWindow,
        segments: list[TranscriptSegment],
        knowledge: tuple[KnowledgeObservation, ...],
    ) -> RankingResult:
        payload = {
            "window": [window.start_seconds, window.end_seconds],
            "events": [
                {
                    "type": item.event_type.value,
                    "confidence": item.confidence,
                    "explanation": item.explanation,
                }
                for item in window.events
            ],
            "transcript": [
                {"start": item.start_seconds, "end": item.end_seconds, "text": item.text}
                for item in segments
            ],
            "knowledge": [
                {
                    "category": item.category.value,
                    "statement": item.statement,
                    "confidence": item.confidence,
                }
                for item in knowledge
            ],
        }
        response = self._provider.generate(
            self._template.format(payload=json.dumps(payload, ensure_ascii=False)),
            model=self._model,
        )
        return parse_ranking_response(response)


def parse_ranking_response(response: str) -> RankingResult:
    data = json.loads(response)
    raw_scores = data.get("category_scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("candidate response requires category_scores")
    scores = {
        category: _probability(raw_scores.get(category.value, 0.0), category.value)
        for category in CandidateCategory
    }
    reasons = data.get("reasons")
    if not isinstance(reasons, list) or not reasons or not all(
        isinstance(item, str) and item.strip() for item in reasons
    ):
        raise ValueError("candidate response requires non-empty reasons")
    return RankingResult(
        scores,
        _probability(data.get("confidence"), "confidence"),
        tuple(item.strip() for item in reasons[:5]),
        _extension(data.get("extend_before_seconds", 0)),
        _extension(data.get("extend_after_seconds", 0)),
    )


def _probability(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return float(value)


def _extension(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("extensions must be numeric")
    return max(0.0, min(float(value), 30.0))
