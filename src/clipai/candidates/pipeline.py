import logging
from collections.abc import Callable
from pathlib import Path

from clipai.candidates.domain import (
    CandidateCategory,
    CandidateJob,
    CandidateWindow,
    ClipCandidate,
)
from clipai.candidates.ranking import (
    BalancedCandidateScorer,
    CandidateScorer,
    LlmCandidateRanker,
    RankingResult,
    relevant_knowledge,
)
from clipai.candidates.repository import CandidateRepository
from clipai.candidates.windowing import construct_windows, overlap_ratio, reduce_windows
from clipai.events.domain import EventType, JsonValue
from clipai.knowledge.provider import LlmProvider

LOGGER = logging.getLogger(__name__)

_EVENT_CATEGORIES = {
    EventType.LAUGHTER: {CandidateCategory.HUMOR},
    EventType.LOUD_REACTION: {CandidateCategory.STRONG_REACTION},
    EventType.SINGING: {CandidateCategory.EMOTIONAL_MOMENT},
    EventType.EMOTIONAL_VOICE: {CandidateCategory.EMOTIONAL_MOMENT},
    EventType.UNUSUAL_SILENCE: {CandidateCategory.STORY_PAYOFF},
    EventType.VICTORY_DEFEAT: {
        CandidateCategory.GREAT_PLAY,
        CandidateCategory.STORY_PAYOFF,
    },
    EventType.MEMORABLE_STATEMENT: {CandidateCategory.MEMORABLE_QUOTE},
    EventType.VIEWER_RESPONSE: {CandidateCategory.VIEWER_INTERACTION},
    EventType.CALLBACK_CONTRADICTION: {CandidateCategory.CALLBACK_RUNNING_JOKE},
}


class CandidatePipeline:
    def __init__(
        self,
        repository: CandidateRepository,
        provider_factory: Callable[[CandidateJob], LlmProvider],
        prompt_root: Path,
        scorer: CandidateScorer | None = None,
    ) -> None:
        self._repository = repository
        self._provider_factory = provider_factory
        self._prompt_root = prompt_root
        self._scorer = scorer or BalancedCandidateScorer()

    def process(self, job: CandidateJob) -> None:
        try:
            config = job.configuration
            windows = construct_windows(
                self._repository.load_events(job),
                minimum_seconds=_number(config, "minimum_seconds"),
                maximum_seconds=_number(config, "maximum_seconds"),
                padding_before_seconds=_number(config, "padding_before_seconds"),
                padding_after_seconds=_number(config, "padding_after_seconds"),
                merge_gap_seconds=_number(config, "merge_gap_seconds"),
            )
            windows = reduce_windows(
                windows,
                target_count=_integer(config, "target_count"),
                overlap_threshold=_number(config, "overlap_threshold"),
            )
            self._repository.update_progress(job.id, 20)
            indexed_knowledge = self._repository.load_knowledge(job)
            knowledge_ids = {item: identity for identity, item in indexed_knowledge}
            observations = tuple(item for _, item in indexed_knowledge)
            ranker = LlmCandidateRanker(
                self._provider_factory(job),
                job.model,
                self._prompt_root / "candidate_ranking" / f"{job.prompt_version}.md",
            )
            ranked: list[ClipCandidate] = []
            maximum = _integer(config, "maximum_knowledge_observations")
            for index, window in enumerate(windows):
                categories = set().union(
                    *(_EVENT_CATEGORIES[item.event_type] for item in window.events)
                )
                knowledge = relevant_knowledge(observations, categories, maximum)
                segments = self._repository.segments_for(
                    job.transcript_id, window.start_seconds, window.end_seconds
                )
                result = ranker.rank(window, segments, knowledge)
                start, end = _extended_window(window, result, config)
                ranked.append(
                    ClipCandidate(
                        0,
                        start,
                        end,
                        result.category_scores,
                        self._scorer.score(result.category_scores),
                        result.confidence,
                        result.reasons,
                        tuple(item.id for item in window.events),
                        tuple(knowledge_ids[item] for item in knowledge),
                        knowledge,
                    )
                )
                self._repository.update_progress(
                    job.id, 20 + round(70 * (index + 1) / max(1, len(windows)))
                )
            final = _rank_and_suppress(
                ranked, _number(config, "overlap_threshold")
            )
            self._repository.save_candidates(job, final)
            LOGGER.info(
                "clip_candidates_completed",
                extra={"job_id": str(job.id), "candidate_count": len(final)},
            )
        except Exception as error:
            LOGGER.exception("clip_candidates_failed", extra={"job_id": str(job.id)})
            self._repository.mark_failed(job.id, str(error) or error.__class__.__name__)


def _extended_window(
    window: CandidateWindow,
    result: RankingResult,
    config: dict[str, JsonValue],
) -> tuple[float, float]:
    minimum = _number(config, "minimum_seconds")
    maximum = _number(config, "maximum_seconds")
    start = max(0.0, window.start_seconds - result.extend_before_seconds)
    end = window.end_seconds + result.extend_after_seconds
    end = max(end, start + minimum)
    if end - start > maximum:
        end = start + maximum
    return start, end


def _rank_and_suppress(
    candidates: list[ClipCandidate], overlap_threshold: float
) -> list[ClipCandidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (-item.overall_score, -item.confidence, item.start_seconds),
    )
    selected: list[ClipCandidate] = []
    for candidate in ordered:
        proxy = _proxy(candidate)
        if all(overlap_ratio(proxy, _proxy(item)) < overlap_threshold for item in selected):
            selected.append(candidate)
    return [
        ClipCandidate(index, item.start_seconds, item.end_seconds, item.category_scores,
                      item.overall_score, item.confidence, item.reasons,
                      item.event_ids, item.knowledge_observation_ids, item.knowledge)
        for index, item in enumerate(selected, start=1)
    ]


def _proxy(candidate: ClipCandidate) -> CandidateWindow:
    return CandidateWindow(candidate.start_seconds, candidate.end_seconds, (), 0)


def _number(configuration: dict[str, JsonValue], key: str) -> float:
    value = configuration.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"candidate configuration {key!r} must be numeric")
    return float(value)


def _integer(configuration: dict[str, JsonValue], key: str) -> int:
    value = configuration.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"candidate configuration {key!r} must be an integer")
    return value
