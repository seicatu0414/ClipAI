import logging
from collections.abc import Callable
from pathlib import Path

from clipai.candidates.boundaries import (
    EndBoundaryDetector,
    LlmEndBoundaryRanker,
    boundary_analysis,
)
from clipai.candidates.domain import (
    CandidateCategory,
    CandidateJob,
    CandidateWindow,
    ClipCandidate,
)
from clipai.candidates.ranking import (
    CandidateScorer,
    LlmCandidateRanker,
    RankingResult,
    WeightedCandidateScorer,
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
        self._scorer = scorer

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
            provider = self._provider_factory(job)
            ranker = LlmCandidateRanker(
                provider,
                job.model,
                self._prompt_root / "candidate_ranking" / f"{job.prompt_version}.md",
            )
            end_detector = EndBoundaryDetector(
                LlmEndBoundaryRanker(
                    provider,
                    job.model,
                    (
                        self._prompt_root
                        / "end_boundary_ranking"
                        / f"{job.prompt_version}.md"
                    ).read_text(encoding="utf-8"),
                )
            )
            scorer = self._scorer or WeightedCandidateScorer(
                self._repository.load_preference_weights(job.preference_version_id)
            )
            ranked: list[ClipCandidate] = []
            maximum = _integer(config, "maximum_knowledge_observations")
            for index, window in enumerate(windows):
                categories = set().union(
                    *(_EVENT_CATEGORIES[item.event_type] for item in window.events)
                )
                knowledge = relevant_knowledge(observations, categories, maximum)
                context_seconds = _number(config, "context_window_seconds")
                anchor = sum(
                    (item.start_seconds + item.end_seconds) / 2
                    for item in window.events
                ) / len(window.events)
                context = self._repository.segments_for(
                    job.transcript_id,
                    max(0.0, anchor - context_seconds / 2),
                    anchor + context_seconds / 2,
                )
                selection = end_detector.detect(
                    window, context, knowledge,
                    minimum_seconds=_number(config, "minimum_seconds"),
                    maximum_seconds=_number(config, "maximum_seconds"),
                    candidate_count=_integer(config, "end_boundary_count"),
                )
                bounded_window = CandidateWindow(
                    window.start_seconds,
                    selection.timestamp,
                    window.events,
                    window.preliminary_score,
                )
                segments = [
                    item
                    for item in context
                    if item.end_seconds >= bounded_window.start_seconds
                    and item.start_seconds <= bounded_window.end_seconds
                ]
                result = ranker.rank(bounded_window, segments, knowledge)
                start, end = _extended_window(bounded_window, result, config)
                ranked.append(
                    ClipCandidate(
                        None,
                        0,
                        start,
                        end,
                        result.category_scores,
                        scorer.score(result.category_scores),
                        result.confidence,
                        result.reasons,
                        tuple(item.id for item in window.events),
                        tuple(knowledge_ids[item] for item in knowledge),
                        knowledge,
                        boundary_analysis(window, selection),
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
    # EndBoundaryDetector has already ranked explicit natural endings. The content
    # ranker's legacy extension must not override that independently selected boundary.
    end = window.end_seconds
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
        ClipCandidate(item.id, index, item.start_seconds, item.end_seconds, item.category_scores,
                      item.overall_score, item.confidence, item.reasons,
                      item.event_ids, item.knowledge_observation_ids, item.knowledge,
                      item.boundary_analysis)
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
