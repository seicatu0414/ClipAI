import json
import math
import re
from dataclasses import dataclass
from typing import Protocol

from clipai.candidates.domain import (
    CandidateWindow,
    EndBoundaryCandidate,
    EndBoundarySelection,
    TopicWindow,
)
from clipai.domain import TranscriptSegment
from clipai.events.domain import JsonValue
from clipai.knowledge.domain import KnowledgeObservation
from clipai.knowledge.provider import LlmProvider

_COMPLETION_MARKERS = ("ということで", "まあいいか", "なるほど", "おしまい", "終わり", "でした", "だね", "ですね")
_TRANSITION_MARKERS = ("はい次", "次は", "さて", "ところで", "話は変わ", "じゃあ", "それでは")


@dataclass(frozen=True)
class _BoundaryPoint:
    timestamp: float
    score: float
    signals: tuple[str, ...]


class EndBoundaryRanker(Protocol):
    def select(
        self,
        window: CandidateWindow,
        topic: TopicWindow,
        candidates: tuple[EndBoundaryCandidate, ...],
        segments: list[TranscriptSegment],
        knowledge: tuple[KnowledgeObservation, ...],
    ) -> tuple[str, float, str]: ...


class LlmEndBoundaryRanker:
    def __init__(self, provider: LlmProvider, model: str, template: str) -> None:
        self._provider = provider
        self._model = model
        self._template = template

    def select(
        self,
        window: CandidateWindow,
        topic: TopicWindow,
        candidates: tuple[EndBoundaryCandidate, ...],
        segments: list[TranscriptSegment],
        knowledge: tuple[KnowledgeObservation, ...],
    ) -> tuple[str, float, str]:
        payload = {
            "clip_start": window.start_seconds,
            "anchor_events": [
                {
                    "type": event.event_type.value,
                    "start": event.start_seconds,
                    "end": event.end_seconds,
                }
                for event in window.events
            ],
            "topic_window": [topic.start_seconds, topic.end_seconds],
            "end_candidates": [
                {
                    "id": item.id,
                    "timestamp": item.timestamp,
                    "confidence": item.confidence,
                    "reason": item.reason,
                    "signals": list(item.source_signals),
                }
                for item in candidates
            ],
            "transcript": [
                {"start": item.start_seconds, "end": item.end_seconds, "text": item.text}
                for item in segments
                if window.start_seconds <= item.end_seconds <= candidates[-1].timestamp + 15
            ],
            "streamer_knowledge": [
                {"statement": item.statement, "confidence": item.confidence} for item in knowledge
            ],
        }
        response = self._provider.generate(
            self._template.format(payload=json.dumps(payload, ensure_ascii=False)),
            model=self._model,
        )
        data = json.loads(response)
        selected_id = data.get("selected_candidate_id")
        reason = data.get("reason")
        confidence = data.get("confidence")
        valid_ids = {item.id for item in candidates}
        if selected_id not in valid_ids:
            raise ValueError("end boundary response must select a supplied candidate id")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("end boundary response requires a reason")
        if not isinstance(confidence, int | float) or isinstance(confidence, bool):
            raise ValueError("end boundary response confidence must be numeric")
        return str(selected_id), max(0.0, min(float(confidence), 1.0)), reason.strip()


class EndBoundaryDetector:
    def __init__(self, ranker: EndBoundaryRanker) -> None:
        self._ranker = ranker

    def detect(
        self,
        window: CandidateWindow,
        context: list[TranscriptSegment],
        knowledge: tuple[KnowledgeObservation, ...],
        *,
        minimum_seconds: float,
        maximum_seconds: float,
        candidate_count: int,
    ) -> EndBoundarySelection:
        points = _boundary_points(context)
        topic = _topic_for_anchor(
            context,
            points,
            _anchor_time(window),
            max(item.end_seconds for item in window.events),
        )
        candidates = _end_candidates(
            window, context, points, topic, minimum_seconds, maximum_seconds, candidate_count
        )
        selected_id, confidence, reason = self._ranker.select(
            window, topic, candidates, context, knowledge
        )
        selected = next(item for item in candidates if item.id == selected_id)
        return EndBoundarySelection(
            selected.timestamp,
            confidence,
            reason,
            selected.source_signals,
            topic,
            candidates,
        )


def boundary_analysis(
    window: CandidateWindow, selection: EndBoundarySelection
) -> dict[str, JsonValue]:
    return {
        "anchor_event_ids": [str(item.id) for item in window.events],
        "start_boundary": {
            "timestamp": window.start_seconds,
            "source": "existing_candidate_window",
        },
        "topic_window": {
            "start": selection.topic_window.start_seconds,
            "end": selection.topic_window.end_seconds,
            "confidence": selection.topic_window.confidence,
            "source_signals": list(selection.topic_window.source_signals),
        },
        "end_boundary_candidates": [
            {
                "id": item.id,
                "timestamp": item.timestamp,
                "confidence": item.confidence,
                "reason": item.reason,
                "source_signals": list(item.source_signals),
            }
            for item in selection.candidates
        ],
        "selected_end_boundary": {
            "timestamp": selection.timestamp,
            "confidence": selection.confidence,
            "reason": selection.reason,
            "source_signals": list(selection.source_signals),
        },
    }


def _anchor_time(window: CandidateWindow) -> float:
    return sum(
        (item.start_seconds + item.end_seconds) / 2 for item in window.events
    ) / len(window.events)


def _boundary_points(segments: list[TranscriptSegment]) -> list[_BoundaryPoint]:
    points: list[_BoundaryPoint] = []
    for previous, current in zip(segments, segments[1:], strict=False):
        gap = max(0.0, current.start_seconds - previous.end_seconds)
        similarity = _similarity(previous.text, current.text)
        previous_rate = _speech_rate(previous)
        current_rate = _speech_rate(current)
        rate_change = abs(previous_rate - current_rate) / max(previous_rate, current_rate, 1.0)
        signals: list[str] = []
        score = 0.0
        if gap >= 1.5:
            signals.append("silence_gap")
            score += min(0.32, gap / 12)
        if similarity < 0.18:
            signals.append("semantic_similarity_drop")
            score += 0.28 * (1 - similarity)
        if previous.text.rstrip().endswith(("。", "！", "？", "!", "?")) or any(
            marker in previous.text for marker in _COMPLETION_MARKERS
        ):
            signals.append("utterance_completion")
            score += 0.18
        if any(marker in current.text for marker in _TRANSITION_MARKERS):
            signals.append("topic_transition_phrase")
            score += 0.3
        if rate_change >= 0.45:
            signals.append("speech_rate_change")
            score += min(0.16, rate_change * 0.16)
        if len(signals) >= 2 and score >= 0.4:
            points.append(_BoundaryPoint(current.start_seconds, min(score, 1.0), tuple(signals)))
    return _smooth_short_detours(points, segments)


def _smooth_short_detours(
    points: list[_BoundaryPoint], segments: list[TranscriptSegment]
) -> list[_BoundaryPoint]:
    kept = list(points)
    index = 0
    while index + 1 < len(kept):
        first, second = kept[index], kept[index + 1]
        if second.timestamp - first.timestamp <= 35:
            before = " ".join(
                item.text
                for item in segments
                if first.timestamp - 25 <= item.end_seconds <= first.timestamp
            )
            after = " ".join(
                item.text
                for item in segments
                if second.timestamp <= item.start_seconds <= second.timestamp + 25
            )
            if _similarity(before, after) >= 0.2:
                del kept[index : index + 2]
                continue
        index += 1
    return kept


def _topic_for_anchor(
    segments: list[TranscriptSegment],
    points: list[_BoundaryPoint],
    anchor: float,
    anchor_end: float,
) -> TopicWindow:
    context_start = segments[0].start_seconds if segments else anchor
    context_end = segments[-1].end_seconds if segments else anchor
    before = [item for item in points if item.timestamp <= anchor]
    after = [
        item
        for item in points
        if item.timestamp > anchor
        and (
            "semantic_similarity_drop" in item.signals
            or "topic_transition_phrase" in item.signals
        )
        and (item.score >= 0.55 or "topic_transition_phrase" in item.signals)
        and (
            item.timestamp >= anchor_end + 5
            or "topic_transition_phrase" in item.signals
        )
    ]
    start_point = before[-1] if before else None
    end_point = after[0] if after else None
    used = tuple(
        dict.fromkeys(
            (start_point.signals if start_point else ())
            + (end_point.signals if end_point else ())
        )
    )
    confidences = [item.score for item in (start_point, end_point) if item is not None]
    return TopicWindow(
        start_point.timestamp if start_point else context_start,
        end_point.timestamp if end_point else context_end,
        round(sum(confidences) / len(confidences), 4) if confidences else 0.35,
        used or ("context_edge",),
    )


def _end_candidates(
    window: CandidateWindow,
    segments: list[TranscriptSegment],
    points: list[_BoundaryPoint],
    topic: TopicWindow,
    minimum: float,
    maximum: float,
    count: int,
) -> tuple[EndBoundaryCandidate, ...]:
    earliest = window.start_seconds + minimum
    latest = window.start_seconds + maximum
    proposals: dict[float, tuple[float, str, tuple[str, ...]]] = {}
    for segment in segments:
        timestamp = min(segment.end_seconds, latest)
        if not earliest <= timestamp <= latest:
            continue
        signals: list[str] = []
        score = 0.2
        if segment.text.rstrip().endswith(("。", "！", "？", "!", "?")):
            signals.append("sentence_completion")
            score += 0.22
        if any(marker in segment.text for marker in _COMPLETION_MARKERS):
            signals.append("conversation_completion_phrase")
            score += 0.28
        next_segment = next(
            (item for item in segments if item.start_seconds >= segment.end_seconds), None
        )
        if next_segment and next_segment.start_seconds - segment.end_seconds >= 1.5:
            signals.append("before_silence")
            score += 0.22
        if signals:
            proposals[round(timestamp, 3)] = (min(score, 1.0), "発話が自然に完結する地点", tuple(signals))
    for point in points:
        if earliest <= point.timestamp <= latest:
            proposals[round(point.timestamp, 3)] = (
                max(0.55, point.score), "推定された話題切り替わりの直前", point.signals
            )
    if earliest <= topic.end_seconds <= latest:
        proposals[round(topic.end_seconds, 3)] = (
            max(0.6, topic.confidence), "対象Eventが属する話題の終了地点", topic.source_signals
        )
    cap_segment = max(
        (item for item in segments if item.end_seconds <= latest),
        key=lambda item: item.end_seconds,
        default=None,
    )
    cap = cap_segment.end_seconds if cap_segment else latest
    proposals.setdefault(
        round(max(earliest, cap), 3),
        (0.35, "120秒制約内の最終発話地点", ("maximum_duration_guard",)),
    )
    eligible_ends = sorted(
        {
            round(item.end_seconds, 3)
            for item in segments
            if earliest <= item.end_seconds <= latest
        }
    )
    while len(proposals) < min(3, count) and eligible_ends:
        target = earliest + (latest - earliest) * (len(proposals) + 1) / 4
        timestamp = min(eligible_ends, key=lambda item: (abs(item - target), item))
        eligible_ends.remove(timestamp)
        proposals.setdefault(
            timestamp,
            (0.25, "発話単位に揃えた補助終了候補", ("utterance_boundary",)),
        )
    fallback_index = 1
    while len(proposals) < min(3, count):
        timestamp = round(
            earliest + (latest - earliest) * fallback_index / min(3, count), 3
        )
        fallback_index += 1
        proposals.setdefault(
            min(timestamp, latest),
            (0.15, "字幕が疎な区間の時間制約内補助候補", ("duration_fallback",)),
        )
    ordered = sorted(
        proposals.items(),
        key=lambda item: (
            abs(item[0] - min(topic.end_seconds, latest)) - item[1][0] * 20,
            item[0],
        ),
    )[:count]
    ordered.sort(key=lambda item: item[0])
    return tuple(
        EndBoundaryCandidate(f"end_{index}", timestamp, round(data[0], 4), data[1], data[2])
        for index, (timestamp, data) in enumerate(ordered, start=1)
    )


def _speech_rate(segment: TranscriptSegment) -> float:
    return len(segment.text.strip()) / max(segment.end_seconds - segment.start_seconds, 0.1)


def _similarity(first: str, second: str) -> float:
    first_tokens = _tokens(first)
    second_tokens = _tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / math.sqrt(len(first_tokens) * len(second_tokens))


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    words = set(re.findall(r"[a-z0-9]+", normalized))
    return words | {normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))}
