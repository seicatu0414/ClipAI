import json
import math
import re
from dataclasses import dataclass
from typing import Protocol

from clipai.candidates.domain import (
    CandidateWindow,
    EndBoundaryCandidate,
    EndBoundarySelection,
    OpenThread,
    ScenePhase,
    SceneWindow,
    TopicWindow,
)
from clipai.candidates.scenes import SceneTimeline, build_scene_timeline
from clipai.domain import TranscriptSegment
from clipai.events.domain import JsonValue
from clipai.knowledge.domain import KnowledgeObservation
from clipai.knowledge.provider import LlmProvider

_COMPLETION_MARKERS = (
    "ということで",
    "まあいいか",
    "なるほど",
    "おしまい",
    "終わり",
    "でした",
    "だね",
    "ですね",
)
_TRANSITION_MARKERS = (
    "はい次",
    "次は",
    "さて",
    "ところで",
    "話は変わ",
    "じゃあ",
    "それでは",
)


@dataclass(frozen=True)
class _BoundaryPoint:
    timestamp: float
    score: float
    signals: tuple[str, ...]


@dataclass(frozen=True)
class _AnalysisPass:
    context: list[TranscriptSegment]
    context_start: float
    context_end: float
    topic: TopicWindow
    scene: SceneWindow
    candidates: tuple[EndBoundaryCandidate, ...]


class EndBoundaryRanker(Protocol):
    def select(
        self,
        window: CandidateWindow,
        topic: TopicWindow,
        scene: SceneWindow,
        candidates: tuple[EndBoundaryCandidate, ...],
        segments: list[TranscriptSegment],
        knowledge: tuple[KnowledgeObservation, ...],
        *,
        detailed: bool,
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
        scene: SceneWindow,
        candidates: tuple[EndBoundaryCandidate, ...],
        segments: list[TranscriptSegment],
        knowledge: tuple[KnowledgeObservation, ...],
        *,
        detailed: bool,
    ) -> tuple[str, float, str]:
        payload = {
            "analysis_mode": "detailed_scene" if detailed else "normal_scene",
            "clip_start": window.start_seconds,
            "anchor_events": [
                {
                    "type": event.event_type.value,
                    "start": event.start_seconds,
                    "end": event.end_seconds,
                    "explanation": event.explanation,
                }
                for event in window.events
            ],
            "topic_window": {
                "start": topic.start_seconds,
                "end": topic.end_seconds,
                "confidence": topic.confidence,
            },
            "scene": _scene_payload(scene),
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
                if window.start_seconds <= item.end_seconds <= candidates[-1].timestamp + 20
            ],
            "streamer_knowledge": [
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
        valid_ids = {item.id for item in candidates}
        try:
            return _parse_ranking_response(response, valid_ids)
        except ValueError:
            repair_payload = {
                "allowed_ids": sorted(valid_ids),
                "candidate_summaries": [
                    {
                        "id": item.id,
                        "timestamp": item.timestamp,
                        "reason": item.reason,
                        "signals": list(item.source_signals),
                    }
                    for item in candidates
                ],
                "invalid_response": response[:1000],
            }
            repaired = self._provider.generate(
                "Select one allowed end-boundary ID. Return JSON only with "
                "selected_candidate_id, confidence (0..1), and a short Japanese "
                "reason. Do not summarize a scene or invent an ID. Input: "
                + json.dumps(repair_payload, ensure_ascii=False),
                model=self._model,
            )
            return _parse_ranking_response(repaired, valid_ids)


def _parse_ranking_response(
    response: str, valid_ids: set[str]
) -> tuple[str, float, str]:
    data = json.loads(response)
    selected_id = data.get("selected_candidate_id", data.get("candidate_id"))
    reason = data.get("reason")
    confidence = data.get("confidence")
    if selected_id not in valid_ids:
        raise ValueError("end boundary response must select a supplied candidate id")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("end boundary response requires a reason")
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise ValueError("end boundary response confidence must be numeric")
    return (
        str(selected_id),
        max(0.0, min(float(confidence), 1.0)),
        reason.strip(),
    )


class EndBoundaryDetector:
    def __init__(self, ranker: EndBoundaryRanker) -> None:
        self._ranker = ranker

    def detect(
        self,
        window: CandidateWindow,
        segments: list[TranscriptSegment],
        knowledge: tuple[KnowledgeObservation, ...],
        *,
        minimum_seconds: float,
        maximum_seconds: float,
        candidate_count: int,
        context_window_seconds: float = 600.0,
        maximum_context_seconds: float = 1800.0,
        context_expansion_seconds: float = 300.0,
        detailed_confidence_threshold: float = 0.65,
        timeline: SceneTimeline | None = None,
    ) -> EndBoundarySelection:
        timeline = timeline or build_scene_timeline(segments)
        anchor = _anchor_time(window)
        initial_start = max(0.0, anchor - context_window_seconds / 2)
        initial_end = anchor + context_window_seconds / 2
        normal = _analyze(
            window,
            segments,
            timeline,
            initial_start,
            initial_end,
            minimum_seconds,
            maximum_seconds,
            candidate_count,
        )
        selected_id, confidence, reason = self._ranker.select(
            window,
            normal.topic,
            normal.scene,
            normal.candidates,
            normal.context,
            knowledge,
            detailed=False,
        )
        selected = _selected(normal.candidates, selected_id)
        detailed = _needs_detailed_analysis(
            normal,
            selected,
            confidence,
            window,
            maximum_seconds,
            detailed_confidence_threshold,
        )
        final_pass = normal
        if detailed:
            while True:
                expanded_start, expanded_end = _expanded_context(
                    final_pass,
                    anchor,
                    maximum_context_seconds,
                    context_expansion_seconds,
                )
                if (
                    expanded_start == final_pass.context_start
                    and expanded_end == final_pass.context_end
                ):
                    break
                final_pass = _analyze(
                    window,
                    segments,
                    timeline,
                    expanded_start,
                    expanded_end,
                    minimum_seconds,
                    maximum_seconds,
                    candidate_count,
                )
                duration = final_pass.context_end - final_pass.context_start
                if (
                    "context_edge" not in final_pass.scene.source_signals
                    or duration >= maximum_context_seconds
                ):
                    break
            selected_id, confidence, reason = self._ranker.select(
                window,
                final_pass.topic,
                final_pass.scene,
                final_pass.candidates,
                final_pass.context,
                knowledge,
                detailed=True,
            )
            selected = _selected(final_pass.candidates, selected_id)
        selected, reason = _stabilize_selection(
            selected,
            reason,
            final_pass.candidates,
            final_pass.scene,
        )
        confidence = _calibrated_confidence(
            confidence,
            selected,
            final_pass.scene,
        )
        return EndBoundarySelection(
            selected.timestamp,
            confidence,
            reason,
            selected.source_signals,
            final_pass.topic,
            final_pass.scene,
            final_pass.candidates,
            final_pass.context_start,
            final_pass.context_end,
            True,
            detailed,
        )


def _stabilize_selection(
    selected: EndBoundaryCandidate,
    reason: str,
    candidates: tuple[EndBoundaryCandidate, ...],
    scene: SceneWindow,
) -> tuple[EndBoundaryCandidate, str]:
    important_open = any(item.confidence >= 0.65 for item in scene.open_threads)
    if important_open:
        return selected, reason
    natural = [
        item
        for item in candidates
        if item.confidence >= 0.65
        and (
            scene.reaction_state not in ("reaction_active", "reaction_pending")
            or "aftermath_completion" in item.source_signals
        )
        and any(
            signal in item.source_signals
            for signal in (
                "scene_boundary",
                "aftermath_completion",
                "open_thread_resolved",
            )
        )
    ]
    if not natural:
        return selected, reason
    aftermath = [
        item for item in natural if "aftermath_completion" in item.source_signals
    ]
    if ScenePhase.AFTERMATH in scene.phases and aftermath:
        natural = aftermath
    compact = min(natural, key=lambda item: item.timestamp)
    if (
        selected.timestamp < compact.timestamp
        and not any(
            signal in selected.source_signals
            for signal in (
                "scene_boundary",
                "aftermath_completion",
                "open_thread_resolved",
            )
        )
    ):
        return (
            compact,
            "Scene完結前の局所境界を避け、最初の自然な完結候補まで保持",
        )
    if selected.timestamp - compact.timestamp < 60:
        return selected, reason
    return (
        compact,
        "重要なOpen ThreadがなくScene完結済みのため、後続Topicを含めない最短候補へ安定化",
    )


def _calibrated_confidence(
    llm_confidence: float,
    selected: EndBoundaryCandidate,
    scene: SceneWindow,
) -> float:
    value = (
        llm_confidence * 0.55
        + selected.confidence * 0.25
        + scene.completion_confidence * 0.2
    )
    if any(item.confidence >= 0.65 for item in scene.open_threads):
        value -= 0.1
    if "hard_maximum_guard" in selected.source_signals:
        value = min(value, 0.45)
    return round(max(0.0, min(value, 1.0)), 4)


def boundary_analysis(
    window: CandidateWindow, selection: EndBoundarySelection
) -> dict[str, JsonValue]:
    scene = selection.scene_window
    return {
        "anchor_event_ids": [str(item.id) for item in window.events],
        "start_boundary": {
            "timestamp": window.start_seconds,
            "source": "existing_candidate_window",
        },
        "context_window": {
            "start": selection.context_start_seconds,
            "end": selection.context_end_seconds,
            "duration_seconds": (
                selection.context_end_seconds - selection.context_start_seconds
            ),
        },
        "topic_window": {
            "start": selection.topic_window.start_seconds,
            "end": selection.topic_window.end_seconds,
            "confidence": selection.topic_window.confidence,
            "source_signals": list(selection.topic_window.source_signals),
        },
        "scene_window": _scene_payload(scene),
        "scene_phase": scene.phase.value,
        "open_threads": [_thread_payload(item) for item in scene.open_threads],
        "resolved_threads": [_thread_payload(item) for item in scene.resolved_threads],
        "emotional_state": scene.emotional_state,
        "reaction_state": scene.reaction_state,
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
        "boundary_confidence": selection.confidence,
        "scene_completion_confidence": scene.completion_confidence,
        "llm_used": selection.llm_used,
        "detailed_analysis_used": selection.detailed_analysis_used,
    }


def _analyze(
    window: CandidateWindow,
    all_segments: list[TranscriptSegment],
    timeline: SceneTimeline,
    context_start: float,
    context_end: float,
    minimum: float,
    maximum: float,
    candidate_count: int,
) -> _AnalysisPass:
    context = [
        item
        for item in all_segments
        if item.end_seconds >= context_start and item.start_seconds <= context_end
    ]
    points = _boundary_points(context)
    topic = _topic_for_anchor(
        context,
        points,
        _anchor_time(window),
        max(item.end_seconds for item in window.events),
    )
    scene = timeline.clipped_scene(_anchor_time(window), context_start, context_end)
    candidates = _end_candidates(
        window,
        context,
        timeline,
        points,
        topic,
        scene,
        minimum,
        maximum,
        candidate_count,
    )
    return _AnalysisPass(
        context,
        context_start,
        context_end,
        topic,
        scene,
        candidates,
    )


def _needs_detailed_analysis(
    analysis: _AnalysisPass,
    selected: EndBoundaryCandidate,
    confidence: float,
    window: CandidateWindow,
    maximum_seconds: float,
    threshold: float,
) -> bool:
    scores = sorted(
        (item.confidence for item in analysis.candidates), reverse=True
    )
    close_scores = (
        len(scores) >= 2
        and scores[0] - scores[1] < 0.08
        and confidence < max(0.8, threshold)
    )
    near_hard_maximum = (
        selected.timestamp - window.start_seconds >= maximum_seconds * 0.9
    )
    important_open = any(
        item.confidence >= 0.65 for item in analysis.scene.open_threads
    )
    topic_scene_conflict = (
        abs(analysis.topic.end_seconds - analysis.scene.end_seconds) >= 30
    )
    context_incomplete = "context_edge" in analysis.scene.source_signals
    return (
        confidence < threshold
        or close_scores
        or near_hard_maximum
        or important_open
        or topic_scene_conflict
        or context_incomplete
        or analysis.scene.completion_confidence < threshold
    )


def _expanded_context(
    analysis: _AnalysisPass,
    anchor: float,
    maximum_context_seconds: float,
    expansion_seconds: float,
) -> tuple[float, float]:
    desired_start = min(
        analysis.context_start,
        max(0.0, analysis.scene.start_seconds - 30),
    )
    desired_end = max(
        analysis.context_end + expansion_seconds,
        analysis.scene.end_seconds + 60,
    )
    if desired_end - desired_start > maximum_context_seconds:
        desired_start = max(0.0, anchor - maximum_context_seconds / 3)
        desired_end = desired_start + maximum_context_seconds
    return desired_start, desired_end


def _end_candidates(
    window: CandidateWindow,
    segments: list[TranscriptSegment],
    timeline: SceneTimeline,
    points: list[_BoundaryPoint],
    topic: TopicWindow,
    scene: SceneWindow,
    minimum: float,
    maximum: float,
    count: int,
) -> tuple[EndBoundaryCandidate, ...]:
    earliest = window.start_seconds + minimum
    latest = window.start_seconds + maximum
    important_open = any(item.confidence >= 0.65 for item in scene.open_threads)
    if important_open:
        candidate_horizon = latest
    elif scene.reaction_state in ("reaction_active", "reaction_pending"):
        candidate_horizon = min(latest, scene.end_seconds + 240)
    else:
        candidate_horizon = min(latest, scene.end_seconds + 30)
    effective_latest = max(earliest, candidate_horizon)
    proposals: dict[float, tuple[float, str, tuple[str, ...]]] = {}
    context_chunks = [
        item
        for item in timeline.chunks
        if item.end_seconds >= max(earliest, segments[0].start_seconds if segments else earliest)
        and item.start_seconds
        <= min(
            effective_latest,
            segments[-1].end_seconds if segments else effective_latest,
        )
    ]
    for chunk in context_chunks:
        timestamp = chunk.end_seconds
        if not earliest <= timestamp <= effective_latest:
            continue
        if chunk.phase is ScenePhase.AFTERMATH:
            _proposal(
                proposals,
                timestamp,
                0.72,
                "CLIMAX後の感想・余韻が完了する地点",
                ("aftermath_completion",),
            )
        if chunk.phase is ScenePhase.TRANSITION:
            _proposal(
                proposals,
                chunk.start_seconds,
                0.9,
                "次の独立Sceneへ移る直前",
                ("scene_transition",),
            )
    for thread in scene.resolved_threads:
        if (
            thread.resolved_at is not None
            and earliest <= thread.resolved_at <= effective_latest
        ):
            _proposal(
                proposals,
                thread.resolved_at,
                min(0.92, thread.confidence),
                "重要なOpen Threadが解決した地点",
                ("open_thread_resolved",),
            )
    if earliest <= scene.end_seconds <= effective_latest:
        scene_score = max(0.45, scene.completion_confidence)
        signals = tuple(
            dict.fromkeys(scene.source_signals + ("scene_boundary",))
        )
        _proposal(
            proposals,
            scene.end_seconds,
            scene_score,
            "一連のSceneが完結する地点",
            signals,
        )
    for segment in segments:
        timestamp = segment.end_seconds
        if not earliest <= timestamp <= effective_latest:
            continue
        signals: list[str] = []
        score = 0.18
        if segment.text.rstrip().endswith(("。", "！", "？", "!", "?")):
            signals.append("sentence_completion")
            score += 0.18
        if any(marker in segment.text for marker in _COMPLETION_MARKERS):
            signals.append("conversation_completion_phrase")
            score += 0.22
        next_segment = next(
            (
                item
                for item in segments
                if item.start_seconds >= segment.end_seconds
            ),
            None,
        )
        if (
            next_segment
            and next_segment.start_seconds - segment.end_seconds >= 1.5
        ):
            signals.append("before_silence")
            score += 0.16
        if signals:
            _proposal(
                proposals,
                timestamp,
                min(score, 1.0),
                "発話が自然に完結する地点",
                tuple(signals),
            )
    for point in points:
        if earliest <= point.timestamp <= effective_latest:
            _proposal(
                proposals,
                point.timestamp,
                max(0.5, point.score),
                "複数シグナルによるTopic境界",
                point.signals + ("topic_boundary",),
            )
    if earliest <= topic.end_seconds <= effective_latest:
        _proposal(
            proposals,
            topic.end_seconds,
            max(0.5, topic.confidence),
            "対象Eventが属するTopicの終了地点",
            topic.source_signals + ("topic_boundary",),
        )
    cap_segment = max(
        (item for item in segments if item.end_seconds <= effective_latest),
        key=lambda item: item.end_seconds,
        default=None,
    )
    cap = cap_segment.end_seconds if cap_segment else effective_latest
    _proposal(
        proposals,
        max(earliest, cap),
        0.2,
        (
            "15分Hard Maximum内の安全な発話境界"
            if effective_latest >= latest
            else "Anchor Scene周辺の安全な探索上限"
        ),
        (
            ("hard_maximum_guard",)
            if effective_latest >= latest
            else ("scene_candidate_horizon",)
        ),
    )
    _ensure_minimum_candidates(
        proposals,
        segments,
        earliest,
        effective_latest,
        count,
    )
    if important_open:
        for timestamp, data in list(proposals.items()):
            if timestamp < scene.end_seconds:
                proposals[timestamp] = (
                    max(0.0, data[0] - 0.25),
                    data[1],
                    tuple(dict.fromkeys(data[2] + ("open_thread_penalty",))),
                )
    ordered = sorted(
        proposals.items(),
        key=lambda item: (
            -item[1][0],
            abs(item[0] - min(scene.end_seconds, latest)),
            item[0],
        ),
    )[:count]
    cap_key = round(max(earliest, cap), 3)
    if scene.end_seconds >= latest and all(item[0] != cap_key for item in ordered):
        cap_item = (cap_key, proposals[cap_key])
        ordered = ordered[: max(0, count - 1)] + [cap_item]
    ordered.sort(key=lambda item: item[0])
    return tuple(
        EndBoundaryCandidate(
            f"end_{index}",
            timestamp,
            round(data[0], 4),
            data[1],
            data[2],
        )
        for index, (timestamp, data) in enumerate(ordered, start=1)
    )


def _proposal(
    proposals: dict[float, tuple[float, str, tuple[str, ...]]],
    timestamp: float,
    score: float,
    reason: str,
    signals: tuple[str, ...],
) -> None:
    key = round(timestamp, 3)
    previous = proposals.get(key)
    if previous is None:
        proposals[key] = (score, reason, signals)
        return
    merged_signals = tuple(dict.fromkeys(previous[2] + signals))
    if score > previous[0]:
        proposals[key] = (score, reason, merged_signals)
    else:
        proposals[key] = (previous[0], previous[1], merged_signals)


def _ensure_minimum_candidates(
    proposals: dict[float, tuple[float, str, tuple[str, ...]]],
    segments: list[TranscriptSegment],
    earliest: float,
    latest: float,
    count: int,
) -> None:
    required = min(3, count)
    eligible = sorted(
        {
            round(item.end_seconds, 3)
            for item in segments
            if earliest <= item.end_seconds <= latest
        }
    )
    while len(proposals) < required and eligible:
        target = earliest + (latest - earliest) * (len(proposals) + 1) / 4
        timestamp = min(eligible, key=lambda item: (abs(item - target), item))
        eligible.remove(timestamp)
        _proposal(
            proposals,
            timestamp,
            0.18,
            "発話単位に揃えた補助終了候補",
            ("utterance_boundary",),
        )
    fallback_index = 1
    while len(proposals) < required:
        timestamp = round(
            earliest + (latest - earliest) * fallback_index / required,
            3,
        )
        fallback_index += 1
        _proposal(
            proposals,
            min(timestamp, latest),
            0.12,
            "字幕が疎な区間の時間制約内補助候補",
            ("duration_fallback",),
        )


def _selected(
    candidates: tuple[EndBoundaryCandidate, ...], selected_id: str
) -> EndBoundaryCandidate:
    return next(item for item in candidates if item.id == selected_id)


def _scene_payload(scene: SceneWindow) -> dict[str, JsonValue]:
    return {
        "start": scene.start_seconds,
        "end": scene.end_seconds,
        "phase": scene.phase.value,
        "phases": [item.value for item in scene.phases],
        "primary_goal": scene.primary_goal,
        "open_threads": [_thread_payload(item) for item in scene.open_threads],
        "resolved_threads": [
            _thread_payload(item) for item in scene.resolved_threads
        ],
        "emotional_state": scene.emotional_state,
        "reaction_state": scene.reaction_state,
        "transition_signal": scene.transition_signal,
        "confidence": scene.confidence,
        "completion_confidence": scene.completion_confidence,
        "source_signals": list(scene.source_signals),
    }


def _thread_payload(thread: OpenThread) -> dict[str, JsonValue]:
    return {
        "thread": thread.thread,
        "status": thread.status.value,
        "confidence": thread.confidence,
        "opened_at": thread.opened_at,
        "resolved_at": thread.resolved_at,
    }


def _anchor_time(window: CandidateWindow) -> float:
    return sum(
        (item.start_seconds + item.end_seconds) / 2
        for item in window.events
    ) / len(window.events)


def _boundary_points(
    segments: list[TranscriptSegment],
) -> list[_BoundaryPoint]:
    points: list[_BoundaryPoint] = []
    for previous, current in zip(segments, segments[1:], strict=False):
        gap = max(0.0, current.start_seconds - previous.end_seconds)
        similarity = _similarity(previous.text, current.text)
        previous_rate = _speech_rate(previous)
        current_rate = _speech_rate(current)
        rate_change = abs(previous_rate - current_rate) / max(
            previous_rate, current_rate, 1.0
        )
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
            points.append(
                _BoundaryPoint(
                    current.start_seconds,
                    min(score, 1.0),
                    tuple(signals),
                )
            )
    return _smooth_short_detours(points, segments)


def _smooth_short_detours(
    points: list[_BoundaryPoint],
    segments: list[TranscriptSegment],
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
                if second.timestamp
                <= item.start_seconds
                <= second.timestamp + 25
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
        and (
            item.score >= 0.55
            or "topic_transition_phrase" in item.signals
        )
        and (
            item.timestamp >= anchor_end + 5
            or "topic_transition_phrase" in item.signals
        )
    ]
    start_point = before[-1] if before else None
    end_point = after[0] if after else None
    signals = tuple(
        dict.fromkeys(
            (start_point.signals if start_point else ())
            + (end_point.signals if end_point else ())
        )
    )
    confidences = [
        item.score
        for item in (start_point, end_point)
        if item is not None
    ]
    return TopicWindow(
        start_point.timestamp if start_point else context_start,
        end_point.timestamp if end_point else context_end,
        (
            round(sum(confidences) / len(confidences), 4)
            if confidences
            else 0.35
        ),
        signals or ("context_edge",),
    )


def _speech_rate(segment: TranscriptSegment) -> float:
    return len(segment.text.strip()) / max(
        segment.end_seconds - segment.start_seconds,
        0.1,
    )


def _similarity(first: str, second: str) -> float:
    first_tokens = _tokens(first)
    second_tokens = _tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / math.sqrt(
        len(first_tokens) * len(second_tokens)
    )


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    words = set(re.findall(r"[a-z0-9]+", normalized))
    return words | {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
    }
