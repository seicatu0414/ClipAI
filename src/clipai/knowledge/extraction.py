import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from clipai.domain import TranscriptSegment
from clipai.knowledge.domain import (
    Evidence,
    HistoricalStream,
    KnowledgeCategory,
    KnowledgeObservation,
    ObservationOrigin,
    Streamer,
)
from clipai.knowledge.provider import LlmProvider


@dataclass(frozen=True)
class TranscriptChunk:
    stream: HistoricalStream
    segments: tuple[TranscriptSegment, ...]
    rendered: str


def chunk_segments(
    stream: HistoricalStream,
    segments: Iterable[TranscriptSegment],
    *,
    maximum_characters: int,
) -> Iterator[TranscriptChunk]:
    current: list[TranscriptSegment] = []
    lines: list[str] = []
    length = 0
    for segment in segments:
        prefix = f"[{segment.index}] {segment.start_seconds:.2f}-{segment.end_seconds:.2f} "
        line = prefix + segment.text
        if len(line) > maximum_characters:
            line = line[:maximum_characters]
        if lines and length + len(line) + 1 > maximum_characters:
            yield TranscriptChunk(stream, tuple(current), "\n".join(lines))
            current, lines, length = [], [], 0
        current.append(segment)
        lines.append(line)
        length += len(line) + 1
    if lines:
        yield TranscriptChunk(stream, tuple(current), "\n".join(lines))


class KnowledgeExtractor:
    def __init__(
        self,
        provider: LlmProvider,
        *,
        model: str,
        prompt_path: Path,
    ) -> None:
        self._provider = provider
        self._model = model
        self._template = prompt_path.read_text(encoding="utf-8")

    def extract(self, streamer: Streamer, chunk: TranscriptChunk) -> list[KnowledgeObservation]:
        prompt = self._template.format(
            streamer_name=streamer.display_name,
            stream_title=chunk.stream.title,
            transcript_chunk=chunk.rendered,
        )
        raw = self._provider.generate(prompt, model=self._model)
        return self._parse(raw, chunk)

    @staticmethod
    def _parse(raw: str, chunk: TranscriptChunk) -> list[KnowledgeObservation]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("LLM response is not valid JSON") from error
        items = payload.get("observations") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError("LLM response must contain an observations list")
        segment_map = {segment.index: segment for segment in chunk.segments}
        observations: list[KnowledgeObservation] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("each knowledge observation must be an object")
            indexes = item.get("evidence_segment_indexes")
            if not isinstance(indexes, list) or not indexes:
                raise ValueError("every knowledge observation requires evidence")
            evidence: list[Evidence] = []
            for raw_index in indexes:
                if not isinstance(raw_index, int) or raw_index not in segment_map:
                    raise ValueError("knowledge evidence references an unknown segment")
                segment = segment_map[raw_index]
                evidence.append(
                    Evidence(
                        transcript_id=chunk.stream.transcript_id,
                        segment_index=segment.index,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        quote=segment.text,
                    )
                )
            statement = item.get("statement")
            confidence = item.get("confidence")
            if not isinstance(statement, str) or not statement.strip():
                raise ValueError("knowledge statement must not be empty")
            if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
                raise ValueError("knowledge confidence must be between 0 and 1")
            origin = ObservationOrigin(item.get("origin"))
            normalized_confidence = (
                min(float(confidence), 0.7)
                if origin is ObservationOrigin.INFERRED
                else float(confidence)
            )
            observations.append(
                KnowledgeObservation(
                    category=KnowledgeCategory(item.get("category")),
                    statement=statement.strip(),
                    origin=origin,
                    confidence=normalized_confidence,
                    evidence=tuple(evidence),
                )
            )
        return observations


def merge_observations(
    observations: Iterable[KnowledgeObservation],
) -> list[KnowledgeObservation]:
    merged: dict[tuple[KnowledgeCategory, str], KnowledgeObservation] = {}
    for observation in observations:
        key = (observation.category, " ".join(observation.statement.lower().split()))
        previous = merged.get(key)
        if previous is None:
            merged[key] = observation
            continue
        evidence = tuple(
            {
                (item.transcript_id, item.segment_index): item
                for item in [*previous.evidence, *observation.evidence]
            }.values()
        )
        merged[key] = KnowledgeObservation(
            category=previous.category,
            statement=previous.statement,
            origin=(
                ObservationOrigin.OBSERVED
                if ObservationOrigin.OBSERVED in {previous.origin, observation.origin}
                else ObservationOrigin.INFERRED
            ),
            confidence=max(previous.confidence, observation.confidence),
            evidence=evidence,
        )
    return sorted(merged.values(), key=lambda item: (item.category.value, item.statement))


def keep_evidence_supported(
    observations: Iterable[KnowledgeObservation],
) -> list[KnowledgeObservation]:
    recurring_categories = {
        KnowledgeCategory.RECURRING_PHRASE,
        KnowledgeCategory.RECURRING_JOKE,
        KnowledgeCategory.SPEECH_PATTERN,
        KnowledgeCategory.COLLABORATION_PATTERN,
        KnowledgeCategory.CALLBACK,
    }
    supported: list[KnowledgeObservation] = []
    for observation in observations:
        distinct_evidence = {
            (item.transcript_id, item.segment_index) for item in observation.evidence
        }
        if observation.category in recurring_categories and len(distinct_evidence) < 2:
            continue
        supported.append(observation)
    return supported
