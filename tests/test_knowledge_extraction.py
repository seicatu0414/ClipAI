import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from clipai.domain import TranscriptSegment
from clipai.knowledge.domain import (
    HistoricalStream,
    KnowledgeCategory,
    ObservationOrigin,
    Streamer,
)
from clipai.knowledge.extraction import (
    KnowledgeExtractor,
    chunk_segments,
    keep_evidence_supported,
    merge_observations,
)


class FakeProvider:
    name = "fake"

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, model: str) -> str:
        assert model == "test-model"
        self.prompts.append(prompt)
        return json.dumps(self.response, ensure_ascii=False)


def _stream() -> HistoricalStream:
    return HistoricalStream(
        uuid4(), uuid4(), uuid4(), "配信", datetime.now(UTC), 3600, 0, 0, False
    )


def test_chunks_never_exceed_configured_character_limit() -> None:
    segments = [TranscriptSegment(index, index, index + 1, "あ" * 20) for index in range(5)]

    chunks = list(chunk_segments(_stream(), segments, maximum_characters=50))

    assert len(chunks) > 1
    assert all(len(chunk.rendered) <= 50 for chunk in chunks)


def test_extracts_evidence_backed_observation(tmp_path: Path) -> None:
    prompt = tmp_path / "v1.md"
    prompt.write_text("{streamer_name}\n{stream_title}\n{transcript_chunk}", encoding="utf-8")
    provider = FakeProvider(
        {
            "observations": [
                {
                    "category": "recurring_phrase",
                    "statement": "よく『やった』と言う",
                    "origin": "observed",
                    "confidence": 0.8,
                    "evidence_segment_indexes": [3],
                }
            ]
        }
    )
    chunk = next(
        chunk_segments(
            _stream(), [TranscriptSegment(3, 1, 2, "やった")], maximum_characters=100
        )
    )

    observations = KnowledgeExtractor(
        provider, model="test-model", prompt_path=prompt
    ).extract(Streamer(uuid4(), "https://youtube.com/@x", "配信者"), chunk)

    assert observations[0].category is KnowledgeCategory.RECURRING_PHRASE
    assert observations[0].origin is ObservationOrigin.OBSERVED
    assert observations[0].evidence[0].segment_index == 3


def test_rejects_observation_without_evidence(tmp_path: Path) -> None:
    prompt = tmp_path / "v1.md"
    prompt.write_text("{transcript_chunk}", encoding="utf-8")
    provider = FakeProvider(
        {
            "observations": [
                {
                    "category": "callback",
                    "statement": "定番",
                    "origin": "inferred",
                    "confidence": 0.3,
                    "evidence_segment_indexes": [],
                }
            ]
        }
    )
    chunk = next(
        chunk_segments(
            _stream(), [TranscriptSegment(1, 0, 1, "本文")], maximum_characters=100
        )
    )

    with pytest.raises(ValueError, match="requires evidence"):
        KnowledgeExtractor(provider, model="test-model", prompt_path=prompt).extract(
            Streamer(uuid4(), "https://youtube.com/@x", "配信者"), chunk
        )


def test_merges_duplicate_observations_and_evidence(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "v1.md"
    prompt.write_text("{transcript_chunk}", encoding="utf-8")
    provider = FakeProvider({"observations": []})
    extractor = KnowledgeExtractor(provider, model="test-model", prompt_path=prompt)
    stream = _stream()
    first = extractor._parse(
        json.dumps(
            {
                "observations": [
                    {
                        "category": "callback",
                        "statement": "定番ネタがある",
                        "origin": "inferred",
                        "confidence": 0.5,
                        "evidence_segment_indexes": [1],
                    }
                ]
            }
        ),
        next(chunk_segments(stream, [TranscriptSegment(1, 0, 1, "A")], maximum_characters=50)),
    )[0]
    second = extractor._parse(
        json.dumps(
            {
                "observations": [
                    {
                        "category": "callback",
                        "statement": "定番ネタがある",
                        "origin": "observed",
                        "confidence": 0.8,
                        "evidence_segment_indexes": [2],
                    }
                ]
            }
        ),
        next(chunk_segments(stream, [TranscriptSegment(2, 2, 3, "B")], maximum_characters=50)),
    )[0]

    merged = merge_observations([first, second])

    assert len(merged) == 1
    assert merged[0].origin is ObservationOrigin.OBSERVED
    assert merged[0].confidence == 0.8
    assert len(merged[0].evidence) == 2


def test_recurring_claim_requires_two_distinct_evidence_segments(tmp_path: Path) -> None:
    prompt = tmp_path / "v1.md"
    prompt.write_text("{transcript_chunk}", encoding="utf-8")
    extractor = KnowledgeExtractor(
        FakeProvider({"observations": []}), model="test-model", prompt_path=prompt
    )
    stream = _stream()
    observation = extractor._parse(
        json.dumps(
            {
                "observations": [
                    {
                        "category": "recurring_joke",
                        "statement": "定番の冗談",
                        "origin": "inferred",
                        "confidence": 0.6,
                        "evidence_segment_indexes": [1],
                    }
                ]
            }
        ),
        next(chunk_segments(stream, [TranscriptSegment(1, 0, 1, "冗談")], maximum_characters=50)),
    )[0]

    assert keep_evidence_supported([observation]) == []
