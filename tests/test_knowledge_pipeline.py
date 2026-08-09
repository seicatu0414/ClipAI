from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from clipai.domain import TranscriptSegment
from clipai.knowledge.domain import (
    HistoricalStream,
    KnowledgeJob,
    KnowledgeJobStatus,
    Streamer,
)
from clipai.knowledge.pipeline import KnowledgePipeline


class FailingProvider:
    name = "failing"

    def generate(self, prompt: str, *, model: str) -> str:
        raise RuntimeError("provider offline")


class FakeKnowledgeRepository:
    def __init__(self, streamer: Streamer, stream: HistoricalStream) -> None:
        self.streamer = streamer
        self.stream = stream
        self.failure: str | None = None
        self.saved = False

    def get_streamer(self, _streamer_id: object) -> Streamer:
        return self.streamer

    def list_streams(self, _streamer_id: object) -> list[HistoricalStream]:
        return [self.stream]

    def iter_segments(self, _transcript_id: object) -> Iterator[TranscriptSegment]:
        yield TranscriptSegment(1, 0, 1, "本文")

    def update_progress(self, _job_id: object, _progress: int) -> None:
        return None

    def save_version(self, _job: object, _observations: object) -> object:
        self.saved = True
        return uuid4()

    def mark_failed(self, _job_id: object, error: str) -> None:
        self.failure = error


def test_provider_failure_marks_job_failed_without_saving_version(tmp_path: Path) -> None:
    prompt_directory = tmp_path / "streamer_knowledge"
    prompt_directory.mkdir()
    (prompt_directory / "v1.md").write_text("{transcript_chunk}", encoding="utf-8")
    streamer = Streamer(uuid4(), "https://youtube.com/@x", "配信者")
    stream = HistoricalStream(
        uuid4(), streamer.id, uuid4(), "配信", datetime.now(UTC), 3600, 0, 0, False
    )
    repository = FakeKnowledgeRepository(streamer, stream)
    pipeline = KnowledgePipeline(
        repository=repository,  # type: ignore[arg-type]
        provider_factory=lambda _job: FailingProvider(),
        prompt_root=tmp_path,
    )
    job = KnowledgeJob(
        uuid4(),
        streamer.id,
        KnowledgeJobStatus.RUNNING,
        1,
        "ollama",
        "model",
        "v1",
        {
            "max_historical_hours": 50.0,
            "max_representative_streams": 10,
            "chunk_characters": 1000,
        },
    )

    pipeline.process(job)

    assert repository.saved is False
    assert repository.failure == "provider offline"
