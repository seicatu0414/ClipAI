from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from clipai.domain import (
    JobStatus,
    SourceKind,
    SourceSpec,
    TranscriptMetadata,
    TranscriptSegment,
    TranscriptionJob,
)
from clipai.pipeline import TranscriptionPipeline
from clipai.transcription import TranscriptionResult


class FakeRepository:
    def __init__(self) -> None:
        self.progress: list[int] = []
        self.saved_segments: list[TranscriptSegment] = []
        self.completed = False
        self.failure: str | None = None
        self.audio_artifact: str | None = None
        self.discarded = False

    def update_progress(self, _job_id: object, progress: int) -> None:
        self.progress.append(progress)

    def save_transcript(
        self,
        _job_id: object,
        _metadata: TranscriptMetadata,
        segments: Iterable[TranscriptSegment],
    ) -> object:
        self.saved_segments.extend(segments)
        return uuid4()

    def mark_completed(self, _job_id: object) -> None:
        self.completed = True

    def set_audio_artifact(self, _transcript_id: object, path: str) -> None:
        self.audio_artifact = path

    def mark_failed(self, _job_id: object, error: str) -> None:
        self.failure = error

    def discard_transcript(self, _job_id: object) -> None:
        self.discarded = True


class FakeAcquirer:
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path

    def acquire(self, _source: SourceSpec, _work_directory: Path) -> Path:
        return self.source_path


class FakeExtractor:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    def extract(self, _source: Path, destination: Path) -> None:
        if self.failure:
            raise self.failure
        destination.write_bytes(b"wav")


class FakeTranscriber:
    def transcribe(self, _audio_path: Path, *, language: str) -> TranscriptionResult:
        assert language == "ja"
        return TranscriptionResult(
            metadata=TranscriptMetadata(10.0, "ja", 0.99, "tiny", "cpu"),
            segments=iter([TranscriptSegment(0, 0.0, 1.0, "こんにちは")]),
        )


def _job() -> TranscriptionJob:
    return TranscriptionJob(
        id=uuid4(),
        source=SourceSpec(SourceKind.LOCAL_FILE, "stream.mp4"),
        status=JobStatus.RUNNING,
        progress=1,
        model_size="tiny",
        language="ja",
    )


def test_pipeline_orchestrates_and_persists_segments(tmp_path: Path) -> None:
    repository = FakeRepository()
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"video")
    pipeline = TranscriptionPipeline(
        repository=repository,  # type: ignore[arg-type]
        media_acquirer=FakeAcquirer(source),
        audio_extractor=FakeExtractor(),
        transcriber_factory=lambda _job: FakeTranscriber(),
        media_root=tmp_path,
    )

    pipeline.process(_job())

    assert repository.progress == [10, 30, 45, 70]
    assert repository.completed is True
    assert repository.failure is None
    assert [segment.text for segment in repository.saved_segments] == ["こんにちは"]
    assert repository.audio_artifact is not None
    assert Path(repository.audio_artifact).is_file()


def test_pipeline_marks_job_failed_and_cleans_work_directory(tmp_path: Path) -> None:
    repository = FakeRepository()
    job = _job()
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"video")
    pipeline = TranscriptionPipeline(
        repository=repository,  # type: ignore[arg-type]
        media_acquirer=FakeAcquirer(source),
        audio_extractor=FakeExtractor(RuntimeError("ffmpeg failed")),
        transcriber_factory=lambda _job: FakeTranscriber(),
        media_root=tmp_path,
    )

    pipeline.process(job)

    assert repository.completed is False
    assert repository.failure == "ffmpeg failed"
    assert repository.discarded is True
    assert not (tmp_path / "work" / str(job.id)).exists()
