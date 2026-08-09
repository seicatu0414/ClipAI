from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

from clipai.domain import TranscriptSegment
from clipai.events.detectors import JapaneseTranscriptRuleDetector
from clipai.events.domain import EventDetectionJob, EventJobStatus
from clipai.events.features import WaveAudioFeatureExtractor
from clipai.events.pipeline import EventDetectionPipeline


class FakeEventRepository:
    def __init__(self) -> None:
        self.progress: list[int] = []
        self.saved_count = 0
        self.completed = False
        self.failure: str | None = None

    def update_progress(self, _job_id: object, progress: int) -> None:
        self.progress.append(progress)

    def iter_segments(self, _transcript_id: object) -> Iterator[TranscriptSegment]:
        yield TranscriptSegment(0, 1, 2, "みんなコメントありがとう")

    def replace_events(self, _job: object, events: list[object]) -> None:
        self.saved_count = len(events)

    def mark_completed(self, _job_id: object) -> None:
        self.completed = True

    def mark_failed(self, _job_id: object, error: str) -> None:
        self.failure = error


class EmptyFeatureExtractor(WaveAudioFeatureExtractor):
    def extract(self, _audio_path: Path) -> Iterator[object]:  # type: ignore[override]
        return iter(())


class FailingFeatureExtractor(WaveAudioFeatureExtractor):
    def extract(self, _audio_path: Path) -> Iterator[object]:  # type: ignore[override]
        raise ValueError("bad audio")


def _job() -> EventDetectionJob:
    return EventDetectionJob(
        id=uuid4(),
        transcript_id=uuid4(),
        status=EventJobStatus.RUNNING,
        progress=1,
        detector_version="test-v1",
        configuration={"minimum_confidence": 0.5, "merge_gap_seconds": 1.0},
    )


def test_event_pipeline_persists_timeline() -> None:
    repository = FakeEventRepository()
    pipeline = EventDetectionPipeline(
        repository=repository,  # type: ignore[arg-type]
        feature_extractor=EmptyFeatureExtractor(),
        audio_detector_factory=lambda _configuration: [],
        transcript_detectors=[JapaneseTranscriptRuleDetector()],
    )

    pipeline.process(_job(), Path("audio.wav"))

    assert repository.progress == [15, 40, 60, 80]
    assert repository.saved_count == 1
    assert repository.completed is True
    assert repository.failure is None


def test_event_pipeline_records_feature_failure() -> None:
    repository = FakeEventRepository()
    pipeline = EventDetectionPipeline(
        repository=repository,  # type: ignore[arg-type]
        feature_extractor=FailingFeatureExtractor(),
        audio_detector_factory=lambda _configuration: [],
        transcript_detectors=[],
    )

    pipeline.process(_job(), Path("audio.wav"))

    assert repository.completed is False
    assert repository.failure == "bad audio"
