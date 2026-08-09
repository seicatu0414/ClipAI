import logging
import time
from functools import lru_cache
from pathlib import Path

from clipai.config import Settings, get_settings
from clipai.database import apply_migrations
from clipai.domain import TranscriptionJob
from clipai.events.configuration import event_configuration
from clipai.events.detectors import (
    JapaneseTranscriptRuleDetector,
    LoudReactionDetector,
    SilenceDetector,
)
from clipai.events.domain import EventDetectionJob, JsonValue
from clipai.events.features import WaveAudioFeatureExtractor
from clipai.events.pipeline import EventDetectionPipeline
from clipai.events.repository import EventRepository
from clipai.logging import configure_logging
from clipai.media import FfmpegAudioExtractor, SourceMediaAcquirer
from clipai.pipeline import TranscriptionPipeline
from clipai.repository import TranscriptionRepository
from clipai.transcription import FasterWhisperTranscriber

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _transcriber(
    model_size: str,
    device: str,
    cpu_fallback: bool,
) -> FasterWhisperTranscriber:
    return FasterWhisperTranscriber(
        model_size,
        device=device,
        cpu_fallback=cpu_fallback,
    )


def build_pipeline(
    settings: Settings,
    repository: TranscriptionRepository,
) -> TranscriptionPipeline:
    def factory(job: TranscriptionJob) -> FasterWhisperTranscriber:
        return _transcriber(job.model_size, settings.device, settings.cpu_fallback)

    return TranscriptionPipeline(
        repository=repository,
        media_acquirer=SourceMediaAcquirer(),
        audio_extractor=FfmpegAudioExtractor(),
        transcriber_factory=factory,
        media_root=settings.media_root,
    )


def build_event_pipeline(
    repository: EventRepository,
) -> EventDetectionPipeline:
    return EventDetectionPipeline(
        repository=repository,
        feature_extractor=WaveAudioFeatureExtractor(),
        audio_detector_factory=lambda configuration: [
            LoudReactionDetector(_number(configuration, "loudness_delta_db")),
            SilenceDetector(
                _number(configuration, "silence_db"),
                _number(configuration, "silence_minimum_seconds"),
            ),
        ],
        transcript_detectors=[JapaneseTranscriptRuleDetector()],
    )


def _number(configuration: dict[str, JsonValue], key: str) -> float:
    value = configuration.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"event configuration {key!r} must be numeric")
    return float(value)


def process_event_job(
    job: EventDetectionJob,
    transcription_repository: TranscriptionRepository,
    event_repository: EventRepository,
    pipeline: EventDetectionPipeline,
) -> None:
    source = transcription_repository.get_transcript_source(job.transcript_id)
    if source is None:
        event_repository.mark_failed(job.id, "transcript has no normalized audio artifact")
        return
    pipeline.process(job, Path(source.audio_artifact_path))


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    apply_migrations(settings.database_url)
    repository = TranscriptionRepository(settings.database_url)
    recovered = repository.recover_interrupted_jobs()
    if recovered:
        LOGGER.warning("interrupted_jobs_recovered", extra={"job_count": recovered})
    pipeline = build_pipeline(settings, repository)
    event_repository = EventRepository(settings.database_url)
    recovered_events = event_repository.recover_interrupted_jobs()
    if recovered_events:
        LOGGER.warning("interrupted_event_jobs_recovered", extra={"job_count": recovered_events})
    event_pipeline = build_event_pipeline(event_repository)
    LOGGER.info("worker_started")
    while True:
        job = repository.claim_next_job()
        if job is None:
            event_job = event_repository.claim_next_job()
            if event_job is None:
                time.sleep(settings.worker_poll_interval_seconds)
                continue
            LOGGER.info("event_detection_claimed", extra={"job_id": str(event_job.id)})
            process_event_job(event_job, repository, event_repository, event_pipeline)
        else:
            LOGGER.info("transcription_claimed", extra={"job_id": str(job.id)})
            pipeline.process(job)


if __name__ == "__main__":
    run()
