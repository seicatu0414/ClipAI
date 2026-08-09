import logging
import time
from functools import lru_cache

from clipai.config import Settings, get_settings
from clipai.database import apply_migrations
from clipai.domain import TranscriptionJob
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


def build_pipeline(settings: Settings, repository: TranscriptionRepository) -> TranscriptionPipeline:
    def factory(job: TranscriptionJob) -> FasterWhisperTranscriber:
        return _transcriber(job.model_size, settings.device, settings.cpu_fallback)

    return TranscriptionPipeline(
        repository=repository,
        media_acquirer=SourceMediaAcquirer(),
        audio_extractor=FfmpegAudioExtractor(),
        transcriber_factory=factory,
        media_root=settings.media_root,
    )


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    apply_migrations(settings.database_url)
    repository = TranscriptionRepository(settings.database_url)
    recovered = repository.recover_interrupted_jobs()
    if recovered:
        LOGGER.warning("interrupted_jobs_recovered", extra={"job_count": recovered})
    pipeline = build_pipeline(settings, repository)
    LOGGER.info("worker_started")
    while True:
        job = repository.claim_next_job()
        if job is None:
            time.sleep(settings.worker_poll_interval_seconds)
            continue
        LOGGER.info("transcription_claimed", extra={"job_id": str(job.id)})
        pipeline.process(job)


if __name__ == "__main__":
    run()
