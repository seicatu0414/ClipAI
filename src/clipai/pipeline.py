import logging
from collections.abc import Callable
from pathlib import Path

from clipai.domain import TranscriptionJob
from clipai.media import (
    AudioExtractor,
    MediaAcquirer,
    prepare_work_directory,
    remove_work_directory,
)
from clipai.repository import TranscriptionRepository
from clipai.transcription import Transcriber

LOGGER = logging.getLogger(__name__)


class TranscriptionPipeline:
    def __init__(
        self,
        repository: TranscriptionRepository,
        media_acquirer: MediaAcquirer,
        audio_extractor: AudioExtractor,
        transcriber_factory: Callable[[TranscriptionJob], Transcriber],
        media_root: Path,
    ) -> None:
        self._repository = repository
        self._media_acquirer = media_acquirer
        self._audio_extractor = audio_extractor
        self._transcriber_factory = transcriber_factory
        self._media_root = media_root

    def process(self, job: TranscriptionJob) -> None:
        work_directory = prepare_work_directory(self._media_root, job.id)
        try:
            self._repository.update_progress(job.id, 10)
            source_path = self._media_acquirer.acquire(job.source, work_directory)
            self._repository.update_progress(job.id, 30)

            audio_path = work_directory / "normalized.wav"
            self._audio_extractor.extract(source_path, audio_path)
            self._repository.update_progress(job.id, 45)

            transcriber = self._transcriber_factory(job)
            result = transcriber.transcribe(audio_path, language=job.language)
            self._repository.update_progress(job.id, 70)
            self._repository.save_transcript(job.id, result.metadata, result.segments)
            self._repository.mark_completed(job.id)
            LOGGER.info("transcription_completed", extra={"job_id": str(job.id)})
        except Exception as error:
            LOGGER.exception("transcription_failed", extra={"job_id": str(job.id)})
            self._repository.mark_failed(job.id, str(error) or error.__class__.__name__)
        finally:
            remove_work_directory(work_directory)
