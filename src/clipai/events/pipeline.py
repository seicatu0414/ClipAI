import logging
from collections.abc import Callable
from pathlib import Path

from clipai.events.detectors import (
    AudioEventDetector,
    TranscriptEventDetector,
)
from clipai.events.domain import DetectedEvent, EventDetectionJob, JsonValue
from clipai.events.features import WaveAudioFeatureExtractor
from clipai.events.merge import merge_events
from clipai.events.repository import EventRepository

LOGGER = logging.getLogger(__name__)


class EventDetectionPipeline:
    def __init__(
        self,
        repository: EventRepository,
        feature_extractor: WaveAudioFeatureExtractor,
        audio_detector_factory: Callable[
            [dict[str, JsonValue]], list[AudioEventDetector]
        ],
        transcript_detectors: list[TranscriptEventDetector],
    ) -> None:
        self._repository = repository
        self._feature_extractor = feature_extractor
        self._audio_detector_factory = audio_detector_factory
        self._transcript_detectors = transcript_detectors

    def process(self, job: EventDetectionJob, audio_path: Path) -> None:
        try:
            self._repository.update_progress(job.id, 15)
            audio_features = list(self._feature_extractor.extract(audio_path))
            self._repository.update_progress(job.id, 40)

            raw_events: list[DetectedEvent] = []
            for detector in self._audio_detector_factory(job.configuration):
                raw_events.extend(detector.detect(audio_features))
            self._repository.update_progress(job.id, 60)

            for detector in self._transcript_detectors:
                segments = self._repository.iter_segments(job.transcript_id)
                raw_events.extend(detector.detect(segments))
            self._repository.update_progress(job.id, 80)

            configuration = job.configuration
            events = merge_events(
                raw_events,
                minimum_confidence=_number(configuration, "minimum_confidence"),
                merge_gap_seconds=_number(configuration, "merge_gap_seconds"),
            )
            self._repository.replace_events(job, events)
            self._repository.mark_completed(job.id)
            LOGGER.info(
                "event_detection_completed",
                extra={"job_id": str(job.id), "event_count": len(events)},
            )
        except Exception as error:
            LOGGER.exception("event_detection_failed", extra={"job_id": str(job.id)})
            self._repository.mark_failed(job.id, str(error) or error.__class__.__name__)


def _number(configuration: dict[str, JsonValue], key: str) -> float:
    value = configuration.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"event configuration {key!r} must be numeric")
    return float(value)
