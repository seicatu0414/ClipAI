import logging
from collections.abc import Callable
from pathlib import Path

from clipai.events.domain import JsonValue
from clipai.knowledge.domain import KnowledgeJob
from clipai.knowledge.extraction import (
    KnowledgeExtractor,
    chunk_segments,
    keep_evidence_supported,
    merge_observations,
)
from clipai.knowledge.provider import LlmProvider
from clipai.knowledge.repository import KnowledgeRepository
from clipai.knowledge.selection import select_historical_streams

LOGGER = logging.getLogger(__name__)


class KnowledgePipeline:
    def __init__(
        self,
        repository: KnowledgeRepository,
        provider_factory: Callable[[KnowledgeJob], LlmProvider],
        prompt_root: Path,
    ) -> None:
        self._repository = repository
        self._provider_factory = provider_factory
        self._prompt_root = prompt_root

    def process(self, job: KnowledgeJob) -> None:
        try:
            streamer = self._repository.get_streamer(job.streamer_id)
            if streamer is None:
                raise ValueError("streamer does not exist")
            configuration = job.configuration
            streams = select_historical_streams(
                self._repository.list_streams(job.streamer_id),
                max_recent_hours=_number(configuration, "max_historical_hours"),
                max_representative_streams=_integer(
                    configuration, "max_representative_streams"
                ),
            )
            if not streams:
                raise ValueError("streamer has no registered transcript history")
            self._repository.update_progress(job.id, 10)

            extractor = KnowledgeExtractor(
                self._provider_factory(job),
                model=job.model,
                prompt_path=self._prompt_root / "streamer_knowledge" / f"{job.prompt_version}.md",
            )
            observations = []
            for stream_index, stream in enumerate(streams):
                segments = self._repository.iter_segments(stream.transcript_id)
                for chunk in chunk_segments(
                    stream,
                    segments,
                    maximum_characters=_integer(configuration, "chunk_characters"),
                ):
                    observations.extend(extractor.extract(streamer, chunk))
                progress = 10 + round(75 * (stream_index + 1) / len(streams))
                self._repository.update_progress(job.id, progress)

            merged = keep_evidence_supported(merge_observations(observations))
            version_id = self._repository.save_version(job, merged)
            LOGGER.info(
                "streamer_knowledge_completed",
                extra={
                    "job_id": str(job.id),
                    "knowledge_version_id": str(version_id),
                    "observation_count": len(merged),
                },
            )
        except Exception as error:
            LOGGER.exception("streamer_knowledge_failed", extra={"job_id": str(job.id)})
            self._repository.mark_failed(job.id, str(error) or error.__class__.__name__)


def _number(configuration: dict[str, JsonValue], key: str) -> float:
    value = configuration.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"knowledge configuration {key!r} must be numeric")
    return float(value)


def _integer(configuration: dict[str, JsonValue], key: str) -> int:
    value = configuration.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"knowledge configuration {key!r} must be an integer")
    return value
