from collections.abc import Iterator
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from clipai.database import connect
from clipai.domain import TranscriptSegment
from clipai.events.domain import JsonValue
from clipai.knowledge.domain import (
    Evidence,
    HistoricalStream,
    KnowledgeCategory,
    KnowledgeJob,
    KnowledgeJobStatus,
    KnowledgeObservation,
    KnowledgeVersion,
    ObservationOrigin,
    Streamer,
)


class KnowledgeRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def create_streamer(
        self,
        *,
        channel_url: str,
        display_name: str,
        youtube_channel_id: str | None,
    ) -> Streamer:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO streamers (channel_url, display_name, youtube_channel_id)
                VALUES (%s, %s, %s)
                RETURNING id, channel_url, display_name, youtube_channel_id
                """,
                (channel_url, display_name, youtube_channel_id),
            )
            row = self._required_row(cursor.fetchone())
            connection.commit()
        return self._map_streamer(row)

    def get_streamer(self, streamer_id: UUID) -> Streamer | None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, channel_url, display_name, youtube_channel_id
                FROM streamers WHERE id = %s
                """,
                (streamer_id,),
            )
            row = cursor.fetchone()
        return None if row is None else self._map_streamer(row)

    def register_stream(
        self,
        *,
        streamer_id: UUID,
        transcript_id: UUID,
        title: str,
        published_at: datetime,
        duration_seconds: float,
        youtube_video_id: str | None = None,
        source_url: str | None = None,
        view_count: int = 0,
        comment_count: int = 0,
        manually_selected: bool = False,
    ) -> HistoricalStream:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO streams (
                    streamer_id, transcript_id, youtube_video_id, source_url, title,
                    published_at, duration_seconds, view_count, comment_count, manually_selected
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, streamer_id, transcript_id, title, published_at,
                          duration_seconds, view_count, comment_count, manually_selected
                """,
                (
                    streamer_id,
                    transcript_id,
                    youtube_video_id,
                    source_url,
                    title,
                    published_at,
                    duration_seconds,
                    view_count,
                    comment_count,
                    manually_selected,
                ),
            )
            row = self._required_row(cursor.fetchone())
            connection.commit()
        return self._map_stream(row)

    def list_streams(self, streamer_id: UUID) -> list[HistoricalStream]:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, streamer_id, transcript_id, title, published_at,
                       duration_seconds, view_count, comment_count, manually_selected
                FROM streams WHERE streamer_id = %s
                """,
                (streamer_id,),
            )
            rows = cursor.fetchall()
        return [self._map_stream(row) for row in rows]

    def iter_segments(
        self,
        transcript_id: UUID,
        batch_size: int = 500,
    ) -> Iterator[TranscriptSegment]:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT segment_index, start_seconds, end_seconds, text,
                       average_log_probability, no_speech_probability
                FROM transcript_segments WHERE transcript_id = %s ORDER BY segment_index
                """,
                (transcript_id,),
            )
            while rows := cursor.fetchmany(batch_size):
                for row in rows:
                    yield TranscriptSegment(
                        index=row[0],
                        start_seconds=row[1],
                        end_seconds=row[2],
                        text=row[3],
                        average_log_probability=row[4],
                        no_speech_probability=row[5],
                    )

    def create_job(
        self,
        streamer_id: UUID,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        configuration: dict[str, JsonValue],
    ) -> KnowledgeJob:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO knowledge_jobs (
                    streamer_id, provider, model, prompt_version, configuration
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id, streamer_id, status, progress, provider, model,
                          prompt_version, configuration, error
                """,
                (streamer_id, provider, model, prompt_version, Jsonb(configuration)),
            )
            row = self._required_row(cursor.fetchone())
            connection.commit()
        return self._map_job(row)

    def get_job(self, job_id: UUID) -> KnowledgeJob | None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, streamer_id, status, progress, provider, model,
                       prompt_version, configuration, error
                FROM knowledge_jobs WHERE id = %s
                """,
                (job_id,),
            )
            row = cursor.fetchone()
        return None if row is None else self._map_job(row)

    def claim_next_job(self) -> KnowledgeJob | None:
        query = """
            WITH next_job AS (
                SELECT id FROM knowledge_jobs WHERE status = 'pending'
                ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
            )
            UPDATE knowledge_jobs j
            SET status = 'running', progress = 1, error = NULL,
                started_at = now(), updated_at = now()
            FROM next_job WHERE j.id = next_job.id
            RETURNING j.id, j.streamer_id, j.status, j.progress, j.provider, j.model,
                      j.prompt_version, j.configuration, j.error
        """
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
            connection.commit()
        return None if row is None else self._map_job(row)

    def recover_interrupted_jobs(self) -> int:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM streamer_knowledge_versions
                WHERE knowledge_job_id IN (
                    SELECT id FROM knowledge_jobs WHERE status = 'running'
                )
                """
            )
            cursor.execute(
                """
                UPDATE knowledge_jobs
                SET status = 'pending', progress = 0,
                    error = 'worker interrupted; job queued again',
                    started_at = NULL, updated_at = now()
                WHERE status = 'running'
                """
            )
            recovered = cursor.rowcount
            connection.commit()
        return recovered

    def update_progress(self, job_id: UUID, progress: int) -> None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE knowledge_jobs SET progress = %s, updated_at = now() WHERE id = %s",
                (max(1, min(progress, 99)), job_id),
            )
            connection.commit()

    def save_version(
        self,
        job: KnowledgeJob,
        observations: list[KnowledgeObservation],
    ) -> UUID:
        if any(not observation.evidence for observation in observations):
            raise ValueError("all knowledge observations require evidence")
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"knowledge:{job.streamer_id}",),
            )
            cursor.execute(
                """
                SELECT id, version_number FROM streamer_knowledge_versions
                WHERE streamer_id = %s ORDER BY version_number DESC LIMIT 1
                """,
                (job.streamer_id,),
            )
            previous = cursor.fetchone()
            previous_id = None if previous is None else previous[0]
            version_number = next_version_number(
                None if previous is None else previous[1]
            )
            cursor.execute(
                """
                INSERT INTO streamer_knowledge_versions (
                    streamer_id, knowledge_job_id, version_number, previous_version_id,
                    provider, model, prompt_version, configuration
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    job.streamer_id,
                    job.id,
                    version_number,
                    previous_id,
                    job.provider,
                    job.model,
                    job.prompt_version,
                    Jsonb(job.configuration),
                ),
            )
            version_id = self._required_row(cursor.fetchone())[0]
            for observation in observations:
                cursor.execute(
                    """
                    INSERT INTO knowledge_observations (
                        knowledge_version_id, category, statement, origin, confidence
                    ) VALUES (%s, %s, %s, %s, %s) RETURNING id
                    """,
                    (
                        version_id,
                        observation.category.value,
                        observation.statement,
                        observation.origin.value,
                        observation.confidence,
                    ),
                )
                observation_id = self._required_row(cursor.fetchone())[0]
                cursor.executemany(
                    """
                    INSERT INTO knowledge_evidence (
                        observation_id, transcript_id, segment_index,
                        start_seconds, end_seconds, quote
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            observation_id,
                            item.transcript_id,
                            item.segment_index,
                            item.start_seconds,
                            item.end_seconds,
                            item.quote,
                        )
                        for item in observation.evidence
                    ],
                )
            cursor.execute(
                """
                UPDATE knowledge_jobs
                SET status = 'completed', progress = 100, error = NULL,
                    completed_at = now(), updated_at = now()
                WHERE id = %s
                """,
                (job.id,),
            )
            connection.commit()
        return version_id

    def get_current_version(self, streamer_id: UUID) -> KnowledgeVersion | None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, streamer_id, version_number, previous_version_id,
                       provider, model, prompt_version, configuration
                FROM streamer_knowledge_versions
                WHERE streamer_id = %s ORDER BY version_number DESC LIMIT 1
                """,
                (streamer_id,),
            )
            version = cursor.fetchone()
            if version is None:
                return None
            cursor.execute(
                """
                SELECT id, category, statement, origin, confidence
                FROM knowledge_observations
                WHERE knowledge_version_id = %s ORDER BY category, statement
                """,
                (version[0],),
            )
            observation_rows = cursor.fetchall()
            observations: list[KnowledgeObservation] = []
            for row in observation_rows:
                cursor.execute(
                    """
                    SELECT transcript_id, segment_index, start_seconds, end_seconds, quote
                    FROM knowledge_evidence
                    WHERE observation_id = %s
                    ORDER BY transcript_id, segment_index
                    """,
                    (row[0],),
                )
                evidence = tuple(Evidence(*item) for item in cursor.fetchall())
                observations.append(
                    KnowledgeObservation(
                        KnowledgeCategory(row[1]),
                        row[2],
                        ObservationOrigin(row[3]),
                        row[4],
                        evidence,
                    )
                )
        return KnowledgeVersion(
            id=version[0],
            streamer_id=version[1],
            version_number=version[2],
            previous_version_id=version[3],
            provider=version[4],
            model=version[5],
            prompt_version=version[6],
            configuration=version[7],
            observations=tuple(observations),
        )

    def mark_failed(self, job_id: UUID, error: str) -> None:
        self._finish(job_id, KnowledgeJobStatus.FAILED, error[:2000])

    def _finish(self, job_id: UUID, status: KnowledgeJobStatus, error: str | None) -> None:
        progress = 100 if status is KnowledgeJobStatus.COMPLETED else 99
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE knowledge_jobs SET status = %s, progress = %s, error = %s,
                    completed_at = now(), updated_at = now() WHERE id = %s
                """,
                (status.value, progress, error, job_id),
            )
            connection.commit()

    @staticmethod
    def _required_row(row: tuple[Any, ...] | None) -> tuple[Any, ...]:
        if row is None:
            raise RuntimeError("database did not return the expected row")
        return row

    @staticmethod
    def _map_streamer(row: tuple[Any, ...]) -> Streamer:
        return Streamer(row[0], row[1], row[2], row[3])

    @staticmethod
    def _map_stream(row: tuple[Any, ...]) -> HistoricalStream:
        return HistoricalStream(*row)

    @staticmethod
    def _map_job(row: tuple[Any, ...]) -> KnowledgeJob:
        return KnowledgeJob(
            id=row[0],
            streamer_id=row[1],
            status=KnowledgeJobStatus(row[2]),
            progress=row[3],
            provider=row[4],
            model=row[5],
            prompt_version=row[6],
            configuration=row[7],
            error=row[8],
        )


def next_version_number(previous_version_number: int | None) -> int:
    return 1 if previous_version_number is None else previous_version_number + 1
