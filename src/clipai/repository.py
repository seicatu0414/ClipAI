from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from clipai.database import connect
from clipai.domain import (
    JobStatus,
    SourceKind,
    SourceSpec,
    TranscriptMetadata,
    TranscriptSegment,
    TranscriptionJob,
)


@dataclass(frozen=True)
class JobSnapshot:
    job: TranscriptionJob
    transcript_id: UUID | None


class TranscriptionRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def create_job(
        self,
        source: SourceSpec,
        *,
        model_size: str,
        language: str,
    ) -> TranscriptionJob:
        query = """
            INSERT INTO transcription_jobs (source_kind, source, model_size, language)
            VALUES (%s, %s, %s, %s)
            RETURNING id, source_kind, source, status, progress, model_size, language, error
        """
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(query, (source.kind.value, source.value, model_size, language))
            row = cursor.fetchone()
            connection.commit()
        return self._map_job(self._required_row(row))

    def get_job(self, job_id: UUID) -> JobSnapshot | None:
        query = """
            SELECT j.id, j.source_kind, j.source, j.status, j.progress,
                   j.model_size, j.language, j.error, t.id
            FROM transcription_jobs j
            LEFT JOIN transcripts t ON t.job_id = j.id
            WHERE j.id = %s
        """
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(query, (job_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return JobSnapshot(job=self._map_job(row[:8]), transcript_id=row[8])

    def list_segments(
        self,
        transcript_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[TranscriptSegment]:
        query = """
            SELECT segment_index, start_seconds, end_seconds, text,
                   average_log_probability, no_speech_probability
            FROM transcript_segments
            WHERE transcript_id = %s
            ORDER BY segment_index
            OFFSET %s LIMIT %s
        """
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(query, (transcript_id, offset, limit))
            rows = cursor.fetchall()
        return [
            TranscriptSegment(
                index=row[0],
                start_seconds=row[1],
                end_seconds=row[2],
                text=row[3],
                average_log_probability=row[4],
                no_speech_probability=row[5],
            )
            for row in rows
        ]

    def claim_next_job(self) -> TranscriptionJob | None:
        query = """
            WITH next_job AS (
                SELECT id FROM transcription_jobs
                WHERE status = 'pending'
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE transcription_jobs j
            SET status = 'running', progress = 1, error = NULL,
                started_at = now(), updated_at = now()
            FROM next_job
            WHERE j.id = next_job.id
            RETURNING j.id, j.source_kind, j.source, j.status, j.progress,
                      j.model_size, j.language, j.error
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
                DELETE FROM transcripts
                WHERE job_id IN (SELECT id FROM transcription_jobs WHERE status = 'running')
                """
            )
            cursor.execute(
                """
                UPDATE transcription_jobs
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
        bounded = max(1, min(progress, 99))
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE transcription_jobs SET progress = %s, updated_at = now() WHERE id = %s",
                (bounded, job_id),
            )
            connection.commit()

    def save_transcript(
        self,
        job_id: UUID,
        metadata: TranscriptMetadata,
        segments: Iterable[TranscriptSegment],
        *,
        batch_size: int = 100,
    ) -> UUID:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO transcripts (
                    job_id, source_duration_seconds, detected_language,
                    detected_language_probability, model_size, device
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    job_id,
                    metadata.duration_seconds,
                    metadata.detected_language,
                    metadata.language_probability,
                    metadata.model_size,
                    metadata.device,
                ),
            )
            transcript_id = self._required_row(cursor.fetchone())[0]
            connection.commit()
            pending: list[tuple[Any, ...]] = []
            for segment in segments:
                pending.append(
                    (
                        transcript_id,
                        segment.index,
                        segment.start_seconds,
                        segment.end_seconds,
                        segment.text,
                        segment.average_log_probability,
                        segment.no_speech_probability,
                    )
                )
                if len(pending) >= batch_size:
                    cursor.executemany(self._segment_insert_sql(), pending)
                    connection.commit()
                    pending.clear()
            if pending:
                cursor.executemany(self._segment_insert_sql(), pending)
                connection.commit()
        return transcript_id

    def mark_completed(self, job_id: UUID) -> None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE transcription_jobs
                SET status = 'completed', progress = 100, completed_at = now(), updated_at = now()
                WHERE id = %s
                """,
                (job_id,),
            )
            connection.commit()

    def mark_failed(self, job_id: UUID, error: str) -> None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE transcription_jobs
                SET status = 'failed', error = %s, completed_at = now(), updated_at = now()
                WHERE id = %s
                """,
                (error[:2000], job_id),
            )
            connection.commit()

    @staticmethod
    def _segment_insert_sql() -> str:
        return """
            INSERT INTO transcript_segments (
                transcript_id, segment_index, start_seconds, end_seconds, text,
                average_log_probability, no_speech_probability
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

    @staticmethod
    def _required_row(row: tuple[Any, ...] | None) -> tuple[Any, ...]:
        if row is None:
            raise RuntimeError("database did not return the expected row")
        return row

    @staticmethod
    def _map_job(row: tuple[Any, ...]) -> TranscriptionJob:
        return TranscriptionJob(
            id=row[0],
            source=SourceSpec(SourceKind(row[1]), row[2]),
            status=JobStatus(row[3]),
            progress=row[4],
            model_size=row[5],
            language=row[6],
            error=row[7],
        )
