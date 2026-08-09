from collections.abc import Iterator
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from clipai.database import connect
from clipai.domain import TranscriptSegment
from clipai.events.domain import (
    DetectedEvent,
    EventDetectionJob,
    EventJobStatus,
    EventType,
    JsonValue,
)


class EventRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def create_job(
        self,
        transcript_id: UUID,
        *,
        detector_version: str,
        configuration: dict[str, JsonValue],
    ) -> EventDetectionJob:
        query = """
            INSERT INTO event_detection_jobs (
                transcript_id, detector_version, configuration
            ) VALUES (%s, %s, %s)
            RETURNING id, transcript_id, status, progress,
                      detector_version, configuration, error
        """
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(query, (transcript_id, detector_version, Jsonb(configuration)))
            row = self._required_row(cursor.fetchone())
            connection.commit()
        return self._map_job(row)

    def get_job(self, job_id: UUID) -> EventDetectionJob | None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, transcript_id, status, progress,
                       detector_version, configuration, error
                FROM event_detection_jobs WHERE id = %s
                """,
                (job_id,),
            )
            row = cursor.fetchone()
        return None if row is None else self._map_job(row)

    def claim_next_job(self) -> EventDetectionJob | None:
        query = """
            WITH next_job AS (
                SELECT id FROM event_detection_jobs
                WHERE status = 'pending'
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE event_detection_jobs j
            SET status = 'running', progress = 1, error = NULL,
                started_at = now(), updated_at = now()
            FROM next_job
            WHERE j.id = next_job.id
            RETURNING j.id, j.transcript_id, j.status, j.progress,
                      j.detector_version, j.configuration, j.error
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
                DELETE FROM events
                WHERE event_detection_job_id IN (
                    SELECT id FROM event_detection_jobs WHERE status = 'running'
                )
                """
            )
            cursor.execute(
                """
                UPDATE event_detection_jobs
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
                "UPDATE event_detection_jobs SET progress = %s, updated_at = now() WHERE id = %s",
                (max(1, min(progress, 99)), job_id),
            )
            connection.commit()

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
                FROM transcript_segments
                WHERE transcript_id = %s
                ORDER BY segment_index
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

    def replace_events(self, job: EventDetectionJob, events: list[DetectedEvent]) -> None:
        query = """
            INSERT INTO events (
                event_detection_job_id, transcript_id, event_type,
                start_seconds, end_seconds, confidence, source_signals, explanation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        rows = [
            (
                job.id,
                job.transcript_id,
                event.event_type.value,
                event.start_seconds,
                event.end_seconds,
                event.confidence,
                Jsonb(event.source_signals),
                event.explanation,
            )
            for event in events
        ]
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM events WHERE event_detection_job_id = %s", (job.id,))
            if rows:
                cursor.executemany(query, rows)
            connection.commit()

    def list_events(
        self,
        transcript_id: UUID,
        event_detection_job_id: UUID | None = None,
    ) -> list[DetectedEvent]:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT event_type, start_seconds, end_seconds, confidence,
                       source_signals, explanation
                FROM events
                WHERE transcript_id = %s
                  AND event_detection_job_id = COALESCE(
                      %s,
                      (
                          SELECT id FROM event_detection_jobs
                          WHERE transcript_id = %s AND status = 'completed'
                          ORDER BY completed_at DESC
                          LIMIT 1
                      )
                  )
                ORDER BY start_seconds, end_seconds
                """,
                (transcript_id, event_detection_job_id, transcript_id),
            )
            rows = cursor.fetchall()
        return [
            DetectedEvent(
                event_type=EventType(row[0]),
                start_seconds=row[1],
                end_seconds=row[2],
                confidence=row[3],
                source_signals=row[4],
                explanation=row[5],
            )
            for row in rows
        ]

    def mark_completed(self, job_id: UUID) -> None:
        self._finish(job_id, EventJobStatus.COMPLETED, None)

    def mark_failed(self, job_id: UUID, error: str) -> None:
        self._finish(job_id, EventJobStatus.FAILED, error[:2000])

    def _finish(self, job_id: UUID, status: EventJobStatus, error: str | None) -> None:
        progress = 100 if status is EventJobStatus.COMPLETED else 99
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE event_detection_jobs
                SET status = %s, progress = %s, error = %s,
                    completed_at = now(), updated_at = now()
                WHERE id = %s
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
    def _map_job(row: tuple[Any, ...]) -> EventDetectionJob:
        return EventDetectionJob(
            id=row[0],
            transcript_id=row[1],
            status=EventJobStatus(row[2]),
            progress=row[3],
            detector_version=row[4],
            configuration=row[5],
            error=row[6],
        )
