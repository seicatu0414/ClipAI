from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from clipai.candidates.domain import (
    CandidateCategory,
    CandidateEvent,
    CandidateJob,
    CandidateJobStatus,
    ClipCandidate,
)
from clipai.database import connect
from clipai.domain import TranscriptSegment
from clipai.events.domain import EventType, JsonValue
from clipai.knowledge.domain import (
    Evidence,
    KnowledgeCategory,
    KnowledgeObservation,
    ObservationOrigin,
)


class CandidateRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def create_job(
        self,
        streamer_id: UUID,
        transcript_id: UUID,
        *,
        pipeline_version: str,
        provider: str,
        model: str,
        prompt_version: str,
        configuration: dict[str, JsonValue],
        preference_version_id: UUID | None = None,
    ) -> CandidateJob:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1 FROM streams
                WHERE streamer_id = %s AND transcript_id = %s
                """,
                (streamer_id, transcript_id),
            )
            if cursor.fetchone() is None:
                raise ValueError("transcript is not registered to this streamer")
            if preference_version_id is not None:
                cursor.execute(
                    """
                    SELECT 1 FROM streamer_preference_versions
                    WHERE id = %s AND streamer_id = %s
                    """,
                    (preference_version_id, streamer_id),
                )
                if cursor.fetchone() is None:
                    raise ValueError("preference version does not belong to streamer")
            cursor.execute(
                """
                SELECT id FROM event_detection_jobs
                WHERE transcript_id = %s AND status = 'completed'
                ORDER BY completed_at DESC LIMIT 1
                """,
                (transcript_id,),
            )
            event_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT id FROM streamer_knowledge_versions
                WHERE streamer_id = %s ORDER BY version_number DESC LIMIT 1
                """,
                (streamer_id,),
            )
            knowledge_row = cursor.fetchone()
            if event_row is None:
                raise ValueError("transcript has no completed event detection")
            if knowledge_row is None:
                raise ValueError("streamer has no knowledge version")
            cursor.execute(
                """
                INSERT INTO candidate_jobs (
                    streamer_id, transcript_id, event_detection_job_id,
                    knowledge_version_id, pipeline_version, provider, model,
                    prompt_version, configuration, preference_version_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, streamer_id, transcript_id, event_detection_job_id,
                    knowledge_version_id, preference_version_id, status, progress, pipeline_version,
                    provider, model, prompt_version, configuration, error
                """,
                (
                    streamer_id,
                    transcript_id,
                    event_row[0],
                    knowledge_row[0],
                    pipeline_version,
                    provider,
                    model,
                    prompt_version,
                    Jsonb(configuration),
                    preference_version_id,
                ),
            )
            row = _required(cursor.fetchone())
            connection.commit()
        return _map_job(row)

    def get_job(self, job_id: UUID) -> CandidateJob | None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, streamer_id, transcript_id, event_detection_job_id,
                    knowledge_version_id, preference_version_id, status, progress, pipeline_version,
                    provider, model, prompt_version, configuration, error
                FROM candidate_jobs WHERE id = %s
                """,
                (job_id,),
            )
            row = cursor.fetchone()
        return None if row is None else _map_job(row)

    def claim_next_job(self) -> CandidateJob | None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                WITH next_job AS (
                    SELECT id FROM candidate_jobs WHERE status = 'pending'
                    ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE candidate_jobs j
                SET status = 'running', progress = 1, error = NULL,
                    started_at = now(), updated_at = now()
                FROM next_job WHERE j.id = next_job.id
                RETURNING j.id, j.streamer_id, j.transcript_id,
                    j.event_detection_job_id, j.knowledge_version_id,
                    j.preference_version_id, j.status,
                    j.progress, j.pipeline_version, j.provider, j.model,
                    j.prompt_version, j.configuration, j.error
                """
            )
            row = cursor.fetchone()
            connection.commit()
        return None if row is None else _map_job(row)

    def recover_interrupted_jobs(self) -> int:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM clip_candidates WHERE candidate_job_id IN (
                    SELECT id FROM candidate_jobs WHERE status = 'running'
                )
                """
            )
            cursor.execute(
                """
                UPDATE candidate_jobs SET status = 'pending', progress = 0,
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
                "UPDATE candidate_jobs SET progress = %s, updated_at = now() WHERE id = %s",
                (max(1, min(progress, 99)), job_id),
            )
            connection.commit()

    def load_events(self, job: CandidateJob) -> list[CandidateEvent]:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, event_type, start_seconds, end_seconds, confidence,
                    source_signals, explanation
                FROM events WHERE event_detection_job_id = %s
                ORDER BY start_seconds, end_seconds, id
                """,
                (job.event_detection_job_id,),
            )
            rows = cursor.fetchall()
        return [
            CandidateEvent(row[0], EventType(row[1]), *row[2:])
            for row in rows
        ]

    def load_knowledge(
        self, job: CandidateJob
    ) -> list[tuple[UUID, KnowledgeObservation]]:
        result: list[tuple[UUID, KnowledgeObservation]] = []
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, category, statement, origin, confidence
                FROM knowledge_observations WHERE knowledge_version_id = %s
                ORDER BY category, statement, id
                """,
                (job.knowledge_version_id,),
            )
            for row in cursor.fetchall():
                cursor.execute(
                    """
                    SELECT transcript_id, segment_index, start_seconds,
                        end_seconds, quote
                    FROM knowledge_evidence WHERE observation_id = %s
                    ORDER BY transcript_id, segment_index
                    """,
                    (row[0],),
                )
                evidence = tuple(Evidence(*item) for item in cursor.fetchall())
                result.append(
                    (
                        row[0],
                        KnowledgeObservation(
                            KnowledgeCategory(row[1]),
                            row[2],
                            ObservationOrigin(row[3]),
                            row[4],
                            evidence,
                        ),
                    )
                )
        return result

    def load_preference_weights(
        self,
        preference_version_id: UUID | None,
    ) -> dict[CandidateCategory, float]:
        if preference_version_id is None:
            return {category: 1.0 for category in CandidateCategory}
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT category_weights FROM streamer_preference_versions WHERE id = %s
                """,
                (preference_version_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise ValueError("pinned preference version does not exist")
        return {
            CandidateCategory(key): float(value)
            for key, value in row[0].items()
        }

    def segments_for(
        self,
        transcript_id: UUID,
        start: float,
        end: float,
    ) -> list[TranscriptSegment]:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT segment_index, start_seconds, end_seconds, text,
                    average_log_probability, no_speech_probability
                FROM transcript_segments
                WHERE transcript_id = %s AND end_seconds >= %s AND start_seconds <= %s
                ORDER BY segment_index
                """,
                (transcript_id, start, end),
            )
            rows = cursor.fetchall()
        return [TranscriptSegment(*row) for row in rows]

    def save_candidates(
        self,
        job: CandidateJob,
        candidates: list[ClipCandidate],
    ) -> None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM clip_candidates WHERE candidate_job_id = %s", (job.id,))
            for item in candidates:
                cursor.execute(
                    """
                    INSERT INTO clip_candidates (
                        candidate_job_id, transcript_id, rank, start_seconds,
                        end_seconds, category_scores, overall_score, confidence,
                        reasons, event_ids, knowledge_observation_ids
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job.id,
                        job.transcript_id,
                        item.rank,
                        item.start_seconds,
                        item.end_seconds,
                        Jsonb({key.value: value for key, value in item.category_scores.items()}),
                        item.overall_score,
                        item.confidence,
                        Jsonb(list(item.reasons)),
                        Jsonb([str(value) for value in item.event_ids]),
                        Jsonb([str(value) for value in item.knowledge_observation_ids]),
                    ),
                )
            cursor.execute(
                """
                UPDATE candidate_jobs SET status = 'completed', progress = 100,
                    error = NULL, completed_at = now(), updated_at = now()
                WHERE id = %s
                """,
                (job.id,),
            )
            connection.commit()

    def list_candidates(self, job_id: UUID) -> list[ClipCandidate]:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, rank, start_seconds, end_seconds, category_scores,
                    overall_score, confidence, reasons, event_ids,
                    knowledge_observation_ids
                FROM clip_candidates WHERE candidate_job_id = %s ORDER BY rank
                """,
                (job_id,),
            )
            rows = cursor.fetchall()
        return [
            ClipCandidate(
                row[0],
                row[1],
                row[2],
                row[3],
                {CandidateCategory(key): value for key, value in row[4].items()},
                row[5], row[6], tuple(row[7]),
                tuple(UUID(value) for value in row[8]),
                tuple(UUID(value) for value in row[9]),
                (),
            )
            for row in rows
        ]

    def mark_failed(self, job_id: UUID, error: str) -> None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE candidate_jobs SET status = 'failed', progress = 99,
                    error = %s, completed_at = now(), updated_at = now()
                WHERE id = %s
                """,
                (error[:2000], job_id),
            )
            connection.commit()


def _required(row: tuple[Any, ...] | None) -> tuple[Any, ...]:
    if row is None:
        raise RuntimeError("database did not return the expected row")
    return row


def _map_job(row: tuple[Any, ...]) -> CandidateJob:
    return CandidateJob(
        row[0], row[1], row[2], row[3], row[4], row[5],
        CandidateJobStatus(row[6]), row[7], row[8], row[9],
        row[10], row[11], row[12], row[13],
    )
