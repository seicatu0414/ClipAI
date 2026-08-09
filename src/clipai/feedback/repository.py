from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from clipai.candidates.domain import CandidateCategory
from clipai.database import connect
from clipai.feedback.domain import (
    CandidateFeedback,
    EvaluationCandidate,
    FeedbackRating,
    FeedbackReasonTag,
    PreferenceVersion,
)
from clipai.feedback.learning import (
    default_weights,
    next_version_number,
    rollback_weights,
    update_weights,
)


class FeedbackRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def ensure_current_preference(self, streamer_id: UUID) -> PreferenceVersion:
        current = self.get_current_preference(streamer_id)
        if current is not None:
            return current
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"preference:{streamer_id}",),
            )
            cursor.execute(
                """
                SELECT id, streamer_id, version_number, previous_version_id,
                    source_feedback_id, rollback_of_version_id, category_weights,
                    explanation, created_at
                FROM streamer_preference_versions
                WHERE streamer_id = %s ORDER BY version_number DESC LIMIT 1
                """,
                (streamer_id,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    """
                    INSERT INTO streamer_preference_versions (
                        streamer_id, version_number, category_weights, explanation
                    ) VALUES (%s, 1, %s, %s)
                    RETURNING id, streamer_id, version_number, previous_version_id,
                        source_feedback_id, rollback_of_version_id, category_weights,
                        explanation, created_at
                    """,
                    (
                        streamer_id,
                        Jsonb(_weights_json(default_weights())),
                        Jsonb([
                            "strategy=bounded-category-weights-v1",
                            "initial balanced preferences",
                        ]),
                    ),
                )
                row = _required(cursor.fetchone())
            connection.commit()
        return _map_preference(row)

    def get_current_preference(self, streamer_id: UUID) -> PreferenceVersion | None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, streamer_id, version_number, previous_version_id,
                    source_feedback_id, rollback_of_version_id, category_weights,
                    explanation, created_at
                FROM streamer_preference_versions
                WHERE streamer_id = %s ORDER BY version_number DESC LIMIT 1
                """,
                (streamer_id,),
            )
            row = cursor.fetchone()
        return None if row is None else _map_preference(row)

    def get_preference(self, version_id: UUID) -> PreferenceVersion | None:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, streamer_id, version_number, previous_version_id,
                    source_feedback_id, rollback_of_version_id, category_weights,
                    explanation, created_at
                FROM streamer_preference_versions WHERE id = %s
                """,
                (version_id,),
            )
            row = cursor.fetchone()
        return None if row is None else _map_preference(row)

    def list_preferences(self, streamer_id: UUID) -> list[PreferenceVersion]:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, streamer_id, version_number, previous_version_id,
                    source_feedback_id, rollback_of_version_id, category_weights,
                    explanation, created_at
                FROM streamer_preference_versions
                WHERE streamer_id = %s ORDER BY version_number
                """,
                (streamer_id,),
            )
            rows = cursor.fetchall()
        return [_map_preference(row) for row in rows]

    def add_feedback(
        self,
        candidate_id: UUID,
        rating: FeedbackRating,
        reason_tags: tuple[FeedbackReasonTag, ...],
        note: str | None,
    ) -> CandidateFeedback:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT j.streamer_id, c.category_scores
                FROM clip_candidates c
                JOIN candidate_jobs j ON j.id = c.candidate_job_id
                WHERE c.id = %s
                """,
                (candidate_id,),
            )
            candidate = cursor.fetchone()
            if candidate is None:
                raise ValueError("clip candidate not found")
            streamer_id, raw_scores = candidate
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"preference:{streamer_id}",),
            )
            current = self._current_for_update(cursor, streamer_id)
            if current is None:
                current = self._insert_initial(cursor, streamer_id)
            new_weights, explanation = update_weights(
                current.category_weights,
                _map_weights(raw_scores),
                rating,
                reason_tags,
            )
            cursor.execute(
                """
                INSERT INTO streamer_preference_versions (
                    streamer_id, version_number, previous_version_id,
                    category_weights, explanation
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    streamer_id,
                    next_version_number(current.version_number),
                    current.id,
                    Jsonb(_weights_json(new_weights)),
                    Jsonb(list(explanation)),
                ),
            )
            preference_id = _required(cursor.fetchone())[0]
            cursor.execute(
                """
                INSERT INTO candidate_feedback (
                    candidate_id, streamer_id, rating, reason_tags, note,
                    preference_version_id
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, candidate_id, streamer_id, rating, reason_tags,
                    note, preference_version_id, created_at
                """,
                (
                    candidate_id,
                    streamer_id,
                    rating.value,
                    Jsonb([tag.value for tag in reason_tags]),
                    note,
                    preference_id,
                ),
            )
            feedback_row = _required(cursor.fetchone())
            cursor.execute(
                """
                UPDATE streamer_preference_versions
                SET source_feedback_id = %s WHERE id = %s
                """,
                (feedback_row[0], preference_id),
            )
            connection.commit()
        return _map_feedback(feedback_row)

    def list_feedback(self, candidate_id: UUID) -> list[CandidateFeedback]:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, candidate_id, streamer_id, rating, reason_tags,
                    note, preference_version_id, created_at
                FROM candidate_feedback WHERE candidate_id = %s ORDER BY created_at, id
                """,
                (candidate_id,),
            )
            rows = cursor.fetchall()
        return [_map_feedback(row) for row in rows]

    def rollback(self, streamer_id: UUID, target_version_id: UUID) -> PreferenceVersion:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"preference:{streamer_id}",),
            )
            current = self._current_for_update(cursor, streamer_id)
            cursor.execute(
                """
                SELECT id, streamer_id, version_number, previous_version_id,
                    source_feedback_id, rollback_of_version_id, category_weights,
                    explanation, created_at
                FROM streamer_preference_versions
                WHERE id = %s AND streamer_id = %s
                """,
                (target_version_id, streamer_id),
            )
            target_row = cursor.fetchone()
            if current is None or target_row is None:
                raise ValueError("preference version not found for streamer")
            target = _map_preference(target_row)
            cursor.execute(
                """
                INSERT INTO streamer_preference_versions (
                    streamer_id, version_number, previous_version_id,
                    rollback_of_version_id, category_weights, explanation
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, streamer_id, version_number, previous_version_id,
                    source_feedback_id, rollback_of_version_id, category_weights,
                    explanation, created_at
                """,
                (
                    streamer_id,
                    next_version_number(current.version_number),
                    current.id,
                    target.id,
                    Jsonb(_weights_json(rollback_weights(target.category_weights))),
                    Jsonb([
                        "strategy=bounded-category-weights-v1",
                        f"rollback to preference version {target.version_number}",
                    ]),
                ),
            )
            row = _required(cursor.fetchone())
            connection.commit()
        return _map_preference(row)

    def evaluation_candidates(self, streamer_id: UUID) -> list[EvaluationCandidate]:
        with connect(self._database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.id, c.rank, c.category_scores,
                    (
                        SELECT f.rating FROM candidate_feedback f
                        WHERE f.candidate_id = c.id
                        ORDER BY f.created_at DESC, f.id DESC LIMIT 1
                    )
                FROM clip_candidates c
                JOIN candidate_jobs j ON j.id = c.candidate_job_id
                WHERE j.streamer_id = %s
                ORDER BY c.created_at, c.rank
                """,
                (streamer_id,),
            )
            rows = cursor.fetchall()
        return [
            EvaluationCandidate(
                row[0],
                row[1],
                _map_weights(row[2]),
                None if row[3] is None else FeedbackRating(row[3]),
            )
            for row in rows
        ]

    @staticmethod
    def _current_for_update(cursor: Any, streamer_id: UUID) -> PreferenceVersion | None:
        cursor.execute(
            """
            SELECT id, streamer_id, version_number, previous_version_id,
                source_feedback_id, rollback_of_version_id, category_weights,
                explanation, created_at
            FROM streamer_preference_versions
            WHERE streamer_id = %s ORDER BY version_number DESC
            LIMIT 1 FOR UPDATE
            """,
            (streamer_id,),
        )
        row = cursor.fetchone()
        return None if row is None else _map_preference(row)

    @staticmethod
    def _insert_initial(cursor: Any, streamer_id: UUID) -> PreferenceVersion:
        cursor.execute(
            """
            INSERT INTO streamer_preference_versions (
                streamer_id, version_number, category_weights, explanation
            ) VALUES (%s, 1, %s, %s)
            RETURNING id, streamer_id, version_number, previous_version_id,
                source_feedback_id, rollback_of_version_id, category_weights,
                explanation, created_at
            """,
            (
                streamer_id,
                Jsonb(_weights_json(default_weights())),
                Jsonb([
                    "strategy=bounded-category-weights-v1",
                    "initial balanced preferences",
                ]),
            ),
        )
        return _map_preference(_required(cursor.fetchone()))


def _weights_json(weights: dict[CandidateCategory, float]) -> dict[str, float]:
    return {category.value: value for category, value in weights.items()}


def _map_weights(raw: dict[str, float]) -> dict[CandidateCategory, float]:
    return {CandidateCategory(key): float(value) for key, value in raw.items()}


def _map_preference(row: tuple[Any, ...]) -> PreferenceVersion:
    return PreferenceVersion(
        row[0], row[1], row[2], row[3], row[4], row[5],
        _map_weights(row[6]), tuple(row[7]), row[8],
    )


def _map_feedback(row: tuple[Any, ...]) -> CandidateFeedback:
    return CandidateFeedback(
        row[0], row[1], row[2], FeedbackRating(row[3]),
        tuple(FeedbackReasonTag(value) for value in row[4]),
        row[5], row[6], row[7],
    )


def _required(row: tuple[Any, ...] | None) -> tuple[Any, ...]:
    if row is None:
        raise RuntimeError("database did not return the expected row")
    return row
