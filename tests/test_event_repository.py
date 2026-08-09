from uuid import uuid4

from clipai.events.domain import EventJobStatus
from clipai.events.repository import EventRepository


def test_maps_event_job_row_to_domain() -> None:
    job_id = uuid4()
    transcript_id = uuid4()

    job = EventRepository._map_job(
        (
            job_id,
            transcript_id,
            "running",
            40,
            "rules-v1",
            {"minimum_confidence": 0.5},
            None,
        )
    )

    assert job.id == job_id
    assert job.transcript_id == transcript_id
    assert job.status is EventJobStatus.RUNNING
    assert job.configuration["minimum_confidence"] == 0.5
