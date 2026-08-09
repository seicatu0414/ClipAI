from uuid import uuid4

from clipai.domain import JobStatus, SourceKind
from clipai.repository import TranscriptionRepository


def test_maps_database_job_row_to_domain() -> None:
    job_id = uuid4()

    job = TranscriptionRepository._map_job(
        (
            job_id,
            "youtube",
            "https://youtu.be/abc",
            "running",
            45,
            "large-v3",
            "ja",
            None,
        )
    )

    assert job.id == job_id
    assert job.source.kind is SourceKind.YOUTUBE
    assert job.status is JobStatus.RUNNING
    assert job.progress == 45
