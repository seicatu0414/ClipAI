from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from clipai.config import get_settings
from clipai.database import apply_migrations, database_is_ready
from clipai.domain import SourceSpec, TranscriptSegment, TranscriptionJob
from clipai.logging import configure_logging
from clipai.repository import TranscriptionRepository


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ready", "unavailable"]


class CreateJobRequest(BaseModel):
    source: str = Field(min_length=1)
    model_size: str | None = Field(default=None, min_length=1)
    language: str | None = Field(default=None, min_length=2, max_length=8)


class JobResponse(BaseModel):
    id: UUID
    source_kind: str
    source: str
    status: str
    progress: int
    model_size: str
    language: str
    error: str | None
    transcript_id: UUID | None = None

    @classmethod
    def from_job(cls, job: TranscriptionJob, transcript_id: UUID | None = None) -> "JobResponse":
        return cls(
            id=job.id,
            source_kind=job.source.kind.value,
            source=job.source.value,
            status=job.status.value,
            progress=job.progress,
            model_size=job.model_size,
            language=job.language,
            error=job.error,
            transcript_id=transcript_id,
        )


class SegmentResponse(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    text: str
    average_log_probability: float | None
    no_speech_probability: float | None

    @classmethod
    def from_segment(cls, segment: TranscriptSegment) -> "SegmentResponse":
        return cls(**segment.__dict__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    apply_migrations(settings.database_url)
    yield


app = FastAPI(title="ClipAI API", version="0.1.0", lifespan=lifespan)


def _repository() -> TranscriptionRepository:
    return TranscriptionRepository(get_settings().database_url)


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    ready = database_is_ready(get_settings().database_url)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if ready else "degraded",
        database="ready" if ready else "unavailable",
    )


@app.post("/v1/transcription-jobs", response_model=JobResponse, status_code=202)
def create_job(request: CreateJobRequest) -> JobResponse:
    try:
        source = SourceSpec.parse(request.source)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    settings = get_settings()
    job = _repository().create_job(
        source,
        model_size=request.model_size or settings.model_size,
        language=request.language or settings.language,
    )
    return JobResponse.from_job(job)


@app.get("/v1/transcription-jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: UUID) -> JobResponse:
    snapshot = _repository().get_job(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="transcription job not found")
    return JobResponse.from_job(snapshot.job, snapshot.transcript_id)


@app.get("/v1/transcripts/{transcript_id}/segments", response_model=list[SegmentResponse])
def list_segments(
    transcript_id: UUID,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[SegmentResponse]:
    segments = _repository().list_segments(transcript_id, offset=offset, limit=limit)
    return [SegmentResponse.from_segment(segment) for segment in segments]
