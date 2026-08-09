from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from clipai.config import get_settings
from clipai.database import apply_migrations, database_is_ready
from clipai.domain import SourceSpec, TranscriptSegment, TranscriptionJob
from clipai.events.configuration import event_configuration
from clipai.events.detectors import JapaneseTranscriptRuleDetector
from clipai.events.domain import DetectedEvent, EventDetectionJob, JsonValue
from clipai.events.repository import EventRepository
from clipai.knowledge.configuration import knowledge_configuration
from clipai.knowledge.domain import (
    Evidence,
    HistoricalStream,
    KnowledgeJob,
    KnowledgeObservation,
    KnowledgeVersion,
    Streamer,
)
from clipai.knowledge.repository import KnowledgeRepository
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


class CreateEventJobRequest(BaseModel):
    transcript_id: UUID


class EventJobResponse(BaseModel):
    id: UUID
    transcript_id: UUID
    status: str
    progress: int
    detector_version: str
    configuration: dict[str, JsonValue]
    error: str | None

    @classmethod
    def from_job(cls, job: EventDetectionJob) -> "EventJobResponse":
        return cls(
            id=job.id,
            transcript_id=job.transcript_id,
            status=job.status.value,
            progress=job.progress,
            detector_version=job.detector_version,
            configuration=job.configuration,
            error=job.error,
        )


class EventResponse(BaseModel):
    event_type: str
    start_seconds: float
    end_seconds: float
    confidence: float
    source_signals: dict[str, JsonValue]
    explanation: str

    @classmethod
    def from_event(cls, event: DetectedEvent) -> "EventResponse":
        return cls(
            event_type=event.event_type.value,
            start_seconds=event.start_seconds,
            end_seconds=event.end_seconds,
            confidence=event.confidence,
            source_signals=event.source_signals,
            explanation=event.explanation,
        )


class CreateStreamerRequest(BaseModel):
    channel_url: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    youtube_channel_id: str | None = None


class StreamerResponse(BaseModel):
    id: UUID
    channel_url: str
    display_name: str
    youtube_channel_id: str | None

    @classmethod
    def from_streamer(cls, streamer: Streamer) -> "StreamerResponse":
        return cls(**streamer.__dict__)


class RegisterStreamRequest(BaseModel):
    streamer_id: UUID
    transcript_id: UUID
    title: str = Field(min_length=1)
    published_at: datetime
    duration_seconds: float = Field(gt=0)
    youtube_video_id: str | None = None
    source_url: str | None = None
    view_count: int = Field(default=0, ge=0)
    comment_count: int = Field(default=0, ge=0)
    manually_selected: bool = False


class StreamResponse(BaseModel):
    id: UUID
    streamer_id: UUID
    transcript_id: UUID
    title: str
    published_at: datetime
    duration_seconds: float
    view_count: int
    comment_count: int
    manually_selected: bool

    @classmethod
    def from_stream(cls, stream: HistoricalStream) -> "StreamResponse":
        return cls(**stream.__dict__)


class CreateKnowledgeJobRequest(BaseModel):
    streamer_id: UUID


class KnowledgeJobResponse(BaseModel):
    id: UUID
    streamer_id: UUID
    status: str
    progress: int
    provider: str
    model: str
    prompt_version: str
    configuration: dict[str, JsonValue]
    error: str | None

    @classmethod
    def from_job(cls, job: KnowledgeJob) -> "KnowledgeJobResponse":
        return cls(
            id=job.id,
            streamer_id=job.streamer_id,
            status=job.status.value,
            progress=job.progress,
            provider=job.provider,
            model=job.model,
            prompt_version=job.prompt_version,
            configuration=job.configuration,
            error=job.error,
        )


class EvidenceResponse(BaseModel):
    transcript_id: UUID
    segment_index: int
    start_seconds: float
    end_seconds: float
    quote: str

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> "EvidenceResponse":
        return cls(**evidence.__dict__)


class ObservationResponse(BaseModel):
    category: str
    statement: str
    origin: str
    confidence: float
    evidence: list[EvidenceResponse]

    @classmethod
    def from_observation(cls, observation: KnowledgeObservation) -> "ObservationResponse":
        return cls(
            category=observation.category.value,
            statement=observation.statement,
            origin=observation.origin.value,
            confidence=observation.confidence,
            evidence=[EvidenceResponse.from_evidence(item) for item in observation.evidence],
        )


class KnowledgeVersionResponse(BaseModel):
    id: UUID
    streamer_id: UUID
    version_number: int
    previous_version_id: UUID | None
    provider: str
    model: str
    prompt_version: str
    configuration: dict[str, JsonValue]
    observations: list[ObservationResponse]

    @classmethod
    def from_version(cls, version: KnowledgeVersion) -> "KnowledgeVersionResponse":
        return cls(
            id=version.id,
            streamer_id=version.streamer_id,
            version_number=version.version_number,
            previous_version_id=version.previous_version_id,
            provider=version.provider,
            model=version.model,
            prompt_version=version.prompt_version,
            configuration=version.configuration,
            observations=[
                ObservationResponse.from_observation(item) for item in version.observations
            ],
        )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    apply_migrations(settings.database_url)
    yield


app = FastAPI(title="ClipAI API", version="0.3.0", lifespan=lifespan)


def _repository() -> TranscriptionRepository:
    return TranscriptionRepository(get_settings().database_url)


def _event_repository() -> EventRepository:
    return EventRepository(get_settings().database_url)


def _knowledge_repository() -> KnowledgeRepository:
    return KnowledgeRepository(get_settings().database_url)


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


@app.post("/v1/event-detection-jobs", response_model=EventJobResponse, status_code=202)
def create_event_job(request: CreateEventJobRequest) -> EventJobResponse:
    source = _repository().get_transcript_source(request.transcript_id)
    if source is None or not Path(source.audio_artifact_path).is_file():
        raise HTTPException(status_code=422, detail="transcript has no normalized audio artifact")
    settings = get_settings()
    job = _event_repository().create_job(
        request.transcript_id,
        detector_version=f"audio-rms-v1+{JapaneseTranscriptRuleDetector.VERSION}+merge-v1",
        configuration=event_configuration(settings),
    )
    return EventJobResponse.from_job(job)


@app.get("/v1/event-detection-jobs/{job_id}", response_model=EventJobResponse)
def get_event_job(job_id: UUID) -> EventJobResponse:
    job = _event_repository().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event detection job not found")
    return EventJobResponse.from_job(job)


@app.get("/v1/transcripts/{transcript_id}/events", response_model=list[EventResponse])
def list_events(
    transcript_id: UUID,
    event_detection_job_id: UUID | None = None,
) -> list[EventResponse]:
    return [
        EventResponse.from_event(event)
        for event in _event_repository().list_events(transcript_id, event_detection_job_id)
    ]


@app.post("/v1/streamers", response_model=StreamerResponse, status_code=201)
def create_streamer(request: CreateStreamerRequest) -> StreamerResponse:
    parsed = urlparse(request.channel_url)
    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower() not in {
        "youtube.com",
        "www.youtube.com",
    }:
        raise HTTPException(status_code=422, detail="only YouTube channel URLs are supported")
    streamer = _knowledge_repository().create_streamer(
        channel_url=request.channel_url,
        display_name=request.display_name,
        youtube_channel_id=request.youtube_channel_id,
    )
    return StreamerResponse.from_streamer(streamer)


@app.post("/v1/streams", response_model=StreamResponse, status_code=201)
def register_stream(request: RegisterStreamRequest) -> StreamResponse:
    repository = _knowledge_repository()
    if repository.get_streamer(request.streamer_id) is None:
        raise HTTPException(status_code=404, detail="streamer not found")
    if _repository().get_transcript_source(request.transcript_id) is None:
        raise HTTPException(status_code=422, detail="completed transcript not found")
    stream = repository.register_stream(**request.model_dump())
    return StreamResponse.from_stream(stream)


@app.post("/v1/knowledge-jobs", response_model=KnowledgeJobResponse, status_code=202)
def create_knowledge_job(request: CreateKnowledgeJobRequest) -> KnowledgeJobResponse:
    repository = _knowledge_repository()
    if repository.get_streamer(request.streamer_id) is None:
        raise HTTPException(status_code=404, detail="streamer not found")
    settings = get_settings()
    job = repository.create_job(
        request.streamer_id,
        provider="ollama",
        model=settings.ollama_model,
        prompt_version="v1",
        configuration=knowledge_configuration(settings),
    )
    return KnowledgeJobResponse.from_job(job)


@app.get("/v1/knowledge-jobs/{job_id}", response_model=KnowledgeJobResponse)
def get_knowledge_job(job_id: UUID) -> KnowledgeJobResponse:
    job = _knowledge_repository().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="knowledge job not found")
    return KnowledgeJobResponse.from_job(job)


@app.get(
    "/v1/streamers/{streamer_id}/knowledge/current",
    response_model=KnowledgeVersionResponse,
)
def get_current_knowledge(streamer_id: UUID) -> KnowledgeVersionResponse:
    version = _knowledge_repository().get_current_version(streamer_id)
    if version is None:
        raise HTTPException(status_code=404, detail="streamer knowledge not found")
    return KnowledgeVersionResponse.from_version(version)
