from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from clipai.candidates.configuration import (
    PIPELINE_VERSION,
    PROMPT_VERSION,
    candidate_configuration,
)
from clipai.candidates.domain import CandidateJob, ClipCandidate
from clipai.candidates.repository import CandidateRepository
from clipai.config import get_settings
from clipai.database import apply_migrations, database_is_ready
from clipai.domain import SourceSpec, TranscriptSegment, TranscriptionJob
from clipai.events.configuration import event_configuration
from clipai.events.detectors import JapaneseTranscriptRuleDetector
from clipai.events.domain import DetectedEvent, EventDetectionJob, JsonValue
from clipai.events.repository import EventRepository
from clipai.feedback.domain import (
    CandidateFeedback,
    FeedbackRating,
    FeedbackReasonTag,
    PreferenceEvaluation,
    PreferenceVersion,
)
from clipai.feedback.learning import evaluate_preferences
from clipai.feedback.repository import FeedbackRepository
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


class CreateCandidateJobRequest(BaseModel):
    streamer_id: UUID
    transcript_id: UUID


class CandidateJobResponse(BaseModel):
    id: UUID
    streamer_id: UUID
    transcript_id: UUID
    event_detection_job_id: UUID
    knowledge_version_id: UUID
    preference_version_id: UUID | None
    status: str
    progress: int
    pipeline_version: str
    provider: str
    model: str
    prompt_version: str
    configuration: dict[str, JsonValue]
    error: str | None

    @classmethod
    def from_job(cls, job: CandidateJob) -> "CandidateJobResponse":
        data = job.__dict__.copy()
        data["status"] = job.status.value
        return cls(**data)


class CandidateResponse(BaseModel):
    id: UUID
    rank: int
    start_seconds: float
    end_seconds: float
    category_scores: dict[str, float]
    overall_score: float
    confidence: float
    reasons: list[str]
    event_ids: list[UUID]
    knowledge_observation_ids: list[UUID]

    @classmethod
    def from_candidate(cls, candidate: ClipCandidate) -> "CandidateResponse":
        return cls(
            id=_required_candidate_id(candidate),
            rank=candidate.rank,
            start_seconds=candidate.start_seconds,
            end_seconds=candidate.end_seconds,
            category_scores={
                key.value: value for key, value in candidate.category_scores.items()
            },
            overall_score=candidate.overall_score,
            confidence=candidate.confidence,
            reasons=list(candidate.reasons),
            event_ids=list(candidate.event_ids),
            knowledge_observation_ids=list(candidate.knowledge_observation_ids),
        )


class CreateFeedbackRequest(BaseModel):
    rating: Literal["◎", "○", "×"]
    reason_tags: list[FeedbackReasonTag] = Field(default_factory=list)
    note: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: UUID
    candidate_id: UUID
    streamer_id: UUID
    rating: str
    reason_tags: list[str]
    note: str | None
    preference_version_id: UUID
    created_at: datetime

    @classmethod
    def from_feedback(cls, feedback: CandidateFeedback) -> "FeedbackResponse":
        symbols = {
            FeedbackRating.EXCELLENT: "◎",
            FeedbackRating.USABLE: "○",
            FeedbackRating.REJECT: "×",
        }
        return cls(
            id=feedback.id,
            candidate_id=feedback.candidate_id,
            streamer_id=feedback.streamer_id,
            rating=symbols[feedback.rating],
            reason_tags=[tag.value for tag in feedback.reason_tags],
            note=feedback.note,
            preference_version_id=feedback.preference_version_id,
            created_at=feedback.created_at,
        )


class PreferenceResponse(BaseModel):
    id: UUID
    streamer_id: UUID
    version_number: int
    previous_version_id: UUID | None
    source_feedback_id: UUID | None
    rollback_of_version_id: UUID | None
    category_weights: dict[str, float]
    explanation: list[str]
    created_at: datetime

    @classmethod
    def from_preference(cls, preference: PreferenceVersion) -> "PreferenceResponse":
        return cls(
            id=preference.id,
            streamer_id=preference.streamer_id,
            version_number=preference.version_number,
            previous_version_id=preference.previous_version_id,
            source_feedback_id=preference.source_feedback_id,
            rollback_of_version_id=preference.rollback_of_version_id,
            category_weights={
                key.value: value for key, value in preference.category_weights.items()
            },
            explanation=list(preference.explanation),
            created_at=preference.created_at,
        )


class RollbackPreferenceRequest(BaseModel):
    target_version_id: UUID


class EvaluationResultResponse(BaseModel):
    preference_version_id: UUID
    ranked_candidate_ids: list[UUID]
    average_accepted_rank: float | None
    precision_at_20: float | None
    precision_at_30: float | None

    @classmethod
    def from_evaluation(cls, result: PreferenceEvaluation) -> "EvaluationResultResponse":
        return cls(
            preference_version_id=result.preference_version_id,
            ranked_candidate_ids=list(result.ranked_candidate_ids),
            average_accepted_rank=result.average_accepted_rank,
            precision_at_20=result.precision_at_20,
            precision_at_30=result.precision_at_30,
        )


class PreferenceComparisonResponse(BaseModel):
    before: EvaluationResultResponse
    after: EvaluationResultResponse


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    apply_migrations(settings.database_url)
    yield


app = FastAPI(title="ClipAI API", version="0.5.0", lifespan=lifespan)


def _repository() -> TranscriptionRepository:
    return TranscriptionRepository(get_settings().database_url)


def _event_repository() -> EventRepository:
    return EventRepository(get_settings().database_url)


def _knowledge_repository() -> KnowledgeRepository:
    return KnowledgeRepository(get_settings().database_url)


def _candidate_repository() -> CandidateRepository:
    return CandidateRepository(get_settings().database_url)


def _feedback_repository() -> FeedbackRepository:
    return FeedbackRepository(get_settings().database_url)


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


@app.post("/v1/candidate-jobs", response_model=CandidateJobResponse, status_code=202)
def create_candidate_job(request: CreateCandidateJobRequest) -> CandidateJobResponse:
    settings = get_settings()
    if _knowledge_repository().get_streamer(request.streamer_id) is None:
        raise HTTPException(status_code=404, detail="streamer not found")
    try:
        preference = _feedback_repository().ensure_current_preference(request.streamer_id)
        job = _candidate_repository().create_job(
            request.streamer_id,
            request.transcript_id,
            pipeline_version=PIPELINE_VERSION,
            provider="ollama",
            model=settings.ollama_model,
            prompt_version=PROMPT_VERSION,
            configuration=candidate_configuration(settings),
            preference_version_id=preference.id,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return CandidateJobResponse.from_job(job)


@app.get("/v1/candidate-jobs/{job_id}", response_model=CandidateJobResponse)
def get_candidate_job(job_id: UUID) -> CandidateJobResponse:
    job = _candidate_repository().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="candidate job not found")
    return CandidateJobResponse.from_job(job)


@app.get(
    "/v1/candidate-jobs/{job_id}/candidates",
    response_model=list[CandidateResponse],
)
def list_candidates(job_id: UUID) -> list[CandidateResponse]:
    repository = _candidate_repository()
    if repository.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="candidate job not found")
    return [
        CandidateResponse.from_candidate(item)
        for item in repository.list_candidates(job_id)
    ]


def _required_candidate_id(candidate: ClipCandidate) -> UUID:
    if candidate.id is None:
        raise RuntimeError("persisted candidate has no ID")
    return candidate.id


@app.post("/v1/candidates/{candidate_id}/feedback", response_model=FeedbackResponse)
def create_feedback(
    candidate_id: UUID,
    request: CreateFeedbackRequest,
) -> FeedbackResponse:
    ratings = {
        "◎": FeedbackRating.EXCELLENT,
        "○": FeedbackRating.USABLE,
        "×": FeedbackRating.REJECT,
    }
    try:
        feedback = _feedback_repository().add_feedback(
            candidate_id,
            ratings[request.rating],
            tuple(request.reason_tags),
            request.note,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FeedbackResponse.from_feedback(feedback)


@app.get(
    "/v1/candidates/{candidate_id}/feedback",
    response_model=list[FeedbackResponse],
)
def list_feedback(candidate_id: UUID) -> list[FeedbackResponse]:
    return [
        FeedbackResponse.from_feedback(item)
        for item in _feedback_repository().list_feedback(candidate_id)
    ]


@app.get(
    "/v1/streamers/{streamer_id}/preferences",
    response_model=list[PreferenceResponse],
)
def list_preferences(streamer_id: UUID) -> list[PreferenceResponse]:
    return [
        PreferenceResponse.from_preference(item)
        for item in _feedback_repository().list_preferences(streamer_id)
    ]


@app.post(
    "/v1/streamers/{streamer_id}/preferences/rollback",
    response_model=PreferenceResponse,
)
def rollback_preference(
    streamer_id: UUID,
    request: RollbackPreferenceRequest,
) -> PreferenceResponse:
    try:
        preference = _feedback_repository().rollback(
            streamer_id, request.target_version_id
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return PreferenceResponse.from_preference(preference)


@app.get(
    "/v1/streamers/{streamer_id}/preferences/compare",
    response_model=PreferenceComparisonResponse,
)
def compare_preferences(
    streamer_id: UUID,
    before_version_id: UUID,
    after_version_id: UUID,
) -> PreferenceComparisonResponse:
    repository = _feedback_repository()
    before = repository.get_preference(before_version_id)
    after = repository.get_preference(after_version_id)
    if (
        before is None
        or after is None
        or before.streamer_id != streamer_id
        or after.streamer_id != streamer_id
    ):
        raise HTTPException(status_code=404, detail="preference version not found")
    candidates = repository.evaluation_candidates(streamer_id)
    return PreferenceComparisonResponse(
        before=EvaluationResultResponse.from_evaluation(
            evaluate_preferences(candidates, before.category_weights, before.id)
        ),
        after=EvaluationResultResponse.from_evaluation(
            evaluate_preferences(candidates, after.category_weights, after.id)
        ),
    )
