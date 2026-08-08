from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID


class SourceKind(StrEnum):
    LOCAL_FILE = "local_file"
    YOUTUBE = "youtube"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SourceSpec:
    kind: SourceKind
    value: str

    @classmethod
    def parse(cls, value: str, *, require_local_file: bool = True) -> "SourceSpec":
        candidate = value.strip()
        if not candidate:
            raise ValueError("source must not be empty")

        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            host = (parsed.hostname or "").lower()
            if host not in {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}:
                raise ValueError("only YouTube URLs are supported")
            return cls(SourceKind.YOUTUBE, candidate)

        path = Path(candidate).expanduser()
        if require_local_file and not path.is_file():
            raise ValueError(f"local video file does not exist: {path}")
        return cls(SourceKind.LOCAL_FILE, str(path))


@dataclass(frozen=True)
class TranscriptionJob:
    id: UUID
    source: SourceSpec
    status: JobStatus
    progress: int
    model_size: str
    language: str
    error: str | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    index: int
    start_seconds: float
    end_seconds: float
    text: str
    average_log_probability: float | None = None
    no_speech_probability: float | None = None


@dataclass(frozen=True)
class TranscriptMetadata:
    duration_seconds: float | None
    detected_language: str | None
    language_probability: float | None
    model_size: str
    device: str
