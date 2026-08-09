from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLIPAI_",
        extra="ignore",
    )

    environment: str = "local"
    log_level: str = "INFO"
    database_url: str = Field(
        default="postgresql://clipai:clipai@localhost:5432/clipai",
        repr=False,
    )
    media_root: Path = Path("data")
    model_size: str = "large-v3"
    language: str = "ja"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    cpu_fallback: bool = True
    worker_poll_interval_seconds: float = Field(default=2.0, gt=0)
    event_min_confidence: float = Field(default=0.55, ge=0, le=1)
    event_loudness_delta_db: float = Field(default=12.0, gt=0)
    event_silence_db: float = Field(default=-48.0, lt=0)
    event_silence_min_seconds: float = Field(default=2.0, gt=0)
    event_merge_gap_seconds: float = Field(default=2.0, ge=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
