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
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    prompt_root: Path = Path("prompts")
    knowledge_max_historical_hours: float = Field(default=50.0, gt=0)
    knowledge_max_representative_streams: int = Field(default=10, ge=0, le=100)
    knowledge_chunk_characters: int = Field(default=12_000, ge=1_000)
    candidate_target_count: int = Field(default=25, ge=1, le=30)
    candidate_minimum_seconds: float = Field(default=15.0, ge=1)
    candidate_maximum_seconds: float = Field(default=120.0, ge=15)
    candidate_padding_before_seconds: float = Field(default=8.0, ge=0)
    candidate_padding_after_seconds: float = Field(default=12.0, ge=0)
    candidate_merge_gap_seconds: float = Field(default=8.0, ge=0)
    candidate_overlap_threshold: float = Field(default=0.65, gt=0, le=1)
    candidate_maximum_knowledge_observations: int = Field(default=5, ge=0, le=20)


@lru_cache
def get_settings() -> Settings:
    return Settings()
