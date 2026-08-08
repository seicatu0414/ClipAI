from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
