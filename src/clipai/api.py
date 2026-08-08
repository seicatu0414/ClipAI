from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from clipai.config import get_settings
from clipai.database import database_is_ready
from clipai.logging import configure_logging


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ready", "unavailable"]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging(get_settings().log_level)
    yield


app = FastAPI(title="ClipAI API", version="0.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    ready = database_is_ready(get_settings().database_url)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if ready else "degraded",
        database="ready" if ready else "unavailable",
    )
