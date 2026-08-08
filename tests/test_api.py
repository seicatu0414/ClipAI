from unittest.mock import patch

from fastapi.testclient import TestClient

from clipai.api import app


def test_health_is_ok_when_database_is_ready() -> None:
    with patch("clipai.api.database_is_ready", return_value=True):
        response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ready"}


def test_health_is_degraded_when_database_is_unavailable() -> None:
    with patch("clipai.api.database_is_ready", return_value=False):
        response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unavailable"}
