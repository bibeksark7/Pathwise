"""Smoke tests for the application factory and health endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from pathwise import __version__


def test_health_reports_version_and_env(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["env"] == "test"


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers["X-Request-ID"]


def test_supplied_request_id_is_echoed_back(client: TestClient) -> None:
    """Lets a caller correlate its own trace id with our logs."""
    response = client.get("/health", headers={"X-Request-ID": "caller-supplied-id"})
    assert response.headers["X-Request-ID"] == "caller-supplied-id"


def test_readiness_reports_degraded_without_a_database(client: TestClient) -> None:
    """No Postgres is running in the unit suite, so this must fail loudly, not hang."""
    response = client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"].startswith("error:")


def test_openapi_schema_is_served_outside_production(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200
