"""Shared pytest fixtures.

The suite runs offline by default: `PATHWISE_LLM_PROVIDER=fake` returns deterministic
fixture responses, so no test costs money or needs network access. Tests marked
`live` call the real Anthropic API and are skipped unless `--run-live` is passed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Set before any pathwise module imports Settings, so the cached singleton is built
# from test configuration rather than a developer's local .env.
os.environ.setdefault("PATHWISE_ENV", "test")
os.environ.setdefault("PATHWISE_LLM_PROVIDER", "fake")
os.environ.setdefault("PATHWISE_EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("PATHWISE_JWT_SECRET", "test-secret-not-used-in-production-32b+padding")
os.environ.setdefault("PATHWISE_LOG_LEVEL", "WARNING")

from fastapi.testclient import TestClient

from pathwise.config import Settings, get_settings
from pathwise.main import create_app


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run contract tests against the real Anthropic API (costs money).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live"):
        return
    skip = pytest.mark.skip(reason="needs --run-live (calls the real Anthropic API)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A TestClient over a freshly built app."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
