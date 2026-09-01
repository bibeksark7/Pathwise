"""Configuration is the one place a mistake leaks a secret or boots a bad process."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from pathwise.config import Environment, Settings

REAL_SECRET = "a-real-secret-value-of-at-least-32-bytes"


def test_cors_origins_accepts_comma_separated_string() -> None:
    settings = Settings(cors_origins="http://a.test, http://b.test")  # type: ignore[arg-type]
    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_accepts_a_list() -> None:
    settings = Settings(cors_origins=["http://a.test"])
    assert settings.cors_origins == ["http://a.test"]


@pytest.mark.parametrize("secret", ["", "change-me-in-production", "CHANGEME", "secret"])
def test_production_rejects_placeholder_jwt_secret(secret: str) -> None:
    with pytest.raises(PydanticValidationError, match="PATHWISE_JWT_SECRET"):
        Settings(env=Environment.PRODUCTION, debug=False, jwt_secret=secret)


def test_production_rejects_a_jwt_secret_that_is_too_short() -> None:
    """RFC 7518 s3.2: an HS256 key shorter than the hash output weakens the signature
    below what the algorithm advertises. PyJWT only warns; we refuse to start."""
    with pytest.raises(PydanticValidationError, match="at least 32 bytes"):
        Settings(env=Environment.PRODUCTION, debug=False, jwt_secret="short-but-not-a-placeholder")


def test_production_rejects_debug_mode() -> None:
    with pytest.raises(PydanticValidationError, match="PATHWISE_DEBUG"):
        Settings(env=Environment.PRODUCTION, debug=True, jwt_secret=REAL_SECRET)


def test_production_accepts_a_real_secret() -> None:
    settings = Settings(env=Environment.PRODUCTION, debug=False, jwt_secret=REAL_SECRET)
    assert settings.is_production


def test_development_tolerates_a_placeholder_secret() -> None:
    """Local development must not require secret generation to boot."""
    settings = Settings(env=Environment.DEVELOPMENT, jwt_secret="change-me-in-production")
    assert not settings.is_production


def test_settings_are_immutable() -> None:
    settings = Settings()
    with pytest.raises(PydanticValidationError):
        settings.llm_model = "something-else"  # type: ignore[misc]
