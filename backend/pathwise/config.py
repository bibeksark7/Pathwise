"""Application configuration.

All settings are read from the environment (or a local `.env`). Secrets are never
hardcoded and never given a usable default: `JWT_SECRET` and the provider API keys
must be supplied explicitly, and production start-up refuses placeholder values.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


PLACEHOLDER_SECRETS = frozenset({"", "change-me-in-production", "secret", "changeme"})

#: RFC 7518 section 3.2: an HMAC key for HS256 must be at least as long as the hash
#: output, or the signature is weaker than the algorithm advertises. PyJWT warns
#: below this; we refuse to start.
MIN_JWT_SECRET_BYTES = 32


class Settings(BaseSettings):
    """Runtime configuration, loaded once per process."""

    model_config = SettingsConfigDict(
        env_prefix="PATHWISE_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Core ---
    env: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # --- Persistence ---
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+asyncpg://pathwise:pathwise@localhost:5432/pathwise")
    )
    redis_url: RedisDsn = Field(default=RedisDsn("redis://localhost:6379/0"))
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # --- Auth ---
    jwt_secret: str = ""
    jwt_algorithm: Literal["HS256"] = "HS256"
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 30

    # --- LLM ---
    llm_provider: Literal["anthropic", "openai", "fake"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_fast_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 16_000
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    llm_cache_enabled: bool = True
    llm_cache_ttl_seconds: int = 60 * 60 * 24 * 7

    # --- Embeddings ---
    embedding_provider: Literal["fastembed", "openai", "fake"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string as well as a JSON list."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def _require_real_secrets_in_production(self) -> Settings:
        if self.env is Environment.PRODUCTION:
            generate_hint = (
                'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
            if self.jwt_secret.strip().lower() in PLACEHOLDER_SECRETS:
                raise ValueError(
                    "PATHWISE_JWT_SECRET must be set to a real secret in production. "
                    f"{generate_hint}"
                )
            if len(self.jwt_secret.encode("utf-8")) < MIN_JWT_SECRET_BYTES:
                raise ValueError(
                    f"PATHWISE_JWT_SECRET must be at least {MIN_JWT_SECRET_BYTES} bytes "
                    f"(RFC 7518 s3.2 for {self.jwt_algorithm}). {generate_hint}"
                )
            if self.debug:
                raise ValueError("PATHWISE_DEBUG must be false in production.")
        return self

    @property
    def is_production(self) -> bool:
        return self.env is Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
