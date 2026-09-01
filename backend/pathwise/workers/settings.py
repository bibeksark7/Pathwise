"""arq worker configuration.

Background work that must not block a request: roadmap generation, resource URL
validation, embedding backfill, adaptation recomputation, and evaluation runs.
Task functions are registered here as they are built.
"""

from __future__ import annotations

from typing import Any, ClassVar

from arq.connections import RedisSettings

from pathwise.config import get_settings
from pathwise.database.session import dispose_engine
from pathwise.logging_config import configure_logging, get_logger

log = get_logger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.is_production)
    log.info("worker_startup", env=str(settings.env))


async def shutdown(ctx: dict[str, Any]) -> None:
    await dispose_engine()
    log.info("worker_shutdown")


class WorkerSettings:
    """Entry point for `arq pathwise.workers.settings.WorkerSettings`."""

    functions: ClassVar[list[Any]] = []
    cron_jobs: ClassVar[list[Any]] = []
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10
    job_timeout = 600
    keep_result = 3600

    @staticmethod
    def redis_settings() -> RedisSettings:
        return RedisSettings.from_dsn(str(get_settings().redis_url))
