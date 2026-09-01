"""Structured logging.

JSON in production (machine-parseable), coloured key=value in development.
A `request_id` contextvar is bound by middleware so every log line emitted while
handling a request carries it, including logs from deep inside the service layer.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

import structlog

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def _add_request_id(
    _logger: object, _name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    request_id = request_id_var.get()
    if request_id is not None:
        event_dict["request_id"] = request_id
    return event_dict


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog and route stdlib logging through it."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_request_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for `name`."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
