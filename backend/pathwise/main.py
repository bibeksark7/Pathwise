"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from pathwise import __version__
from pathwise.api.errors import PathwiseError
from pathwise.api.routes import auth as auth_routes
from pathwise.config import Environment, Settings, get_settings
from pathwise.database.session import dispose_engine, get_engine
from pathwise.logging_config import configure_logging, get_logger, request_id_var

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shutdown."""
    settings: Settings = app.state.settings
    log.info("startup", env=str(settings.env), version=__version__)
    yield
    await dispose_engine()
    log.info("shutdown")


def _problem(
    request: Request, *, status: int, code: str, message: str, **extra: Any
) -> JSONResponse:
    """Build an RFC 9457-shaped error response."""
    body: dict[str, Any] = {
        "type": f"https://pathwise.dev/errors/{code}",
        "title": code.replace("_", " "),
        "status": status,
        "detail": message,
        "instance": str(request.url.path),
        "request_id": request_id_var.get(),
    }
    body.update(extra)
    return JSONResponse(status_code=status, content=body)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Tests call this directly with overridden settings."""
    settings = settings or get_settings()
    configure_logging(
        level=settings.log_level,
        json_output=settings.env is Environment.PRODUCTION,
    )

    app = FastAPI(
        title="Pathwise API",
        version=__version__,
        description="Adaptive learning platform — knowledge graph, mastery model, "
        "and decision engine.",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a request id to logs and the response, and time the request."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed_ms, 2),
        )
        return response

    # --- Error handlers ------------------------------------------------------- #

    @app.exception_handler(PathwiseError)
    async def handle_domain_error(request: Request, exc: PathwiseError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("domain_error", code=exc.error_code, message=exc.message, **exc.details)
        else:
            log.info("domain_error", code=exc.error_code, message=exc.message)
        return _problem(
            request,
            status=exc.status_code,
            code=exc.error_code,
            message=exc.message,
            **exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _problem(
            request,
            status=422,
            code="request_validation_error",
            message="Request body or parameters failed validation.",
            errors=exc.errors(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception", error=str(exc))
        detail = str(exc) if settings.debug else "An unexpected error occurred."
        return _problem(request, status=500, code="internal_error", message=detail)

    # --- Routers -------------------------------------------------------------- #

    app.include_router(auth_routes.router, prefix="/api")

    # --- Health --------------------------------------------------------------- #

    @app.get("/health", tags=["meta"], summary="Liveness probe")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "env": str(settings.env)}

    @app.get("/health/ready", tags=["meta"], summary="Readiness probe")
    async def readiness(response: Response) -> dict[str, Any]:
        """Verify the dependencies the API cannot serve traffic without.

        Returns 503 when degraded so orchestrators pull the instance out of rotation
        instead of routing traffic at a database that is not answering.
        """
        checks: dict[str, str] = {}
        try:
            async with get_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:  # a probe reports failure, it never raises
            checks["database"] = f"error: {exc}"

        ready = all(value == "ok" for value in checks.values())
        if not ready:
            response.status_code = 503
        return {"status": "ready" if ready else "degraded", "checks": checks}

    return app


app = create_app()
