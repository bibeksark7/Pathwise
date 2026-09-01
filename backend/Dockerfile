# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed_cache

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY pathwise/__init__.py pathwise/__init__.py

# --- development image: editable install with dev extras ---
FROM base AS dev
RUN pip install -e ".[dev]"
COPY . .
EXPOSE 8000
CMD ["uvicorn", "pathwise.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# --- production image ---
FROM base AS prod
RUN pip install .
COPY . .
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "pathwise.main:app", "--host", "0.0.0.0", "--port", "8000"]
