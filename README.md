# Pathwise

An adaptive learning platform for technical subjects. Pathwise builds a personalised
roadmap over a persistent knowledge graph, measures what you actually understand,
finds the prerequisite behind your weak spots, and restructures the path as evidence
accumulates.

> **Design principle: deterministic logic decides, the LLM explains.**
> Graph traversal, mastery estimation, next-topic ranking, and roadmap mutation are
> pure, unit-tested functions. The LLM never invents a resource URL, never computes a
> mastery score, and never picks the next topic — it renders a machine-produced
> decision trace into prose, and that prose is validated against the trace before it
> is stored.

## Status

Under active construction. See `docs/` for the architecture plan and phase order.

## Stack

| Layer | Choice |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind, React Flow |
| Backend | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2 (async) |
| Database | PostgreSQL 16 + pgvector, Alembic migrations |
| Queue/cache | Redis + arq workers |
| LLM | Anthropic (`claude-opus-5`) behind a provider abstraction |
| Embeddings | fastembed (`BAAI/bge-small-en-v1.5`, 384-dim, ONNX — no API key) |
| Testing | pytest, Hypothesis, Vitest, Playwright |

## Prerequisites

- **Docker Desktop** (provides Postgres 16 + pgvector and Redis)
- An **`ANTHROPIC_API_KEY`** — required from Phase 2 onward; the deterministic core
  and its test suite run without one.

## Quickstart

```bash
cp .env.example .env      # then fill in ANTHROPIC_API_KEY and PATHWISE_JWT_SECRET
make up                   # start postgres + redis + api + worker
make migrate              # apply database migrations
make seed                 # load the curated knowledge graph and resource catalog
```

API docs: http://localhost:8000/docs · Health: http://localhost:8000/health

## Development

```bash
make test        # full backend suite (offline — uses the deterministic fake LLM provider)
make test-live   # opt-in contract tests against the real Anthropic API (costs money)
make lint        # ruff + mypy
make fmt         # ruff format
make eval        # AI evaluation suites; exits nonzero on regression
make down        # stop services
```

The test suite never calls a real model. `PATHWISE_LLM_PROVIDER=fake` returns
deterministic fixture responses, so CI is offline, free, and reproducible.

## Layout

```
backend/pathwise/
  api/         FastAPI routes and request/response schemas (no business logic)
  services/    Business logic — knowledge, decision, adaptation, roadmap, tutor, ...
  ai/          Provider abstraction, prompt registry, validators, cost tracking
  models/      SQLAlchemy ORM models
  database/    Session management and Alembic migrations
  workers/     arq background tasks
  evaluation/  Eval runner and scorers
  data/        Curated knowledge-graph and resource seed files (YAML)
frontend/src/
  features/    roadmap, dashboard, tutor, assessments, resources, projects
```

## License

MIT
