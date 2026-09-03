.DEFAULT_GOAL := help
COMPOSE := docker compose
BACKEND := $(COMPOSE) exec -T api

.PHONY: help up down logs build migrate revision seed shell test test-live test-unit lint fmt typecheck eval web web-test web-check web-build fixtures clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up: ## Start all services
	$(COMPOSE) up -d
down: ## Stop all services
	$(COMPOSE) down
logs: ## Tail service logs
	$(COMPOSE) logs -f --tail=100
build: ## Rebuild images
	$(COMPOSE) build

migrate: ## Apply database migrations
	$(BACKEND) alembic upgrade head
revision: ## Autogenerate a migration: make revision m="add x"
	$(BACKEND) alembic revision --autogenerate -m "$(m)"
seed: ## Load knowledge graph + resource catalog
	$(BACKEND) python -m pathwise.cli seed --all
shell: ## Shell into the api container
	$(COMPOSE) exec api bash

test: ## Run the backend suite (offline, fake LLM provider)
	$(BACKEND) pytest -q
test-unit: ## Run unit tests only
	$(BACKEND) pytest -q tests/unit
test-live: ## Opt-in contract tests against the real Anthropic API (costs money)
	$(BACKEND) pytest -q -m live --run-live

lint: ## ruff check + mypy
	$(BACKEND) ruff check .
	$(BACKEND) mypy pathwise
fmt: ## Format with ruff
	$(BACKEND) ruff format .
	$(BACKEND) ruff check --fix .
typecheck: ## mypy only
	$(BACKEND) mypy pathwise

eval: ## Run AI evaluation suites (fails on regression)
	$(BACKEND) python -m pathwise.evaluation.run --suite all

web: ## Run the frontend dev server (no Docker or API key needed)
	cd frontend && npm install && npm run dev
web-test: ## Frontend unit tests
	cd frontend && npm run test
web-check: ## Frontend typecheck
	cd frontend && npm run typecheck
web-build: ## Production build of the frontend
	cd frontend && npm run build

fixtures: ## Regenerate frontend fixtures from the deterministic engines
	cd backend && python -m pathwise.cli_fixtures --out ../frontend/src/lib/fixtures.ts

clean: ## Remove containers and volumes
	$(COMPOSE) down -v
