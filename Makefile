.DEFAULT_GOAL := help
COMPOSE := docker compose
BACKEND := $(COMPOSE) exec -T api

.PHONY: help up down logs build migrate revision seed shell test test-live test-unit lint fmt typecheck eval clean

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

clean: ## Remove containers and volumes
	$(COMPOSE) down -v
