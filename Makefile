.PHONY: setup setup-pip test lint demo clean help smoke-test

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install all packages in development mode (uv workspace)
	uv sync

setup-pip: ## Install via pip (fallback, no uv required)
	pip install -e packages/contracts
	pip install -e packages/common
	pip install -e packages/storage
	pip install -e packages/pdf-tools
	pip install -e packages/cli
	pip install -e packages/extraction
	pip install -e packages/api
	pip install -e packages/dashboard
	pip install -e packages/mcp-server
	pip install -e packages/s2-client
	pip install -e packages/client
	pip install -e packages/daemon

setup-demo: setup ## Set up demo corpus (download + ingest arXiv papers)
	python scripts/setup_demo.py

test: ## Run unit tests
	pytest -m unit

test-integration: ## Run integration tests (needs PostgreSQL)
	pytest -m integration

test-all: ## Run all tests
	pytest

lint: ## Check code style
	black --check packages/ --line-length 100
	ruff check packages/

format: ## Format code
	black packages/ --line-length 100
	ruff check packages/ --fix

typecheck: ## Run type checking
	mypy packages/

demo: ## Start full demo stack (PostgreSQL + API + Dashboard)
	docker compose --profile demo --profile api up -d

demo-down: ## Stop demo stack
	docker compose --profile demo --profile api down

services: ## Start core services (PostgreSQL + GROBID)
	docker compose up -d

smoke-test: ## Run CLI smoke tests against live database
	@echo "=== Sources sub-app ==="
	.venv/bin/research-kb sources list
	.venv/bin/research-kb sources stats
	.venv/bin/research-kb sources extraction-status
	@echo "=== Search sub-app ==="
	.venv/bin/research-kb search query "instrumental variables"
	.venv/bin/research-kb search audit-assumptions "double machine learning" --no-ollama
	@echo "=== Graph sub-app ==="
	.venv/bin/research-kb graph concepts "IV"
	.venv/bin/research-kb graph neighborhood "double machine learning" --hops 2
	.venv/bin/research-kb graph path "IV" "unconfoundedness"
	@echo "=== Citations sub-app ==="
	.venv/bin/research-kb citations stats
	@echo "=== Discover + Enrich (help only) ==="
	.venv/bin/research-kb discover --help
	.venv/bin/research-kb enrich --help
	@echo "All smoke tests passed"

clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
