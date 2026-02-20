.PHONY: setup test lint demo clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install all packages in development mode
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

clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
