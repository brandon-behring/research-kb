# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Research Knowledge Base: A semantic search system for research literature across 35+ domains with graph-boosted retrieval. Combines full-text search, vector similarity (BGE-large-en-v1.5, 1024 dims), and knowledge graph signals.

**Docs entry point**: [`docs/INDEX.md`](docs/INDEX.md)

## Commands

```bash
# Testing
pytest                             # All tests
pytest packages/storage/tests/ -v  # By package

# Installation
uv sync                            # Recommended (uv workspace)

# Docker
docker-compose up -d               # PostgreSQL + GROBID

# Code quality
black packages/                    # Format (100-char lines)
ruff check packages/               # Lint
mypy packages/                     # Type check
```

## Architecture

### Package Dependency Graph

```
contracts (pure Pydantic models)
    |
common (logging, retry, instrumentation)
    |--- storage (PostgreSQL + pgvector + KuzuDB)
    |      |--- cli
    |      |--- pdf-tools
    |      |--- extraction
    |      |--- api
    |      |--- dashboard
    |      |--- daemon
    |      |--- mcp-server
    |      '--- client
    |--- pdf-tools
    |--- extraction
    '--- s2-client
```

### Package Responsibilities

| Package | Purpose |
|---------|---------|
| **contracts** | Pure Pydantic schemas - zero business logic |
| **common** | Cross-cutting: logging (structlog), retry (tenacity), tracing (OpenTelemetry) |
| **storage** | Exclusive database ownership (asyncpg, pgvector) |
| **pdf-tools** | PDF extraction (Docling, GROBID) + embeddings (sentence-transformers) |
| **cli** | Typer-based interface, thin wrapper |
| **extraction** | Concept extraction via Ollama LLM |
| **api** | FastAPI REST endpoints with health checks and metrics |
| **dashboard** | Streamlit visualization for search and graph exploration |
| **s2-client** | Semantic Scholar API client with rate limiting and caching |
| **daemon** | Low-latency query service via Unix socket (JSON-RPC 2.0) |
| **mcp-server** | Model Context Protocol server for Claude Code integration |
| **client** | DaemonClient SDK (JSON-RPC 2.0) with CLI fallback |

### Database Schema

**Core tables:** `sources`, `chunks`, `citations`, `source_citations`
**Knowledge graph:** `concepts`, `concept_relationships`, `chunk_concepts`, `methods`, `assumptions`
**Assumption auditing:** `method_assumption_cache`, `method_aliases`

## Key Patterns

### Async Throughout

All storage operations are async. Use `asyncpg` connection pooling (2-10 connections).

```python
async with pool.acquire() as conn:
    result = await conn.fetch("SELECT ...")
```

### JSONB Extensibility

Unknown fields -> `metadata` JSONB column. Promote to dedicated table when patterns emerge.

### Error Handling

Custom errors from `research_kb_common`: `IngestionError`, `StorageError`, `SearchError`

### Embeddings

Single model: BGE-large-en-v1.5 (1024 dimensions). All vector columns are `vector(1024)`.

### Hybrid Search (Summary)

Default 3-way: FTS + vector + citation. Graph disabled pending re-extraction. Context types: building (20/80), auditing (50/50), balanced (30/70). See `.claude/rules/architecture.md` for weight tuning details.

## Data Protection

### Safe Docker Operations

**CRITICAL**: Use the safe docker wrapper to prevent accidental data loss:

```bash
# Add alias to ~/.bashrc
alias dc='./scripts/docker-safe.sh'

# Usage (intercepts dangerous operations)
dc down -v    # Warns, shows data counts, requires 'DELETE' confirmation
dc up -d      # Works normally
```

### Backups

- **Automatic**: Created before every extraction run (unless `--skip-backup`)
- **Manual**: `./scripts/backup_db.sh`
- **Location**: `backups/` directory (last 5 kept)
- **Recovery**: See [`docs/RECOVERY.md`](docs/RECOVERY.md)

## Gotchas

- GROBID takes ~60s to start (healthcheck has 60s start_period)
- Graph search gracefully falls back to FTS+vector if concepts not extracted
- Table name is `concept_relationships` (not `relationships`)
- Uses `uv` workspaces for package resolution; `pip install -e` still works as fallback
- **NEVER use `docker compose down -v`** without the safe wrapper — it deletes all data

## Documentation Protocol

When modifying code, update docs accordingly:

| Change Type | Required Doc Updates |
|-------------|---------------------|
| New CLI command | CLAUDE.md, README.md |
| New package | Create README.md, add to architecture above |
| New extraction backend | packages/extraction/README.md comparison table |
| External path change | docs/INTEGRATION.md |
| New API endpoint | Run `scripts/generate_package_docs.py` |
| Database schema | docs/phases/ relevant phase doc |

Run `python scripts/audit_docs.py` periodically to detect drift.
