# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research Knowledge Base: A semantic search system for causal inference literature with graph-boosted retrieval. Combines full-text search, vector similarity (BGE-large-en-v1.5, 1024 dimensions), and knowledge graph signals.

## Commands

### Testing

```bash
# All tests
pytest

# By package
pytest packages/cli/tests/ -v
pytest packages/storage/tests/ -v
pytest packages/pdf-tools/tests/ -v
pytest packages/extraction/tests/ -v

# By marker
pytest -m "unit"                    # Fast, isolated
pytest -m "integration"             # Multi-component
pytest -m "e2e"                     # Full pipeline
pytest -m "smoke"                   # Real PDFs
pytest -m "scripts"                 # Script validation
pytest -m "not slow"                # Skip >5s tests
pytest -m "not requires_ollama"     # Skip Ollama tests
pytest -m "not requires_embedding"  # Skip embedding tests
```

### Installation

```bash
pip install -e packages/cli
pip install -e packages/storage
pip install -e packages/pdf-tools
pip install -e packages/contracts
pip install -e packages/common
pip install -e packages/extraction  # Optional
```

### Docker Services

```bash
docker-compose up -d                    # PostgreSQL + GROBID
docker-compose --profile dev up -d      # Include pgAdmin
```

### Code Quality

```bash
black packages/           # Format (100-char lines)
ruff check packages/      # Lint
mypy packages/            # Type check
```

### Data Operations

```bash
python scripts/ingest_corpus.py                  # Ingest corpus
python scripts/extract_concepts.py --limit 1000  # Extract concepts (requires Ollama)
python scripts/eval_retrieval.py                 # Validate retrieval quality
python scripts/run_quality_checks.py             # Quality metrics
```

### CLI Usage

```bash
research-kb query "instrumental variables"        # Default (graph-boosted)
research-kb query "test" --no-graph               # Without graph
research-kb sources                               # List sources
research-kb stats                                 # Database statistics
research-kb concepts "IV"                         # Concept search
research-kb graph "double machine learning" --hops 2  # Graph exploration
```

## Architecture

### Package Dependency Graph

```
contracts (pure Pydantic models)
    ↓
common (logging, retry, instrumentation)
    ├─→ storage (PostgreSQL + pgvector)
    │     ├─→ cli
    │     ├─→ pdf-tools
    │     └─→ extraction
    ├─→ pdf-tools
    └─→ extraction
```

### Package Responsibilities

| Package | Purpose |
|---------|---------|
| **contracts** | Pure Pydantic schemas - zero business logic |
| **common** | Cross-cutting: logging (structlog), retry (tenacity), tracing (OpenTelemetry) |
| **storage** | Exclusive database ownership (asyncpg, pgvector) |
| **pdf-tools** | PDF extraction (PyMuPDF, GROBID) + embeddings (sentence-transformers) |
| **cli** | Typer-based interface, thin wrapper |
| **extraction** | Concept extraction via Ollama LLM |

### Database Schema

**Core tables:** `sources`, `chunks`, `citations`
**Knowledge graph:** `concepts`, `concept_relationships`, `chunk_concepts`, `methods`, `assumptions`

Key enums:
- `ConceptType`: METHOD, ASSUMPTION, PROBLEM, DEFINITION, THEOREM
- `RelationshipType`: REQUIRES, USES, ADDRESSES, GENERALIZES, SPECIALIZES, ALTERNATIVE_TO, EXTENDS

### Hybrid Search

```
score = fts_weight × fts + vector_weight × vector + graph_weight × graph
```

Context types adjust weights:
- **building**: 20% FTS, 80% vector (favor semantic breadth)
- **auditing**: 50% FTS, 50% vector (favor precision)
- **balanced**: 30% FTS, 70% vector (default)

## Key Patterns

### Async Throughout

All storage operations are async. Use `asyncpg` connection pooling (2-10 connections).

```python
async with pool.acquire() as conn:
    result = await conn.fetch("SELECT ...")
```

### JSONB Extensibility

Unknown fields → `metadata` JSONB column. Promote to dedicated table when patterns emerge.

### Testing

- All tests use `pytest-asyncio` with `asyncio_mode = auto`
- Function-scoped event loops
- Mock fixtures: `mock_ollama`, `mock_embedding_client`
- Float comparisons: use `pytest.approx(value, rel=1e-5)`

### Error Handling

Custom errors from `research_kb_common`: `IngestionError`, `StorageError`, `SearchError`

### Embeddings

Single model: BGE-large-en-v1.5 (1024 dimensions). All vector columns are `vector(1024)`.

## CI/CD Tiers

1. **PR Checks** (<10 min): Unit + CLI tests with mocked services
2. **Daily Validation** (3 min): Quality checks with cached DB
3. **Weekly Full Rebuild** (60 min): Complete from-scratch validation

## Gotchas

- GROBID takes ~60s to start (healthcheck has 60s start_period)
- Graph search gracefully falls back to FTS+vector if concepts not extracted
- Table name is `concept_relationships` (not `relationships`)
- CLI adds packages to `sys.path` for development mode imports
