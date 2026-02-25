# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Entry Point

**Start here**: [`docs/INDEX.md`](docs/INDEX.md) - Navigation hub with current status, phase overview, and links to all documentation.

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

# By marker (comprehensive coverage as of Phase G)
pytest -m "unit"                    # ~1,570 tests (fast, mocked)
pytest -m "integration"             # ~343 tests (needs PostgreSQL)
pytest -m "requires_embedding"      # ~19 tests (needs embed_server)
pytest -m "requires_reranker"       # ~20 tests (needs reranker model)
pytest -m "requires_ollama"         # ~32 tests (needs Ollama)

# CI-safe (excludes all service-dependent tests)
pytest -m "unit and not requires_embedding and not requires_ollama and not requires_reranker and not requires_grobid"
pytest -m "integration and not requires_embedding and not requires_ollama and not requires_reranker"
```

### Installation

```bash
# Recommended (uv workspace — single command)
uv sync

# Fallback (pip — manual editable installs)
make setup-pip
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

### Ingestion Best Practices

```bash
# Recommended: Use quiet mode for Claude Code monitoring (minimal output)
python scripts/ingest_missing_textbooks.py --quiet

# JSON output for programmatic parsing
python scripts/ingest_missing_textbooks.py --quiet --json > ingestion_report.json

# Full verbose output for debugging
python scripts/ingest_missing_textbooks.py
```

**Error Recovery:**
- Failed files with `recoverable: true` can be re-ingested later
- Memory errors indicate PDF too large (contact maintainer)
- Embedding service errors: ensure embed_server is running
- Database errors: check PostgreSQL connection and disk space

### Extraction Profiles

**Fast Profile** (Ollama, GPU): ~50 chunks/min on NVIDIA GPU (8GB+ VRAM)
```bash
python scripts/extract_concepts.py \
  --backend ollama \
  --model llama3.1:8b \
  --concurrency 2 \
  --metrics-file /tmp/extraction_metrics.txt
```

**Quality Profile** (Anthropic): Higher accuracy, ~20 chunks/min
```bash
python scripts/extract_concepts.py \
  --backend anthropic \
  --model haiku \
  --concurrency 4
```

**Ollama Optimization** (already applied via systemd override):
```bash
# /etc/systemd/system/ollama.service.d/override.conf
OLLAMA_FLASH_ATTENTION=1   # Enable flash attention
OLLAMA_NUM_PARALLEL=2      # Allow 2 parallel streams
OLLAMA_KV_CACHE_TYPE=q8_0  # Quantized KV cache
```

### CLI Usage

```bash
# Search and retrieval (4-way ranking: FTS + vector + graph + citation)
research-kb search query "instrumental variables"            # Default (all signals)
research-kb search query "test" --no-graph                   # Without graph
research-kb search query "IV" --no-citations                 # Without citation authority
research-kb search query "IV" --citation-weight 0.25         # Boost citation influence
research-kb search query "IV" --context building             # Context-tuned weights

# Source management
research-kb sources list                                     # List sources
research-kb sources stats                                    # Database statistics
research-kb sources extraction-status                        # Extraction pipeline stats

# Knowledge graph
research-kb graph concepts "IV"                              # Concept search
research-kb graph neighborhood "DML" --hops 2                # Graph exploration
research-kb graph path "IV" "unconfoundedness"               # Shortest path between concepts

# Citation network
research-kb citations list <source>                          # List citations from a source
research-kb citations cited-by <source>                      # Find sources citing this one
research-kb citations cites <source>                         # Find sources this one cites
research-kb citations stats                                  # Corpus citation statistics
research-kb citations similar <source>                       # Find similar sources (shared refs)

# Assumption auditing (North Star feature)
research-kb search audit-assumptions "double machine learning"  # Get required assumptions
research-kb search audit-assumptions "IV" --no-ollama           # Graph only, no LLM fallback
research-kb search audit-assumptions "DML" --format json        # JSON output

# Semantic Scholar discovery (s2-client)
research-kb discover search "double machine learning"  # Search S2 for papers
research-kb discover topics                            # Browse by topic
research-kb discover author "Chernozhukov"             # Find by author
research-kb enrich citations                           # Enrich corpus with S2 metadata
research-kb enrich status                              # Show enrichment status
research-kb enrich job-status                          # Check enrichment job status
```

## Architecture

### Package Dependency Graph

```
contracts (pure Pydantic models)
    ↓
common (logging, retry, instrumentation)
    ├─→ storage (PostgreSQL + pgvector + KuzuDB)
    │     ├─→ cli
    │     ├─→ pdf-tools
    │     ├─→ extraction
    │     ├─→ api
    │     ├─→ dashboard
    │     ├─→ daemon
    │     ├─→ mcp-server
    │     └─→ client
    ├─→ pdf-tools
    ├─→ extraction
    └─→ s2-client
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
| **api** | FastAPI REST endpoints with health checks and metrics |
| **dashboard** | Streamlit visualization for search and graph exploration |
| **s2-client** | Semantic Scholar API client with rate limiting and caching |
| **daemon** | Low-latency query service via Unix socket (JSON-RPC 2.0) |
| **mcp-server** | Model Context Protocol server for Claude Code integration |
| **client** | DaemonClient SDK (JSON-RPC 2.0) with CLI fallback |

### Database Schema

**Core tables:** `sources`, `chunks`, `citations`, `source_citations`
**Knowledge graph:** `concepts`, `concept_relationships`, `chunk_concepts`, `methods`, `assumptions`
**Assumption auditing:** `method_assumption_cache`, `method_aliases` (Phase 4.1)

### KuzuDB Graph Engine

KuzuDB serves as the primary graph traversal engine, with PostgreSQL recursive CTEs as fallback:

- **Data**: `~/.research_kb/kuzu/research_kb.kuzu` (~110MB, mirrors PostgreSQL concepts/relationships)
- **Sync**: `python scripts/sync_kuzu.py` (run after ingestion/extraction)
- **Performance**: ~150ms batch scoring vs ~96s PostgreSQL CTEs
- **Fallback**: 2-second timeout on PostgreSQL path (`GRAPH_SCORE_TIMEOUT = 2.0`)
- **Code**: `packages/storage/src/research_kb_storage/kuzu_store.py` (749 lines)
- **Integration**: `graph_queries.py` tries KuzuDB first, PostgreSQL on failure

Key enums:
- `ConceptType`: METHOD, ASSUMPTION, PROBLEM, DEFINITION, THEOREM, CONCEPT, PRINCIPLE, TECHNIQUE, MODEL
- `RelationshipType`: REQUIRES, USES, ADDRESSES, GENERALIZES, SPECIALIZES, ALTERNATIVE_TO, EXTENDS

### Hybrid Search

4-way ranking combines text matching, semantic similarity, knowledge graph, and citation authority:

```
score = fts_weight × fts + vector_weight × vector + graph_weight × graph + citation_weight × citation
```

**Signals:**
- **FTS**: PostgreSQL full-text search (keyword matching)
- **Vector**: BGE-large cosine similarity (semantic matching)
- **Graph**: Concept co-occurrence boost (knowledge graph)
- **Citation**: PageRank-style authority score (citation network)

Context types adjust weights (base FTS + vector; graph and citation are **enabled by default** in CLI/MCP; disable with `--no-graph`/`--no-citations`):
- **building**: 20% FTS, 80% vector (favor semantic breadth)
- **auditing**: 50% FTS, 50% vector (favor precision)
- **balanced**: 30% FTS, 70% vector (default)

When `use_graph` or `use_citations` are active, each adds 15% weight (FTS/vector reduced proportionally).

### HyDE (Hypothetical Document Embeddings)

Optional query expansion that generates a hypothetical document to improve embedding quality for terse queries.

```python
from research_kb_storage import HydeConfig, get_hyde_embedding

# Configure HyDE (Ollama default, Anthropic for production)
config = HydeConfig(
    enabled=True,
    backend="ollama",  # or "anthropic"
    model="llama3.1:8b",  # or "claude-3-5-haiku-20241022"
    max_length=200,  # words in hypothetical document
)

# Get embedding using HyDE
embedding = await get_hyde_embedding("IV assumptions", config)
```

Benefits:
- 5-10% improvement on terse queries ("IV", "DML")
- Graceful fallback if LLM unavailable
- Configurable backend: Ollama (dev), Anthropic (prod)

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

1. **PR Checks** (<10 min): Unit + integration tests with mocked services, pytest-cov coverage reports (XML), doc freshness gate
2. **Weekly Integration** (15 min, manual trigger): Search pipeline + quality tests + script tests + doc freshness gate (`audit_docs.py`, `generate_status.py --check`)
3. **Full Rebuild** (45 min, manual trigger): Demo data load, embedding generation, retrieval eval against golden dataset with per-domain metrics (`weekly-full-rebuild.yml`)

## Data Protection

### Safe Docker Operations

**CRITICAL**: Use the safe docker wrapper to prevent accidental data loss:

```bash
# Recommended: Add alias to ~/.bashrc or ~/.zshrc
alias dc='./scripts/docker-safe.sh'

# Usage (intercepts dangerous operations)
dc down -v    # Warns, shows data counts, requires 'DELETE' confirmation
dc up -d      # Works normally
```

### Backups

- **Automatic**: Created before every extraction run (unless `--skip-backup`)
- **Manual**: `./scripts/backup_db.sh`
- **Location**: `backups/` directory (last 5 kept)

### Recovery

See [`docs/RECOVERY.md`](docs/RECOVERY.md) for detailed recovery procedures.

## Gotchas

- GROBID takes ~60s to start (healthcheck has 60s start_period)
- Graph search gracefully falls back to FTS+vector if concepts not extracted
- Table name is `concept_relationships` (not `relationships`)
- Uses `uv` workspaces for package resolution (`uv sync`); `pip install -e` still works as fallback
- **NEVER use `docker compose down -v`** without the safe wrapper — it deletes all data

## Documentation Protocol

When modifying code, update docs accordingly:

| Change Type | Required Doc Updates |
|-------------|---------------------|
| New CLI command | CLAUDE.md (CLI Usage), README.md |
| New package | Create README.md, add to CLAUDE.md architecture |
| New extraction backend | packages/extraction/README.md comparison table |
| External path change | docs/INTEGRATION.md |
| New API endpoint | Run `scripts/generate_package_docs.py` |
| Database schema | docs/phases/ relevant phase doc |

Run `python scripts/audit_docs.py` periodically to detect drift.

## Integration

research-kb exposes three integration surfaces:
- **Daemon service**: Unix socket at `/tmp/research_kb_daemon_${USER}.sock` (JSON-RPC 2.0)
- **REST API**: FastAPI at `http://localhost:8000` (Swagger at `/docs`)
- **MCP server**: Claude Code integration via Model Context Protocol

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for full details.

### Daemon Service

Low-latency query service via Unix domain socket (~200ms vector search, ~20ms health).

**Starting the daemon:**
```bash
# Direct start
research-kb-daemon

# Systemd user service
systemctl --user start research-kb-daemon
systemctl --user enable research-kb-daemon  # Auto-start on login
systemctl --user status research-kb-daemon
```

**Protocol:** JSON-RPC 2.0

**Methods:**
| Method | Description |
|--------|-------------|
| `search` | Hybrid search with optional graph/citation boosting |
| `fast_search` | Vector-only search for low latency (~200ms) |
| `graph_path` | Find path between concepts (KuzuDB accelerated) |
| `citation_info` | Get citation authority for sources |
| `health` | System health check (database, embed server, Kuzu, uptime) |
| `stats` | Database statistics |

**Example request:**
```bash
echo '{"jsonrpc":"2.0","method":"health","id":1}' | nc -U /tmp/research_kb_daemon_$USER.sock
```

**Pre-warming:**
The daemon pre-warms KuzuDB on startup to avoid 60s cold-start latency.
- Warming runs in background; `health` and `fast_search` work immediately
- Typical warming time: 5-15s (depends on page cache state)
- Skip with `--no-warm` flag for testing
- Monitor via `health` endpoint (`kuzu_warmup` field) or Prometheus gauge

**Installation:**
```bash
./scripts/install_daemon.sh install
```

### MCP Server (Claude Code Integration)

The `mcp-server` package exposes research-kb to Claude Code via MCP protocol.

**Available Tools (20 total):**
| Tool | Description |
|------|-------------|
| `research_kb_search` | Hybrid search (FTS + vector + graph + citation), optional HyDE via `use_hyde` |
| `research_kb_fast_search` | Fast vector-only search (~200ms), skips FTS/graph/citation/reranking |
| `research_kb_list_sources` | List sources (papers, textbooks) |
| `research_kb_get_source` | Get source details and chunks |
| `research_kb_get_source_citations` | Get citations for a source |
| `research_kb_get_citing_sources` | Find sources citing this one |
| `research_kb_get_cited_sources` | Find sources this one cites |
| `research_kb_citation_network` | Bidirectional citation network for a source |
| `research_kb_biblio_coupling` | Find similar sources by shared references |
| `research_kb_list_concepts` | List/search concepts |
| `research_kb_get_concept` | Get concept with relationships |
| `research_kb_chunk_concepts` | Get concepts linked to a chunk |
| `research_kb_find_similar_concepts` | Find semantically similar concepts |
| `research_kb_cross_domain_concepts` | Find concepts spanning multiple domains |
| `research_kb_graph_neighborhood` | Explore concept neighborhood |
| `research_kb_graph_path` | Find path between concepts (KuzuDB-accelerated) |
| `research_kb_list_domains` | List available knowledge domains |
| `research_kb_audit_assumptions` | Get required assumptions for a method (North Star; uses Anthropic backend) |
| `research_kb_stats` | Database statistics |
| `research_kb_health` | Health check (includes KuzuDB status) |

**Installation in Claude Code:**
```json
// In ~/.config/claude-code/config.json
{
  "mcpServers": {
    "research-kb": {
      "command": "research-kb-mcp",
      "args": []
    }
  }
}
```
