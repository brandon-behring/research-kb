# Research Knowledge Base

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PR Checks](https://github.com/brandonmbehring-dev/research-kb/actions/workflows/pr-checks.yml/badge.svg)](https://github.com/brandonmbehring-dev/research-kb/actions/workflows/pr-checks.yml)
[![Weekly Full Rebuild](https://github.com/brandonmbehring-dev/research-kb/actions/workflows/weekly-full-rebuild.yml/badge.svg)](https://github.com/brandonmbehring-dev/research-kb/actions/workflows/weekly-full-rebuild.yml)

Graph-boosted semantic search for research literature.

Combines full-text search (BM25), vector similarity (BGE-large 1024d), knowledge graph traversal (KuzuDB), and citation authority scoring (PageRank) into a single ranked result set. Ships as a 20-tool MCP server for Claude Code, a CLI, a REST API, and a Streamlit dashboard.

## Features

- **4-signal hybrid search** -- BM25 + vector + knowledge graph + citation authority, with context-aware weight profiles
- **20-tool MCP server** -- plug into Claude Code for conversational access to search, graph exploration, citation networks, and assumption auditing
- **Knowledge graph** -- 307K concepts and 742K relationships extracted from research literature, served by KuzuDB
- **Citation authority** -- PageRank-style scoring over 15K+ citation links; bibliographic coupling for related-work discovery
- **Multi-domain** -- causal inference, time series, RAG/LLM, and extensible to new domains
- **Demo corpus** -- ships with scripts to download and ingest open-access arXiv papers
- **Production monitoring** -- SLOs, Prometheus metrics, structured logging, health checks

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- Ollama (optional, for concept extraction)

### 1. Start infrastructure

```bash
docker-compose up -d   # PostgreSQL + pgvector
```

> Schema is auto-applied on first container creation. For an existing database:
> `psql -h localhost -U postgres -d research_kb -f packages/storage/schema.sql`

### 2. Install packages

```bash
pip install -e packages/contracts \
            -e packages/common \
            -e packages/storage \
            -e packages/pdf-tools \
            -e packages/cli
```

### 3. Set up demo corpus

**Option A -- Pre-built fixtures (fast, no downloads):**

```bash
python scripts/load_demo_data.py          # Load 9 causal-inference papers + concepts
python scripts/sync_kuzu.py               # Sync concepts to KuzuDB (enables graph search)
python -m research_kb_pdf.embed_server &   # Start embedding server
python scripts/embed_missing.py            # Generate embeddings (~5 min on CPU)
```

**Option B -- Full pipeline (downloads from arXiv):**

```bash
python scripts/setup_demo.py               # Download + ingest 25 open-access papers
```

**Option C -- Bring your own PDFs:**

```bash
research-kb ingest <your-pdfs-directory>
```

### 4. Search

```bash
research-kb query "instrumental variables"
```

### 5. Start the MCP server (optional)

```bash
research-kb-mcp serve
```

Then add to your Claude Code MCP config to access all 20 tools from conversation.

## How It Works

### Search Pipeline

```
Query
  |
  +---> Embed (BGE-large-en-v1.5, 1024d)
  |
  +---> Execute in parallel:
  |       FTS (PostgreSQL ts_rank)
  |       Vector (pgvector cosine similarity)
  |       Graph (KuzuDB concept traversal)
  |       Citation (PageRank authority)
  |
  +---> Weighted fusion
  |       score = w_fts * BM25 + w_vec * cosine + w_graph * graph + w_cite * pagerank
  |
  +---> Cross-encoder rerank (optional)
  |
  +---> Return top-K results
```

### Context-Aware Weights

The weight profile adapts to the search intent:

| Context | FTS | Vector | Graph | Citation | Use Case |
|---------|-----|--------|-------|----------|----------|
| `building` | 20% | 80% | -- | -- | Broad research -- cast a wide semantic net |
| `auditing` | 50% | 50% | -- | -- | Precise lookup -- keyword accuracy matters |
| `balanced` | 30% | 70% | -- | -- | Default -- good general performance |

Graph (15%) and citation (15%) signals are opt-in via `use_graph` and `use_citations` flags. When enabled, FTS and vector weights are reduced proportionally.

## Architecture

```
┌───────────────────────────────────────────────────┐
│  Interfaces                                       │
│  ┌─────┐  ┌─────────┐  ┌─────┐  ┌───────────┐   │
│  │ CLI │  │MCP (20) │  │ API │  │ Dashboard │   │
│  └──┬──┘  └────┬────┘  └──┬──┘  └─────┬─────┘   │
│     └──────────┴──────────┴────────────┘          │
│                     │                              │
│  ┌──────────────────┴──────────────────────────┐  │
│  │           Storage Layer                      │  │
│  │  SourceStore · ChunkStore · ConceptStore     │  │
│  │  CitationStore · KuzuStore                   │  │
│  │  HybridSearch (4-signal fusion)              │  │
│  └──────────────────┬──────────────────────────┘  │
│                     │                              │
│  ┌──────────────────┴──────────────────────────┐  │
│  │     PostgreSQL + pgvector  |  KuzuDB        │  │
│  │     (FTS, vectors, schema) | (graph engine) │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

### Packages

| Package | Purpose | Key Technology |
|---------|---------|----------------|
| `contracts` | Shared data models | Pydantic |
| `common` | Logging, retry, instrumentation | structlog, tenacity |
| `storage` | Database + search orchestration | asyncpg, pgvector, KuzuDB |
| `pdf-tools` | PDF extraction + embedding | PyMuPDF, GROBID, BGE-large |
| `extraction` | Concept extraction from text | Ollama LLM |
| `cli` | Command-line interface | Typer |
| `api` | REST endpoints + health checks | FastAPI |
| `daemon` | Background service (embedding, sync) | asyncio |
| `mcp-server` | MCP tool server for Claude Code | MCP SDK |
| `dashboard` | Visual search + graph explorer | Streamlit |
| `s2-client` | Semantic Scholar API client | httpx, rate limiting |
| `client` | Python client library | httpx |

### Key Design Decisions

| Decision | Rationale | Alternative Rejected |
|----------|-----------|---------------------|
| BGE-large-en-v1.5 (1024d) | Single model ensures embedding consistency across 178K chunks | Multi-model (marginal quality gain, consistency cost) |
| KuzuDB embedded graph | Solved O(N*M) recursive CTE bottleneck: 85s -> 2.1s | PostgreSQL-only graph (too slow at scale) |
| Weighted sum over RRF | Validated 5-1 superiority on golden dataset | Reciprocal Rank Fusion (loses magnitude signal) |
| asyncpg connection pooling | Handles concurrent MCP + API + CLI requests | Synchronous psycopg2 (blocks on I/O) |
| JSONB metadata columns | Extensible without schema migrations | Rigid columns (migration overhead) |

## Performance

### Retrieval Quality

Evaluated on a golden dataset of 177 queries across 14 domains with known-relevant chunks:

| Metric | Score |
|--------|-------|
| Hit Rate@K | 92.9% |
| MRR | 0.849 |
| NDCG@5 | 0.823 |

### Latency

| Operation | p50 | p95 |
|-----------|-----|-----|
| Health check | 20ms | 22ms |
| Vector search (fast path) | 208ms | 212ms |
| Graph-boosted search (warm) | 2.1s | -- |
| Graph path query (KuzuDB) | 3.1s | 5.8s |

The graph-boosted warm latency of 2.1s represents a **40x improvement** from the pre-KuzuDB architecture (85s). Full optimization story: [`docs/design/latency_analysis.md`](docs/design/latency_analysis.md).

### Corpus Scale

| Dimension | Count |
|-----------|-------|
| Sources (papers, textbooks) | 452 |
| Text chunks (100% embedded) | 178K |
| Concepts (9 types) | 307K |
| Relationships | 742K |
| Citations | 15,869 |

## MCP Server

20 tools organized by function, designed for conversational use in Claude Code:

| Category | Tools | Description |
|----------|-------|-------------|
| **Search** | `research_kb_search`, `research_kb_fast_search` | Hybrid search with domain filtering and context profiles |
| **Sources** | `list_sources`, `get_source`, `get_source_citations`, `get_citing_sources`, `get_cited_sources` | Browse corpus and citation links |
| **Concepts** | `list_concepts`, `get_concept`, `chunk_concepts`, `find_similar_concepts` | Search and inspect knowledge graph nodes |
| **Graph** | `graph_neighborhood`, `graph_path`, `cross_domain_concepts` | Traverse concept relationships |
| **Citations** | `citation_network`, `biblio_coupling` | Upstream/downstream influence, bibliographic coupling |
| **Health** | `health`, `stats`, `list_domains` | System status and corpus metrics |
| **Advanced** | `audit_assumptions` | Method assumption extraction with docstring generation |

## Testing

- **~2,040 test functions** across 90+ test files
- **Tiered CI/CD**: PR checks (<10 min, with pytest-cov) -> Weekly integration (10 min, doc freshness gate) -> Full rebuild (45 min, demo data + embeddings + retrieval eval)
- **Golden evaluation dataset**: 177 queries across 14 domains with known-relevant chunks
- **Retrieval eval**: 55 test cases with per-domain reporting (`--per-domain` flag)
- **RRF validation study**: Weighted sum vs. Reciprocal Rank Fusion ([`docs/design/rrf_validation.md`](docs/design/rrf_validation.md))

```bash
# Run all tests
pytest

# Run by package
pytest packages/storage/tests/ -v
pytest packages/mcp-server/tests/ -v

# Run with markers
pytest -m "unit"
```

## CLI Reference

Full command reference with examples: [`docs/CLI.md`](docs/CLI.md)

Quick reference:

```bash
research-kb query "instrumental variables"        # Hybrid search
research-kb sources                               # List sources
research-kb stats                                 # Database statistics
research-kb concepts "IV"                         # Concept search
research-kb graph "double machine learning"       # Graph exploration
research-kb path "IV" "unconfoundedness"          # Shortest path
research-kb audit-assumptions "IV"                # Assumption auditing
research-kb discover search "causal inference"    # S2 discovery
research-kb enrich citations                      # Enrich with S2 metadata
```

## Development

### Adding a New Domain

1. Create a domain config in `packages/storage/src/research_kb_storage/domains/`
2. Ingest domain-specific PDFs: `research-kb ingest --domain <name> <pdf-dir>`
3. Extract concepts: `python scripts/extract_concepts.py --domain <name>`
4. Sync to KuzuDB: `python scripts/sync_kuzu.py`

### Extending the MCP Server

1. Create a tool module in `packages/mcp-server/src/research_kb_mcp/tools/`
2. Implement tools with `@mcp.tool()` decorators
3. Register in `tools/__init__.py`
4. Add tests in `packages/mcp-server/tests/`

### Running the Full Stack

```bash
# Infrastructure
docker-compose up -d

# Embedding server
python -m research_kb_pdf.embed_server

# MCP daemon
research-kb-mcp serve

# API server
uvicorn research_kb_api.app:app --port 8000

# Dashboard
streamlit run packages/dashboard/app.py
```

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) | Architecture, package dependencies, schema |
| [`docs/design/latency_analysis.md`](docs/design/latency_analysis.md) | 85s -> 2.1s graph optimization story |
| [`docs/design/rrf_validation.md`](docs/design/rrf_validation.md) | Weighted sum vs. RRF empirical comparison |
| [`docs/SLO.md`](docs/SLO.md) | Service level objectives |
| [`docs/CLI.md`](docs/CLI.md) | Full CLI command reference |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT
