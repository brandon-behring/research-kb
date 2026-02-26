# Research Knowledge Base

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PR Checks](https://github.com/brandonmbehring-dev/research-kb/actions/workflows/pr-checks.yml/badge.svg)](https://github.com/brandonmbehring-dev/research-kb/actions/workflows/pr-checks.yml)

Graph-boosted semantic search for research literature.

Combines full-text search (BM25), vector similarity (BGE-large 1024d), knowledge graph traversal (KuzuDB), and citation authority scoring (PageRank) into a single ranked result set. Ships as a 20-tool MCP server for Claude Code, a CLI, a REST API, and a Streamlit dashboard.

## Features

- **4-signal hybrid search** -- BM25 + vector + knowledge graph + citation authority, with context-aware weight profiles
- **20-tool MCP server** -- plug into Claude Code for conversational access to search, graph exploration, citation networks, and assumption auditing
- **Knowledge graph** -- 307K concepts and 742K relationships extracted from research literature, served by KuzuDB
- **Citation authority** -- PageRank-style scoring over 15K+ citation links; bibliographic coupling for related-work discovery
- **Multi-domain** -- 19 corpus domains, 20 extraction prompt configs, extensible to new domains
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
python scripts/ingest_corpus.py            # Ingest PDFs from configured corpus directory
```

### 4. Search

```bash
research-kb search query "instrumental variables"
```

### 5. Start the MCP server (optional)

```bash
research-kb-mcp
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

Graph (15%) and citation (15%) signals are **enabled by default** in CLI and MCP interfaces. Disable with `--no-graph` / `--no-citations`. When active, FTS and vector weights are reduced proportionally.

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

<!-- AUTO-GEN:packages:START -->
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
| `daemon` | Low-latency query service (JSON-RPC 2.0) | asyncio, Unix socket |
| `mcp-server` | MCP tool server for Claude Code | MCP SDK |
| `dashboard` | Visual search + graph explorer | Streamlit |
| `s2-client` | Semantic Scholar API client | httpx, rate limiting |
| `client` | DaemonClient SDK (JSON-RPC 2.0) | Unix socket |
<!-- AUTO-GEN:packages:END -->

### Key Design Decisions

| Decision | Rationale | Alternative Rejected |
|----------|-----------|---------------------|
| BGE-large-en-v1.5 (1024d) | Single model ensures embedding consistency across 226K chunks | Multi-model (marginal quality gain, consistency cost) |
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
| Sources (papers, textbooks) | 485 |
| Text chunks (100% embedded) | 226K |
| Concepts (9 types) | 307K |
| Relationships | 742K |
| Citations | 15,166 |

<!-- AUTO-GEN:mcp-tools:START -->
## MCP Server

20 tools organized by function, designed for conversational use in Claude Code:

| Category | Tools | Description |
|----------|-------|-------------|
| **Search** | `research_kb_search`, `research_kb_fast_search` | Hybrid search with domain filtering and context profiles |
| **Sources** | `research_kb_list_sources`, `research_kb_get_source`, `research_kb_get_source_citations`, `research_kb_get_citing_sources`, `research_kb_get_cited_sources` | Browse corpus and citation links |
| **Concepts** | `research_kb_list_concepts`, `research_kb_get_concept`, `research_kb_chunk_concepts`, `research_kb_find_similar_concepts` | Search and inspect knowledge graph nodes |
| **Graph** | `research_kb_graph_neighborhood`, `research_kb_graph_path`, `research_kb_cross_domain_concepts` | Traverse concept relationships |
| **Citations** | `research_kb_citation_network`, `research_kb_biblio_coupling` | Upstream/downstream influence, bibliographic coupling |
| **Health** | `research_kb_health`, `research_kb_stats`, `research_kb_list_domains` | System status and corpus metrics |
| **Advanced** | `research_kb_audit_assumptions` | Method assumption extraction (uses Anthropic backend) |
<!-- AUTO-GEN:mcp-tools:END -->

## Testing

- **~2,400+ test functions** across 100+ test files
- **Tiered CI/CD**: PR checks (<10 min, with pytest-cov) -> Weekly integration (15 min, doc freshness gate) -> Full rebuild (45 min, demo data + embeddings + retrieval eval)
- **Golden evaluation dataset**: 177 queries across 14 domains with known-relevant chunks (benchmark)
- **Retrieval eval**: 55 YAML test cases with per-domain reporting (`--per-domain` flag)
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

<!-- AUTO-GEN:cli-commands:START -->
## CLI Reference

Full command reference with examples: [`docs/CLI.md`](docs/CLI.md)

Quick reference:

```bash
# Search and retrieval
research-kb search query "instrumental variables"        # Hybrid search (all 4 signals)
research-kb search query "IV" --no-graph                 # Disable graph signal
research-kb search query "IV" --no-citations             # Disable citation authority

# Source management
research-kb sources list                                 # List sources
research-kb sources stats                                # Database statistics
research-kb sources extraction-status                    # Extraction pipeline stats

# Knowledge graph
research-kb graph concepts "IV"                          # Concept search
research-kb graph neighborhood "double machine learning" # Graph exploration
research-kb graph path "IV" "unconfoundedness"           # Shortest path

# Citation network
research-kb citations list <source>                      # List citations from a source
research-kb citations cited-by <source>                  # Find sources citing this one
research-kb citations cites <source>                     # Find sources this one cites
research-kb citations stats                              # Corpus citation statistics
research-kb citations similar <source>                   # Find similar sources (shared refs)

# Assumption auditing (North Star feature)
research-kb search audit-assumptions "IV"                # Get required assumptions
research-kb search audit-assumptions "IV" --no-ollama    # Graph-only (no LLM fallback)
research-kb search audit-assumptions "DML" --format json # JSON output

# Semantic Scholar discovery
research-kb discover search "causal inference"           # Search S2 for papers
research-kb discover topics                              # Browse by topic
research-kb discover author "Chernozhukov"               # Find by author
research-kb enrich citations                             # Enrich with S2 metadata
research-kb enrich status                                # Show enrichment status
research-kb enrich job-status                            # Check enrichment job status
```
<!-- AUTO-GEN:cli-commands:END -->

## Multi-Domain Support

research-kb supports 19 corpus domains with 20 extraction prompt configurations:

| Domain | Sources | Description |
|--------|---------|-------------|
| `causal_inference` | 89 | Causal inference, structural models, treatment effects |
| `rag_llm` | 73 | Retrieval-augmented generation, language models |
| `time_series` | 48 | Time series analysis, forecasting, temporal methods |
| `software_engineering` | 30 | Design patterns, testing, architecture, DevOps |
| `deep_learning` | 35 | Neural networks, transformers, optimization |
| `econometrics` | 35 | Econometric theory and estimation |
| `mathematics` | 28 | Pure and applied mathematics |
| `interview_prep` | 23 | Technical interview preparation |
| `finance` | 23 | Quantitative finance and risk |
| `machine_learning` | 14 | General ML algorithms and theory |
| `statistics` | 18 | Statistical theory and methods |
| `ml_engineering` | 17 | ML systems, MLOps, production ML |
| `data_science` | 12 | Data analysis and visualization |
| `portfolio_management` | 11 | Portfolio theory and optimization |
| `functional_programming` | 8 | FP concepts and languages |
| `algorithms` | 12 | Algorithm design and analysis |
| `forecasting` | 5 | Forecasting methods and evaluation |
| `fitness` | 2 | Exercise science and training |
| `economics` | 2 | Economic theory |

All search, concept extraction, and graph operations support domain filtering via the `--domain` flag.

### Adding Your Own Domain

See the full tutorial: [`docs/tutorial_new_domain.md`](docs/tutorial_new_domain.md)

Quick version:

1. Create a SQL migration to register the domain
2. (Optional) Configure domain-specific prompts in `domain_prompts.py`
3. Ingest PDFs: `python scripts/ingest_corpus.py --domain <name>`
4. Extract concepts: `python scripts/extract_concepts.py --domain <name>`
5. Sync to KuzuDB: `python scripts/sync_kuzu.py`

## Development

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

# MCP server
research-kb-mcp

# API server
uvicorn research_kb_api.main:app --port 8000

# Dashboard
streamlit run packages/dashboard/src/research_kb_dashboard/app.py
```

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) | Architecture, package dependencies, schema |
| [`docs/design/latency_analysis.md`](docs/design/latency_analysis.md) | 85s -> 2.1s graph optimization story |
| [`docs/design/rrf_validation.md`](docs/design/rrf_validation.md) | Weighted sum vs. RRF empirical comparison |
| [`docs/SLO.md`](docs/SLO.md) | Service level objectives |
| [`docs/CLI.md`](docs/CLI.md) | Full CLI command reference |
| [`docs/tutorial_new_domain.md`](docs/tutorial_new_domain.md) | Step-by-step guide to adding a new domain |

## Ecosystem

Part of the **Rigorous AI Engineering** ecosystem:

| Project | Description |
|---------|-------------|
| **research-kb** (this repo) | Graph-boosted semantic search for research literature |
| [ir-eval](https://github.com/brandonmbehring-dev/ir-eval) | Statistical retrieval evaluation with drift detection |
| [temporalcv](https://github.com/brandonmbehring-dev/temporalcv) | Temporal cross-validation with leakage detection |

research-kb's 177-query golden evaluation dataset is used by ir-eval for retrieval quality benchmarking and regression detection.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT
