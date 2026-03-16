# Research Knowledge Base

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PR Checks](https://github.com/brandon-behring/research-kb/actions/workflows/pr-checks.yml/badge.svg)](https://github.com/brandon-behring/research-kb/actions/workflows/pr-checks.yml)

Graph-boosted semantic search for research literature.

Combines full-text search (BM25), vector similarity (BGE-large 1024d), knowledge graph traversal (KuzuDB), and citation authority scoring (PageRank) into a single ranked result set. Ships as a 21-tool MCP server for Claude Code, a CLI, a REST API, and a Streamlit dashboard.

## Features

- **4-signal hybrid search** -- BM25 + vector + knowledge graph + citation authority, with context-aware weight profiles
- **21-tool MCP server** -- plug into Claude Code for conversational access to search, graph exploration, citation networks, assumption auditing, and concept synthesis
- **Knowledge graph** -- 310K concepts and 744K relationships extracted from research literature, served by KuzuDB
- **Citation authority** -- PageRank-style scoring over 15K+ citation links; bibliographic coupling for related-work discovery
- **Multi-domain** -- 22 corpus domains, 20 extraction prompt configs, extensible to new domains
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

Then add to your Claude Code MCP config to access all 21 tools from conversation.

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
│  │ CLI │  │MCP (21) │  │ API │  │ Dashboard │   │
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

| Package | Purpose |
|---------|---------|
| `api` | FastAPI REST API for research-kb semantic search |
| `cli` | CLI for querying the research knowledge base |
| `client` | Python client SDK for research-kb daemon and CLI |
| `common` | Common utilities for research-kb system (logging, retry, instrumentation) |
| `contracts` | Pure Pydantic schemas for research-kb system |
| `daemon` | Low-latency daemon for research-kb queries via Unix socket |
| `dashboard` | Streamlit visualization dashboard for research-kb |
| `extraction` | Concept extraction for research knowledge base using Ollama LLM |
| `mcp-server` | MCP server exposing research-kb knowledge base to Claude Code |
| `pdf-tools` | PDF extraction and chunking for research-kb system |
| `s2-client` | Semantic Scholar API client for research-kb (async, rate-limited, cached) |
| `storage` | PostgreSQL storage layer for research-kb system |
<!-- AUTO-GEN:packages:END -->

### Key Design Decisions

| Decision | Rationale | Alternative Rejected |
|----------|-----------|---------------------|
| BGE-large-en-v1.5 (1024d) | Single model ensures embedding consistency across 228K chunks | Multi-model (marginal quality gain, consistency cost) |
| KuzuDB embedded graph | Solved O(N*M) recursive CTE bottleneck: 85s -> 2.1s | PostgreSQL-only graph (too slow at scale) |
| Weighted sum over RRF | Validated 5-1 superiority on golden dataset | Reciprocal Rank Fusion (loses magnitude signal) |
| asyncpg connection pooling | Handles concurrent MCP + API + CLI requests | Synchronous psycopg2 (blocks on I/O) |
| JSONB metadata columns | Extensible without schema migrations | Rigid columns (migration overhead) |

## Performance

### Retrieval Quality

Evaluated on 98 YAML test cases across 20 domains with known-relevant chunks (`fixtures/eval/retrieval_test_cases.yaml`):

| Metric | Full Corpus (98 cases) |
|--------|------------------------|
| Hit Rate@K | 91.8% |
| MRR | 0.729 |
| NDCG@5 | 0.714 |

The full-corpus MRR reflects the inclusion of 15 interview_prep cases (Phase AE) and thin domains (finance, sql, machine_learning) with few test cases. Core domains with 5+ test cases (causal_inference, econometrics, statistics, interview_prep) average higher MRR individually.

> CI gate: `--fail-below 0.85` scoped to core domains via `--gate-domains` in `weekly-full-rebuild.yml`. A deprecated 177-query JSON benchmark exists in `fixtures/eval/` for historical reference.

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
| Sources (papers, textbooks) | 495 |
| Text chunks (100% embedded) | 228K |
| Concepts (9 types) | 310K |
| Relationships | 744K |
| Citations | 15,166 |

<!-- AUTO-GEN:mcp-tools:START -->
## MCP Server

9 tools organized by function, designed for conversational use in Claude Code:

| Tool | Description |
|------|-------------|
| `research_kb_chunk_concepts` | Get all concepts linked to a specific chunk. |
| `research_kb_find_similar_concepts` | Find concepts semantically similar to a given concept. |
| `research_kb_stats` | Get statistics about the research knowledge base. |
| `research_kb_health` | Check the health of the research-kb system. |
| `research_kb_list_domains` | List available knowledge domains and their statistics. |
| `research_kb_get_source` | Get detailed information about a specific source. |
| `research_kb_get_source_citations` | Get citation relationships for a source. |
| `research_kb_get_citing_sources` | Find all sources that cite a given source. |
| `research_kb_get_cited_sources` | Find all sources that a given source cites. |
<!-- AUTO-GEN:mcp-tools:END -->

## Testing

- **~2,700+ test functions** across 111 test files
- **Tiered CI/CD**: PR checks (<10 min, pytest-cov 70% gate) -> Manual integration (15 min, doc freshness gate) -> Full rebuild (45 min, demo data + embeddings + retrieval eval)
- **Retrieval eval**: 98 YAML test cases across 20 domains with per-domain reporting (`--per-domain` flag, MRR >= 0.85 CI gate on core domains)
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
research-kb enrich status                              # Show citation enrichment status.
```
<!-- AUTO-GEN:cli-commands:END -->

## Multi-Domain Support

research-kb supports 22 corpus domains with 20 extraction prompt configurations:

| Domain | Sources | Description |
|--------|---------|-------------|
| `causal_inference` | 89 | Causal inference, structural models, treatment effects |
| `rag_llm` | 76 | Retrieval-augmented generation, language models |
| `time_series` | 48 | Time series analysis, forecasting, temporal methods |
| `deep_learning` | 35 | Neural networks, transformers, optimization |
| `econometrics` | 35 | Econometric theory and estimation |
| `software_engineering` | 30 | Design patterns, testing, architecture, DevOps |
| `mathematics` | 28 | Pure and applied mathematics |
| `finance` | 23 | Quantitative finance and risk |
| `interview_prep` | 23 | Technical interview preparation |
| `statistics` | 18 | Statistical theory and methods |
| `ml_engineering` | 17 | ML systems, MLOps, production ML |
| `machine_learning` | 14 | General ML algorithms and theory |
| `algorithms` | 12 | Algorithm design and analysis |
| `data_science` | 12 | Data analysis and visualization |
| `portfolio_management` | 11 | Portfolio theory and optimization |
| `functional_programming` | 8 | FP concepts and languages |
| `forecasting` | 5 | Forecasting methods and evaluation |
| `recommender_systems` | 3 | Recommender systems, collaborative filtering |
| `sql` | 2 | SQL, databases, query optimization |
| `adtech` | 2 | Advertising technology, auction mechanisms |
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
| [ir-eval](https://github.com/brandon-behring/ir-eval) | Statistical retrieval evaluation with drift detection |
| [temporalcv](https://github.com/brandon-behring/temporalcv) | Temporal cross-validation with leakage detection |

research-kb's retrieval evaluation dataset is used by ir-eval for retrieval quality benchmarking and regression detection.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT
