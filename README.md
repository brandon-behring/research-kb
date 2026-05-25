# Research Knowledge Base

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PR Checks](https://github.com/brandon-behring/research-kb/actions/workflows/pr-checks.yml/badge.svg)](https://github.com/brandon-behring/research-kb/actions/workflows/pr-checks.yml)

Graph-boosted semantic search for research literature.

Combines full-text search (BM25), vector similarity (BGE-large 1024d), and citation authority scoring (PageRank) into a single ranked result set. Knowledge graph traversal (KuzuDB, 310K concepts) is available but disabled by default while chunk-concept links are being rebuilt. Ships as a 22-tool MCP server for Claude Code, a CLI, a REST API, and a Streamlit dashboard.

## Features

- **3-signal hybrid search** -- BM25 + vector + citation authority, with context-aware weight profiles (knowledge graph available via `--graph` flag, disabled by default pending KG re-extraction)
- **22-tool MCP server** -- plug into Claude Code for conversational access to search, graph exploration, citation networks, assumption auditing, and concept synthesis
- **Knowledge graph** -- 310K concepts and 744K relationships extracted from research literature, served by KuzuDB
- **Citation authority** -- PageRank-style scoring over 45K+ citation links; bibliographic coupling for related-work discovery
- **Multi-domain** -- 36 corpus domains, 20 extraction prompt configs, extensible to new domains
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

Then add to your Claude Code MCP config to access all 22 tools from conversation.

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

Citation (15%) signals are **enabled by default**. Graph is **disabled by default** while knowledge graph chunk links are being rebuilt — enable with `--graph` flag. When active, FTS and vector weights are reduced proportionally.

## Architecture

```
┌───────────────────────────────────────────────────┐
│  Interfaces                                       │
│  ┌─────┐  ┌─────────┐  ┌─────┐  ┌───────────┐   │
│  │ CLI │  │MCP (22) │  │ API │  │ Dashboard │   │
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
| BGE-large-en-v1.5 (1024d) | Single model ensures embedding consistency across 1M+ chunks | Multi-model (marginal quality gain, consistency cost) |
| KuzuDB embedded graph | Solved O(N*M) recursive CTE bottleneck: 85s -> 2.1s | PostgreSQL-only graph (too slow at scale) |
| Weighted sum over RRF | Validated 5-1 superiority on golden dataset | Reciprocal Rank Fusion (loses magnitude signal) |
| asyncpg connection pooling | Handles concurrent MCP + API + CLI requests | Synchronous psycopg2 (blocks on I/O) |
| JSONB metadata columns | Extensible without schema migrations | Rigid columns (migration overhead) |

## Performance

### Retrieval Quality

108 YAML test cases across 36 domains with known-relevant chunks (`fixtures/eval/retrieval_test_cases.yaml`). Run `python scripts/eval_retrieval.py --per-domain` for current metrics.

Core domains with 5+ test cases (causal_inference, econometrics, statistics) average MRR > 0.85 individually. Thin domains with few test cases pull down aggregate metrics.

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

Run `python scripts/generate_status.py` for current metrics. See [`docs/status/CURRENT_STATUS.md`](docs/status/CURRENT_STATUS.md) for the latest auto-generated snapshot.

<!-- AUTO-GEN:mcp-tools:START -->
## MCP Server

22 tools organized by function, designed for conversational use in Claude Code:

| Tool | Description |
|------|-------------|
| `research_kb_audit_assumptions` | Get required assumptions for a statistical/ML method. |
| `research_kb_citation_network` | Get bidirectional citation network for a source. |
| `research_kb_biblio_coupling` | Find sources similar by bibliographic coupling. |
| `research_kb_list_concepts` | List or search concepts in the knowledge graph. |
| `research_kb_get_concept` | Get detailed information about a specific concept. |
| `research_kb_chunk_concepts` | Get all concepts linked to a specific chunk. |
| `research_kb_find_similar_concepts` | Find concepts semantically similar to a given concept. |
| `research_kb_graph_neighborhood` | Explore the neighborhood of a concept in the knowledge graph. |
| `research_kb_graph_path` | Find the shortest path between two concepts. |
| `research_kb_cross_domain_concepts` | Find equivalent or related concepts across knowledge domains. |
| `research_kb_explain_connection` | Explain how two concepts connect through the knowledge graph with evidence. |
| `research_kb_stats` | Get statistics about the research knowledge base. |
| `research_kb_health` | Check the health of the research-kb system. |
| `research_kb_list_domains` | List available knowledge domains and their statistics. |
| `research_kb_literature_review` | Generate a structured literature review for a topic from the knowledge base. |
| `research_kb_search` | Search the research knowledge base across multiple domains. |
| `research_kb_fast_search` | Fast vector-only search (~200ms). Skips FTS, graph, citation, reranking. |
| `research_kb_list_sources` | List sources (papers and textbooks) in the knowledge base. |
| `research_kb_get_source` | Get detailed information about a specific source. |
| `research_kb_get_source_citations` | Get citation relationships for a source. |
| `research_kb_get_citing_sources` | Find all sources that cite a given source. |
| `research_kb_get_cited_sources` | Find all sources that a given source cites. |
<!-- AUTO-GEN:mcp-tools:END -->

## Testing

- **~2,700+ test functions** across 111 test files
- **Tiered CI/CD**: PR checks (<10 min, pytest-cov 70% gate) -> Manual integration (15 min, doc freshness gate) -> Full rebuild (45 min, demo data + embeddings + retrieval eval)
- **Retrieval eval**: 108 YAML test cases across 36 domains with per-domain reporting (`--per-domain` flag, MRR >= 0.85 CI gate on core domains)
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
research-kb search audit-assumptions                   # Get required assumptions for a statistical/ML method.
research-kb search query                               # Search the research knowledge base with hybrid search and reranking.

research-kb graph concepts                             # Search for concepts in the knowledge graph.
research-kb graph neighborhood                         # Visualize concept neighborhood in the knowledge graph.
research-kb graph path                                 # Find shortest path between two concepts in the knowledge graph.
research-kb graph explain                              # Explain how two concepts are connected with evidence and synthesis.
research-kb graph export                               # Export a topic-filtered graph as JSON for downstream viz consumers.

research-kb citations list                             # List citations extracted from a source.
research-kb citations cited-by                         # Find sources that cite a given source.
research-kb citations cites                            # Find sources that a given source cites.
research-kb citations stats                            # Show corpus-wide citation graph statistics.
research-kb citations similar                          # Find sources with similar research focus via bibliographic coupling.

research-kb sources list                               # List all ingested sources in the knowledge base.
research-kb sources extraction-status                  # Show extraction pipeline statistics.
research-kb sources stats                              # Show knowledge base statistics.

research-kb discover search                            # Search for papers on Semantic Scholar.
research-kb discover topics                            # Discover papers for all pre-configured research topics.
research-kb discover author                            # Get recent papers by a specific author.

research-kb enrich citations                           # Enrich citations with Semantic Scholar metadata.
research-kb enrich status                              # Show citation enrichment status.
research-kb enrich job-status                          # Check status of enrichment jobs (running or completed).

research-kb review generate                            # Generate a structured literature review for a topic.
```
<!-- AUTO-GEN:cli-commands:END -->

## Multi-Domain Support

research-kb supports 36 corpus domains with 20 extraction prompt configurations:

| Domain | Sources | Description |
|--------|---------|-------------|
| `machine_learning` | 352 | General ML algorithms and theory |
| `software_engineering` | 164 | Design patterns, testing, architecture, DevOps |
| `deep_learning` | 152 | Neural networks, transformers, optimization |
| `causal_inference` | 131 | Causal inference, structural models, treatment effects |
| `rag_llm` | 113 | Retrieval-augmented generation, language models |
| `time_series` | 72 | Time series analysis, forecasting, temporal methods |
| `mathematics` | 69 | Pure and applied mathematics |
| `finance` | 65 | Quantitative finance and risk |
| `numerical_methods` | 52 | Numerical analysis and computational methods |
| `linear_algebra` | 47 | Linear algebra and matrix theory |
| `econometrics` | 44 | Econometric theory and estimation |
| `statistics` | 40 | Statistical theory and methods |
| `data_science` | 36 | Data analysis and visualization |
| `healthcare` | 35 | Healthcare analytics and clinical data |
| `algorithms` | 32 | Algorithm design and analysis |
| `ml_engineering` | 30 | ML systems, MLOps, production ML |
| `actuarial_insurance` | 25 | Actuarial science and insurance modeling |
| `functional_programming` | 25 | FP concepts and languages |
| `dynamical_systems` | 23 | Dynamical systems and control theory |
| `fitness` | 22 | Exercise science and training |
| `biology_neuroscience` | 21 | Biology, neuroscience, computational models |
| `probability_theory` | 21 | Probability theory and stochastic processes |
| `interview_prep` | 20 | Technical interview preparation |
| `portfolio_management` | 20 | Portfolio theory and optimization |
| `reinforcement_learning` | 19 | Reinforcement learning and decision processes |
| `analysis` | 18 | Real and functional analysis |
| `optimization` | 18 | Mathematical optimization and operations research |
| `algebra` | 18 | Abstract algebra and algebraic structures |
| `physics` | 16 | Physics and mathematical physics |
| `topology_geometry` | 16 | Topology, geometry, and manifolds |
| `sql` | 14 | SQL, databases, query optimization |
| `signal_processing` | 11 | Signal processing and spectral methods |
| `recommender_systems` | 5 | Recommender systems, collaborative filtering |
| `forecasting` | 5 | Forecasting methods and evaluation |
| `adtech` | 3 | Advertising technology, auction mechanisms |
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
