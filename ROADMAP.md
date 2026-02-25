# Research KB Roadmap

A semantic search system for research literature with graph-boosted retrieval. Combines full-text search, vector similarity (BGE-large-en-v1.5, 1024 dimensions), knowledge graph signals (KuzuDB), and citation authority scoring.

## Phase 1: Foundation (Weeks 1-2) ✅ COMPLETE

- PostgreSQL + pgvector (1024-dim embeddings)
- PDF extraction (PyMuPDF + GROBID)
- Hybrid search (FTS + vector, configurable weights)
- CLI: `research-kb query`, `stats`, `sources`

## Phase 2: Knowledge Graph (Weeks 3-4) ✅ COMPLETE

- Concept extraction with LLM (Ollama llama3.1:8b or Claude Haiku)
- Method & assumption database
- Relationship ontology (REQUIRES, USES, ADDRESSES, etc.)
- Graph traversal queries (SQL recursive CTEs)
- Hybrid retrieval (vector + FTS + graph signals)

## Phase 3: Enhanced Retrieval (Weeks 5-6) ✅ COMPLETE

- Query expansion with concept synonyms (synonym_map.json)
- Cross-encoder re-ranking (BGE reranker)
- Citation graph integration (5,044 citations, 275 internal edges)
- 4-way hybrid search (FTS + vector + graph + citation)

## Phase 4: API & Dashboard (Weeks 7-8) ✅ COMPLETE

- FastAPI REST API with health checks and metrics
- Streamlit + PyVis dashboard
- Citation network visualization
- Concept graph explorer with N-hop neighborhoods
- MCP server for Claude Code integration (20 tools)

## Phase D: Observability ✅ COMPLETE

- KuzuDB graph engine (replaces PostgreSQL recursive CTEs for graph traversal)
- Prometheus metrics on port 9001
- Grafana dashboard for daemon monitoring
- Systemd timer for daily KuzuDB sync (3 AM)
- Benchmark suite: fast_search 208ms, graph_path 1.7-5.8s

## Phase E: RAG/LLM Extraction ✅ COMPLETE

- 23,307 RAG/LLM concepts extracted via Claude Haiku 4.5 (50 min, ~$30)
- KuzuDB sync: 307K concepts, 742K relationships
- 87.4% coverage across RAG/LLM domain

## Phase F: Cross-Repo Integration ✅ COMPLETE

- Lever of Archimedes health monitoring
- Interview readiness evaluation
- Glossary validation pipeline

## Phase G: Repository Hygiene ✅ COMPLETE

- Pytest consolidation (~2,040 tests with markers)
- Pre-commit hooks (black, ruff, mypy)
- Scripts archive for one-off utilities
- CI marker integration (unit, integration, requires_*)

## Phase H: Multi-Domain Extraction ✅ COMPLETE

- 9 new domain prompt configs (14 total: econometrics, software_engineering, deep_learning, mathematics, machine_learning, finance, statistics, ml_engineering, data_science)
- 162 domain prompt tests
- Domain-specific concept type guidance for higher extraction quality

## Phase I: CI Hardening & Coverage ✅ COMPLETE

- pytest-cov in PR checks with XML reports
- Doc freshness gate in weekly integration workflow
- ROADMAP and INDEX documentation refresh

## Phase J: Retrieval Quality Eval Expansion ✅ COMPLETE

- Retrieval test case expansion (34 → 55 cases across 9 domains)
- Per-domain eval reporting (`--per-domain` flag on eval_retrieval.py)
- Method assumption expansion (20 → 29 methods: added ARIMA, GARCH, VAR, LASSO, random forest, SVM, MCMC, EM, KDE)

## Phase K: Documentation Consolidation ✅ COMPLETE

- CLAUDE.md accuracy pass (CI tiers updated, counts corrected)
- README.md CLI quick reference added (zero audit_docs.py warnings)
- MEMORY.md refresh with post-Phase-K metrics
- docs/INDEX.md final pass with all H-K rows

## Phase M: Code Quality Hardening ✅ COMPLETE

- Coverage threshold enforcement (`--cov-fail-under=40` in PR checks)
- MCP search tool: `use_hyde` parameter for HyDE query expansion
- MCP `fast_search` tool: vector-only search (~200ms, 20th tool)
- Daemon connection timeout 5s→10s (graph queries take 1.7-5.8s)
- Dashboard AppTest suite: 24 new tests (search + citations pages)

## Phase N: Domain Gap Expansion ✅ COMPLETE

- 3 new domains registered: `sql`, `recommender_systems`, `adtech` (22 total)
- 5 new domain prompt configs: sql, recommender_systems, adtech, algorithms, forecasting (19 total)
- 232 domain prompt tests (up from 162)
- 97 sidecar JSON domain corrections in `fixtures/textbooks/migrated/`
- 88 DB source retags (causal_inference fallback → correct domain)
- 27,965 chunk domain_id propagation via `sync_chunk_domains.py`
- Ingestion infrastructure updated: VALID_DOMAINS, DOMAIN_KEYWORDS

## Phase O: Eval & Quality Hardening ✅ COMPLETE

- Retrieval eval expansion: 55 → 82 test cases across all 19 populated domains + 9 future-tagged
- `expected_concepts` added to 10 more existing test cases (concept recall now meaningful)
- `portfolio_management` extraction prompt config (20th domain, 11 DB sources)
- CI weekly rebuild threshold tightened: `--fail-below 0.5` → `--fail-below 0.85`
- CI weekly rebuild installs all 12 packages (was 4)
- Chunker Unicode normalization fix (NFKC for NBSP/SHY/ligatures in section tracking)
- TEST_COVERAGE_AUDIT.md refreshed (2,182 test functions across 91 test files)
- Primary textbook acquisition wishlist for empty/thin domains

---

**Current Status**: All phases H-K, M, N, and O complete.

**Key Metrics** (as of 2026-02-25):
- Sources: 485 (across 22 domains, 3 empty)
- Chunks: ~226,000 (100% with embeddings)
- Concepts: 307,000 (742,000 relationships)
- KuzuDB: ~110MB graph engine
- Tests: ~2,182 test functions across 91 test files
- Domains: 23 registered — 20 with prompt configs (causal_inference 299, time_series 47, mathematics 27, interview_prep 22, rag_llm 17, software_engineering 15, portfolio_management 11, statistics 10, machine_learning 10, algorithms 9, deep_learning 9, functional_programming 8, data_science 4, ml_engineering 3, fitness 2, economics 1, finance 1, forecasting 1) + 3 empty (sql, recommender_systems, adtech)
- Retrieval eval: 82 test cases covering all populated domains + 9 future-tagged for empty domains
- CI threshold: 0.85 (catches 8%+ regressions)
- Method cache: 10/10 top methods, 55 cached assumptions, 87.5% readiness

**Architecture**: 12 packages (contracts → common → storage → cli/daemon/api/dashboard/mcp-server/client/pdf-tools/extraction/s2-client)

**Documentation**: Run `python scripts/generate_status.py` to refresh status docs. Run `python scripts/audit_docs.py` to check documentation health.

---

## Future Work (Contributions Welcome)

### Retrieval Quality
- **Learned weight optimization**: Tune FTS/vector/graph/citation weights on golden dataset
- **Multi-vector retrieval**: ColBERT-style late interaction for fine-grained matching
- **Adaptive chunking**: Use document structure (sections, paragraphs) instead of fixed token windows

### New Domains
- **Biology/Genomics**: Pathway analysis literature
- **Climate science**: Climate modeling papers
- **Your domain**: See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new domain

### Infrastructure
- **Kubernetes deployment**: Helm chart for production deployment
- **Streaming search**: Server-sent events for real-time result streaming
- **Multi-user**: Authentication and per-user corpora

### Knowledge Graph
- **Temporal reasoning**: Track how assumptions/methods evolve across publications
- **Contradiction detection**: Flag conflicting claims across papers
- **Automated literature review**: Generate structured reviews from graph traversal

### Dashboard
- **Screenshot/GIF capture**: Visual showcase for README and docs
- **Performance page**: Benchmark visualization from `fixtures/benchmarks/`
- **Comparison mode**: Side-by-side search results with different weight profiles
