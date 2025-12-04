# Research KB Roadmap

A causal inference knowledge base for research context retrieval.

## Phase 1: Foundation (Weeks 1-2) ✅ COMPLETE

- PostgreSQL + pgvector (1024-dim embeddings)
- PDF extraction (PyMuPDF + GROBID)
- Hybrid search (FTS + vector, configurable weights)
- CLI: `research-kb query`, `stats`, `sources`
- Corpus: 20 sources (2 textbooks + 18 papers)
- Eval: 14/14 tests, 100% P@5

## Phase 2: Knowledge Graph (Weeks 3-4)

- Concept extraction with LLM (Ollama llama3.2 baseline)
- Method & assumption database
- Relationship ontology (REQUIRES, USES, ADDRESSES, etc.)
- Graph traversal queries (SQL recursive CTEs)
- Hybrid retrieval (vector + FTS + graph signals)

## Phase 3: Enhanced Retrieval (Weeks 5-6)

- Query expansion with concept synonyms
- Cross-encoder re-ranking
- Multi-hop reasoning chains
- Citation graph integration

## Phase 4: Production (Weeks 7-8)

- FastAPI REST API
- Authentication & rate limiting
- Observability (OpenTelemetry, structured logging)
- Deployment automation

---

**Current Status**: Phase 1 complete, Phase 2 in design phase.

**Key Metrics**:
- Precision@5: 100% (target: 90%)
- Chunks: 3,975
- Sources: 20
