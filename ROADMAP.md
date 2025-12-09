# Research KB Roadmap

A causal inference knowledge base for research context retrieval.

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
- Performance: 2.11ms for 2-hop queries (target: <100ms)

## Phase 3: Enhanced Retrieval (Weeks 5-6) 📋 READY TO START

- Query expansion with concept synonyms
- Cross-encoder re-ranking
- Multi-hop reasoning chains
- Citation graph integration

## Phase 4: Production (Weeks 7-8) 📋 PLANNED

- FastAPI REST API
- Authentication & rate limiting
- Observability (OpenTelemetry, structured logging)
- Deployment automation

---

**Current Status**: Phase 2 complete. Ready for Phase 3.

**Key Metrics** (auto-generated, see `docs/status/CURRENT_STATUS.md`):
- Sources: 137 (65 textbooks + 72 papers)
- Chunks: 33,973 (100% with embeddings)
- Concepts: 19,458 (91.6% with embeddings)
- Relationships: 20,423
- Tests: 502 functions across 32 files
- 2-hop graph query: 2.11ms (target: <100ms) ✅

**Documentation Note**: Run `python scripts/generate_status.py` to update status docs from database.
