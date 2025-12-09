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

## Phase 3: Enhanced Retrieval (Weeks 5-6) 🔄 IN PROGRESS (75%)

- ✅ Query expansion with concept synonyms (synonym_map.json)
- ✅ Cross-encoder re-ranking (BGE reranker)
- 📋 Multi-hop reasoning chains (planned)
- 📋 Citation graph integration (requires citation extraction first)

## Phase 4: Production (Weeks 7-8) 📋 PLANNED

- FastAPI REST API
- Authentication & rate limiting
- Observability (OpenTelemetry, structured logging)
- Deployment automation

---

**Current Status**: Phase 3 in progress (75% complete).

**Key Metrics** (as of 2025-12-09):
- Sources: 138 (textbooks + papers + CFA materials)
- Chunks: 32,727 (100% with embeddings)
- Concepts: 17,819
- Relationships: 18,542
- Tests: 502 functions across 32 files
- 2-hop graph query: 2.11ms (target: <100ms) ✅

**Phase 3 Progress**:
- ✅ Query expansion with synonym map
- ✅ Cross-encoder reranking (BGE model)
- 📋 Multi-hop reasoning (planned)
- 📋 Citation graph (requires citation extraction)

**Documentation Note**: Run `python scripts/generate_status.py` to update status docs from database.
