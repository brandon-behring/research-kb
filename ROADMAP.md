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

## Phase 3: Enhanced Retrieval (Weeks 5-6) ✅ COMPLETE

- ✅ Query expansion with concept synonyms (synonym_map.json)
- ✅ Cross-encoder re-ranking (BGE reranker)
- ✅ Citation graph integration (5,044 citations, 275 internal edges)
- ✅ 4-way hybrid search (FTS + vector + graph + citation)

## Phase 4: Visualization & MCP (Weeks 7-8) 📋 PLANNED

- Streamlit + PyVis dashboard
- Citation network visualization
- Concept graph explorer with N-hop neighborhoods
- Multi-hop path finder between concepts
- MCP server for Claude Code integration

---

**Current Status**: Phase 3 complete. Phase 4 (Visualization) ready to start.

**Key Metrics** (as of 2025-12-09):
- Sources: 136 (65 textbooks + 71 papers)
- Chunks: 46,263 (100% with embeddings)
- Citations: 5,044 (275 internal edges)
- Concepts: rebuilding (extraction in progress)
- Tests: 502 functions
- 2-hop graph query: 2.11ms (target: <100ms) ✅

**Phase 3 Deliverables**:
- ✅ Query expansion with synonym map
- ✅ Cross-encoder reranking (BGE model)
- ✅ Citation graph with PageRank authority scores
- ✅ 4-way hybrid retrieval

**Documentation Note**: Run `python scripts/generate_status.py` to update status docs from database.
