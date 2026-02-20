# Research-KB System Design

**Version**: Summary v1.0

---

## Purpose & Overview

Research-KB is a semantic search system for causal inference literature that combines:
- **Full-text search** (PostgreSQL FTS)
- **Vector similarity** (BGE-large-en-v1.5, 1024 dimensions)
- **Knowledge graph signals** (concept relationships)

**Problem Solved**: Efficiently search and connect concepts across academic papers, textbooks, and code repositories with domain-aware relevance ranking.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                           │
│                    (research-kb commands)                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                              │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │SourceStore│  │ChunkStore │  │ConceptStore│  │CitationStore│ │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘    │
│                               │                                 │
│            ┌──────────────────┴──────────────────┐              │
│            │         Hybrid Search               │              │
│            │   FTS + Vector + Graph Scoring      │              │
│            └─────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PostgreSQL + pgvector                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐│
│  │ sources │  │ chunks  │  │concepts │  │concept_relationships││
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Package Dependency Graph

```
contracts (pure Pydantic models)
    ↓
common (logging, retry, instrumentation)
    ├─→ storage (PostgreSQL + pgvector)
    │     ├─→ cli
    │     ├─→ pdf-tools
    │     ├─→ extraction
    │     ├─→ api
    │     └─→ dashboard
    ├─→ pdf-tools
    ├─→ extraction
    └─→ s2-client
```

| Package | Purpose |
|---------|---------|
| **contracts** | Pure Pydantic schemas - zero business logic |
| **common** | Cross-cutting: logging (structlog), retry (tenacity), tracing |
| **storage** | Exclusive database ownership (asyncpg, pgvector) |
| **pdf-tools** | PDF extraction (PyMuPDF, GROBID) + embeddings |
| **cli** | Typer-based interface, thin wrapper |
| **extraction** | Concept extraction via Ollama LLM |
| **api** | FastAPI REST endpoints with health checks and metrics |
| **dashboard** | Streamlit visualization for search and graph exploration |
| **s2-client** | Semantic Scholar API client with rate limiting and caching |

---

## Key Data Flow

### Search Query Flow
```
Query → Embed Query → Execute in Parallel:
  ├─ FTS Search (PostgreSQL ts_rank)
  ├─ Vector Search (pgvector cosine distance)
  └─ Graph Score (concept relationship traversal)
           ↓
    Weighted Combination:
    score = fts_weight × fts + vector_weight × vector + graph_weight × graph
           ↓
    Re-rank top-K → Return Results
```

### Ingestion Flow
```
PDF → Hash Check (idempotency) → GROBID/PyMuPDF →
  Chunk (300 tokens) → Embed (BGE-large) → Store
```

---

## Important Decisions

### 1. BGE-large-en-v1.5 (1024 dimensions)
**Rationale**: Best open-source model for semantic search quality. Single model for all embeddings ensures consistency.

### 2. Hybrid Search Weights
**Formula**: `score = fts × 0.3 + vector × 0.6 + graph × 0.1`

| Context | FTS | Vector | Graph |
|---------|-----|--------|-------|
| building | 20% | 70% | 10% |
| auditing | 45% | 45% | 10% |
| balanced | 30% | 60% | 10% |

### 3. JSONB Extensibility
**Strategy**: Use JSONB `metadata` columns for unknown future use cases. Promote to dedicated tables when patterns emerge.

### 4. Async Throughout
All storage operations are async using `asyncpg` with connection pooling (2-10 connections).

---

## Database Schema Overview

**Core Tables**:
- `sources` - Document metadata (type, title, authors, year, file_hash)
- `chunks` - Searchable text (content, embedding[1024], page_start/end, metadata)
- `citations` - Extracted references (DOI, arXiv, BibTeX)

**Knowledge Graph Tables**:
- `concepts` - Extracted concepts with types and confidence
- `concept_relationships` - Edges between concepts
- `chunk_concepts` - Links chunks to concepts

**Key Indexes**:
- GIN on `fts_vector` (full-text search)
- IVFFlat on `embedding` (vector similarity)
- UNIQUE on `file_hash` (idempotency)

---

## Full Documentation

For complete details including:
- All schema definitions
- API specifications
- Performance benchmarks
- Error handling patterns
- Deployment configuration

See the architecture documentation in [CLAUDE.md](../CLAUDE.md) for full details.

---

## Quick Reference

| Component | Location |
|-----------|----------|
| Schema | `packages/storage/schema.sql` |
| Search | `packages/storage/src/research_kb_storage/search.py` |
| Graph queries | `packages/storage/src/research_kb_storage/graph_queries.py` |
| PDF extraction | `packages/pdf-tools/src/research_kb_pdf/` |
| Concept extraction | `packages/extraction/src/research_kb_extraction/` |
| CLI | `packages/cli/src/research_kb_cli/main.py` |
