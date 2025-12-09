# Phase 1: Foundation

**Status**: ✅ **COMPLETE**
**Duration**: Week 1 (Days 1-6)
**Date Completed**: 2025-11-29

---

## Overview

Phase 1 established the complete foundation for the research-kb system with a proven minimal schema, layered package architecture, and working storage layer.

---

## Deliverables

### 1. Packages

#### contracts (v1.0.0) - FROZEN
**Purpose**: Pure Pydantic schemas with zero business logic

- `Source` (textbook/paper/code_repo)
- `Chunk` (content + 1024-dim embeddings)
- `SearchResult` (hybrid FTS + vector)
- `IngestionStatus` (pipeline state machine)

**Dependencies**: `pydantic` only
**Tests**: 21/21 passed

---

#### common (v1.0.0)
**Purpose**: Shared utilities across all packages

- Structured logging (`structlog`): JSON + human-readable modes
- Retry/backoff patterns (`tenacity`): Exponential backoff
- OpenTelemetry instrumentation: Tracing helpers
- Custom error types: `IngestionError`, `StorageError`, `SearchError`

**Dependencies**: `structlog`, `tenacity`, `opentelemetry-api/sdk`, `research-kb-contracts`
**Tests**: 22/22 passed

---

#### storage (v1.0.0)
**Purpose**: PostgreSQL storage layer with exclusive DB ownership

- `DatabaseConfig`: Connection configuration
- `SourceStore`: CRUD operations for sources table
- `ChunkStore`: CRUD operations for chunks table (with pgvector support)
- `search_hybrid()`: FTS + vector similarity search

**Key Implementation Details**:
- Asyncpg connection pooling (2-10 connections)
- JSONB codec configuration for metadata
- pgvector 1024-dim embeddings (BGE-large-en-v1.5)
- Hybrid search with weighted FTS + vector scores

---

### 2. Database Schema

**File**: `packages/storage/schema.sql`

**Tables**:
| Table | Columns | Purpose |
|-------|---------|---------|
| `sources` | id, source_type, title, authors, year, file_path, file_hash (UNIQUE), metadata (JSONB) | Document metadata |
| `chunks` | id, source_id, content, content_hash, location, page_start/end, embedding (vector(1024)), metadata (JSONB) | Searchable text chunks |

**Indexes**:
- Full-text search: GIN index on `fts_vector` (generated column with location boosting)
- Vector search: IVFFlat index on `embedding` (lists=100)
- Foreign key: `chunks.source_id` → `sources.id` (CASCADE delete)
- Deduplication: UNIQUE index on `sources.file_hash`

---

### 3. Infrastructure

**File**: `docker-compose.yml`

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL 16 + pgvector | 5432 | Vector search database |
| GROBID | 8070 | PDF structure extraction |
| pgAdmin | 5050 (dev profile) | Database GUI |

---

## Key Decisions

### 1. JSONB Extensibility Strategy
- **Decision**: Use JSONB `metadata` columns instead of rigid schema
- **Rationale**: Unknown future use cases (flashcards, concept linking, etc.)
- **Migration Path**: JSONB → dedicated table when patterns emerge

### 2. Separate `common` from `contracts`
- **Decision**: Instrumentation in `common`, NOT in `contracts`
- **Rationale**: Keeps schemas pure (Pydantic only)

### 3. Asyncpg Connection Pooling
- **Decision**: Global connection pool with 2-10 connections
- **Rationale**: Reuse connections across requests for performance
- **Tradeoff**: Must manage pool lifecycle carefully

### 4. JSONB Codec Configuration
- **Decision**: Set JSON codec on each connection
- **Rationale**: Asyncpg doesn't auto-convert dict ↔ JSONB

### 5. 1024-dim Embeddings
- **Decision**: BGE-large-en-v1.5 (1024 dimensions)
- **Rationale**: Best open-source model for semantic search quality

---

## Test Results

| Package | Tests | Status |
|---------|-------|--------|
| contracts | 21/21 | ✅ |
| common | 22/22 | ✅ |
| storage | Core CRUD | ✅ |
| **Total** | 43+ | ✅ |

---

## Metrics

| Metric | Value |
|--------|-------|
| Packages delivered | 3/6 (Phase 1) |
| Tests written | 43 |
| Lines of code | ~3,500 |
| Schema tables | 2 (sources, chunks) |
| JSONB fields | 2 (extensibility) |
| Vector dimensions | 1024 (BGE-large-en-v1.5) |

---

## Key Learnings

1. **JSONB flexibility essential** - Future use cases leverage metadata
2. **Asyncpg JSON codec** - Must configure explicitly for JSONB ↔ dict conversion
3. **Schema minimalism works** - 2 tables + JSONB handles all source types

---

## Next Phase

→ [Phase 1.5: PDF Ingestion](../phase1.5/PDF_INGESTION.md)
