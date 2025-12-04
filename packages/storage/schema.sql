-- Research Knowledge Base - Phase 1 Minimal Schema
-- Created: 2025-11-29
-- Purpose: Minimal schema with JSONB extensibility for unknown future use cases

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ===========================================================================
-- SOURCES: Books, Papers, Code Repositories
-- ===========================================================================
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Core metadata
    source_type TEXT NOT NULL,  -- textbook, paper, code_repo
    title TEXT NOT NULL,
    authors TEXT[],
    year INTEGER,

    -- File tracking (for idempotency)
    file_path TEXT,
    file_hash TEXT UNIQUE NOT NULL,  -- SHA256 hash for deduplication

    -- Extensibility: Unknown future metadata
    -- Examples: doi, arxiv_id, isbn, git_url, importance_tier, notes
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for sources
CREATE INDEX idx_sources_type ON sources(source_type);
CREATE INDEX idx_sources_metadata ON sources USING gin(metadata);

-- ===========================================================================
-- CHUNKS: Extracted content units
-- ===========================================================================
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,

    -- Content
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- For detecting duplicate chunks

    -- Location (for citations)
    location TEXT,  -- Human-readable: "Chapter 3, Section 3.4, p. 73"
    page_start INTEGER,
    page_end INTEGER,

    -- Semantic search (Phase 1: single embedding model)
    embedding vector(1024),  -- BGE-large-en-v1.5 embeddings (1024 dimensions)

    -- Future A/B testing (Phase 2+):
    -- embedding_alt vector(3072),  -- OpenAI text-embedding-3-large
    -- embedding_model TEXT,  -- Track which model(s) generated embeddings

    -- Extensibility: Future fields without schema migration
    -- Examples: chunk_type, parent_chunk_id, concepts[], theorem_text,
    --           flashcard_front/back, section_name, paragraph_num
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Full-text search (FTS) support
-- Generated column for FTS with weighted locations
ALTER TABLE chunks ADD COLUMN fts_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(location, '')), 'A') ||
        setweight(to_tsvector('english', content), 'B')
    ) STORED;

-- Indexes for chunks
CREATE INDEX idx_chunks_fts ON chunks USING gin(fts_vector);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_source ON chunks(source_id);
CREATE INDEX idx_chunks_metadata ON chunks USING gin(metadata);
CREATE INDEX idx_chunks_content_hash ON chunks(content_hash);

-- ===========================================================================
-- CITATIONS: Extracted references from GROBID (Phase 1.5.2)
-- ===========================================================================
CREATE TABLE citations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,

    -- Core citation metadata
    authors TEXT[] DEFAULT '{}',
    title TEXT,
    year INTEGER,
    venue TEXT,                    -- Journal, conference, publisher
    doi TEXT,
    arxiv_id TEXT,
    raw_string TEXT NOT NULL,      -- Original citation text

    -- BibTeX generation
    bibtex TEXT,                   -- Generated BibTeX entry

    -- Extraction metadata
    extraction_method TEXT,        -- "grobid", "manual"
    confidence_score REAL,         -- 0.0 to 1.0

    -- Extensibility
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for citations
CREATE INDEX idx_citations_source_id ON citations(source_id);
CREATE INDEX idx_citations_doi ON citations(doi) WHERE doi IS NOT NULL;
CREATE INDEX idx_citations_arxiv ON citations(arxiv_id) WHERE arxiv_id IS NOT NULL;
CREATE INDEX idx_citations_year ON citations(year) WHERE year IS NOT NULL;
CREATE INDEX idx_citations_metadata ON citations USING gin(metadata);

COMMENT ON TABLE citations IS 'Citations extracted from source documents via GROBID';
COMMENT ON COLUMN citations.raw_string IS 'Original citation string as it appeared in the document';
COMMENT ON COLUMN citations.bibtex IS 'Generated BibTeX entry for the citation';
COMMENT ON COLUMN citations.confidence_score IS 'GROBID extraction confidence (0.0 to 1.0)';

-- ===========================================================================
-- SCHEMA EVOLUTION NOTES
-- ===========================================================================
--
-- JSONB Extensibility Strategy:
--
-- 1. Experiment in JSONB first (sources.metadata, chunks.metadata)
-- 2. When use case solidifies, promote to dedicated table
-- 3. Example migration path for flashcards:
--    - Start: chunks.metadata->>'flashcard'
--    - Later: CREATE TABLE flashcards (chunk_id UUID REFERENCES chunks)
--
-- Known Future Extensions:
-- - Flashcards (Anki-style learning)
-- - Parent-child chunk relationships (hierarchical)
-- - Multi-embedding A/B testing (different models)
--
-- ===========================================================================
-- Phase 2 Knowledge Graph (in migrations/002_knowledge_graph.sql):
-- - concepts: Extracted knowledge entities
-- - concept_relationships: Directed graph edges
-- - chunk_concepts: Junction table (chunk ↔ concept)
-- - methods: Specialized method attributes
-- - assumptions: Specialized assumption attributes
-- - find_related_concepts(): PostgreSQL graph traversal function
-- ===========================================================================

COMMENT ON TABLE sources IS 'Source documents (textbooks, papers, code repos)';
COMMENT ON TABLE chunks IS 'Content units extracted from sources';
COMMENT ON COLUMN sources.file_hash IS 'SHA256 hash for idempotency - prevents duplicate ingestion';
COMMENT ON COLUMN chunks.embedding IS 'BGE-large-en-v1.5 1024-dim embeddings for semantic search';
COMMENT ON COLUMN chunks.fts_vector IS 'Generated tsvector for full-text search with location boosting';
COMMENT ON COLUMN sources.metadata IS 'Flexible JSONB for future extensions without schema migration';
COMMENT ON COLUMN chunks.metadata IS 'Flexible JSONB for chunk_type, concepts, flashcards, etc.';
