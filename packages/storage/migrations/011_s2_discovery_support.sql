-- Migration 011: S2 Auto-Discovery Support
-- Purpose: Enable automated paper discovery via Semantic Scholar integration
-- Created: 2025-12-31

-- ===========================================================================
-- FAST S2 ID LOOKUPS ON METADATA JSONB
-- ===========================================================================
-- These indexes enable fast deduplication when discovering papers from S2

CREATE INDEX idx_sources_s2_paper_id ON sources
    USING btree ((metadata->>'s2_paper_id'))
    WHERE metadata->>'s2_paper_id' IS NOT NULL;

CREATE INDEX idx_sources_doi ON sources
    USING btree ((metadata->>'doi'))
    WHERE metadata->>'doi' IS NOT NULL;

CREATE INDEX idx_sources_arxiv ON sources
    USING btree ((metadata->>'arxiv_id'))
    WHERE metadata->>'arxiv_id' IS NOT NULL;

COMMENT ON INDEX idx_sources_s2_paper_id IS 'Fast lookup by Semantic Scholar paper ID';
COMMENT ON INDEX idx_sources_doi IS 'Fast lookup by DOI for deduplication';
COMMENT ON INDEX idx_sources_arxiv IS 'Fast lookup by arXiv ID for deduplication';

-- ===========================================================================
-- DISCOVERY AUDIT LOG
-- ===========================================================================
-- Tracks all discovery operations for auditing and monitoring

CREATE TABLE discovery_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Discovery metadata
    discovery_method TEXT NOT NULL,         -- 'keyword_search', 'author_search', 'citation_traverse', 'topic_batch'
    query TEXT,                             -- Search query or paper ID for traversal
    domain_id TEXT DEFAULT 'causal_inference' REFERENCES domains(id),

    -- Results summary
    papers_found INTEGER DEFAULT 0,         -- Total papers returned by S2
    papers_acquired INTEGER DEFAULT 0,      -- PDFs successfully downloaded
    papers_ingested INTEGER DEFAULT 0,      -- Successfully processed into corpus
    papers_skipped INTEGER DEFAULT 0,       -- Skipped (duplicates)
    papers_failed INTEGER DEFAULT 0,        -- Failed to process

    -- Performance
    duration_seconds REAL,

    -- Extensibility
    metadata JSONB DEFAULT '{}',            -- Additional context (year_from, min_citations, etc.)

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_discovery_log_method ON discovery_log(discovery_method);
CREATE INDEX idx_discovery_log_domain ON discovery_log(domain_id);
CREATE INDEX idx_discovery_log_created ON discovery_log(created_at DESC);

COMMENT ON TABLE discovery_log IS 'Audit log of all S2 discovery operations';
COMMENT ON COLUMN discovery_log.discovery_method IS 'Discovery method: keyword_search, author_search, citation_traverse, topic_batch';
COMMENT ON COLUMN discovery_log.papers_found IS 'Number of papers returned by S2 API';
COMMENT ON COLUMN discovery_log.papers_skipped IS 'Papers skipped due to deduplication';

-- ===========================================================================
-- INGESTION QUEUE
-- ===========================================================================
-- Async processing queue for discovered papers

CREATE TABLE ingestion_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Paper identifiers
    s2_paper_id TEXT UNIQUE,                -- Semantic Scholar paper ID (unique)
    doi TEXT,                               -- DOI if available
    arxiv_id TEXT,                          -- arXiv ID if available

    -- Download info
    pdf_url TEXT,                           -- URL to download PDF
    pdf_path TEXT,                          -- Local path after download

    -- Paper metadata
    title TEXT NOT NULL,
    authors TEXT[],                         -- Author names
    year INTEGER,
    venue TEXT,                             -- Publication venue

    -- Processing
    domain_id TEXT DEFAULT 'causal_inference' REFERENCES domains(id),
    status TEXT DEFAULT 'pending',          -- pending, downloading, extracting, embedding, completed, failed
    priority INTEGER DEFAULT 0,             -- Higher = process first
    error_message TEXT,                     -- Last error if failed
    retry_count INTEGER DEFAULT 0,          -- Number of retry attempts

    -- Extensibility
    metadata JSONB DEFAULT '{}',            -- S2 metadata, citation count, etc.

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Processing indexes
CREATE INDEX idx_ingestion_queue_status ON ingestion_queue(status);
CREATE INDEX idx_ingestion_queue_priority ON ingestion_queue(priority DESC, created_at);
CREATE INDEX idx_ingestion_queue_domain ON ingestion_queue(domain_id);

-- Deduplication indexes
CREATE INDEX idx_ingestion_queue_doi ON ingestion_queue(doi) WHERE doi IS NOT NULL;
CREATE INDEX idx_ingestion_queue_arxiv ON ingestion_queue(arxiv_id) WHERE arxiv_id IS NOT NULL;

COMMENT ON TABLE ingestion_queue IS 'Queue for async paper ingestion from S2 discovery';
COMMENT ON COLUMN ingestion_queue.status IS 'Processing status: pending, downloading, extracting, embedding, completed, failed';
COMMENT ON COLUMN ingestion_queue.priority IS 'Processing priority: higher values processed first';
COMMENT ON COLUMN ingestion_queue.retry_count IS 'Number of failed processing attempts';

-- ===========================================================================
-- HELPER FUNCTION: Update ingestion_queue.updated_at
-- ===========================================================================
CREATE OR REPLACE FUNCTION update_ingestion_queue_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_ingestion_queue_updated_at
    BEFORE UPDATE ON ingestion_queue
    FOR EACH ROW
    EXECUTE FUNCTION update_ingestion_queue_updated_at();

-- ===========================================================================
-- MIGRATION NOTES
-- ===========================================================================
--
-- This migration enables S2 auto-discovery:
--
-- 1. Metadata Indexes: Fast S2 ID / DOI / arXiv lookup for deduplication
-- 2. Discovery Log: Audit trail of all discovery operations
-- 3. Ingestion Queue: Async processing of discovered papers
--
-- Workflow:
--   1. Discovery script searches S2 for papers matching criteria
--   2. Papers not in corpus are added to ingestion_queue (status='pending')
--   3. Discovery operation is logged to discovery_log
--   4. Queue processor downloads PDFs, runs ingestion pipeline
--   5. Queue item status updated (completed/failed)
--
-- Usage:
--   -- Check discovery history
--   SELECT * FROM discovery_log ORDER BY created_at DESC LIMIT 10;
--
--   -- Process queue
--   SELECT * FROM ingestion_queue WHERE status = 'pending'
--   ORDER BY priority DESC, created_at LIMIT 50;
--
--   -- Check existing papers by S2 ID
--   SELECT id FROM sources WHERE metadata->>'s2_paper_id' = 'abc123';
--
