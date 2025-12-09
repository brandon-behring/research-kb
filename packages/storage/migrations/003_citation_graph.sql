-- Migration 003: Citation Graph
-- Date: 2025-12-09
-- Purpose: Add source-to-source citation links and PageRank authority scores
--
-- This enables:
-- - "Who cites this source?" queries
-- - "What does this source cite?" queries
-- - Citation authority scoring for search ranking
-- - Type-aware citation statistics (paper→paper, textbook→paper, etc.)

-- ============================================================================
-- Table: source_citations
-- ============================================================================
-- Links citing sources to cited sources via the extracted citation record.
-- - citing_source_id: The source that contains the citation (always in corpus)
-- - cited_source_id: The source being cited (NULL if external to corpus)
-- - citation_id: The raw citation record with metadata
-- - context: Reserved for future citation context extraction (sentence where cited)

CREATE TABLE IF NOT EXISTS source_citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The source that contains this citation
    citing_source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,

    -- The source being cited (NULL if not in our corpus)
    cited_source_id UUID REFERENCES sources(id) ON DELETE SET NULL,

    -- Reference to the raw citation record
    citation_id UUID NOT NULL REFERENCES citations(id) ON DELETE CASCADE,

    -- Citation context (deferred - NULL for now)
    -- Future: Sentence/paragraph where citation appears
    context TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate edges
    UNIQUE(citing_source_id, citation_id)
);

-- Index for "who cites this source?" queries
CREATE INDEX IF NOT EXISTS idx_source_citations_cited
ON source_citations(cited_source_id)
WHERE cited_source_id IS NOT NULL;

-- Index for "what does this source cite?" queries
CREATE INDEX IF NOT EXISTS idx_source_citations_citing
ON source_citations(citing_source_id);

-- ============================================================================
-- Column: sources.citation_authority
-- ============================================================================
-- Precomputed PageRank-style authority score.
-- Higher = more cited by other corpus sources, especially highly-cited ones.
-- Recomputed after citation graph updates.

ALTER TABLE sources
ADD COLUMN IF NOT EXISTS citation_authority REAL DEFAULT 0.0;

-- Index for ranking by citation authority
CREATE INDEX IF NOT EXISTS idx_sources_citation_authority
ON sources(citation_authority DESC);

-- ============================================================================
-- Helper views for citation graph analysis
-- ============================================================================

-- View: Citation counts by source with type breakdown
CREATE OR REPLACE VIEW source_citation_stats AS
SELECT
    s.id AS source_id,
    s.source_type,
    s.title,
    s.citation_authority,
    -- Incoming citations (who cites this)
    COUNT(DISTINCT CASE WHEN sc.cited_source_id = s.id THEN sc.citing_source_id END) AS cited_by_count,
    COUNT(DISTINCT CASE WHEN sc.cited_source_id = s.id AND citing.source_type = 'paper' THEN sc.citing_source_id END) AS cited_by_papers,
    COUNT(DISTINCT CASE WHEN sc.cited_source_id = s.id AND citing.source_type = 'textbook' THEN sc.citing_source_id END) AS cited_by_textbooks,
    -- Outgoing citations (what this cites)
    COUNT(DISTINCT CASE WHEN sc.citing_source_id = s.id THEN sc.cited_source_id END) AS cites_count,
    COUNT(DISTINCT CASE WHEN sc.citing_source_id = s.id AND cited.source_type = 'paper' THEN sc.cited_source_id END) AS cites_papers,
    COUNT(DISTINCT CASE WHEN sc.citing_source_id = s.id AND cited.source_type = 'textbook' THEN sc.cited_source_id END) AS cites_textbooks
FROM sources s
LEFT JOIN source_citations sc ON sc.cited_source_id = s.id OR sc.citing_source_id = s.id
LEFT JOIN sources citing ON sc.citing_source_id = citing.id
LEFT JOIN sources cited ON sc.cited_source_id = cited.id
GROUP BY s.id, s.source_type, s.title, s.citation_authority;

-- View: Corpus-wide citation summary
CREATE OR REPLACE VIEW corpus_citation_summary AS
SELECT
    COUNT(DISTINCT c.id) AS total_citations,
    COUNT(DISTINCT sc.id) AS total_edges,
    COUNT(DISTINCT CASE WHEN sc.cited_source_id IS NOT NULL THEN sc.id END) AS internal_edges,
    COUNT(DISTINCT CASE WHEN sc.cited_source_id IS NULL THEN sc.id END) AS external_edges,
    -- By source type combination
    COUNT(DISTINCT CASE
        WHEN citing.source_type = 'paper' AND cited.source_type = 'paper'
        THEN sc.id END) AS paper_to_paper,
    COUNT(DISTINCT CASE
        WHEN citing.source_type = 'paper' AND cited.source_type = 'textbook'
        THEN sc.id END) AS paper_to_textbook,
    COUNT(DISTINCT CASE
        WHEN citing.source_type = 'textbook' AND cited.source_type = 'paper'
        THEN sc.id END) AS textbook_to_paper,
    COUNT(DISTINCT CASE
        WHEN citing.source_type = 'textbook' AND cited.source_type = 'textbook'
        THEN sc.id END) AS textbook_to_textbook
FROM citations c
LEFT JOIN source_citations sc ON c.id = sc.citation_id
LEFT JOIN sources citing ON sc.citing_source_id = citing.id
LEFT JOIN sources cited ON sc.cited_source_id = cited.id;

-- ============================================================================
-- Comments for documentation
-- ============================================================================

COMMENT ON TABLE source_citations IS 'Links citing sources to cited sources for citation graph traversal';
COMMENT ON COLUMN source_citations.citing_source_id IS 'Source that contains this citation (always in corpus)';
COMMENT ON COLUMN source_citations.cited_source_id IS 'Source being cited (NULL if external to corpus)';
COMMENT ON COLUMN source_citations.context IS 'Reserved: sentence where citation appears';
COMMENT ON COLUMN sources.citation_authority IS 'PageRank-style authority score (0-1, higher = more authoritative)';
