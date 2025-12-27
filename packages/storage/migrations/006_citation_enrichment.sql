-- Phase 2.2: Citation Enrichment Indexes
-- Adds indexes for S2 enrichment queries and tracking

-- S2 paper ID column for direct linking
ALTER TABLE citations ADD COLUMN IF NOT EXISTS s2_paper_id TEXT;

-- Index for S2 paper lookup
CREATE INDEX IF NOT EXISTS idx_citations_s2_paper
    ON citations(s2_paper_id)
    WHERE s2_paper_id IS NOT NULL;

-- Index for enrichment status queries
CREATE INDEX IF NOT EXISTS idx_citations_s2_status
    ON citations((metadata->>'s2_match_status'));

-- Index for staleness checking (when was citation last enriched)
CREATE INDEX IF NOT EXISTS idx_citations_s2_enriched_at
    ON citations((metadata->>'s2_enriched_at'));

-- Index for citations with citation counts (for sorting by impact)
CREATE INDEX IF NOT EXISTS idx_citations_citation_count
    ON citations(((metadata->>'citation_count')::int))
    WHERE metadata->>'citation_count' IS NOT NULL;

COMMENT ON COLUMN citations.s2_paper_id IS 'Semantic Scholar paper ID from enrichment';
