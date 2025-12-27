-- Phase 2.3: Bibliographic Coupling
-- Sources sharing references → similarity signal

-- Precomputed bibliographic coupling scores
CREATE TABLE IF NOT EXISTS bibliographic_coupling (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source pair (ordered to avoid duplicates)
    source_a_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    source_b_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,

    -- Coupling metrics
    shared_references INT NOT NULL,
    coupling_strength REAL NOT NULL,  -- Jaccard: shared / (a_refs + b_refs - shared)

    -- Shared reference details (for explanation)
    shared_source_ids UUID[] DEFAULT '{}',  -- Which corpus sources are shared references

    -- Metadata
    computed_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    UNIQUE(source_a_id, source_b_id),
    CHECK(source_a_id < source_b_id)  -- Enforce ordering to prevent duplicates
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_biblio_coupling_source_a
    ON bibliographic_coupling(source_a_id);

CREATE INDEX IF NOT EXISTS idx_biblio_coupling_source_b
    ON bibliographic_coupling(source_b_id);

CREATE INDEX IF NOT EXISTS idx_biblio_coupling_strength
    ON bibliographic_coupling(coupling_strength DESC)
    WHERE coupling_strength > 0.1;  -- Only meaningful coupling

-- Composite index for looking up a specific pair
CREATE INDEX IF NOT EXISTS idx_biblio_coupling_pair
    ON bibliographic_coupling(source_a_id, source_b_id);

-- View for easy querying of similar sources
CREATE OR REPLACE VIEW similar_sources AS
SELECT
    bc.source_a_id,
    sa.title AS source_a_title,
    bc.source_b_id,
    sb.title AS source_b_title,
    bc.shared_references,
    bc.coupling_strength,
    bc.computed_at
FROM bibliographic_coupling bc
JOIN sources sa ON sa.id = bc.source_a_id
JOIN sources sb ON sb.id = bc.source_b_id
WHERE bc.coupling_strength > 0.1
ORDER BY bc.coupling_strength DESC;

COMMENT ON TABLE bibliographic_coupling IS 'Precomputed similarity between sources based on shared references (Jaccard coefficient)';
COMMENT ON COLUMN bibliographic_coupling.coupling_strength IS 'Jaccard similarity: |shared| / |union| where 1.0 = identical references';
