-- Research Knowledge Base - Phase 4 Assumption Auditing
-- Created: 2026-01-08
-- Purpose: Cache method assumptions for quick auditing
-- Depends: 002_knowledge_graph.sql (concepts, concept_relationships)

-- ===========================================================================
-- METHOD_ASSUMPTION_CACHE: LLM-extracted assumption data
-- ===========================================================================
-- This table caches structured assumption data for methods.
-- Used when:
-- 1. Graph query returns <3 assumptions (triggers LLM extraction)
-- 2. We want consistent formatting for Claude-first output
-- 3. We want to persist verification approaches and formal statements
--
-- Populated by: Ollama fallback extraction, manual curation
-- Queried by: research_kb_audit_assumptions MCP tool, CLI

CREATE TABLE IF NOT EXISTS method_assumption_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to method concept
    method_concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,

    -- Assumption details (may or may not have concept record)
    assumption_name TEXT NOT NULL,
    assumption_concept_id UUID REFERENCES concepts(id) ON DELETE SET NULL,

    -- Claude-first structured fields
    formal_statement TEXT,           -- Mathematical notation, e.g., "Y(t) ⊥ T | X"
    plain_english TEXT,              -- Human-readable, e.g., "No unmeasured confounders"
    importance TEXT CHECK (importance IN ('critical', 'standard', 'technical')),
    violation_consequence TEXT,      -- What goes wrong if violated
    verification_approaches TEXT[],  -- How to check, e.g., ["DAG review", "Sensitivity analysis"]

    -- Source provenance
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    source_citation TEXT,            -- "Chernozhukov et al. (2018), Section 2.1"

    -- Extraction metadata
    extraction_method TEXT,          -- "graph", "ollama:llama3.1:8b", "manual"
    confidence_score REAL CHECK (confidence_score >= 0 AND confidence_score <= 1),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate assumption-method pairs
    UNIQUE(method_concept_id, assumption_name)
);

COMMENT ON TABLE method_assumption_cache IS 'Cached assumption data for methods, used by audit_assumptions MCP tool';
COMMENT ON COLUMN method_assumption_cache.formal_statement IS 'Mathematical statement in LaTeX-like notation';
COMMENT ON COLUMN method_assumption_cache.importance IS 'critical=identification fails, standard=bias, technical=efficiency';
COMMENT ON COLUMN method_assumption_cache.extraction_method IS 'Source: graph query, LLM extraction, or manual curation';

-- ===========================================================================
-- METHOD_ALIASES: Common abbreviations and alternative names
-- ===========================================================================
-- Enables lookup by "DML", "debiased ML", "double machine learning", etc.

CREATE TABLE IF NOT EXISTS method_aliases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    method_concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,  -- TRUE for the canonical name

    UNIQUE(alias)  -- Each alias maps to exactly one method
);

CREATE INDEX IF NOT EXISTS idx_method_aliases_alias ON method_aliases(LOWER(alias));
CREATE INDEX IF NOT EXISTS idx_method_aliases_concept ON method_aliases(method_concept_id);

COMMENT ON TABLE method_aliases IS 'Maps abbreviations and alternative names to method concepts';

-- ===========================================================================
-- INDEXES for assumption auditing performance
-- ===========================================================================

-- Fast lookup by method concept
CREATE INDEX IF NOT EXISTS idx_method_assumption_cache_method
    ON method_assumption_cache(method_concept_id);

-- Find cached data needing refresh (by extraction method)
CREATE INDEX IF NOT EXISTS idx_method_assumption_cache_extraction
    ON method_assumption_cache(extraction_method);

-- ===========================================================================
-- TRIGGER: Update timestamp on modification
-- ===========================================================================

CREATE OR REPLACE FUNCTION update_method_assumption_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_method_assumption_updated ON method_assumption_cache;
CREATE TRIGGER trg_method_assumption_updated
    BEFORE UPDATE ON method_assumption_cache
    FOR EACH ROW
    EXECUTE FUNCTION update_method_assumption_timestamp();

-- ===========================================================================
-- HELPER FUNCTION: Get assumptions for a method from graph
-- ===========================================================================
-- Returns assumptions linked via REQUIRES or USES relationships
-- This is the primary query path before falling back to LLM

CREATE OR REPLACE FUNCTION get_method_assumptions_from_graph(
    method_id UUID
)
RETURNS TABLE (
    assumption_id UUID,
    assumption_name TEXT,
    assumption_canonical_name TEXT,
    relationship_type TEXT,
    relationship_strength REAL,
    definition TEXT,
    evidence_chunk_ids UUID[]
) AS $$
    SELECT
        c.id AS assumption_id,
        c.name AS assumption_name,
        c.canonical_name AS assumption_canonical_name,
        cr.relationship_type,
        cr.strength AS relationship_strength,
        c.definition,
        cr.evidence_chunk_ids
    FROM concept_relationships cr
    JOIN concepts c ON c.id = cr.target_concept_id
    WHERE cr.source_concept_id = method_id
      AND cr.relationship_type IN ('REQUIRES', 'USES')
      AND c.concept_type = 'assumption'
    ORDER BY
        CASE cr.relationship_type
            WHEN 'REQUIRES' THEN 1
            WHEN 'USES' THEN 2
        END,
        cr.strength DESC;
$$ LANGUAGE SQL STABLE;

COMMENT ON FUNCTION get_method_assumptions_from_graph IS
    'Query knowledge graph for assumptions linked to a method via REQUIRES/USES';

-- ===========================================================================
-- HELPER FUNCTION: Find method by name or alias
-- ===========================================================================

CREATE OR REPLACE FUNCTION find_method_concept(
    query_name TEXT
)
RETURNS TABLE (
    concept_id UUID,
    name TEXT,
    canonical_name TEXT,
    aliases TEXT[],
    definition TEXT
) AS $$
    -- First, check method_aliases table
    SELECT
        c.id AS concept_id,
        c.name,
        c.canonical_name,
        c.aliases,
        c.definition
    FROM method_aliases ma
    JOIN concepts c ON c.id = ma.method_concept_id
    WHERE LOWER(ma.alias) = LOWER(query_name)

    UNION

    -- Fallback: search concepts directly
    SELECT
        c.id AS concept_id,
        c.name,
        c.canonical_name,
        c.aliases,
        c.definition
    FROM concepts c
    WHERE c.concept_type = 'method'
      AND (
          LOWER(c.canonical_name) = LOWER(query_name)
          OR LOWER(c.name) = LOWER(query_name)
          OR LOWER(query_name) = ANY(SELECT LOWER(unnest(c.aliases)))
      )

    LIMIT 1;
$$ LANGUAGE SQL STABLE;

COMMENT ON FUNCTION find_method_concept IS
    'Find method concept by name, canonical name, or alias (case-insensitive)';
