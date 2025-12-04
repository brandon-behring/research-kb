-- Research Knowledge Base - Phase 2 Knowledge Graph Schema
-- Created: 2025-12-01
-- Purpose: Concept extraction, relationships, and graph queries
-- Depends: schema.sql (sources, chunks, citations tables)

-- ===========================================================================
-- CONCEPTS: Extracted knowledge entities
-- ===========================================================================
CREATE TABLE IF NOT EXISTS concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core identity
    name TEXT NOT NULL,
    canonical_name TEXT NOT NULL UNIQUE,
    aliases TEXT[] DEFAULT '{}',

    -- Classification
    concept_type TEXT NOT NULL CHECK (concept_type IN (
        'method', 'assumption', 'problem', 'definition', 'theorem'
    )),
    category TEXT,  -- identification, estimation, testing, etc.
    definition TEXT,

    -- Semantic search (reuses BGE-large-en-v1.5 from chunks)
    -- Embeds: canonical_name + definition
    embedding vector(1024),

    -- Extraction metadata
    extraction_method TEXT,  -- "ollama:llama3.1:8b", "manual"
    confidence_score REAL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    validated BOOLEAN DEFAULT FALSE,

    -- Extensibility
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE concepts IS 'Knowledge entities extracted from research documents';
COMMENT ON COLUMN concepts.canonical_name IS 'Normalized unique name for deduplication';
COMMENT ON COLUMN concepts.embedding IS 'BGE-large-en-v1.5 1024-dim embedding of canonical_name + definition';
COMMENT ON COLUMN concepts.validated IS 'TRUE if manually reviewed and confirmed';

-- ===========================================================================
-- CONCEPT_RELATIONSHIPS: Directed edges in knowledge graph
-- ===========================================================================
CREATE TABLE IF NOT EXISTS concept_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Graph edge endpoints
    source_concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    target_concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,

    -- Relationship semantics
    relationship_type TEXT NOT NULL CHECK (relationship_type IN (
        'REQUIRES',       -- Method requires assumption
        'USES',           -- Method uses technique
        'ADDRESSES',      -- Method solves problem
        'GENERALIZES',    -- Broader concept (Panel → DiD)
        'SPECIALIZES',    -- Narrower concept (LATE → treatment effect)
        'ALTERNATIVE_TO', -- Competing approaches (Matching vs Regression)
        'EXTENDS'         -- Builds upon (DML → ML + CI)
    )),
    is_directed BOOLEAN DEFAULT TRUE,

    -- Strength and evidence
    strength REAL DEFAULT 1.0 CHECK (strength >= 0 AND strength <= 1),
    evidence_chunk_ids UUID[],  -- Chunks where relationship was observed
    confidence_score REAL CHECK (confidence_score >= 0 AND confidence_score <= 1),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate edges of same type
    UNIQUE(source_concept_id, target_concept_id, relationship_type)
);

COMMENT ON TABLE concept_relationships IS 'Directed edges between concepts in knowledge graph';
COMMENT ON COLUMN concept_relationships.evidence_chunk_ids IS 'References to chunks where this relationship was observed';

-- ===========================================================================
-- CHUNK_CONCEPTS: Junction table (many-to-many)
-- Links chunks to the concepts they mention
-- ===========================================================================
CREATE TABLE IF NOT EXISTS chunk_concepts (
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,

    -- How the concept appears in the chunk
    mention_type TEXT DEFAULT 'reference' CHECK (mention_type IN (
        'defines',    -- Chunk defines the concept
        'reference',  -- Chunk mentions/uses the concept
        'example'     -- Chunk provides an example of the concept
    )),
    relevance_score REAL CHECK (relevance_score >= 0 AND relevance_score <= 1),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Composite PK allows same concept with different mention types
    PRIMARY KEY (chunk_id, concept_id, mention_type)
);

COMMENT ON TABLE chunk_concepts IS 'Links chunks to concepts they mention';
COMMENT ON COLUMN chunk_concepts.mention_type IS 'How concept appears: defines, reference, or example';

-- ===========================================================================
-- METHODS: Specialized attributes for method concepts
-- ===========================================================================
CREATE TABLE IF NOT EXISTS methods (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE UNIQUE,

    -- Method-specific attributes
    required_assumptions TEXT[],  -- List of assumption concept names
    problem_types TEXT[],         -- ATE, ATT, LATE, CATE, etc.
    common_estimators TEXT[]      -- OLS, 2SLS, matching, etc.
);

COMMENT ON TABLE methods IS 'Extended attributes for method-type concepts';
COMMENT ON COLUMN methods.required_assumptions IS 'Assumption concepts this method requires';

-- ===========================================================================
-- ASSUMPTIONS: Specialized attributes for assumption concepts
-- ===========================================================================
CREATE TABLE IF NOT EXISTS assumptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE UNIQUE,

    -- Assumption-specific attributes
    mathematical_statement TEXT,
    is_testable BOOLEAN,
    common_tests TEXT[],
    violation_consequences TEXT
);

COMMENT ON TABLE assumptions IS 'Extended attributes for assumption-type concepts';
COMMENT ON COLUMN assumptions.is_testable IS 'Whether assumption can be empirically tested';

-- ===========================================================================
-- INDEXES for graph traversal performance
-- Target: <100ms for 2-hop traversal queries
-- ===========================================================================

-- Concept lookups
CREATE INDEX IF NOT EXISTS idx_concepts_canonical_name ON concepts(canonical_name);
CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_concepts_category ON concepts(category) WHERE category IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_concepts_metadata ON concepts USING gin(metadata);

-- Concept embedding similarity search
CREATE INDEX IF NOT EXISTS idx_concepts_embedding ON concepts
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Graph traversal (both directions for relationship queries)
CREATE INDEX IF NOT EXISTS idx_relationships_source ON concept_relationships(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON concept_relationships(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON concept_relationships(relationship_type);

-- Chunk-concept lookups (both directions)
CREATE INDEX IF NOT EXISTS idx_chunk_concepts_chunk ON chunk_concepts(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_concepts_concept ON chunk_concepts(concept_id);
CREATE INDEX IF NOT EXISTS idx_chunk_concepts_mention ON chunk_concepts(mention_type);

-- ===========================================================================
-- HELPER FUNCTIONS for graph queries
-- ===========================================================================

-- Find concepts related within N hops (for Neo4j-less fallback)
CREATE OR REPLACE FUNCTION find_related_concepts(
    start_concept_id UUID,
    max_hops INTEGER DEFAULT 2
)
RETURNS TABLE (
    concept_id UUID,
    concept_name TEXT,
    relationship_path TEXT[],
    hop_distance INTEGER
) AS $$
WITH RECURSIVE concept_graph AS (
    -- Base case: starting concept
    SELECT
        c.id AS concept_id,
        c.name AS concept_name,
        ARRAY[]::TEXT[] AS path,
        0 AS distance
    FROM concepts c
    WHERE c.id = start_concept_id

    UNION ALL

    -- Recursive case: follow relationships
    SELECT
        CASE
            WHEN cr.source_concept_id = cg.concept_id THEN cr.target_concept_id
            ELSE cr.source_concept_id
        END AS concept_id,
        c2.name AS concept_name,
        cg.path || cr.relationship_type,
        cg.distance + 1
    FROM concept_graph cg
    JOIN concept_relationships cr
        ON cr.source_concept_id = cg.concept_id
        OR (cr.target_concept_id = cg.concept_id AND NOT cr.is_directed)
    JOIN concepts c2
        ON c2.id = CASE
            WHEN cr.source_concept_id = cg.concept_id THEN cr.target_concept_id
            ELSE cr.source_concept_id
        END
    WHERE cg.distance < max_hops
      AND c2.id != start_concept_id  -- Don't return to start
)
SELECT DISTINCT ON (concept_id)
    concept_id,
    concept_name,
    path AS relationship_path,
    distance AS hop_distance
FROM concept_graph
WHERE distance > 0  -- Exclude starting concept
ORDER BY concept_id, distance ASC;
$$ LANGUAGE SQL STABLE;

COMMENT ON FUNCTION find_related_concepts IS 'PostgreSQL fallback for graph traversal when Neo4j unavailable';

-- ===========================================================================
-- SCHEMA EVOLUTION NOTES
-- ===========================================================================
--
-- Neo4j Integration (per /iterate Q3 decision):
-- - Primary graph store: Neo4j (Docker container)
-- - PostgreSQL maintains authoritative data
-- - Sync service: research_kb_extraction/graph_sync.py
-- - Use find_related_concepts() as fallback if Neo4j unavailable
--
-- Embedding Strategy:
-- - Concept embeddings use same BGE-large-en-v1.5 as chunk embeddings
-- - Embeds: canonical_name + definition
-- - Enables semantic deduplication (similarity > 0.95 = duplicate)
--
-- ===========================================================================
