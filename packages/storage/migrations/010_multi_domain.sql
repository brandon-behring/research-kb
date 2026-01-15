-- Migration 010: Multi-Domain Support
-- Purpose: Enable research-kb to support multiple knowledge domains (causal inference, time series, etc.)
-- Created: 2025-12-30

-- ===========================================================================
-- DOMAIN REGISTRY: Track available knowledge domains
-- ===========================================================================
CREATE TABLE domains (
    id TEXT PRIMARY KEY,                        -- 'causal_inference', 'time_series'
    name TEXT NOT NULL,                         -- Human-readable: "Causal Inference"
    description TEXT,                           -- Extended description

    -- Domain configuration
    config JSONB DEFAULT '{}',                  -- Domain-specific configuration
    concept_types TEXT[] DEFAULT '{}',          -- Allowed concept types for this domain
    relationship_types TEXT[] DEFAULT '{}',     -- Allowed relationship types

    -- Default search weights for this domain
    default_fts_weight REAL DEFAULT 0.3,
    default_vector_weight REAL DEFAULT 0.7,
    default_graph_weight REAL DEFAULT 0.1,
    default_citation_weight REAL DEFAULT 0.15,

    -- Status
    status TEXT DEFAULT 'active',               -- 'active', 'inactive', 'deprecated'

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE domains IS 'Registry of knowledge domains supported by research-kb';
COMMENT ON COLUMN domains.config IS 'Domain-specific configuration (extraction prompts, etc.)';
COMMENT ON COLUMN domains.concept_types IS 'Concept types allowed in this domain';

-- ===========================================================================
-- ADD domain_id TO CORE TABLES
-- ===========================================================================

-- Sources: Which domain does this source belong to?
ALTER TABLE sources ADD COLUMN domain_id TEXT NOT NULL DEFAULT 'causal_inference';
ALTER TABLE sources ADD CONSTRAINT fk_sources_domain FOREIGN KEY (domain_id) REFERENCES domains(id);

-- Chunks: Inherit domain from source (denormalized for query performance)
ALTER TABLE chunks ADD COLUMN domain_id TEXT NOT NULL DEFAULT 'causal_inference';
ALTER TABLE chunks ADD CONSTRAINT fk_chunks_domain FOREIGN KEY (domain_id) REFERENCES domains(id);

-- Concepts: Which domain does this concept belong to?
ALTER TABLE concepts ADD COLUMN domain_id TEXT NOT NULL DEFAULT 'causal_inference';
ALTER TABLE concepts ADD CONSTRAINT fk_concepts_domain FOREIGN KEY (domain_id) REFERENCES domains(id);

-- ===========================================================================
-- CROSS-DOMAIN CONCEPT LINKS
-- ===========================================================================
CREATE TABLE cross_domain_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    target_concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,

    -- Link metadata
    link_type TEXT NOT NULL,                    -- 'SAME_AS', 'RELATED_TO', 'APPLIES_IN'
    confidence_score REAL,                      -- 0.0 to 1.0
    evidence TEXT,                              -- Why this link exists

    -- Extensibility
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate links
    UNIQUE(source_concept_id, target_concept_id)
);

COMMENT ON TABLE cross_domain_links IS 'Links between concepts across different knowledge domains';
COMMENT ON COLUMN cross_domain_links.link_type IS 'SAME_AS (identical), RELATED_TO (related), APPLIES_IN (used in context of)';

-- ===========================================================================
-- INDEXES FOR DOMAIN-SCOPED QUERIES
-- ===========================================================================
CREATE INDEX idx_sources_domain ON sources(domain_id);
CREATE INDEX idx_chunks_domain ON chunks(domain_id);
CREATE INDEX idx_concepts_domain ON concepts(domain_id);
CREATE INDEX idx_cross_domain_source ON cross_domain_links(source_concept_id);
CREATE INDEX idx_cross_domain_target ON cross_domain_links(target_concept_id);
CREATE INDEX idx_cross_domain_type ON cross_domain_links(link_type);

-- Composite indexes for common query patterns
CREATE INDEX idx_chunks_domain_source ON chunks(domain_id, source_id);
CREATE INDEX idx_concepts_domain_type ON concepts(domain_id, concept_type);

-- ===========================================================================
-- SEED INITIAL DOMAINS
-- ===========================================================================
INSERT INTO domains (id, name, description, concept_types, relationship_types) VALUES
(
    'causal_inference',
    'Causal Inference',
    'Econometrics, causal machine learning, and treatment effect estimation',
    ARRAY['method', 'assumption', 'problem', 'definition', 'theorem', 'concept', 'principle', 'technique', 'model'],
    ARRAY['REQUIRES', 'USES', 'ADDRESSES', 'GENERALIZES', 'SPECIALIZES', 'ALTERNATIVE_TO', 'EXTENDS', 'RELATED_TO']
),
(
    'time_series',
    'Time Series',
    'Forecasting, temporal modeling, and sequential data analysis',
    ARRAY['method', 'assumption', 'problem', 'definition', 'theorem', 'concept', 'principle', 'technique', 'model'],
    ARRAY['REQUIRES', 'USES', 'ADDRESSES', 'GENERALIZES', 'SPECIALIZES', 'ALTERNATIVE_TO', 'EXTENDS', 'RELATED_TO']
);

-- ===========================================================================
-- MIGRATION NOTES
-- ===========================================================================
--
-- This migration enables multi-domain support:
--
-- 1. Domain Registry: Central registry of knowledge domains with configuration
-- 2. Domain Scoping: All core tables (sources, chunks, concepts) now have domain_id
-- 3. Cross-Domain Links: Concepts can be linked across domains
-- 4. Search Performance: Indexes optimized for domain-scoped queries
--
-- Usage:
--   -- Domain-scoped query
--   SELECT * FROM chunks WHERE domain_id = 'time_series';
--
--   -- Cross-domain search
--   SELECT * FROM chunks WHERE domain_id IN ('causal_inference', 'time_series');
--
-- Future Extensions:
-- - Domain-specific search weights could be stored in domains.config
-- - Cross-domain concept discovery (automatic link detection)
-- - Domain inheritance (parent/child domains)
