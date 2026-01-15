-- Migration 012: Healthcare Analytics Domain
-- Purpose: Add healthcare domain for Oscar Health interview prep and healthcare ML research
-- Created: 2026-01-03

-- ===========================================================================
-- ADD HEALTHCARE DOMAIN
-- ===========================================================================

-- Healthcare analytics: risk adjustment, population health, clinical outcomes
INSERT INTO domains (id, name, description, concept_types, relationship_types) VALUES
(
    'healthcare',
    'Healthcare Analytics',
    'Healthcare ML, risk adjustment, population health, and clinical outcomes',
    ARRAY['method', 'assumption', 'problem', 'definition', 'theorem', 'concept', 'principle', 'technique', 'model'],
    ARRAY['REQUIRES', 'USES', 'ADDRESSES', 'GENERALIZES', 'SPECIALIZES', 'ALTERNATIVE_TO', 'EXTENDS', 'RELATED_TO']
)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    concept_types = EXCLUDED.concept_types,
    relationship_types = EXCLUDED.relationship_types;

-- Also ensure rag_llm domain exists (defined in domain_prompts.py but may not be in DB)
INSERT INTO domains (id, name, description, concept_types, relationship_types) VALUES
(
    'rag_llm',
    'RAG & LLM',
    'Retrieval-augmented generation, language models, prompting, embeddings',
    ARRAY['method', 'technique', 'model', 'concept', 'problem', 'definition', 'assumption', 'theorem', 'principle'],
    ARRAY['REQUIRES', 'USES', 'ADDRESSES', 'GENERALIZES', 'SPECIALIZES', 'ALTERNATIVE_TO', 'EXTENDS', 'RELATED_TO']
)
ON CONFLICT (id) DO NOTHING;

-- ===========================================================================
-- VERIFY DOMAINS
-- ===========================================================================
-- Expected domains after this migration:
--   - causal_inference (seeded in 010)
--   - time_series (seeded in 010)
--   - rag_llm (added here)
--   - healthcare (added here)

-- ===========================================================================
-- MIGRATION NOTES
-- ===========================================================================
--
-- Healthcare domain is designed for Oscar Health interview prep:
-- - CMS-HCC risk adjustment methodology
-- - Claims data analysis
-- - Population health modeling
-- - Healthcare quality measurement
--
-- The domain uses the same concept types as causal_inference for consistency,
-- but has healthcare-specific abbreviation deduplication (see domain_prompts.py).
--
-- Cross-domain linking (causal_inference <-> healthcare) should be run after
-- extraction to discover related concepts (e.g., "propensity score" <-> "risk adjustment").
