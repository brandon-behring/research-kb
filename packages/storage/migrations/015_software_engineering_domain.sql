-- Migration 015: Software Engineering Domain
-- Purpose: Add software engineering domain for SE research and best practices
-- Created: 2026-02-21

-- ===========================================================================
-- ADD SOFTWARE ENGINEERING DOMAIN
-- ===========================================================================

-- Software engineering: design patterns, testing, architecture, DevOps, CI/CD
INSERT INTO domains (id, name, description, concept_types, relationship_types) VALUES
(
    'software_engineering',
    'Software Engineering',
    'Software engineering practices: design patterns, testing strategies, architecture, DevOps, CI/CD, code quality, and development methodology',
    ARRAY['method', 'assumption', 'problem', 'definition', 'theorem', 'concept', 'principle', 'technique', 'model'],
    ARRAY['REQUIRES', 'USES', 'ADDRESSES', 'GENERALIZES', 'SPECIALIZES', 'ALTERNATIVE_TO', 'EXTENDS', 'RELATED_TO']
)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    concept_types = EXCLUDED.concept_types,
    relationship_types = EXCLUDED.relationship_types;

-- ===========================================================================
-- VERIFY DOMAINS
-- ===========================================================================
-- Expected domains after this migration:
--   - causal_inference (seeded in 010)
--   - time_series (seeded in 010)
--   - rag_llm (seeded in 012)
--   - healthcare (seeded in 012)
--   - software_engineering (added here)

-- ===========================================================================
-- MIGRATION NOTES
-- ===========================================================================
--
-- Software engineering domain covers:
-- - Design patterns (GoF, architectural patterns, microservices)
-- - Testing strategies (unit, integration, property-based, mutation)
-- - Architecture (clean architecture, hexagonal, event-driven)
-- - DevOps and CI/CD (deployment patterns, infrastructure as code)
-- - Code quality (refactoring, technical debt, code review)
-- - Development methodology (agile, TDD, BDD)
--
-- The domain uses the same concept types as other domains for consistency.
-- Domain-specific abbreviation deduplication is configured in domain_prompts.py.
--
-- Cross-domain linking (software_engineering <-> rag_llm) should be run after
-- extraction to discover related concepts (e.g., "CI/CD" <-> "evaluation pipeline").
