-- Migration 017: Add 'web' to the source_type CHECK constraint
--
-- Admits frontier web-native primaries (vendor docs, standards bodies,
-- practitioner blogs) as a first-class source_type, tagged with
-- metadata.source_class + a short half-life (stale_after_days). This is the
-- D2 decision settled 2026-05-28: web sources are the ONLY primaries for the
-- agents / ml_security domains (83% / 49% web), so research-kb admits them
-- distinctly from peer-reviewed 'paper'/'textbook' and from 'blog'.
--
-- Pairs with: research_kb_contracts.SourceType.WEB (models.py).
--
-- Pre-flight:
--     SELECT DISTINCT source_type FROM sources;
-- Expected: subset of {textbook, paper, code_repo, blog}.
--
-- Rollback:
--     ALTER TABLE sources DROP CONSTRAINT source_type_check;
--     ALTER TABLE sources ADD CONSTRAINT source_type_check
--         CHECK (source_type IN ('textbook', 'paper', 'code_repo', 'blog'));

BEGIN;

ALTER TABLE sources DROP CONSTRAINT source_type_check;

ALTER TABLE sources
    ADD CONSTRAINT source_type_check
    CHECK (source_type IN ('textbook', 'paper', 'code_repo', 'blog', 'web'));

COMMIT;
