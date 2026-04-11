-- Migration 016: Add BLOG to SourceType + CHECK constraint
--
-- Adds the 'blog' variant to SourceType and introduces a CHECK constraint
-- on sources.source_type that was previously missing. The old schema only
-- enforced the enum via a comment; this tightens the invariant at the DB
-- level.
--
-- Pre-flight (run before applying):
--     SELECT DISTINCT source_type FROM sources;
-- Expected values: a subset of {textbook, paper, code_repo, blog}. Abort
-- if anything else appears.
--
-- Rollback:
--     ALTER TABLE sources DROP CONSTRAINT source_type_check;

BEGIN;

ALTER TABLE sources
    ADD CONSTRAINT source_type_check
    CHECK (source_type IN ('textbook', 'paper', 'code_repo', 'blog'));

COMMIT;
