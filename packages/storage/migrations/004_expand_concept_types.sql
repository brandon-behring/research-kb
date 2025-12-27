-- Migration 004: Expand concept_type CHECK constraint
--
-- Problem: Migration 002 created a CHECK constraint allowing only 5 concept types,
-- but the codebase and extraction pipeline use 9 types.
--
-- This migration is idempotent and safe to run on:
-- - Fresh installs (applies new constraint)
-- - Production databases (constraint may already be expanded manually)
--
-- Concept types:
--   Original (002): method, assumption, problem, definition, theorem
--   Extended: concept, principle, technique, model

-- Safely drop existing constraint (may not exist on fresh installs)
DO $$ BEGIN
    ALTER TABLE concepts DROP CONSTRAINT IF EXISTS concepts_concept_type_check;
EXCEPTION WHEN others THEN
    -- Constraint might not exist or have a different name
    NULL;
END $$;

-- Add the expanded constraint with all 9 types
ALTER TABLE concepts ADD CONSTRAINT concepts_concept_type_check
    CHECK (concept_type IN (
        'method',
        'assumption',
        'problem',
        'definition',
        'theorem',
        'concept',
        'principle',
        'technique',
        'model'
    ));

-- Verify the migration succeeded
DO $$
DECLARE
    constraint_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO constraint_count
    FROM information_schema.table_constraints
    WHERE table_name = 'concepts'
    AND constraint_name = 'concepts_concept_type_check';

    IF constraint_count = 0 THEN
        RAISE EXCEPTION 'Migration 004 failed: constraint not created';
    END IF;
END $$;
