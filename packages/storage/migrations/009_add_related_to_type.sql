-- Migration: Add RELATED_TO relationship type
-- Purpose: Enable semantic similarity-based concept linking for orphan recovery
-- Date: 2025-12-27

-- Update the relationship_type CHECK constraint to include RELATED_TO
ALTER TABLE concept_relationships DROP CONSTRAINT IF EXISTS concept_relationships_relationship_type_check;

ALTER TABLE concept_relationships ADD CONSTRAINT concept_relationships_relationship_type_check
    CHECK (relationship_type = ANY (ARRAY[
        'REQUIRES',
        'USES',
        'ADDRESSES',
        'GENERALIZES',
        'SPECIALIZES',
        'ALTERNATIVE_TO',
        'EXTENDS',
        'RELATED_TO'
    ]));

-- Add comment explaining the new type
COMMENT ON CONSTRAINT concept_relationships_relationship_type_check ON concept_relationships IS
    'Valid relationship types. RELATED_TO added for semantic similarity-based orphan re-linking.';
