-- Migration 008: Citation Context
-- Date: 2025-12-27
-- Purpose: Add context column to citations table for storing citing sentences
--
-- This enables:
-- - Storage of the sentence/paragraph where each citation appears in the source
-- - Richer citation context for understanding how papers are cited
-- - Better provenance tracking for research synthesis

-- ============================================================================
-- Column: citations.context
-- ============================================================================
-- The citing sentence or surrounding text where the citation appears.
-- Extracted from GROBID TEI-XML by finding <ref type="bibr"> elements
-- and extracting their surrounding sentence context.

ALTER TABLE citations ADD COLUMN IF NOT EXISTS context TEXT;

-- ============================================================================
-- Comments for documentation
-- ============================================================================

COMMENT ON COLUMN citations.context IS 'Citing sentence/context where this citation appears in the source document';
