-- Migration 005: Add provenance columns for LLM-enriched attributes
--
-- These columns track:
-- - evidence_chunk_ids: Source chunks that justify inferred attributes
-- - inference_confidence: LLM's confidence in the inference (0.0-1.0)
--
-- Supports verification and traceability of enrichment data.

-- Add provenance columns to methods table
ALTER TABLE methods ADD COLUMN IF NOT EXISTS evidence_chunk_ids UUID[] DEFAULT '{}';
ALTER TABLE methods ADD COLUMN IF NOT EXISTS inference_confidence FLOAT;
ALTER TABLE methods ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ;

-- Add provenance columns to assumptions table
ALTER TABLE assumptions ADD COLUMN IF NOT EXISTS evidence_chunk_ids UUID[] DEFAULT '{}';
ALTER TABLE assumptions ADD COLUMN IF NOT EXISTS inference_confidence FLOAT;
ALTER TABLE assumptions ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ;

-- Create indexes for evidence lookup
CREATE INDEX IF NOT EXISTS idx_methods_evidence_chunks ON methods USING gin(evidence_chunk_ids);
CREATE INDEX IF NOT EXISTS idx_assumptions_evidence_chunks ON assumptions USING gin(evidence_chunk_ids);
