-- Migration: Upgrade embedding dimension from 384 to 1024
-- Reason: BGE-large-en-v1.5 uses 1024-dim embeddings (not 384)
-- Date: 2025-11-29
-- IMPORTANT: This will drop all existing embeddings!

BEGIN;

-- Drop old embedding index
DROP INDEX IF EXISTS idx_chunks_embedding;

-- Alter column to new dimension
ALTER TABLE chunks
ALTER COLUMN embedding TYPE vector(1024);

-- Recreate index with new dimension
CREATE INDEX idx_chunks_embedding
ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Update comment to reflect correct dimension
COMMENT ON COLUMN chunks.embedding IS 'BGE-large-en-v1.5 1024-dim embeddings for semantic search';

COMMIT;
