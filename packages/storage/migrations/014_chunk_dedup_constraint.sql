-- Migration 014: Add unique constraint on (source_id, content_hash)
-- Prevents duplicate chunks from being created within the same source.
--
-- Prerequisite: Run `python scripts/dedup_chunks.py --apply` first to remove
-- existing duplicates, otherwise this migration will fail.

ALTER TABLE chunks
    ADD CONSTRAINT uq_chunks_source_content_hash UNIQUE (source_id, content_hash);

-- Existing idx_chunks_content_hash stays (useful for cross-source dedup queries)
