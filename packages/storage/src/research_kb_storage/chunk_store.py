"""ChunkStore - CRUD operations for chunks table.

Provides:
- Create chunk records with embeddings
- Retrieve chunks by ID or source
- Update chunk metadata and embeddings
- Delete chunks
- Batch operations for ingestion pipeline
"""

import json
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import asyncpg
from pgvector.asyncpg import register_vector
from research_kb_common import StorageError, get_logger
from research_kb_contracts import Chunk, ChunkMetadata

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class ChunkStore:
    """Storage operations for Chunk entities.

    All operations use the global connection pool.
    """

    @staticmethod
    async def create(
        source_id: UUID,
        content: str,
        content_hash: str,
        domain_id: str,
        location: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        embedding: Optional[list[float]] = None,
        metadata: Optional[ChunkMetadata] = None,
    ) -> Chunk:
        """Create a new chunk record.

        Args:
            source_id: Parent source UUID
            content: Chunk text content
            content_hash: Hash of content for deduplication
            location: Human-readable location (e.g., "Chapter 3, p. 73")
            page_start: Starting page number
            page_end: Ending page number
            embedding: 1024-dim BGE-large-en-v1.5 embedding vector
            metadata: Extensible JSONB metadata
            domain_id: Knowledge domain (inherited from source)

        Returns:
            Created Chunk

        Raises:
            StorageError: If creation fails

        Example:
            >>> chunk = await ChunkStore.create(
            ...     source_id=source.id,
            ...     content="The backdoor criterion...",
            ...     content_hash="sha256:chunk123",
            ...     location="Chapter 3, p. 73",
            ...     embedding=[0.1] * 384,
            ...     metadata={"chunk_type": "theorem"},
            ...     domain_id="causal_inference",
            ... )
        """
        pool = await get_connection_pool()
        chunk_id = uuid4()
        now = datetime.now(timezone.utc)

        try:
            async with pool.acquire() as conn:
                # Register pgvector type
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    """
                    INSERT INTO chunks (
                        id, source_id, content, content_hash,
                        location, page_start, page_end,
                        embedding, metadata, domain_id, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (source_id, content_hash) DO NOTHING
                    RETURNING *
                    """,
                    chunk_id,
                    source_id,
                    content,
                    content_hash,
                    location,
                    page_start,
                    page_end,
                    embedding,  # pgvector handles conversion
                    metadata or {},  # Pass dict directly
                    domain_id,
                    now,
                )

                if row is None:
                    # Conflict: chunk already exists, fetch existing
                    row = await conn.fetchrow(
                        "SELECT * FROM chunks WHERE source_id = $1 AND content_hash = $2",
                        source_id,
                        content_hash,
                    )
                    logger.info(
                        "chunk_already_exists",
                        chunk_id=str(row["id"]),
                        source_id=str(source_id),
                    )
                else:
                    logger.info(
                        "chunk_created",
                        chunk_id=str(chunk_id),
                        source_id=str(source_id),
                        content_length=len(content),
                        has_embedding=embedding is not None,
                        domain_id=domain_id,
                    )

                return await _row_to_chunk(row, conn)

        except Exception as e:
            logger.error("chunk_creation_failed", error=str(e))
            raise StorageError(f"Failed to create chunk: {e}") from e

    @staticmethod
    async def get_by_id(chunk_id: UUID) -> Optional[Chunk]:
        """Retrieve chunk by ID.

        Args:
            chunk_id: Chunk UUID

        Returns:
            Chunk if found, None otherwise
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM chunks WHERE id = $1",
                    chunk_id,
                )

                if row is None:
                    return None

                return await _row_to_chunk(row, conn)

        except Exception as e:
            logger.error("chunk_get_failed", chunk_id=str(chunk_id), error=str(e))
            raise StorageError(f"Failed to retrieve chunk: {e}") from e

    @staticmethod
    async def list_by_source(
        source_id: UUID,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Chunk]:
        """List chunks for a source with pagination.

        Args:
            source_id: Source UUID
            limit: Maximum number of results (default: 1000)
            offset: Number of results to skip (default: 0)

        Returns:
            List of chunks

        Example:
            >>> chunks = await ChunkStore.list_by_source(source.id, limit=100)
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                rows = await conn.fetch(
                    """
                    SELECT * FROM chunks
                    WHERE source_id = $1
                    ORDER BY created_at ASC
                    LIMIT $2 OFFSET $3
                    """,
                    source_id,
                    limit,
                    offset,
                )

                return [await _row_to_chunk(row, conn) for row in rows]

        except Exception as e:
            logger.error("chunk_list_failed", source_id=str(source_id), error=str(e))
            raise StorageError(f"Failed to list chunks: {e}") from e

    @staticmethod
    async def list_all(
        limit: int = 10000,
        offset: int = 0,
        domain_id: Optional[str] = None,
    ) -> list[Chunk]:
        """List all chunks with pagination.

        Args:
            limit: Maximum number of results (default: 10000)
            offset: Number of results to skip (default: 0)
            domain_id: Optional filter by domain (None = all domains)

        Returns:
            List of chunks ordered by creation time
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                if domain_id:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM chunks
                        WHERE domain_id = $1
                        ORDER BY created_at ASC
                        LIMIT $2 OFFSET $3
                        """,
                        domain_id,
                        limit,
                        offset,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM chunks
                        ORDER BY created_at ASC
                        LIMIT $1 OFFSET $2
                        """,
                        limit,
                        offset,
                    )

                return [await _row_to_chunk(row, conn) for row in rows]

        except Exception as e:
            logger.error("chunk_list_all_failed", error=str(e))
            raise StorageError(f"Failed to list chunks: {e}") from e

    @staticmethod
    async def update_embedding(chunk_id: UUID, embedding: list[float]) -> Chunk:
        """Update chunk embedding vector.

        Args:
            chunk_id: Chunk UUID
            embedding: 1024-dim embedding vector (BGE-large-en-v1.5)

        Returns:
            Updated Chunk

        Raises:
            StorageError: If chunk not found or update fails
        """
        if len(embedding) != 1024:
            raise ValueError(
                f"Embedding must be 1024 dimensions (BGE-large-en-v1.5), got {len(embedding)}"
            )

        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    """
                    UPDATE chunks
                    SET embedding = $1
                    WHERE id = $2
                    RETURNING *
                    """,
                    embedding,
                    chunk_id,
                )

                if row is None:
                    raise StorageError(f"Chunk not found: {chunk_id}")

                logger.info("chunk_embedding_updated", chunk_id=str(chunk_id))
                return await _row_to_chunk(row, conn)

        except StorageError:
            raise
        except Exception as e:
            logger.error("chunk_update_failed", chunk_id=str(chunk_id), error=str(e))
            raise StorageError(f"Failed to update chunk: {e}") from e

    @staticmethod
    async def batch_create(chunks_data: list[dict]) -> list[Chunk]:
        """Batch create multiple chunks (for ingestion pipeline).

        Args:
            chunks_data: List of chunk dictionaries with keys:
                        source_id, content, content_hash, location, etc.

        Returns:
            List of created Chunks

        Raises:
            StorageError: If batch creation fails

        Example:
            >>> chunks = await ChunkStore.batch_create([
            ...     {"source_id": source.id, "content": "...", "content_hash": "..."},
            ...     {"source_id": source.id, "content": "...", "content_hash": "..."},
            ... ])
        """
        if not chunks_data:
            return []

        pool = await get_connection_pool()
        now = datetime.now(timezone.utc)
        created_chunks = []

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                async with conn.transaction():
                    for chunk_dict in chunks_data:
                        chunk_id = uuid4()

                        row = await conn.fetchrow(
                            """
                            INSERT INTO chunks (
                                id, source_id, content, content_hash,
                                location, page_start, page_end,
                                embedding, metadata, domain_id, created_at
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            ON CONFLICT (source_id, content_hash) DO NOTHING
                            RETURNING *
                            """,
                            chunk_id,
                            chunk_dict["source_id"],
                            chunk_dict["content"],
                            chunk_dict["content_hash"],
                            chunk_dict.get("location"),
                            chunk_dict.get("page_start"),
                            chunk_dict.get("page_end"),
                            chunk_dict.get("embedding"),
                            chunk_dict.get("metadata", {}),  # Pass dict directly
                            chunk_dict.get("domain_id", "causal_inference"),
                            now,
                        )

                        if row is None:
                            # Conflict: fetch existing chunk
                            row = await conn.fetchrow(
                                "SELECT * FROM chunks WHERE source_id = $1 AND content_hash = $2",
                                chunk_dict["source_id"],
                                chunk_dict["content_hash"],
                            )

                        created_chunks.append(await _row_to_chunk(row, conn))

                logger.info(
                    "chunks_batch_created",
                    count=len(created_chunks),
                    source_id=str(chunks_data[0]["source_id"]),
                )

                return created_chunks

        except Exception as e:
            logger.error("chunk_batch_creation_failed", count=len(chunks_data), error=str(e))
            raise StorageError(f"Failed to batch create chunks: {e}") from e

    @staticmethod
    async def delete(chunk_id: UUID) -> bool:
        """Delete chunk by ID.

        Args:
            chunk_id: Chunk UUID

        Returns:
            True if deleted, False if not found
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM chunks WHERE id = $1",
                    chunk_id,
                )

                deleted: bool = result == "DELETE 1"

                if deleted:
                    logger.info("chunk_deleted", chunk_id=str(chunk_id))
                else:
                    logger.warning("chunk_not_found_for_delete", chunk_id=str(chunk_id))

                return deleted

        except Exception as e:
            logger.error("chunk_delete_failed", chunk_id=str(chunk_id), error=str(e))
            raise StorageError(f"Failed to delete chunk: {e}") from e

    @staticmethod
    async def count_by_source(source_id: UUID) -> int:
        """Count chunks for a source.

        Args:
            source_id: Source UUID

        Returns:
            Number of chunks
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM chunks WHERE source_id = $1",
                    source_id,
                )

                return int(count)

        except Exception as e:
            logger.error("chunk_count_failed", source_id=str(source_id), error=str(e))
            raise StorageError(f"Failed to count chunks: {e}") from e


async def _row_to_chunk(row: asyncpg.Record, conn: asyncpg.Connection) -> Chunk:
    """Convert database row to Chunk model.

    Args:
        row: Database row from chunks table
        conn: Database connection (for vector conversion)

    Returns:
        Chunk instance
    """
    # Convert embedding from pgvector to list[float]
    embedding = None
    if row["embedding"] is not None:
        # pgvector returns a list-like object that needs conversion
        embedding = list(row["embedding"])

    return Chunk(
        id=row["id"],
        source_id=row["source_id"],
        domain_id=row["domain_id"],
        content=row["content"],
        content_hash=row["content_hash"],
        location=row["location"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        embedding=embedding,
        metadata=row["metadata"],
        created_at=row["created_at"],
    )
