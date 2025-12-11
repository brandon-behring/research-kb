"""SourceStore - CRUD operations for sources table.

Provides:
- Create source records
- Retrieve sources by ID or file hash
- Update source metadata
- Delete sources (cascades to chunks)
- List sources with filtering
"""

import json
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import asyncpg
from research_kb_common import StorageError, get_logger
from research_kb_contracts import Source, SourceMetadata, SourceType

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class SourceStore:
    """Storage operations for Source entities.

    All operations use the global connection pool.
    """

    @staticmethod
    async def create(
        source_type: SourceType,
        title: str,
        file_hash: str,
        authors: Optional[list[str]] = None,
        year: Optional[int] = None,
        file_path: Optional[str] = None,
        metadata: Optional[SourceMetadata] = None,
    ) -> Source:
        """Create a new source record.

        Args:
            source_type: Type of source (textbook, paper, code_repo)
            title: Source title
            file_hash: SHA256 hash for deduplication (must be unique)
            authors: List of author names
            year: Publication year
            file_path: Path to source file
            metadata: Extensible JSONB metadata

        Returns:
            Created Source

        Raises:
            StorageError: If creation fails (e.g., duplicate file_hash)

        Example:
            >>> source = await SourceStore.create(
            ...     source_type=SourceType.TEXTBOOK,
            ...     title="Causality",
            ...     file_hash="sha256:abc123",
            ...     authors=["Judea Pearl"],
            ...     metadata={"isbn": "978-0521895606"}
            ... )
        """
        pool = await get_connection_pool()
        source_id = uuid4()
        now = datetime.now(timezone.utc)

        try:
            async with pool.acquire() as conn:
                # Set JSON codec for this connection
                await conn.set_type_codec(
                    "jsonb",
                    encoder=json.dumps,
                    decoder=json.loads,
                    schema="pg_catalog",
                )

                row = await conn.fetchrow(
                    """
                    INSERT INTO sources (
                        id, source_type, title, authors, year,
                        file_path, file_hash, metadata,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING *
                    """,
                    source_id,
                    source_type.value,
                    title,
                    authors or [],
                    year,
                    file_path,
                    file_hash,
                    metadata or {},  # Pass dict directly - asyncpg will encode
                    now,
                    now,
                )

                logger.info(
                    "source_created",
                    source_id=str(source_id),
                    source_type=source_type.value,
                    title=title,
                )

                return _row_to_source(row)

        except asyncpg.UniqueViolationError as e:
            logger.error(
                "source_creation_failed_duplicate", file_hash=file_hash, error=str(e)
            )
            raise StorageError(
                f"Source with file_hash '{file_hash}' already exists"
            ) from e
        except Exception as e:
            logger.error("source_creation_failed", error=str(e))
            raise StorageError(f"Failed to create source: {e}") from e

    @staticmethod
    async def get_by_id(source_id: UUID) -> Optional[Source]:
        """Retrieve source by ID.

        Args:
            source_id: Source UUID

        Returns:
            Source if found, None otherwise

        Example:
            >>> source = await SourceStore.get_by_id(uuid.UUID("..."))
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM sources WHERE id = $1",
                    source_id,
                )

                if row is None:
                    return None

                return _row_to_source(row)

        except Exception as e:
            logger.error("source_get_failed", source_id=str(source_id), error=str(e))
            raise StorageError(f"Failed to retrieve source: {e}") from e

    @staticmethod
    async def get_by_file_hash(file_hash: str) -> Optional[Source]:
        """Retrieve source by file hash.

        Useful for checking if a file has already been ingested.

        Args:
            file_hash: SHA256 file hash

        Returns:
            Source if found, None otherwise

        Example:
            >>> source = await SourceStore.get_by_file_hash("sha256:abc123")
            >>> if source:
            ...     print(f"File already ingested: {source.id}")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM sources WHERE file_hash = $1",
                    file_hash,
                )

                if row is None:
                    return None

                return _row_to_source(row)

        except Exception as e:
            logger.error("source_get_by_hash_failed", file_hash=file_hash, error=str(e))
            raise StorageError(f"Failed to retrieve source by hash: {e}") from e

    @staticmethod
    async def update_metadata(source_id: UUID, metadata: SourceMetadata) -> Source:
        """Update source metadata (JSONB merge).

        Args:
            source_id: Source UUID
            metadata: New metadata to merge with existing

        Returns:
            Updated Source

        Raises:
            StorageError: If source not found or update fails

        Example:
            >>> source = await SourceStore.update_metadata(
            ...     source_id=uuid.UUID("..."),
            ...     metadata={"citations_count": 1200}
            ... )
        """
        pool = await get_connection_pool()
        now = datetime.now(timezone.utc)

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    """
                    UPDATE sources
                    SET metadata = metadata || $1,
                        updated_at = $2
                    WHERE id = $3
                    RETURNING *
                    """,
                    metadata,  # Pass dict directly
                    now,
                    source_id,
                )

                if row is None:
                    raise StorageError(f"Source not found: {source_id}")

                logger.info("source_metadata_updated", source_id=str(source_id))
                return _row_to_source(row)

        except StorageError:
            raise
        except Exception as e:
            logger.error("source_update_failed", source_id=str(source_id), error=str(e))
            raise StorageError(f"Failed to update source: {e}") from e

    @staticmethod
    async def delete(source_id: UUID) -> bool:
        """Delete source and all associated chunks (CASCADE).

        Args:
            source_id: Source UUID

        Returns:
            True if deleted, False if not found

        Example:
            >>> deleted = await SourceStore.delete(uuid.UUID("..."))
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM sources WHERE id = $1",
                    source_id,
                )

                deleted = result == "DELETE 1"

                if deleted:
                    logger.info("source_deleted", source_id=str(source_id))
                else:
                    logger.warning(
                        "source_not_found_for_delete", source_id=str(source_id)
                    )

                return deleted

        except Exception as e:
            logger.error("source_delete_failed", source_id=str(source_id), error=str(e))
            raise StorageError(f"Failed to delete source: {e}") from e

    @staticmethod
    async def list_all(
        limit: int = 100,
        offset: int = 0,
        source_type: Optional[SourceType] = None,
    ) -> list[Source]:
        """List all sources with optional filtering and pagination.

        Args:
            limit: Maximum number of results (default: 100)
            offset: Number of results to skip (default: 0)
            source_type: Optional filter by source type

        Returns:
            List of sources

        Example:
            >>> sources = await SourceStore.list_all(limit=50)
            >>> papers = await SourceStore.list_all(source_type=SourceType.PAPER)
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                if source_type:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM sources
                        WHERE source_type = $1
                        ORDER BY created_at DESC
                        LIMIT $2 OFFSET $3
                        """,
                        source_type.value,
                        limit,
                        offset,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM sources
                        ORDER BY created_at DESC
                        LIMIT $1 OFFSET $2
                        """,
                        limit,
                        offset,
                    )

                return [_row_to_source(row) for row in rows]

        except Exception as e:
            logger.error("source_list_all_failed", error=str(e))
            raise StorageError(f"Failed to list sources: {e}") from e

    @staticmethod
    async def list_by_type(
        source_type: SourceType,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Source]:
        """List sources by type with pagination.

        Args:
            source_type: Filter by source type
            limit: Maximum number of results (default: 100)
            offset: Number of results to skip (default: 0)

        Returns:
            List of sources

        Example:
            >>> textbooks = await SourceStore.list_by_type(
            ...     source_type=SourceType.TEXTBOOK,
            ...     limit=50
            ... )
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                rows = await conn.fetch(
                    """
                    SELECT * FROM sources
                    WHERE source_type = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    source_type.value,
                    limit,
                    offset,
                )

                return [_row_to_source(row) for row in rows]

        except Exception as e:
            logger.error(
                "source_list_failed", source_type=source_type.value, error=str(e)
            )
            raise StorageError(f"Failed to list sources: {e}") from e


def _row_to_source(row: asyncpg.Record) -> Source:
    """Convert database row to Source model.

    Args:
        row: Database row from sources table

    Returns:
        Source instance
    """
    return Source(
        id=row["id"],
        source_type=SourceType(row["source_type"]),
        title=row["title"],
        authors=row["authors"],
        year=row["year"],
        file_path=row["file_path"],
        file_hash=row["file_hash"],
        metadata=row["metadata"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
