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
        domain_id: str = "causal_inference",
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
            domain_id: Knowledge domain (default: causal_inference)

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
            ...     metadata={"isbn": "978-0521895606"},
            ...     domain_id="causal_inference",
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
                        file_path, file_hash, metadata, domain_id,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
                    domain_id,
                    now,
                    now,
                )

                logger.info(
                    "source_created",
                    source_id=str(source_id),
                    source_type=source_type.value,
                    title=title,
                    domain_id=domain_id,
                )

                return _row_to_source(row)

        except asyncpg.UniqueViolationError as e:
            logger.error("source_creation_failed_duplicate", file_hash=file_hash, error=str(e))
            raise StorageError(f"Source with file_hash '{file_hash}' already exists") from e
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
    async def get_by_s2_paper_id(s2_paper_id: str) -> Optional[Source]:
        """Retrieve source by Semantic Scholar paper ID.

        Uses the indexed metadata->>'s2_paper_id' for fast lookup.

        Args:
            s2_paper_id: Semantic Scholar paper ID

        Returns:
            Source if found, None otherwise

        Example:
            >>> source = await SourceStore.get_by_s2_paper_id("649def34f8be52c8b66281af98ae884c09aef38b")
            >>> if source:
            ...     print(f"Paper already ingested: {source.title}")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM sources WHERE metadata->>'s2_paper_id' = $1",
                    s2_paper_id,
                )

                if row is None:
                    return None

                return _row_to_source(row)

        except Exception as e:
            logger.error("source_get_by_s2_id_failed", s2_paper_id=s2_paper_id, error=str(e))
            raise StorageError(f"Failed to retrieve source by S2 ID: {e}") from e

    @staticmethod
    async def get_by_doi(doi: str) -> Optional[Source]:
        """Retrieve source by DOI.

        Uses the indexed metadata->>'doi' for fast lookup.

        Args:
            doi: Digital Object Identifier

        Returns:
            Source if found, None otherwise

        Example:
            >>> source = await SourceStore.get_by_doi("10.1111/ectj.12097")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM sources WHERE metadata->>'doi' = $1",
                    doi,
                )

                if row is None:
                    return None

                return _row_to_source(row)

        except Exception as e:
            logger.error("source_get_by_doi_failed", doi=doi, error=str(e))
            raise StorageError(f"Failed to retrieve source by DOI: {e}") from e

    @staticmethod
    async def get_by_arxiv_id(arxiv_id: str) -> Optional[Source]:
        """Retrieve source by arXiv ID.

        Uses the indexed metadata->>'arxiv_id' for fast lookup.

        Args:
            arxiv_id: arXiv identifier (e.g., "1607.00698")

        Returns:
            Source if found, None otherwise

        Example:
            >>> source = await SourceStore.get_by_arxiv_id("1607.00698")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM sources WHERE metadata->>'arxiv_id' = $1",
                    arxiv_id,
                )

                if row is None:
                    return None

                return _row_to_source(row)

        except Exception as e:
            logger.error("source_get_by_arxiv_failed", arxiv_id=arxiv_id, error=str(e))
            raise StorageError(f"Failed to retrieve source by arXiv ID: {e}") from e

    @staticmethod
    async def get_existing_identifiers(
        domain_id: Optional[str] = None,
    ) -> dict[str, set[str]]:
        """Get all existing identifiers for fast deduplication.

        Returns sets of S2 paper IDs, DOIs, and arXiv IDs that are already
        in the corpus. Use this for batch deduplication when discovering
        new papers.

        Args:
            domain_id: Optional filter by domain (None = all domains)

        Returns:
            Dict with keys 's2_ids', 'dois', 'arxiv_ids', each a set of strings

        Example:
            >>> existing = await SourceStore.get_existing_identifiers("causal_inference")
            >>> new_papers = [p for p in discovered if p.s2_id not in existing['s2_ids']]
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                # Build query with optional domain filter
                domain_filter = ""
                params: list = []
                if domain_id:
                    domain_filter = "AND domain_id = $1"
                    params.append(domain_id)

                # Get all S2 paper IDs
                s2_rows = await conn.fetch(
                    f"""
                    SELECT metadata->>'s2_paper_id' as s2_id
                    FROM sources
                    WHERE metadata->>'s2_paper_id' IS NOT NULL
                    {domain_filter}
                    """,
                    *params,
                )

                # Get all DOIs
                doi_rows = await conn.fetch(
                    f"""
                    SELECT metadata->>'doi' as doi
                    FROM sources
                    WHERE metadata->>'doi' IS NOT NULL
                    {domain_filter}
                    """,
                    *params,
                )

                # Get all arXiv IDs
                arxiv_rows = await conn.fetch(
                    f"""
                    SELECT metadata->>'arxiv_id' as arxiv_id
                    FROM sources
                    WHERE metadata->>'arxiv_id' IS NOT NULL
                    {domain_filter}
                    """,
                    *params,
                )

                return {
                    "s2_ids": {row["s2_id"] for row in s2_rows},
                    "dois": {row["doi"] for row in doi_rows},
                    "arxiv_ids": {row["arxiv_id"] for row in arxiv_rows},
                }

        except Exception as e:
            logger.error("source_get_identifiers_failed", domain_id=domain_id, error=str(e))
            raise StorageError(f"Failed to get existing identifiers: {e}") from e

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
                    logger.warning("source_not_found_for_delete", source_id=str(source_id))

                return deleted

        except Exception as e:
            logger.error("source_delete_failed", source_id=str(source_id), error=str(e))
            raise StorageError(f"Failed to delete source: {e}") from e

    @staticmethod
    async def list_all(
        limit: int = 100,
        offset: int = 0,
        source_type: Optional[SourceType] = None,
        domain_id: Optional[str] = None,
    ) -> list[Source]:
        """List all sources with optional filtering and pagination.

        Args:
            limit: Maximum number of results (default: 100)
            offset: Number of results to skip (default: 0)
            source_type: Optional filter by source type
            domain_id: Optional filter by domain (None = all domains)

        Returns:
            List of sources

        Example:
            >>> sources = await SourceStore.list_all(limit=50)
            >>> papers = await SourceStore.list_all(source_type=SourceType.PAPER)
            >>> ts_sources = await SourceStore.list_all(domain_id="time_series")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                # Build query with optional filters
                conditions = []
                params = []
                param_idx = 1

                if source_type:
                    conditions.append(f"source_type = ${param_idx}")
                    params.append(source_type.value)
                    param_idx += 1

                if domain_id:
                    conditions.append(f"domain_id = ${param_idx}")
                    params.append(domain_id)
                    param_idx += 1

                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

                params.extend([limit, offset])
                query = f"""
                    SELECT * FROM sources
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """

                rows = await conn.fetch(query, *params)
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
            logger.error("source_list_failed", source_type=source_type.value, error=str(e))
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
        domain_id=row.get("domain_id", "causal_inference"),
        file_path=row["file_path"],
        file_hash=row["file_hash"],
        metadata=row["metadata"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
