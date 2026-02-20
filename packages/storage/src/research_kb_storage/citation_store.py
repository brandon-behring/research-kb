"""CitationStore - CRUD operations for citations table.

Provides:
- Create citation records from GROBID extraction
- Retrieve citations by ID or source
- List citations with filtering
- Batch operations for ingestion pipeline

Phase 1.5.2: Storage layer for extracted citations.
"""

import json
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import asyncpg
from research_kb_common import StorageError, get_logger
from research_kb_contracts import Citation

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class CitationStore:
    """Storage operations for Citation entities.

    All operations use the global connection pool.
    """

    @staticmethod
    async def create(
        source_id: UUID,
        raw_string: str,
        authors: Optional[list[str]] = None,
        title: Optional[str] = None,
        year: Optional[int] = None,
        venue: Optional[str] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        bibtex: Optional[str] = None,
        extraction_method: Optional[str] = None,
        confidence_score: Optional[float] = None,
        context: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Citation:
        """Create a new citation record.

        Args:
            source_id: Parent source UUID
            raw_string: Original citation text as it appeared
            authors: List of author names
            title: Citation title
            year: Publication year
            venue: Journal, conference, or publisher
            doi: DOI identifier
            arxiv_id: arXiv identifier
            bibtex: Generated BibTeX entry
            extraction_method: "grobid" or "manual"
            confidence_score: Extraction confidence (0.0 to 1.0)
            context: Citing sentence/context where citation appears
            metadata: Extensible JSONB metadata

        Returns:
            Created Citation

        Raises:
            StorageError: If creation fails

        Example:
            >>> citation = await CitationStore.create(
            ...     source_id=source.id,
            ...     raw_string="Pearl, J. (2009). Causality. Cambridge University Press.",
            ...     authors=["Pearl, Judea"],
            ...     title="Causality",
            ...     year=2009,
            ...     extraction_method="grobid",
            ... )
        """
        pool = await get_connection_pool()
        citation_id = uuid4()
        now = datetime.now(timezone.utc)

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb",
                    encoder=json.dumps,
                    decoder=json.loads,
                    schema="pg_catalog",
                )

                row = await conn.fetchrow(
                    """
                    INSERT INTO citations (
                        id, source_id, authors, title, year, venue,
                        doi, arxiv_id, raw_string, bibtex,
                        extraction_method, confidence_score, context,
                        metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    RETURNING *
                    """,
                    citation_id,
                    source_id,
                    authors or [],
                    title,
                    year,
                    venue,
                    doi,
                    arxiv_id,
                    raw_string,
                    bibtex,
                    extraction_method,
                    confidence_score,
                    context,
                    metadata or {},
                    now,
                )

                logger.info(
                    "citation_created",
                    citation_id=str(citation_id),
                    source_id=str(source_id),
                    has_doi=doi is not None,
                    has_arxiv=arxiv_id is not None,
                )

                return _row_to_citation(row)

        except Exception as e:
            logger.error("citation_creation_failed", source_id=str(source_id), error=str(e))
            raise StorageError(f"Failed to create citation: {e}") from e

    @staticmethod
    async def get_by_id(citation_id: UUID) -> Optional[Citation]:
        """Retrieve citation by ID.

        Args:
            citation_id: Citation UUID

        Returns:
            Citation if found, None otherwise
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb",
                    encoder=json.dumps,
                    decoder=json.loads,
                    schema="pg_catalog",
                )

                row = await conn.fetchrow(
                    "SELECT * FROM citations WHERE id = $1",
                    citation_id,
                )

                if row is None:
                    return None

                return _row_to_citation(row)

        except Exception as e:
            logger.error("citation_get_failed", citation_id=str(citation_id), error=str(e))
            raise StorageError(f"Failed to retrieve citation: {e}") from e

    @staticmethod
    async def list_by_source(
        source_id: UUID,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Citation]:
        """List citations for a source with pagination.

        Args:
            source_id: Source UUID
            limit: Maximum number of results (default: 1000)
            offset: Number of results to skip (default: 0)

        Returns:
            List of citations

        Example:
            >>> citations = await CitationStore.list_by_source(source.id)
            >>> print(f"Found {len(citations)} citations")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb",
                    encoder=json.dumps,
                    decoder=json.loads,
                    schema="pg_catalog",
                )

                rows = await conn.fetch(
                    """
                    SELECT * FROM citations
                    WHERE source_id = $1
                    ORDER BY created_at ASC
                    LIMIT $2 OFFSET $3
                    """,
                    source_id,
                    limit,
                    offset,
                )

                return [_row_to_citation(row) for row in rows]

        except Exception as e:
            logger.error("citation_list_failed", source_id=str(source_id), error=str(e))
            raise StorageError(f"Failed to list citations: {e}") from e

    @staticmethod
    async def count_by_source(source_id: UUID) -> int:
        """Count citations for a source.

        Args:
            source_id: Source UUID

        Returns:
            Number of citations
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM citations WHERE source_id = $1",
                    source_id,
                )

                return count

        except Exception as e:
            logger.error("citation_count_failed", source_id=str(source_id), error=str(e))
            raise StorageError(f"Failed to count citations: {e}") from e

    @staticmethod
    async def batch_create(citations_data: list[dict]) -> list[Citation]:
        """Batch create multiple citations (for ingestion pipeline).

        Args:
            citations_data: List of citation dictionaries with keys:
                           source_id, raw_string, authors, title, etc.

        Returns:
            List of created Citations

        Raises:
            StorageError: If batch creation fails

        Example:
            >>> citations = await CitationStore.batch_create([
            ...     {"source_id": source.id, "raw_string": "...", "authors": [...]},
            ...     {"source_id": source.id, "raw_string": "...", "authors": [...]},
            ... ])
        """
        if not citations_data:
            return []

        pool = await get_connection_pool()
        now = datetime.now(timezone.utc)
        created_citations = []

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb",
                    encoder=json.dumps,
                    decoder=json.loads,
                    schema="pg_catalog",
                )

                async with conn.transaction():
                    for cit_dict in citations_data:
                        citation_id = uuid4()

                        row = await conn.fetchrow(
                            """
                            INSERT INTO citations (
                                id, source_id, authors, title, year, venue,
                                doi, arxiv_id, raw_string, bibtex,
                                extraction_method, confidence_score, context,
                                metadata, created_at
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                            RETURNING *
                            """,
                            citation_id,
                            cit_dict["source_id"],
                            cit_dict.get("authors", []),
                            cit_dict.get("title"),
                            cit_dict.get("year"),
                            cit_dict.get("venue"),
                            cit_dict.get("doi"),
                            cit_dict.get("arxiv_id"),
                            cit_dict["raw_string"],
                            cit_dict.get("bibtex"),
                            cit_dict.get("extraction_method"),
                            cit_dict.get("confidence_score"),
                            cit_dict.get("context"),
                            cit_dict.get("metadata", {}),
                            now,
                        )

                        created_citations.append(_row_to_citation(row))

                logger.info(
                    "citations_batch_created",
                    count=len(created_citations),
                    source_id=str(citations_data[0]["source_id"]),
                )

                return created_citations

        except Exception as e:
            logger.error(
                "citation_batch_creation_failed",
                count=len(citations_data),
                error=str(e),
            )
            raise StorageError(f"Failed to batch create citations: {e}") from e

    @staticmethod
    async def delete(citation_id: UUID) -> bool:
        """Delete citation by ID.

        Args:
            citation_id: Citation UUID

        Returns:
            True if deleted, False if not found
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM citations WHERE id = $1",
                    citation_id,
                )

                deleted = result == "DELETE 1"

                if deleted:
                    logger.info("citation_deleted", citation_id=str(citation_id))
                else:
                    logger.warning("citation_not_found_for_delete", citation_id=str(citation_id))

                return deleted

        except Exception as e:
            logger.error("citation_delete_failed", citation_id=str(citation_id), error=str(e))
            raise StorageError(f"Failed to delete citation: {e}") from e

    @staticmethod
    async def find_by_doi(doi: str) -> Optional[Citation]:
        """Find citation by DOI.

        Args:
            doi: DOI identifier

        Returns:
            Citation if found, None otherwise
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb",
                    encoder=json.dumps,
                    decoder=json.loads,
                    schema="pg_catalog",
                )

                row = await conn.fetchrow(
                    "SELECT * FROM citations WHERE doi = $1 LIMIT 1",
                    doi,
                )

                if row is None:
                    return None

                return _row_to_citation(row)

        except Exception as e:
            logger.error("citation_find_by_doi_failed", doi=doi, error=str(e))
            raise StorageError(f"Failed to find citation by DOI: {e}") from e

    @staticmethod
    async def find_by_arxiv(arxiv_id: str) -> Optional[Citation]:
        """Find citation by arXiv ID.

        Args:
            arxiv_id: arXiv identifier

        Returns:
            Citation if found, None otherwise
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb",
                    encoder=json.dumps,
                    decoder=json.loads,
                    schema="pg_catalog",
                )

                row = await conn.fetchrow(
                    "SELECT * FROM citations WHERE arxiv_id = $1 LIMIT 1",
                    arxiv_id,
                )

                if row is None:
                    return None

                return _row_to_citation(row)

        except Exception as e:
            logger.error("citation_find_by_arxiv_failed", arxiv_id=arxiv_id, error=str(e))
            raise StorageError(f"Failed to find citation by arXiv ID: {e}") from e


def _row_to_citation(row: asyncpg.Record) -> Citation:
    """Convert database row to Citation model.

    Args:
        row: Database row from citations table

    Returns:
        Citation instance
    """
    return Citation(
        id=row["id"],
        source_id=row["source_id"],
        authors=row["authors"] or [],
        title=row["title"],
        year=row["year"],
        venue=row["venue"],
        doi=row["doi"],
        arxiv_id=row["arxiv_id"],
        raw_string=row["raw_string"],
        context=row.get("context"),  # May not exist in older schema
        bibtex=row["bibtex"],
        extraction_method=row["extraction_method"],
        confidence_score=row["confidence_score"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"],
    )
