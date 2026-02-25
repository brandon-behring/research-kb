"""QueueStore - CRUD operations for ingestion_queue table.

Provides async processing queue for discovered papers.
Supports status tracking, priority ordering, and retry logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from research_kb_common import StorageError, get_logger

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class QueueStatus:
    """Queue item status constants."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"


class QueueStore:
    """Storage operations for ingestion_queue table.

    Provides async processing queue for papers discovered via S2.
    """

    @staticmethod
    async def add(
        s2_paper_id: str,
        title: str,
        domain_id: str,
        pdf_url: Optional[str] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        authors: Optional[list[str]] = None,
        year: Optional[int] = None,
        venue: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """Add a paper to the ingestion queue.

        Args:
            s2_paper_id: Semantic Scholar paper ID (unique)
            title: Paper title
            pdf_url: URL to download PDF
            doi: DOI if available
            arxiv_id: arXiv ID if available
            authors: Author names
            year: Publication year
            venue: Publication venue
            domain_id: Target domain
            priority: Processing priority (higher = first)
            metadata: Additional S2 metadata

        Returns:
            UUID of created queue item

        Raises:
            StorageError: If paper already in queue (duplicate s2_paper_id)

        Example:
            >>> queue_id = await QueueStore.add(
            ...     s2_paper_id="649def34f8be52c8b66281af98ae884c09aef38b",
            ...     title="Double/Debiased Machine Learning",
            ...     pdf_url="https://arxiv.org/pdf/1607.00698.pdf",
            ...     authors=["Chernozhukov", "Chetverikov", "Demirer"],
            ...     year=2018,
            ...     priority=100,  # High priority for influential papers
            ... )
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                now = datetime.now(timezone.utc)
                queue_id = await conn.fetchval(
                    """
                    INSERT INTO ingestion_queue (
                        s2_paper_id, title, pdf_url, doi, arxiv_id,
                        authors, year, venue, domain_id,
                        status, priority, metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    RETURNING id
                    """,
                    s2_paper_id,
                    title,
                    pdf_url,
                    doi,
                    arxiv_id,
                    authors or [],
                    year,
                    venue,
                    domain_id,
                    QueueStatus.PENDING,
                    priority,
                    metadata or {},
                    now,
                    now,
                )

                logger.info(
                    "queue_item_added",
                    queue_id=str(queue_id),
                    s2_paper_id=s2_paper_id,
                    title=title[:50],
                )

                return UUID(queue_id)

        except Exception as e:
            if "unique" in str(e).lower():
                logger.warning(
                    "queue_item_duplicate",
                    s2_paper_id=s2_paper_id,
                )
                raise StorageError(f"Paper already in queue: {s2_paper_id}") from e
            logger.error("queue_add_failed", s2_paper_id=s2_paper_id, error=str(e))
            raise StorageError(f"Failed to add to queue: {e}") from e

    @staticmethod
    async def add_batch(papers: list[dict[str, Any]]) -> tuple[int, int]:
        """Add multiple papers to the queue, skipping duplicates.

        Args:
            papers: List of paper dicts with keys matching add() params

        Returns:
            Tuple of (added_count, skipped_count)

        Example:
            >>> papers = [
            ...     {"s2_paper_id": "abc", "title": "Paper 1"},
            ...     {"s2_paper_id": "def", "title": "Paper 2"},
            ... ]
            >>> added, skipped = await QueueStore.add_batch(papers)
        """
        added = 0
        skipped = 0

        for paper in papers:
            try:
                await QueueStore.add(
                    s2_paper_id=paper["s2_paper_id"],
                    title=paper["title"],
                    pdf_url=paper.get("pdf_url"),
                    doi=paper.get("doi"),
                    arxiv_id=paper.get("arxiv_id"),
                    authors=paper.get("authors"),
                    year=paper.get("year"),
                    venue=paper.get("venue"),
                    domain_id=paper["domain_id"],
                    priority=paper.get("priority", 0),
                    metadata=paper.get("metadata"),
                )
                added += 1
            except StorageError:
                skipped += 1

        logger.info("queue_batch_added", added=added, skipped=skipped)
        return added, skipped

    @staticmethod
    async def get_pending(
        limit: int = 50,
        domain_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get pending items from the queue.

        Returns items ordered by priority (desc) then created_at (asc).

        Args:
            limit: Maximum items to return
            domain_id: Optional filter by domain

        Returns:
            List of queue items

        Example:
            >>> pending = await QueueStore.get_pending(limit=10)
            >>> for item in pending:
            ...     print(f"Processing: {item['title']}")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                domain_filter = ""
                params: list[Any] = [QueueStatus.PENDING, limit]
                if domain_id:
                    domain_filter = "AND domain_id = $3"
                    params.append(domain_id)

                rows = await conn.fetch(
                    f"""
                    SELECT * FROM ingestion_queue
                    WHERE status = $1
                    {domain_filter}
                    ORDER BY priority DESC, created_at ASC
                    LIMIT $2
                    """,
                    *params,
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error("queue_get_pending_failed", error=str(e))
            raise StorageError(f"Failed to get pending items: {e}") from e

    @staticmethod
    async def update_status(
        queue_id: UUID,
        status: str,
        error_message: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> None:
        """Update queue item status.

        Args:
            queue_id: Queue item UUID
            status: New status
            error_message: Error message if failed
            pdf_path: Local path after download

        Example:
            >>> await QueueStore.update_status(
            ...     queue_id=uuid,
            ...     status=QueueStatus.COMPLETED,
            ...     pdf_path="/data/papers/paper.pdf",
            ... )
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                # Build update fields
                set_clauses = ["status = $1", "updated_at = $2"]
                params: list[Any] = [status, datetime.now(timezone.utc)]
                param_idx = 3

                if error_message is not None:
                    set_clauses.append(f"error_message = ${param_idx}")
                    params.append(error_message)
                    param_idx += 1

                if pdf_path is not None:
                    set_clauses.append(f"pdf_path = ${param_idx}")
                    params.append(pdf_path)
                    param_idx += 1

                # Increment retry_count on failure
                if status == QueueStatus.FAILED:
                    set_clauses.append("retry_count = retry_count + 1")

                params.append(queue_id)
                query = f"""
                    UPDATE ingestion_queue
                    SET {', '.join(set_clauses)}
                    WHERE id = ${param_idx}
                """

                result = await conn.execute(query, *params)

                if result == "UPDATE 0":
                    raise StorageError(f"Queue item not found: {queue_id}")

                logger.info(
                    "queue_status_updated",
                    queue_id=str(queue_id),
                    status=status,
                )

        except StorageError:
            raise
        except Exception as e:
            logger.error("queue_update_failed", queue_id=str(queue_id), error=str(e))
            raise StorageError(f"Failed to update queue status: {e}") from e

    @staticmethod
    async def get_by_s2_paper_id(s2_paper_id: str) -> Optional[dict[str, Any]]:
        """Get queue item by S2 paper ID.

        Args:
            s2_paper_id: Semantic Scholar paper ID

        Returns:
            Queue item dict or None

        Example:
            >>> item = await QueueStore.get_by_s2_paper_id("abc123")
            >>> if item:
            ...     print(f"Status: {item['status']}")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM ingestion_queue WHERE s2_paper_id = $1",
                    s2_paper_id,
                )

                return dict(row) if row else None

        except Exception as e:
            logger.error("queue_get_by_s2_id_failed", s2_paper_id=s2_paper_id, error=str(e))
            raise StorageError(f"Failed to get queue item: {e}") from e

    @staticmethod
    async def delete_completed(
        older_than_days: int = 7,
    ) -> int:
        """Delete completed items older than specified days.

        Args:
            older_than_days: Delete items older than this many days

        Returns:
            Number of items deleted

        Example:
            >>> deleted = await QueueStore.delete_completed(older_than_days=30)
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM ingestion_queue
                    WHERE status = $1
                    AND updated_at < NOW() - make_interval(days => $2)
                    """,
                    QueueStatus.COMPLETED,
                    older_than_days,
                )

                # Parse "DELETE N" result
                deleted = int(result.split()[-1]) if result else 0

                if deleted > 0:
                    logger.info(
                        "queue_cleanup",
                        deleted=deleted,
                        older_than_days=older_than_days,
                    )

                return deleted

        except Exception as e:
            logger.error("queue_cleanup_failed", error=str(e))
            raise StorageError(f"Failed to cleanup queue: {e}") from e

    @staticmethod
    async def get_stats() -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dict with counts by status and domain

        Example:
            >>> stats = await QueueStore.get_stats()
            >>> print(f"Pending: {stats['by_status'].get('pending', 0)}")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                # Get counts by status
                status_rows = await conn.fetch(
                    """
                    SELECT status, COUNT(*) as count
                    FROM ingestion_queue
                    GROUP BY status
                    """
                )

                # Get counts by domain
                domain_rows = await conn.fetch(
                    """
                    SELECT domain_id, COUNT(*) as count
                    FROM ingestion_queue
                    GROUP BY domain_id
                    """
                )

                # Get retry stats
                retry_row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE retry_count > 0) as with_retries,
                        MAX(retry_count) as max_retries,
                        AVG(retry_count) FILTER (WHERE retry_count > 0) as avg_retries
                    FROM ingestion_queue
                    """
                )

                return {
                    "by_status": {r["status"]: r["count"] for r in status_rows},
                    "by_domain": {r["domain_id"]: r["count"] for r in domain_rows},
                    "total": sum(r["count"] for r in status_rows),
                    "with_retries": retry_row["with_retries"] or 0,
                    "max_retries": retry_row["max_retries"] or 0,
                    "avg_retries": float(retry_row["avg_retries"] or 0),
                }

        except Exception as e:
            logger.error("queue_stats_failed", error=str(e))
            raise StorageError(f"Failed to get queue stats: {e}") from e

    @staticmethod
    async def retry_failed(
        max_retries: int = 3,
        domain_id: Optional[str] = None,
    ) -> int:
        """Reset failed items for retry (if under max retries).

        Args:
            max_retries: Maximum retry attempts
            domain_id: Optional filter by domain

        Returns:
            Number of items reset

        Example:
            >>> reset = await QueueStore.retry_failed(max_retries=3)
            >>> print(f"Reset {reset} items for retry")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                domain_filter = ""
                params: list[Any] = [
                    QueueStatus.PENDING,
                    QueueStatus.FAILED,
                    max_retries,
                ]
                if domain_id:
                    domain_filter = "AND domain_id = $4"
                    params.append(domain_id)

                result = await conn.execute(
                    f"""
                    UPDATE ingestion_queue
                    SET status = $1, error_message = NULL, updated_at = NOW()
                    WHERE status = $2
                    AND retry_count < $3
                    {domain_filter}
                    """,
                    *params,
                )

                reset = int(result.split()[-1]) if result else 0

                if reset > 0:
                    logger.info("queue_retry_reset", reset=reset)

                return reset

        except Exception as e:
            logger.error("queue_retry_failed", error=str(e))
            raise StorageError(f"Failed to retry failed items: {e}") from e
