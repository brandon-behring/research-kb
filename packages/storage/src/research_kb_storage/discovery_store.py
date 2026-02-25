"""DiscoveryStore - CRUD operations for discovery_log table.

Provides audit logging for S2 auto-discovery operations.
Tracks discovery method, results, and performance metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from research_kb_common import StorageError, get_logger

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class DiscoveryMethod:
    """Discovery method constants."""

    KEYWORD_SEARCH = "keyword_search"
    AUTHOR_SEARCH = "author_search"
    CITATION_TRAVERSE = "citation_traverse"
    TOPIC_BATCH = "topic_batch"


class DiscoveryStore:
    """Storage operations for discovery_log table.

    Provides audit logging for all S2 discovery operations.
    """

    @staticmethod
    async def log_discovery(
        discovery_method: str,
        domain_id: str,
        query: Optional[str] = None,
        papers_found: int = 0,
        papers_acquired: int = 0,
        papers_ingested: int = 0,
        papers_skipped: int = 0,
        papers_failed: int = 0,
        duration_seconds: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """Log a discovery operation.

        Args:
            discovery_method: Method used (keyword_search, author_search, etc.)
            query: Search query or paper ID for traversal
            domain_id: Target domain
            papers_found: Total papers returned by S2
            papers_acquired: PDFs successfully downloaded
            papers_ingested: Successfully processed into corpus
            papers_skipped: Skipped (duplicates)
            papers_failed: Failed to process
            duration_seconds: Operation duration
            metadata: Additional context (year_from, min_citations, etc.)

        Returns:
            UUID of created log entry

        Example:
            >>> log_id = await DiscoveryStore.log_discovery(
            ...     discovery_method=DiscoveryMethod.KEYWORD_SEARCH,
            ...     query="double machine learning",
            ...     papers_found=50,
            ...     papers_ingested=12,
            ...     papers_skipped=35,
            ...     duration_seconds=45.2,
            ...     metadata={"year_from": 2020, "min_citations": 50},
            ... )
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                log_id = await conn.fetchval(
                    """
                    INSERT INTO discovery_log (
                        discovery_method, query, domain_id,
                        papers_found, papers_acquired, papers_ingested,
                        papers_skipped, papers_failed,
                        duration_seconds, metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                    """,
                    discovery_method,
                    query,
                    domain_id,
                    papers_found,
                    papers_acquired,
                    papers_ingested,
                    papers_skipped,
                    papers_failed,
                    duration_seconds,
                    metadata or {},
                    datetime.now(timezone.utc),
                )

                logger.info(
                    "discovery_logged",
                    log_id=str(log_id),
                    method=discovery_method,
                    papers_found=papers_found,
                    papers_ingested=papers_ingested,
                )

                return UUID(log_id)

        except Exception as e:
            logger.error("discovery_log_failed", method=discovery_method, error=str(e))
            raise StorageError(f"Failed to log discovery: {e}") from e

    @staticmethod
    async def get_recent(
        limit: int = 20,
        domain_id: Optional[str] = None,
        discovery_method: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get recent discovery log entries.

        Args:
            limit: Maximum entries to return
            domain_id: Optional filter by domain
            discovery_method: Optional filter by method

        Returns:
            List of log entries (most recent first)

        Example:
            >>> recent = await DiscoveryStore.get_recent(limit=10)
            >>> for entry in recent:
            ...     print(f"{entry['discovery_method']}: {entry['papers_ingested']} ingested")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                # Build query with optional filters
                conditions = []
                params: list[Any] = []
                param_idx = 1

                if domain_id:
                    conditions.append(f"domain_id = ${param_idx}")
                    params.append(domain_id)
                    param_idx += 1

                if discovery_method:
                    conditions.append(f"discovery_method = ${param_idx}")
                    params.append(discovery_method)
                    param_idx += 1

                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

                params.append(limit)
                query = f"""
                    SELECT * FROM discovery_log
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ${param_idx}
                """

                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error("discovery_get_recent_failed", error=str(e))
            raise StorageError(f"Failed to get recent discoveries: {e}") from e

    @staticmethod
    async def get_stats(
        domain_id: Optional[str] = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get discovery statistics.

        Args:
            domain_id: Optional filter by domain
            days: Look back period in days

        Returns:
            Dict with:
                - total_discoveries: Count of discovery operations
                - total_papers_found: Sum of papers_found
                - total_papers_ingested: Sum of papers_ingested
                - total_papers_skipped: Sum of papers_skipped
                - by_method: Breakdown by discovery method

        Example:
            >>> stats = await DiscoveryStore.get_stats(days=7)
            >>> print(f"Last 7 days: {stats['total_papers_ingested']} papers ingested")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                # Build domain filter
                domain_filter = ""
                params: list[Any] = [days]
                if domain_id:
                    domain_filter = "AND domain_id = $2"
                    params.append(domain_id)

                # Get aggregate stats
                # Note: Use make_interval() for parameterized days
                row = await conn.fetchrow(
                    f"""
                    SELECT
                        COUNT(*) as total_discoveries,
                        COALESCE(SUM(papers_found), 0) as total_papers_found,
                        COALESCE(SUM(papers_acquired), 0) as total_papers_acquired,
                        COALESCE(SUM(papers_ingested), 0) as total_papers_ingested,
                        COALESCE(SUM(papers_skipped), 0) as total_papers_skipped,
                        COALESCE(SUM(papers_failed), 0) as total_papers_failed,
                        COALESCE(AVG(duration_seconds), 0) as avg_duration_seconds
                    FROM discovery_log
                    WHERE created_at >= NOW() - make_interval(days => $1)
                    {domain_filter}
                    """,
                    *params,
                )

                # Get breakdown by method
                method_rows = await conn.fetch(
                    f"""
                    SELECT
                        discovery_method,
                        COUNT(*) as count,
                        COALESCE(SUM(papers_ingested), 0) as papers_ingested
                    FROM discovery_log
                    WHERE created_at >= NOW() - make_interval(days => $1)
                    {domain_filter}
                    GROUP BY discovery_method
                    ORDER BY count DESC
                    """,
                    *params,
                )

                return {
                    "total_discoveries": row["total_discoveries"],
                    "total_papers_found": row["total_papers_found"],
                    "total_papers_acquired": row["total_papers_acquired"],
                    "total_papers_ingested": row["total_papers_ingested"],
                    "total_papers_skipped": row["total_papers_skipped"],
                    "total_papers_failed": row["total_papers_failed"],
                    "avg_duration_seconds": float(row["avg_duration_seconds"]),
                    "days": days,
                    "by_method": {
                        r["discovery_method"]: {
                            "count": r["count"],
                            "papers_ingested": r["papers_ingested"],
                        }
                        for r in method_rows
                    },
                }

        except Exception as e:
            logger.error("discovery_stats_failed", error=str(e))
            raise StorageError(f"Failed to get discovery stats: {e}") from e
