"""DomainStore - CRUD operations for domains table.

Provides:
- List available knowledge domains
- Get domain by ID
- Register new domains
- Update domain configuration
- Get domain statistics
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
from research_kb_common import StorageError, get_logger
from research_kb_contracts import Domain

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class DomainStore:
    """Storage operations for Domain entities.

    All operations use the global connection pool.
    """

    @staticmethod
    async def list_all() -> list[Domain]:
        """List all registered domains.

        Returns:
            List of all domains

        Example:
            >>> domains = await DomainStore.list_all()
            >>> for d in domains:
            ...     print(f"{d.id}: {d.name}")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                rows = await conn.fetch("""
                    SELECT * FROM domains
                    WHERE status = 'active'
                    ORDER BY name
                    """)

                return [_row_to_domain(row) for row in rows]

        except Exception as e:
            logger.error("domain_list_failed", error=str(e))
            raise StorageError(f"Failed to list domains: {e}") from e

    @staticmethod
    async def get_by_id(domain_id: str) -> Optional[Domain]:
        """Retrieve domain by ID.

        Args:
            domain_id: Domain identifier (e.g., 'causal_inference')

        Returns:
            Domain if found, None otherwise

        Example:
            >>> domain = await DomainStore.get_by_id("causal_inference")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    "SELECT * FROM domains WHERE id = $1",
                    domain_id,
                )

                if row is None:
                    return None

                return _row_to_domain(row)

        except Exception as e:
            logger.error("domain_get_failed", domain_id=domain_id, error=str(e))
            raise StorageError(f"Failed to retrieve domain: {e}") from e

    @staticmethod
    async def create(
        domain_id: str,
        name: str,
        description: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        concept_types: Optional[list[str]] = None,
        relationship_types: Optional[list[str]] = None,
        default_fts_weight: float = 0.3,
        default_vector_weight: float = 0.7,
        default_graph_weight: float = 0.1,
        default_citation_weight: float = 0.15,
    ) -> Domain:
        """Register a new knowledge domain.

        Args:
            domain_id: Unique domain identifier (e.g., 'time_series')
            name: Human-readable name (e.g., 'Time Series')
            description: Extended description
            config: Domain-specific configuration
            concept_types: Allowed concept types
            relationship_types: Allowed relationship types
            default_fts_weight: Default FTS weight for search
            default_vector_weight: Default vector weight for search
            default_graph_weight: Default graph weight for search
            default_citation_weight: Default citation weight for search

        Returns:
            Created Domain

        Raises:
            StorageError: If creation fails (e.g., duplicate ID)

        Example:
            >>> domain = await DomainStore.create(
            ...     domain_id="time_series",
            ...     name="Time Series",
            ...     description="Forecasting and temporal modeling",
            ...     concept_types=["method", "assumption", "model"],
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
                    INSERT INTO domains (
                        id, name, description, config,
                        concept_types, relationship_types,
                        default_fts_weight, default_vector_weight,
                        default_graph_weight, default_citation_weight,
                        status, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    RETURNING *
                    """,
                    domain_id,
                    name,
                    description,
                    config or {},
                    concept_types or [],
                    relationship_types or [],
                    default_fts_weight,
                    default_vector_weight,
                    default_graph_weight,
                    default_citation_weight,
                    "active",
                    now,
                )

                logger.info(
                    "domain_created",
                    domain_id=domain_id,
                    name=name,
                )

                return _row_to_domain(row)

        except asyncpg.UniqueViolationError as e:
            logger.error("domain_creation_failed_duplicate", domain_id=domain_id, error=str(e))
            raise StorageError(f"Domain '{domain_id}' already exists") from e
        except Exception as e:
            logger.error("domain_creation_failed", error=str(e))
            raise StorageError(f"Failed to create domain: {e}") from e

    @staticmethod
    async def update_config(domain_id: str, config: dict[str, Any]) -> Domain:
        """Update domain configuration (JSONB merge).

        Args:
            domain_id: Domain identifier
            config: New configuration to merge with existing

        Returns:
            Updated Domain

        Raises:
            StorageError: If domain not found or update fails

        Example:
            >>> domain = await DomainStore.update_config(
            ...     domain_id="time_series",
            ...     config={"extraction_prompt_version": "v2"}
            ... )
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    """
                    UPDATE domains
                    SET config = config || $1
                    WHERE id = $2
                    RETURNING *
                    """,
                    config,
                    domain_id,
                )

                if row is None:
                    raise StorageError(f"Domain not found: {domain_id}")

                logger.info("domain_config_updated", domain_id=domain_id)
                return _row_to_domain(row)

        except StorageError:
            raise
        except Exception as e:
            logger.error("domain_update_failed", domain_id=domain_id, error=str(e))
            raise StorageError(f"Failed to update domain: {e}") from e

    @staticmethod
    async def get_stats(domain_id: str) -> Optional[dict[str, Any]]:
        """Get statistics for a domain.

        Args:
            domain_id: Domain identifier

        Returns:
            Dictionary with statistics:
                - domain_id: Domain identifier
                - name: Domain name
                - source_count: Number of sources
                - chunk_count: Number of chunks
                - concept_count: Number of concepts
            Returns None if domain not found.

        Example:
            >>> stats = await DomainStore.get_stats("causal_inference")
            >>> print(f"Sources: {stats['source_count']}")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                # Get domain info and counts in one query
                row = await conn.fetchrow(
                    """
                    SELECT
                        d.id as domain_id,
                        d.name,
                        COALESCE(s.source_count, 0) as source_count,
                        COALESCE(c.chunk_count, 0) as chunk_count,
                        COALESCE(co.concept_count, 0) as concept_count
                    FROM domains d
                    LEFT JOIN (
                        SELECT domain_id, COUNT(*) as source_count
                        FROM sources WHERE domain_id = $1 GROUP BY domain_id
                    ) s ON d.id = s.domain_id
                    LEFT JOIN (
                        SELECT domain_id, COUNT(*) as chunk_count
                        FROM chunks WHERE domain_id = $1 GROUP BY domain_id
                    ) c ON d.id = c.domain_id
                    LEFT JOIN (
                        SELECT domain_id, COUNT(*) as concept_count
                        FROM concepts WHERE domain_id = $1 GROUP BY domain_id
                    ) co ON d.id = co.domain_id
                    WHERE d.id = $1 AND d.status = 'active'
                    """,
                    domain_id,
                )

                if row is None:
                    return None

                return {
                    "domain_id": row["domain_id"],
                    "name": row["name"],
                    "source_count": row["source_count"],
                    "chunk_count": row["chunk_count"],
                    "concept_count": row["concept_count"],
                }

        except Exception as e:
            logger.error("domain_stats_failed", domain_id=domain_id, error=str(e))
            raise StorageError(f"Failed to get domain stats: {e}") from e

    @staticmethod
    async def get_all_stats() -> list[dict[str, Any]]:
        """Get statistics for all domains.

        Returns:
            List of statistics dictionaries for each domain

        Example:
            >>> all_stats = await DomainStore.get_all_stats()
            >>> for s in all_stats:
            ...     print(f"{s['domain_id']}: {s['source_count']} sources")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT
                        d.id as domain_id,
                        d.name,
                        COALESCE(s.source_count, 0) as source_count,
                        COALESCE(c.chunk_count, 0) as chunk_count,
                        COALESCE(co.concept_count, 0) as concept_count
                    FROM domains d
                    LEFT JOIN (
                        SELECT domain_id, COUNT(*) as source_count
                        FROM sources GROUP BY domain_id
                    ) s ON d.id = s.domain_id
                    LEFT JOIN (
                        SELECT domain_id, COUNT(*) as chunk_count
                        FROM chunks GROUP BY domain_id
                    ) c ON d.id = c.domain_id
                    LEFT JOIN (
                        SELECT domain_id, COUNT(*) as concept_count
                        FROM concepts GROUP BY domain_id
                    ) co ON d.id = co.domain_id
                    WHERE d.status = 'active'
                    ORDER BY d.name
                    """)

                return [
                    {
                        "domain_id": row["domain_id"],
                        "name": row["name"],
                        "source_count": row["source_count"],
                        "chunk_count": row["chunk_count"],
                        "concept_count": row["concept_count"],
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error("domain_all_stats_failed", error=str(e))
            raise StorageError(f"Failed to get domain stats: {e}") from e


def _row_to_domain(row: asyncpg.Record) -> Domain:
    """Convert database row to Domain model.

    Args:
        row: Database row from domains table

    Returns:
        Domain instance
    """
    return Domain(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        config=row["config"] if row["config"] else {},
        concept_types=row["concept_types"] if row["concept_types"] else [],
        relationship_types=(row["relationship_types"] if row["relationship_types"] else []),
        default_fts_weight=row["default_fts_weight"],
        default_vector_weight=row["default_vector_weight"],
        default_graph_weight=row["default_graph_weight"],
        default_citation_weight=row["default_citation_weight"],
        status=row["status"],
        created_at=row["created_at"],
    )
