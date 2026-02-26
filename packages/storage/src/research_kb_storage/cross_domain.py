"""Cross-domain concept discovery and linking.

Discovers semantic connections between concepts across different knowledge domains
using embedding similarity. Populates the cross_domain_links table.

Link Types:
    - EQUIVALENT: Same concept, different terminology (e.g., "ATE" ↔ "treatment effect")
    - ANALOGOUS: Similar role in different contexts (e.g., "IV" in economics ↔ "random assignment" in stats)
    - RELATED: Semantically related but distinct concepts
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from research_kb_common import StorageError, get_logger

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class CrossDomainLinkType:
    """Cross-domain link type constants."""

    EQUIVALENT = "EQUIVALENT"  # Same concept, different terminology
    ANALOGOUS = "ANALOGOUS"  # Similar role in different contexts
    RELATED = "RELATED"  # Semantically related but distinct


class CrossDomainStore:
    """Storage operations for cross-domain concept links.

    Provides discovery and storage of semantic connections between
    concepts across different knowledge domains.
    """

    @staticmethod
    async def discover_links(
        source_domain: str,
        target_domain: str,
        similarity_threshold: float = 0.85,
        limit_per_concept: int = 5,
        source_concept_limit: int = 1000,
        min_relationship_count: int = 2,
    ) -> list[dict[str, Any]]:
        """Discover cross-domain concept links using embedding similarity.

        Finds concepts in source_domain that have semantically similar
        counterparts in target_domain based on embedding cosine similarity.

        Args:
            source_domain: Domain to find concepts from (e.g., 'causal_inference')
            target_domain: Domain to find matches in (e.g., 'time_series')
            similarity_threshold: Minimum cosine similarity (0.0-1.0, default 0.85)
            limit_per_concept: Max matches per source concept (default 5)
            source_concept_limit: Max concepts to process from source (default 1000)
            min_relationship_count: Only process concepts with >= N relationships (default 2)

        Returns:
            List of discovered links with:
                - source_concept_id: UUID
                - source_name: str
                - target_concept_id: UUID
                - target_name: str
                - similarity: float
                - source_domain: str
                - target_domain: str

        Example:
            >>> links = await CrossDomainStore.discover_links(
            ...     source_domain="causal_inference",
            ...     target_domain="time_series",
            ...     similarity_threshold=0.85,
            ... )
            >>> for link in links:
            ...     print(f"{link['source_name']} ↔ {link['target_name']} ({link['similarity']:.3f})")
        """
        pool = await get_connection_pool()
        discovered = []

        try:
            async with pool.acquire() as conn:
                # Get source concepts with embeddings, ordered by importance
                # (relationship count as proxy for importance)
                source_concepts = await conn.fetch(
                    """
                    SELECT c.id, c.name, c.canonical_name, c.concept_type, c.embedding
                    FROM concepts c
                    LEFT JOIN (
                        SELECT source_concept_id as concept_id, COUNT(*) as rel_count
                        FROM concept_relationships
                        GROUP BY source_concept_id
                        UNION ALL
                        SELECT target_concept_id as concept_id, COUNT(*) as rel_count
                        FROM concept_relationships
                        GROUP BY target_concept_id
                    ) r ON c.id = r.concept_id
                    WHERE c.domain_id = $1
                      AND c.embedding IS NOT NULL
                    GROUP BY c.id, c.name, c.canonical_name, c.concept_type, c.embedding
                    HAVING COALESCE(SUM(r.rel_count), 0) >= $2
                    ORDER BY COALESCE(SUM(r.rel_count), 0) DESC
                    LIMIT $3
                    """,
                    source_domain,
                    min_relationship_count,
                    source_concept_limit,
                )

                logger.info(
                    "cross_domain_discovery_started",
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_concepts=len(source_concepts),
                    threshold=similarity_threshold,
                )

                # For each source concept, find similar concepts in target domain
                for src in source_concepts:
                    src_embedding = src["embedding"]

                    # Vector similarity search in target domain
                    matches = await conn.fetch(
                        """
                        SELECT
                            id,
                            name,
                            canonical_name,
                            concept_type,
                            1 - (embedding <=> $1::vector) as similarity
                        FROM concepts
                        WHERE domain_id = $2
                          AND embedding IS NOT NULL
                          AND 1 - (embedding <=> $1::vector) >= $3
                        ORDER BY embedding <=> $1::vector
                        LIMIT $4
                        """,
                        src_embedding,
                        target_domain,
                        similarity_threshold,
                        limit_per_concept,
                    )

                    for match in matches:
                        discovered.append(
                            {
                                "source_concept_id": src["id"],
                                "source_name": src["name"],
                                "source_canonical": src["canonical_name"],
                                "source_type": src["concept_type"],
                                "target_concept_id": match["id"],
                                "target_name": match["name"],
                                "target_canonical": match["canonical_name"],
                                "target_type": match["concept_type"],
                                "similarity": float(match["similarity"]),
                                "source_domain": source_domain,
                                "target_domain": target_domain,
                            }
                        )

                logger.info(
                    "cross_domain_discovery_complete",
                    source_domain=source_domain,
                    target_domain=target_domain,
                    links_found=len(discovered),
                )

                return discovered

        except Exception as e:
            logger.error(
                "cross_domain_discovery_failed",
                source_domain=source_domain,
                target_domain=target_domain,
                error=str(e),
            )
            raise StorageError(f"Failed to discover cross-domain links: {e}") from e

    @staticmethod
    async def store_link(
        source_concept_id: UUID,
        target_concept_id: UUID,
        link_type: str,
        confidence_score: float,
        evidence: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """Store a cross-domain concept link.

        Args:
            source_concept_id: Source concept UUID
            target_concept_id: Target concept UUID
            link_type: Type of link (EQUIVALENT, ANALOGOUS, RELATED)
            confidence_score: Confidence score (0.0-1.0)
            evidence: Optional explanation/evidence for the link
            metadata: Optional additional metadata

        Returns:
            UUID of created link

        Raises:
            StorageError: If link creation fails

        Example:
            >>> link_id = await CrossDomainStore.store_link(
            ...     source_concept_id=uuid.UUID("..."),
            ...     target_concept_id=uuid.UUID("..."),
            ...     link_type=CrossDomainLinkType.EQUIVALENT,
            ...     confidence_score=0.92,
            ...     evidence="Both refer to Average Treatment Effect",
            ... )
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                link_id = await conn.fetchval(
                    """
                    INSERT INTO cross_domain_links (
                        source_concept_id, target_concept_id, link_type,
                        confidence_score, evidence, metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (source_concept_id, target_concept_id) DO UPDATE
                    SET link_type = EXCLUDED.link_type,
                        confidence_score = EXCLUDED.confidence_score,
                        evidence = EXCLUDED.evidence,
                        metadata = EXCLUDED.metadata,
                        created_at = EXCLUDED.created_at
                    RETURNING id
                    """,
                    source_concept_id,
                    target_concept_id,
                    link_type,
                    confidence_score,
                    evidence,
                    metadata or {},
                    datetime.now(timezone.utc),
                )

                logger.info(
                    "cross_domain_link_stored",
                    link_id=str(link_id),
                    source_id=str(source_concept_id),
                    target_id=str(target_concept_id),
                    link_type=link_type,
                )

                return UUID(link_id)

        except Exception as e:
            logger.error(
                "cross_domain_link_failed",
                source_id=str(source_concept_id),
                target_id=str(target_concept_id),
                error=str(e),
            )
            raise StorageError(f"Failed to store cross-domain link: {e}") from e

    @staticmethod
    async def store_discovered_links(
        links: list[dict[str, Any]],
        min_similarity: float = 0.90,
    ) -> int:
        """Bulk store discovered links above similarity threshold.

        Automatically classifies link types based on similarity:
            - >= 0.95: EQUIVALENT
            - >= 0.90: ANALOGOUS
            - < 0.90: RELATED

        Args:
            links: List of discovered links from discover_links()
            min_similarity: Minimum similarity to store (default 0.90)

        Returns:
            Number of links stored

        Example:
            >>> links = await CrossDomainStore.discover_links("causal_inference", "time_series")
            >>> stored = await CrossDomainStore.store_discovered_links(links, min_similarity=0.90)
            >>> print(f"Stored {stored} cross-domain links")
        """
        stored = 0

        for link in links:
            if link["similarity"] < min_similarity:
                continue

            # Classify link type based on similarity
            if link["similarity"] >= 0.95:
                link_type = CrossDomainLinkType.EQUIVALENT
            elif link["similarity"] >= 0.90:
                link_type = CrossDomainLinkType.ANALOGOUS
            else:
                link_type = CrossDomainLinkType.RELATED

            try:
                await CrossDomainStore.store_link(
                    source_concept_id=link["source_concept_id"],
                    target_concept_id=link["target_concept_id"],
                    link_type=link_type,
                    confidence_score=link["similarity"],
                    evidence=f"Embedding similarity: {link['similarity']:.3f}",
                    metadata={
                        "source_domain": link["source_domain"],
                        "target_domain": link["target_domain"],
                        "source_name": link["source_name"],
                        "target_name": link["target_name"],
                        "discovery_method": "embedding_similarity",
                    },
                )
                stored += 1
            except StorageError:
                # Log error but continue with other links
                pass

        logger.info("cross_domain_links_stored", count=stored, total=len(links))
        return stored

    @staticmethod
    async def get_cross_domain_concepts(
        concept_id: UUID,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get cross-domain links for a concept.

        Args:
            concept_id: Concept UUID to find links for
            direction: 'source', 'target', or 'both'

        Returns:
            List of linked concepts with link metadata

        Example:
            >>> links = await CrossDomainStore.get_cross_domain_concepts(concept_id)
            >>> for link in links:
            ...     print(f"{link['linked_concept_name']} ({link['link_type']})")
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                results = []

                if direction in ("source", "both"):
                    # Concept is source, find targets
                    rows = await conn.fetch(
                        """
                        SELECT
                            cdl.id as link_id,
                            cdl.link_type,
                            cdl.confidence_score,
                            cdl.evidence,
                            cdl.metadata,
                            c.id as linked_concept_id,
                            c.name as linked_concept_name,
                            c.canonical_name as linked_canonical,
                            c.concept_type as linked_type,
                            c.domain_id as linked_domain,
                            'target' as direction
                        FROM cross_domain_links cdl
                        JOIN concepts c ON cdl.target_concept_id = c.id
                        WHERE cdl.source_concept_id = $1
                        ORDER BY cdl.confidence_score DESC
                        """,
                        concept_id,
                    )
                    results.extend([dict(r) for r in rows])

                if direction in ("target", "both"):
                    # Concept is target, find sources
                    rows = await conn.fetch(
                        """
                        SELECT
                            cdl.id as link_id,
                            cdl.link_type,
                            cdl.confidence_score,
                            cdl.evidence,
                            cdl.metadata,
                            c.id as linked_concept_id,
                            c.name as linked_concept_name,
                            c.canonical_name as linked_canonical,
                            c.concept_type as linked_type,
                            c.domain_id as linked_domain,
                            'source' as direction
                        FROM cross_domain_links cdl
                        JOIN concepts c ON cdl.source_concept_id = c.id
                        WHERE cdl.target_concept_id = $1
                        ORDER BY cdl.confidence_score DESC
                        """,
                        concept_id,
                    )
                    results.extend([dict(r) for r in rows])

                return results

        except Exception as e:
            logger.error(
                "cross_domain_get_failed",
                concept_id=str(concept_id),
                error=str(e),
            )
            raise StorageError(f"Failed to get cross-domain links: {e}") from e

    @staticmethod
    async def get_stats() -> dict[str, Any]:
        """Get cross-domain link statistics.

        Returns:
            Dictionary with:
                - total_links: Total number of links
                - by_type: Counts per link type
                - by_domain_pair: Counts per domain pair
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                total = await conn.fetchval("SELECT COUNT(*) FROM cross_domain_links")

                type_counts = await conn.fetch("""
                    SELECT link_type, COUNT(*) as count
                    FROM cross_domain_links
                    GROUP BY link_type
                    ORDER BY count DESC
                    """)

                domain_counts = await conn.fetch("""
                    SELECT
                        metadata->>'source_domain' as source_domain,
                        metadata->>'target_domain' as target_domain,
                        COUNT(*) as count
                    FROM cross_domain_links
                    WHERE metadata->>'source_domain' IS NOT NULL
                    GROUP BY metadata->>'source_domain', metadata->>'target_domain'
                    ORDER BY count DESC
                    """)

                return {
                    "total_links": total or 0,
                    "by_type": {r["link_type"]: r["count"] for r in type_counts},
                    "by_domain_pair": [
                        {
                            "source": r["source_domain"],
                            "target": r["target_domain"],
                            "count": r["count"],
                        }
                        for r in domain_counts
                    ],
                }

        except Exception as e:
            logger.error("cross_domain_stats_failed", error=str(e))
            raise StorageError(f"Failed to get cross-domain stats: {e}") from e
