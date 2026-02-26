"""Bibliographic Coupling Store - Similarity based on shared references.

Bibliographic coupling measures similarity between sources based on their
shared references. If sources A and B both cite sources C, D, E, they are
likely topically related.

Coupling strength uses Jaccard similarity:
    coupling = |shared_refs| / |union_refs|

Where:
- shared_refs = references cited by both A and B
- union_refs = A_refs + B_refs - shared_refs

Example:
    >>> from research_kb_storage import BiblioStore
    >>> similar = await BiblioStore.get_similar_sources(source_id, limit=10)
    >>> for s in similar:
    ...     print(f"{s['coupling_strength']:.2f}: {s['title']}")
"""

from typing import Optional
from uuid import UUID

from research_kb_common import get_logger
from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)


class BiblioStore:
    """Store for bibliographic coupling computations."""

    @staticmethod
    async def compute_coupling_for_source(
        source_id: UUID,
        min_coupling: float = 0.1,
    ) -> list[dict]:
        """Compute bibliographic coupling for a single source.

        Finds all other sources that share at least one cited source,
        computes Jaccard similarity, and returns results above threshold.

        Args:
            source_id: Source to compute coupling for
            min_coupling: Minimum coupling strength threshold

        Returns:
            List of dicts with: other_source_id, shared_count, coupling_strength
        """
        pool = await get_connection_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH my_citations AS (
                    -- Get all sources I cite (internal only)
                    SELECT DISTINCT sc.cited_source_id
                    FROM source_citations sc
                    WHERE sc.citing_source_id = $1
                      AND sc.cited_source_id IS NOT NULL
                ),
                my_count AS (
                    SELECT COUNT(*) AS cnt FROM my_citations
                ),
                other_sources AS (
                    -- Find other sources citing at least one of my cited sources
                    SELECT DISTINCT sc.citing_source_id
                    FROM source_citations sc
                    WHERE sc.cited_source_id IN (SELECT cited_source_id FROM my_citations)
                      AND sc.citing_source_id != $1
                ),
                other_with_shared AS (
                    -- For each other source, count shared citations
                    SELECT
                        os.citing_source_id AS other_id,
                        COUNT(DISTINCT sc.cited_source_id) AS shared_count,
                        (SELECT COUNT(DISTINCT cited_source_id)
                         FROM source_citations
                         WHERE citing_source_id = os.citing_source_id
                           AND cited_source_id IS NOT NULL) AS other_count
                    FROM other_sources os
                    JOIN source_citations sc ON sc.citing_source_id = os.citing_source_id
                    WHERE sc.cited_source_id IN (SELECT cited_source_id FROM my_citations)
                    GROUP BY os.citing_source_id
                )
                SELECT
                    other_id,
                    shared_count,
                    (SELECT cnt FROM my_count) AS my_count,
                    other_count,
                    shared_count::real / NULLIF(
                        (SELECT cnt FROM my_count) + other_count - shared_count,
                        0
                    ) AS coupling_strength
                FROM other_with_shared
                WHERE shared_count::real / NULLIF(
                    (SELECT cnt FROM my_count) + other_count - shared_count,
                    0
                ) >= $2
                ORDER BY coupling_strength DESC
            """,
                source_id,
                min_coupling,
            )

            return [
                {
                    "other_source_id": row["other_id"],
                    "shared_count": row["shared_count"],
                    "my_count": row["my_count"],
                    "other_count": row["other_count"],
                    "coupling_strength": (
                        float(row["coupling_strength"]) if row["coupling_strength"] else 0.0
                    ),
                }
                for row in rows
            ]

    @staticmethod
    async def compute_all_coupling(
        min_coupling: float = 0.1,
        batch_size: int = 100,
    ) -> dict:
        """Compute bibliographic coupling for all source pairs.

        Iterates through all sources with internal citations and computes
        coupling with all other sources. Stores results in bibliographic_coupling table.

        Args:
            min_coupling: Minimum coupling threshold to store
            batch_size: Batch size for inserts

        Returns:
            Dict with statistics: total_pairs, stored, skipped
        """
        pool = await get_connection_pool()
        stats = {"total_sources": 0, "pairs_computed": 0, "pairs_stored": 0}

        async with pool.acquire() as conn:
            # Get all sources with internal citations
            sources = await conn.fetch("""
                SELECT DISTINCT citing_source_id
                FROM source_citations
                WHERE cited_source_id IS NOT NULL
            """)
            stats["total_sources"] = len(sources)

            logger.info(
                "computing_bibliographic_coupling",
                source_count=len(sources),
                min_coupling=min_coupling,
            )

            # Clear existing coupling data
            await conn.execute("TRUNCATE bibliographic_coupling")

            # Process each source
            batch = []
            for idx, row in enumerate(sources):
                source_id = row["citing_source_id"]

                # Compute coupling
                couplings = await BiblioStore.compute_coupling_for_source(source_id, min_coupling)
                stats["pairs_computed"] += len(couplings)

                for c in couplings:
                    other_id = c["other_source_id"]
                    # Enforce ordering to avoid duplicates
                    if source_id < other_id:
                        a_id, b_id = source_id, other_id
                    else:
                        a_id, b_id = other_id, source_id

                    batch.append(
                        (
                            a_id,
                            b_id,
                            c["shared_count"],
                            c["coupling_strength"],
                        )
                    )

                # Insert batch
                if len(batch) >= batch_size:
                    await conn.executemany(
                        """
                        INSERT INTO bibliographic_coupling
                            (source_a_id, source_b_id, shared_references, coupling_strength)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (source_a_id, source_b_id) DO UPDATE
                        SET shared_references = EXCLUDED.shared_references,
                            coupling_strength = EXCLUDED.coupling_strength,
                            computed_at = NOW()
                    """,
                        batch,
                    )
                    stats["pairs_stored"] += len(batch)
                    batch = []

                if (idx + 1) % 50 == 0:
                    logger.info(
                        "coupling_progress",
                        processed=idx + 1,
                        total=len(sources),
                        pairs_stored=stats["pairs_stored"],
                    )

            # Insert remaining
            if batch:
                await conn.executemany(
                    """
                    INSERT INTO bibliographic_coupling
                        (source_a_id, source_b_id, shared_references, coupling_strength)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (source_a_id, source_b_id) DO UPDATE
                    SET shared_references = EXCLUDED.shared_references,
                        coupling_strength = EXCLUDED.coupling_strength,
                        computed_at = NOW()
                """,
                    batch,
                )
                stats["pairs_stored"] += len(batch)

        logger.info(
            "bibliographic_coupling_complete",
            **stats,
        )

        return stats

    @staticmethod
    async def get_similar_sources(
        source_id: UUID,
        limit: int = 10,
    ) -> list[dict]:
        """Get sources most similar to the given source by bibliographic coupling.

        Args:
            source_id: Source to find similar sources for
            limit: Maximum number of results

        Returns:
            List of dicts with: source_id, title, authors, year, coupling_strength
        """
        pool = await get_connection_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    CASE
                        WHEN bc.source_a_id = $1 THEN bc.source_b_id
                        ELSE bc.source_a_id
                    END AS similar_source_id,
                    s.title,
                    s.authors,
                    s.year,
                    s.source_type,
                    bc.shared_references,
                    bc.coupling_strength
                FROM bibliographic_coupling bc
                JOIN sources s ON s.id = CASE
                    WHEN bc.source_a_id = $1 THEN bc.source_b_id
                    ELSE bc.source_a_id
                END
                WHERE bc.source_a_id = $1 OR bc.source_b_id = $1
                ORDER BY bc.coupling_strength DESC
                LIMIT $2
            """,
                source_id,
                limit,
            )

            return [
                {
                    "source_id": row["similar_source_id"],
                    "title": row["title"],
                    "authors": row["authors"],
                    "year": row["year"],
                    "source_type": row["source_type"],
                    "shared_references": row["shared_references"],
                    "coupling_strength": float(row["coupling_strength"]),
                }
                for row in rows
            ]

    @staticmethod
    async def get_coupling_score(
        source_a_id: UUID,
        source_b_id: UUID,
    ) -> Optional[float]:
        """Get coupling strength between two specific sources.

        Args:
            source_a_id: First source
            source_b_id: Second source

        Returns:
            Coupling strength (0.0-1.0) or None if no coupling exists
        """
        pool = await get_connection_pool()

        # Enforce ordering
        if source_a_id > source_b_id:
            source_a_id, source_b_id = source_b_id, source_a_id

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT coupling_strength
                FROM bibliographic_coupling
                WHERE source_a_id = $1 AND source_b_id = $2
            """,
                source_a_id,
                source_b_id,
            )

            return float(row["coupling_strength"]) if row else None

    @staticmethod
    async def get_stats() -> dict:
        """Get bibliographic coupling statistics.

        Returns:
            Dict with: total_pairs, avg_coupling, max_coupling, sources_with_coupling
        """
        pool = await get_connection_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    COUNT(*) AS total_pairs,
                    AVG(coupling_strength) AS avg_coupling,
                    MAX(coupling_strength) AS max_coupling,
                    COUNT(DISTINCT source_a_id) + COUNT(DISTINCT source_b_id) AS sources_involved
                FROM bibliographic_coupling
            """)

            return {
                "total_pairs": row["total_pairs"],
                "avg_coupling": (float(row["avg_coupling"]) if row["avg_coupling"] else 0.0),
                "max_coupling": (float(row["max_coupling"]) if row["max_coupling"] else 0.0),
                "sources_involved": row["sources_involved"],
            }
