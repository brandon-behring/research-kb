#!/usr/bin/env python3
"""Re-link orphan concepts using semantic similarity.

For orphan concepts (those with 0 relationships), finds semantically similar
concepts that ARE connected in the graph and creates RELATED_TO relationships.

This helps integrate isolated concepts back into the knowledge graph.

Requirements:
- Orphan must have an embedding
- Similar concept must have at least one existing relationship
- Similarity threshold: 0.80 (cosine similarity)

Usage:
    # Dry run (default)
    python scripts/relink_orphans.py

    # Execute with lower threshold
    python scripts/relink_orphans.py --threshold 0.75 --execute

    # Limit number of orphans to process
    python scripts/relink_orphans.py --limit 1000 --execute
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
from research_kb_common import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 0.80
BATCH_SIZE = 100


async def get_orphans_with_embeddings(
    conn: asyncpg.Connection, limit: int | None = None
) -> list[dict]:
    """Get orphan concepts that have embeddings."""

    sql = """
        SELECT
            c.id,
            c.canonical_name,
            c.concept_type,
            c.embedding
        FROM concepts c
        WHERE c.embedding IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM concept_relationships cr
            WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
        )
        ORDER BY c.canonical_name
    """
    if limit:
        sql += f" LIMIT {limit}"

    rows = await conn.fetch(sql)
    return [dict(row) for row in rows]


async def find_similar_connected_concepts(
    conn: asyncpg.Connection,
    embedding: list[float],
    exclude_id: UUID,
    threshold: float,
    limit: int = 5,
) -> list[dict]:
    """Find concepts similar to the given embedding that have relationships.

    Returns concepts that:
    1. Have cosine similarity >= threshold
    2. Have at least one relationship in the graph
    3. Are not the excluded concept itself
    """

    # Use pgvector cosine distance (1 - cosine_similarity)
    # So we want distance < (1 - threshold)
    max_distance = 1.0 - threshold

    rows = await conn.fetch(
        """
        SELECT
            c.id,
            c.canonical_name,
            c.concept_type,
            1 - (c.embedding <=> $1::vector) as similarity
        FROM concepts c
        WHERE c.id != $2
        AND c.embedding IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM concept_relationships cr
            WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
        )
        AND (c.embedding <=> $1::vector) < $3
        ORDER BY c.embedding <=> $1::vector
        LIMIT $4
    """,
        str(embedding),
        exclude_id,
        max_distance,
        limit,
    )

    return [dict(row) for row in rows]


async def create_relationship(
    conn: asyncpg.Connection,
    source_id: UUID,
    target_id: UUID,
    relationship_type: str = "RELATED_TO",
    strength: float = 1.0,
) -> bool:
    """Create a relationship between two concepts."""

    try:
        await conn.execute(
            """
            INSERT INTO concept_relationships (source_concept_id, target_concept_id, relationship_type, strength)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (source_concept_id, target_concept_id, relationship_type) DO NOTHING
        """,
            source_id,
            target_id,
            relationship_type,
            strength,
        )
        return True
    except Exception as e:
        logger.error("create_relationship_failed", error=str(e))
        return False


async def relink_orphan(
    conn: asyncpg.Connection, orphan: dict, threshold: float, execute: bool = False
) -> dict:
    """Attempt to re-link a single orphan concept.

    Returns dict with:
    - orphan_id: UUID of the orphan
    - orphan_name: Name of the orphan
    - similar_concepts: List of similar connected concepts found
    - relationships_created: Count of relationships created (if execute=True)
    """

    result = {
        "orphan_id": orphan["id"],
        "orphan_name": orphan["canonical_name"],
        "orphan_type": orphan["concept_type"],
        "similar_concepts": [],
        "relationships_created": 0,
    }

    # Find similar connected concepts
    similar = await find_similar_connected_concepts(
        conn,
        orphan["embedding"],
        orphan["id"],
        threshold,
        limit=3,  # Connect to top 3 similar concepts
    )

    result["similar_concepts"] = similar

    if execute and similar:
        for concept in similar:
            # Use similarity as strength (normalized)
            strength = concept["similarity"]

            # Create RELATED_TO relationship
            success = await create_relationship(
                conn, orphan["id"], concept["id"], "RELATED_TO", strength
            )
            if success:
                result["relationships_created"] += 1

    return result


async def main():
    """Run orphan re-linking."""
    parser = argparse.ArgumentParser(description="Re-link orphan concepts via semantic similarity")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--limit", type=int, help="Limit number of orphans to process")
    parser.add_argument("--execute", action="store_true", help="Actually create relationships")
    args = parser.parse_args()

    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="research_kb",
    )

    mode = "EXECUTING" if args.execute else "DRY RUN"
    print(f"{mode}: Orphan Re-linking")
    print("=" * 60)
    print(f"Similarity threshold: {args.threshold}")
    if args.limit:
        print(f"Processing limit: {args.limit}")

    # Get orphans with embeddings
    print("\nFetching orphans with embeddings...")
    orphans = await get_orphans_with_embeddings(conn, args.limit)
    print(f"Found {len(orphans):,} orphans to process")

    if not orphans:
        print("\nNo orphans to process!")
        await conn.close()
        return

    # Process orphans
    total_relinked = 0
    total_relationships = 0
    samples = []

    print("\nProcessing orphans...")
    for i, orphan in enumerate(orphans):
        if (i + 1) % 1000 == 0:
            print(f"  [{i + 1:,}/{len(orphans):,}] {orphan['canonical_name'][:40]}")

        result = await relink_orphan(conn, orphan, args.threshold, args.execute)

        if result["similar_concepts"]:
            total_relinked += 1
            total_relationships += (
                len(result["similar_concepts"])
                if not args.execute
                else result["relationships_created"]
            )

            # Save samples for reporting
            if len(samples) < 20:
                samples.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Orphans processed:      {len(orphans):,}")
    print(f"Orphans with matches:   {total_relinked:,}")
    print(f"Orphans still isolated: {len(orphans) - total_relinked:,}")

    if args.execute:
        print(f"Relationships created:  {total_relationships:,}")
    else:
        print(f"Relationships (potential): {total_relationships:,}")

    # Show samples
    if samples:
        print("\n--- Sample Re-links ---")
        for s in samples[:10]:
            print(f"\n  {s['orphan_name']} ({s['orphan_type']})")
            for sim in s["similar_concepts"][:2]:
                print(
                    f"    -> {sim['canonical_name']} ({sim['concept_type']}) "
                    f"sim={sim['similarity']:.3f}"
                )

    if not args.execute:
        print("\n" + "=" * 60)
        print("This was a DRY RUN. Use --execute to create relationships.")
        print("=" * 60)

    # Verify final state
    if args.execute:
        print("\nVerifying final orphan count...")
        remaining = await conn.fetchval("""
            SELECT COUNT(*) FROM concepts c
            WHERE NOT EXISTS (
                SELECT 1 FROM concept_relationships cr
                WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
            )
        """)
        print(f"Remaining orphans: {remaining:,}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
