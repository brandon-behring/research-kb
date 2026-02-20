#!/usr/bin/env python3
"""Diagnose orphan concepts in the knowledge graph.

Orphan concepts have no relationships in concept_relationships table.
This script categorizes them as:
- Salvageable: Has chunk_concepts entries (can be re-linked via semantic similarity)
- Dead: No chunk_concepts, low confidence (candidates for deletion)

Usage:
    python scripts/diagnose_orphans.py
    python scripts/diagnose_orphans.py --output orphans_report.csv
"""

import argparse
import asyncio
import csv
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
from research_kb_common import get_logger

logger = get_logger(__name__)


async def get_orphan_statistics(conn: asyncpg.Connection) -> dict:
    """Get comprehensive orphan statistics."""

    stats = {}

    # Total counts
    stats["total_concepts"] = await conn.fetchval("SELECT COUNT(*) FROM concepts")
    stats["total_relationships"] = await conn.fetchval("SELECT COUNT(*) FROM concept_relationships")

    # Orphan counts by type
    orphans_by_type = await conn.fetch(
        """
        SELECT concept_type, COUNT(*) as count
        FROM concepts c
        WHERE NOT EXISTS (
            SELECT 1 FROM concept_relationships cr
            WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
        )
        GROUP BY concept_type
        ORDER BY count DESC
    """
    )
    stats["orphans_by_type"] = {row["concept_type"]: row["count"] for row in orphans_by_type}
    stats["total_orphans"] = sum(stats["orphans_by_type"].values())

    # Salvageable vs dead
    stats["salvageable"] = await conn.fetchval(
        """
        SELECT COUNT(*) FROM concepts c
        WHERE NOT EXISTS (
            SELECT 1 FROM concept_relationships cr
            WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
        )
        AND EXISTS (SELECT 1 FROM chunk_concepts cc WHERE cc.concept_id = c.id)
    """
    )

    stats["dead"] = await conn.fetchval(
        """
        SELECT COUNT(*) FROM concepts c
        WHERE NOT EXISTS (
            SELECT 1 FROM concept_relationships cr
            WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
        )
        AND NOT EXISTS (SELECT 1 FROM chunk_concepts cc WHERE cc.concept_id = c.id)
    """
    )

    # Low confidence orphans (candidates for deletion)
    stats["low_confidence_orphans"] = await conn.fetchval(
        """
        SELECT COUNT(*) FROM concepts c
        WHERE NOT EXISTS (
            SELECT 1 FROM concept_relationships cr
            WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
        )
        AND c.confidence_score < 0.5
    """
    )

    # Orphans with embeddings (needed for semantic re-linking)
    stats["orphans_with_embeddings"] = await conn.fetchval(
        """
        SELECT COUNT(*) FROM concepts c
        WHERE NOT EXISTS (
            SELECT 1 FROM concept_relationships cr
            WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
        )
        AND c.embedding IS NOT NULL
    """
    )

    return stats


async def get_orphan_samples(conn: asyncpg.Connection, limit: int = 20) -> list[dict]:
    """Get sample orphan concepts for review."""

    rows = await conn.fetch(
        """
        SELECT
            c.id::text,
            c.canonical_name,
            c.concept_type,
            c.confidence_score,
            c.embedding IS NOT NULL as has_embedding,
            (SELECT COUNT(*) FROM chunk_concepts cc WHERE cc.concept_id = c.id) as chunk_count
        FROM concepts c
        WHERE NOT EXISTS (
            SELECT 1 FROM concept_relationships cr
            WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
        )
        ORDER BY c.confidence_score DESC
        LIMIT $1
    """,
        limit,
    )

    return [dict(row) for row in rows]


async def export_orphans_csv(conn: asyncpg.Connection, output_path: Path) -> int:
    """Export all orphans to CSV for manual review."""

    rows = await conn.fetch(
        """
        SELECT
            c.id::text as concept_id,
            c.canonical_name,
            c.concept_type,
            c.definition,
            c.confidence_score,
            c.embedding IS NOT NULL as has_embedding,
            (SELECT COUNT(*) FROM chunk_concepts cc WHERE cc.concept_id = c.id) as chunk_count,
            CASE
                WHEN EXISTS (SELECT 1 FROM chunk_concepts cc WHERE cc.concept_id = c.id)
                THEN 'salvageable'
                ELSE 'dead'
            END as status
        FROM concepts c
        WHERE NOT EXISTS (
            SELECT 1 FROM concept_relationships cr
            WHERE cr.source_concept_id = c.id OR cr.target_concept_id = c.id
        )
        ORDER BY c.concept_type, c.confidence_score DESC
    """
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concept_id",
                "canonical_name",
                "concept_type",
                "definition",
                "confidence_score",
                "has_embedding",
                "chunk_count",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    return len(rows)


async def main():
    """Run orphan diagnosis."""
    parser = argparse.ArgumentParser(description="Diagnose orphan concepts")
    parser.add_argument("--output", "-o", type=Path, help="Output CSV file path")
    args = parser.parse_args()

    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="research_kb",
    )

    print("=" * 60)
    print("Orphan Concept Diagnosis")
    print("=" * 60)

    # Get statistics
    stats = await get_orphan_statistics(conn)

    print("\n--- Overview ---")
    print(f"Total concepts:     {stats['total_concepts']:,}")
    print(f"Total relationships: {stats['total_relationships']:,}")
    print(
        f"Total orphans:      {stats['total_orphans']:,} ({100*stats['total_orphans']/stats['total_concepts']:.1f}%)"
    )

    print("\n--- Orphan Classification ---")
    print(f"Salvageable (has chunk_concepts): {stats['salvageable']:,}")
    print(f"Dead (no chunk_concepts):         {stats['dead']:,}")
    print(f"Low confidence (<0.5):            {stats['low_confidence_orphans']:,}")
    print(f"Has embeddings:                   {stats['orphans_with_embeddings']:,}")

    print("\n--- Orphans by Type ---")
    for concept_type, count in stats["orphans_by_type"].items():
        pct = 100 * count / stats["total_orphans"] if stats["total_orphans"] > 0 else 0
        print(f"  {concept_type:15s}: {count:6,} ({pct:5.1f}%)")

    # Show samples
    print("\n--- Sample Orphans (Top by Confidence) ---")
    samples = await get_orphan_samples(conn)
    for s in samples[:10]:
        status = "salvageable" if s["chunk_count"] > 0 else "dead"
        print(
            f"  [{s['concept_type']:10s}] {s['canonical_name'][:40]:40s} "
            f"conf={s['confidence_score']:.2f} chunks={s['chunk_count']} ({status})"
        )

    # Export to CSV if requested
    if args.output:
        print("\n--- Exporting to CSV ---")
        count = await export_orphans_csv(conn, args.output)
        print(f"Exported {count:,} orphans to {args.output}")

    await conn.close()

    # Print recommendations
    print("\n--- Recommendations ---")
    if stats["salvageable"] > 0:
        print(f"1. Re-link {stats['salvageable']:,} salvageable orphans using semantic similarity")
        print("   Run: python scripts/relink_orphans.py")
    if stats["dead"] > 0:
        print(f"2. Consider deleting {stats['dead']:,} dead orphans (no chunk references)")
    if stats["low_confidence_orphans"] > 0:
        print(
            f"3. Review {stats['low_confidence_orphans']:,} low-confidence orphans for quality issues"
        )


if __name__ == "__main__":
    asyncio.run(main())
