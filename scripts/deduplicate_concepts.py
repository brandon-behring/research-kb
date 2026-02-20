#!/usr/bin/env python3
"""Deduplicate concepts by merging singular/plural pairs.

Finds concepts where one is the plural of another (e.g., "eigenvalue" vs "eigenvalues")
and merges them, keeping the singular form as canonical.

Usage:
    # Dry run (default) - show what would be merged
    python scripts/deduplicate_concepts.py

    # Execute the merge
    python scripts/deduplicate_concepts.py --execute

    # Limit to first N pairs (for testing)
    python scripts/deduplicate_concepts.py --limit 10 --execute
"""

import argparse
import asyncio
from dataclasses import dataclass
from uuid import UUID

import asyncpg

from research_kb_common import get_logger

logger = get_logger(__name__)


@dataclass
class DuplicatePair:
    """A pair of duplicate concepts to merge."""

    singular_id: UUID
    singular_name: str
    plural_id: UUID
    plural_name: str
    singular_type: str
    plural_type: str


async def find_singular_plural_pairs(
    conn: asyncpg.Connection, limit: int | None = None
) -> list[DuplicatePair]:
    """Find all singular/plural concept pairs.

    Matches concepts where plural_name = singular_name + 's'
    Excludes empty names and single-character names.
    """
    sql = """
    SELECT
        c1.id AS singular_id,
        c1.canonical_name AS singular_name,
        c1.concept_type AS singular_type,
        c2.id AS plural_id,
        c2.canonical_name AS plural_name,
        c2.concept_type AS plural_type
    FROM concepts c1
    JOIN concepts c2 ON c2.canonical_name = c1.canonical_name || 's'
    WHERE LENGTH(c1.canonical_name) > 1  -- Exclude empty and single-char names
    ORDER BY c1.canonical_name
    """
    if limit:
        sql += f" LIMIT {limit}"

    rows = await conn.fetch(sql)

    return [
        DuplicatePair(
            singular_id=row["singular_id"],
            singular_name=row["singular_name"],
            singular_type=row["singular_type"],
            plural_id=row["plural_id"],
            plural_name=row["plural_name"],
            plural_type=row["plural_type"],
        )
        for row in rows
    ]


async def get_reference_counts(conn: asyncpg.Connection, concept_id: UUID) -> dict[str, int]:
    """Get counts of references to a concept from each table."""
    counts = {}

    counts["chunk_concepts"] = await conn.fetchval(
        "SELECT COUNT(*) FROM chunk_concepts WHERE concept_id = $1", concept_id
    )
    counts["relationships_source"] = await conn.fetchval(
        "SELECT COUNT(*) FROM concept_relationships WHERE source_concept_id = $1",
        concept_id,
    )
    counts["relationships_target"] = await conn.fetchval(
        "SELECT COUNT(*) FROM concept_relationships WHERE target_concept_id = $1",
        concept_id,
    )
    counts["methods"] = await conn.fetchval(
        "SELECT COUNT(*) FROM methods WHERE concept_id = $1", concept_id
    )
    counts["assumptions"] = await conn.fetchval(
        "SELECT COUNT(*) FROM assumptions WHERE concept_id = $1", concept_id
    )

    return counts


async def merge_concept_pair(
    conn: asyncpg.Connection,
    pair: DuplicatePair,
    dry_run: bool = True,
) -> dict:
    """Merge a singular/plural pair, keeping the singular.

    Steps:
    1. Update chunk_concepts references
    2. Update concept_relationships (source and target)
    3. Update methods references
    4. Update assumptions references
    5. Merge aliases from plural into singular
    6. Delete the plural concept

    Returns stats about what was updated.
    """
    stats = {
        "singular_name": pair.singular_name,
        "plural_name": pair.plural_name,
        "chunk_concepts_updated": 0,
        "relationships_source_updated": 0,
        "relationships_target_updated": 0,
        "methods_updated": 0,
        "assumptions_updated": 0,
        "aliases_merged": 0,
        "concept_deleted": False,
        "errors": [],
    }

    if dry_run:
        # Just count what would be updated
        counts = await get_reference_counts(conn, pair.plural_id)
        stats["chunk_concepts_updated"] = counts["chunk_concepts"]
        stats["relationships_source_updated"] = counts["relationships_source"]
        stats["relationships_target_updated"] = counts["relationships_target"]
        stats["methods_updated"] = counts["methods"]
        stats["assumptions_updated"] = counts["assumptions"]

        # Check aliases
        plural_aliases = await conn.fetchval(
            "SELECT aliases FROM concepts WHERE id = $1", pair.plural_id
        )
        stats["aliases_merged"] = len(plural_aliases) if plural_aliases else 0
        stats["concept_deleted"] = True

        return stats

    # Execute the merge
    try:
        async with conn.transaction():
            # 1. Update chunk_concepts (handle duplicates with ON CONFLICT)
            # First, delete any that would create duplicates
            await conn.execute(
                """
                DELETE FROM chunk_concepts cc1
                WHERE cc1.concept_id = $1
                AND EXISTS (
                    SELECT 1 FROM chunk_concepts cc2
                    WHERE cc2.chunk_id = cc1.chunk_id
                    AND cc2.concept_id = $2
                )
            """,
                pair.plural_id,
                pair.singular_id,
            )

            # Then update the rest
            result = await conn.execute(
                """
                UPDATE chunk_concepts
                SET concept_id = $1
                WHERE concept_id = $2
            """,
                pair.singular_id,
                pair.plural_id,
            )
            stats["chunk_concepts_updated"] = int(result.split()[-1])

            # 2. Update concept_relationships (source)
            # Delete duplicates first
            await conn.execute(
                """
                DELETE FROM concept_relationships cr1
                WHERE cr1.source_concept_id = $1
                AND EXISTS (
                    SELECT 1 FROM concept_relationships cr2
                    WHERE cr2.source_concept_id = $2
                    AND cr2.target_concept_id = cr1.target_concept_id
                    AND cr2.relationship_type = cr1.relationship_type
                )
            """,
                pair.plural_id,
                pair.singular_id,
            )

            result = await conn.execute(
                """
                UPDATE concept_relationships
                SET source_concept_id = $1
                WHERE source_concept_id = $2
            """,
                pair.singular_id,
                pair.plural_id,
            )
            stats["relationships_source_updated"] = int(result.split()[-1])

            # 3. Update concept_relationships (target)
            await conn.execute(
                """
                DELETE FROM concept_relationships cr1
                WHERE cr1.target_concept_id = $1
                AND EXISTS (
                    SELECT 1 FROM concept_relationships cr2
                    WHERE cr2.target_concept_id = $2
                    AND cr2.source_concept_id = cr1.source_concept_id
                    AND cr2.relationship_type = cr1.relationship_type
                )
            """,
                pair.plural_id,
                pair.singular_id,
            )

            result = await conn.execute(
                """
                UPDATE concept_relationships
                SET target_concept_id = $1
                WHERE target_concept_id = $2
            """,
                pair.singular_id,
                pair.plural_id,
            )
            stats["relationships_target_updated"] = int(result.split()[-1])

            # 4. Update methods (delete duplicates first)
            await conn.execute(
                """
                DELETE FROM methods
                WHERE concept_id = $1
                AND EXISTS (SELECT 1 FROM methods WHERE concept_id = $2)
            """,
                pair.plural_id,
                pair.singular_id,
            )

            result = await conn.execute(
                """
                UPDATE methods
                SET concept_id = $1
                WHERE concept_id = $2
            """,
                pair.singular_id,
                pair.plural_id,
            )
            stats["methods_updated"] = int(result.split()[-1])

            # 5. Update assumptions (delete duplicates first)
            await conn.execute(
                """
                DELETE FROM assumptions
                WHERE concept_id = $1
                AND EXISTS (SELECT 1 FROM assumptions WHERE concept_id = $2)
            """,
                pair.plural_id,
                pair.singular_id,
            )

            result = await conn.execute(
                """
                UPDATE assumptions
                SET concept_id = $1
                WHERE concept_id = $2
            """,
                pair.singular_id,
                pair.plural_id,
            )
            stats["assumptions_updated"] = int(result.split()[-1])

            # 6. Merge aliases
            plural_aliases = await conn.fetchval(
                "SELECT aliases FROM concepts WHERE id = $1", pair.plural_id
            )
            if plural_aliases:
                # Add plural name and its aliases to singular's aliases
                await conn.execute(
                    """
                    UPDATE concepts
                    SET aliases = array_cat(
                        aliases,
                        array_append($1::text[], $2)
                    )
                    WHERE id = $3
                """,
                    plural_aliases,
                    pair.plural_name,
                    pair.singular_id,
                )
                stats["aliases_merged"] = len(plural_aliases) + 1
            else:
                # Just add the plural name as an alias
                await conn.execute(
                    """
                    UPDATE concepts
                    SET aliases = array_append(aliases, $1)
                    WHERE id = $2
                """,
                    pair.plural_name,
                    pair.singular_id,
                )
                stats["aliases_merged"] = 1

            # 7. Delete the plural concept
            await conn.execute("DELETE FROM concepts WHERE id = $1", pair.plural_id)
            stats["concept_deleted"] = True

    except Exception as e:
        stats["errors"].append(str(e))
        logger.error("merge_failed", pair=pair.singular_name, error=str(e))

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Deduplicate singular/plural concept pairs")
    parser.add_argument(
        "--execute", action="store_true", help="Execute the merge (default: dry run)"
    )
    parser.add_argument("--limit", type=int, help="Limit to first N pairs (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details for each pair")
    args = parser.parse_args()

    dry_run = not args.execute

    # Connect to database
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
    )

    try:
        print(f"{'DRY RUN' if dry_run else 'EXECUTING'}: Concept Deduplication")
        print("=" * 60)

        # Find pairs
        print("\nFinding singular/plural pairs...")
        pairs = await find_singular_plural_pairs(conn, limit=args.limit)
        print(f"Found {len(pairs)} pairs to merge")

        if not pairs:
            print("No duplicates found!")
            return

        # Process pairs
        total_stats = {
            "pairs_processed": 0,
            "chunk_concepts_updated": 0,
            "relationships_updated": 0,
            "methods_updated": 0,
            "assumptions_updated": 0,
            "aliases_merged": 0,
            "concepts_deleted": 0,
            "errors": 0,
        }

        print(f"\n{'Processing' if not dry_run else 'Analyzing'} pairs...")

        for i, pair in enumerate(pairs):
            if args.verbose or (i + 1) % 1000 == 0:
                print(f"  [{i+1}/{len(pairs)}] {pair.singular_name} <- {pair.plural_name}")

            stats = await merge_concept_pair(conn, pair, dry_run=dry_run)

            total_stats["pairs_processed"] += 1
            total_stats["chunk_concepts_updated"] += stats["chunk_concepts_updated"]
            total_stats["relationships_updated"] += (
                stats["relationships_source_updated"] + stats["relationships_target_updated"]
            )
            total_stats["methods_updated"] += stats["methods_updated"]
            total_stats["assumptions_updated"] += stats["assumptions_updated"]
            total_stats["aliases_merged"] += stats["aliases_merged"]
            if stats["concept_deleted"]:
                total_stats["concepts_deleted"] += 1
            if stats["errors"]:
                total_stats["errors"] += 1
                if args.verbose:
                    print(f"    ERROR: {stats['errors']}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Pairs processed:        {total_stats['pairs_processed']:,}")
        print(f"Chunk-concepts updated: {total_stats['chunk_concepts_updated']:,}")
        print(f"Relationships updated:  {total_stats['relationships_updated']:,}")
        print(f"Methods updated:        {total_stats['methods_updated']:,}")
        print(f"Assumptions updated:    {total_stats['assumptions_updated']:,}")
        print(f"Aliases merged:         {total_stats['aliases_merged']:,}")
        print(f"Concepts deleted:       {total_stats['concepts_deleted']:,}")
        print(f"Errors:                 {total_stats['errors']}")

        if dry_run:
            print("\n⚠️  This was a DRY RUN. Use --execute to apply changes.")
        else:
            print("\n✅ Deduplication complete!")

            # Verify final counts
            print("\nVerifying final counts...")
            final_count = await conn.fetchval("SELECT COUNT(*) FROM concepts")
            print(f"  Concepts remaining: {final_count:,}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
