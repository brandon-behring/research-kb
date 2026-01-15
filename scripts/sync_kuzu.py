#!/usr/bin/env python3
"""Sync PostgreSQL graph data to KuzuDB.

Phase: KuzuDB Migration

This script exports concepts and relationships from PostgreSQL
and loads them into KuzuDB for fast graph traversal.

Usage:
    # Full sync (clear and reload)
    python scripts/sync_kuzu.py

    # Incremental sync (add new only)
    python scripts/sync_kuzu.py --incremental

    # Verify data integrity
    python scripts/sync_kuzu.py --verify-only

    # Custom database path
    python scripts/sync_kuzu.py --kuzu-path /path/to/kuzu.db

Performance:
    - Full sync: ~2-3 minutes for 284K concepts, 726K relationships
    - Incremental: Depends on delta size
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))

from research_kb_common import get_logger, configure_logging
from research_kb_storage.connection import get_connection_pool, close_connection_pool
from research_kb_storage.kuzu_store import (
    get_kuzu_connection,
    close_kuzu_connection,
    clear_all_data,
    bulk_insert_concepts,
    bulk_insert_relationships,
    get_stats as get_kuzu_stats,
    DEFAULT_KUZU_PATH,
)

# Ensure parent directory exists for default path
DEFAULT_KUZU_PATH.parent.mkdir(parents=True, exist_ok=True)

configure_logging()
logger = get_logger(__name__)


async def fetch_concepts_from_postgres() -> list[dict]:
    """Fetch all concepts from PostgreSQL.

    Returns:
        List of concept dicts with id, name, canonical_name, concept_type
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, canonical_name, concept_type
            FROM concepts
            ORDER BY id
            """
        )

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "canonical_name": row["canonical_name"],
                "concept_type": row["concept_type"],
            }
            for row in rows
        ]


async def fetch_relationships_from_postgres() -> list[dict]:
    """Fetch all relationships from PostgreSQL.

    Returns:
        List of relationship dicts with source_id, target_id, relationship_type, strength
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_concept_id, target_concept_id, relationship_type, strength
            FROM concept_relationships
            ORDER BY id
            """
        )

        return [
            {
                "source_id": row["source_concept_id"],
                "target_id": row["target_concept_id"],
                "relationship_type": row["relationship_type"],
                "strength": float(row["strength"]) if row["strength"] else 1.0,
            }
            for row in rows
        ]


async def get_postgres_counts() -> tuple[int, int]:
    """Get concept and relationship counts from PostgreSQL."""
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        concept_count = await conn.fetchval("SELECT COUNT(*) FROM concepts")
        rel_count = await conn.fetchval("SELECT COUNT(*) FROM concept_relationships")

    return concept_count, rel_count


async def full_sync(kuzu_path: Path) -> dict:
    """Perform full sync: clear KuzuDB and reload from PostgreSQL.

    Args:
        kuzu_path: Path to KuzuDB database

    Returns:
        Sync statistics dict
    """
    start_time = time.time()
    stats = {
        "mode": "full",
        "concepts_synced": 0,
        "relationships_synced": 0,
        "duration_seconds": 0,
        "errors": [],
    }

    try:
        # Initialize connections
        logger.info("sync_starting", mode="full", kuzu_path=str(kuzu_path))
        get_kuzu_connection(kuzu_path)

        # Step 1: Clear existing KuzuDB data
        logger.info("clearing_kuzu_data")
        cleared = await clear_all_data()
        logger.info("kuzu_data_cleared", concepts_cleared=cleared)

        # Step 2: Fetch concepts from PostgreSQL
        logger.info("fetching_concepts_from_postgres")
        concepts = await fetch_concepts_from_postgres()
        logger.info("concepts_fetched", count=len(concepts))

        # Step 3: Insert concepts into KuzuDB
        logger.info("inserting_concepts_into_kuzu")
        stats["concepts_synced"] = await bulk_insert_concepts(concepts)
        logger.info("concepts_inserted", count=stats["concepts_synced"])

        # Step 4: Fetch relationships from PostgreSQL
        logger.info("fetching_relationships_from_postgres")
        relationships = await fetch_relationships_from_postgres()
        logger.info("relationships_fetched", count=len(relationships))

        # Step 5: Insert relationships into KuzuDB
        logger.info("inserting_relationships_into_kuzu")
        stats["relationships_synced"] = await bulk_insert_relationships(relationships)
        logger.info("relationships_inserted", count=stats["relationships_synced"])

        stats["duration_seconds"] = time.time() - start_time
        logger.info(
            "sync_completed",
            concepts=stats["concepts_synced"],
            relationships=stats["relationships_synced"],
            duration=f"{stats['duration_seconds']:.1f}s",
        )

    except Exception as e:
        stats["errors"].append(str(e))
        logger.error("sync_failed", error=str(e))
        raise

    finally:
        await close_connection_pool()
        close_kuzu_connection()

    return stats


async def verify_sync(kuzu_path: Path) -> dict:
    """Verify data integrity between PostgreSQL and KuzuDB.

    Args:
        kuzu_path: Path to KuzuDB database

    Returns:
        Verification results dict
    """
    results = {
        "postgres_concepts": 0,
        "postgres_relationships": 0,
        "kuzu_concepts": 0,
        "kuzu_relationships": 0,
        "concepts_match": False,
        "relationships_match": False,
        "verified": False,
    }

    try:
        # Get PostgreSQL counts
        logger.info("verifying_postgres_counts")
        pg_concepts, pg_rels = await get_postgres_counts()
        results["postgres_concepts"] = pg_concepts
        results["postgres_relationships"] = pg_rels

        # Get KuzuDB counts
        logger.info("verifying_kuzu_counts")
        get_kuzu_connection(kuzu_path)
        kuzu_stats = await get_kuzu_stats()
        results["kuzu_concepts"] = kuzu_stats["concept_count"]
        results["kuzu_relationships"] = kuzu_stats["relationship_count"]

        # Compare
        results["concepts_match"] = pg_concepts == results["kuzu_concepts"]
        results["relationships_match"] = pg_rels == results["kuzu_relationships"]
        results["verified"] = results["concepts_match"] and results["relationships_match"]

        if results["verified"]:
            logger.info(
                "verification_passed",
                concepts=pg_concepts,
                relationships=pg_rels,
            )
        else:
            logger.warning(
                "verification_failed",
                pg_concepts=pg_concepts,
                kuzu_concepts=results["kuzu_concepts"],
                pg_relationships=pg_rels,
                kuzu_relationships=results["kuzu_relationships"],
            )

    finally:
        await close_connection_pool()
        close_kuzu_connection()

    return results


def print_stats(stats: dict) -> None:
    """Pretty-print sync statistics."""
    print("\n" + "=" * 60)
    print("KuzuDB Sync Results")
    print("=" * 60)

    if "mode" in stats:
        print(f"Mode: {stats['mode']}")
        print(f"Concepts synced: {stats['concepts_synced']:,}")
        print(f"Relationships synced: {stats['relationships_synced']:,}")
        print(f"Duration: {stats['duration_seconds']:.1f}s")

        if stats["errors"]:
            print("\nErrors:")
            for err in stats["errors"]:
                print(f"  - {err}")
    else:
        # Verification results
        print(f"PostgreSQL concepts: {stats['postgres_concepts']:,}")
        print(f"KuzuDB concepts: {stats['kuzu_concepts']:,}")
        print(f"PostgreSQL relationships: {stats['postgres_relationships']:,}")
        print(f"KuzuDB relationships: {stats['kuzu_relationships']:,}")
        print()
        print(f"Concepts match: {'✓' if stats['concepts_match'] else '✗'}")
        print(f"Relationships match: {'✓' if stats['relationships_match'] else '✗'}")
        print()
        if stats["verified"]:
            print("✓ Verification PASSED")
        else:
            print("✗ Verification FAILED")

    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(
        description="Sync PostgreSQL graph data to KuzuDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full sync (clear and reload)
    python scripts/sync_kuzu.py

    # Verify only (no changes)
    python scripts/sync_kuzu.py --verify-only

    # Custom path
    python scripts/sync_kuzu.py --kuzu-path /data/research_kb/kuzu.db
        """,
    )

    parser.add_argument(
        "--kuzu-path",
        type=Path,
        default=DEFAULT_KUZU_PATH,
        help=f"Path to KuzuDB database (default: {DEFAULT_KUZU_PATH})",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify data integrity, don't sync",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental sync (not yet implemented)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    if args.incremental:
        print("Incremental sync not yet implemented. Use full sync for now.")
        sys.exit(1)

    try:
        if args.verify_only:
            results = await verify_sync(args.kuzu_path)
            if not args.quiet:
                print_stats(results)
            sys.exit(0 if results["verified"] else 1)
        else:
            stats = await full_sync(args.kuzu_path)
            if not args.quiet:
                print_stats(stats)

            # Verify after sync
            print("\nVerifying sync...")
            results = await verify_sync(args.kuzu_path)
            if not args.quiet:
                print_stats(results)

            sys.exit(0 if results["verified"] else 1)

    except KeyboardInterrupt:
        print("\nSync interrupted.")
        sys.exit(130)
    except Exception as e:
        logger.error("sync_script_failed", error=str(e))
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
