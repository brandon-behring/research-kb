#!/usr/bin/env python3
"""Backfill embeddings for existing concepts.

This script populates the `embedding` column for all concepts that don't have
embeddings yet. Embeddings are generated from: "{canonical_name}: {definition}"

Prerequisites:
- Embedding server must be running: python -m research_kb_pdf.embed_server
- Database must be accessible

Usage:
    python scripts/backfill_concept_embeddings.py [--batch-size 100] [--dry-run]

Example:
    # Check embedding server is running
    curl -s http://localhost:8765/health || python -m research_kb_pdf.embed_server &

    # Run backfill
    python scripts/backfill_concept_embeddings.py

    # Verify
    psql -c "SELECT COUNT(*) FROM concepts WHERE embedding IS NOT NULL"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))

from research_kb_common import get_logger
from research_kb_pdf.embedding_client import EmbeddingClient
from research_kb_storage.concept_store import ConceptStore

logger = get_logger(__name__)


async def get_concepts_without_embeddings(batch_size: int = 1000) -> list:
    """Fetch all concepts that don't have embeddings yet.

    Uses pagination to handle large numbers of concepts.
    """
    all_concepts = []
    offset = 0

    while True:
        concepts = await ConceptStore.list_all(limit=batch_size, offset=offset)
        if not concepts:
            break

        # Filter to only concepts without embeddings
        concepts_needing_embeddings = [c for c in concepts if c.embedding is None]
        all_concepts.extend(concepts_needing_embeddings)

        offset += batch_size
        logger.debug(
            "fetched_concepts_batch",
            offset=offset,
            batch_count=len(concepts),
            needing_embeddings=len(concepts_needing_embeddings),
        )

    return all_concepts


def format_embedding_text(concept) -> str:
    """Format concept data for embedding.

    Uses canonical_name + definition for semantic richness.
    If no definition, uses just the canonical name.
    """
    if concept.definition:
        return f"{concept.canonical_name}: {concept.definition}"
    return concept.canonical_name


async def backfill_embeddings(
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict:
    """Backfill embeddings for all concepts without them.

    Args:
        batch_size: Number of concepts to process per batch
        dry_run: If True, don't actually update database

    Returns:
        Stats dictionary with counts
    """
    logger.info("backfill_starting", batch_size=batch_size, dry_run=dry_run)

    # Check embedding server is available
    try:
        client = EmbeddingClient()
        status = client.ping()
        logger.info("embedding_server_connected", status=status)
    except ConnectionError as e:
        logger.error("embedding_server_not_available", error=str(e))
        print("\n❌ Embedding server not running!")
        print("Start it with: python -m research_kb_pdf.embed_server")
        return {"error": "embedding_server_not_available"}

    # Fetch concepts without embeddings
    concepts = await get_concepts_without_embeddings()
    total_count = len(concepts)

    if total_count == 0:
        logger.info("no_concepts_need_embeddings")
        print("\n✅ All concepts already have embeddings!")
        return {"total": 0, "updated": 0, "skipped": 0}

    logger.info("concepts_to_backfill", count=total_count)
    print(f"\n📊 Found {total_count} concepts without embeddings")

    if dry_run:
        print("🔍 DRY RUN - no changes will be made")
        # Sample a few concepts to show what would be embedded
        for concept in concepts[:5]:
            text = format_embedding_text(concept)
            print(f"  - {concept.canonical_name}: '{text[:80]}...'")
        return {"total": total_count, "updated": 0, "skipped": 0, "dry_run": True}

    # Process in batches
    updated = 0
    errors = 0

    for i in range(0, total_count, batch_size):
        batch = concepts[i : i + batch_size]
        texts = [format_embedding_text(c) for c in batch]

        try:
            # Generate embeddings for batch
            embeddings = client.embed_batch(texts)

            # Update each concept
            for concept, embedding in zip(batch, embeddings):
                try:
                    await ConceptStore.update(concept.id, embedding=embedding)
                    updated += 1
                except Exception as e:
                    logger.error(
                        "concept_update_failed",
                        concept_id=str(concept.id),
                        error=str(e),
                    )
                    errors += 1

            # Progress update
            progress = min(i + batch_size, total_count)
            pct = (progress / total_count) * 100
            print(f"  Progress: {progress}/{total_count} ({pct:.1f}%)", end="\r")

        except Exception as e:
            logger.error("batch_embedding_failed", batch_start=i, error=str(e))
            errors += len(batch)

    print()  # Newline after progress
    logger.info(
        "backfill_complete",
        total=total_count,
        updated=updated,
        errors=errors,
    )

    return {"total": total_count, "updated": updated, "errors": errors}


async def verify_embeddings() -> dict:
    """Verify embedding coverage after backfill."""
    from research_kb_storage.connection import get_connection_pool

    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM concepts")
        with_embeddings = await conn.fetchval(
            "SELECT COUNT(*) FROM concepts WHERE embedding IS NOT NULL"
        )
        without_embeddings = total - with_embeddings

    return {
        "total_concepts": total,
        "with_embeddings": with_embeddings,
        "without_embeddings": without_embeddings,
        "coverage_pct": (with_embeddings / total * 100) if total > 0 else 0,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for concepts without them"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of concepts to process per batch (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually update database, just show what would be done",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify current embedding coverage, don't backfill",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Concept Embedding Backfill")
    print("=" * 60)

    if args.verify_only:
        stats = await verify_embeddings()
        print(f"\n📊 Current embedding coverage:")
        print(f"  Total concepts: {stats['total_concepts']}")
        print(f"  With embeddings: {stats['with_embeddings']}")
        print(f"  Without embeddings: {stats['without_embeddings']}")
        print(f"  Coverage: {stats['coverage_pct']:.1f}%")
        return

    # Run backfill
    result = await backfill_embeddings(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    if "error" in result:
        sys.exit(1)

    # Verify final state
    if not args.dry_run:
        print("\n📊 Final verification:")
        stats = await verify_embeddings()
        print(f"  Total concepts: {stats['total_concepts']}")
        print(f"  With embeddings: {stats['with_embeddings']}")
        print(f"  Coverage: {stats['coverage_pct']:.1f}%")

        if stats["without_embeddings"] > 0:
            print(f"\n⚠️  {stats['without_embeddings']} concepts still without embeddings")
        else:
            print("\n✅ All concepts now have embeddings!")


if __name__ == "__main__":
    asyncio.run(main())
