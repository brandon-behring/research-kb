#!/usr/bin/env python3
"""Link interview_prep chunks to causal_inference concepts via embedding similarity.

Creates chunk_concepts links between interview Q&A cards and formal research concepts
using semantic similarity matching on BGE-large embeddings.

Usage:
    # Link vol2 cards to CI concepts (dry run)
    python scripts/link_interview_to_concepts.py --dry-run --limit 20

    # Execute full linking
    python scripts/link_interview_to_concepts.py

    # Custom thresholds
    python scripts/link_interview_to_concepts.py --threshold 0.85 --max-concepts 3

    # Filter by source title
    python scripts/link_interview_to_concepts.py --source-filter "vol2_causal_inference"
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from pgvector.asyncpg import register_vector
from research_kb_common import get_logger, configure_logging
from research_kb_storage import (
    DatabaseConfig,
    get_connection_pool,
    close_connection_pool,
)
from research_kb_storage.chunk_concept_store import ChunkConceptStore

configure_logging()
logger = get_logger(__name__)


async def get_chunks_with_embeddings(
    source_domain: str,
    source_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[tuple[UUID, list[float], str]]:
    """Load chunks from a domain with their embeddings.

    Args:
        source_domain: Domain ID (e.g., 'interview_prep')
        source_filter: Optional substring filter on source title
        limit: Maximum chunks to load

    Returns:
        List of (chunk_id, embedding, content_preview) tuples
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        await register_vector(conn)

        query = """
            SELECT c.id, c.embedding, substring(c.content, 1, 100) as preview
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE c.domain_id = $1
              AND c.embedding IS NOT NULL
        """
        params = [source_domain]

        if source_filter:
            query += " AND s.title ILIKE $2"
            params.append(f"%{source_filter}%")

        query += " ORDER BY c.created_at"

        if limit:
            query += f" LIMIT {limit}"

        rows = await conn.fetch(query, *params)

        return [(row["id"], list(row["embedding"]), row["preview"]) for row in rows]


async def find_similar_concepts(
    embedding: list[float],
    target_domain: str,
    threshold: float,
    max_concepts: int,
) -> list[tuple[UUID, str, float]]:
    """Find concepts in target domain similar to embedding.

    Args:
        embedding: Query embedding (1024-dim)
        target_domain: Domain to search (e.g., 'causal_inference')
        threshold: Minimum similarity (0.0-1.0)
        max_concepts: Maximum concepts to return

    Returns:
        List of (concept_id, concept_name, similarity) tuples
    """
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        await register_vector(conn)

        rows = await conn.fetch(
            """
            SELECT
                id,
                name,
                1.0 - (embedding <=> $1::vector(1024)) / 2.0 AS similarity
            FROM concepts
            WHERE domain_id = $2
              AND embedding IS NOT NULL
              AND 1.0 - (embedding <=> $1::vector(1024)) / 2.0 >= $3
            ORDER BY embedding <=> $1::vector(1024) ASC
            LIMIT $4
            """,
            embedding,
            target_domain,
            threshold,
            max_concepts,
        )

        return [(row["id"], row["name"], row["similarity"]) for row in rows]


async def count_existing_links(source_domain: str) -> int:
    """Count existing chunk_concepts links for a domain."""
    pool = await get_connection_pool()

    async with pool.acquire() as conn:
        count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM chunk_concepts cc
            JOIN chunks c ON cc.chunk_id = c.id
            WHERE c.domain_id = $1
            """,
            source_domain,
        )
        return count or 0


async def link_chunks_to_concepts(
    source_domain: str = "interview_prep",
    target_domain: str = "causal_inference",
    source_filter: Optional[str] = None,
    threshold: float = 0.80,
    max_concepts: int = 5,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """Link chunks from source domain to concepts in target domain.

    Args:
        source_domain: Domain containing chunks to link
        target_domain: Domain containing concepts to link to
        source_filter: Optional substring filter on source title
        threshold: Minimum similarity for creating link
        max_concepts: Max concepts to link per chunk
        dry_run: If True, don't create links
        limit: Max chunks to process

    Returns:
        Statistics dict
    """
    stats = {
        "chunks_processed": 0,
        "links_created": 0,
        "chunks_with_matches": 0,
        "avg_similarity": 0.0,
        "existing_links": 0,
    }

    # Check existing links
    stats["existing_links"] = await count_existing_links(source_domain)
    logger.info("existing_links", count=stats["existing_links"], domain=source_domain)

    # Load chunks
    logger.info("loading_chunks", domain=source_domain, filter=source_filter, limit=limit)
    chunks = await get_chunks_with_embeddings(source_domain, source_filter, limit)
    logger.info("chunks_loaded", count=len(chunks))

    if not chunks:
        logger.warning("no_chunks_found")
        return stats

    total_similarity = 0.0
    total_links = 0

    for i, (chunk_id, embedding, preview) in enumerate(chunks):
        # Find similar concepts
        similar = await find_similar_concepts(embedding, target_domain, threshold, max_concepts)

        if similar:
            stats["chunks_with_matches"] += 1

            for concept_id, concept_name, similarity in similar:
                total_similarity += similarity
                total_links += 1

                if dry_run:
                    if i < 5:  # Show first 5 examples
                        logger.info(
                            "would_create_link",
                            chunk_preview=preview[:50],
                            concept=concept_name,
                            similarity=f"{similarity:.3f}",
                        )
                else:
                    try:
                        await ChunkConceptStore.create(
                            chunk_id=chunk_id,
                            concept_id=concept_id,
                            mention_type="reference",
                            relevance_score=similarity,
                        )
                        stats["links_created"] += 1
                    except Exception as e:
                        # Link may already exist
                        if "duplicate" in str(e).lower():
                            pass  # Skip duplicates
                        else:
                            logger.warning("link_failed", error=str(e))

        stats["chunks_processed"] += 1

        # Progress logging
        if (i + 1) % 50 == 0:
            logger.info(
                "progress",
                processed=i + 1,
                total=len(chunks),
                links=total_links,
            )

    # Calculate average similarity
    if total_links > 0:
        stats["avg_similarity"] = total_similarity / total_links

    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="Link interview_prep chunks to causal_inference concepts"
    )
    parser.add_argument(
        "--source-domain",
        default="interview_prep",
        help="Source domain with chunks (default: interview_prep)",
    )
    parser.add_argument(
        "--target-domain",
        default="causal_inference",
        help="Target domain with concepts (default: causal_inference)",
    )
    parser.add_argument(
        "--source-filter",
        help="Filter source titles (e.g., 'vol2_causal_inference')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum similarity threshold (default: 0.80)",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=5,
        help="Max concepts per chunk (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without creating links",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max chunks to process",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    args = parser.parse_args()

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    try:
        logger.info(
            "starting_linkage",
            source=args.source_domain,
            target=args.target_domain,
            filter=args.source_filter,
            threshold=args.threshold,
            max_concepts=args.max_concepts,
            dry_run=args.dry_run,
        )

        stats = await link_chunks_to_concepts(
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            source_filter=args.source_filter,
            threshold=args.threshold,
            max_concepts=args.max_concepts,
            dry_run=args.dry_run,
            limit=args.limit,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Cross-Domain Linking Results")
        print("=" * 60)
        print(f"Source domain: {args.source_domain}")
        print(f"Target domain: {args.target_domain}")
        print(f"Source filter: {args.source_filter or 'None'}")
        print(f"Threshold: {args.threshold}")
        print(f"Max concepts per chunk: {args.max_concepts}")
        print(f"Dry run: {args.dry_run}")
        print("-" * 60)
        print(f"Chunks processed: {stats['chunks_processed']}")
        print(f"Chunks with matches: {stats['chunks_with_matches']}")
        print(f"Links created: {stats['links_created']}")
        print(f"Average similarity: {stats['avg_similarity']:.3f}")
        print(f"Existing links (before): {stats['existing_links']}")
        print("=" * 60)

    finally:
        await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
