#!/usr/bin/env python3
"""Build citation graph from extracted citations.

This script:
1. Matches extracted citations to corpus sources
2. Creates source_citations edges
3. Computes PageRank-style authority scores
4. Reports statistics by source type

Usage:
    python scripts/build_citation_graph.py [--skip-pagerank]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_storage import (
    DatabaseConfig,
    close_connection_pool,
    get_connection_pool,
    build_citation_graph,
    compute_pagerank_authority,
    get_corpus_citation_summary,
    get_most_cited_sources,
)

logger = get_logger(__name__)


async def main():
    """Build citation graph and compute PageRank."""
    parser = argparse.ArgumentParser(description="Build citation graph")
    parser.add_argument("--skip-pagerank", action="store_true", help="Skip PageRank computation")
    rebuild_group = parser.add_mutually_exclusive_group()
    rebuild_group.add_argument(
        "--rebuild-unmatched",
        action="store_true",
        help=(
            "Delete source_citations rows with cited_source_id IS NULL before "
            "rebuild, so the matcher re-evaluates previously-external citations "
            "against the current corpus (e.g. after adding new anchor sources)."
        ),
    )
    rebuild_group.add_argument(
        "--full-rebuild",
        action="store_true",
        help=(
            "Delete ALL source_citations rows before rebuild. Use when matcher "
            "logic has been tightened (e.g. fewer false-positive partial-LIKE "
            "matches) and pre-existing edges may need re-evaluation. Slower."
        ),
    )
    args = parser.parse_args()

    # Initialize database
    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    # Check citation count
    async with pool.acquire() as conn:
        citation_count = await conn.fetchval("SELECT COUNT(*) FROM citations")
        source_count = await conn.fetchval("SELECT COUNT(*) FROM sources")
        existing_edges = await conn.fetchval("SELECT COUNT(*) FROM source_citations")

    print(f"Found {citation_count} citations from {source_count} sources")
    print(f"Existing citation graph edges: {existing_edges}")

    if citation_count == 0:
        print("No citations to process. Run extract_citations.py first.")
        return

    # Optionally delete prior edges to force re-evaluation under current matcher.
    if args.full_rebuild:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM source_citations")
            after = await conn.fetchval("SELECT COUNT(*) FROM source_citations")
        print(f"\n--full-rebuild: deleted {existing_edges} edges (now {after})")
    elif args.rebuild_unmatched:
        async with pool.acquire() as conn:
            null_before = await conn.fetchval(
                "SELECT COUNT(*) FROM source_citations WHERE cited_source_id IS NULL"
            )
            await conn.execute("DELETE FROM source_citations WHERE cited_source_id IS NULL")
        print(
            f"\n--rebuild-unmatched: deleted {null_before} NULL-target edges; "
            f"non-NULL edges preserved"
        )

    # Build citation graph
    print("\n" + "=" * 70)
    print("BUILDING CITATION GRAPH")
    print("=" * 70)

    stats = await build_citation_graph()

    print("\nGraph building complete:")
    print(f"  Total processed: {stats['total_processed']}")
    print(
        f"  Matched to corpus: {stats['matched']} ({100*stats['matched']/max(stats['total_processed'],1):.1f}%)"
    )
    print(f"  External (unmatched): {stats['unmatched']}")
    print(f"  Errors: {stats['errors']}")

    if stats["by_type"]:
        print("\nBy source type:")
        for key, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
            print(f"  {key}: {count}")

    # Compute PageRank
    if not args.skip_pagerank:
        print("\n" + "=" * 70)
        print("COMPUTING PAGERANK AUTHORITY")
        print("=" * 70)

        pr_stats = await compute_pagerank_authority(iterations=20, damping=0.85)

        print("\nPageRank computation complete:")
        print(f"  Sources: {pr_stats['sources']}")
        print(f"  Min score: {pr_stats['min_score']:.4f}")
        print(f"  Max score: {pr_stats['max_score']:.4f}")
        print(f"  Mean score: {pr_stats['mean_score']:.4f}")

    # Show summary
    print("\n" + "=" * 70)
    print("CITATION GRAPH SUMMARY")
    print("=" * 70)

    summary = await get_corpus_citation_summary()
    print("\nCorpus-wide statistics:")
    print(f"  Total citations: {summary.get('total_citations', 0)}")
    print(f"  Total edges: {summary.get('total_edges', 0)}")
    print(f"  Internal edges: {summary.get('internal_edges', 0)}")
    print(f"  External edges: {summary.get('external_edges', 0)}")
    print("\nBy type combination:")
    print(f"  Paper → Paper: {summary.get('paper_to_paper', 0)}")
    print(f"  Paper → Textbook: {summary.get('paper_to_textbook', 0)}")
    print(f"  Textbook → Paper: {summary.get('textbook_to_paper', 0)}")
    print(f"  Textbook → Textbook: {summary.get('textbook_to_textbook', 0)}")

    # Show most cited
    print("\n" + "-" * 70)
    print("TOP 10 MOST CITED SOURCES")
    print("-" * 70)

    most_cited = await get_most_cited_sources(limit=10)
    for i, source in enumerate(most_cited, 1):
        print(f"{i:2}. [{source['source_type']:8}] {source['title'][:50]}...")
        print(
            f"    Cited by: {source['cited_by_count']} | Authority: {source['citation_authority']:.4f}"
        )

    await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
