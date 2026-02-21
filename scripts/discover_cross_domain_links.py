#!/usr/bin/env python3
"""Discover and store cross-domain concept links.

Finds semantically similar concepts across knowledge domains using embedding
similarity. Stores high-confidence matches to cross_domain_links table.

Usage:
    # Discover links between domains (dry run)
    python scripts/discover_cross_domain_links.py --dry-run

    # Discover and store links
    python scripts/discover_cross_domain_links.py

    # Custom thresholds
    python scripts/discover_cross_domain_links.py --threshold 0.80 --store-threshold 0.90

    # Full discovery (more concepts)
    python scripts/discover_cross_domain_links.py --full
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))

from research_kb_common import get_logger
from research_kb_storage import (
    CrossDomainStore,
    DatabaseConfig,
    get_connection_pool,
)

logger = get_logger(__name__)


async def discover_and_store(
    source_domain: str,
    target_domain: str,
    discovery_threshold: float = 0.80,
    store_threshold: float = 0.90,
    source_concept_limit: int = 100,
    limit_per_concept: int = 5,
    dry_run: bool = False,
) -> dict:
    """Discover and optionally store cross-domain links.

    Args:
        source_domain: Domain to search from
        target_domain: Domain to find matches in
        discovery_threshold: Minimum similarity for discovery (default 0.80)
        store_threshold: Minimum similarity to store (default 0.90)
        source_concept_limit: Max concepts to process (default 100)
        limit_per_concept: Max matches per concept (default 5)
        dry_run: If True, don't store links

    Returns:
        Summary statistics
    """
    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    print(f"\n{'='*60}")
    print(f"Cross-Domain Discovery: {source_domain} → {target_domain}")
    print(f"{'='*60}")
    print(f"Discovery threshold: {discovery_threshold:.2f}")
    print(f"Store threshold: {store_threshold:.2f}")
    print(f"Source concept limit: {source_concept_limit}")
    print(f"Dry run: {dry_run}")
    print()

    # Discover links
    print("Discovering links...")
    links = await CrossDomainStore.discover_links(
        source_domain=source_domain,
        target_domain=target_domain,
        similarity_threshold=discovery_threshold,
        source_concept_limit=source_concept_limit,
        limit_per_concept=limit_per_concept,
        min_relationship_count=2,
    )

    print(f"Found {len(links)} potential links\n")

    if not links:
        return {"discovered": 0, "stored": 0}

    # Sort by similarity
    links.sort(key=lambda x: x["similarity"], reverse=True)

    # Classify links
    equivalent = [l for l in links if l["similarity"] >= 0.95]
    analogous = [l for l in links if 0.90 <= l["similarity"] < 0.95]
    related = [l for l in links if l["similarity"] < 0.90]

    print("Classification (by similarity):")
    print(f"  EQUIVALENT (≥0.95): {len(equivalent)}")
    print(f"  ANALOGOUS (0.90-0.95): {len(analogous)}")
    print(f"  RELATED (<0.90): {len(related)}")
    print()

    # Show top matches
    print("Top 15 matches:")
    for i, link in enumerate(links[:15], 1):
        link_type = (
            "EQUIVALENT"
            if link["similarity"] >= 0.95
            else "ANALOGOUS" if link["similarity"] >= 0.90 else "RELATED"
        )
        print(
            f"  {i:2}. {link['source_name'][:30]:30} ↔ "
            f"{link['target_name'][:30]:30} [{link['similarity']:.3f}] {link_type}"
        )
    print()

    # Store if not dry run
    stored = 0
    if not dry_run:
        print(f"Storing links with similarity ≥ {store_threshold:.2f}...")
        stored = await CrossDomainStore.store_discovered_links(
            links=links,
            min_similarity=store_threshold,
        )
        print(f"Stored {stored} links\n")
    else:
        would_store = len([l for l in links if l["similarity"] >= store_threshold])
        print(f"Would store {would_store} links (dry run)\n")

    # Get stats
    stats = await CrossDomainStore.get_stats()
    print("Database Statistics:")
    print(f"  Total cross-domain links: {stats['total_links']}")
    if stats["by_type"]:
        print(f"  By type: {stats['by_type']}")

    return {
        "discovered": len(links),
        "equivalent": len(equivalent),
        "analogous": len(analogous),
        "related": len(related),
        "stored": stored,
    }


async def bidirectional_discovery(
    source_domain: str,
    target_domain: str,
    threshold: float = 0.85,
    store_threshold: float = 0.90,
    full: bool = False,
    dry_run: bool = False,
) -> None:
    """Run discovery in both directions between two domains.

    Args:
        source_domain: First domain
        target_domain: Second domain
        threshold: Discovery threshold
        store_threshold: Storage threshold
        full: Run full discovery (more concepts)
        dry_run: Don't store links
    """
    limit = 500 if full else 100

    # source → target
    result1 = await discover_and_store(
        source_domain=source_domain,
        target_domain=target_domain,
        discovery_threshold=threshold,
        store_threshold=store_threshold,
        source_concept_limit=limit,
        dry_run=dry_run,
    )

    # target → source
    result2 = await discover_and_store(
        source_domain=target_domain,
        target_domain=source_domain,
        discovery_threshold=threshold,
        store_threshold=store_threshold,
        source_concept_limit=limit,
        dry_run=dry_run,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"{source_domain} → {target_domain}: {result1['discovered']} discovered, {result1['stored']} stored"
    )
    print(
        f"{target_domain} → {source_domain}: {result2['discovered']} discovered, {result2['stored']} stored"
    )
    print(f"Total stored: {result1['stored'] + result2['stored']}")


def main():
    parser = argparse.ArgumentParser(
        description="Discover cross-domain concept links",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source_domain",
        help="Source domain to search from (e.g. causal_inference)",
    )
    parser.add_argument(
        "target_domain",
        help="Target domain to find matches in (e.g. time_series)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum similarity for discovery (default: 0.85)",
    )
    parser.add_argument(
        "--store-threshold",
        type=float,
        default=0.90,
        help="Minimum similarity to store (default: 0.90)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full discovery (500 concepts vs 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't store links, just show what would be stored",
    )
    parser.add_argument(
        "--one-way",
        action="store_true",
        help="Only run source → target (skip reverse direction)",
    )

    args = parser.parse_args()

    if args.one_way:
        asyncio.run(
            discover_and_store(
                source_domain=args.source_domain,
                target_domain=args.target_domain,
                discovery_threshold=args.threshold,
                store_threshold=args.store_threshold,
                source_concept_limit=500 if args.full else 100,
                dry_run=args.dry_run,
            )
        )
    else:
        asyncio.run(
            bidirectional_discovery(
                source_domain=args.source_domain,
                target_domain=args.target_domain,
                threshold=args.threshold,
                store_threshold=args.store_threshold,
                full=args.full,
                dry_run=args.dry_run,
            )
        )


if __name__ == "__main__":
    main()
