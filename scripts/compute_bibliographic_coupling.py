#!/usr/bin/env python3
"""Compute bibliographic coupling for all sources.

Bibliographic coupling measures similarity between sources based on
their shared references. Higher coupling = more similar research focus.

Usage:
    python scripts/compute_bibliographic_coupling.py
    python scripts/compute_bibliographic_coupling.py --min-coupling 0.15

After running, use:
    research-kb biblio-similar "source title" --limit 10
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_storage import BiblioStore, DatabaseConfig, get_connection_pool

logger = get_logger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Compute bibliographic coupling")
    parser.add_argument(
        "--min-coupling",
        type=float,
        default=0.1,
        help="Minimum coupling strength to store (0.0-1.0, default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for database inserts (default: 100)",
    )
    args = parser.parse_args()

    # Initialize connection pool
    config = DatabaseConfig()
    await get_connection_pool(config)

    print(f"Computing bibliographic coupling (min_coupling={args.min_coupling})...")
    print()

    # Compute all coupling
    stats = await BiblioStore.compute_all_coupling(
        min_coupling=args.min_coupling,
        batch_size=args.batch_size,
    )

    print()
    print("=" * 60)
    print("BIBLIOGRAPHIC COUPLING COMPLETE")
    print("=" * 60)
    print(f"Sources processed: {stats['total_sources']}")
    print(f"Pairs computed: {stats['pairs_computed']}")
    print(f"Pairs stored (coupling >= {args.min_coupling}): {stats['pairs_stored']}")
    print()

    # Show statistics
    coupling_stats = await BiblioStore.get_stats()
    print("Statistics:")
    print(f"  Total pairs: {coupling_stats['total_pairs']}")
    print(f"  Avg coupling: {coupling_stats['avg_coupling']:.3f}")
    print(f"  Max coupling: {coupling_stats['max_coupling']:.3f}")
    print(f"  Sources with coupling: {coupling_stats['sources_involved']}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
