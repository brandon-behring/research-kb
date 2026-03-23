#!/usr/bin/env python3
"""
Re-tag existing mathematics sources to new sub-domains.

Queries the 29 sources with domain_id='mathematics', classifies each
by title keywords, and updates both sources.domain_id and chunks.domain_id.

Usage:
    python scripts/catalog_library/retag_math_sources.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import defaultdict

import asyncpg

# Same keyword taxonomy as catalog_books.py — math sub-domains only
MATH_SUB_DOMAINS: dict[str, list[str]] = {
    "algebra": [
        "algebra", "galois", "group theory", "ring", "field theory",
        "commutative", "homological", "representation theory", "module",
        "abstract algebra", "lie algebra", "lie group",
    ],
    "analysis": [
        "real analysis", "complex analysis", "functional analysis",
        "harmonic analysis", "measure theory", "fourier analysis",
        "operator theory", "spectral theory", "lebesgue", "banach",
        "hilbert space", "sobolev",
    ],
    "topology_geometry": [
        "topology", "manifold", "riemannian", "cohomology", "homotopy",
        "differential geometry", "algebraic geometry", "knot theory",
        "symplectic", "differential forms",
    ],
    "dynamical_systems": [
        "dynamical system", "bifurcation", "chaos", "hamiltonian",
        "perturbation", "stability", "nonlinear", "celestial mechanics",
        "n-body", "ergodic", "kam",
    ],
    "probability_theory": [
        "probability", "stochastic", "markov", "martingale",
        "random", "brownian", "measure-theoretic probability",
    ],
    "numerical_methods": [
        "numerical analysis", "optimization", "convex", "finite element",
        "approximation theory", "interpolation", "computational mathematics",
    ],
    "linear_algebra": [
        "linear algebra", "matrix", "eigenvalue", "vector space",
        "svd", "tensor",
    ],
}


def classify_math_source(title: str) -> tuple[str, str]:
    """Classify a mathematics source into a sub-domain.

    Returns (domain_id, confidence). Falls back to 'mathematics' if no match.
    """
    title_lower = title.lower()
    scores: dict[str, int] = defaultdict(int)

    for domain, keywords in MATH_SUB_DOMAINS.items():
        for kw in keywords:
            if kw in title_lower:
                scores[domain] += 1

    if not scores:
        return "mathematics", "low"

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_domain, best_score = ranked[0]
    confidence = "high" if best_score >= 2 else "medium"
    return best_domain, confidence


async def retag_math_sources(dry_run: bool = True) -> None:
    """Re-tag mathematics sources to sub-domains."""
    dsn = "postgresql://localhost/research_kb"
    conn = await asyncpg.connect(dsn)

    try:
        rows = await conn.fetch(
            "SELECT id, title, domain_id FROM sources WHERE domain_id = 'mathematics' ORDER BY title"
        )
        print(f"Found {len(rows)} sources with domain_id='mathematics'\n")

        if not rows:
            print("Nothing to retag.")
            return

        changes: list[tuple[str, str, str, str]] = []  # (id, title, old_domain, new_domain)
        unchanged: list[tuple[str, str]] = []

        for row in rows:
            source_id = str(row["id"])
            title = row["title"]
            new_domain, confidence = classify_math_source(title)

            if new_domain != "mathematics":
                changes.append((source_id, title, "mathematics", new_domain))
                marker = f"  -> {new_domain} ({confidence})"
            else:
                unchanged.append((source_id, title))
                marker = "  -> mathematics (no match, keeping)"

            print(f"  {title[:70]:<70s}{marker}")

        print(f"\n--- Summary ---")
        print(f"  Re-tagged:  {len(changes)}")
        print(f"  Unchanged:  {len(unchanged)}")

        if dry_run:
            print("\n[DRY RUN] No changes applied. Remove --dry-run to apply.")
            return

        # Apply changes
        print("\nApplying changes...")
        for source_id, title, old_domain, new_domain in changes:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE sources SET domain_id = $1 WHERE id = $2::uuid",
                    new_domain, source_id,
                )
                updated = await conn.execute(
                    "UPDATE chunks SET domain_id = $1 WHERE source_id = $2::uuid",
                    new_domain, source_id,
                )
                chunk_count = int(updated.split()[-1])
                print(f"  {title[:60]}: {old_domain} -> {new_domain} ({chunk_count} chunks)")

        print(f"\nDone. Re-tagged {len(changes)} sources.")
        print("Remember to run: python scripts/sync_kuzu.py  (to update KuzuDB)")

    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-tag mathematics sources to sub-domains.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show proposed changes without applying (default: True)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply the changes (overrides --dry-run)",
    )
    args = parser.parse_args()
    dry_run = not args.apply

    asyncio.run(retag_math_sources(dry_run=dry_run))


if __name__ == "__main__":
    main()
