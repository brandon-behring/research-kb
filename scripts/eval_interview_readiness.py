#!/usr/bin/env python3
"""Evaluate interview readiness based on research-kb knowledge base.

Measures four readiness dimensions:
1. Method query hit rate (top 10 methods)
2. Assumption audit completeness (>= 3 assumptions per method)
3. Domain readiness (domains with >1000 chunks)
4. Overall readiness score (weighted average)

Output: JSON + human-readable table.

Usage:
    python scripts/eval_interview_readiness.py
    python scripts/eval_interview_readiness.py --json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool

# Top 10 causal inference methods every interview prep candidate should cover
TOP_METHODS = [
    "instrumental variables",
    "double machine learning",
    "difference-in-differences",
    "propensity score matching",
    "inverse probability weighting",
    "doubly robust estimation",
    "regression discontinuity design",
    "causal forests",
    "CUPED",
    "LATE",
]

# Minimum assumptions per method for "complete" coverage
MIN_ASSUMPTIONS = 3

# Domains with expected chunk counts for readiness
DOMAIN_THRESHOLDS = {
    "causal_inference": 1000,
    "time_series": 500,
    "interview_prep": 1000,
    "rag_llm": 1000,
}


async def check_method_hit_rate(conn) -> dict:
    """Check how many top methods have search hits.

    Args:
        conn: asyncpg connection

    Returns:
        Dict with per-method hit status and overall rate
    """
    results = {}
    hits = 0

    for method in TOP_METHODS:
        count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE c.content ILIKE $1
            """,
            f"%{method}%",
        )
        hit = count > 0
        results[method] = {"chunks": count, "hit": hit}
        if hit:
            hits += 1

    return {
        "methods": results,
        "hits": hits,
        "total": len(TOP_METHODS),
        "rate": hits / len(TOP_METHODS),
    }


async def check_assumption_completeness(conn) -> dict:
    """Check assumption audit completeness per method.

    Args:
        conn: asyncpg connection

    Returns:
        Dict with per-method assumption counts and completeness rate
    """
    results = {}
    complete = 0

    for method in TOP_METHODS:
        # Check method_assumption_cache (pre-computed audits via concept join)
        cached = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM method_assumption_cache mac
            JOIN concepts c ON mac.method_concept_id = c.id
            WHERE c.canonical_name ILIKE $1
               OR c.name ILIKE $1
            """,
            f"%{method}%",
        )

        # Also check concepts table for assumption-type concepts related to this method
        # Uses both directions (source→target and target→source) since relationship direction varies
        concept_count = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT a.id)
            FROM concepts a
            WHERE a.concept_type = 'ASSUMPTION'
            AND (
                EXISTS (
                    SELECT 1 FROM concept_relationships cr
                    JOIN concepts m ON cr.source_concept_id = m.id
                    WHERE cr.target_concept_id = a.id
                    AND (m.canonical_name ILIKE $1 OR m.name ILIKE $1)
                )
                OR EXISTS (
                    SELECT 1 FROM concept_relationships cr
                    JOIN concepts m ON cr.target_concept_id = m.id
                    WHERE cr.source_concept_id = a.id
                    AND (m.canonical_name ILIKE $1 OR m.name ILIKE $1)
                )
            )
            """,
            f"%{method}%",
        )

        total = cached + concept_count
        is_complete = total >= MIN_ASSUMPTIONS
        results[method] = {
            "cached_audits": cached,
            "graph_assumptions": concept_count,
            "total": total,
            "complete": is_complete,
        }
        if is_complete:
            complete += 1

    return {
        "methods": results,
        "complete": complete,
        "total": len(TOP_METHODS),
        "rate": complete / len(TOP_METHODS),
    }


async def check_domain_readiness(conn) -> dict:
    """Check domain readiness by chunk count.

    Args:
        conn: asyncpg connection

    Returns:
        Dict with per-domain chunk counts and readiness status
    """
    results = {}
    ready = 0

    for domain, threshold in DOMAIN_THRESHOLDS.items():
        # Count chunks via source domain tag (domain is on sources, not chunks)
        count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE s.metadata->>'domain' = $1
            """,
            domain,
        )

        # Count sources with this domain
        source_count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM sources s
            WHERE s.metadata->>'domain' = $1
            """,
            domain,
        )

        is_ready = count >= threshold
        results[domain] = {
            "chunks": count,
            "sources": source_count,
            "threshold": threshold,
            "ready": is_ready,
        }
        if is_ready:
            ready += 1

    # Get total corpus counts for context
    total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
    total_sources = await conn.fetchval("SELECT COUNT(*) FROM sources")

    return {
        "domains": results,
        "ready": ready,
        "total": len(DOMAIN_THRESHOLDS),
        "rate": ready / len(DOMAIN_THRESHOLDS),
        "corpus_chunks": total_chunks,
        "corpus_sources": total_sources,
    }


def compute_readiness_score(
    method_rate: float,
    assumption_rate: float,
    domain_rate: float,
) -> float:
    """Compute weighted overall readiness score.

    Weights:
    - Method hit rate: 40% (must be able to find content for key methods)
    - Assumption completeness: 35% (North Star feature)
    - Domain readiness: 25% (corpus breadth)

    Args:
        method_rate: Fraction of methods with search hits (0-1)
        assumption_rate: Fraction of methods with sufficient assumptions (0-1)
        domain_rate: Fraction of domains above chunk threshold (0-1)

    Returns:
        Weighted score (0-1)
    """
    return 0.40 * method_rate + 0.35 * assumption_rate + 0.25 * domain_rate


async def main():
    """Run readiness evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate interview readiness from research-kb")
    parser.add_argument("--json", action="store_true", help="JSON output only")
    args = parser.parse_args()

    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    async with pool.acquire() as conn:
        method_results = await check_method_hit_rate(conn)
        assumption_results = await check_assumption_completeness(conn)
        domain_results = await check_domain_readiness(conn)

    overall_score = compute_readiness_score(
        method_results["rate"],
        assumption_results["rate"],
        domain_results["rate"],
    )

    output = {
        "overall_score": round(overall_score, 3),
        "overall_pct": f"{overall_score * 100:.1f}%",
        "method_hit_rate": method_results,
        "assumption_completeness": assumption_results,
        "domain_readiness": domain_results,
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    # Human-readable output
    print("=" * 70)
    print("INTERVIEW READINESS EVALUATION")
    print("=" * 70)

    print(f"\nOverall Score: {overall_score * 100:.1f}%")
    print()

    # Method hit rate
    print(f"Method Hit Rate: {method_results['hits']}/{method_results['total']} "
          f"({method_results['rate']*100:.0f}%)")
    print("-" * 50)
    for method, data in method_results["methods"].items():
        status = "HIT" if data["hit"] else "MISS"
        print(f"  {method:<35} {status:>5} ({data['chunks']} chunks)")

    print()

    # Assumption completeness
    print(f"Assumption Completeness: {assumption_results['complete']}/{assumption_results['total']} "
          f"({assumption_results['rate']*100:.0f}%)")
    print("-" * 50)
    for method, data in assumption_results["methods"].items():
        status = "OK" if data["complete"] else "LOW"
        print(f"  {method:<35} {status:>4} ({data['total']} assumptions)")

    print()

    # Domain readiness
    print(f"Domain Readiness: {domain_results['ready']}/{domain_results['total']} "
          f"({domain_results['rate']*100:.0f}%)")
    print("-" * 50)
    for domain, data in domain_results["domains"].items():
        status = "READY" if data["ready"] else "NEEDS WORK"
        print(f"  {domain:<25} {status:>10} "
              f"({data['chunks']:,} chunks / {data['threshold']:,} threshold)")

    print(f"\n  Total corpus: {domain_results['corpus_sources']:,} sources, "
          f"{domain_results['corpus_chunks']:,} chunks")

    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
