#!/usr/bin/env python3
"""Validate Haiku 4.5 extraction on gold eval set.

Runs extraction on gold eval chunks and compares with Round 1 baseline.
Uses real-time API (not batch) for quick validation.
"""

import asyncio
import os
import sys
from pathlib import Path
from uuid import UUID

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "extraction" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

import asyncpg

from research_kb_extraction import get_llm_client
from research_kb_extraction.chunk_filter import filter_chunk, FilterDecision


async def main():
    """Run validation on gold eval set."""
    # Ensure API key is set
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key.startswith("  "):
        print("ERROR: ANTHROPIC_API_KEY not set or has leading spaces")
        return

    # Connect to database
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="research_kb",
    )

    # Get gold eval chunks with content
    rows = await conn.fetch("""
        SELECT
            g.chunk_id,
            g.category,
            g.r1_concept_count,
            c.content
        FROM gold_eval_chunks g
        JOIN chunks c ON g.chunk_id = c.id
        ORDER BY g.category, g.r1_concept_count
    """)

    print(f"=== Gold Eval Validation ===")
    print(f"Total chunks: {len(rows)}")

    # Initialize client
    client = get_llm_client("anthropic", model="haiku-4.5")

    # Track results
    results = {
        "hard": {"count": 0, "r1_total": 0, "new_total": 0, "new_relations": 0},
        "high_quality": {"count": 0, "r1_total": 0, "new_total": 0, "new_relations": 0},
    }

    type_distribution = {}
    errors = []

    # Process each chunk
    for i, row in enumerate(rows):
        chunk_id = row["chunk_id"]
        category = row["category"]
        r1_count = row["r1_concept_count"]
        content = row["content"]

        # Check filter
        filter_result = filter_chunk(content)
        if filter_result.decision == FilterDecision.SKIP:
            print(f"  [{i+1}/{len(rows)}] SKIP: {chunk_id} - {filter_result.reason}")
            continue

        try:
            # Extract
            extraction = await client.extract_concepts(content, "full")

            # Count concepts and relationships
            new_count = extraction.concept_count
            new_relations = extraction.relationship_count

            # Track type distribution
            for concept in extraction.concepts:
                t = concept.concept_type
                type_distribution[t] = type_distribution.get(t, 0) + 1

            # Update results
            results[category]["count"] += 1
            results[category]["r1_total"] += r1_count
            results[category]["new_total"] += new_count
            results[category]["new_relations"] += new_relations

            # Print progress
            status = "✓" if new_count >= r1_count else "!"
            print(
                f"  [{i+1}/{len(rows)}] {status} {category[:4]}: "
                f"R1={r1_count} → H45={new_count} concepts, {new_relations} rels"
            )

        except Exception as e:
            errors.append((chunk_id, str(e)))
            print(f"  [{i+1}/{len(rows)}] ERROR: {e}")

    await client.close()
    await conn.close()

    # Summary
    print("\n=== Summary ===")
    for cat, data in results.items():
        if data["count"] > 0:
            r1_avg = data["r1_total"] / data["count"]
            new_avg = data["new_total"] / data["count"]
            rel_avg = data["new_relations"] / data["count"]
            improvement = (new_avg - r1_avg) / max(r1_avg, 0.1) * 100
            print(f"\n{cat.upper()}:")
            print(f"  Chunks processed: {data['count']}")
            print(f"  Round 1 avg concepts: {r1_avg:.1f}")
            print(f"  Haiku 4.5 avg concepts: {new_avg:.1f} ({improvement:+.0f}%)")
            print(f"  Haiku 4.5 avg relations: {rel_avg:.1f}")

    print("\n=== Type Distribution ===")
    total_concepts = sum(type_distribution.values())
    for t, count in sorted(type_distribution.items(), key=lambda x: -x[1]):
        pct = count / max(total_concepts, 1) * 100
        print(f"  {t}: {count} ({pct:.1f}%)")

    if errors:
        print(f"\n=== Errors ({len(errors)}) ===")
        for chunk_id, err in errors[:5]:
            print(f"  {chunk_id}: {err[:50]}")


if __name__ == "__main__":
    asyncio.run(main())
