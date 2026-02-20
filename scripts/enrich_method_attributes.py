#!/usr/bin/env python3
"""Enrich method and assumption concepts with structured attributes.

Uses Anthropic Batch API (Haiku 4.5) to infer:
- For methods: required_assumptions, problem_types, common_estimators
- For assumptions: mathematical_statement, is_testable, common_tests, violation_consequences

Features:
- Processes ~25K method/assumption concepts
- 50% batch API discount
- Evidence tracking via chunk_ids
- Inference confidence scores

Usage:
    python scripts/enrich_method_attributes.py
    python scripts/enrich_method_attributes.py --dry-run
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

import asyncpg

from research_kb_common import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("<PROJECT_ROOT>/batch_results/enrichment")
BATCH_SIZE = 10000  # Stay under 256MB limit

# Prompt templates for attribute enrichment
METHOD_PROMPT = """You are a causal inference expert. Given a method concept from a causal inference knowledge base, infer its structured attributes.

**Concept Name:** {name}
**Definition:** {definition}
**Source Chunk ID:** {chunk_id}

Analyze this method and provide:
1. **required_assumptions**: List of statistical/causal assumptions this method requires (e.g., "unconfoundedness", "overlap", "SUTVA")
2. **problem_types**: Types of causal inference problems this method addresses (e.g., "ATE estimation", "heterogeneous treatment effects", "mediation analysis")
3. **common_estimators**: Specific estimators or algorithms that implement this method (e.g., "IPW estimator", "doubly robust estimator")
4. **confidence**: Your confidence in these inferences (0.0-1.0)

Respond with a JSON object only, no additional text."""

ASSUMPTION_PROMPT = """You are a causal inference expert. Given an assumption concept from a causal inference knowledge base, infer its structured attributes.

**Concept Name:** {name}
**Definition:** {definition}
**Source Chunk ID:** {chunk_id}

Analyze this assumption and provide:
1. **mathematical_statement**: Formal mathematical expression if applicable (e.g., "Y(1), Y(0) ⊥ T | X")
2. **is_testable**: Whether this assumption can be empirically tested (true/false)
3. **common_tests**: List of statistical tests to check this assumption (e.g., "balance tests", "sensitivity analysis")
4. **violation_consequences**: What happens if this assumption is violated (e.g., "biased treatment effect estimates")
5. **confidence**: Your confidence in these inferences (0.0-1.0)

Respond with a JSON object only, no additional text."""


async def get_concepts_to_enrich(conn) -> tuple[list[dict], list[dict]]:
    """Fetch method and assumption concepts that need enrichment."""

    # Get methods not yet in methods table
    methods = await conn.fetch(
        """
        SELECT c.id::text, c.canonical_name, c.definition,
               (SELECT cc.chunk_id::text
                FROM chunk_concepts cc
                WHERE cc.concept_id = c.id
                ORDER BY cc.relevance_score DESC NULLS LAST
                LIMIT 1) as evidence_chunk_id
        FROM concepts c
        WHERE c.concept_type = 'method'
        AND NOT EXISTS (
            SELECT 1 FROM methods m WHERE m.concept_id = c.id
        )
        ORDER BY c.canonical_name
    """
    )

    # Get assumptions not yet in assumptions table
    assumptions = await conn.fetch(
        """
        SELECT c.id::text, c.canonical_name, c.definition,
               (SELECT cc.chunk_id::text
                FROM chunk_concepts cc
                WHERE cc.concept_id = c.id
                ORDER BY cc.relevance_score DESC NULLS LAST
                LIMIT 1) as evidence_chunk_id
        FROM concepts c
        WHERE c.concept_type = 'assumption'
        AND NOT EXISTS (
            SELECT 1 FROM assumptions a WHERE a.concept_id = c.id
        )
        ORDER BY c.canonical_name
    """
    )

    return [dict(r) for r in methods], [dict(r) for r in assumptions]


def build_batch_request(concept: dict, concept_type: str) -> dict:
    """Build a single batch request for enrichment."""

    if concept_type == "method":
        prompt = METHOD_PROMPT.format(
            name=concept["canonical_name"],
            definition=concept["definition"] or "No definition available",
            chunk_id=concept["evidence_chunk_id"] or "unknown",
        )
    else:
        prompt = ASSUMPTION_PROMPT.format(
            name=concept["canonical_name"],
            definition=concept["definition"] or "No definition available",
            chunk_id=concept["evidence_chunk_id"] or "unknown",
        )

    # custom_id must match ^[a-zA-Z0-9_-]{1,64}$ - use underscore separator
    # UUID hyphens are replaced with underscores to fit pattern
    safe_id = concept["id"].replace("-", "_")
    return {
        "custom_id": f"{concept_type}_{safe_id}",
        "params": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 1024,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


async def submit_batch(client, requests: list[dict], batch_name: str) -> str:
    """Submit a batch to Anthropic API."""

    # Write requests to JSONL file
    jsonl_file = OUTPUT_DIR / f"{batch_name}.jsonl"
    with open(jsonl_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Submit batch
    with open(jsonl_file, "rb") as f:
        batch = client.messages.batches.create(
            requests=[json.loads(line) for line in f],
        )

    logger.info("batch_submitted", batch_id=batch.id, request_count=len(requests))
    return batch.id


async def main():
    """Submit method/assumption enrichment batches."""
    import argparse

    parser = argparse.ArgumentParser(description="Enrich method/assumption concepts")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="research_kb",
    )

    print("=" * 60)
    print("Method/Assumption Enrichment")
    print("=" * 60)

    # Get concepts to enrich
    methods, assumptions = await get_concepts_to_enrich(conn)
    print(f"\nMethods to enrich: {len(methods)}")
    print(f"Assumptions to enrich: {len(assumptions)}")
    print(f"Total: {len(methods) + len(assumptions)}")

    if len(methods) == 0 and len(assumptions) == 0:
        print("\n✅ All concepts already enriched!")
        await conn.close()
        return

    # Show samples
    print("\n--- Sample Methods ---")
    for m in methods[:3]:
        print(f"  {m['canonical_name']}: {(m['definition'] or '')[:60]}...")

    print("\n--- Sample Assumptions ---")
    for a in assumptions[:3]:
        print(f"  {a['canonical_name']}: {(a['definition'] or '')[:60]}...")

    # Cost estimate
    total_concepts = len(methods) + len(assumptions)
    est_input_tokens = total_concepts * 500  # ~500 tokens per concept with prompt
    est_output_tokens = total_concepts * 200  # ~200 tokens per response
    est_cost = (est_input_tokens * 0.80 + est_output_tokens * 4.00) / 1_000_000 * 0.5

    print("\n=== Cost Estimate ===")
    print(f"Input tokens: ~{est_input_tokens/1e6:.2f}M")
    print(f"Output tokens: ~{est_output_tokens/1e6:.2f}M")
    print(f"Estimated cost: ~${est_cost:.2f} (with 50% batch discount)")

    if args.dry_run:
        print("\n🔍 DRY RUN - no batches submitted")

        # Show sample request
        if methods:
            sample = build_batch_request(methods[0], "method")
            print("\nSample method request:")
            print(json.dumps(sample, indent=2)[:500] + "...")

        await conn.close()
        return

    # Confirm
    print("\n" + "=" * 50)
    response = input("Submit enrichment batch? [y/N]: ")
    if response.lower() != "y":
        print("Aborted.")
        await conn.close()
        return

    # Initialize Anthropic client
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Build all requests
    all_requests = []
    for m in methods:
        all_requests.append(build_batch_request(m, "method"))
    for a in assumptions:
        all_requests.append(build_batch_request(a, "assumption"))

    # Submit in batches
    batch_ids = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(0, len(all_requests), BATCH_SIZE):
        batch_requests = all_requests[i : i + BATCH_SIZE]
        batch_name = f"enrichment_{timestamp}_batch{len(batch_ids) + 1}"

        print(f"\nSubmitting batch {len(batch_ids) + 1}: {len(batch_requests)} requests...")

        # Write JSONL file
        jsonl_file = OUTPUT_DIR / f"{batch_name}.jsonl"
        with open(jsonl_file, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")

        # Submit batch
        batch = client.messages.batches.create(
            requests=batch_requests,
        )

        batch_ids.append(batch.id)
        print(f"  ✓ Batch ID: {batch.id}")

    # Save batch info
    info_file = OUTPUT_DIR / f"enrichment_{timestamp}_info.json"
    with open(info_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "batch_ids": batch_ids,
                "method_count": len(methods),
                "assumption_count": len(assumptions),
                "total_requests": len(all_requests),
            },
            f,
            indent=2,
        )

    print(f"\n✅ Submitted {len(batch_ids)} batch(es)")
    print(f"Batch IDs saved to: {info_file}")
    print("\nCheck status with: python scripts/check_batch_status.py")
    print("Process results with: python scripts/process_method_enrichment.py")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
